//! Upscale stage: whole-frame inference when the picture fits the engine's profile,
//! otherwise overlapping tiles blended with feathered weights.

use super::pool::{Pool, Scratch, Tensor};
use super::stage::{Item, Stage};
use super::tiling::{self, round_up, TileLayout};
use crate::models::ResolvedEngine;
use crate::native::{self, DType, Engine, Stream};
use anyhow::{bail, Context, Result};
use std::sync::Arc;

/// Overlap between neighbouring tiles in *input* pixels.
pub const TILE_OVERLAP: i32 = 32;

pub struct Upscaler {
    engine: Engine,
    stream: Stream,
    pool: Arc<Pool>,
    scratch: Scratch,
    pub scale: i32,
    pm: i32,
    min_hw: (i32, i32),
    max_hw: (i32, i32),
    eng_dtype: DType,
    in_name: String,
    out_name: String,
    tile_limit: Option<i32>,
    cur_shape: Option<(i32, i32)>,
    last: Option<(Arc<Tensor>, Arc<Tensor>)>,
    pub frames: u64,
    busy: std::time::Duration,
}

// scratch slots
const S_TILE_IN: usize = 0;
const S_TILE_OUT: usize = 1;
const S_ACCUM: usize = 2;
const S_WEIGHT: usize = 3;
const S_CVT_IN: usize = 4;
const S_CVT_OUT: usize = 5;
const S_PADDED: usize = 6;

impl Upscaler {
    pub fn new(info: &ResolvedEngine, pool: Arc<Pool>, tile: Option<u32>) -> Result<Upscaler> {
        let engine = Engine::load(&info.path)
            .with_context(|| format!("loading engine {}", info.path.display()))?;
        let descs = engine.io_descs()?;
        let inp = descs.iter().find(|d| d.is_input).context("engine has no input tensor")?;
        let outp = descs.iter().find(|d| !d.is_input).context("engine has no output tensor")?;
        if inp.dims.len() != 4 || inp.min_dims.len() != 4 {
            bail!("upscale engine '{}' input is not NCHW (dims {:?})", info.id, inp.dims);
        }
        let mut engine = engine;
        // verify the scale factor against the engine itself
        let (mh, mw) = (inp.min_dims[2].max(1), inp.min_dims[3].max(1));
        engine.set_input_shape(&inp.name, &[1, 3, mh, mw])?;
        let oshape = engine.io_shape(&outp.name)?;
        let scale = if oshape.len() == 4 && mh > 0 { (oshape[2] / mh) as i32 } else { info.scale as i32 };
        if scale != info.scale as i32 {
            tracing::warn!("engine {} reports scale {} but the manifest says {}", info.id, scale, info.scale);
        }
        let min_hw = (inp.min_dims[2] as i32, inp.min_dims[3] as i32);
        let max_hw = (inp.max_dims[2] as i32, inp.max_dims[3] as i32);
        Ok(Upscaler {
            stream: Stream::new()?,
            engine,
            pool,
            scratch: Scratch::default(),
            scale,
            pm: info.pad_multiple.max(1) as i32,
            min_hw,
            max_hw,
            eng_dtype: inp.dtype,
            in_name: inp.name.clone(),
            out_name: outp.name.clone(),
            tile_limit: tile.map(|t| t as i32),
            cur_shape: None,
            last: None,
            frames: 0,
            busy: std::time::Duration::ZERO,
        })
    }

    /// Tile layout this upscaler would use for a `w x h` frame.
    pub fn layout_for(&self, w: i32, h: i32) -> TileLayout {
        let (mut max_w, mut max_h) = (self.max_hw.1, self.max_hw.0);
        if let Some(t) = self.tile_limit {
            max_w = max_w.min(t);
            max_h = max_h.min(t);
        }
        tiling::layout(w, h, max_w, max_h, TILE_OVERLAP, self.pm, self.min_hw.1, self.min_hw.0)
    }

    fn out_dims(&self, w: i32, h: i32) -> (i32, i32) {
        (w * self.scale, h * self.scale)
    }

    /// Run the engine on a `tw x th` fp16 tensor at `src`, returning a pointer to the
    /// `(tw*s) x (th*s)` fp16 result (valid until the next call).
    fn infer(&mut self, src: u64, tw: i32, th: i32) -> Result<u64> {
        let s = self.scale;
        let n_in = (3 * tw as usize) * th as usize;
        let n_out = n_in * (s * s) as usize;
        let (in_ptr, out_ptr);
        if self.eng_dtype == DType::Fp16 {
            in_ptr = src;
            out_ptr = self.scratch.get(S_TILE_OUT, n_out * 2)?;
        } else {
            in_ptr = self.scratch.get(S_CVT_IN, n_in * 4)?;
            native::convert(src, DType::Fp16, in_ptr, DType::Fp32, tw, th, Some(&self.stream))?;
            out_ptr = self.scratch.get(S_CVT_OUT, n_out * 4)?;
        }
        if self.cur_shape != Some((tw, th)) {
            self.engine.set_input_shape(&self.in_name, &[1, 3, th as i64, tw as i64])?;
            self.cur_shape = Some((tw, th));
        }
        self.engine.set_tensor(&self.in_name, in_ptr)?;
        self.engine.set_tensor(&self.out_name, out_ptr)?;
        self.engine.run(&self.stream)?;
        if self.eng_dtype == DType::Fp16 {
            Ok(out_ptr)
        } else {
            let o16 = self.scratch.get(S_TILE_OUT, n_out * 2)?;
            native::convert(out_ptr, DType::Fp32, o16, DType::Fp16, tw * s, th * s, Some(&self.stream))?;
            Ok(o16)
        }
    }

    pub fn upscale(&mut self, t: &Arc<Tensor>) -> Result<Arc<Tensor>> {
        let t_start = std::time::Instant::now();
        if !t.is_exact() {
            bail!("internal error: upscaler needs an unpadded tensor");
        }
        let (w, h) = (t.w, t.h);
        let s = self.scale;
        let (ow, oh) = self.out_dims(w, h);
        let out = Tensor::alloc_exact(&self.pool, ow, oh, DType::Fp16)?;
        let layout = self.layout_for(w, h);
        let st = &self.stream;
        if layout.is_single() {
            let (tw, th) = (layout.tile_w, layout.tile_h);
            let src = if tw != w || th != h {
                let p = self.scratch.get(S_PADDED, 3 * tw as usize * th as usize * 2)?;
                native::pad(t.ptr(), DType::Fp16, w, h, p, tw, th, Some(st))?;
                p
            } else {
                t.ptr()
            };
            let res = self.infer(src, tw, th)?;
            native::crop(res, DType::Fp16, tw * s, th * s, 0, 0, out.ptr(), ow, oh, Some(&self.stream))?;
        } else {
            let (tw, th) = (layout.tile_w, layout.tile_h);
            let acc = self.scratch.get(S_ACCUM, 3 * ow as usize * oh as usize * 4)?;
            let wgt = self.scratch.get(S_WEIGHT, ow as usize * oh as usize * 4)?;
            native::memset(acc, 0, 3 * ow as usize * oh as usize * 4, Some(&self.stream))?;
            native::memset(wgt, 0, ow as usize * oh as usize * 4, Some(&self.stream))?;
            let tile_in = self.scratch.get(S_TILE_IN, 3 * tw as usize * th as usize * 2)?;
            for r in &layout.tiles {
                native::tile_split(t.ptr(), DType::Fp16, w, h, r.x, r.y, tile_in, tw, th, Some(&self.stream))?;
                let res = self.infer(tile_in, tw, th)?;
                native::tile_merge(
                    res,
                    DType::Fp16,
                    tw * s,
                    th * s,
                    acc,
                    wgt,
                    ow,
                    oh,
                    r.x * s,
                    r.y * s,
                    layout.overlap * s,
                    r.border,
                    Some(&self.stream),
                )?;
            }
            native::tile_finalize(acc, wgt, ow, oh, out.ptr(), DType::Fp16, Some(&self.stream))?;
        }
        self.stream.sync()?;
        self.frames += 1;
        self.busy += t_start.elapsed();
        Ok(Arc::new(out))
    }

    /// Scratch estimate (bytes) for a frame, to fail early with a helpful message.
    pub fn estimate_scratch(&self, w: i32, h: i32) -> usize {
        let l = self.layout_for(w, h);
        let s = self.scale as usize;
        let tile = 3 * l.tile_w as usize * l.tile_h as usize;
        let mut b = tile * 2 * (1 + s * s) * 2;
        if self.eng_dtype == DType::Fp32 {
            b += tile * 4 * (1 + s * s);
        }
        if !l.is_single() {
            let o = (w as usize * s) * (h as usize * s);
            b += o * 4 * 4;
        }
        b
    }
}

impl Stage for Upscaler {
    fn name(&self) -> &'static str {
        "upscale"
    }

    fn process(&mut self, item: Item, out: &mut Vec<Item>) -> Result<()> {
        // exact copies (duplicated frames) reuse the previous result
        if let Some((src, res)) = &self.last {
            if Arc::ptr_eq(src, &item.t) {
                out.push(Item { t: res.clone(), cut: item.cut });
                return Ok(());
            }
        }
        let src = if item.t.is_exact() { item.t.clone() } else { Arc::new(crop_exact(&item.t, &self.pool, &self.stream)?) };
        let res = self.upscale(&src)?;
        self.last = Some((item.t.clone(), res.clone()));
        out.push(Item { t: res, cut: item.cut });
        Ok(())
    }

    fn finish(&mut self, _out: &mut Vec<Item>) -> Result<()> {
        if self.frames > 0 {
            tracing::info!("upscale: {} frames, {:.1} ms/frame busy", self.frames, self.busy.as_secs_f64() * 1000.0 / self.frames as f64);
        }
        Ok(())
    }
}

/// Copy the `w x h` content of a padded tensor into an exact-size tensor.
pub fn crop_exact(t: &Tensor, pool: &Arc<Pool>, stream: &Stream) -> Result<Tensor> {
    let out = Tensor::alloc_exact(pool, t.w, t.h, t.dtype)?;
    native::crop(t.ptr(), t.dtype, t.pw, t.ph, 0, 0, out.ptr(), t.w, t.h, Some(stream))?;
    Ok(out)
}

/// Round `v` up to the engine's alignment.
pub fn aligned(v: i32, pm: i32) -> i32 {
    round_up(v, pm)
}
