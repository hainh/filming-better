//! Frame interpolation stage (RIFE-style engines: img0, img1, timestep -> output).

use super::pool::{f32_to_f16_bits, Pool, Scratch, Tensor};
use super::stage::{Item, Stage};
use super::tiling::round_up;
use crate::models::ResolvedEngine;
use crate::native::{self, DType, DeviceBuffer, Engine, Stream};
use anyhow::{bail, Context, Result};
use std::sync::Arc;

pub struct Interpolator {
    engine: Engine,
    stream: Stream,
    pool: Arc<Pool>,
    scratch: Scratch,
    /// Output frames per input frame (2..=4).
    pub factor: u32,
    pm: i32,
    min_hw: (i32, i32),
    max_hw: (i32, i32),
    eng_dtype: DType,
    names: [String; 4], // img0, img1, timestep, output
    /// timestep tensors for k = 1..factor-1 in the engine dtype
    ts: Vec<DeviceBuffer>,
    cur_shape: Option<(i32, i32)>,
    prev: Option<Arc<Tensor>>,
    pad_cache: Option<(Arc<Tensor>, Arc<Tensor>)>,
    pub interpolated: u64,
    pub duplicated: u64,
    busy: std::time::Duration,
}

const S_A: usize = 0;
const S_B: usize = 1;
const S_OUT: usize = 2;

impl Interpolator {
    pub fn new(info: &ResolvedEngine, factor: u32, pool: Arc<Pool>) -> Result<Interpolator> {
        let mut engine = Engine::load(&info.path)
            .with_context(|| format!("loading engine {}", info.path.display()))?;
        let descs = engine.io_descs()?;
        let inputs: Vec<_> = descs.iter().filter(|d| d.is_input).collect();
        let output = descs.iter().find(|d| !d.is_input).context("interp engine has no output tensor")?;
        let find = |n: &str, idx: usize| -> Result<&native::TensorDesc> {
            inputs
                .iter()
                .find(|d| d.name == n)
                .or_else(|| inputs.get(idx))
                .copied()
                .with_context(|| format!("interp engine has no input '{n}'"))
        };
        let i0 = find("img0", 0)?;
        let i1 = find("img1", 1)?;
        let its = find("timestep", 2)?;
        if i0.dims.len() != 4 || i0.min_dims.len() != 4 {
            bail!("interp engine '{}' input is not NCHW", info.id);
        }
        let eng_dtype = i0.dtype;
        // timestep tensors
        let mut ts = Vec::new();
        for k in 1..factor {
            let v = k as f32 / factor as f32;
            let buf = DeviceBuffer::alloc(eng_dtype.bytes())?;
            match eng_dtype {
                DType::Fp16 => buf.upload(&[f32_to_f16_bits(v)], None)?,
                DType::Fp32 => buf.upload(&[v], None)?,
            }
            ts.push(buf);
        }
        engine.set_input_shape(&its.name, &[1, 1, 1, 1])?;
        Ok(Interpolator {
            stream: Stream::new()?,
            engine,
            pool,
            scratch: Scratch::default(),
            factor,
            pm: info.pad_multiple.max(1) as i32,
            min_hw: (i0.min_dims[2] as i32, i0.min_dims[3] as i32),
            max_hw: (i0.max_dims[2] as i32, i0.max_dims[3] as i32),
            eng_dtype,
            names: [i0.name.clone(), i1.name.clone(), its.name.clone(), output.name.clone()],
            ts,
            cur_shape: None,
            prev: None,
            pad_cache: None,
            interpolated: 0,
            duplicated: 0,
            busy: std::time::Duration::ZERO,
        })
    }

    /// Allocation size a `w x h` frame must have for this engine.
    pub fn padded_dims(&self, w: i32, h: i32) -> (i32, i32) {
        (round_up(w, self.pm).max(round_up(self.min_hw.1, self.pm)), round_up(h, self.pm).max(round_up(self.min_hw.0, self.pm)))
    }

    pub fn check_fit(&self, w: i32, h: i32) -> Result<()> {
        let (pw, ph) = self.padded_dims(w, h);
        if pw > self.max_hw.1 || ph > self.max_hw.0 {
            bail!(
                "frame {w}x{h} (padded {pw}x{ph}) is larger than the interpolation engine supports ({}x{}). \
                 Fix: use --order interp-first so interpolation runs at the source resolution, or --interp 1",
                self.max_hw.1,
                self.max_hw.0
            );
        }
        Ok(())
    }

    fn ensure_padded(&mut self, t: &Arc<Tensor>) -> Result<Arc<Tensor>> {
        let (pw, ph) = self.padded_dims(t.w, t.h);
        if t.pw == pw && t.ph == ph {
            return Ok(t.clone());
        }
        if let Some((src, dst)) = &self.pad_cache {
            if Arc::ptr_eq(src, t) {
                return Ok(dst.clone());
            }
        }
        let out = Tensor::alloc(&self.pool, t.w, t.h, pw, ph, t.dtype)?;
        if t.is_exact() {
            native::pad(t.ptr(), t.dtype, t.w, t.h, out.ptr(), pw, ph, Some(&self.stream))?;
        } else {
            // content is w x h inside a larger allocation: crop then pad
            let tmp = super::upscale::crop_exact(t, &self.pool, &self.stream)?;
            native::pad(tmp.ptr(), t.dtype, t.w, t.h, out.ptr(), pw, ph, Some(&self.stream))?;
            self.stream.sync()?;
        }
        let out = Arc::new(out);
        self.pad_cache = Some((t.clone(), out.clone()));
        Ok(out)
    }

    /// Interpolated frame between `a` and `b` at t = k / factor (k in 1..factor).
    fn interpolate(&mut self, a: &Arc<Tensor>, b: &Arc<Tensor>, k: usize) -> Result<Arc<Tensor>> {
        let (pw, ph) = (a.pw, a.ph);
        let n = 3 * pw as usize * ph as usize;
        let out = Tensor::alloc(&self.pool, a.w, a.h, pw, ph, DType::Fp16)?;
        let (pa, pb, po);
        if self.eng_dtype == DType::Fp16 {
            pa = a.ptr();
            pb = b.ptr();
            po = out.ptr();
        } else {
            pa = self.scratch.get(S_A, n * 4)?;
            pb = self.scratch.get(S_B, n * 4)?;
            po = self.scratch.get(S_OUT, n * 4)?;
            native::convert(a.ptr(), DType::Fp16, pa, DType::Fp32, pw, ph, Some(&self.stream))?;
            native::convert(b.ptr(), DType::Fp16, pb, DType::Fp32, pw, ph, Some(&self.stream))?;
        }
        if self.cur_shape != Some((pw, ph)) {
            let dims = [1, 3, ph as i64, pw as i64];
            self.engine.set_input_shape(&self.names[0], &dims)?;
            self.engine.set_input_shape(&self.names[1], &dims)?;
            self.cur_shape = Some((pw, ph));
        }
        self.engine.set_tensor(&self.names[0], pa)?;
        self.engine.set_tensor(&self.names[1], pb)?;
        self.engine.set_tensor(&self.names[2], self.ts[k - 1].ptr())?;
        self.engine.set_tensor(&self.names[3], po)?;
        self.engine.run(&self.stream)?;
        if self.eng_dtype == DType::Fp32 {
            native::convert(po, DType::Fp32, out.ptr(), DType::Fp16, pw, ph, Some(&self.stream))?;
        }
        self.interpolated += 1;
        Ok(Arc::new(out))
    }

    fn emit_pair(&mut self, prev: &Arc<Tensor>, cur: &Arc<Tensor>, cut: bool, out: &mut Vec<Item>) -> Result<()> {
        out.push(Item { t: prev.clone(), cut: false });
        let same = Arc::ptr_eq(prev, cur);
        for k in 1..self.factor as usize {
            if cut || same {
                self.duplicated += 1;
                out.push(Item { t: prev.clone(), cut: false });
            } else {
                let t = self.interpolate(prev, cur, k)?;
                out.push(Item { t, cut: false });
            }
        }
        self.stream.sync()?;
        Ok(())
    }
}

impl Stage for Interpolator {
    fn name(&self) -> &'static str {
        "interpolate"
    }

    fn process(&mut self, item: Item, out: &mut Vec<Item>) -> Result<()> {
        let t_start = std::time::Instant::now();
        let cur = self.ensure_padded(&item.t)?;
        if let Some(prev) = self.prev.take() {
            self.emit_pair(&prev, &cur, item.cut, out)?;
        }
        self.stream.sync()?;
        self.prev = Some(cur);
        self.busy += t_start.elapsed();
        Ok(())
    }

    fn finish(&mut self, out: &mut Vec<Item>) -> Result<()> {
        if self.interpolated > 0 {
            tracing::info!("interp: {} frames, {:.1} ms/frame busy", self.interpolated, self.busy.as_secs_f64() * 1000.0 / self.interpolated as f64);
        }
        if let Some(prev) = self.prev.take() {
            for _ in 0..self.factor {
                out.push(Item { t: prev.clone(), cut: false });
            }
            self.duplicated += (self.factor - 1) as u64;
        }
        Ok(())
    }
}
