//! The processing pipeline: decode -> (interpolate | upscale) -> encode, one thread per stage,
//! bounded channels for backpressure, one CUDA stream per stage. Frames never leave VRAM.

pub mod interp;
pub mod pool;
pub mod stage;
pub mod tiling;
pub mod timing;
pub mod upscale;

use crate::io::decode::{GpuFrame, SidePacket, VideoSource};
use crate::io::encode::{EncoderConfig, VideoCodec, VideoEncoder};
use crate::io::mux::Output;
use crate::io::probe::{resolve_color, ColorSpec, ColorTags};
use crate::io::{AVRational, CudaDevice, NOPTS};
use crate::models::ResolvedEngine;
use crate::native::{self, DType, Stream};
use anyhow::{anyhow, bail, Context, Result};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use pool::{Pool, Tensor};
use stage::{run_stage, Item, Stage};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use timing::{CfrGrid, Slot};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    InterpFirst,
    UpscaleFirst,
}

impl Order {
    pub fn parse(s: &str) -> Option<Order> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "interp-first" | "interpfirst" => Some(Order::InterpFirst),
            "upscale-first" | "upscalefirst" => Some(Order::UpscaleFirst),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Order::InterpFirst => "interp-first",
            Order::UpscaleFirst => "upscale-first",
        }
    }
}

/// Everything needed to process one file.
#[derive(Clone)]
pub struct JobSpec {
    pub input: PathBuf,
    pub output: PathBuf,
    pub upscale: Option<ResolvedEngine>,
    pub interp_engine: Option<ResolvedEngine>,
    /// Output frames per input frame (1 = no interpolation).
    pub interp_factor: u32,
    pub order: Order,
    pub codec: VideoCodec,
    pub cq: u32,
    pub nvenc_preset: String,
    pub scene_threshold: f32,
    pub tile: Option<u32>,
    pub device: i32,
    pub cpu_decode: bool,
}

/// Shared control block: cancellation, abort-on-error, first error, progress counters.
pub struct Ctl {
    pub device: i32,
    cancel: AtomicBool,
    abort: AtomicBool,
    error: Mutex<Option<anyhow::Error>>,
    /// Output frames written to the encoder so far.
    pub done: AtomicU64,
    /// Estimated total output frames (0 = unknown).
    pub total: AtomicU64,
}

impl Ctl {
    pub fn new(device: i32) -> Arc<Ctl> {
        Arc::new(Ctl {
            device,
            cancel: AtomicBool::new(false),
            abort: AtomicBool::new(false),
            error: Mutex::new(None),
            done: AtomicU64::new(0),
            total: AtomicU64::new(0),
        })
    }
    /// Graceful stop: finish what was decoded so far, write a valid file.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
    pub fn aborted(&self) -> bool {
        self.abort.load(Ordering::SeqCst)
    }
    pub fn fail(&self, e: anyhow::Error) {
        let mut g = self.error.lock().unwrap();
        if g.is_none() {
            *g = Some(e);
        }
        self.abort.store(true, Ordering::SeqCst);
    }
    fn take_error(&self) -> Option<anyhow::Error> {
        self.error.lock().unwrap().take()
    }
}

#[derive(Debug, Clone, Default)]
pub struct JobStats {
    pub frames_decoded: u64,
    pub frames_out: u64,
    pub out_w: i32,
    pub out_h: i32,
    pub out_fps: f64,
    pub elapsed: Duration,
    pub cfr_dropped: u64,
    pub cfr_duplicated: u64,
    pub scene_cuts: u64,
    pub hw_decode: bool,
    pub cancelled: bool,
    pub skipped_streams: Vec<String>,
}

impl JobStats {
    pub fn fps(&self) -> f64 {
        let s = self.elapsed.as_secs_f64();
        if s > 0.0 {
            self.frames_out as f64 / s
        } else {
            0.0
        }
    }
}

/// Parameters the decode thread needs.
struct DecodeParams {
    pw: i32,
    ph: i32,
    color: ColorSpec,
    fps: AVRational,
    want_scene: bool,
    scene_threshold: f32,
}

#[derive(Default)]
struct DecodeStats {
    decoded: u64,
    dropped: u64,
    duplicated: u64,
    cuts: u64,
}

/// Colour conversion parameters from stream tags, falling back to the first frame's tags.
pub fn pick_color(info: &crate::io::probe::VideoInfo, frame_tags: ColorTags, w: i32, h: i32) -> ColorSpec {
    let pick = |a: i32, b: i32| if a != 0 && a != 2 { a } else { b };
    let tags = ColorTags {
        space: pick(info.color.space, frame_tags.space),
        range: if info.color.range == 1 || info.color.range == 2 { info.color.range } else { frame_tags.range },
        primaries: pick(info.color.primaries, frame_tags.primaries),
        trc: pick(info.color.trc, frame_tags.trc),
    };
    resolve_color(tags, w, h)
}

/// Open the video source, decoding with NVDEC if possible and falling back to the CPU when the
/// hardware path fails on the very first frame. Returns the source with its first frame queued.
pub fn open_source(path: &std::path::Path, device: &CudaDevice, force_cpu: bool, forward: bool) -> Result<(VideoSource, bool)> {
    if !force_cpu {
        match VideoSource::open(path, device, true, forward) {
            Ok(mut src) => {
                if !src.hw {
                    return prime(src, false);
                }
                match src.next_frame() {
                    Ok(Some(f)) => {
                        src.push_back(f);
                        return Ok((src, true));
                    }
                    Ok(None) => bail!("the video stream contains no decodable frames"),
                    Err(e) => tracing::warn!("NVDEC decode failed ({e}); falling back to CPU decoding"),
                }
            }
            Err(e @ crate::io::IoError::Ff { .. }) if e.to_string().starts_with("cannot open") => return Err(e.into()),
            Err(e) => tracing::warn!("hardware decoder unavailable ({e}); falling back to CPU decoding"),
        }
    }
    let src = VideoSource::open(path, device, false, forward)?;
    prime(src, false)
}

fn prime(mut src: VideoSource, hw: bool) -> Result<(VideoSource, bool)> {
    match src.next_frame()? {
        Some(f) => {
            src.push_back(f);
            Ok((src, hw))
        }
        None => bail!("the video stream contains no decodable frames"),
    }
}

fn decode_loop(
    src: &mut VideoSource,
    p: &DecodeParams,
    pool: &Arc<Pool>,
    tx: &Sender<Item>,
    side_tx: &Sender<SidePacket>,
    ctl: &Ctl,
    stats: &mut DecodeStats,
) -> Result<()> {
    native::init(ctl.device)?;
    let stream = Stream::new()?;
    let tb = src.info.time_base;
    let mut grid = CfrGrid::new(tb, p.fps);
    let mut prev: Option<Arc<Tensor>> = None;
    let (w0, h0) = (src.info.width, src.info.height);
    let mut dims: Option<(i32, i32)> = None;
    let _ = (w0, h0);
    loop {
        if ctl.cancelled() || ctl.aborted() {
            break;
        }
        let Some(gf) = src.next_frame().context("decoding video")? else { break };
        for sp in src.take_side() {
            let _ = side_tx.send(sp);
        }
        stats.decoded += 1;
        let (w, h) = (gf.width, gf.height);
        match dims {
            None => dims = Some((w, h)),
            Some(d) if d != (w, h) => bail!("the video changes resolution mid-stream ({}x{} -> {w}x{h}), which is not supported", d.0, d.1),
            _ => {}
        }
        match grid.push(gf.pts) {
            Slot::Drop => continue,
            Slot::Emit { dups } => {
                let t = Tensor::alloc(pool, w, h, p.pw.max(w), p.ph.max(h), DType::Fp16)?;
                native::yuv_to_rgb(&gf.yuv(), p.color.matrix, p.color.range, t.ptr(), DType::Fp16, t.pw, t.ph, Some(&stream))?;
                stream.sync()?;
                drop(gf); // release the NVDEC surface
                let t = Arc::new(t);
                let mut cut = false;
                if let (true, Some(pv)) = (p.want_scene, prev.as_ref()) {
                    let score = native::scene_score(pv.ptr(), t.ptr(), DType::Fp16, t.pw, t.ph, Some(&stream))?;
                    cut = score > p.scene_threshold;
                    tracing::debug!("scene score {score:.4}{}", if cut { " (cut)" } else { "" });
                    if cut {
                        stats.cuts += 1;
                    }
                }
                if let Some(pv) = prev.as_ref() {
                    for _ in 0..dups {
                        if tx.send(Item { t: pv.clone(), cut: false }).is_err() {
                            return Ok(());
                        }
                    }
                }
                if tx.send(Item { t: t.clone(), cut }).is_err() {
                    return Ok(());
                }
                prev = Some(t);
            }
        }
    }
    // hand over remaining audio / subtitle packets
    for sp in src.take_side() {
        let _ = side_tx.send(sp);
    }
    stats.dropped = grid.dropped;
    stats.duplicated = grid.duplicated;
    Ok(())
}

struct EncodeParams {
    color: ColorSpec,
    base_ticks: i64,
}

fn encode_loop(
    rx: &Receiver<Item>,
    side_rx: &Receiver<SidePacket>,
    enc: &mut VideoEncoder,
    out: &mut Option<Output>,
    p: &EncodeParams,
    ctl: &Ctl,
) -> Result<u64> {
    native::init(ctl.device)?;
    let stream = Stream::new()?;
    let mut m: i64 = 0;
    while let Ok(item) = rx.recv() {
        if ctl.aborted() {
            return Ok(m as u64);
        }
        let t = &item.t;
        let (frame, mut yuv) = enc.alloc_frame()?;
        yuv.width = enc.cfg.width;
        yuv.height = enc.cfg.height;
        if t.w < yuv.width || t.h < yuv.height {
            bail!("internal error: frame {}x{} smaller than the encoder size {}x{}", t.w, t.h, yuv.width, yuv.height);
        }
        native::rgb_to_yuv(t.ptr(), t.dtype, t.pw, t.ph, &yuv, p.color.matrix, p.color.range, Some(&stream))?;
        stream.sync()?;
        drop(item);
        let o = out.as_mut().unwrap();
        enc.encode(Some((&frame, p.base_ticks + m)), &mut |pkt| {
            for mut sp in side_rx.try_iter() {
                o.write_side(&mut sp)?;
            }
            o.write_video(pkt)?;
            Ok(())
        })?;
        m += 1;
        ctl.done.store(m as u64, Ordering::Relaxed);
    }
    if ctl.aborted() {
        return Ok(m as u64);
    }
    // flush the encoder, then the rest of the side packets, then the trailer
    let o = out.as_mut().unwrap();
    enc.encode(None, &mut |pkt| {
        for mut sp in side_rx.try_iter() {
            o.write_side(&mut sp)?;
        }
        o.write_video(pkt)?;
        Ok(())
    })?;
    while let Ok(mut sp) = side_rx.recv() {
        o.write_side(&mut sp)?;
    }
    Ok(m as u64)
}

/// Run one file through the pipeline.
pub fn run_job(spec: &JobSpec, ctl: &Arc<Ctl>) -> Result<JobStats> {
    let t_start = Instant::now();
    let device = CudaDevice::shared(spec.device).ok();
    native::init(spec.device)?;
    let gpu = native::gpu_info(spec.device)?;
    let device = match device {
        Some(d) => d,
        None => CudaDevice::shared(spec.device)?,
    };

    // ---- source + first frame ----
    let (mut src, hw) = open_source(&spec.input, &device, spec.cpu_decode, true)?;
    let info = src.info.clone();
    let first: GpuFrame = src.next_frame()?.context("no frames")?;
    let (in_w, in_h) = (first.width, first.height);
    let first_pts = first.pts;
    let ten_bit = first.format == native::YuvFormat::P010;
    let frame_tags = first.color;
    src.push_back(first);

    let fps = info.fps.ok_or_else(|| anyhow!("cannot determine the frame rate of '{}'", spec.input.display()))?;
    let color = pick_color(&info, frame_tags, in_w, in_h);
    let tags = color.tags;
    if tags.is_hdr() {
        tracing::warn!("HDR input detected: v1 processes SDR only, colours will look wrong");
    }
    if info.field_order > 1 {
        tracing::warn!("input may be interlaced: interlaced video is not supported in v1");
    }

    // ---- stages ----
    let pool = Pool::new();
    let factor = spec.interp_factor.clamp(1, 4);
    let mut interp = match (&spec.interp_engine, factor) {
        (Some(e), f) if f > 1 => Some(interp::Interpolator::new(e, f, pool.clone())?),
        (None, f) if f > 1 => bail!("interpolation requested but no interpolation engine was resolved"),
        _ => None,
    };
    let up = match &spec.upscale {
        Some(e) => Some(upscale::Upscaler::new(e, pool.clone(), spec.tile)?),
        None => None,
    };
    let scale = up.as_ref().map(|u| u.scale).unwrap_or(1);
    let (up_w, up_h) = (in_w * scale, in_h * scale);
    // Output is exactly input x scale; a dimension is cropped by 1 px only if it is odd (NVENC 4:2:0).
    let crop_odd = |v: i32| if v % 2 != 0 { v - 1 } else { v };
    let (out_w, out_h) = (crop_odd(up_w), crop_odd(up_h));
    if out_w < 16 || out_h < 16 {
        bail!("output size {out_w}x{out_h} is too small to encode");
    }
    let interp_first = spec.order == Order::InterpFirst || up.is_none() || interp.is_none();
    if let Some(i) = &interp {
        if interp_first {
            i.check_fit(in_w, in_h)?;
        } else {
            i.check_fit(up_w, up_h)?;
        }
    }
    let (pad_w, pad_h) = match (&interp, interp_first) {
        (Some(i), true) => i.padded_dims(in_w, in_h),
        _ => (in_w, in_h),
    };
    tracing::info!(
        "input {in_w}x{in_h} {} fps {}/{} ({}), output {out_w}x{out_h}, interp x{factor}, scale x{scale}, order {}",
        info.codec_name,
        fps.num,
        fps.den,
        if hw { "NVDEC" } else { "CPU decode" },
        if interp_first { "interp-first" } else { "upscale-first" }
    );

    let mut stages: Vec<Box<dyn Stage>> = Vec::new();
    match (interp.take(), up) {
        (Some(i), Some(u)) => {
            if interp_first {
                stages.push(Box::new(i));
                stages.push(Box::new(u));
            } else {
                stages.push(Box::new(u));
                stages.push(Box::new(i));
            }
        }
        (Some(i), None) => stages.push(Box::new(i)),
        (None, Some(u)) => stages.push(Box::new(u)),
        (None, None) => {}
    }

    // ---- encoder + muxer ----
    let out_fps = AVRational { num: fps.num * factor as i32, den: fps.den };
    let global_header = true; // mp4 / mkv / mov all want extradata in the container
    let ecfg = EncoderConfig {
        codec: spec.codec,
        cq: spec.cq,
        preset: spec.nvenc_preset.clone(),
        width: out_w,
        height: out_h,
        fps: out_fps,
        ten_bit,
        color,
        sar: info.sample_aspect_ratio,
        global_header,
        gpu_cc: Some((gpu.cc_major, gpu.cc_minor)),
    };
    let mut enc = VideoEncoder::open(&ecfg, &device)?;
    let output = Output::create(&spec.output, src.input(), &info, &enc)?;
    let skipped_streams = output.skipped.clone();
    let mut output = Some(output);

    let base_ticks = if first_pts == NOPTS { 0 } else { timing::start_ticks(first_pts, info.time_base, out_fps) };
    let est = info.estimated_frames();
    ctl.total.store(est * factor as u64, Ordering::Relaxed);
    ctl.done.store(0, Ordering::Relaxed);

    // ---- channels & threads ----
    let cap = 3;
    let n_stages = stages.len();
    let mut chans: Vec<(Sender<Item>, Receiver<Item>)> = (0..=n_stages).map(|_| bounded(cap)).collect();
    let (side_tx, side_rx) = unbounded::<SidePacket>();
    let dp = DecodeParams {
        pw: pad_w,
        ph: pad_h,
        color,
        fps,
        want_scene: factor > 1,
        scene_threshold: spec.scene_threshold,
    };
    let ep = EncodeParams { color, base_ticks };
    let mut dstats = DecodeStats::default();
    let mut frames_out = 0u64;

    let ctl_ref: &Ctl = ctl;
    std::thread::scope(|scope| {
        // channel i: stage i input; last channel feeds the encoder
        let mut rxs: Vec<Receiver<Item>> = Vec::new();
        let mut txs: Vec<Sender<Item>> = Vec::new();
        for (tx, rx) in chans.drain(..) {
            txs.push(tx);
            rxs.push(rx);
        }
        // decode -> txs[0]; stage k: rxs[k] -> txs[k+1]; encode <- rxs[n]
        let enc_rx = rxs.pop().unwrap();
        let dec_tx = txs.remove(0);
        let mut stage_io: Vec<(Receiver<Item>, Sender<Item>)> = Vec::new();
        for _ in 0..n_stages {
            stage_io.push((rxs.remove(0), txs.remove(0)));
        }
        drop(txs);
        drop(rxs);

        let src_ref = &mut src;
        let pool_ref = &pool;
        let dstats_ref = &mut dstats;
        scope.spawn(move || {
            let r = decode_loop(src_ref, &dp, pool_ref, &dec_tx, &side_tx, ctl_ref, dstats_ref);
            if let Err(e) = r {
                ctl_ref.fail(e.context("decode thread"));
            }
            // channels drop here, after the failure flag is set
        });
        for (st, (rx, tx)) in stages.drain(..).zip(stage_io.drain(..)) {
            scope.spawn(move || {
                if let Err(e) = run_stage(st, &rx, &tx, ctl_ref) {
                    ctl_ref.fail(e);
                }
            });
        }
        let enc_ref = &mut enc;
        let out_ref = &mut output;
        let fo = &mut frames_out;
        scope.spawn(move || {
            match encode_loop(&enc_rx, &side_rx, enc_ref, out_ref, &ep, ctl_ref) {
                Ok(n) => *fo = n,
                Err(e) => ctl_ref.fail(e.context("encode thread")),
            }
        });
    });

    if let Some(e) = ctl.take_error() {
        if let Some(o) = output.take() {
            o.abort();
        }
        return Err(e);
    }
    let out = output.take().unwrap();
    out.finish().context("finalizing the output file")?;
    Ok(JobStats {
        frames_decoded: dstats.decoded,
        frames_out,
        out_w,
        out_h,
        out_fps: out_fps.num as f64 / out_fps.den as f64,
        elapsed: t_start.elapsed(),
        cfr_dropped: dstats.dropped,
        cfr_duplicated: dstats.duplicated,
        scene_cuts: dstats.cuts,
        hw_decode: hw,
        cancelled: ctl.cancelled(),
        skipped_streams,
    })
}
