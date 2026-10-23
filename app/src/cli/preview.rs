//! `filming-better preview`: process a single frame and write a before/after PNG.

use super::run::{resolve_engines, Settings, DEFAULT_INTERP_MODEL};
use super::usage;
use crate::io::{CudaDevice, NOPTS};
use crate::native::{self, DType, DeviceBuffer, Stream};
use crate::pipeline::interp::Interpolator;
use crate::pipeline::pool::{Pool, Tensor};
use crate::pipeline::stage::{Item, Stage};
use crate::pipeline::upscale::{crop_exact, Upscaler};
use crate::pipeline::{open_source, pick_color, Order};
use anyhow::{bail, Context, Result};
use clap::Args;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Args, Debug, Clone)]
#[command(after_help = "\
Examples:
  filming-better preview --input in.mp4 --at 00:01:23 --upscale span_x2 --out preview.png
  filming-better preview --input in.mp4 --at 5.5 --upscale span_x2 --interp 2 --out p.png --split

The PNG shows, left to right: the original frame scaled up with bicubic filtering (same size as
the result), the upscaled frame, and - with --interp - an interpolated in-between frame.
With --split every picture is written to its own file (<out>_before.png, _after.png, _interp.png).")]
pub struct PreviewArgs {
    /// Input video
    #[arg(short, long, value_name = "FILE")]
    pub input: PathBuf,
    /// Position of the frame: HH:MM:SS(.ms), MM:SS or seconds
    #[arg(long, value_name = "TIME", default_value = "0")]
    pub at: String,
    /// Upscale model id or `none`
    #[arg(long, value_name = "MODEL|none")]
    pub upscale: Option<String>,
    /// Also show an interpolated frame (2, 3 or 4)
    #[arg(long, value_name = "N", default_value_t = 1)]
    pub interp: u32,
    /// Interpolation model id
    #[arg(long, value_name = "MODEL", default_value = DEFAULT_INTERP_MODEL)]
    pub interp_model: String,
    /// Output PNG (with --split: base name of the files)
    #[arg(short, long, value_name = "FILE", default_value = "preview.png")]
    pub out: PathBuf,
    /// Write separate before/after files instead of one side-by-side picture
    #[arg(long)]
    pub split: bool,
    /// Maximum tile edge for upscaling
    #[arg(long, value_name = "PX")]
    pub tile: Option<u32>,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    /// Decode on the CPU
    #[arg(long)]
    pub cpu_decode: bool,
}

/// `HH:MM:SS(.ms)`, `MM:SS` or plain seconds -> seconds.
pub fn parse_time(s: &str) -> Result<f64> {
    let bad = || anyhow::Error::new(super::errors::CliError::Usage(format!("cannot parse time '{s}' (use HH:MM:SS.ms, MM:SS or seconds)")));
    let parts: Vec<&str> = s.trim().split(':').collect();
    if parts.is_empty() || parts.len() > 3 {
        return Err(bad());
    }
    let mut total = 0.0f64;
    for p in &parts {
        let v: f64 = p.trim().parse().map_err(|_| bad())?;
        if v < 0.0 || !v.is_finite() {
            return Err(bad());
        }
        total = total * 60.0 + v;
    }
    Ok(total)
}

// ------------------------------------------------------------------ image helpers

/// Planar f32 RGB image (CHW, values nominally 0..1).
pub struct PlanarImage {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

fn cubic_weight(x: f32) -> f32 {
    // Catmull-Rom (a = -0.5)
    let a = -0.5f32;
    let x = x.abs();
    if x <= 1.0 {
        (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
    } else if x < 2.0 {
        a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
    } else {
        0.0
    }
}

/// Per output pixel: (first source index, weights).
fn taps(src: usize, dst: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = src as f32 / dst as f32;
    let support = if scale > 1.0 { 2.0 * scale } else { 2.0 };
    (0..dst)
        .map(|o| {
            let center = (o as f32 + 0.5) * scale - 0.5;
            let lo = (center - support).ceil().max(0.0) as usize;
            let hi = ((center + support).floor() as isize).min(src as isize - 1).max(lo as isize) as usize;
            let f = if scale > 1.0 { 1.0 / scale } else { 1.0 };
            let mut w: Vec<f32> = (lo..=hi).map(|i| cubic_weight((i as f32 - center) * f)).collect();
            let sum: f32 = w.iter().sum();
            if sum.abs() > 1e-6 {
                w.iter_mut().for_each(|v| *v /= sum);
            }
            (lo, w)
        })
        .collect()
}

/// Separable bicubic (Catmull-Rom) resize.
pub fn resize_bicubic(img: &PlanarImage, dw: usize, dh: usize) -> PlanarImage {
    if img.w == dw && img.h == dh {
        return PlanarImage { w: dw, h: dh, data: img.data.clone() };
    }
    let tx = taps(img.w, dw);
    let ty = taps(img.h, dh);
    let mut out = vec![0f32; 3 * dw * dh];
    let mut tmp = vec![0f32; img.h * dw];
    for c in 0..3 {
        let plane = &img.data[c * img.w * img.h..(c + 1) * img.w * img.h];
        for y in 0..img.h {
            let row = &plane[y * img.w..(y + 1) * img.w];
            for (x, (lo, w)) in tx.iter().enumerate() {
                let mut acc = 0.0;
                for (k, wk) in w.iter().enumerate() {
                    acc += row[lo + k] * wk;
                }
                tmp[y * dw + x] = acc;
            }
        }
        let o = &mut out[c * dw * dh..(c + 1) * dw * dh];
        for (y, (lo, w)) in ty.iter().enumerate() {
            for x in 0..dw {
                let mut acc = 0.0;
                for (k, wk) in w.iter().enumerate() {
                    acc += tmp[(lo + k) * dw + x] * wk;
                }
                o[y * dw + x] = acc;
            }
        }
    }
    PlanarImage { w: dw, h: dh, data: out }
}

/// Planar f32 -> interleaved RGB8.
pub fn to_rgb8(img: &PlanarImage) -> Vec<u8> {
    let n = img.w * img.h;
    let mut out = vec![0u8; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            let v = img.data[c * n + i].clamp(0.0, 1.0);
            out[3 * i + c] = (v * 255.0 + 0.5) as u8;
        }
    }
    out
}

/// Place panels left to right separated by `gap` pixels of mid gray.
pub fn compose_side_by_side(panels: &[(usize, usize, Vec<u8>)], gap: usize) -> (usize, usize, Vec<u8>) {
    let h = panels.iter().map(|p| p.1).max().unwrap_or(0);
    let w: usize = panels.iter().map(|p| p.0).sum::<usize>() + gap * panels.len().saturating_sub(1);
    let mut out = vec![128u8; 3 * w * h];
    let mut x0 = 0;
    for (pw, ph, px) in panels {
        for y in 0..*ph {
            let dst = (y * w + x0) * 3;
            out[dst..dst + pw * 3].copy_from_slice(&px[y * pw * 3..(y + 1) * pw * 3]);
        }
        x0 += pw + gap;
    }
    (w, h, out)
}

pub fn write_png(path: &Path, w: usize, h: usize, rgb: &[u8]) -> Result<()> {
    if let Some(d) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(d).ok();
    }
    let f = std::fs::File::create(path).with_context(|| format!("cannot create '{}'", path.display()))?;
    let mut enc = png::Encoder::new(BufWriter::new(f), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_compression(png::Compression::Fast);
    let mut wr = enc.write_header()?;
    wr.write_image_data(rgb)?;
    wr.finish()?;
    Ok(())
}

fn download(t: &Tensor, stream: &Stream) -> Result<PlanarImage> {
    if !t.is_exact() {
        bail!("internal error: preview tensor is padded");
    }
    let n = 3 * t.w as usize * t.h as usize;
    let data = match t.dtype {
        DType::Fp32 => {
            let tmp = DeviceBuffer::alloc(n * 4)?;
            native::memcpy(tmp.ptr(), t.ptr(), n * 4, Some(stream))?;
            tmp.download::<f32>(n, Some(stream))?
        }
        DType::Fp16 => {
            let tmp = DeviceBuffer::alloc(n * 4)?;
            native::convert(t.ptr(), DType::Fp16, tmp.ptr(), DType::Fp32, t.w, t.h, Some(stream))?;
            tmp.download::<f32>(n, Some(stream))?
        }
    };
    Ok(PlanarImage { w: t.w as usize, h: t.h as usize, data })
}

fn decode_frame_at(src: &mut crate::io::decode::VideoSource, at: f64) -> Result<crate::io::decode::GpuFrame> {
    if at > 0.0 {
        src.seek(at)?;
    }
    let tb = crate::io::q2d(src.info.time_base);
    let target = src.info.start_time + (at / tb).round() as i64;
    let mut last = None;
    while let Some(f) = src.next_frame().context("decoding")? {
        let reached = f.pts != NOPTS && f.pts >= target;
        if reached || at <= 0.0 {
            return Ok(f);
        }
        last = Some(f);
    }
    last.context("the video has no frame at that position")
}

pub fn cmd_preview(a: PreviewArgs) -> Result<()> {
    super::run::init_logging(None)?;
    if !(1..=4).contains(&a.interp) {
        return usage(format!("--interp must be 1, 2, 3 or 4 (got {})", a.interp));
    }
    let up_id = a.upscale.clone().filter(|u| {
        let l = u.trim().to_ascii_lowercase();
        !(l.is_empty() || l == "none" || l == "off")
    });
    if up_id.is_none() && a.interp == 1 {
        return usage("nothing to preview: give --upscale <model> and/or --interp 2..4");
    }
    let at = parse_time(&a.at)?;
    super::prepare_device(a.device);
    native::init(a.device)?;
    let gpu = super::gpu_caps(a.device)?;
    let settings = Settings {
        upscale: up_id,
        interp: a.interp,
        interp_model: a.interp_model.clone(),
        codec: crate::io::encode::VideoCodec::Hevc,
        cq: 20,
        order: Order::InterpFirst,
        scene_threshold: 1.0,
        tile: a.tile,
        nvenc_preset: "p5".into(),
        output_dir: None,
        name_template: String::new(),
        log_file: None,
    };
    let (up_e, ip_e) = resolve_engines(&settings, &gpu)?;

    let device = CudaDevice::shared(a.device)?;
    let (mut src, _hw) = open_source(&a.input, &device, a.cpu_decode, false)?;
    let info = src.info.clone();
    let f0 = decode_frame_at(&mut src, at)?;
    let f1 = if a.interp > 1 { src.next_frame()? } else { None };
    let pool = Pool::new();
    let stream = Stream::new()?;
    let (w, h) = (f0.width, f0.height);
    let color = pick_color(&info, f0.color, w, h);
    let to_rgb = |f: &crate::io::decode::GpuFrame| -> Result<Arc<Tensor>> {
        let t = Tensor::alloc_exact(&pool, f.width, f.height, DType::Fp16)?;
        native::yuv_to_rgb(&f.yuv(), color.matrix, color.range, t.ptr(), DType::Fp16, f.width, f.height, Some(&stream))?;
        stream.sync()?;
        Ok(Arc::new(t))
    };
    let t0 = to_rgb(&f0)?;
    let t1 = match &f1 {
        Some(f) => Some(to_rgb(f)?),
        None => None,
    };
    drop((f0, f1));

    let mut upscaler = match &up_e {
        Some(e) => Some(Upscaler::new(e, pool.clone(), a.tile)?),
        None => None,
    };
    let scale = upscaler.as_ref().map(|u| u.scale as usize).unwrap_or(1);
    let (ow, oh) = (w as usize * scale, h as usize * scale);

    // before: the source frame, bicubic-scaled to the result size
    let before_img = resize_bicubic(&download(&t0, &stream)?, ow, oh);
    let mut panels: Vec<(String, PlanarImage)> = vec![("before".into(), before_img)];

    if let Some(u) = upscaler.as_mut() {
        let r = u.upscale(&t0)?;
        panels.push(("after".into(), download(&r, &stream)?));
    }
    if let Some(e) = &ip_e {
        let mut ip = Interpolator::new(e, a.interp, pool.clone())?;
        ip.check_fit(w, h)?;
        let t1 = t1.clone().unwrap_or_else(|| t0.clone());
        if Arc::ptr_eq(&t1, &t0) {
            eprintln!("note: no frame follows the selected position; the interpolated panel repeats the frame");
        }
        let mut out = Vec::new();
        ip.process(Item { t: t0.clone(), cut: false }, &mut out)?;
        ip.process(Item { t: t1, cut: false }, &mut out)?;
        // out = [frame0, mid(1/N), ...]
        let mid = out.get(1).context("interpolation produced no frame")?.t.clone();
        let mid = if mid.is_exact() { mid } else { Arc::new(crop_exact(&mid, &pool, &stream)?) };
        stream.sync()?;
        let mid = match upscaler.as_mut() {
            Some(u) => u.upscale(&mid)?,
            None => mid,
        };
        panels.push(("interp".into(), download(&mid, &stream)?));
    }

    if a.split {
        let stem = a.out.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "preview".into());
        let dir = a.out.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        for (name, img) in &panels {
            let p = dir.join(format!("{stem}_{name}.png"));
            write_png(&p, img.w, img.h, &to_rgb8(img))?;
            println!("wrote {}", p.display());
        }
    } else {
        let rgb: Vec<(usize, usize, Vec<u8>)> = panels.iter().map(|(_, i)| (i.w, i.h, to_rgb8(i))).collect();
        let (cw, ch, px) = compose_side_by_side(&rgb, 6);
        write_png(&a.out, cw, ch, &px)?;
        let legend: Vec<&str> = panels.iter().map(|(n, _)| n.as_str()).collect();
        println!("wrote {} ({}x{}; left to right: {})", a.out.display(), cw, ch, legend.join(" | "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_parsing() {
        assert_eq!(parse_time("83.5").unwrap(), 83.5);
        assert_eq!(parse_time("01:23").unwrap(), 83.0);
        assert_eq!(parse_time("00:01:23").unwrap(), 83.0);
        assert!((parse_time("1:02:03.25").unwrap() - 3723.25).abs() < 1e-9);
        assert!(parse_time("abc").is_err());
        assert!(parse_time("1:2:3:4").is_err());
        assert!(parse_time("-5").is_err());
    }

    #[test]
    fn resize_identity_and_constant() {
        let img = PlanarImage { w: 4, h: 3, data: (0..36).map(|v| v as f32 / 36.0).collect() };
        let same = resize_bicubic(&img, 4, 3);
        assert_eq!(same.data, img.data);
        let flat = PlanarImage { w: 5, h: 5, data: vec![0.4; 75] };
        let up = resize_bicubic(&flat, 10, 10);
        assert!(up.data.iter().all(|v| (v - 0.4).abs() < 1e-5));
        let down = resize_bicubic(&flat, 2, 3);
        assert!(down.data.iter().all(|v| (v - 0.4).abs() < 1e-5));
        assert_eq!((up.w, up.h, up.data.len()), (10, 10, 300));
    }

    #[test]
    fn compose_and_rgb8() {
        let a = (2usize, 2usize, vec![255u8; 12]);
        let b = (2usize, 2usize, vec![0u8; 12]);
        let (w, h, px) = compose_side_by_side(&[a, b], 1);
        assert_eq!((w, h), (5, 2));
        assert_eq!(&px[0..3], &[255, 255, 255]);
        assert_eq!(&px[6..9], &[128, 128, 128]); // gap
        assert_eq!(&px[9..12], &[0, 0, 0]);
        let img = PlanarImage { w: 1, h: 1, data: vec![1.5, 0.0, 0.5] };
        assert_eq!(to_rgb8(&img), vec![255, 0, 128]);
    }
}
