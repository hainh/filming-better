//! `filming-better gpu-info`: GPU, VRAM, driver, runtimes and matching engines.

use crate::io::encode::{nvenc_supports, VideoCodec};
use crate::models::ModelStore;
use crate::native;
use anyhow::Result;
use std::fmt::Write as _;

fn cuda_ver(v: i32) -> String {
    format!("{}.{}", v / 1000, (v % 1000) / 10)
}

fn mib(b: u64) -> u64 {
    b / (1024 * 1024)
}

/// Multi-line description of the machine (also embedded in `report` zips).
pub fn system_info(device: i32) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "filming-better {}", env!("CARGO_PKG_VERSION"));
    let _ = writeln!(s, "os: {} {}", std::env::consts::OS, std::env::consts::ARCH);
    match native::init(device).and_then(|_| native::gpu_info(device)) {
        Ok(g) => {
            let _ = writeln!(
                s,
                "gpu: {} (device {}, compute capability {}.{}, {} SMs)",
                g.name, g.device_index, g.cc_major, g.cc_minor, g.multiprocessor_count
            );
            let mem = native::mem_info().ok();
            match mem {
                Some((free, total)) => {
                    let _ = writeln!(s, "vram: {} MiB total, {} MiB free", mib(total), mib(free));
                }
                None => {
                    let _ = writeln!(s, "vram: {} MiB total", mib(g.total_mem));
                }
            }
            let ok = g.driver_version >= native::MIN_DRIVER_VERSION;
            let _ = writeln!(
                s,
                "driver: supports CUDA {} (needs {} or newer: {})",
                cuda_ver(g.driver_version),
                cuda_ver(native::MIN_DRIVER_VERSION),
                if ok { "ok" } else { "TOO OLD - update the NVIDIA driver" }
            );
            let _ = writeln!(s, "cuda runtime (fb_native): {}", cuda_ver(g.runtime_version));
            let enc = |c: VideoCodec| if nvenc_supports(c, g.cc_major, g.cc_minor) { "yes" } else { "no" };
            let _ = writeln!(
                s,
                "nvenc: h264 {}, hevc {}, av1 {}{}",
                enc(VideoCodec::H264),
                enc(VideoCodec::Hevc),
                enc(VideoCodec::Av1),
                if g.cc_major * 10 + g.cc_minor < 89 { " (AV1 encoding needs RTX 40-series or newer)" } else { "" }
            );
            match native::trt_version() {
                Ok(v) => {
                    let _ = writeln!(s, "tensorrt lean runtime: {}.{}.{}", v / 10000, (v / 100) % 100, v % 100);
                }
                Err(e) => {
                    let _ = writeln!(s, "tensorrt lean runtime: NOT AVAILABLE ({})", e.message);
                }
            }
            let caps = crate::models::GpuCaps {
                name: g.name.clone(),
                cc_major: g.cc_major,
                cc_minor: g.cc_minor,
                driver_version: g.driver_version,
                total_mem: g.total_mem,
            };
            match ModelStore::open() {
                Ok(store) => {
                    let m = store.matching_engines(&caps);
                    if m.is_empty() {
                        let _ = writeln!(s, "engines: none match this GPU");
                    } else {
                        let _ = writeln!(s, "engines for this GPU:");
                        for (id, arch, installed) in m {
                            let _ = writeln!(
                                s,
                                "  {id:<20} {arch:<8} {}",
                                if installed { "installed" } else { "not installed (filming-better models pull)" }
                            );
                        }
                    }
                }
                Err(e) => {
                    let _ = writeln!(s, "engines: {e:#}");
                }
            }
        }
        Err(e) => {
            let _ = writeln!(s, "gpu: NOT AVAILABLE ({e})");
        }
    }
    unsafe {
        let v = rsmpeg::ffi::av_version_info();
        let v = if v.is_null() { String::new() } else { std::ffi::CStr::from_ptr(v).to_string_lossy().into_owned() };
        let _ = writeln!(s, "ffmpeg: {v} (avcodec {})", rsmpeg::ffi::avcodec_version() >> 16);
    }
    s
}

pub fn cmd_gpu_info(device: i32) -> Result<()> {
    // surface "no GPU" / "driver too old" as proper exit codes, but still print what we know
    let init = native::init(device);
    print!("{}", system_info(device));
    init.map(|_| ()).map_err(Into::into)
}
