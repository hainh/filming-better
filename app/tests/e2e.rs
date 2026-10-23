//! End-to-end tests: run the built exe on Download.mp4 (repo root). They are skipped when the
//! sample, a CUDA GPU or the engines are not available.

use std::path::{Path, PathBuf};
use std::process::Command;

fn exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_filming-better"))
}

fn sample() -> Option<PathBuf> {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..3 {
        let p = d.join("Download.mp4");
        if p.is_file() {
            return Some(p);
        }
        if !d.pop() {
            break;
        }
    }
    None
}

fn gpu_ok() -> bool {
    Command::new(exe())
        .arg("gpu-info")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn engines_ok() -> bool {
    let o = Command::new(exe()).arg("gpu-info").output().unwrap();
    let t = String::from_utf8_lossy(&o.stdout);
    t.contains("span_x2") && t.lines().any(|l| l.contains("span_x2") && l.trim_end().ends_with("installed") && !l.contains("not installed"))
        && t.lines().any(|l| l.contains("rife_v4 ") && !l.contains("not installed"))
}

fn probe(path: &Path, entries: &str) -> String {
    let o = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries", entries, "-of", "compact"])
        .arg(path)
        .output();
    match o {
        Ok(o) => String::from_utf8_lossy(&o.stdout).into_owned(),
        Err(_) => String::new(),
    }
}

fn outdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fb_e2e_{}_{}", name, std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn passthrough_keeps_duration_and_audio() {
    let Some(src) = sample() else { return };
    if !gpu_ok() {
        return;
    }
    let out = outdir("pass").join("o.mp4");
    let st = Command::new(exe())
        .args(["run", "--upscale", "none", "--interp", "1", "--progress", "none", "--input"])
        .arg(&src)
        .arg("--output")
        .arg(&out)
        .status()
        .unwrap();
    assert!(st.success());
    let p = probe(&out, "stream=codec_name,width,height,nb_frames");
    assert!(p.contains("hevc") && p.contains("width=576") && p.contains("nb_frames=214"), "{p}");
    assert!(p.contains("aac"), "audio must be copied: {p}");
}

#[test]
fn upscale_and_interpolate() {
    let Some(src) = sample() else { return };
    if !gpu_ok() || !engines_ok() {
        return;
    }
    let out = outdir("full").join("o.mp4");
    let st = Command::new(exe())
        .args(["run", "--upscale", "span_x2", "--interp", "2", "--codec", "hevc", "--cq", "20", "--progress", "none", "--input"])
        .arg(&src)
        .arg("--output")
        .arg(&out)
        .status()
        .unwrap();
    assert!(st.success());
    let p = probe(&out, "stream=codec_name,width,height,avg_frame_rate,nb_frames");
    assert!(p.contains("width=1152") && p.contains("height=2048"), "{p}");
    assert!(p.contains("avg_frame_rate=60/1") && p.contains("nb_frames=428"), "{p}");
}

#[test]
fn av1_is_rejected_on_ampere_with_code_6() {
    let Some(src) = sample() else { return };
    let o = Command::new(exe()).arg("gpu-info").output().unwrap();
    let t = String::from_utf8_lossy(&o.stdout);
    if !o.status.success() || !t.contains("av1 no") {
        return; // GPU has AV1 NVENC (or no GPU): nothing to assert
    }
    let out = outdir("av1").join("o.mp4");
    let st = Command::new(exe())
        .args(["run", "--upscale", "none", "--interp", "1", "--codec", "av1", "--progress", "none", "--input"])
        .arg(&src)
        .arg("--output")
        .arg(&out)
        .status()
        .unwrap();
    assert_eq!(st.code(), Some(6));
}

#[test]
fn usage_errors_exit_2() {
    let st = Command::new(exe()).args(["run"]).status().unwrap();
    assert_eq!(st.code(), Some(2));
    let st = Command::new(exe()).args(["run", "--input", "no_such_file.mp4"]).status().unwrap();
    assert_eq!(st.code(), Some(2));
}
