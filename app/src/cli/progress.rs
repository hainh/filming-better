//! Progress reporting: an indicatif bar on stderr, or JSON lines on stdout.

use crate::pipeline::Ctl;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressMode {
    Bar,
    Json,
    None,
}

impl ProgressMode {
    pub fn parse(s: &str) -> Option<ProgressMode> {
        match s.to_ascii_lowercase().as_str() {
            "bar" => Some(ProgressMode::Bar),
            "json" => Some(ProgressMode::Json),
            "none" | "off" => Some(ProgressMode::None),
            _ => None,
        }
    }
}

/// One JSON progress line: `{"frame":N,"total":T,"fps":F,"eta":S}` (`total`/`eta` are null when unknown).
pub fn json_line(frame: u64, total: u64, fps: f64, eta: Option<f64>) -> String {
    let total_s = if total > 0 { total.to_string() } else { "null".to_string() };
    let eta_s = match eta {
        Some(e) if e.is_finite() => format!("{e:.1}"),
        _ => "null".to_string(),
    };
    format!("{{\"frame\":{frame},\"total\":{total_s},\"fps\":{fps:.2},\"eta\":{eta_s}}}")
}

pub fn eta_secs(done: u64, total: u64, fps: f64) -> Option<f64> {
    if total == 0 || fps <= 0.0 {
        return None;
    }
    Some(total.saturating_sub(done) as f64 / fps)
}

pub struct Reporter {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Reporter {
    pub fn start(ctl: Arc<Ctl>, mode: ProgressMode, label: String) -> Reporter {
        let stop = Arc::new(AtomicBool::new(false));
        if mode == ProgressMode::None {
            return Reporter { stop, handle: None };
        }
        let stop2 = stop.clone();
        let handle = std::thread::spawn(move || {
            let t0 = Instant::now();
            let bar = if mode == ProgressMode::Bar {
                let b = ProgressBar::with_draw_target(None, ProgressDrawTarget::stderr_with_hz(5));
                b.set_style(
                    ProgressStyle::with_template("{prefix} [{bar:30.cyan/blue}] {pos}/{len} frames  {msg}  ETA {eta}")
                        .unwrap()
                        .progress_chars("=> "),
                );
                b.set_prefix(label);
                Some(b)
            } else {
                None
            };
            let mut last_json = Instant::now() - Duration::from_secs(10);
            let mut last = (0u64, 0u64, 0.0f64);
            loop {
                let done = ctl.done.load(Ordering::Relaxed);
                let total = ctl.total.load(Ordering::Relaxed);
                let el = t0.elapsed().as_secs_f64();
                let fps = if el > 0.0 { done as f64 / el } else { 0.0 };
                last = (done, total, fps);
                if let Some(b) = &bar {
                    if total > 0 {
                        b.set_length(total.max(done));
                    }
                    b.set_position(done);
                    b.set_message(format!("{fps:5.1} fps"));
                } else if last_json.elapsed() >= Duration::from_millis(500) {
                    last_json = Instant::now();
                    let line = json_line(done, total, fps, eta_secs(done, total, fps));
                    let mut out = std::io::stdout().lock();
                    let _ = writeln!(out, "{line}");
                    let _ = out.flush();
                }
                if stop2.load(Ordering::Relaxed) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            if let Some(b) = bar {
                b.finish_and_clear();
            } else {
                let (done, _, fps) = last;
                let mut out = std::io::stdout().lock();
                let _ = writeln!(out, "{}", json_line(done, done, fps, Some(0.0)));
                let _ = out.flush();
            }
        });
        Reporter { stop, handle: Some(handle) }
    }

    pub fn finish(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Reporter {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// `1h02m03s` style.
pub fn fmt_duration(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    if s >= 3600 {
        format!("{}h{:02}m{:02}s", s / 3600, (s % 3600) / 60, s % 60)
    } else if s >= 60 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_shape() {
        let l = json_line(120, 1000, 59.97, Some(14.7));
        let v: serde_json::Value = serde_json::from_str(&l).unwrap();
        assert_eq!(v["frame"], 120);
        assert_eq!(v["total"], 1000);
        assert!((v["fps"].as_f64().unwrap() - 59.97).abs() < 1e-6);
        assert!((v["eta"].as_f64().unwrap() - 14.7).abs() < 1e-6);
        let l = json_line(5, 0, 0.0, None);
        let v: serde_json::Value = serde_json::from_str(&l).unwrap();
        assert!(v["total"].is_null() && v["eta"].is_null());
    }

    #[test]
    fn eta() {
        assert_eq!(eta_secs(50, 100, 10.0), Some(5.0));
        assert_eq!(eta_secs(0, 0, 10.0), None);
        assert_eq!(eta_secs(10, 100, 0.0), None);
        assert_eq!(eta_secs(120, 100, 10.0), Some(0.0));
    }

    #[test]
    fn durations() {
        assert_eq!(fmt_duration(5.2), "5s");
        assert_eq!(fmt_duration(65.0), "1m05s");
        assert_eq!(fmt_duration(3723.0), "1h02m03s");
    }

    #[test]
    fn modes() {
        assert_eq!(ProgressMode::parse("JSON"), Some(ProgressMode::Json));
        assert_eq!(ProgressMode::parse("bar"), Some(ProgressMode::Bar));
        assert_eq!(ProgressMode::parse("x"), None);
    }
}
