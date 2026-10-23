//! Exit codes (docs/contracts.md section 4) and the mapping from errors to (code, message, hint).

use crate::io::IoError;
use crate::models::ModelError;
use crate::native::{self, NativeError};

pub const EXIT_OK: i32 = 0;
pub const EXIT_GENERIC: i32 = 1;
pub const EXIT_USAGE: i32 = 2;
pub const EXIT_NO_GPU: i32 = 3;
pub const EXIT_DRIVER_TOO_OLD: i32 = 4;
pub const EXIT_OUT_OF_VRAM: i32 = 5;
pub const EXIT_UNSUPPORTED: i32 = 6;
pub const EXIT_MODEL: i32 = 7;
pub const EXIT_BATCH_FAILED: i32 = 8;
pub const EXIT_CANCELLED: i32 = 130;

/// Errors raised by the CLI layer itself.
#[derive(Debug, thiserror::Error)]
pub enum CliError {
    #[error("{0}")]
    Usage(String),
    #[error("cancelled")]
    Cancelled,
    #[error("{failed} of {total} files failed")]
    BatchFailed { failed: usize, total: usize },
}

/// What to show the user and which code to exit with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    pub code: i32,
    pub message: String,
    pub hint: Option<String>,
}

fn cuda_ver(v: i32) -> String {
    format!("{}.{}", v / 1000, (v % 1000) / 10)
}

fn native_report(e: &NativeError) -> Report {
    use native::code::*;
    let (code, hint): (i32, Option<String>) = match e.code {
        NO_GPU => (
            EXIT_NO_GPU,
            Some("an NVIDIA GPU (RTX 20-series or newer) with an up-to-date driver is required".into()),
        ),
        DRIVER_TOO_OLD => (
            EXIT_DRIVER_TOO_OLD,
            Some(format!(
                "your driver supports CUDA {}, filming-better needs CUDA {} or newer. Update the NVIDIA driver from nvidia.com/drivers",
                if e.driver_version > 0 { cuda_ver(e.driver_version) } else { "?".into() },
                cuda_ver(native::MIN_DRIVER_VERSION)
            )),
        ),
        OUT_OF_MEMORY => (
            EXIT_OUT_OF_VRAM,
            Some("out of GPU memory: close other GPU programs, lower the load with --tile 512 or use --upscale none".into()),
        ),
        TRT_LOAD => (
            EXIT_MODEL,
            Some("TensorRT runtime (nvinfer_lean_11.dll) not found: keep it next to filming-better.exe or set FB_TRT_LEAN_DLL".into()),
        ),
        ENGINE => (
            EXIT_MODEL,
            Some("the engine could not be loaded or run: run `filming-better models verify`, then `models pull <id> --force`".into()),
        ),
        UNSUPPORTED => (EXIT_UNSUPPORTED, None),
        IO => (EXIT_MODEL, Some("run `filming-better models verify` to check the engine files".into())),
        _ => (EXIT_GENERIC, None),
    };
    let message = if e.code == DRIVER_TOO_OLD {
        "NVIDIA driver too old".to_string()
    } else {
        e.to_string()
    };
    Report { code, message, hint }
}

fn model_report(e: &ModelError) -> Report {
    let code = match e {
        ModelError::UnsupportedGpu { .. } => EXIT_NO_GPU,
        ModelError::DriverTooOld { .. } => EXIT_DRIVER_TOO_OLD,
        _ => EXIT_MODEL,
    };
    // ModelError messages already carry their own "Fix: ..." text.
    Report { code, message: e.to_string(), hint: None }
}

fn io_report(e: &IoError) -> Report {
    match e {
        IoError::Unsupported(m) => Report {
            code: EXIT_UNSUPPORTED,
            message: m.clone(),
            // encoder errors already say what to do; only format errors get the generic hint
            hint: if m.contains("encoder") || m.contains("NVENC") {
                None
            } else {
                Some("supported: H.264/HEVC/AV1/VP9 8/10-bit 4:2:0 SDR progressive video; output through NVENC".into())
            },
        },
        IoError::Ff { what, code, msg } => {
            let hint = if what.starts_with("cannot open") {
                Some("check that the path exists and the file is a valid video".to_string())
            } else {
                None
            };
            let _ = code;
            Report { code: EXIT_GENERIC, message: format!("{what}: {msg}"), hint }
        }
        IoError::Other(m) => Report { code: EXIT_GENERIC, message: m.clone(), hint: None },
    }
}

/// Classify an error chain. The innermost recognised typed error wins (context layers are
/// plain strings), the message is the whole chain.
pub fn classify(err: &anyhow::Error) -> Report {
    let mut found: Option<Report> = None;
    for cause in err.chain() {
        if let Some(c) = cause.downcast_ref::<CliError>() {
            found = Some(match c {
                CliError::Usage(m) => Report { code: EXIT_USAGE, message: m.clone(), hint: Some("see `filming-better --help`".into()) },
                CliError::Cancelled => Report { code: EXIT_CANCELLED, message: "cancelled".into(), hint: None },
                CliError::BatchFailed { .. } => Report { code: EXIT_BATCH_FAILED, message: c.to_string(), hint: None },
            });
        } else if let Some(n) = cause.downcast_ref::<NativeError>() {
            found = Some(native_report(n));
        } else if let Some(m) = cause.downcast_ref::<ModelError>() {
            found = Some(model_report(m));
        } else if let Some(i) = cause.downcast_ref::<IoError>() {
            found = Some(io_report(i));
        }
    }
    let mut r = found.unwrap_or(Report { code: EXIT_GENERIC, message: String::new(), hint: None });
    if r.message.is_empty() || !matches!(r.code, EXIT_USAGE | EXIT_CANCELLED | EXIT_BATCH_FAILED) {
        let chain = format!("{err:#}");
        if r.code == EXIT_DRIVER_TOO_OLD && r.message == "NVIDIA driver too old" {
            // keep the short message
        } else {
            r.message = chain;
        }
    }
    r
}

/// Print `error: ...` (+ `hint: ...`) to stderr.
pub fn print_report(r: &Report) {
    eprintln!("error: {}", r.message);
    if let Some(h) = &r.hint {
        eprintln!("hint: {h}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    #[test]
    fn native_codes_map() {
        let cases = [
            (native::code::NO_GPU, EXIT_NO_GPU),
            (native::code::DRIVER_TOO_OLD, EXIT_DRIVER_TOO_OLD),
            (native::code::OUT_OF_MEMORY, EXIT_OUT_OF_VRAM),
            (native::code::TRT_LOAD, EXIT_MODEL),
            (native::code::ENGINE, EXIT_MODEL),
            (native::code::UNSUPPORTED, EXIT_UNSUPPORTED),
            (native::code::CUDA, EXIT_GENERIC),
        ];
        for (nc, exp) in cases {
            let e: anyhow::Error = NativeError::new(nc, "op", "msg").into();
            assert_eq!(classify(&e).code, exp, "native code {nc}");
        }
    }

    #[test]
    fn driver_hint_mentions_versions() {
        let mut n = NativeError::new(native::code::DRIVER_TOO_OLD, "fb_init", "old");
        n.driver_version = 11040;
        let r = classify(&n.into());
        assert_eq!(r.code, EXIT_DRIVER_TOO_OLD);
        let h = r.hint.unwrap();
        assert!(h.contains("11.4") && h.contains("12.8"), "{h}");
    }

    #[test]
    fn typed_error_found_through_context() {
        let e = anyhow::Error::new(IoError::Unsupported("AV1 NVENC not supported".into()))
            .context("encode thread");
        let r = classify(&e);
        assert_eq!(r.code, EXIT_UNSUPPORTED);
        assert!(r.message.contains("encode thread") && r.message.contains("AV1"));
    }

    #[test]
    fn model_errors() {
        let e = anyhow::Error::new(ModelError::NotInstalled { id: "x".into(), file: "x.engine".into() });
        assert_eq!(classify(&e).code, EXIT_MODEL);
        let e = anyhow::Error::new(ModelError::UnsupportedGpu { name: "g".into(), major: 6, minor: 1 });
        assert_eq!(classify(&e).code, EXIT_NO_GPU);
        let e = anyhow::Error::new(ModelError::DriverTooOld { have: "11.0".into(), need: 12 });
        assert_eq!(classify(&e).code, EXIT_DRIVER_TOO_OLD);
    }

    #[test]
    fn cli_errors() {
        let e: anyhow::Error = CliError::Usage("bad".into()).into();
        assert_eq!(classify(&e).code, EXIT_USAGE);
        let e: anyhow::Error = CliError::Cancelled.into();
        assert_eq!(classify(&e).code, EXIT_CANCELLED);
        let e: anyhow::Error = CliError::BatchFailed { failed: 1, total: 3 }.into();
        assert_eq!(classify(&e).code, EXIT_BATCH_FAILED);
        let e = anyhow::anyhow!("boom").context("while doing x");
        let r = classify(&e);
        assert_eq!(r.code, EXIT_GENERIC);
        assert!(r.message.contains("boom"));
        let r: Result<(), _> = Err(std::io::Error::other("io"));
        let e = r.context("reading").unwrap_err();
        assert_eq!(classify(&e).code, EXIT_GENERIC);
    }
}
