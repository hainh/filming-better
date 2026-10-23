//! Test helpers (no external tempfile dependency).
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

pub struct TempDir(PathBuf);
static N: AtomicU32 = AtomicU32::new(0);

impl TempDir {
    pub fn new(tag: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!(
            "fb-test-{}-{}-{}",
            tag,
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
    pub fn path(&self) -> &Path {
        &self.0
    }
    pub fn join(&self, s: &str) -> PathBuf {
        self.0.join(s)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Serializes tests that touch process environment variables.
pub static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(data))
}

/// A manifest with two models whose engine files are blobs; returns (json, [(file, bytes)]).
pub fn sample_manifest(base_url: &str) -> (String, Vec<(String, Vec<u8>)>) {
    let a = vec![7u8; 1000];
    let b = vec![9u8; 2500];
    let c = vec![3u8; 300];
    let json = format!(
        r#"{{
  "schema": 1, "trt_version": "11.4.0", "cuda_major": 12, "base_url": "{base}",
  "future_top_level_field": 42,
  "models": [
    {{ "id": "span_x2", "kind": "upscale", "scale": 2, "io_dtype": "fp16", "pad_multiple": 1,
       "description": "SPAN", "license": "x", "source": "y", "something_new": true,
       "engines": [
         {{ "arch": "ampere+", "file": "span_x2.ampere+.engine", "sha256": "{ha}", "size": {sa},
            "min_shape": [1,3,64,64], "opt_shape": [1,3,540,960], "max_shape": [1,3,1088,1920],
            "fps_1080p": 41.2, "build_gpu": "RTX 3050" }},
         {{ "arch": "sm75", "file": "span_x2.sm75.engine", "sha256": "{hc}", "size": {sc},
            "min_shape": [1,3,64,64], "opt_shape": [1,3,540,960], "max_shape": [1,3,1088,1920],
            "fps_1080p": null, "build_gpu": null }}
       ] }},
    {{ "id": "rife_v4", "kind": "interp", "scale": 1, "io_dtype": "fp32", "pad_multiple": 64,
       "description": "RIFE", "license": "MIT", "source": "z",
       "engines": [
         {{ "arch": "ampere+", "file": "rife_v4.ampere+.engine", "sha256": "{hb}", "size": {sb},
            "min_shape": [1,3,64,64], "opt_shape": [1,3,576,960], "max_shape": [1,3,1088,1920],
            "fps_1080p": 80.0, "build_gpu": "RTX 3050" }}
       ] }}
  ]
}}"#,
        base = base_url,
        ha = sha256_hex(&a),
        sa = a.len(),
        hb = sha256_hex(&b),
        sb = b.len(),
        hc = sha256_hex(&c),
        sc = c.len()
    );
    (
        json,
        vec![
            ("span_x2.ampere+.engine".into(), a),
            ("rife_v4.ampere+.engine".into(), b),
            ("span_x2.sm75.engine".into(), c),
        ],
    )
}

pub fn gpu(cc_major: i32, cc_minor: i32) -> super::GpuCaps {
    super::GpuCaps {
        name: format!("Test GPU {}.{}", cc_major, cc_minor),
        cc_major,
        cc_minor,
        driver_version: 13010,
        total_mem: 6 << 30,
    }
}

/// Adds an `sm86` engine (file `span_x2.sm86.engine`, 200 bytes of 5s) after the other
/// `span_x2` engines in a `sample_manifest` json; returns the new json and the blob.
pub fn add_sm86_engine(json: &str) -> (String, (String, Vec<u8>)) {
    let blob = vec![5u8; 200];
    let mut m: super::manifest::Manifest = serde_json::from_str(json).unwrap();
    let span = m.models.iter_mut().find(|x| x.id == "span_x2").unwrap();
    let mut e = span.engines[0].clone();
    e.arch = "sm86".into();
    e.file = "span_x2.sm86.engine".into();
    e.sha256 = sha256_hex(&blob);
    e.size = blob.len() as u64;
    e.fps_1080p = Some(55.5);
    span.engines.push(e);
    (serde_json::to_string_pretty(&m).unwrap(), ("span_x2.sm86.engine".into(), blob))
}
