//! Model/engine management: manifest, local store, engine selection, download, verify.
//!
//! Public API pinned in docs/contracts.md section 3; additions are marked "extra".

pub mod download;
pub mod error;
pub mod list;
pub mod manifest;
pub mod select;
pub mod store;
pub mod verify;

#[cfg(test)]
pub(crate) mod testutil;

use std::path::PathBuf;

pub use download::{cmd_pull, pull_dir};
pub use error::ModelError;
pub use list::cmd_list;
pub use manifest::{EngineEntry, Manifest, ModelEntry};
pub use store::{ManifestSource, ModelStore};
pub use verify::{cmd_verify, sha256_file, VerifyResult, VerifyStatus};

/// GPU capabilities as reported by `fb_gpu_info`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GpuCaps {
    pub name: String,
    pub cc_major: i32,
    pub cc_minor: i32,
    /// CUDA driver version (1000*major + 10*minor), e.g. 13010.
    pub driver_version: i32,
    pub total_mem: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    Upscale,
    Interp,
}

impl std::fmt::Display for ModelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ModelKind::Upscale => "upscale",
            ModelKind::Interp => "interp",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedEngine {
    pub id: String,
    pub kind: ModelKind,
    pub scale: u32,
    pub io_dtype_fp16: bool,
    pub pad_multiple: u32,
    pub path: PathBuf,
    pub min_shape: [i64; 4],
    pub opt_shape: [i64; 4],
    pub max_shape: [i64; 4],
    /// extra: engine arch actually selected ("ampere+" or "sm75").
    pub arch: String,
}
