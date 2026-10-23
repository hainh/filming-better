//! Typed errors so the CLI can map them to exit codes
//! (`err.downcast_ref::<ModelError>()` -> exit code 7, except `UnsupportedGpu`/`DriverTooOld`).

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("GPU not supported (needs RTX 20 or newer): {name}, compute capability {major}.{minor}")]
    UnsupportedGpu { name: String, major: i32, minor: i32 },

    #[error("unknown model '{id}'. Available models: {available}. Run `filming-better models list`")]
    UnknownModel { id: String, available: String },

    #[error("model '{id}' is ambiguous (matches: {matches}); use the full id")]
    AmbiguousModel { id: String, matches: String },

    #[error("model '{id}' has no engine for {arch} GPUs in the manifest")]
    NoEngineForArch { id: String, arch: String },

    #[error("engine for '{id}' is not installed (looked for {file}). Fix: run `filming-better models pull {id}`")]
    NotInstalled { id: String, file: String },

    #[error("engine file {path} looks damaged (size {actual} bytes, expected {expected}). Fix: run `filming-better models pull {id} --force`")]
    Corrupt {
        id: String,
        path: String,
        actual: u64,
        expected: u64,
    },

    #[error("NVIDIA driver too old: it supports CUDA {have}, engines need CUDA {need}.x. Fix: update the NVIDIA driver")]
    DriverTooOld { have: String, need: u32 },

    #[error("engines were built for TensorRT {built} but the runtime is {runtime}; they are not compatible. Fix: update filming-better and run `filming-better models pull --force`")]
    TrtMismatch { built: String, runtime: String },

    #[error("download failed: {0}")]
    Download(String),
}
