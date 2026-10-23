# Contracts between components (v1 implementation)

This file pins the interfaces so components can be built in parallel. Change it only
together with every consumer.

## 0. Dev environment (no system installs; everything project-local)

- Python venv: `.venv/` (Python 3.12, created with `uv`). Use `.venv/Scripts/python.exe`.
  Packages: tensorrt-cu12 11.4 (builder + lean/dispatch runtimes as DLLs under
  `.venv/Lib/site-packages/tensorrt_libs/`), torch/torchvision (CPU, export only), onnx,
  onnxruntime (CPU), onnxslim (used instead of onnxsim), spandrel, numpy, opencv-python,
  cmake, ninja, libclang (for bindgen), nvidia-cuda-nvcc-cu12 12.9, nvidia-cuda-runtime-cu12 12.9,
  nvidia-cuda-cccl-cu12.
- CUDA 12.9 toolkit (nvcc, cudart, cccl from NVIDIA redist archives): `.tools/cuda`
  (`bin/nvcc.exe`, `include/`, `lib/x64/cudart.lib`, `bin/cudart64_12.dll`). Verified working on sm_86.
- TensorRT 11.4 public headers: `.tools/tensorrt/include` (from github NVIDIA/TensorRT v11.4).
  No .lib files: fb_native loads `nvinfer_lean_11.dll` dynamically (LoadLibrary +
  GetProcAddress of the extern "C" `createInferRuntime_INTERNAL`).
  Lean runtime DLL: `.venv/Lib/site-packages/tensorrt_lean_libs/nvinfer_lean_11.dll`;
  full builder DLLs (python `tensorrt` module): `.venv/Lib/site-packages/tensorrt_libs/`.
- **`tools\devenv.cmd <cmd...>`** sets up MSVC (vcvars64), CUDA_PATH, PATH (nvcc, cmake, ninja,
  TRT DLLs, FFmpeg DLLs), LIBCLANG_PATH, FFMPEG_DIR, FB_TRT_LEAN_DLL. From Git Bash run e.g.
  `cmd //c "tools\\devenv.cmd cmake --version"`.
- MSVC: VS 2022 Community, MSVC 14.44 (`vcvars64.bat`).
- FFmpeg (dev): `C:/ffmpeg-8.0.1-full_build-shared` (headers `include/`, import libs `lib/`,
  DLLs `bin/`). This build is GPL and is ONLY for development; release must use the LGPL
  build from `third_party/ffmpeg/`.
- GPU on this machine: RTX 3050 6 GB, sm_86, driver 591.74 (CUDA 13.1). No Turing card ->
  `sm75` engines cannot be built here (documented, skipped).
- Decision (phase 0): **cu12** (CUDA 12.9 runtime + TensorRT 11.4 cu12). TRT 11 still ships cu12.

## 1. Engine tensor conventions (models/ <-> app/)

All engines: batch 1, NCHW, RGB planar, values in [0,1]. I/O dtype given per model in the
manifest (`io_dtype`: `"fp16"` or `"fp32"`).

| kind | inputs | outputs |
|---|---|---|
| `upscale` | `input` `[1,3,H,W]` | `output` `[1,3,H*scale,W*scale]` |
| `interp`  | `img0` `[1,3,H,W]`, `img1` `[1,3,H,W]`, `timestep` `[1,1,1,1]` (same dtype, value in (0,1)) | `output` `[1,3,H,W]` |

- `pad_multiple`: H and W fed to the engine must be multiples of this (RIFE: 64; upscale: usually 1 or 8).
- Dynamic H/W with one optimization profile; the min/opt/max of profile 0 are recorded in the
  manifest (and also readable at runtime through `fb_engine_io_desc`).
- Engines are built with `VERSION_COMPATIBLE` + `EXCLUDE_LEAN_RUNTIME` + hardware compatibility
  `AMPERE_PLUS` (portable fallback; optional per-GPU `SAME_COMPUTE_CAPABILITY` engines `sm<cc>`, see section 2),
  and are run through the TRT **lean** runtime (`nvinfer_lean_11.dll`).

## 2. `models/manifest.json` schema (schema 1)

```json
{
  "schema": 1,
  "trt_version": "11.4.0",
  "cuda_major": 12,
  "base_url": "https://github.com/<owner>/filming-better/releases/download/engines-v1/",
  "models": [
    {
      "id": "span_x2",
      "kind": "upscale",               // "upscale" | "interp"
      "scale": 2,                      // upscale factor; 1 for interp
      "io_dtype": "fp16",
      "pad_multiple": 1,
      "description": "SPAN 2x (fast default)",
      "license": "...", "source": "https://...",
      "engines": [
        {
          "arch": "ampere+",           // "ampere+" (portable, cc >= 8.0) | "sm<major><minor>" (exact cc: sm75, sm80, sm86, sm89, sm120 ...)
          "file": "span_x2.ampere+.engine",
          "sha256": "<hex>",
          "size": 12345678,
          "min_shape": [1,3,64,64], "opt_shape": [1,3,540,960], "max_shape": [1,3,1088,1920],
          "fps_1080p": 41.2,           // measured, null if not measured
          "build_gpu": "NVIDIA GeForce RTX 3050"
        }
      ]
    }
  ]
}
```

Engine file URL = `base_url + file`. Local dev engines live in `models/engines/`.

**Arch tags and selection (still schema 1; additive).** A model may list several engines, one per tag:
`ampere+` is the portable engine (hardware compatibility `AMPERE_PLUS`, runs on every cc >= 8.0, but is
slower and ~3x larger), and `sm<major><minor>` (`SAME_COMPUTE_CAPABILITY`, file `<id>.sm86.engine`) is the
exact-GPU engine (near native speed). For a GPU with compute capability `M.m` the app uses the first
available of:

1. the engine whose arch is exactly `sm{M}{m}` (cc 8.6 -> `sm86`, cc 12.0 -> `sm120`);
2. `ampere+`, when cc >= 8.0;
3. (cc == 7.5: the exact tag is `sm75`, and there is no other fallback);
4. otherwise an error (GPU unsupported, or the model has no engine for it).

`ampere+` is never used on Turing, and a foreign tag (`sm86` on a cc 8.9 GPU) is never used. The same order
applies to `resolve`, `matching_engines`, `cmd_list` and `cmd_pull`. `pull` downloads the best-ranked engine for
the GPU. `resolve`/`list` rank the same way but an engine that is *installed* beats a better-ranked one that is
not (so a portable bundle that ships only `ampere+` keeps working with a manifest that also lists `sm86`). Older
apps that only know `ampere+`/`sm75` ignore the other entries, so a manifest with extra `sm<cc>` engines stays
valid for them. Keep the `ampere+` entry for every model.

## 3. Rust app module ownership (`app/src/`)

- **core agent**: `main.rs`, `native.rs`, `build.rs`, `io/`, `pipeline/`, `cli/mod.rs`,
  `cli/run.rs`, `cli/preview.rs`, `cli/progress.rs`, `cli/errors.rs`, `cli/gpu_info.rs`.
- **models/config agent**: `models/` (manifest, store, selection, pull, verify, list),
  `cli/config.rs` (TOML config + presets), `report.rs` (zip).

Public API that `models/config` agent provides (core agent consumes):

```rust
// models/mod.rs
pub struct GpuCaps { pub name: String, pub cc_major: i32, pub cc_minor: i32,
                     pub driver_version: i32, pub total_mem: u64 }
pub enum ModelKind { Upscale, Interp }
pub struct ResolvedEngine {
    pub id: String, pub kind: ModelKind, pub scale: u32, pub io_dtype_fp16: bool,
    pub pad_multiple: u32, pub path: std::path::PathBuf,
    pub min_shape: [i64;4], pub opt_shape: [i64;4], pub max_shape: [i64;4],
}
pub struct ModelStore { /* search dirs */ }
impl ModelStore {
    /// Search order: $FB_MODELS_DIR, <exe_dir>/models, %LOCALAPPDATA%/filming-better/models,
    /// and (dev) <repo>/models/engines found by walking up from exe dir. Manifest is read from
    /// the first dir that has manifest.json, falling back to the embedded copy.
    pub fn open() -> anyhow::Result<ModelStore>;
    pub fn manifest(&self) -> &Manifest;
    /// Pick the right engine for this GPU and return its local path (error with a fix hint
    /// "run `filming-better models pull <id>`" if not present locally).
    pub fn resolve(&self, model_id: &str, gpu: &GpuCaps) -> anyhow::Result<ResolvedEngine>;
}
pub fn cmd_list(gpu: Option<&GpuCaps>) -> anyhow::Result<()>;
pub fn cmd_pull(ids: &[String], gpu: &GpuCaps, force: bool) -> anyhow::Result<()>;
pub fn cmd_verify() -> anyhow::Result<()>;

// cli/config.rs
#[derive(Default, Clone, serde::Deserialize)]
pub struct RunDefaults { pub preset: Option<String>, pub upscale: Option<String>,
    pub interp: Option<u32>, pub codec: Option<String>, pub cq: Option<u32>,
    pub order: Option<String>, pub scene_threshold: Option<f32>,
    pub output_dir: Option<PathBuf>, pub name_template: Option<String>,
    pub tile: Option<u32>, pub log_file: Option<PathBuf> }
pub fn load_config(explicit: Option<&Path>) -> anyhow::Result<RunDefaults>; // --config or %APPDATA%/filming-better/config.toml
pub struct Preset { pub upscale: Option<&'static str>, pub interp: u32, pub codec: &'static str, pub cq: u32 }
pub fn preset(name: &str) -> Option<Preset>; // fast | balanced | quality

// report.rs
pub fn create_report(out_zip: &Path, log_file: Option<&Path>, system_info: &str) -> anyhow::Result<()>;
```

## 4. Exit codes (cli/errors.rs)

| code | meaning |
|---|---|
| 0 | success |
| 1 | generic error |
| 2 | bad usage / arguments |
| 3 | no NVIDIA GPU |
| 4 | driver too old |
| 5 | out of VRAM |
| 6 | unsupported codec / format |
| 7 | model/engine missing or invalid |
| 8 | some files in a batch failed |
| 130 | cancelled by Ctrl+C |
