# models/ - dev-side model pipeline (blueprint phase 1)

Turns public pretrained weights into TensorRT 11.4 engines + `manifest.json` that the Rust app loads
through the TensorRT **lean** runtime. Runs on the dev/CI machine only (needs the `.venv`, see
`docs/contracts.md` section 0). Tensor conventions and the manifest schema are pinned in
`docs/contracts.md` sections 1 and 2.

## Run everything

```bash
.venv/Scripts/python.exe models/build_all.py                 # span_x2 animevideov3_x4 rife_v4 rife_v4_lite
.venv/Scripts/python.exe models/build_all.py x4plus          # the big optional model (separate run)
.venv/Scripts/python.exe models/build_all.py --compat        # + portable-vs-native speed check
```

`build_all.py` runs, in order (each script also works standalone, all take model ids):

| step | script | output |
|---|---|---|
| 1 | `download_weights.py` | `models/weights/*` (gitignored). URL, pinned sha256, license are in the `SOURCES` table. |
| 2 | `export_onnx.py` | `models/onnx/<id>.fp32.onnx` and `<id>.fp16.onnx` (opset 17, dynamic H/W, onnxslim). |
| 3 | `validate_onnx.py` | ONNX Runtime (CPU) vs PyTorch on 20 frames of `Download.mp4`; fails unless PSNR > 40 dB. |
| 4 | `build_engines.py` | `models/engines/<id>.ampere+.engine` (portable fallback) + `models/engines/manifest.json` (copied to `models/manifest.json`). |
| 4b | `build_engines.py --arch sm86` | per-GPU engine `<id>.sm86.engine` (`SAME_COMPUTE_CAPABILITY`), added to the manifest next to the `ampere+` entry; run it on a GPU of that cc (any `sm<major><minor>`; refuses on a different GPU). `--arch sm75` needs a Turing GPU, **skipped on this machine** (see below). |
| 5 | `bench_engines.py [--arch sm86]` | lean-runtime load test (every engine whose file exists), fps, TRT-vs-ORT PSNR; fills `fps_1080p` in the manifest. |
| opt | `compat_check.py` | portable (`ampere+`) and per-GPU (`sm<cc>`) vs native-built engine speed (phase-0 criterion). |

Run the python scripts with the project venv (`.venv/Scripts/python.exe`). They add the TensorRT and CUDA
DLL directories to `PATH` themselves (`common.setup_dll_dirs`), so `tools\devenv.cmd` is not required.
Support files: `common.py` (registry of model ids/profiles, frame loading), `rife_arch.py` (inlined RIFE
IFNet), `trt_runner.py` (tiny TensorRT runner on cudart via ctypes; no torch-cuda needed).

## Models

| id | model | kind | scale | license (weights) | source |
|---|---|---|---|---|---|
| `span_x2` | SPAN 2x, `2xNomosUni_span_multijpg` (JPEG/compression-robust, general) | upscale | 2 | CC-BY-4.0 (attribution: Helaman / Phhofm); SPAN arch Apache-2.0 | github.com/Phhofm/models release `2xNomosUni_span_multijpg` |
| `animevideov3_x4` | Real-ESRGAN `realesr-animevideov3` (SRVGGNetCompact) | upscale | 4 | BSD-3-Clause | github.com/xinntao/Real-ESRGAN release v0.2.5.0 |
| `rife_v4` | RIFE 4.26 (Practical-RIFE, `flownet_v4.26.pkl`) | interp | 1 | MIT | github.com/HolyWu/vs-rife release `model` |
| `rife_v4_lite` | RIFE 4.22.lite (`flownet_v4.22.lite.pkl`) | interp | 1 | MIT | same |
| `x4plus` (optional) | Real-ESRGAN `RealESRGAN_x4plus` (RRDBNet 23 blocks) | upscale | 4 | BSD-3-Clause | github.com/xinntao/Real-ESRGAN release v0.1.0 |

Notes
* CC-BY-4.0 for `span_x2` needs attribution in `THIRD_PARTY_NOTICES` (author Helaman; model
  `2xNomosUni_span_multijpg`; https://github.com/Phhofm/models). The official SPAN 2x weights
  (`spanx2_ch48`, Apache-2.0, OpenModelDB) are a drop-in alternative but are trained on clean bicubic data.
* The RIFE architecture (`rife_arch.py`) is inlined from Practical-RIFE / vs-rife (both MIT, notices in the file header).
  The weights contain a training-only `teacher.*` branch that is dropped.
* Upscalers load through spandrel (`SPAN`, `SRVGGNetCompact`, `RRDBNet`).

## Engine conventions (as built, = contracts.md section 1)

Batch 1, NCHW, RGB planar, values in [0,1], **FP16 I/O for every engine** (`io_dtype: "fp16"`).

| engine | inputs (dtype, shape; profile 0 min / opt / max) | output | pad_multiple |
|---|---|---|---|
| `span_x2` | `input` HALF `[1,3,H,W]`; 64x64 / 512x512 / 1088x1088 | `output` HALF `[1,3,2H,2W]` | 1 |
| `animevideov3_x4` | `input` HALF `[1,3,H,W]`; 64x64 / 512x512 / 1088x1088 | `output` HALF `[1,3,4H,4W]` | 1 |
| `rife_v4`, `rife_v4_lite` | `img0`, `img1` HALF `[1,3,H,W]` and `timestep` HALF `[1,1,1,1]` (fixed shape in the profile); H,W 64x64 / 576x1024 / **1920x1920** | `output` HALF `[1,3,H,W]` | 64 |

* RIFE max 1920x1920 covers 1088x1920 (landscape) **and** 1920x1088 (portrait). H and W must each be a
  multiple of 64 (pad with replicate/edge padding, crop afterwards). `timestep` is the interpolation position in (0,1).
* RIFE runs with `scale=1`, `ensemble=False` fixed. The warp / `grid_sample` path is kept in FP32 inside
  the graph (casts in the ONNX), because FP16 sampling-grid coordinates are too coarse at 1080p.
* Upscale engines: the app tiles frames larger than `max_shape` (max 1088x1088). No padding needed.
* Execution-context scratch memory is sized by the max profile: `device_memory_size_v2` =
  975 MiB (span_x2), 506 MiB (animevideov3_x4), 872 MiB (rife_v4 and rife_v4_lite), plus the I/O tensors.
  A user-managed context allocation can use `getDeviceMemorySizeForProfileV2` / per-shape sizes if tighter control is wanted.

### TensorRT 11 specifics

TensorRT 11 has no `BuilderFlag.FP16`: networks are strongly typed, so an engine's precision is the
precision of the ONNX. `export_onnx.py` therefore writes two ONNX files per model: the FP32 one is
validated against PyTorch; the FP16 one (exported from the half-cast torch module, I/O FP16) is what
`build_engines.py` compiles. Engine flags: `VERSION_COMPATIBLE | EXCLUDE_LEAN_RUNTIME`,
`hardware_compatibility_level = AMPERE_PLUS` (portable engine) or `SAME_COMPUTE_CAPABILITY` (`--arch sm<cc>` engine), `STRONGLY_TYPED` network, one profile, workspace 3 GiB
(the build script halves it and retries if the shared GPU runs out of memory, also when TRT silently
skips tactics for lack of memory).

## Lean-runtime loading requirements (for the C++ side)

Verified with the python `tensorrt_lean` bindings (same `nvinfer_lean_11.dll` the C++ code loads), in a process
where **only** `nvinfer_lean_11.dll` was on the search path:

1. The **only** TensorRT file needed at run time is `nvinfer_lean_11.dll` (41 MB; its only static import is
   `KERNEL32.dll`; it loads `nvcuda.dll` from the driver dynamically). Not needed: `nvinfer_vc_plugin_11.dll`,
   `nvinfer_11.dll`, builder resources, cuBLAS/cuDNN, `cudart` (cudart is only loaded because the app/benchmark uses it for
   allocations/streams). (The python package `tensorrt_lean_libs/__init__.py` blindly `LoadLibrary`s every dll in its folder,
   which is why `nvinfer_vc_plugin_11.dll` shows up as loaded in the python case; it is not required.)
2. Entry point: `createInferRuntime_INTERNAL(ILogger&, int32_t NV_TENSORRT_VERSION)` (exported `extern "C"`),
   i.e. `nvinfer1::createInferRuntime(logger)` from the TRT 11.4 headers. Then `runtime->deserializeCudaEngine(blob, size)`.
3. **`setEngineHostCodeAllowed` / `engine_host_code_allowed` is NOT needed for our engines.** Measured:
   * `VERSION_COMPATIBLE` + `EXCLUDE_LEAN_RUNTIME` (what we ship, 5-32 MB): deserializes with the default
     (`getEngineHostCodeAllowed() == false`) and also with `true`.
   * `VERSION_COMPATIBLE` **without** `EXCLUDE_LEAN_RUNTIME` embeds the lean runtime in the plan (span_x2: 42.8 MB instead of 1.4 MB)
     and the lean runtime then refuses it: *"Cannot deserialize engine with lean runtime since
     IRuntime::getEngineHostCodeAllowed() is false (runtime.cpp:187)"*. Only that kind of engine needs
     `runtime->setEngineHostCodeAllowed(true)`. Recommendation: leave it `false` (it is the safer default; it runs code from the
     plan) - and keep verifying the engine sha256 from the manifest before loading.
4. Engines must be built with the same TensorRT 11.4 version family as the lean DLL (`trt_version` in the manifest; version-compatible
   engines are meant to load across later minor versions, not tested here). cu12 only.
5. Execution: `engine->createExecutionContext()`; for each input `context->setInputShape(name, dims)` (must lie inside profile 0),
   `setTensorAddress(name, devptr)` for every input/output (outputs shapes via `getTensorShape` after the input shapes are set),
   `enqueueV3(stream)`. Dynamic H/W with a single profile means no `setOptimizationProfileAsync` call is needed (profile 0 is the default).
   Input shapes outside min/max fail in `setInputShape`.
6. GPU/driver: tested with driver 591.74 (CUDA 13.1 capable). A missing GPU or too-old driver shows up as a null
   runtime/engine and a message on the logger (not tested); the min driver for the cu12 build is decided in phase 0.

## Results (RTX 3050 6 GB, sm_86, TensorRT 11.4.0.106, lean runtime; **GPU was shared at ~100% load with other jobs while measuring**)

fps = best-of-60 runs (min latency; median is 0-50% higher latency because of the other GPU users). So these are
approximations of an idle card, and the real numbers should be re-measured on a quiet GPU.
Upscale rows: the largest supported tile (1088x1088) timed and scaled to a 1920x1080 input frame
(`1 / (t_tile * 1920*1080 / (1088*1088))`). Interp row: one 1920x1088 frame pair.

| model | engine MB | latency (best) | fps_1080p | ONNX FP32 vs PyTorch PSNR (20 frames, min / mean) | TRT FP16 (lean) vs ORT FP32 PSNR (3 real frames, min) |
|---|---|---|---|---|---|
| `span_x2` | 4.73 | 119 ms / 1088x1088 tile | 4.8 | 129.1 / 130.7 dB | 64.7 dB |
| `animevideov3_x4` | 5.95 | 119 ms / 1088x1088 tile | 4.8 | 143.6 / 144.0 dB | 72.3 dB |
| `rife_v4` (4.26) | 32.13 | 70.8 ms / 1920x1088 pair | 14.1 | 126.6 / 133.4 dB | 67.5 dB |
| `rife_v4_lite` (4.22.lite) | 28.84 | 59.9 ms / 1920x1088 pair | 16.7 | 128.9 / 135.0 dB | 71.6 dB |
| `x4plus` (optional) | 154.13 | 910 ms / 512x512 tile | 0.14 | 121.6 / 122.8 dB (2 frames) | not measured |

* All PSNR thresholds are met by a huge margin (ONNX > 40 dB required, TRT FP16 vs FP32 RIFE > 35 dB required).
* RIFE sanity check (`validate_onnx.py`, informational): interpolating frames k and k+2 at t=0.5 gives 34.7 dB against the
  real frame k+1 versus 29.6 dB for a plain 50/50 blend, for both RIFE variants.
* Total of the four default `ampere+` engines: 72 MB (budget: 200 MB; `pull` fetches one engine per model, the sm86 set is another 33 MB). `x4plus` is not in that set: its `ampere+` engine is 154 MB and it runs at
  0.14 fps (profile 64 / 256 / 512 tiles), so it is a download-on-demand quality option only.
* SPAN 2x at 1080p input is compute bound (about 9 TFLOP per frame); expect about 4 fps on this 3050 (shared), faster on bigger cards.
  `animevideov3_x4` is similar. Use smaller inputs / the lite RIFE for speed-sensitive presets.

### Portable engine speed (phase-0 criterion: not more than 15% slower than a natively built engine)

`compat_check.py`, same shapes (profile opt), best-of-40, shared GPU (noisy, +-10%):

| model | native (no compat flags, full runtime) | `SAME_COMPUTE_CAPABILITY` + VC | shipped `AMPERE_PLUS` + VC (lean) |
|---|---|---|---|
| `span_x2` @512x512 | 22.1 ms (1.4 MB) | 24.2 ms (x1.10, 1.5 MB) | 25.7 ms (**x1.16**, 5.0 MB; the later rebuild is 4.7 MB) |
| `animevideov3_x4` @512x512 | 24.4 ms (2.0 MB) | 26.4 ms (x1.08) | 26.5 ms (x1.08, 5.9 MB) |
| `rife_v4` @576x1024 | 13.2 ms (14.0 MB) | 15.7 ms (x1.19) | 19.7 ms (**x1.49**, 32.1 MB) |

Findings: `VERSION_COMPATIBLE` itself costs about 0-2%; the lean runtime itself costs nothing; the cost comes from
`AMPERE_PLUS` hardware compatibility (restricts tactics to kernels valid on every cc >= 8.0 and makes the engines 3x
larger). So the `ampere+` engines **miss the 15% criterion for RIFE (and marginally SPAN)**. Fix (next section): per-compute-capability
engines (`--arch sm86`, `SAME_COMPUTE_CAPABILITY`, near native speed, half the size) listed in the manifest next to the
`ampere+` ones. The app prefers the exact `sm<major><minor>` engine and falls back to `ampere+` (docs/contracts.md section 2).

### Per-GPU `sm86` engines vs portable `ampere+` (RTX 3050, sm_86)

`sm86` is a contract arch tag. The Rust resolver (`app/src/models/select.rs`) picks, for a GPU with cc `M.m`: the engine tagged
exactly `sm{M}{m}`, else `ampere+` (cc >= 8.0), else `sm75` (cc 7.5), else an error. `resolve`, `list`, `matching_engines`
and `pull` use the same order (`pull` downloads the best-ranked engine; `resolve`/`list` prefer an *installed* engine over a
better-ranked missing one, so the portable zip that bundles only `ampere+` keeps working). Engines were built with
`build_engines.py --arch sm86` with the same profiles as the `ampere+` ones; the `ampere+` entries are kept.

Speed, same shapes as above (profile opt: 512x512 / 576x1024), lean runtime. **The GPU was shared (other jobs at 10-99% load)
during every measurement, so all numbers are best-of-N under noise (+-10%; absolute ms swing by 1.5x with clocks and
neighbours). The "ampere+ / sm86" column comes from in-process interleaved runs (8 rounds x 25 runs, min latency, two
repetitions) and is the most reliable; re-measure on an idle card for final numbers.**

| model | ampere+ MB | sm86 MB | ampere+ ms | sm86 ms | ampere+ / sm86 time (interleaved, 2 runs) | sm86 vs native-built (2 compat_check runs) | ampere+ vs native (first table, 2 runs) |
|---|---|---|---|---|---|---|---|
| `span_x2` @512x512 | 4.73 | 1.59 | 37.9 | 32.8 | 1.16 - 1.17x | 0.97x / 0.97x (as fast as native) | 1.14x / 1.23x |
| `animevideov3_x4` @512x512 | 5.95 | 2.24 | 41.6 | 41.9 | 0.98 - 1.00x (no gain) | 1.04x / 1.09x | 1.02x / 1.17x |
| `rife_v4` @576x1024 | 32.13 | 15.59 | 29.3 | 23.9 | 1.19 - 1.23x (quiet-GPU `fb_native` run: 19.9 -> 16.6 ms, 1.19x) | 1.13x / 0.99x | 1.42x / 1.23x |
| `rife_v4_lite` @576x1024 | 28.84 | 13.43 | 23.8 | 21.4 | 1.12 - 1.16x (quiet-GPU `fb_native` run: 16.7 -> 14.9 ms, 1.12x) | 1.22x / 1.50x (noisy, see below) | 1.36x / 1.81x |

* The 4 sm86 engines total 32.9 MB (the 4 ampere+ ones: 71.6 MB). The portable zip still ships only `ampere+`
  (`packaging/make_portable.ps1 -Arches ampere+,sm86` adds the sm86 files).
* The 15% criterion (vs a natively built engine) is met by `sm86` for `span_x2`, `animevideov3_x4` and `rife_v4`; `rife_v4_lite` is
  borderline/noisy: a `SAME_COMPUTE_CAPABILITY` engine built in-process and run with the full runtime was 1.20x - 1.75x the native one
  as well, so that gap is tactic-selection noise on a shared GPU (the native lite engine ran anywhere between 12 and 15 ms), not the lean
  runtime. `animevideov3_x4` is not hurt by `AMPERE_PLUS` at all (all its kernels are the same either way).
* `fps_1080p` in the manifest was re-measured for all eight engines in one session (GPU busy), so the `ampere+` values are lower than in the first
  table: span_x2 3.8 (sm86 4.5), animevideov3_x4 3.7 (3.7), rife_v4 10.7 (13.1), rife_v4_lite 12.7 (14.4).
* TRT lean runtime check: python `tensorrt_lean` (`bench_engines.py`) deserializes all four `sm86` engines with
  `engine_host_code_allowed` False and True; PSNR vs ORT FP32 equals the `ampere+` engines (64.7 / 72.3 / 68.7 / 71.6 dB min). `fb_native`
  (`fb_engine_load` / `fb_engine_run` through `nvinfer_lean_11.dll`, one opt-shape tile, driven by a ctypes script) loads and runs all four
  `sm86` and `ampere+` engines with finite output.

## sm75 (Turing)

(`sm75` is a regular arch tag: for cc 7.5 the app uses the `sm75` engine only, never `ampere+`.)
`build_engines.py --arch sm75` is implemented (hardware level `SAME_COMPUTE_CAPABILITY`, entry `arch: "sm75"`, file
`<id>.sm75.engine`) but **not run here**: engines for cc 7.5 can only be built on a Turing GPU, and this machine only has an
RTX 3050 (cc 8.6). The script prints a SKIP notice and exits 0 on any other GPU. To produce them, run
`python models/build_engines.py --arch sm75` on an RTX 20xx/GTX 16xx machine (the manifest entries are merged
into the existing `models/engines/manifest.json`), then `bench_engines.py --arch sm75` there. The same applies to any other cc
(`--arch sm89` on an RTX 40xx, `sm120` on an RTX 50xx ...): build on that card, merge the manifest entries, keep `ampere+`.

## Misc

* `build_engines.py <ids> --arch sm86` must run on an sm_86 card; it writes `<id>.sm86.engine` and merges the entry (sha256, size, profile,
  `build_gpu`) into the manifest after the `ampere+` entry. `--no-manifest` writes only the engine file.
* Everything under `models/weights`, `models/onnx` and `*.engine` is gitignored; `models/manifest.json` (copy of
  `models/engines/manifest.json`) is tracked. `base_url` is the placeholder `https://github.com/OWNER/filming-better/releases/download/engines-v1/`.
* Rebuilds are not bit-exact (tactic timing depends on GPU load); `sha256` in the manifest always matches the file next to it.
* Build times on the shared 3050: span 2 min, animevideov3 0.5-1.5 min, rife 3.3 min each, x4plus 3 min.
* The end-to-end `build_all.py span_x2` run (download, export, validate, build, bench) was exercised once and passes; the other
  models were run step by step with the same scripts.
