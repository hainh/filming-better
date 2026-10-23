# fb_native — C++/CUDA core (`fb_native.dll`)

The only boundary to Rust is `include/fb_native.h` (plain C, `int` status codes, thread-local `fb_last_error()`).

## Layout

```
include/fb_native.h   public C API (consumed by bindgen)
src/                  device / stream / memory / error handling, kernel argument validation
cuda/kernels.{h,cu}   NV12/P010 <-> RGB, pad/crop/tile split, tile merge/finalize, scene score
trt/                  TensorRT *lean* runtime wrapper (dynamic LoadLibrary of nvinfer_lean_11.dll)
tests/                fb_tests.exe (CPU-reference tests), make_test_engines.py (tiny VC engines)
```

## Build

Everything is project-local; from the repo root (Git Bash):

```
cmd //c "tools\\devenv.cmd cmake -S app\\native -B .build\\native -G Ninja -DFB_BUILD_TESTS=ON"
cmd //c "tools\\devenv.cmd cmake --build .build\\native"
cmd //c "tools\\devenv.cmd cmake --build .build\\native --target install"   # optional: <prefix>/{bin,lib,include}
cd .build/native && cmd //c "..\\..\\tools\\devenv.cmd ctest --output-on-failure"
```

Needs (read by CMake): `CUDA_PATH` (or `-DCUDAToolkit_ROOT`; CUDA 12.9: nvcc, cudart_static, cudart headers) and
`TENSORRT_INCLUDE` (or `-DTENSORRT_INCLUDE_DIR`; TRT 11.x headers only, no `.lib`). Defaults: Release, C++20, static
cudart, static MSVC CRT (`-DFB_STATIC_CRT=OFF` to use `/MD`), CUDA archs `75-real;86-real;89-real;120` (SASS + PTX for 120).

Outputs: `<build>/bin/fb_native.dll` (+ `.pdb`), `<build>/lib/fb_native.lib` (import lib). `install` copies them to
`<prefix>/bin`, `<prefix>/lib` and the header to `<prefix>/include`. The `cmake` crate flow
(`cmake::Config::new("native").generator("Ninja")`, build + install, `-MD`/`-MDd` injected via `CMAKE_CXX_FLAGS`) was
checked in Debug and Release.

Tests: `bin/fb_tests.exe [name-filter]`; ctest groups: `color_ tensor_ memory_ tiling_ device_ error_ streams_ scene_
engine_ leak_` (the `engine_`/`leak_` groups first run `make_test_engines.py` with `.venv` python). `engine_real_*` also
loads `models/engines/*.engine` when present.

## Runtime dependencies

`dumpbin /dependents fb_native.dll` -> **only `KERNEL32.dll`**. cudart is linked statically (it loads `nvcuda.dll`, the
driver, at runtime), the MSVC CRT is static. Ship next to the exe:

* `fb_native.dll`
* `nvinfer_lean_11.dll` (TensorRT lean runtime; depends only on KERNEL32). Search order: `fb_trt_set_library()`
  (strict, if used) -> env `FB_TRT_LEAN_DLL` -> directory of `fb_native.dll` -> default DLL search path.

No `cudart64_12.dll` is needed. The NVIDIA driver (>= CUDA 12.8, `FB_MIN_DRIVER_VERSION`) is the only system requirement.

## API notes / rules for the Rust side

* Call `fb_init(device, &driver_version)` once first. Statuses: `FB_ERR_NO_GPU`, `FB_ERR_DRIVER_TOO_OLD` (driver version
  is still returned), `FB_ERR_INVALID_ARG`. Every other call (except `fb_api_version`, `fb_last_error`, `fb_device_count`,
  `fb_gpu_info`, `fb_trt_*`) returns `FB_ERR_NOT_INITIALIZED` before `fb_init`. `fb_shutdown` does **not** reset the
  primary context (FFmpeg shares it).
* Threads: any thread may call any function; each call makes the selected device current on the calling thread.
  `fb_last_error()` is thread-local and cleared by every successful call.
* `fb_stream` is created `cudaStreamNonBlocking`, so it does **not** implicitly synchronize with the legacy default
  stream used by FFmpeg/NVDEC. The caller must make sure decoded frames are complete (FFmpeg returns them synchronized)
  and that work on our stream is finished (`fb_stream_sync`) before handing a frame to NVENC. `NULL` stream = legacy
  default stream. A stream may be used by one thread at a time.
* `fb_engine`: no internal locking, no thread affinity (create on one thread, run/free on another is fine) but never use
  one engine from two threads at once. Different engines may run concurrently. One engine = one execution context,
  profile 0. `fb_engine_run` is `enqueueV3` (async on the stream); input shapes must be set first and **all** I/O
  tensor addresses set (`fb_engine_set_tensor`) — addresses may be re-set between runs. Tensor memory from `fb_alloc` is
  256-byte aligned; if you sub-allocate keep >= 16-byte alignment. Engines load with host code disabled (they are
  built with `EXCLUDE_LEAN_RUNTIME`), so `setEngineHostCodeAllowed` is not used.
* `fb_free`/`fb_engine_free`/`fb_stream_destroy` may synchronize the device (cudaFree) — pool buffers instead of
  alloc/free per frame.
* TensorRT logging: errors/warnings go to `fb_last_error()` of the failing call; set env `FB_TRT_VERBOSE=1` (info) or
  `2` (verbose) to also print to stderr.
* Colour conventions: chroma is treated as centre-sited (bilinear upsample; 2x2 average on the way back), so
  YUV->RGB->YUV is neutral. `P010` uses the 10 MSBs of each 16-bit sample. Full-range uses `/255` (`/1023`) scaling.
  `fb_yuv_to_rgb` output is clamped to [0,1]. `fb_tile_finalize` does not clamp.
* Tile merge weights: per edge not flagged as image border, linear ramp `(t+0.5)/feather` over the first/last `feather`
  pixels (in tile/output pixels; pass `feather*scale` for upscaled tiles). `border_flags`: 1 left, 2 top, 4 right, 8 bottom.
* `fb_scene_score` uses BT.601 luma weights on the RGB tensors, 64x36 area cells; it blocks until the result is ready.
* Slightly-too-large `fb_alloc` requests on WDDM can take a long time to fail (tens of seconds) — size buffers from
  `fb_mem_info`.
