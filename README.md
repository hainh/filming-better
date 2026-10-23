# filming-better

**Upscale (x2/x4) và tăng FPS (nội suy x2/x3/x4) video hoàn toàn trên GPU NVIDIA** — ứng dụng dòng lệnh cho Windows.
Giải mã bằng NVDEC, suy luận bằng TensorRT (lean runtime), mã hoá bằng NVENC; frame không rời VRAM.
Audio, phụ đề, chapter được giữ nguyên (copy stream).

*English: a Windows command-line tool that upscales (x2/x4) and interpolates frames (x2/x3/x4) of videos entirely on an
NVIDIA GPU: NVDEC decode -> TensorRT inference (SPAN / Real-ESRGAN / RIFE) -> NVENC encode (H.264 / HEVC / AV1).
Audio, subtitles and chapters are stream-copied.*

> Trạng thái: đang phát triển, chưa có bản phát hành. Kế hoạch đầy đủ: [`docs/blueprint.md`](docs/blueprint.md);
> giao ước giữa các thành phần: [`docs/contracts.md`](docs/contracts.md).

## Yêu cầu

| | |
|---|---|
| Hệ điều hành | Windows 10 / 11 x64 |
| GPU | NVIDIA **RTX 20 series trở lên** (Turing sm75, Ampere sm86, Ada sm89, Blackwell sm120). Bản v1 chưa hỗ trợ AMD / Intel |
| Driver NVIDIA | **R570 trở lên** (CUDA 12.8+; trên Windows từ 570.65). Đây là `FB_MIN_DRIVER_VERSION 12080` trong `app/native/include/fb_native.h` — app được build với CUDA 12.9 runtime và cần driver báo CUDA ≥ 12.8 để có mã máy cho RTX 50 (sm_120). Cũng là mức tối thiểu của NVENC SDK 13 (`nv-codec-headers n13.0.19.1`). Driver cũ hơn: app thoát với mã 4 và báo phiên bản cần cập nhật |
| Phần mềm khác | **Không cần** cài CUDA Toolkit / TensorRT / FFmpeg: runtime đi kèm gói (cudart liên kết tĩnh, TensorRT lean, FFmpeg LGPL) |
| Đầu vào | H.264 / HEVC / AV1 / VP9, 8 hoặc 10-bit, SDR, progressive. AV1 cần GPU Ampere+ để giải mã bằng NVDEC; H.264 10-bit giải mã bằng CPU |
| Đầu ra | H.264 / HEVC / AV1 (NVENC) trong MP4 / MKV / MOV |

## Cài đặt (khi có bản phát hành)

```powershell
# 1) Portable: giải nén filming-better-<ver>-win64.zip (kèm engine mặc định span_x2 + rife_v4) và chạy
.\filming-better.exe gpu-info

# 2) Installer (không kèm engine, tự thêm vào PATH của người dùng)
filming-better-<ver>-setup.exe
filming-better models pull span_x2 rife_v4

# 3) Package manager
winget install OWNER.FilmingBetter
scoop install filming-better        # scoop bucket / manifest: packaging/scoop/filming-better.json
```

## Bắt đầu nhanh

```powershell
filming-better.exe gpu-info                                   # GPU, VRAM, driver, engine phù hợp

# upscale x2 + nội suy x2, xuất HEVC
filming-better.exe run --input in.mp4 --output out.mkv --upscale span_x2 --interp 2 --codec hevc --cq 20

# preset + xử lý hàng loạt (wildcard / thư mục), bỏ qua file đã xong
filming-better.exe run --input "D:\clips\*.mp4" --output-dir D:\out --preset balanced --skip-existing

# xem trước 1 frame (PNG ghép trước/sau)
filming-better.exe preview --input in.mp4 --at 00:01:23 --upscale span_x2 --out preview.png

# quản lý model (engine tải từ GitHub Releases, kiểm sha256, lưu ở %LOCALAPPDATA%)
filming-better.exe models list
filming-better.exe models pull span_x2
filming-better.exe models verify

# tiến độ dạng JSON lines {frame,total,fps,eta} cho script; báo lỗi
filming-better.exe run ... --progress json
filming-better.exe report                                     # gom log + thông tin hệ thống thành 1 file zip
```

Mã thoát: `0` ok · `2` sai tham số · `3` không có GPU NVIDIA · `4` driver quá cũ · `5` hết VRAM ·
`6` codec/định dạng không hỗ trợ · `7` thiếu/hỏng engine · `8` một số file trong lô bị lỗi · `130` Ctrl+C.
Chi tiết: [`docs/contracts.md`](docs/contracts.md) §4. Ctrl+C huỷ an toàn và vẫn đóng file output hợp lệ.

## Môi trường phát triển (Windows)

Mọi thứ nằm **trong thư mục dự án**, không cài vào hệ thống:

| Thư mục | Nội dung |
|---|---|
| `.venv/` | Python 3.12 (tạo bằng `uv`): TensorRT 11.4 cu12 (builder + lean), torch CPU, onnx, spandrel... chỉ dùng trên máy dev/CI cho pipeline `models/` |
| `.tools/` | CUDA 12.9 (nvcc, cudart, cccl từ NVIDIA redist), TensorRT 11.4 headers |
| `C:\ffmpeg-8.0.1-full_build-shared` | FFmpeg **GPL** chỉ để phát triển và sinh dữ liệu test — không bao giờ đóng gói |

Cần sẵn: VS 2022 Build Tools (MSVC 14.4x), Rust stable x86_64-pc-windows-msvc, LLVM/libclang (bindgen).

```bat
:: thiết lập môi trường (vcvars64, CUDA_PATH, PATH, FFMPEG_DIR, FB_TRT_LEAN_DLL...) rồi chạy 1 lệnh
tools\devenv.cmd cargo build --release --manifest-path app\Cargo.toml
:: -> app\target\release\ : filming-better.exe, fb_native.dll, nvinfer_lean_11.dll, DLL FFmpeg

:: từ Git Bash
cmd //c "tools\\devenv.cmd cargo test --manifest-path app/Cargo.toml"
```

Pipeline model (chạy trên máy có GPU, xem `models/`):

```bat
.venv\Scripts\python.exe models\download_weights.py
.venv\Scripts\python.exe models\export_onnx.py
.venv\Scripts\python.exe models\validate_onnx.py          :: PSNR > 40 dB so với PyTorch
.venv\Scripts\python.exe models\build_engines.py          :: -> models\engines\*.engine + manifest.json
.venv\Scripts\python.exe models\bench_engines.py          :: FPS + kiểm lean runtime
```

FFmpeg LGPL tối giản (bản phát hành dùng bản này, không phải bản GPL ở trên) — cần MSYS2 + VS 2022:

```bat
third_party\ffmpeg\build_ffmpeg_lgpl.cmd --dry-run        :: in lệnh configure
third_party\ffmpeg\build_ffmpeg_lgpl.cmd                  :: build; FAIL nếu có --enable-gpl / x264 / x265; ghi BUILD_CONFIG.txt
set FFMPEG_DIR=%CD%\third_party\ffmpeg\dist               :: rồi build lại app để dùng bản LGPL
```

Đóng gói (sau khi đã có exe, engine và FFmpeg LGPL):

```powershell
powershell -ExecutionPolicy Bypass -File packaging\make_portable.ps1          # dist\filming-better-<ver>-win64.zip, fail nếu > 200 MB
iscc /DAppVersion=<ver> packaging\installer.iss                                # dist\filming-better-<ver>-setup.exe (limit 70 MB)
```

Kiểm thử: xem [`tests/README.md`](tests/README.md) (ma trận test, `make_samples.sh`, `run_matrix.sh`, `quality.sh`, `bench.sh`).
CI: `.github/workflows/` — `ci.yml` (fmt, clippy, test), `engines.yml` (self-hosted GPU, upload engine lên release `engines-v1`),
`release.yml` (tag `v*` → zip + installer + kiểm dung lượng), `ffmpeg-lgpl.yml` (build FFmpeg LGPL, có cache).

## Cấu trúc repo

```
filming-better/
├── app/                  # Rust CLI (filming-better.exe) + native/ C++20/CUDA (fb_native.dll)
│   ├── src/              #   cli/ io/ pipeline/ models/ native.rs
│   └── native/           #   include/fb_native.h = ranh giới C duy nhất Rust <-> C++
├── models/               # Python (chỉ máy dev/CI): export ONNX -> engine TensorRT + manifest.json
├── packaging/            # make_portable.ps1, installer.iss, winget/, scoop/, about.toml/hbs
├── third_party/ffmpeg/   # script build FFmpeg LGPL tối giản (MSVC) + kiểm tra không dính GPL
├── tests/                # script sinh sample, ma trận test, chất lượng (PSNR/VMAF), benchmark
├── tools/devenv.cmd      # môi trường dev (MSVC, CUDA, PATH)
├── docs/                 # blueprint.md (kế hoạch), contracts.md (giao ước giữa các phần)
├── .github/              # workflows CI / engines / release + action setup-native
├── LICENSE               # MIT (mã nguồn dự án)
└── THIRD_PARTY_NOTICES   # FFmpeg LGPL, TensorRT lean, CUDA runtime, model weights, crates Rust
```

## Giấy phép

Mã nguồn: [MIT](LICENSE). Gói phát hành còn chứa FFmpeg (LGPL-2.1+, liên kết động, cấu hình build trong
`BUILD_CONFIG.txt`), TensorRT lean runtime (NVIDIA SLA), CUDA runtime (NVIDIA EULA) và trọng số model với giấy phép riêng
(SPAN weights CC-BY-4.0, Real-ESRGAN BSD-3-Clause, RIFE MIT) — xem [`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES).
