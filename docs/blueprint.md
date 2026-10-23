# Kế hoạch triển khai app Upscale + Tăng FPS (CLI)

## Phạm vi bản v1.0

Bản đầu tiên nên chốt phạm vi hẹp để làm xong được:

- **Hình thức**: chỉ là ứng dụng dòng lệnh (CLI), không có GUI. Mọi thao tác đều làm qua tham số dòng lệnh hoặc file cấu hình.
- **Nền tảng**: chỉ Windows 10/11 x64 và GPU NVIDIA (RTX 20 trở lên).
- **Đầu vào**: video SDR, progressive (không interlaced). Codec vào là H.264, HEVC, AV1 hoặc VP9, bản 8-bit và 10-bit.
- **Đầu ra**: H.264, HEVC hoặc AV1 qua NVENC. Audio và phụ đề giữ nguyên bằng cách copy stream.
- **Tính năng**: upscale x2/x4, nội suy x2/x3/x4, xử lý hàng loạt, xem trước bằng cách xuất ảnh PNG so sánh trước/sau.
- **Ngân sách dung lượng**: dưới 200 MB.

Chưa làm ở v1: GUI, HDR, video interlaced, GPU AMD/Intel, Linux, cho người dùng nạp model tùy ý.

## Cấu trúc repo

```
filming-better/
├── models/              # Python, CHỈ chạy trên máy dev/CI
│   ├── export_onnx.py
│   ├── validate_onnx.py
│   ├── build_engines.py
│   └── manifest.json
├── app/                 # Rust → filming-better.exe
│   ├── Cargo.toml
│   ├── build.rs         # gọi CMake build native/, sinh binding từ fb_native.h
│   ├── src/
│   │   ├── main.rs
│   │   ├── cli/         # parse lệnh, preset, config, in tiến độ, mã lỗi
│   │   ├── io/          # demux, decode, encode, mux (FFmpeg qua rsmpeg)
│   │   ├── pipeline/    # các luồng, hàng đợi, timestamp, chuyển cảnh
│   │   ├── models/      # manifest, tải, kiểm tra, chọn engine
│   │   └── native.rs    # lớp bọc an toàn quanh API của fb_native
│   └── native/          # C++20 + CUDA → fb_native.dll
│       ├── CMakeLists.txt
│       ├── include/fb_native.h   # API kiểu C, ranh giới duy nhất với Rust
│       ├── trt/         # bọc TensorRT lean runtime
│       └── cuda/        # kernel đổi màu, pad, tiling, phát hiện chuyển cảnh
├── packaging/           # script tạo bản zip portable và installer
├── third_party/ffmpeg/  # script build FFmpeg LGPL tối giản
├── tests/               # video mẫu, golden frames
└── .github/workflows/
```

## Kiến trúc: Rust bên ngoài, C++/CUDA bên trong

- **Rust (`app/src/`)** lo phần lớn code: CLI, preset, xử lý hàng loạt, đọc/ghi video qua FFmpeg, pipeline đa luồng, quản lý model.
- **C++/CUDA (`app/native/`)** chỉ gồm những thứ bắt buộc phải là C++: bọc TensorRT (chỉ có API C++) và các kernel CUDA. Đóng gói thành `fb_native.dll`, xuất ra một API kiểu C hẹp.
- Rust không gọi CUDA trực tiếp. Mọi thao tác CUDA (stream, cấp phát VRAM, đọc thông tin GPU) đều đi qua `fb_native`, để chỉ có một nơi quản lý tài nguyên GPU.
- Hiệu năng ngang bản thuần C++ vì việc nặng đều chạy trên GPU. Frame không bao giờ rời VRAM khi đi qua ranh giới Rust ↔ C++, chỉ có con trỏ thiết bị được truyền qua.

## Giai đoạn 0: Chuẩn bị và kiểm chứng kỹ thuật (2 tuần)

Mục tiêu là trả lời "go/no-go" cho kiến trúc trước khi viết code thật.

1. **Công cụ**:
   - VS 2022 Build Tools, CMake, Ninja.
   - Rust stable (toolchain `x86_64-pc-windows-msvc`), `bindgen` (cần LLVM/libclang).
   - CUDA Toolkit và TensorRT 11.x SDK. Phiên bản CUDA (cu12 hoặc cu13) phải khớp với bản TensorRT đang dùng.
   - Python 3.11 kèm PyTorch CUDA, chỉ dùng trên máy dev.
   - MSYS2 (để build FFmpeg).
2. **Phần cứng test**: tối thiểu 1 GPU dòng Ampere trở lên (RTX 30/40/50). Nên có thêm 1 card Turing (RTX 20) nếu muốn hỗ trợ dòng này.
3. **Thử nghiệm bằng `trtexec`**:
   - Export ONNX cho RIFE 4.x và một model SPAN x2.
   - Build engine với `--versionCompatible --excludeLeanRuntime --hardwareCompatibilityLevel=ampere+`.
   - Chạy engine bằng **lean runtime** trong một chương trình C++ khoảng 50 dòng.
   - Đo FPS ở 1080p.
4. **Thử ghép Rust ↔ C++/CUDA**:
   - Dựng khung `app/` với `build.rs` gọi CMake build `fb_native.dll` rồi sinh binding bằng `bindgen`. Chỉ cần một lệnh `cargo build` là ra cả exe lẫn DLL.
   - Rust dùng `rsmpeg` decode 1 frame bằng NVDEC, truyền con trỏ thiết bị NV12 sang `fb_native` để đổi màu, chạy engine SPAN, đổi màu ngược lại rồi encode bằng NVENC.
   - Xác nhận FFmpeg và `fb_native` dùng chung một CUDA context (primary context, cờ `AV_CUDA_USE_PRIMARY_CONTEXT`) và không có copy GPU → CPU nào.
5. **Chốt cu12 hay cu13**:
   - App tự mang theo CUDA runtime (`cudart64_*.dll`) và TensorRT lean runtime, người dùng không cần cài CUDA Toolkit. Thứ duy nhất app phụ thuộc trên máy người dùng là **driver NVIDIA**.
   - Driver tương thích ngược: app cu12 chạy được trên driver hỗ trợ CUDA 13, nhưng app cu13 không chạy trên driver chỉ hỗ trợ CUDA 12. Nếu người dùng cài sẵn CUDA Toolkit khác phiên bản cũng không sao, vì Windows ưu tiên nạp DLL nằm cạnh exe.
   - **Mặc định nghiêng về cu12 (12.8 trở lên)** để chạy được trên nhiều máy hơn. Phải từ 12.8 mới có mã máy sẵn cho RTX 50 (sm_120).
   - Kiểm tra TensorRT 11 còn bản cho cu12 không. Nếu không còn, buộc phải dùng cu13 và chấp nhận yêu cầu driver mới hơn. Engine và lean runtime phải cùng một bản (cùng cu12 hoặc cùng cu13), không trộn.
   - Ghi lại **phiên bản driver tối thiểu** của app. Con số này là mức cao hơn trong hai yêu cầu: driver tối thiểu của bản CUDA đã chọn, và driver tối thiểu cho NVDEC/NVENC (tùy bản `nv-codec-headers` dùng khi build FFmpeg).
   - Đồng thời kiểm tra lại các cờ build trong tài liệu TRT 11.

**Điều kiện qua giai đoạn**: lean runtime nạp được engine đã build sẵn, tốc độ của engine `ampere+` không chậm hơn engine build riêng quá 15%, và bài thử Rust ↔ C++ chạy trọn một frame trong VRAM.

## Giai đoạn 1: Pipeline model phía dev (1–2 tuần)

1. **Chọn model** và kiểm tra license từng file trọng số:

   | Vai trò | Model | Ghi chú |
   |---|---|---|
   | Upscale nhanh | `realesr-animevideov3` x2/x4, SPAN x2/x4 | Dùng làm mặc định |
   | Upscale chất lượng | Real-ESRGAN `x4plus` | Chậm hơn |
   | Nội suy | RIFE 4.x (bản mới nhất, kèm một bản "lite") | MIT |

2. **`export_onnx.py`**:
   - Upscale: load model qua spandrel và export ONNX FP32 với opset 17 trở lên. Đặt dynamic axes cho chiều H/W, hoặc dùng shape tile cố định.
   - RIFE: viết module wrapper nhận `(img0, img1, timestep)` và trả về frame giữa. Cố định các tham số `scale` và `ensemble`, yêu cầu input được pad về bội số của 64.
   - Chạy `onnxsim` để rút gọn graph.
3. **`validate_onnx.py`**: so sánh output của ONNX Runtime với PyTorch trên 20 frame mẫu. Yêu cầu PSNR trên 40 dB.
4. **`build_engines.py`**:
   - Build theo ma trận model × {`ampere+`, `sm75`} × profile shape.
   - Xuất `manifest.json` gồm: id, kiến trúc, phiên bản TRT, shape, sha256 và kích thước file.
   - Lưu ý: engine `sm75` phải build trên chính một GPU Turing.
5. **CI**: dùng GitHub Actions trên self-hosted runner có GPU. Engine được build lại mỗi khi đổi model hoặc nâng TensorRT, rồi upload lên GitHub Releases.

**Kết quả cần có**: các file engine cùng manifest. Mỗi engine kèm số đo FPS và kích thước.

## Giai đoạn 2: App Rust + native C++/CUDA (3–5 tuần, phần lớn nhất)

### 2.1 Video I/O (Rust, `src/io/`)
- Viết script build FFmpeg LGPL tối giản (đã có trong câu trả lời trước), link động. Rust dùng crate `rsmpeg` trỏ vào bản FFmpeg này.
- **Demux**: dùng libavformat. Đọc metadata màu (matrix, range, primaries), xoay hình, timebase.
- **Decode**: dùng libavcodec với `hw_device_ctx` CUDA, nhờ đó frame NV12/P010 được giữ ngay trong VRAM. Nếu NVDEC không hỗ trợ codec của file, chuyển sang decode bằng CPU rồi upload lên GPU.
- **Encode**: dùng `hevc_nvenc`/`av1_nvenc`/`h264_nvenc` với `hw_frames_ctx` để không phải copy frame.
- **Mux**: copy nguyên audio, phụ đề, chapter và metadata. Gắn đúng color tag cho output.
- Phần gọi FFmpeg phần lớn là `unsafe`. Gom vào vài struct có `Drop` để giải phóng đúng, phần còn lại của app chỉ dùng API an toàn.

### 2.2 API ranh giới `fb_native.h`
- Chỉ dùng kiểu C: handle mờ (`fb_engine*`, `fb_stream*`), con trỏ thiết bị (`CUdeviceptr`), pitch, kích thước, và mã lỗi `int`. Không truyền exception hay kiểu C++ qua ranh giới. Mọi hàm bọc trong `try/catch` và trả về mã lỗi, kèm `fb_last_error()` để lấy thông báo.
- Nhóm hàm dự kiến:
  - Thiết bị: `fb_init`, `fb_gpu_info`, `fb_mem_info`. `fb_init` gọi `cuDriverGetVersion` và trả về mã lỗi riêng nếu driver thấp hơn mức tối thiểu đã chốt ở giai đoạn 0. Phía Rust in ra phiên bản đang có và phiên bản cần cập nhật, thay vì để app crash giữa chừng.
  - Stream và bộ nhớ: `fb_stream_create/destroy`, `fb_alloc/free`, `fb_stream_sync`.
  - Engine: `fb_engine_load` (từ đường dẫn file), `fb_engine_io_shape`, `fb_engine_run`, `fb_engine_free`.
  - Kernel: `fb_nv12_to_rgb`, `fb_rgb_to_nv12` (và bản P010), `fb_pad`, `fb_tile_split/merge`, `fb_scene_score`.
- **Quy ước sở hữu**: bên nào cấp phát thì bên đó giải phóng. Frame decode/encode thuộc FFmpeg (Rust quản lý), buffer trung gian và engine thuộc `fb_native`. Phía Rust bọc mỗi handle trong struct có `Drop` để không rò VRAM.
- Lớp `native.rs` biến mã lỗi thành `Result<T, NativeError>` và đánh dấu kiểu nào được `Send` giữa các luồng.

### 2.3 Kernel CUDA (`native/cuda/`)
- Build fatbin gồm mã máy cho sm_75, sm_86, sm_89, sm_120, cộng PTX của kiến trúc mới nhất, để GPU đời sau vẫn tự JIT được.
- Đổi NV12/P010 sang RGB planar FP16 và ngược lại. Hỗ trợ BT.601/709 và cả limited/full range.
- Pad frame về bội số của 64, rồi cắt bỏ phần pad sau khi xử lý.
- Tách frame thành tile có overlap, sau đó ghép lại với blend feather ở mép để không lộ đường nối.
- Phát hiện chuyển cảnh: so sánh frame thu nhỏ 64×36 bằng MAD hoặc histogram.

### 2.4 Inference (`native/trt/` + `src/models/`)
- Phía Rust có trait `InferBackend` để sau này thêm được backend DirectML. Bản v1 chỉ có `TrtBackend` gọi xuống `fb_engine_*`.
- `native/trt/`: dùng lean runtime để deserialize engine, quản lý execution context, binding và CUDA stream.
- Chọn engine tự động (Rust): lấy compute capability và phiên bản driver qua `fb_gpu_info`, rồi đối chiếu với manifest.
- Tự chọn kích thước tile dựa theo VRAM còn trống (`fb_mem_info`, bên trong gọi `cudaMemGetInfo`).

### 2.5 Pipeline (Rust, `src/pipeline/`)
- Chạy 4 luồng: decode → upscale → interpolate → encode, nối với nhau bằng hàng đợi có giới hạn (`crossbeam-channel` bounded) để tạo backpressure. Mỗi tầng dùng CUDA stream riêng tạo qua `fb_stream_create`.
- **Thứ tự xử lý**: cho cấu hình được. Mặc định nội suy ở độ phân giải gốc trước rồi mới upscale, vì cách này nhanh hơn.
- **Timestamp**: video VFR được chuyển sang CFR trước khi xử lý. FPS đầu ra bằng FPS đầu vào nhân N.
- Khi gặp chuyển cảnh, nhân bản frame thay vì nội suy để tránh nhòe hình.

### 2.6 Lệnh xử lý cơ bản (Rust, `src/cli/`)
- Crate dự kiến: `clap` (lệnh và tham số), `indicatif` (thanh tiến độ), `serde`/`serde_json`, `ctrlc`, `anyhow`/`thiserror`, `tracing` (log).
```
filming-better.exe run --input in.mp4 --output out.mkv --upscale span_x2 --interp 2 --codec hevc --cq 20
filming-better.exe gpu-info
```
- Mặc định in tiến độ dạng thanh trên terminal (frame, fps, ETA). Thêm `--progress json` để in JSON lines `{frame, total, fps, eta}` ra stdout, tiện cho script hoặc công cụ khác đọc.
- Ctrl+C hủy tác vụ an toàn và vẫn đóng file output hợp lệ.
- Lỗi trả về theo exit code riêng: không có GPU NVIDIA, driver quá cũ, hết VRAM, codec không hỗ trợ. Thông báo lỗi in ra stderr, ngắn gọn và nói rõ cách khắc phục.

**Tiêu chí hoàn thành**:
- Video 1080p, upscale x2 cùng nội suy 2x, đạt tốc độ mục tiêu (cần đặt con số sau khi có kết quả giai đoạn 0).
- Lệch tiếng và hình không quá 1 frame.
- Chạy một video 1 giờ mà không rò VRAM.
- `fb_native` có test riêng bằng C++ (so kernel với kết quả tham chiếu trên CPU). Phía Rust có `cargo test` cho pipeline và CLI.

## Giai đoạn 3: Hoàn thiện CLI (1–2 tuần, toàn bộ bằng Rust)

Crate dự kiến: `reqwest` (rustls) và `sha2` để tải và kiểm tra engine, `toml` cho file cấu hình, `glob` cho wildcard, `zip` cho lệnh `report`.

1. **Preset**: `--preset fast|balanced|quality` chọn sẵn model, hệ số và chất lượng encode. Tham số riêng lẻ vẫn ghi đè được preset.
2. **Xem trước**: lấy một frame, xử lý rồi xuất PNG ghép trước/sau cạnh nhau (hoặc 2 file riêng):
   ```
   filming-better.exe preview --input in.mp4 --at 00:01:23 --upscale span_x2 --out preview.png
   ```
3. **Xử lý hàng loạt**: nhận nhiều file, wildcard hoặc cả thư mục, xuất vào `--output-dir` với mẫu tên file cấu hình được. Bỏ qua file đã xử lý xong (`--skip-existing`), một file lỗi không làm dừng cả lô, cuối lô in bảng tổng kết.
   ```
   filming-better.exe run --input "D:\clips\*.mp4" --output-dir D:\out --preset balanced
   ```
4. **Quản lý model**: đọc manifest từ GitHub Releases, tải đúng engine cho GPU của máy, kiểm tra sha256, lưu vào `%LOCALAPPDATA%`.
   ```
   filming-better.exe models list
   filming-better.exe models pull span_x2
   filming-better.exe models verify
   ```
5. **File cấu hình**: đọc tùy chọn mặc định từ file TOML (`--config` hoặc `%APPDATA%`), để không phải gõ lại tham số dài.
6. **Chẩn đoán**: `gpu-info` in GPU, VRAM, driver và engine phù hợp. `--log-file` ghi log chi tiết, `filming-better.exe report` gom log và thông tin hệ thống thành một file zip để gửi khi báo lỗi.
7. **Trợ giúp**: `--help` cho từng lệnh con, kèm ví dụ dùng thực tế.

## Giai đoạn 4: Đóng gói và phát hành (1 tuần)

1. **Bản portable**: file zip gồm `filming-better.exe`, `fb_native.dll`, các DLL FFmpeg, lean runtime và 1 bộ engine mặc định. Giải nén là chạy được.
2. **Installer (tùy chọn)**: NSIS hoặc Inno Setup, nén LZMA, tự thêm thư mục cài vào `PATH`.
3. **Phân phối qua package manager**: viết manifest cho winget và Scoop để người dùng cài/cập nhật bằng một lệnh.
4. **Kiểm soát dung lượng trong CI**: build thất bại nếu gói cài vượt 70 MB.
5. **Cập nhật**: bản chương trình cập nhật qua winget/Scoop. `filming-better.exe` kiểm tra phiên bản mới khi chạy `gpu-info` hoặc `--version` và chỉ in thông báo. Engine được cập nhật riêng qua `models pull`.
6. **Ký code**: dự án mã nguồn mở có thể dùng SignPath miễn phí, nếu không thì mua chứng chỉ. Không ký thì Windows SmartScreen và một số phần mềm diệt virus sẽ cảnh báo người dùng.
7. **License**:
   - Đặt file `THIRD_PARTY_NOTICES` trong gói phát hành.
   - FFmpeg LGPL phải link động và công bố cấu hình build cùng mã nguồn.
   - Chỉ phân phối những file TensorRT mà NVIDIA SLA liệt kê là redistributable.

## Giai đoạn 5: Kiểm thử (song song từ giai đoạn 2, cộng 1–2 tuần cuối)

| Nhóm | Trường hợp cần test |
|---|---|
| Định dạng | H.264/HEVC/AV1/VP9, 8-bit và 10-bit, MP4/MKV/MOV/TS |
| Dữ liệu khó | VFR, độ phân giải lẻ (1366×768, chiều lẻ), metadata xoay hình, nhiều track audio, phụ đề, file dài trên 2 giờ |
| GPU | RTX 20/30/40/50, laptop 4 GB VRAM, driver cũ (phải báo lỗi rõ ràng) |
| CLI | Đường dẫn có dấu cách và Unicode, wildcard, exit code, Ctrl+C giữa chừng, output JSON hợp lệ |
| Chất lượng | VMAF/PSNR so với bản tham chiếu, kiểm tra nhấp nháy giữa các tile |
| Hiệu năng | Bộ benchmark cố định chạy mỗi khi release để phát hiện tụt tốc độ |

## Giai đoạn 6 (sau v1.0): Mở rộng

1. Thêm backend ONNX Runtime + DirectML cho GPU AMD/Intel, tải về khi cần (khoảng 30–50 MB).
2. Cho người dùng import model ONNX riêng. Cần một gói builder TensorRT tải thêm (khoảng 1 GB) để build engine trên máy người dùng.
3. Hỗ trợ HDR, deinterlace, và có thể thêm Linux.

## Mốc thời gian (1 dev, toàn thời gian)

| Mốc | Tuần | Kết quả |
|---|---|---|
| M0 | 2 | Kiểm chứng xong (gồm bài thử Rust ↔ C++), quyết định go/no-go |
| M1 | 4 | Có engine và manifest, CI build tự động |
| M2 | 9 | `filming-better.exe run` chạy trọn từ đầu vào đến đầu ra (bản alpha) |
| M3 | 11 | CLI đủ tính năng: preset, preview, hàng loạt, quản lý model |
| M4 | 12 | Có gói portable/installer, bản beta cho người dùng thử |
| M5 | 13–15 | Phát hành v1.0 |

## Rủi ro chính

| Rủi ro | Cách giảm thiểu |
|---|---|
| Lean runtime hoặc cờ `ampere+` hoạt động khác trên TRT 11 | Đã đưa vào kiểm chứng ở giai đoạn 0, trước khi viết code thật |
| Engine bị lỗi thời khi NVIDIA ra GPU kiến trúc mới | Manifest hỗ trợ engine theo kiến trúc. Khi chưa có engine phù hợp, dùng tạm DirectML |
| Hết VRAM trên laptop | Tự giảm kích thước tile và độ sâu hàng đợi |
| Lệch tiếng-hình với video VFR | Chuyển sang CFR trước khi xử lý, có test riêng cho VFR |
| Dính GPL do FFmpeg | Script build kiểm tra `--disable-gpl` và không bật x264/x265 |
| Lỗi ở ranh giới Rust ↔ C++ (rò VRAM, dùng con trỏ sau khi giải phóng, sai CUDA context) | API C hẹp, quy ước sở hữu rõ ràng, mọi handle bọc trong `Drop`. Kiểm chứng sớm ở giai đoạn 0, chạy test dài với `compute-sanitizer` |
| Build hai bộ công cụ (Cargo + CMake/nvcc) phức tạp trên Windows và CI | `build.rs` tự gọi CMake, một lệnh `cargo build` ra đủ file. Ghim phiên bản CUDA, TensorRT, Rust trong CI |
| CLI khó dùng với người không quen dòng lệnh | Preset hợp lý, `--help` có ví dụ, thông báo lỗi kèm cách khắc phục |

Nếu bạn muốn bắt đầu, tôi đề xuất làm giai đoạn 0 trước: script export ONNX cho RIFE và SPAN, chương trình C++ nhỏ để thử nạp engine bằng lean runtime, và khung `app/` để thử gọi `fb_native` từ Rust. Kết quả bước này sẽ quyết định kiến trúc có đi tiếp được không.
