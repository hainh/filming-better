# Kiểm thử filming-better (Giai đoạn 5)

Ma trận test lấy từ `docs/blueprint.md` (Giai đoạn 5). Cột **Tự động** cho biết script nào phủ hàng đó;
`[ ]` = checklist cần tích tay trước mỗi release (copy bảng này vào issue/PR của release).

```
tests/
├── README.md                  # file này
├── samples/                   # video mẫu SINH RA (git-ignored) - make_samples.sh
├── out/                       # output của lần chạy (git-ignored)
├── bench/baseline.json        # baseline tốc độ theo GPU (commit khi chủ động cập nhật)
└── scripts/
    ├── make_samples.sh        # sinh ~30 clip 2-3 s bằng ffmpeg cục bộ
    ├── run_matrix.sh          # chạy exe trên mọi sample + kiểm exit code + ffprobe output + hành vi CLI
    ├── check_output.py        # (do run_matrix gọi) so thuộc tính ffprobe input/output
    ├── quality.sh             # PSNR / SSIM / VMAF / flicker so với bản tham chiếu
    ├── bench.sh               # benchmark cố định, so với baseline, phát hiện tụt tốc độ
    └── mock_fb.sh             # exe GIẢ (ffmpeg thuần) chỉ để thử chính bộ script, không cần GPU
```

Yêu cầu: Git Bash (hoặc WSL/MSYS), ffmpeg + ffprobe có `libvmaf` (mặc định tìm trong `PATH`, rồi
`C:/ffmpeg-8.0.1-full_build-shared/bin`; ghi đè bằng `FFMPEG` / `FFPROBE`), Python 3 (tự dùng `.venv` nếu có).
FFmpeg GPL cục bộ **chỉ** dùng để sinh dữ liệu test và đo chất lượng, không bao giờ đóng gói.

## Chạy nhanh

```bash
bash tests/scripts/make_samples.sh                 # ~8 MB, vài chục giây;  --long thêm file 2h20m
cargo build --release --manifest-path app/Cargo.toml
bash tests/scripts/run_matrix.sh                   # toàn bộ ma trận + CLI;  --filter 'hevc|vp9'  --ext mp4
bash tests/scripts/run_matrix.sh --exe tests/scripts/mock_fb.sh   # thử harness không cần GPU
bash tests/scripts/bench.sh --update-baseline      # lần đầu trên GPU mới; sau đó chỉ `bash tests/scripts/bench.sh`
```

Tham số `run` mặc định của ma trận: `--upscale span_x2 --interp 2 --codec hevc --cq 20`
(đổi bằng `--args "..."`; script tự suy ra hệ số scale/interp để tính kích thước và fps mong đợi).

## Ma trận test

### Định dạng — `make_samples.sh` + `run_matrix.sh`

| | Mẫu (trong `tests/samples/`) | Tự động |
|---|---|---|
| [ ] H.264 8-bit (MP4 / MOV / TS) | `h264_8bit.mp4`, `.mov`, `.ts` | có |
| [ ] H.264 10-bit (**NVDEC không giải mã được** → phải fallback CPU) | `h264_10bit.mkv` | có |
| [ ] HEVC 8-bit (MKV / TS) và 10-bit (MP4 / MOV) | `hevc_8bit.mkv`, `hevc_8bit.ts`, `hevc_10bit.mp4`, `hevc_10bit.mov` | có |
| [ ] AV1 8/10-bit (cần Ampere+ để NVDEC; RTX 20 → phải báo/ fallback rõ ràng) | `av1_8bit.mp4`, `av1_10bit.mkv` | có |
| [ ] VP9 8/10-bit (WebM / MKV) | `vp9_8bit.webm`, `vp9_10bit.mkv` | có |
| [ ] Đầu ra: H.264 / HEVC / AV1 qua NVENC × MP4 / MKV / MOV | chạy `run_matrix.sh --args "... --codec h264" --ext mp4` v.v. | có (lặp tay từng tổ hợp) |

### Dữ liệu khó

| | Mẫu | Tự động |
|---|---|---|
| [ ] VFR → xuất CFR, không lệch tiếng/hình quá 1 frame | `h264_vfr.mp4` (gap ngẫu nhiên, ~17,7 fps trung bình) | có (output phải CFR + duration) |
| [ ] Các fps chuẩn 23,976 / 24 / 29,97 / 30 / 60 | `h264_23_976fps.mp4`, `h264_29_97fps.mp4`, `h264_60fps.mp4`, ... | có |
| [ ] 1366×768 (không chia hết 16) | `h264_1366x768.mp4`, `hevc_1366x768.mkv` | có |
| [ ] Kích thước lẻ 853×479 (chỉ VP9/AV1 mới có chiều lẻ 4:2:0) | `vp9_odd_853x479.webm`, `av1_odd_853x479.mkv` | có |
| [ ] Metadata xoay hình (display matrix 90°) | `h264_rot90.mp4` (640×360 mã hoá, hiển thị 360×640) | có (kích thước hiển thị ×scale; giữ tag hoặc đã xoay thật) |
| [ ] Nhiều track audio (copy nguyên) | `h264_2audio.mkv` | có (số lượng + codec + sample rate) |
| [ ] Phụ đề (SRT ×2 trong MKV, mov_text trong MP4) | `h264_subs.mkv`, `h264_subs.mp4` | có (số stream). Lưu ý: xuất `.mkv` từ `mov_text` cần chuyển sang srt/ass |
| [ ] Color tag BT.709 / BT.601 full-range giữ nguyên | `h264_bt709.mp4`, `h264_bt601_pc.mp4` | có |
| [ ] Chuyển cảnh cứng (nhân bản frame, không nội suy) | `h264_scenecut.mp4` (cắt ở 1,5 s) | **tay**: xem `--at 00:00:01.4` trước/sau cắt, không có frame nhòe |
| [ ] Không có audio | `h264_noaudio.mp4` | có |
| [ ] File dài > 2 giờ, không rò VRAM, lệch A/V ≤ 1 frame | `make_samples.sh --long` → `long_2h20.mkv` | tay: chạy kèm `nvidia-smi -l 60`, VRAM phải phẳng |
| [ ] Chạy 1 giờ liên tục dưới `compute-sanitizer` (rò bộ nhớ native) | — | tay (Giai đoạn 2) |

### GPU (tay, mỗi GPU một lượt)

| | Cách kiểm | Kết quả mong đợi |
|---|---|---|
| [ ] RTX 20 (sm75) | `gpu-info`; chạy ma trận | chọn engine `sm75`; AV1 input → lỗi/fallback rõ ràng |
| [ ] RTX 30 (sm86) — máy dev RTX 3050 6 GB | ma trận + bench | pass |
| [ ] RTX 40 (sm89) | ma trận + bench | pass |
| [ ] RTX 50 (sm120) | ma trận + bench | pass (cần CUDA ≥ 12.8) |
| [ ] Laptop 4 GB VRAM | `run` 1080p→4K | tự giảm tile / hàng đợi, không `exit 5` |
| [ ] Driver cũ (< R570) | chạy trên máy/VM driver cũ | `exit 4` + thông báo ghi rõ phiên bản đang có và cần có |
| [ ] Không có GPU NVIDIA | máy không GPU | `exit 3` |

### CLI — `run_matrix.sh` (phần "CLI behaviour")

| | Tự động |
|---|---|
| [ ] `--help`, `run --help`, `--version`, `gpu-info` exit 0 | có |
| [ ] Lệnh/tham số sai → exit 2; file không tồn tại → ≠ 0; model lạ → 2 hoặc 7 | có |
| [ ] Đường dẫn có dấu cách + Unicode (input **và** output) | có (`tests/samples/path with spaces`, `unicode`) |
| [ ] Wildcard `*.mp4`, thư mục, `--output-dir`, `--skip-existing` | có |
| [ ] Một file hỏng không làm dừng cả lô → exit 8, bảng tổng kết cuối lô | có |
| [ ] `--progress json` mỗi dòng là JSON hợp lệ `{frame,total,fps,eta}` | có |
| [ ] Ctrl+C giữa chừng → exit 130, file output vẫn mở được (đóng container hợp lệ) | **tay** (Git Bash không gửi Ctrl+C được cho exe Windows): chạy `run` trên `h264_1920x1080.mp4`, nhấn Ctrl+C sau ~1 s, `ffprobe` output |
| [ ] `models list/pull/verify`, sha256 sai bị từ chối | tay (cần mạng / release) |
| [ ] `report` tạo zip log + system info | tay |
| [ ] `preview --at ... --out preview.png` ra PNG trước/sau | tay |

### Chất lượng — `quality.sh`

```bash
bash tests/scripts/quality.sh --make-lowres ref_1080p.mp4 --out in_540p.mp4          # input = tham chiếu thu nhỏ ×2
filming-better.exe run --input in_540p.mp4 --output out.mkv --upscale span_x2 --codec hevc --cq 18
bash tests/scripts/quality.sh out.mkv ref_1080p.mp4 --flicker --min-psnr 28 --min-vmaf 60 --json q.json
```

| | Ngưỡng gợi ý |
|---|---|
| [ ] PSNR / SSIM / VMAF so với bản tham chiếu (clip có chi tiết thật, không phải testsrc) | đặt ngưỡng sau khi có số liệu của model mặc định; dùng `--min-psnr/--min-vmaf` để chặn tụt chất lượng |
| [ ] Nhấp nháy giữa các tile (`--flicker`: tỉ lệ sai khác giữa frame liên tiếp của output / tham chiếu) | ≤ 1,25 (mặc định), thử với `--tile` nhỏ và ảnh có vùng phẳng + gradient |
| [ ] Nội suy: không bóng ma/nhòe ở chuyển cảnh | tay, xem clip `h264_scenecut.mp4` |

### Hiệu năng — `bench.sh`

Chạy mỗi lần release trên cùng máy, cùng driver. `bench.sh` tạo 2 input cố định (1080p và 540p, 10 s, 24 fps, nhiễu có seed),
chạy từng cấu hình `--runs 3` lần (+1 warm-up), lấy **trung vị**, ghi `tests/out/bench/bench-*.json` và so với
`tests/bench/baseline.json` theo tên GPU. Tụt quá `--tolerance 0.10` → exit 1 (`REGRESSION`).
`--full` thêm cấu hình mục tiêu Giai đoạn 2: 1080p → 4K, upscale ×2 + interp ×2.

| [ ] | Cấu hình | Ghi chú |
|---|---|---|
| [ ] | `interp2_1080p` | RIFE 1080p |
| [ ] | `up2_540p_to_1080p` | SPAN ×2 |
| [ ] | `up2_interp2_540p` | cả hai |
| [ ] | `up2_interp2_1080p_to_4k` (`--full`) | mục tiêu "đặt con số sau Giai đoạn 0" |

## Ghi chú về các sample

* Mọi clip dài 2–3 giây, 640×360 trừ khi tên file nói khác (tổng ~8 MB + 2 file bench). Nguồn video là `testsrc2`, nguồn audio sine
  440 Hz — đủ để kiểm cấu trúc/đồng bộ, **không** đủ để đánh giá chất lượng cảm nhận (dùng clip thật cho `quality.sh`).
* H.264/HEVC 4:2:0 không thể có chiều lẻ (độ chi tiết crop = 2 pixel) nên chiều lẻ thử bằng VP9/AV1.
* `make_samples.sh` idempotent (bỏ qua file đã có; `--force` để sinh lại) và tự xác minh bằng ffprobe
  (codec, pix_fmt, kích thước, fps, số stream, rotation) rồi in bảng.
* `tests/samples/` và `tests/out/` nằm trong `.gitignore`.
