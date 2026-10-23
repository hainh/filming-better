#!/usr/bin/env bash
# Generates small synthetic test clips (2-3 s each) covering the Phase-5 "Dinh dang" / "Du lieu kho" matrix:
#   H.264 / HEVC / AV1 / VP9, 8 and 10 bit, MP4 / MKV / MOV / TS / WebM, VFR, 1366x768, odd dimensions,
#   rotation metadata, two audio tracks, subtitles, a hard scene cut, Unicode + space path.
# Output: tests/samples/ (git-ignored). Uses the local full FFmpeg build (GPL is fine for TEST DATA only).
#
# Usage: tests/scripts/make_samples.sh [--out DIR] [--long] [--force]
#   --long   additionally create a 2h20m 320x180 file (long-file test, takes a few minutes)
#   --force  regenerate files that already exist
# Env:   FFMPEG / FFPROBE override the binaries (default: ffmpeg on PATH, else C:/ffmpeg-8.0.1-full_build-shared/bin)
#
# NOTES
#  * H.264 / HEVC in 4:2:0 cannot have odd dimensions (chroma subsampling + coded-size cropping granularity of 2),
#    so the odd-size (853x479) clips are VP9 and AV1 (both allow odd sizes). 1366x768 (not a multiple of 16) is
#    produced for H.264 and HEVC.
#  * H.264 10-bit (High10) is NOT decodable by NVDEC on any GPU -> the app must fall back to CPU decode.
#  * AV1 clips need an Ampere+ GPU for NVDEC (RTX 20 has no AV1 decoder).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$ROOT/tests/samples"
LONG=0
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift ;;
    --long) LONG=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done

DEFAULT_FF_DIR="C:/ffmpeg-8.0.1-full_build-shared/bin"
if [[ -z "${FFMPEG:-}" ]]; then
  if command -v ffmpeg >/dev/null 2>&1; then FFMPEG=ffmpeg; else FFMPEG="$DEFAULT_FF_DIR/ffmpeg.exe"; fi
fi
if [[ -z "${FFPROBE:-}" ]]; then
  if command -v ffprobe >/dev/null 2>&1; then FFPROBE=ffprobe; else FFPROBE="$DEFAULT_FF_DIR/ffprobe.exe"; fi
fi
FF=("$FFMPEG" -hide_banner -loglevel error -y)

mkdir -p "$OUT"
cd "$OUT"

declare -a CHECKS=()   # "file|codec|pix_fmt-regex|WxH"
FAILED=0

# ---------------------------------------------------------------------------------------------
# mk <file> <W> <H> <fps> <dur> <expect-codec> <expect-pixfmt-regex> <video-encoder-args...>
#   video: testsrc2 ; audio: 440 Hz sine (aac, or opus for .webm) unless NOAUDIO=1
# ---------------------------------------------------------------------------------------------
mk() {
  local file="$1" w="$2" h="$3" fps="$4" dur="$5" ecodec="$6" epix="$7"; shift 7
  CHECKS+=("$file|$ecodec|$epix|${w}x${h}")
  if [[ -s "$file" && $FORCE -eq 0 ]]; then echo "  skip (exists) $file"; return; fi
  local ac=(-f lavfi -i "sine=frequency=440:sample_rate=48000:duration=$dur" -c:a aac -b:a 64k)
  [[ "$file" == *.webm ]] && ac=(-f lavfi -i "sine=frequency=440:sample_rate=48000:duration=$dur" -c:a libopus -b:a 48k)
  [[ "${NOAUDIO:-0}" == 1 ]] && ac=(-an)
  echo "  make $file"
  "${FF[@]}" -f lavfi -i "testsrc2=size=${SRCSIZE:-${w}x${h}}:rate=${fps}:duration=${dur}" "${ac[@]}" \
      "$@" -shortest "$file" || { echo "FAILED: $file" >&2; FAILED=1; }
}

X264=(-c:v libx264 -preset veryfast -crf 24)
X265=(-c:v libx265 -preset ultrafast -crf 28 -x265-params log-level=error)
SVT=(-c:v libsvtav1 -preset 10 -crf 40)
VP9=(-c:v libvpx-vp9 -b:v 0 -crf 40 -deadline realtime -cpu-used 8 -row-mt 1)

echo "== codec x bit depth x container"
mk h264_8bit.mp4        640 360 30 3 h264 'yuv420p$'          "${X264[@]}" -pix_fmt yuv420p
mk h264_10bit.mkv       640 360 30 3 h264 'yuv420p10le'       "${X264[@]}" -pix_fmt yuv420p10le -profile:v high10
mk hevc_8bit.mkv        640 360 30 3 hevc 'yuv420p$'          "${X265[@]}" -pix_fmt yuv420p
mk hevc_10bit.mp4       640 360 30 3 hevc 'yuv420p10le'       "${X265[@]}" -pix_fmt yuv420p10le -tag:v hvc1
mk av1_8bit.mp4         640 360 30 3 av1  'yuv420p$'          "${SVT[@]}" -pix_fmt yuv420p
mk av1_10bit.mkv        640 360 30 3 av1  'yuv420p10le'       "${SVT[@]}" -pix_fmt yuv420p10le
mk vp9_8bit.webm        640 360 30 3 vp9  'yuv420p$'          "${VP9[@]}" -pix_fmt yuv420p
mk vp9_10bit.mkv        640 360 30 3 vp9  'yuv420p10le'       "${VP9[@]}" -pix_fmt yuv420p10le -profile:v 2

echo "== containers (H.264 / HEVC in MOV and MPEG-TS)"
mk h264_8bit.mov        640 360 30 2 h264 'yuv420p$'          "${X264[@]}" -pix_fmt yuv420p
mk h264_8bit.ts         640 360 30 2 h264 'yuv420p$'          "${X264[@]}" -pix_fmt yuv420p -f mpegts
mk hevc_8bit.ts         640 360 30 2 hevc 'yuv420p$'          "${X265[@]}" -pix_fmt yuv420p -f mpegts
mk hevc_10bit.mov       640 360 30 2 hevc 'yuv420p10le'       "${X265[@]}" -pix_fmt yuv420p10le -tag:v hvc1

echo "== resolutions"
mk h264_1366x768.mp4    1366 768 30 2 h264 'yuv420p$'         "${X264[@]}" -pix_fmt yuv420p
mk hevc_1366x768.mkv    1366 768 30 2 hevc 'yuv420p$'         "${X265[@]}" -pix_fmt yuv420p
# (lavfi sources round to even sizes -> generate 854x480 and crop to the odd size, exact=1 keeps odd chroma)
SRCSIZE=854x480 mk vp9_odd_853x479.webm 853 479 30 2 vp9  'yuv420p$' "${VP9[@]}" -vf crop=853:479:0:0:exact=1 -pix_fmt yuv420p
SRCSIZE=854x480 mk av1_odd_853x479.mkv  853 479 30 2 av1  'yuv420p$' -c:v libaom-av1 -cpu-used 8 -crf 45 -b:v 0 -vf crop=853:479:0:0:exact=1 -pix_fmt yuv420p
mk h264_1920x1080.mp4   1920 1080 24 2 h264 'yuv420p$'        "${X264[@]}" -pix_fmt yuv420p

echo "== frame rates: 24 / 23.976 / 29.97 / 60 / VFR"
mk h264_23_976fps.mp4   640 360 24000/1001 2 h264 'yuv420p$'  "${X264[@]}" -pix_fmt yuv420p
mk h264_29_97fps.mp4    640 360 30000/1001 2 h264 'yuv420p$'  "${X264[@]}" -pix_fmt yuv420p
mk h264_60fps.mp4       640 360 60 2 h264 'yuv420p$'          "${X264[@]}" -pix_fmt yuv420p
# VFR: drop 3 of every 7 frames but keep the original timestamps (irregular gaps), -fps_mode vfr
mk h264_vfr.mp4         640 360 30 3 h264 'yuv420p$'          "${X264[@]}" -pix_fmt yuv420p \
    -vf "select='lt(mod(n\,7)\,4)'" -fps_mode vfr

echo "== audio / subtitles / metadata"
# two audio tracks (440 Hz eng, 880 Hz jpn), H.264 in MKV
if [[ ! -s h264_2audio.mkv || $FORCE -eq 1 ]]; then
  echo "  make h264_2audio.mkv"
  "${FF[@]}" -f lavfi -i "testsrc2=size=640x360:rate=30:duration=3" \
      -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=3" \
      -f lavfi -i "sine=frequency=880:sample_rate=48000:duration=3" \
      -map 0:v -map 1:a -map 2:a "${X264[@]}" -pix_fmt yuv420p -c:a aac -b:a 64k \
      -metadata:s:a:0 language=eng -metadata:s:a:1 language=jpn -shortest h264_2audio.mkv || FAILED=1
fi
CHECKS+=("h264_2audio.mkv|h264|yuv420p\$|640x360")

# subtitles: SRT (mkv, 2 tracks) and mov_text (mp4)
cat > subs_en.srt <<'SRT'
1
00:00:00,200 --> 00:00:01,000
Hello subtitle one

2
00:00:01,200 --> 00:00:02,000
Second line
with two rows

3
00:00:02,200 --> 00:00:02,900
Third \u2013 unicode test: dấu tiếng Việt
SRT
cat > subs_fr.srt <<'SRT'
1
00:00:00,200 --> 00:00:01,000
Bonjour sous-titre
SRT
if [[ ! -s h264_subs.mkv || $FORCE -eq 1 ]]; then
  echo "  make h264_subs.mkv"
  "${FF[@]}" -f lavfi -i "testsrc2=size=640x360:rate=30:duration=3" \
      -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=3" -i subs_en.srt -i subs_fr.srt \
      -map 0:v -map 1:a -map 2 -map 3 "${X264[@]}" -pix_fmt yuv420p -c:a aac -b:a 64k -c:s srt \
      -metadata:s:s:0 language=eng -metadata:s:s:1 language=fra -t 3 h264_subs.mkv || FAILED=1
fi
CHECKS+=("h264_subs.mkv|h264|yuv420p\$|640x360")
if [[ ! -s h264_subs.mp4 || $FORCE -eq 1 ]]; then
  echo "  make h264_subs.mp4"
  "${FF[@]}" -f lavfi -i "testsrc2=size=640x360:rate=30:duration=3" \
      -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=3" -i subs_en.srt \
      -map 0:v -map 1:a -map 2 "${X264[@]}" -pix_fmt yuv420p -c:a aac -b:a 64k -c:s mov_text \
      -metadata:s:s:0 language=eng -t 3 h264_subs.mp4 || FAILED=1
fi
CHECKS+=("h264_subs.mp4|h264|yuv420p\$|640x360")

# rotation metadata (display matrix) written by stream copy: coded 640x360, displayed 360x640
if [[ ! -s h264_rot90.mp4 || $FORCE -eq 1 ]]; then
  echo "  make h264_rot90.mp4 (display_rotation=90)"
  "${FF[@]}" -display_rotation 90 -i h264_8bit.mp4 -c copy h264_rot90.mp4 || FAILED=1
fi
CHECKS+=("h264_rot90.mp4|h264|yuv420p\$|640x360")

# color tags: BT.709 limited, plus a BT.601/full-range variant
mk h264_bt709.mp4       640 360 30 2 h264 'yuv420p$'          "${X264[@]}" -pix_fmt yuv420p \
    -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv
mk h264_bt601_pc.mp4    640 360 30 2 h264 'yuvj?420p'         "${X264[@]}" -pix_fmt yuvj420p \
    -colorspace smpte170m -color_primaries smpte170m -color_trc smpte170m -color_range pc

echo "== hard scene cut (testsrc2 -> smptehdbars at t=1.5 s) + video-only clip"
if [[ ! -s h264_scenecut.mp4 || $FORCE -eq 1 ]]; then
  echo "  make h264_scenecut.mp4"
  "${FF[@]}" -f lavfi -i "testsrc2=size=640x360:rate=30:duration=1.5" \
      -f lavfi -i "smptehdbars=size=640x360:rate=30:duration=1.5" \
      -filter_complex "[0:v][1:v]concat=n=2:v=1:a=0[v]" -map "[v]" "${X264[@]}" -pix_fmt yuv420p -an h264_scenecut.mp4 || FAILED=1
fi
CHECKS+=("h264_scenecut.mp4|h264|yuv420p\$|640x360")
NOAUDIO=1 mk h264_noaudio.mp4 640 360 30 2 h264 'yuv420p$' "${X264[@]}" -pix_fmt yuv420p

echo "== paths with spaces / Unicode"
mkdir -p "path with spaces" "unicode"
cp -f h264_8bit.mp4 "path with spaces/video with spaces.mp4"
cp -f h264_8bit.mp4 "unicode/phim_thử_nghiệm_日本語_ünï.mp4"

if [[ $LONG -eq 1 ]]; then
  echo "== long file (2h20m, 320x180, 15 fps) - this takes a while"
  if [[ ! -s long_2h20.mkv || $FORCE -eq 1 ]]; then
    "${FF[@]}" -f lavfi -i "testsrc2=size=320x180:rate=15:duration=8400" \
        -f lavfi -i "sine=frequency=330:sample_rate=48000:duration=8400" \
        -c:v libx264 -preset ultrafast -crf 32 -pix_fmt yuv420p -c:a aac -b:a 32k -g 150 long_2h20.mkv || FAILED=1
  fi
  CHECKS+=("long_2h20.mkv|h264|yuv420p\$|320x180")
fi

# ---------------------------------------------------------------------------------------------
echo
echo "== verification (ffprobe)"
printf '%-24s %-6s %-14s %-10s %-9s %-7s %-3s %-3s %-4s\n' file codec pix_fmt size fps dur aud sub rot
for c in "${CHECKS[@]}"; do
  IFS='|' read -r f ec ep es <<<"$c"
  if [[ ! -s "$f" ]]; then echo "MISSING: $f"; FAILED=1; continue; fi
  j="$("$FFPROBE" -v error -show_entries stream=index,codec_type,codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate:stream_side_data=rotation:format=duration -of json "$f")"
  # tiny json extraction without jq: use python from the repo venv or PATH, else grep
  PY="$ROOT/.venv/Scripts/python.exe"; [[ -x "$PY" ]] || PY="$(command -v python3 || command -v python || true)"
  if [[ -z "$PY" ]]; then echo "no python for json parsing; file exists: $f"; continue; fi
  line="$("$PY" -I -c '
import json,sys
d=json.loads(sys.stdin.read()); ec,ep,es=sys.argv[1:4]
import re
v=[s for s in d["streams"] if s["codec_type"]=="video"][0]
a=sum(1 for s in d["streams"] if s["codec_type"]=="audio")
s_=sum(1 for s in d["streams"] if s["codec_type"]=="subtitle")
rot=""
for sd in v.get("side_data_list",[]): rot=str(sd.get("rotation",rot))
def fr(x):
    n,dn=x.split("/"); return float(n)/float(dn) if float(dn) else 0
size="%sx%s"%(v["width"],v["height"])
ok = v["codec_name"]==ec and re.search(ep,v["pix_fmt"]) is not None and size==es
print("%s|%s|%s|%.3f|%.2f|%d|%d|%s|%s"%(v["codec_name"],v["pix_fmt"],size,fr(v["avg_frame_rate"]),float(d["format"]["duration"]),a,s_,rot,"OK" if ok else "BAD"))
' "$ec" "$ep" "$es" <<<"$j")"
  IFS='|' read -r vc vp vs vf vd va vsub vrot vok <<<"$line"
  printf '%-24s %-6s %-14s %-10s %-9s %-7s %-3s %-3s %-4s %s\n' "$f" "$vc" "$vp" "$vs" "$vf" "$vd" "$va" "$vsub" "${vrot:--}" "$vok"
  [[ "$vok" == OK ]] || { echo "  expected codec=$ec pix_fmt~$ep size=$es"; FAILED=1; }
done

echo
du -sh "$OUT" | sed 's/^/total size: /'
if [[ $FAILED -ne 0 ]]; then echo "SOME SAMPLES FAILED" >&2; exit 1; fi
echo "OK: ${#CHECKS[@]} samples in $OUT"
