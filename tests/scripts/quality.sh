#!/usr/bin/env bash
# Objective quality metrics of a processed video against a reference: PSNR, SSIM, VMAF (libvmaf),
# and a temporal-flicker ratio (tile seams / frame-to-frame instability).
#
# Usage:
#   quality.sh <distorted> <reference> [options]
#   quality.sh --make-lowres <reference> [--factor 2] [--out lowres.mp4]    # builds the app INPUT from a reference
#
# Options:
#   --no-vmaf            skip VMAF (slow)
#   --flicker            also compute the temporal flicker ratio (distorted vs reference)
#   --min-psnr  <dB>     exit 1 if average PSNR is lower
#   --min-vmaf  <score>  exit 1 if VMAF mean is lower
#   --max-flicker <x>    exit 1 if flicker ratio is higher (default 1.25 when --flicker)
#   --frames N           only compare the first N frames
#   --json <file>        write the results as JSON
#
# The distorted video is scaled (bicubic) to the reference size if they differ, and both are converted
# to yuv420p and timestamp-aligned. If the two have different frame rates (e.g. after 2x interpolation) the
# distorted one is resampled to the reference fps; use a reference of the same fps for a fair score.
#
# Typical workflow for the "Chat luong" row of the test matrix (see tests/README.md):
#   quality.sh --make-lowres ref_1080p.mp4 --out in_540p.mp4
#   filming-better run --input in_540p.mp4 --output out_1080p.mkv --upscale span_x2 --codec hevc --cq 18
#   quality.sh out_1080p.mkv ref_1080p.mp4 --flicker --min-psnr 28 --min-vmaf 60
#
# Env: FFMPEG / FFPROBE override the binaries (default: PATH, then C:/ffmpeg-8.0.1-full_build-shared/bin).
set -euo pipefail

DEFAULT_FF_DIR="C:/ffmpeg-8.0.1-full_build-shared/bin"
if [[ -z "${FFMPEG:-}" ]]; then
  if command -v ffmpeg >/dev/null 2>&1; then FFMPEG=ffmpeg; else FFMPEG="$DEFAULT_FF_DIR/ffmpeg.exe"; fi
fi
if [[ -z "${FFPROBE:-}" ]]; then
  if command -v ffprobe >/dev/null 2>&1; then FFPROBE=ffprobe; else FFPROBE="$DEFAULT_FF_DIR/ffprobe.exe"; fi
fi

usage() { sed -n '2,28p' "${BASH_SOURCE[0]}"; exit "${1:-2}"; }

# ---- --make-lowres mode ---------------------------------------------------------------------
if [[ "${1:-}" == "--make-lowres" ]]; then
  shift
  ref="${1:-}"; [[ -f "$ref" ]] || usage; shift
  factor=2; out=""
  while [[ $# -gt 0 ]]; do
    case "$1" in --factor) factor="$2"; shift ;; --out) out="$2"; shift ;; *) usage ;; esac; shift
  done
  [[ -n "$out" ]] || out="${ref%.*}_lowres_x${factor}.mp4"
  "$FFMPEG" -hide_banner -loglevel error -y -i "$ref" -vf "scale=iw/${factor}:ih/${factor}:flags=area" \
      -c:v libx264 -crf 12 -preset medium -pix_fmt yuv420p -an "$out"
  echo "$out"
  exit 0
fi

# ---- metrics mode ---------------------------------------------------------------------------
[[ $# -ge 2 ]] || usage
dist="$1"; ref="$2"; shift 2
DO_VMAF=1; DO_FLICKER=0; MIN_PSNR=""; MIN_VMAF=""; MAX_FLICKER="1.25"; FRAMES=""; JSON=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-vmaf) DO_VMAF=0 ;;
    --flicker) DO_FLICKER=1 ;;
    --min-psnr) MIN_PSNR="$2"; shift ;;
    --min-vmaf) MIN_VMAF="$2"; shift ;;
    --max-flicker) MAX_FLICKER="$2"; shift ;;
    --frames) FRAMES="$2"; shift ;;
    --json) JSON="$2"; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option $1" >&2; usage ;;
  esac
  shift
done
[[ -f "$dist" ]] || { echo "not found: $dist" >&2; exit 2; }
[[ -f "$ref"  ]] || { echo "not found: $ref" >&2; exit 2; }

probe() { "$FFPROBE" -v error -select_streams v:0 -show_entries "stream=$1" -of csv=p=0 "$2" | head -1 | tr -d '\r'; }
RW="$(probe width "$ref")"; RH="$(probe height "$ref")"
RFPS="$(probe r_frame_rate "$ref")"
DW="$(probe width "$dist")"; DH="$(probe height "$dist")"
echo "reference : $ref  ${RW}x${RH} @ $RFPS"
echo "distorted : $dist  ${DW}x${DH}"

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
LIM=(); [[ -n "$FRAMES" ]] && LIM=(-frames:v "$FRAMES")

# normalised streams: [d] distorted scaled to the reference size, [r] reference; both yuv420p, fps aligned
PRE="[0:v]scale=${RW}:${RH}:flags=bicubic,fps=${RFPS},setpts=PTS-STARTPTS,format=yuv420p[d];[1:v]fps=${RFPS},setpts=PTS-STARTPTS,format=yuv420p[r]"

run() { "$FFMPEG" -hide_banner -nostats -loglevel info -i "$dist" -i "$ref" "$@" 2>&1; }

# ---- PSNR + SSIM (one pass) -----------------------------------------------------------------
psnr_ssim_log="$(run -filter_complex "${PRE};[d]split[d1][d2];[r]split[r1][r2];[d1][r1]psnr;[d2][r2]ssim" "${LIM[@]}" -f null - | tr -d '\r')"
PSNR_AVG="$(grep -o 'PSNR y:[^ ]* u:[^ ]* v:[^ ]* average:[^ ]*' <<<"$psnr_ssim_log" | tail -1 | sed 's/.*average://')"
PSNR_Y="$(grep -o 'PSNR y:[^ ]*' <<<"$psnr_ssim_log" | tail -1 | sed 's/PSNR y://')"
SSIM_ALL="$(grep -o 'SSIM Y:[^ ]* ([^)]*) U:[^ ]* ([^)]*) V:[^ ]* ([^)]*) All:[^ ]*' <<<"$psnr_ssim_log" | tail -1 | sed 's/.*All://')"
[[ -n "$PSNR_AVG" ]] || { echo "$psnr_ssim_log" | tail -15; echo "could not parse PSNR" >&2; exit 3; }
printf 'PSNR  avg %s dB   (Y %s dB)\n' "$PSNR_AVG" "$PSNR_Y"
printf 'SSIM  all %s\n' "${SSIM_ALL:-n/a}"

# ---- VMAF -----------------------------------------------------------------------------------
VMAF=""
if [[ $DO_VMAF -eq 1 ]]; then
  if "$FFMPEG" -hide_banner -filters 2>/dev/null | grep -q libvmaf; then
    # relative log path + run from the temp dir avoids the C: / backslash escaping problems of filter options
    vlog="vmaf.json"
    adist="$(cd "$(dirname "$dist")" && pwd)/$(basename "$dist")"
    aref="$(cd "$(dirname "$ref")" && pwd)/$(basename "$ref")"
    vout="$( cd "$tmp" && "$FFMPEG" -hide_banner -nostats -loglevel info -i "$adist" -i "$aref" \
             -filter_complex "${PRE};[d][r]libvmaf=log_fmt=json:log_path=${vlog}:n_threads=4" "${LIM[@]}" -f null - 2>&1 | tr -d '\r' )"
    VMAF="$(grep -o 'VMAF score: [0-9.]*' <<<"$vout" | tail -1 | sed 's/VMAF score: //')"
    if [[ -z "$VMAF" ]]; then echo "$vout" | tail -10; echo "could not parse VMAF" >&2; else printf 'VMAF  mean %s\n' "$VMAF"; fi
  else
    echo "VMAF  skipped (this ffmpeg has no libvmaf)"
  fi
fi

# ---- flicker: mean |frame[n]-frame[n-1]| (luma) of distorted vs reference --------------------
FLICK=""
if [[ $DO_FLICKER -eq 1 ]]; then
  tdiff() {  # $1 = file ; prints mean YAVG of the temporal difference image
    "$FFMPEG" -hide_banner -nostats -loglevel error -i "$1" \
      -vf "scale=${RW}:${RH}:flags=bicubic,fps=${RFPS},format=gray,tblend=all_mode=difference,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-" \
      "${LIM[@]}" -f null - 2>&1 | tr -d '\r' | awk -F= '/YAVG/ {s+=$2; n++} END {if (n) printf "%.6f", s/n; else print "nan"}'
  }
  td="$(tdiff "$dist")"; tr_="$(tdiff "$ref")"
  FLICK="$(awk -v a="$td" -v b="$tr_" 'BEGIN { if (b+0 > 0) printf "%.3f", a/b; else print "nan" }')"
  printf 'FLICKER temporal-diff distorted %s / reference %s = ratio %s (limit %s)\n' "$td" "$tr_" "$FLICK" "$MAX_FLICKER"
fi

# ---- thresholds / json ----------------------------------------------------------------------
rc=0
lt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a+0 < b+0) }'; }
if [[ -n "$MIN_PSNR" ]] && lt "$PSNR_AVG" "$MIN_PSNR"; then echo "FAIL: PSNR $PSNR_AVG < $MIN_PSNR"; rc=1; fi
if [[ -n "$MIN_VMAF" ]]; then
  if [[ -z "$VMAF" ]]; then echo "FAIL: VMAF requested but not available"; rc=1
  elif lt "$VMAF" "$MIN_VMAF"; then echo "FAIL: VMAF $VMAF < $MIN_VMAF"; rc=1; fi
fi
if [[ -n "$FLICK" && "$FLICK" != nan ]] && lt "$MAX_FLICKER" "$FLICK"; then echo "FAIL: flicker ratio $FLICK > $MAX_FLICKER"; rc=1; fi
json_num() { case "$1" in ""|nan|NaN) echo null ;; inf|Inf) echo 9999 ;; *) echo "$1" ;; esac; }
if [[ -n "$JSON" ]]; then
  printf '{"distorted":"%s","reference":"%s","psnr_avg":%s,"ssim_all":%s,"vmaf":%s,"flicker_ratio":%s}\n' \
    "${dist//\\/\/}" "${ref//\\/\/}" "$(json_num "$PSNR_AVG")" "$(json_num "$SSIM_ALL")" "$(json_num "$VMAF")" "$(json_num "$FLICK")" > "$JSON"
fi
[[ $rc -eq 0 ]] && echo "quality: OK"
exit $rc
