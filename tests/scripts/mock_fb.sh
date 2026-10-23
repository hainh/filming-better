#!/usr/bin/env bash
# MOCK of filming-better.exe used ONLY to test the test harness (run_matrix.sh) without a GPU / without the
# real binary:   tests/scripts/run_matrix.sh --exe tests/scripts/mock_fb.sh
# It implements the documented CLI surface with plain ffmpeg (scale + frame duplication + libx265).
set -u
FF="${FFMPEG:-ffmpeg}"; command -v "$FF" >/dev/null 2>&1 || FF="C:/ffmpeg-8.0.1-full_build-shared/bin/ffmpeg.exe"
FP="${FFPROBE:-ffprobe}"; command -v "$FP" >/dev/null 2>&1 || FP="C:/ffmpeg-8.0.1-full_build-shared/bin/ffprobe.exe"
cmd="${1:-}"; [[ $# -gt 0 ]] && shift
case "$cmd" in
  --help|-h) echo "filming-better mock: run | gpu-info | models | preview"; exit 0 ;;
  --version) echo "filming-better 0.0.0-mock"; exit 0 ;;
  gpu-info) echo "mock GPU"; exit 0 ;;
  models) echo "span_x2 rife_v4"; exit 0 ;;
  run) ;;
  *) echo "error: unrecognized subcommand '$cmd'" >&2; exit 2 ;;
esac
in=""; out=""; outdir=""; up=""; interp=1; codec=hevc; cq=23; skip=0; progress=bar
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) echo "usage: run --input IN --output OUT [--upscale ID] [--interp N] [--codec C] [--cq N]"; exit 0 ;;
    --input) in="$2"; shift ;; --output) out="$2"; shift ;; --output-dir) outdir="$2"; shift ;;
    --upscale) up="$2"; shift ;; --interp) interp="$2"; shift ;; --codec) codec="$2"; shift ;; --cq) cq="$2"; shift ;;
    --progress) progress="$2"; shift ;; --skip-existing) skip=1 ;; --preset) shift ;;
    *) echo "error: unexpected argument '$1'" >&2; exit 2 ;;
  esac; shift
done
[[ -n "$in" ]] || { echo "error: --input is required" >&2; exit 2; }
scale=1
case "$up" in "") ;; *_x2) scale=2 ;; *_x4) scale=4 ;; *) echo "model '$up' not found" >&2; exit 7 ;; esac
one() { # in out
  local i="$1" o="$2"
  [[ -f "$i" ]] || { echo "input not found: $i" >&2; return 1; }
  local fps; fps="$("$FP" -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 "$i" 2>/dev/null | tr -d '\r')"
  [[ -n "$fps" ]] || { echo "cannot read $i" >&2; return 6; }
  local vf="scale=iw*$scale:ih*$scale:flags=bicubic"
  [[ "$interp" -gt 1 ]] && vf="fps=$(awk -v f="$fps" -v n="$interp" 'BEGIN{split(f,a,"/"); printf "%d/%d", a[1]*n, a[2]}'),$vf"
  local enc=(-c:v libx265 -preset ultrafast -crf "$cq" -x265-params log-level=error)
  [[ "$codec" == h264 ]] && enc=(-c:v libx264 -preset ultrafast -crf "$cq")
  [[ "$codec" == av1 ]] && enc=(-c:v libsvtav1 -preset 12 -crf 40)
  [[ "$progress" == json ]] && echo '{"frame":1,"total":1,"fps":1.0,"eta":0}'
  "$FF" -hide_banner -loglevel error -y -i "$i" -map 0 -vf "$vf" "${enc[@]}" -c:a copy -c:s copy -fps_mode cfr "$o" 2>&1 | head -3 >&2
  [[ ${PIPESTATUS[0]} -eq 0 ]] || return 6
}
if [[ -z "$outdir" ]]; then one "$in" "$out"; exit $?; fi
mkdir -p "$outdir"
shopt -s nullglob
if [[ -d "$in" ]]; then files=("$in"/*); else files=($in); fi
fail=0
for f in "${files[@]}"; do
  b="$(basename "$f")"; o="$outdir/${b%.*}_${b##*.}.mkv"
  [[ $skip -eq 1 && -s "$o" ]] && continue
  one "$f" "$o" || { fail=1; rm -f "$o"; }
done
[[ $fail -eq 1 ]] && exit 8
exit 0
