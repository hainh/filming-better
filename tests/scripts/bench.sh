#!/usr/bin/env bash
# Fixed performance benchmark ("Hieu nang" row of the test matrix): run on every release to catch speed regressions.
#
# * builds two deterministic inputs (10 s, 24 fps, noisy testsrc2): 1080p and 540p (tests/samples/bench_*.mp4)
# * runs a fixed list of configurations N times each (default 3, + 1 warm-up), takes the MEDIAN wall time
# * metric: input frames / second (fps_in) and output frames / second (fps_out)
# * writes tests/out/bench/bench-<date>-<gpu>.json and compares fps_in with tests/bench/baseline.json
#   (per GPU name); exit 1 if any config is slower than baseline by more than --tolerance (default 10 %)
#
# Usage: tests/scripts/bench.sh [--exe PATH] [--runs N] [--tolerance 0.10] [--update-baseline] [--full] [--dry-run]
#   --full             adds the heavy 1080p -> 4K config (upscale x2 + interp x2 on 1080p, the phase-2 target)
#   --update-baseline  store this run as the new baseline for this GPU (do it deliberately, commit the file)
# Env: FFMPEG / FFPROBE / PYTHON override tools.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
EXE="$ROOT/app/target/release/filming-better.exe"
RUNS=3; TOL="0.10"; UPDATE=0; FULL=0; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --exe) EXE="$2"; shift ;; --runs) RUNS="$2"; shift ;; --tolerance) TOL="$2"; shift ;;
    --update-baseline) UPDATE=1 ;; --full) FULL=1 ;; --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done

DEFAULT_FF_DIR="C:/ffmpeg-8.0.1-full_build-shared/bin"
if [[ -z "${FFMPEG:-}" ]]; then command -v ffmpeg >/dev/null 2>&1 && FFMPEG=ffmpeg || FFMPEG="$DEFAULT_FF_DIR/ffmpeg.exe"; fi
if [[ -z "${FFPROBE:-}" ]]; then command -v ffprobe >/dev/null 2>&1 && FFPROBE=ffprobe || FFPROBE="$DEFAULT_FF_DIR/ffprobe.exe"; fi
PY="${PYTHON:-}"; [[ -z "$PY" && -x "$ROOT/.venv/Scripts/python.exe" ]] && PY="$ROOT/.venv/Scripts/python.exe"
[[ -z "$PY" ]] && PY="$(command -v python3 || command -v python || true)"
[[ -n "$PY" ]] || { echo "python not found" >&2; exit 2; }

SAMPLES="$ROOT/tests/samples"; OUT="$ROOT/tests/out/bench"; BASE="$ROOT/tests/bench/baseline.json"
mkdir -p "$SAMPLES" "$OUT" "$(dirname "$BASE")"

# ---- fixed inputs (deterministic: testsrc2 + seeded noise, x264 crf 16) -----------------------
IN1080="$SAMPLES/bench_1080p24_10s.mp4"; IN540="$SAMPLES/bench_540p24_10s.mp4"
if [[ ! -s "$IN1080" ]]; then
  echo "creating $IN1080"
  "$FFMPEG" -hide_banner -loglevel error -y -f lavfi -i "testsrc2=size=1920x1080:rate=24:duration=10" \
    -vf "noise=alls=10:allf=t:all_seed=1234,format=yuv420p" -c:v libx264 -preset veryfast -crf 16 -an "$IN1080" || exit 2
fi
if [[ ! -s "$IN540" ]]; then
  "$FFMPEG" -hide_banner -loglevel error -y -i "$IN1080" -vf "scale=960:540:flags=area" -c:v libx264 -preset veryfast -crf 14 -pix_fmt yuv420p -an "$IN540" || exit 2
fi
FRAMES=240

# name | input | args
CONFIGS=(
  "interp2_1080p|$IN1080|--interp 2 --codec hevc --cq 22"
  "up2_540p_to_1080p|$IN540|--upscale span_x2 --codec hevc --cq 22"
  "up2_interp2_540p|$IN540|--upscale span_x2 --interp 2 --codec hevc --cq 22"
)
[[ $FULL -eq 1 ]] && CONFIGS+=("up2_interp2_1080p_to_4k|$IN1080|--upscale span_x2 --interp 2 --codec hevc --cq 22")

GPU="unknown"; DRIVER="unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d '\r')"
  DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '\r')"
fi
GIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "GPU      : $GPU (driver $DRIVER)"
echo "exe      : $EXE"
echo "runs     : $RUNS (+1 warm-up), tolerance $TOL"
if [[ $DRY -eq 1 ]]; then
  for c in "${CONFIGS[@]}"; do IFS='|' read -r n i a <<<"$c"; echo "would run [$n]: run --input $(basename "$i") --output <out>.mkv $a"; done
  exit 0
fi
[[ -f "$EXE" ]] || { echo "exe not found: $EXE" >&2; exit 2; }

now() { date +%s.%N; }
declare -A MEDIAN
TMPRES="$(mktemp)"
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r name input args <<<"$c"
  times=()
  for ((r = 0; r <= RUNS; r++)); do
    o="$OUT/${name}.mkv"; rm -f "$o"
    t0=$(now)
    # shellcheck disable=SC2086
    "$EXE" run --input "$input" --output "$o" $args >"$OUT/$name.log" 2>&1; rc=$?
    t1=$(now)
    if [[ $rc -ne 0 ]]; then echo "[$name] run failed with exit $rc (see $OUT/$name.log)"; tail -3 "$OUT/$name.log"; exit 1; fi
    dt="$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')"
    if [[ $r -eq 0 ]]; then echo "[$name] warm-up ${dt}s"; else times+=("$dt"); echo "[$name] run $r: ${dt}s"; fi
  done
  med="$(printf '%s\n' "${times[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')"
  interp=1; [[ "$args" =~ --interp[=\ ]+([0-9]+) ]] && interp="${BASH_REMATCH[1]}"
  echo "$name $med $interp" >> "$TMPRES"
done

# ---- report + baseline comparison ---------------------------------------------------------
STAMP="$(date +%Y%m%d-%H%M%S)"
SAFE_GPU="$(echo "$GPU" | tr -c 'A-Za-z0-9\n' '_')"
RES="$OUT/bench-$STAMP-$SAFE_GPU.json"
"$PY" -I - "$TMPRES" "$RES" "$BASE" "$GPU" "$DRIVER" "$GIT" "$FRAMES" "$TOL" "$UPDATE" <<'PYEOF'
import json, os, sys
tmp, res, base, gpu, driver, git, frames, tol, update = sys.argv[1:10]
frames = int(frames); tol = float(tol); update = update == "1"
rows = {}
for line in open(tmp):
    n, med, interp = line.split()
    med = float(med); interp = int(interp)
    rows[n] = {"seconds": med, "fps_in": round(frames / med, 2), "fps_out": round(frames * interp / med, 2)}
doc = {"gpu": gpu, "driver": driver, "git": git, "frames_in": frames, "results": rows}
json.dump(doc, open(res, "w"), indent=2)
print("\nresults written to", res)
b = json.load(open(base)) if os.path.exists(base) else {}
ref = b.get(gpu, {})
bad = 0
print("%-26s %9s %9s %9s %9s" % ("config", "seconds", "fps_in", "fps_out", "baseline"))
for n, r in rows.items():
    bl = ref.get(n)
    flag = ""
    if bl:
        change = r["fps_in"] / bl - 1.0
        flag = "%+.1f%%" % (change * 100)
        if change < -tol:
            flag += "  REGRESSION"; bad += 1
    else:
        flag = "(no baseline)"
    print("%-26s %9.2f %9.2f %9.2f %9s %s" % (n, r["seconds"], r["fps_in"], r["fps_out"], bl if bl else "-", flag))
if update:
    b[gpu] = {n: r["fps_in"] for n, r in rows.items()}
    os.makedirs(os.path.dirname(base), exist_ok=True)
    json.dump(b, open(base, "w"), indent=2)
    print("baseline updated for", gpu)
    bad = 0
sys.exit(1 if bad else 0)
PYEOF
rc=$?
rm -f "$TMPRES"
exit $rc
