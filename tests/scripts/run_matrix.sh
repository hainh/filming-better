#!/usr/bin/env bash
# Runs filming-better over every sample in tests/samples and checks, per file:
#   * exit code (0 expected),
#   * ffprobe properties of the output vs the input: resolution (x scale), fps (x interp, or CFR for VFR),
#     duration within 1 frame, audio / subtitle stream counts, codec, color tags (see check_output.py);
# then a set of CLI behaviour checks (exit codes, --progress json, unicode/space paths, wildcard batch,
# --skip-existing, partial batch failure).  Exit status 1 if anything failed.
#
# Usage: tests/scripts/run_matrix.sh [options]
#   --exe PATH       filming-better.exe  (default: app/target/release/filming-better.exe)
#   --samples DIR    default tests/samples (create them with tests/scripts/make_samples.sh)
#   --out DIR        default tests/out/matrix
#   --args "..."     arguments appended to `run`  (default: --upscale span_x2 --interp 2 --codec hevc --cq 20)
#   --ext EXT        output container (default mkv; try mp4 / mov)
#   --filter REGEX   only samples whose file name matches
#   --timeout SEC    per-run timeout (default 600)
#   --no-cli         skip the CLI behaviour checks
#   --cli-only       only the CLI checks
#   --dry-run        list what would run and the expected output properties; do not execute the exe
#
# The expected scale/interp are parsed from --args (--upscale *_x2 / *_x4 -> 2/4 ; --interp N).
# Env: FFMPEG/FFPROBE/PYTHON override tools. Manual-only rows (Ctrl+C, 2 h file, GPUs, old driver) are in tests/README.md.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
EXE="$ROOT/app/target/release/filming-better.exe"
SAMPLES="$ROOT/tests/samples"
OUT="$ROOT/tests/out/matrix"
ARGS="--upscale span_x2 --interp 2 --codec hevc --cq 20"
EXT="mkv"; FILTER=""; TIMEOUT=600; DO_MATRIX=1; DO_CLI=1; DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exe) EXE="$2"; shift ;;
    --samples) SAMPLES="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --args) ARGS="$2"; shift ;;
    --ext) EXT="$2"; shift ;;
    --filter) FILTER="$2"; shift ;;
    --timeout) TIMEOUT="$2"; shift ;;
    --no-cli) DO_CLI=0 ;;
    --cli-only) DO_MATRIX=0 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

DEFAULT_FF_DIR="C:/ffmpeg-8.0.1-full_build-shared/bin"
if [[ -z "${FFPROBE:-}" ]]; then
  if command -v ffprobe >/dev/null 2>&1; then FFPROBE=ffprobe; else FFPROBE="$DEFAULT_FF_DIR/ffprobe.exe"; fi
fi
PY="${PYTHON:-}"
[[ -z "$PY" && -x "$ROOT/.venv/Scripts/python.exe" ]] && PY="$ROOT/.venv/Scripts/python.exe"
[[ -z "$PY" ]] && PY="$(command -v python3 || command -v python || true)"
[[ -n "$PY" ]] || { echo "python not found (needed by check_output.py)" >&2; exit 2; }

[[ -d "$SAMPLES" ]] || { echo "no samples dir $SAMPLES - run tests/scripts/make_samples.sh first" >&2; exit 2; }
if [[ $DRY -eq 0 && ! -x "$EXE" && ! -f "$EXE" ]]; then
  echo "filming-better.exe not found: $EXE" >&2
  echo "build it: cargo build --release --manifest-path app/Cargo.toml  (or pass --exe)" >&2
  exit 2
fi
mkdir -p "$OUT"

# expected transformation from --args
SCALE=1
if [[ "$ARGS" =~ --upscale[=\ ]+[A-Za-z0-9_]*x([0-9]) ]]; then SCALE="${BASH_REMATCH[1]}"; fi
INTERP=1
if [[ "$ARGS" =~ --interp[=\ ]+([0-9]+) ]]; then INTERP="${BASH_REMATCH[1]}"; fi
CODEC=""
if [[ "$ARGS" =~ --codec[=\ ]+([a-z0-9]+) ]]; then CODEC="${BASH_REMATCH[1]}"; fi
echo "exe      : $EXE"
echo "samples  : $SAMPLES"
echo "args     : run --input <sample> --output <out.$EXT> $ARGS   (expect scale=$SCALE interp=$INTERP codec=${CODEC:-any})"

declare -a SUMMARY=()
FAIL=0
record() { SUMMARY+=("$1|$2|$3"); [[ "$2" == FAIL ]] && FAIL=$((FAIL+1)); return 0; }

run_exe() {  # run_exe <logfile> args... ; returns the exe exit status (124 on timeout)
  local log="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$TIMEOUT" "$EXE" "$@" >"$log.out" 2>"$log.err"
  else
    "$EXE" "$@" >"$log.out" 2>"$log.err"
  fi
}

# ===================================================================================================
if [[ $DO_MATRIX -eq 1 ]]; then
  echo; echo "=== format / difficult-data matrix ==="
  # bench_* inputs belong to bench.sh; long_* only with --filter long
  mapfile -t FILES < <(find "$SAMPLES" -maxdepth 1 -type f \( -name '*.mp4' -o -name '*.mkv' -o -name '*.mov' -o -name '*.ts' -o -name '*.webm' \) ! -name 'bench_*' | sort)
  for f in "${FILES[@]}"; do
    name="$(basename "$f")"
    [[ -n "$FILTER" && ! "$name" =~ $FILTER ]] && continue
    [[ -z "$FILTER" && "$name" == long_* ]] && continue
    stem="${name%.*}_${name##*.}"
    out="$OUT/${stem}.$EXT"
    vfr=()
    [[ "$name" == *vfr* ]] && vfr=(--vfr)
    echo "--- $name"
    if [[ $DRY -eq 1 ]]; then
      echo "    would run: run --input $name --output ${stem}.$EXT $ARGS"
      echo "    input: $("$FFPROBE" -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate -of csv=p=0 "$f" | tr -d '
') -> expect scale x$SCALE, fps x$INTERP"
      record "$name" SKIP "dry-run"
      continue
    fi
    rm -f "$out"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    run_exe "$OUT/$stem.log" run --input "$f" --output "$out" $ARGS
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [[ $rc -ne 0 ]]; then
      echo "    exit code $rc (expected 0) after ${dt}s; stderr:"; tail -5 "$OUT/$stem.log.err" | sed 's/^/      /'
      record "$name" FAIL "exit $rc"
      continue
    fi
    if [[ ! -s "$out" ]]; then echo "    output missing"; record "$name" FAIL "no output"; continue; fi
    cargs=(--scale "$SCALE" --interp "$INTERP" --ffprobe "$FFPROBE")
    [[ -n "$CODEC" ]] && cargs+=(--codec "$CODEC")
    if "$PY" -I "$HERE/check_output.py" "$f" "$out" "${cargs[@]}" "${vfr[@]}"; then
      record "$name" PASS "${dt}s"
    else
      record "$name" FAIL "output properties"
    fi
  done
fi

# ===================================================================================================
# Git Bash does not convert /c/... to C:/... inside arguments containing wildcards: pass mixed-style paths
WP() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }

cli_check() {  # cli_check <name> <expected-rc-regex> <logbase> <args...>
  local name="$1" want="$2" log="$3"; shift 3
  if [[ $DRY -eq 1 ]]; then echo "    would run: $*"; record "cli: $name" SKIP "dry-run"; return 0; fi
  run_exe "$log" "$@"; local rc=$?
  if [[ "$rc" =~ ^($want)$ ]]; then echo "    [PASS] $name (exit $rc)"; record "cli: $name" PASS "exit $rc"
  else echo "    [FAIL] $name: exit $rc, expected /$want/"; tail -3 "$log.err" | sed 's/^/        /'; record "cli: $name" FAIL "exit $rc"; fi
  LAST_RC=$rc
}

if [[ $DO_CLI -eq 1 ]]; then
  echo; echo "=== CLI behaviour ==="
  C="$OUT/cli"; mkdir -p "$C"
  SMALL="$SAMPLES/h264_8bit.mp4"
  [[ -f "$SMALL" ]] || { echo "need $SMALL"; exit 2; }

  cli_check "--help" 0 "$C/help" --help
  cli_check "run --help" 0 "$C/runhelp" run --help
  cli_check "--version" 0 "$C/ver" --version
  cli_check "gpu-info" 0 "$C/gpu" gpu-info
  cli_check "unknown subcommand => usage error (2)" 2 "$C/bad1" frobnicate
  cli_check "missing required args => usage error (2)" 2 "$C/bad2" run
  cli_check "input file does not exist => non-zero" '[1-9][0-9]*' "$C/nofile" run --input "$C/does_not_exist.mp4" --output "$C/x.$EXT"
  # shellcheck disable=SC2086
  cli_check "unknown model id => 2 or 7" '2|7' "$C/nomodel" run --input "$SMALL" --output "$C/nm.$EXT" --upscale no_such_model
  cli_check "models list" 0 "$C/mlist" models list

  # --progress json: every non-empty stdout line must be a JSON object with frame/total/fps/eta
  if [[ $DRY -eq 0 ]]; then
    # shellcheck disable=SC2086
    run_exe "$C/json" run --input "$SMALL" --output "$C/json.$EXT" $ARGS --progress json; rc=$?
    if [[ $rc -eq 0 ]] && "$PY" -I - "$C/json.out" <<'PYEOF'
import json, sys
lines = [l for l in open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines() if l.strip()]
if not lines: sys.exit("no JSON progress lines on stdout")
for l in lines:
    d = json.loads(l)
    missing = [k for k in ("frame", "total", "fps", "eta") if k not in d]
    if missing: sys.exit("line %r misses %s" % (l, missing))
print("    %d JSON progress lines OK" % len(lines))
PYEOF
    then echo "    [PASS] --progress json (exit 0, valid JSON lines)"; record "cli: --progress json" PASS ""
    else echo "    [FAIL] --progress json (exit $rc)"; tail -3 "$C/json.err" | sed 's/^/        /'; record "cli: --progress json" FAIL "exit $rc"; fi
  else record "cli: --progress json" SKIP dry-run; fi

  # paths with spaces / Unicode
  if [[ -f "$SAMPLES/path with spaces/video with spaces.mp4" ]]; then
    mkdir -p "$C/out with spaces ünï"
    # shellcheck disable=SC2086
    cli_check "path with spaces" 0 "$C/spaces" run --input "$SAMPLES/path with spaces/video with spaces.mp4" --output "$C/out with spaces ünï/o ut.$EXT" $ARGS
    [[ $DRY -eq 0 && ! -s "$C/out with spaces ünï/o ut.$EXT" ]] && { echo "    [FAIL] output file with spaces not created"; record "cli: spaces output" FAIL ""; }
  fi
  U="$(find "$SAMPLES/unicode" -type f 2>/dev/null | head -1)"
  if [[ -n "$U" ]]; then
    mkdir -p "$C/ünïcödé_日本語"
    # shellcheck disable=SC2086
    cli_check "unicode path" 0 "$C/unicode" run --input "$U" --output "$C/ünïcödé_日本語/出力.$EXT" $ARGS
    [[ $DRY -eq 0 && ! -s "$C/ünïcödé_日本語/出力.$EXT" ]] && { echo "    [FAIL] unicode output file not created"; record "cli: unicode output" FAIL ""; }
  fi

  # wildcard batch + --skip-existing + one bad file in the batch (exit 8)
  B="$C/batch"; rm -rf "$B"; mkdir -p "$B/in" "$B/out"
  cp "$SAMPLES"/h264_8bit.mp4 "$SAMPLES"/h264_8bit.mov "$SAMPLES"/h264_8bit.ts "$B/in/" 2>/dev/null
  # shellcheck disable=SC2086
  cli_check "wildcard batch" 0 "$C/wild" run --input "$(WP "$B/in")/h264_8bit.*" --output-dir "$B/out" $ARGS
  if [[ $DRY -eq 0 ]]; then
    n=$(find "$B/out" -type f | wc -l)
    if [[ $n -ge 3 ]]; then echo "    [PASS] wildcard produced $n files"; record "cli: wildcard output count" PASS "$n"
    else echo "    [FAIL] wildcard produced $n files (expected 3)"; record "cli: wildcard output count" FAIL "$n"; fi
    before="$(find "$B/out" -type f -printf '%p %T@\n' 2>/dev/null | sort | md5sum)"
    # shellcheck disable=SC2086
    cli_check "--skip-existing re-run" 0 "$C/skip" run --input "$(WP "$B/in")/h264_8bit.*" --output-dir "$B/out" $ARGS --skip-existing
    after="$(find "$B/out" -type f -printf '%p %T@\n' 2>/dev/null | sort | md5sum)"
    if [[ "$before" == "$after" ]]; then echo "    [PASS] existing outputs untouched"; record "cli: skip-existing untouched" PASS ""
    else echo "    [FAIL] --skip-existing modified existing outputs"; record "cli: skip-existing untouched" FAIL ""; fi
    # corrupt file + 1 good file => exit 8, the good one still produced
    rm -rf "$B/in2" "$B/out2"; mkdir -p "$B/in2" "$B/out2"
    head -c 4096 /dev/urandom > "$B/in2/a_corrupt.mp4"
    cp "$SMALL" "$B/in2/b_good.mp4"
    # shellcheck disable=SC2086
    cli_check "batch with a corrupt file => exit 8" 8 "$C/partial" run --input "$B/in2" --output-dir "$B/out2" $ARGS
    if find "$B/out2" -type f -name 'b_good*' | grep -q .; then echo "    [PASS] good file still processed"; record "cli: good file survives bad one" PASS ""
    else echo "    [FAIL] good file missing in $B/out2"; record "cli: good file survives bad one" FAIL ""; fi
  fi
  echo "    (manual only: Ctrl+C mid-run => exit 130 + playable output; see tests/README.md)"
fi

# ===================================================================================================
echo; echo "=== summary ==="
printf '%-34s %-5s %s\n' item result detail
npass=0; nfail=0; nskip=0
for r in "${SUMMARY[@]}"; do
  IFS='|' read -r n s d <<<"$r"
  printf '%-34s %-5s %s\n' "$n" "$s" "$d"
  case "$s" in PASS) npass=$((npass+1));; FAIL) nfail=$((nfail+1));; *) nskip=$((nskip+1));; esac
done
echo "pass=$npass fail=$nfail skip=$nskip"
[[ $nfail -eq 0 ]]
