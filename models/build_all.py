"""Run the whole dev-side model pipeline (blueprint phase 1):

  download weights -> export ONNX -> validate ONNX (ORT vs PyTorch) -> build TRT engines
  -> bench engines with the lean runtime (fills fps_1080p in the manifest)

  .venv/Scripts/python.exe models/build_all.py [id ...] [--skip-validate] [--skip-bench] [--compat]

Default ids: span_x2 animevideov3_x4 rife_v4 rife_v4_lite   (add `x4plus` explicitly for the big model)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DEFAULT_IDS  # noqa: E402


def run(script, *args):
    cmd = [sys.executable, str(HERE / script), *args]
    print(f"\n>>> {' '.join(Path(c).name if c.endswith('.py') else c for c in cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"<<< {script} done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--compat", action="store_true", help="also run compat_check.py (portable vs native engine speed)")
    ap.add_argument("--workspace-mb", type=int, default=3072)
    a = ap.parse_args()
    ids = a.ids or DEFAULT_IDS
    run("download_weights.py", *ids)
    run("export_onnx.py", *ids)
    if not a.skip_validate:
        run("validate_onnx.py", *ids)
    run("build_engines.py", *ids, "--workspace-mb", str(a.workspace_mb))
    run("build_engines.py", "--arch", "sm75")  # prints a SKIP notice unless run on a Turing GPU
    if not a.skip_bench:
        run("bench_engines.py", *ids)
    if a.compat:
        run("compat_check.py", *ids)
    print("\nDone. Engines + manifest.json are in models/engines/ (manifest also copied to models/manifest.json).")


if __name__ == "__main__":
    main()
