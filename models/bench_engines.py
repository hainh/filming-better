"""Load the engines with the TensorRT LEAN runtime only (python `tensorrt_lean`), run them on the
GPU, measure speed and check accuracy against ONNX Runtime (FP32).

  python models/bench_engines.py [id ...] [--arch ampere+|sm86|all] [--iters 30] [--no-manifest]

This is the blueprint phase-0 go/no-go for "lean runtime can deserialize version-compatible
engines built with EXCLUDE_LEAN_RUNTIME". The full `tensorrt` module is NOT imported here.
CUDA memory / streams are driven through ctypes on cudart64_12.dll (no torch-cuda needed).

Reports:
  * what the lean runtime needs to deserialize (engine_host_code_allowed False vs True)
  * per-engine tensor names / dtypes / profile shapes
  * best-of-N ms per run and fps_1080p (upscale: largest tile scaled to a 1920x1080 frame equivalent;
    interp: one 1088x1920 pair) -> written to the manifest
  * PSNR of the TRT FP16 output vs ONNX Runtime FP32 on real frames
"""
import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

import common
from trt_runner import Cuda, Engine, make_logger, random_feeds
from common import ENGINES, HERE, MODELS, onnx_path, psnr, to_chw01, pad_to

common.setup_dll_dirs()
import tensorrt_lean as trt  # noqa: E402  (lean runtime python bindings, nvinfer_lean_11.dll)

assert "tensorrt" not in sys.modules, "bench must use the lean runtime only"


def LeanEngine(cu, path, host_code=True):
    return Engine(trt, cu, path, host_code)


def accuracy(mid, eng, n=3):
    """PSNR (dB) of TRT fp16 vs ONNX Runtime fp32 on real frames."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path(mid, "fp32")), providers=["CPUExecutionProvider"])
    out = []
    if MODELS[mid]["kind"] == "upscale":
        for f in common.load_frames(n):
            x = to_chw01(f)
            ref = sess.run(["output"], {"input": x})[0]
            got = eng.run({"input": x.astype(np.float16)})[0]["output"].astype(np.float32)
            out.append(psnr(np.clip(got, 0, 1), np.clip(ref, 0, 1)))
    else:
        for i, (a, b) in enumerate(common.load_frame_pairs(n, step=2)):
            xa, _ = pad_to(to_chw01(a), 64)
            xb, _ = pad_to(to_chw01(b), 64)
            t = np.full((1, 1, 1, 1), [0.5, 0.25, 0.75][i % 3], dtype=np.float32)
            ref = sess.run(["output"], {"img0": xa, "img1": xb, "timestep": t})[0]
            got = eng.run({"img0": xa.astype(np.float16), "img1": xb.astype(np.float16),
                           "timestep": t.astype(np.float16)})[0]["output"].astype(np.float32)
            out.append(psnr(np.clip(got, 0, 1), np.clip(ref, 0, 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--arch", default="all", help="only engines with this arch tag (default: every engine whose file exists)")
    ap.add_argument("--skip-accuracy", action="store_true")
    a = ap.parse_args()
    manifest_path = ENGINES / "manifest.json"
    man = json.loads(manifest_path.read_text())
    cu = Cuda()
    print(f"tensorrt_lean {trt.__version__}; GPU free/total MiB = {[x >> 20 for x in cu.mem_info()]}")
    results = []
    for mid in (a.ids or [m["id"] for m in man["models"]]):
        entry = next(m for m in man["models"] if m["id"] == mid)
        for e in entry["engines"]:
            if a.arch != "all" and e["arch"] != a.arch:
                continue
            path = ENGINES / e["file"]
            if not path.exists():
                print(f"\n(skip {mid} {e['arch']}: {path.name} not built)")
                continue
            print(f"\n=== {mid}  {e['file']}  {path.stat().st_size/1e6:.2f} MB")
            # --- what does the lean runtime need to deserialize?
            for hc in (False, True):
                try:
                    rt = trt.Runtime(make_logger(trt))
                    rt.engine_host_code_allowed = hc
                    eng_try = rt.deserialize_cuda_engine(path.read_bytes())
                    print(f"  deserialize with engine_host_code_allowed={hc}: {'OK' if eng_try is not None else 'FAILED (None)'}")
                    del eng_try, rt
                except Exception as ex:  # noqa
                    print(f"  deserialize with engine_host_code_allowed={hc}: EXCEPTION {ex}")
            le = LeanEngine(cu, path, host_code=True)
            for n, d in le.describe().items():
                print(f"  {n:9s} {d}")
            print(f"  engine.device_memory_size_v2 (context scratch at max profile): {le.engine.device_memory_size_v2 / 2**20:.0f} MiB")
            mx = e["max_shape"]
            if entry["kind"] == "upscale":
                th, tw = mx[2], mx[3]
                _, ms = le.run(random_feeds(mid, th, tw), a.iters, warmup=5)
                fps = 1000.0 / (ms * (1920 * 1080) / (th * tw))
                label = (f"tile {tw}x{th}: best {ms:.2f} ms (median {le.last_stats['median']:.2f}) -> "
                         f"{fps:.1f} fps per 1080p-equivalent")
            else:
                _, ms = le.run(random_feeds(mid, 1088, 1920), a.iters, warmup=5)
                fps = 1000.0 / ms
                label = (f"1920x1088 pair: best {ms:.2f} ms (median {le.last_stats['median']:.2f}) -> "
                         f"{fps:.1f} interpolated frames/s")
            print("  " + label)
            free, tot = cu.mem_info()
            print(f"  VRAM free after context+buffers: {free >> 20} / {tot >> 20} MiB (shared GPU)")
            ps = []
            if not a.skip_accuracy:
                ps = accuracy(mid, le)
                print("  PSNR TRT-fp16(lean) vs ORT-fp32 on real frames: " + ", ".join(f"{p:.2f}" for p in ps) + " dB")
            e["fps_1080p"] = round(fps, 2 if fps < 1 else 1)
            results.append((mid, path.stat().st_size / 1e6, ms, fps, min(ps) if ps else None, label))
            le.close()
    if not a.no_manifest:
        manifest_path.write_text(common.dump_manifest(man))
        (HERE / "manifest.json").write_text(common.dump_manifest(man))
        print("\nmanifest updated with fps_1080p")
    print(f"\n{'model':18s} {'MB':>7s} {'ms':>8s} {'fps_1080p':>10s} {'PSNR min':>9s}")
    for mid, mb, ms, fps, p, _ in results:
        print(f"{mid:18s} {mb:7.2f} {ms:8.2f} {fps:10.1f} {('%.2f' % p) if p else '-':>9s}")


if __name__ == "__main__":
    main()
