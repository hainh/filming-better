"""Compare ONNX Runtime (CPU) with PyTorch on 20 frames of Download.mp4. Requires PSNR > 40 dB.

Upscalers run on full frames; RIFE runs on 20 consecutive frame pairs with timesteps cycling
through 0.25/0.5/0.75. The PSNR is computed between the ONNX Runtime output and the PyTorch
output (both FP32), i.e. it measures export fidelity, not model quality.

Usage: python models/validate_onnx.py [id ...] [--frames 20] [--threshold 40]
"""
import argparse
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

import common
from common import DEFAULT_IDS, MODELS, onnx_path, psnr, to_chw01, pad_to
import export_onnx


def validate(mid, n, prec="fp32"):
    kind = MODELS[mid]["kind"]
    sess = ort.InferenceSession(str(onnx_path(mid, prec)), providers=["CPUExecutionProvider"])
    model = export_onnx.load_torch_model(mid)
    if prec == "fp16":
        model = model.half()
    dt = np.float16 if prec == "fp16" else np.float32
    tdt = torch.float16 if prec == "fp16" else torch.float32
    res = []
    if kind == "upscale":
        frames = common.load_frames(n)
        # last frame is cropped to an odd size to exercise the dynamic axes
        frames[-1] = frames[-1][:301, :437]
        for f in frames:
            x = to_chw01(f).astype(dt)
            with torch.no_grad():
                ref = model(torch.from_numpy(x).to(tdt)).float().numpy()
            out = sess.run(["output"], {"input": x})[0].astype(np.float32)
            assert out.shape == ref.shape == (1, 3, f.shape[0] * MODELS[mid]["scale"], f.shape[1] * MODELS[mid]["scale"]), (out.shape, ref.shape)
            res.append(psnr(out, ref))
    else:
        pairs = common.load_frame_pairs(n)
        ts = [0.25, 0.5, 0.75]
        for i, (a, b) in enumerate(pairs):
            xa, _ = pad_to(to_chw01(a), 64)
            xb, _ = pad_to(to_chw01(b), 64)
            t = np.full((1, 1, 1, 1), ts[i % 3], dtype=dt)
            xa, xb = xa.astype(dt), xb.astype(dt)
            with torch.no_grad():
                ref = model(torch.from_numpy(xa).to(tdt), torch.from_numpy(xb).to(tdt), torch.from_numpy(t).to(tdt)).float().numpy()
            out = sess.run(["output"], {"img0": xa, "img1": xb, "timestep": t})[0].astype(np.float32)
            assert out.shape == ref.shape, (out.shape, ref.shape)
            res.append(psnr(out, ref))
    return res


def rife_quality(mid, n=4):
    """Informational sanity check of the exported RIFE graph: interpolate frame k and k+2 at t=0.5 and
    compare with the real frame k+1, against a plain 50/50 blend baseline."""
    import cv2
    sess = ort.InferenceSession(str(onnx_path(mid, "fp32")), providers=["CPUExecutionProvider"])
    cap = cv2.VideoCapture(str(common.SAMPLE_VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    for i in range(n):
        start = 20 + i * (total - 60) // max(n - 1, 1)
        fr = []
        for k in range(3):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start + k)
            ok, f = cap.read()
            fr.append(pad_to(to_chw01(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), 64)[0])
        out = sess.run(["output"], {"img0": fr[0], "img1": fr[2], "timestep": np.full((1, 1, 1, 1), 0.5, np.float32)})[0]
        rows.append((psnr(out, fr[1]), psnr((fr[0] + fr[2]) / 2, fr[1])))
    cap.release()
    print(f"[{mid}] quality sanity (t=0.5 from frames k,k+2 vs real k+1): "
          f"RIFE {np.mean([r[0] for r in rows]):.2f} dB vs blend {np.mean([r[1] for r in rows]):.2f} dB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=40.0)
    ap.add_argument("--prec", default="fp32")
    a = ap.parse_args()
    ids = a.ids or DEFAULT_IDS
    rows = []
    ok_all = True
    for mid in ids:
        t0 = time.time()
        r = validate(mid, a.frames, a.prec)
        ok = min(r) > a.threshold
        ok_all &= ok
        rows.append((mid, len(r), min(r), float(np.mean(r)), max(r), ok, time.time() - t0))
        if MODELS[mid]["kind"] == "interp" and a.prec == "fp32":
            rife_quality(mid)
        print(f"[{mid}] done in {time.time()-t0:.0f}s min={min(r):.2f} mean={np.mean(r):.2f}", flush=True)
    print(f"\n{'model':18s} {'frames':>6s} {'PSNR min':>9s} {'mean':>8s} {'max':>8s}  result")
    for mid, n, lo, mean, hi, ok, _ in rows:
        print(f"{mid:18s} {n:6d} {lo:9.2f} {mean:8.2f} {hi:8.2f}  {'PASS' if ok else 'FAIL'} (>{a.threshold} dB)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
