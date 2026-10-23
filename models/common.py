"""Shared helpers / model registry for the models/ pipeline."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEIGHTS = HERE / "weights"
ONNX = HERE / "onnx"
ENGINES = HERE / "engines"
SAMPLE_VIDEO = ROOT / "Download.mp4"

sys.path.insert(0, str(HERE))


def setup_dll_dirs():
    """Make the TensorRT / CUDA runtime DLLs of the project venv loadable (Windows)."""
    sp = ROOT / ".venv" / "Lib" / "site-packages"
    dirs = [sp / "tensorrt_libs", sp / "tensorrt_lean_libs", ROOT / ".tools" / "cuda" / "bin"]
    for d in dirs:
        if d.is_dir():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(d))


# id -> build info. `profile` = (min, opt, max) as [H, W] for engine inputs.
MODELS = {
    "span_x2": dict(kind="upscale", scale=2, pad_multiple=1,
                    profile=((64, 64), (512, 512), (1088, 1088))),
    "animevideov3_x4": dict(kind="upscale", scale=4, pad_multiple=1,
                            profile=((64, 64), (512, 512), (1088, 1088))),
    "x4plus": dict(kind="upscale", scale=4, pad_multiple=1,
                   profile=((64, 64), (256, 256), (512, 512))),
    "rife_v4": dict(kind="interp", scale=1, pad_multiple=64, rife_version="4.26",
                    profile=((64, 64), (576, 1024), (1920, 1920))),
    "rife_v4_lite": dict(kind="interp", scale=1, pad_multiple=64, rife_version="4.22.lite",
                         profile=((64, 64), (576, 1024), (1920, 1920))),
}
DEFAULT_IDS = ["span_x2", "animevideov3_x4", "rife_v4", "rife_v4_lite"]


def load_frames(n=20, scale_down=None):
    """Return n evenly spaced RGB uint8 frames (H,W,3) from the sample video."""
    import cv2
    cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = [int(i * (total - 2) / max(n - 1, 1)) for i in range(n)]
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {i} of {SAMPLE_VIDEO}")
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_frame_pairs(n=20, step=1):
    """n pairs of consecutive frames (RGB uint8)."""
    import cv2
    cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = [int(i * (total - 2 - step) / max(n - 1, 1)) for i in range(n)]
    pairs = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok0, a = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, i + step)
        ok1, b = cap.read()
        if not (ok0 and ok1):
            raise RuntimeError("cannot read frame pair")
        pairs.append((cv2.cvtColor(a, cv2.COLOR_BGR2RGB), cv2.cvtColor(b, cv2.COLOR_BGR2RGB)))
    cap.release()
    return pairs


def to_chw01(img_u8):
    import numpy as np
    return (img_u8.astype("float32") / 255.0).transpose(2, 0, 1)[None].copy()


def psnr(a, b, peak=1.0):
    import numpy as np
    mse = float(np.mean((a.astype("float64") - b.astype("float64")) ** 2))
    if mse == 0:
        return 99.0
    return 10.0 * np.log10(peak * peak / mse)


def pad_to(x, mult):
    """Pad NCHW numpy array (replicate) so H and W are multiples of mult; returns (padded, (H, W))."""
    import numpy as np
    h, w = x.shape[2:]
    ph = (-h) % mult
    pw = (-w) % mult
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="edge")
    return x, (h, w)


def onnx_path(mid, prec):
    return ONNX / f"{mid}.{prec}.onnx"


def dump_manifest(man):
    """json.dumps with indent=2 but short integer arrays kept on one line (as in docs/contracts.md)."""
    import json
    import re
    text = json.dumps(man, indent=2)
    return re.sub(r"\[\s+(-?\d+(?:,\s+-?\d+)*)\s+\]",
                  lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", text) + "\n"
