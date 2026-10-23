"""Phase-0 criterion: the portable engine (VERSION_COMPATIBLE + AMPERE_PLUS, run with the LEAN runtime)
must not be more than 15% slower than an engine built specifically for this GPU (no compat flags,
FULL runtime).

  python models/compat_check.py [id ...]          (default: span_x2 rife_v4)

Builds a throw-away "native" engine (kept in %TEMP%, not in models/engines/), times it with the full
`tensorrt` runtime and then the shipped ampere+ engine with the lean runtime. The two runtimes are
run in separate subprocesses (they must not be loaded in the same process). Uses the opt shape of
the profile. Run on an otherwise idle GPU for stable numbers.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import common
from common import ENGINES, MODELS, onnx_path

common.setup_dll_dirs()


def worker_native(mid, iters, hw="none"):
    import tensorrt as trt
    from trt_runner import Cuda, Engine, make_logger, random_feeds
    logger = make_logger(trt)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, logger)
    assert parser.parse_from_file(str(onnx_path(mid, "fp16")))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3072 << 20)
    if hw == "same_cc":  # portable across GPUs with the same compute capability, version-compatible
        cfg.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
        cfg.set_flag(trt.BuilderFlag.EXCLUDE_LEAN_RUNTIME)
        cfg.hardware_compatibility_level = trt.HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY
    mn, op, mx = MODELS[mid]["profile"]
    prof = builder.create_optimization_profile()
    for i in range(net.num_inputs):
        t = net.get_input(i)
        if t.name == "timestep":
            prof.set_shape(t.name, [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1])
        else:
            prof.set_shape(t.name, [1, 3, *mn], [1, 3, *op], [1, 3, *mx])
    cfg.add_optimization_profile(prof)
    blob = builder.build_serialized_network(net, cfg)
    path = Path(tempfile.gettempdir()) / f"{mid}.{hw}.engine"
    path.write_bytes(bytes(blob))
    cu = Cuda()
    eng = Engine(trt, cu, path)
    _, ms = eng.run(random_feeds(mid, *op), iters, warmup=10)
    print(f"{hw} engine {path.stat().st_size/1e6:.2f} MB, full runtime, {op}: ms={ms:.3f}")


def worker_lean(mid, iters, arch="ampere+"):
    import tensorrt_lean as trt
    from trt_runner import Cuda, Engine, random_feeds
    cu = Cuda()
    path = ENGINES / f"{mid}.{arch}.engine"
    eng = Engine(trt, cu, path)
    op = MODELS[mid]["profile"][1]
    _, ms = eng.run(random_feeds(mid, *op), iters, warmup=10)
    print(f"{arch} engine {path.stat().st_size/1e6:.2f} MB, lean runtime, {op}: ms={ms:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", default=None)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--worker", choices=["native", "same_cc", "lean"])
    ap.add_argument("--id")
    ap.add_argument("--arch", default="ampere+", help="engine arch file used by the lean worker")
    a = ap.parse_args()
    if a.worker:
        (worker_lean(a.id, a.iters, a.arch) if a.worker == "lean" else worker_native(a.id, a.iters, "none" if a.worker == "native" else "same_cc"))
        return
    cc = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True).splitlines()[0]
    sm = "sm" + cc.strip().replace(".", "")
    for mid in a.ids or ["span_x2", "rife_v4"]:
        ms = {}
        runs = [("native", "native", "ampere+"), ("same_cc", "same_cc", "ampere+"), ("lean", "lean", "ampere+")]
        if (ENGINES / f"{mid}.{sm}.engine").exists():
            runs.append((sm, "lean", sm))
        for key, w, arch in runs:
            out = subprocess.check_output([sys.executable, __file__, "--worker", w, "--id", mid, "--arch", arch,
                                           "--iters", str(a.iters)], text=True)
            print(f"[{mid}] {out.strip().splitlines()[-1]}")
            ms[key] = float(out.strip().splitlines()[-1].rsplit("ms=", 1)[1])
        labels = {"lean": "shipped ampere+ (AMPERE_PLUS, lean runtime)",
                  "same_cc": "SAME_COMPUTE_CAPABILITY built in-process (full runtime)",
                  sm: f"shipped {sm} (SAME_COMPUTE_CAPABILITY, lean runtime)"}
        for k, label in labels.items():
            if k not in ms:
                continue
            ratio = ms[k] / ms["native"]
            print(f"[{mid}] {label} / native time ratio = {ratio:.3f} -> {'OK (<= 1.15)' if ratio <= 1.15 else 'MORE THAN 15% SLOWER'}")
        if sm in ms:
            print(f"[{mid}] {sm} vs ampere+ (both lean): {ms[sm] / ms['lean']:.3f}x the time")

if __name__ == "__main__":
    main()
