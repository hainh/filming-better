"""Build TensorRT engines (version-compatible, lean-runtime-only, ampere+) from the FP16 ONNX files.

  python models/build_engines.py [id ...] [--arch ampere+|sm75|sm86|smXY] [--workspace-mb 3072]

Engine flags (docs/contracts.md section 1):
  VERSION_COMPATIBLE + EXCLUDE_LEAN_RUNTIME, hardware_compatibility_level = AMPERE_PLUS,
  one optimization profile with dynamic H/W, batch 1.
TensorRT 11 has no BuilderFlag.FP16: the network is strongly typed, so the engine runs in
the precision of the ONNX (models/onnx/<id>.fp16.onnx) and its I/O is FP16 (io_dtype "fp16").

Writes models/engines/<id>.<arch>.engine and merges the entry into models/engines/manifest.json
(copied to models/manifest.json). `fps_1080p` is filled in later by bench_engines.py.

sm75 (Turing) engines can only be built on a Turing GPU: hardware_compatibility_level
SAME_COMPUTE_CAPABILITY (engine bound to cc 7.5). `--arch sm75` is implemented but skips itself on any
other GPU. Any `--arch sm<major><minor>` (sm80, sm86, sm89, sm120 ...) works the same way for the GPU's own
cc and is a contract arch tag (docs/contracts.md section 2): the app prefers the exact-cc engine and falls back to
the portable `ampere+` one. Such engines run nearly as fast as a fully native engine while AMPERE_PLUS engines
are up to 1.5x slower (see README). Both kinds are kept in the manifest.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time

import common
from common import DEFAULT_IDS, ENGINES, HERE, MODELS, onnx_path
import download_weights as dw

common.setup_dll_dirs()
import tensorrt as trt  # noqa: E402

BASE_URL = "https://github.com/OWNER/filming-better/releases/download/engines-v1/"
MANIFEST = ENGINES / "manifest.json"


def gpu_name():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], text=True)
        name, cc = [s.strip() for s in out.strip().splitlines()[0].split(",")]
        return name, cc
    except Exception:
        return "unknown", "0.0"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def shapes_for(mid):
    mn, op, mx = MODELS[mid]["profile"]
    return [[1, 3, *s] for s in (mn, op, mx)]


class Logger(trt.ILogger):
    def __init__(self, verbose=False):
        trt.ILogger.__init__(self)
        self.verbose = verbose
        self.bad = False  # an ERROR (e.g. out of VRAM while timing tactics) was reported

    def log(self, severity, msg):
        if severity <= trt.ILogger.ERROR:
            self.bad = True
        if severity <= trt.ILogger.WARNING or self.verbose:
            print(f"  [TRT {severity.name}] {msg}")


def _attempt(mid, arch, ws_mb, verbose):
    """One build attempt; returns (blob or None, had_errors, seconds)."""
    onnx_file = onnx_path(mid, "fp16")
    logger = Logger(verbose)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_file)):
        for i in range(parser.num_errors):
            print("  ONNX parse error:", parser.get_error(i))
        raise SystemExit(f"parse failed for {mid}")
    mn, op, mx = shapes_for(mid)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
    config.set_flag(trt.BuilderFlag.EXCLUDE_LEAN_RUNTIME)
    config.hardware_compatibility_level = (trt.HardwareCompatibilityLevel.AMPERE_PLUS if arch == "ampere+"
                                           else trt.HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_mb << 20)
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        t = network.get_input(i)
        if t.name == "timestep":
            profile.set_shape(t.name, [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1])
        else:
            profile.set_shape(t.name, mn, op, mx)
    config.add_optimization_profile(profile)
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    return blob, logger.bad, time.time() - t0


def build_one(mid, arch, workspace_mb, verbose=False, attempts=6):
    if not onnx_path(mid, "fp16").exists():
        raise SystemExit(f"{onnx_path(mid, 'fp16')} missing; run export_onnx.py first")
    mn, op, mx = shapes_for(mid)
    ws = workspace_mb
    blob = None
    for k in range(attempts):
        print(f"[{mid}] building {arch} engine (attempt {k + 1}): min={mn} opt={op} max={mx} workspace={ws}MB", flush=True)
        blob, bad, dt = _attempt(mid, arch, ws, verbose)
        if blob is not None and not bad:
            break
        # The GPU is shared: an out-of-memory while timing tactics silently drops tactics ("Skipping this
        # backend strategy") and yields a slower engine, so retry in that case too.
        print(f"[{mid}] build {'failed' if blob is None else 'reported errors (tactics skipped, engine may be slow)'}; "
              f"retrying with a smaller workspace after a pause", flush=True)
        blob = None
        ws = max(512, ws // 2)
        time.sleep(20)
    if blob is None:
        raise SystemExit(f"[{mid}] engine build failed")
    ENGINES.mkdir(parents=True, exist_ok=True)
    path = ENGINES / f"{mid}.{arch}.engine"
    with open(path, "wb") as f:
        f.write(blob)
    print(f"[{mid}] wrote {path.name} {path.stat().st_size/1e6:.2f} MB in {dt:.0f}s")
    return path, mn, op, mx


def update_manifest(mid, arch, path, mn, op, mx, gpu):
    src = dw.SOURCES[mid]
    man = {"schema": 1, "trt_version": trt.__version__.rsplit(".", 1)[0], "cuda_major": 12,
           "base_url": BASE_URL, "models": []}
    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text())
    man["trt_version"] = trt.__version__.rsplit(".", 1)[0]
    man["base_url"] = man.get("base_url") or BASE_URL
    info = MODELS[mid]
    entry = next((m for m in man["models"] if m["id"] == mid), None)
    if entry is None:
        entry = {"id": mid, "engines": []}
        man["models"].append(entry)
    entry.update(kind=info["kind"], scale=info["scale"], io_dtype="fp16", pad_multiple=info["pad_multiple"],
                 description=src["description"], license=src["license"], source=src["source"])
    eng = {"arch": arch, "file": path.name, "sha256": sha256_file(path), "size": path.stat().st_size,
           "min_shape": mn, "opt_shape": op, "max_shape": mx, "fps_1080p": None, "build_gpu": gpu}
    # portable ampere+ first, then the per-cc engines (sm75, sm86 ...) in name order
    engines = [e for e in entry["engines"] if e["arch"] != arch] + [eng]
    entry["engines"] = sorted(engines, key=lambda e: (e["arch"] != "ampere+", e["arch"]))
    # keep key order of the contract: model keys first
    order = ["id", "kind", "scale", "io_dtype", "pad_multiple", "description", "license", "source", "engines"]
    man["models"] = [{k: m[k] for k in order if k in m} for m in man["models"]]
    MANIFEST.write_text(common.dump_manifest(man))
    shutil.copyfile(MANIFEST, HERE / "manifest.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--arch", default="ampere+", help="ampere+ (default) | sm75 | smXY (engine bound to that exact compute capability)")
    ap.add_argument("--workspace-mb", type=int, default=3072)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-manifest", action="store_true", help="only write the engine file, do not touch the manifest")
    a = ap.parse_args()
    name, cc = gpu_name()
    if a.arch != "ampere+":
        want = a.arch[2:] if re.fullmatch(r"sm\d{2,3}", a.arch) else None
        if want is None:
            raise SystemExit(f"bad --arch {a.arch}")
        if cc.replace(".", "") != want:
            print(f"SKIP: --arch {a.arch} must be built on a GPU with compute capability {want[0]}.{want[1]}; this machine is "
                  f"'{name}' cc {cc}. Run this script on such a card to produce <id>.{a.arch}.engine entries.")
            return
    elif float(cc) < 8.0:
        raise SystemExit(f"ampere+ engines need a GPU with cc >= 8.0 (found {name} cc {cc})")
    for mid in (a.ids or DEFAULT_IDS):
        path, mn, op, mx = build_one(mid, a.arch, a.workspace_mb, a.verbose)
        if a.no_manifest:
            print(f"[{mid}] engine written, manifest NOT updated (--no-manifest)")
        else:
            update_manifest(mid, a.arch, path, mn, op, mx, name)


if __name__ == "__main__":
    main()
