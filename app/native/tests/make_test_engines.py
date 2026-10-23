"""Builds tiny TensorRT test engines for fb_native's tests.

Run with the project venv:  .venv/Scripts/python.exe make_test_engines.py <out_dir>

Engines (all built like the real ones: VERSION_COMPATIBLE + EXCLUDE_LEAN_RUNTIME + hw compat AMPERE_PLUS,
one dynamic-H/W optimization profile, batch 1 NCHW RGB):
  up2_fp16.engine / up2_fp32.engine : input [1,3,H,W]  -> output [1,3,2H,2W]   (nearest 2x resize)
  interp_fp16.engine                : img0,img1 [1,3,H,W], timestep [1,1,1,1] -> output = img0 + (img1-img0)*t
"""
import os
import sys

import tensorrt as trt


def build(out_path, make_network, profiles):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    make_network(network)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
    config.set_flag(trt.BuilderFlag.EXCLUDE_LEAN_RUNTIME)
    config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS
    prof = builder.create_optimization_profile()
    for name, (mn, op, mx) in profiles.items():
        prof.set_shape(name, mn, op, mx)
    config.add_optimization_profile(prof)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise SystemExit("engine build failed: " + out_path)
    with open(out_path, "wb") as f:
        f.write(bytes(blob))
    print("wrote", out_path, blob.nbytes, "bytes")


def up2(dtype):
    def net(n):
        x = n.add_input("input", dtype, (1, 3, -1, -1))
        r = n.add_resize(x)
        r.resize_mode = trt.InterpolationMode.NEAREST
        r.scales = [1.0, 1.0, 2.0, 2.0]
        o = r.get_output(0)
        o.name = "output"
        n.mark_output(o)

    return net


def interp(n):
    dt = trt.float16
    a = n.add_input("img0", dt, (1, 3, -1, -1))
    b = n.add_input("img1", dt, (1, 3, -1, -1))
    t = n.add_input("timestep", dt, (1, 1, 1, 1))
    d = n.add_elementwise(b, a, trt.ElementWiseOperation.SUB).get_output(0)
    m = n.add_elementwise(d, t, trt.ElementWiseOperation.PROD).get_output(0)
    o = n.add_elementwise(a, m, trt.ElementWiseOperation.SUM).get_output(0)
    o.name = "output"
    n.mark_output(o)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    hw = lambda lo, op, hi: ((1, 3) + lo, (1, 3) + op, (1, 3) + hi)
    build(os.path.join(out, "up2_fp16.engine"), up2(trt.float16),
          {"input": hw((16, 16), (128, 128), (512, 512))})
    build(os.path.join(out, "up2_fp32.engine"), up2(trt.float32),
          {"input": hw((16, 16), (128, 128), (512, 512))})
    build(os.path.join(out, "interp_fp16.engine"), interp, {
        "img0": hw((64, 64), (128, 128), (256, 256)),
        "img1": hw((64, 64), (128, 128), (256, 256)),
        "timestep": ((1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1)),
    })


if __name__ == "__main__":
    main()
