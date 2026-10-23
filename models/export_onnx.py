"""Export the pretrained models to ONNX (dynamic H/W), then simplify with onnxslim.

For every model id two files are written to models/onnx/:
  <id>.fp32.onnx   FP32 reference (validated against PyTorch by validate_onnx.py)
  <id>.fp16.onnx   FP16 graph with fp16 I/O. TensorRT 11 has no BuilderFlag.FP16 (networks are
                   strongly typed), so the precision of an engine is the precision of this ONNX.
                   RIFE keeps the warp/grid_sample path in fp32 inside the graph (Cast nodes).

Conventions (docs/contracts.md section 1):
  upscale: input `input` [1,3,H,W] -> `output` [1,3,H*s,W*s]
  interp : `img0`,`img1` [1,3,H,W], `timestep` [1,1,1,1] -> `output` [1,3,H,W]  (H,W % 64 == 0)

Usage: python models/export_onnx.py [id ...] [--opset 17]
"""
import argparse
import sys
import warnings

import torch
import torch.nn as nn

import common
from common import MODELS, DEFAULT_IDS, ONNX, onnx_path
import download_weights as dw

warnings.filterwarnings("ignore")
OPSET = 17


class UpscaleWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input):
        return self.m(input)


def load_upscaler(mid):
    import spandrel
    desc = spandrel.ModelLoader().load_from_file(str(dw.weights_path(mid)))
    assert desc.input_channels == 3 and desc.output_channels == 3
    assert desc.scale == MODELS[mid]["scale"], (desc.scale, MODELS[mid]["scale"])
    return UpscaleWrap(desc.model.eval())


def load_rife_wrap(mid):
    import rife_arch
    return rife_arch.load_rife(dw.weights_path(mid), MODELS[mid]["rife_version"]).eval()


def load_torch_model(mid):
    """Return the FP32 eval-mode torch module with the contract forward signature."""
    return load_upscaler(mid) if MODELS[mid]["kind"] == "upscale" else load_rife_wrap(mid)


def dummy_inputs(mid, dtype, h=None, w=None):
    kind = MODELS[mid]["kind"]
    if kind == "upscale":
        h, w = h or 72, w or 88
        return (torch.rand(1, 3, h, w, dtype=dtype),)
    h, w = h or 128, w or 192
    return (torch.rand(1, 3, h, w, dtype=dtype), torch.rand(1, 3, h, w, dtype=dtype),
            torch.full((1, 1, 1, 1), 0.5, dtype=dtype))


def io_spec(mid):
    if MODELS[mid]["kind"] == "upscale":
        return ["input"], ["output"], {"input": {2: "height", 3: "width"}, "output": {2: "oh", 3: "ow"}}
    return (["img0", "img1", "timestep"], ["output"],
            {"img0": {2: "height", 3: "width"}, "img1": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}})


def export_one(mid, prec, opset=OPSET):
    import onnx
    from onnxslim import slim
    model = load_torch_model(mid)
    dtype = torch.float32
    if prec == "fp16":
        model = model.half()
        dtype = torch.float16
    args = dummy_inputs(mid, dtype)
    in_names, out_names, dyn = io_spec(mid)
    out = onnx_path(mid, prec)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, args, str(out), input_names=in_names, output_names=out_names,
                          dynamic_axes=dyn, opset_version=opset, do_constant_folding=True, dynamo=False)
    raw = out.stat().st_size
    m = onnx.load(str(out))
    m = slim(m)
    onnx.save(m, str(out))
    onnx.checker.check_model(str(out))
    print(f"[{mid}] {prec}: {out.name} raw={raw/1e6:.2f}MB slim={out.stat().st_size/1e6:.2f}MB")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--opset", type=int, default=OPSET)
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16"])
    a = ap.parse_args()
    ids = a.ids or DEFAULT_IDS
    for mid in ids:
        dw.download(mid, dw.SOURCES[mid])
        for prec in a.precisions:
            export_one(mid, prec, a.opset)


if __name__ == "__main__":
    main()
