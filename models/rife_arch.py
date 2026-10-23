"""RIFE 4.x IFNet (Practical-RIFE / vs-rife architecture), inlined for ONNX export.

Derived from hzwer/Practical-RIFE (MIT, Copyright (c) 2021 hzwer) and HolyWu/vs-rife
(MIT, Copyright (c) 2021 HolyWu). Changes: one parametrised class covering the 4.22.lite
and 4.25/4.26 layouts, warp grid computed from the dynamic input shape (so the traced
graph has dynamic H/W), timestep given as a [1,1,1,1] tensor, ensemble / scale fixed.

`RifeInterp.forward(img0, img1, timestep) -> output`, all [1,3,H,W] (timestep [1,1,1,1]),
H and W must be multiples of 64 (caller pads).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# version -> (head_ch, head_out, [(in_planes, c)], scale_list, lastconv_out_ch)
CONFIGS = {
    "4.26": dict(
        head=(16, 4),
        blocks=[(7 + 8, 192), (8 + 4 + 8 + 8, 128), (8 + 4 + 8 + 8, 96), (8 + 4 + 8 + 8, 64), (8 + 4 + 8 + 8, 32)],
        scales=[16, 8, 4, 2, 1],
        out_ch=13,
    ),
    "4.22.lite": dict(
        head=(16, 4),
        blocks=[(7 + 8, 192), (8 + 4 + 8 + 8, 128), (8 + 4 + 8 + 8, 64), (8 + 4 + 8 + 8, 32)],
        scales=[8, 4, 2, 1],
        out_ch=13,
    ),
}


def conv(i, o, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, p, bias=True), nn.LeakyReLU(0.2, True))


class Head(nn.Module):
    def __init__(self, ch, out):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, ch, 3, 2, 1)
        self.cnn1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.cnn2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(ch, out, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = x.clamp(0.0, 1.0)
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c, out_ch):
        super().__init__()
        self.conv0 = nn.Sequential(conv(in_planes, c // 2, 3, 2, 1), conv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * out_ch, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            if scale != 1:
                flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear")
            flow = flow / scale
            x = torch.cat((x, flow), 1)
        feat = self.convblock(self.conv0(x))
        tmp = self.lastconv(feat)
        if scale != 1:
            tmp = F.interpolate(tmp, scale_factor=float(scale), mode="bilinear")
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


def warp(x, flow, grid, div_w, div_h):
    """Backward warp in fp32 (fp16 grid coordinates are too coarse at 1080p)."""
    dt = x.dtype
    x = x.float()
    flow = flow.float()
    flow = torch.cat([flow[:, 0:1] / div_w, flow[:, 1:2] / div_h], 1)
    g = (grid + flow).permute(0, 2, 3, 1)
    out = F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=True)
    return out.to(dt)


class IFNet(nn.Module):
    def __init__(self, version):
        super().__init__()
        cfg = CONFIGS[version]
        self.version = version
        self.blocks = nn.ModuleList([IFBlock(i, c, cfg["out_ch"]) for i, c in cfg["blocks"]])
        self.encode = Head(*cfg["head"])
        self.scales = cfg["scales"]
        # parameter names must match the checkpoint: block0..blockN
        for i, b in enumerate(self.blocks):
            self.add_module(f"block{i}", b)
        del self.blocks

    def forward(self, img0, img1, timestep):
        n = len(self.scales)
        dt = img0.dtype
        h, w = img0.shape[2], img0.shape[3]
        # base sampling grid in [-1,1] (fp32), computed from the dynamic shape
        xs = torch.arange(w, dtype=torch.float32) * (2.0 / (w - 1)) - 1.0
        ys = torch.arange(h, dtype=torch.float32) * (2.0 / (h - 1)) - 1.0
        grid = torch.stack([xs.view(1, 1, 1, w).expand(1, 1, h, w), ys.view(1, 1, h, 1).expand(1, 1, h, w)], 1)
        grid = grid.reshape(1, 2, h, w)
        div_w = (w - 1.0) / 2.0
        div_h = (h - 1.0) / 2.0
        t = timestep.expand(1, 1, h, w)

        img0 = img0.clamp(0.0, 1.0)
        img1 = img1.clamp(0.0, 1.0)
        f0 = self.encode(img0)
        f1 = self.encode(img1)
        w0, w1 = img0, img1
        flow = mask = feat = None
        for i in range(n):
            blk = getattr(self, f"block{i}")
            if flow is None:
                flow, mask, feat = blk(torch.cat((img0, img1, f0, f1, t), 1), None, self.scales[i])
            else:
                wf0 = warp(f0, flow[:, :2], grid, div_w, div_h)
                wf1 = warp(f1, flow[:, 2:4], grid, div_w, div_h)
                fd, m0, feat = blk(torch.cat((w0, w1, wf0, wf1, t, mask, feat), 1), flow, self.scales[i])
                mask = m0
                flow = flow + fd
            w0 = warp(img0, flow[:, :2], grid, div_w, div_h)
            w1 = warp(img1, flow[:, 2:4], grid, div_w, div_h)
        mask = torch.sigmoid(mask)
        return w0 * mask + w1 * (1 - mask)


def load_rife(path, version):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items() if "attn_mask" not in k and "HW" not in k}
    m = IFNet(version)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if not k.startswith(("teacher.", "caltime."))]  # training-only
    if missing or unexpected:
        raise RuntimeError(f"rife {version}: missing={missing[:5]} unexpected={unexpected[:5]}")
    return m.eval()
