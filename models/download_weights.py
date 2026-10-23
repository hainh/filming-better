"""Download pretrained weights into models/weights/ (gitignored).

Usage: python models/download_weights.py [id ...]     (default: all required ids)
       python models/download_weights.py --list

Every file is verified against the pinned sha256 below. If a sha256 is empty the
file is downloaded and the actual hash is printed (paste it back into this table).
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights"

# id -> metadata. `file` is the local file name under models/weights/.
SOURCES = {
    "span_x2": dict(
        kind="upscale", scale=2,
        url="https://github.com/Phhofm/models/releases/download/2xNomosUni_span_multijpg/2xNomosUni_span_multijpg.pth",
        file="2xNomosUni_span_multijpg.pth",
        sha256="1a981d0a11e2fcbb6cda7ac635f31b97d9903492ba7e4dda09b8be0b274f1ac5",
        license="CC-BY-4.0 (attribution: Helaman / Phhofm, 2xNomosUni_span_multijpg; SPAN arch Apache-2.0 hongyuanyu/SPAN)",
        source="https://github.com/Phhofm/models/releases/tag/2xNomosUni_span_multijpg",
        description="SPAN 2x (2xNomosUni_span_multijpg), fast default upscaler",
        required=True,
    ),
    "animevideov3_x4": dict(
        kind="upscale", scale=4,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        file="realesr-animevideov3.pth",
        sha256="b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d",
        license="BSD-3-Clause (xinntao/Real-ESRGAN)",
        source="https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0",
        description="Real-ESRGAN realesr-animevideov3 (SRVGGNetCompact) 4x",
        required=True,
    ),
    "rife_v4": dict(
        kind="interp", scale=1,
        url="https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.26.pkl",
        file="flownet_v4.26.pkl",
        sha256="45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f",
        license="MIT (hzwer/Practical-RIFE, HolyWu/vs-rife)",
        source="https://github.com/HolyWu/vs-rife/releases/tag/model",
        description="RIFE 4.26 frame interpolation",
        required=True,
    ),
    "rife_v4_lite": dict(
        kind="interp", scale=1,
        url="https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.22.lite.pkl",
        file="flownet_v4.22.lite.pkl",
        sha256="8f6fb9105ba9e946762ee7190acbca3ca1cf14193eb81ca0955d492fb8558692",
        license="MIT (hzwer/Practical-RIFE, HolyWu/vs-rife)",
        source="https://github.com/HolyWu/vs-rife/releases/tag/model",
        description="RIFE 4.22 lite frame interpolation (lighter, faster)",
        required=False,
    ),
    "x4plus": dict(
        kind="upscale", scale=4,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        file="RealESRGAN_x4plus.pth",
        sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        license="BSD-3-Clause (xinntao/Real-ESRGAN)",
        source="https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0",
        description="Real-ESRGAN x4plus (RRDBNet, large and slow)",
        required=False,
    ),
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(mid: str, info: dict, force: bool = False) -> Path:
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    dst = WEIGHTS / info["file"]
    if dst.exists() and not force:
        got = sha256_file(dst)
        if not info["sha256"] or got == info["sha256"]:
            print(f"[{mid}] present {dst.name} sha256={got}")
            return dst
        print(f"[{mid}] hash mismatch, re-downloading")
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[{mid}] GET {info['url']}")
    req = urllib.request.Request(info["url"], headers={"User-Agent": "filming-better-models/1"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    got = sha256_file(tmp)
    if info["sha256"] and got != info["sha256"]:
        tmp.unlink()
        raise SystemExit(f"[{mid}] sha256 mismatch: expected {info['sha256']} got {got}")
    tmp.replace(dst)
    print(f"[{mid}] ok {dst.name} {dst.stat().st_size} bytes sha256={got}")
    return dst


def weights_path(mid: str) -> Path:
    return WEIGHTS / SOURCES[mid]["file"]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--list" in sys.argv:
        for k, v in SOURCES.items():
            print(f"{k:16s} {v['license']:60s} {v['url']}")
        return
    ids = args or [k for k, v in SOURCES.items() if v["required"]]
    if "--all" in sys.argv:
        ids = list(SOURCES)
    for mid in ids:
        if mid not in SOURCES:
            raise SystemExit(f"unknown id {mid}; known: {list(SOURCES)}")
        download(mid, SOURCES[mid], force="--force" in sys.argv)


if __name__ == "__main__":
    main()
