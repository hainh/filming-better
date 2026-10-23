#!/usr/bin/env python3
"""Compare the ffprobe properties of a filming-better OUTPUT with its INPUT.

usage: check_output.py INPUT OUTPUT [--scale N] [--interp N] [--codec h264|hevc|av1]
                       [--ffprobe PATH] [--vfr]

Checks (each prints PASS/FAIL, exit status 1 if any FAIL):
  * display resolution  == input display resolution * scale     (rotation metadata is honoured:
                           either the display matrix is kept or the rotation was baked in)
  * frame rate          == input fps * interp (+-0.3 %)  /  for VFR input: output must be CFR
  * duration            video packet span within 1 INPUT frame (+ half an output frame)
  * A/V start offset    unchanged within 1 input frame
  * audio streams       same count, same codec, same sample rate (stream copy)
  * subtitle streams    same count
  * video codec         == --codec (if given)
  * color tags          colorspace / primaries / transfer / range preserved when the input has them
  * 10-bit input        10-bit output (warning only)
Only the Python standard library is used.
"""
import argparse
import json
import subprocess
import sys


def probe(ffprobe, path):
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise SystemExit("ffprobe failed on %s:\n%s" % (path, out.stderr))
    return json.loads(out.stdout)


def video_span(ffprobe, path):
    """(first_pts, end) of the first video stream from its packets (container independent)."""
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,duration_time",
         "-of", "csv=p=0", path], capture_output=True, text=True, encoding="utf-8", errors="replace")
    first, end = None, 0.0
    for line in out.stdout.splitlines():
        parts = line.strip().split(",")
        try:
            pts = float(parts[0])
        except (ValueError, IndexError):
            continue
        dur = 0.0
        try:
            dur = float(parts[1])
        except (ValueError, IndexError):
            pass
        first = pts if first is None else min(first, pts)
        end = max(end, pts + dur)
    return (first or 0.0), end


def frac(x):
    try:
        n, d = x.split("/")
        d = float(d)
        return float(n) / d if d else 0.0
    except Exception:
        try:
            return float(x)
        except Exception:
            return 0.0


def rotation(v):
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            return int(round(float(sd["rotation"])))
    r = (v.get("tags") or {}).get("rotate")
    return int(r) if r else 0


def display_dims(v):
    w, h = int(v["width"]), int(v["height"])
    if abs(rotation(v)) % 180 == 90:
        return h, w
    return w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--interp", type=int, default=1)
    ap.add_argument("--codec", default=None)
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--vfr", action="store_true", help="input is variable frame rate")
    a = ap.parse_args()

    i, o = probe(a.ffprobe, a.input), probe(a.ffprobe, a.output)
    iv = [s for s in i["streams"] if s["codec_type"] == "video" and s.get("disposition", {}).get("attached_pic", 0) == 0][0]
    ovs = [s for s in o["streams"] if s["codec_type"] == "video" and s.get("disposition", {}).get("attached_pic", 0) == 0]
    results = []

    def check(name, ok, detail="", warn=False):
        results.append((name, ok, detail, warn))

    if not ovs:
        check("video stream present", False, "output has no video stream")
        return report(results)
    ov = ovs[0]

    # resolution
    iw, ih = display_dims(iv)
    ow, oh = display_dims(ov)
    exp_w, exp_h = iw * a.scale, ih * a.scale
    if (ow, oh) == (exp_w, exp_h):
        check("resolution", True)
    else:
        # odd input sizes: encoders need even dimensions, rounding the INPUT up to even first is acceptable (warning)
        alt = (((iw + 1) // 2) * 2 * a.scale, ((ih + 1) // 2) * 2 * a.scale)
        ok_alt = (iw % 2 or ih % 2) and (ow, oh) == alt
        check("resolution" + (" (odd input rounded up to even first)" if ok_alt else ""), False,
              "in %dx%d (rot %d) -> out %dx%d (rot %d), expected %dx%d" % (iw, ih, rotation(iv), ow, oh, rotation(ov), exp_w, exp_h),
              warn=bool(ok_alt))

    # frame rate
    ifps = frac(iv.get("avg_frame_rate", "0/1")) or frac(iv.get("r_frame_rate", "0/1"))
    ofps = frac(ov.get("avg_frame_rate", "0/1")) or frac(ov.get("r_frame_rate", "0/1"))
    orfps = frac(ov.get("r_frame_rate", "0/1"))
    if a.vfr:
        cfr = ofps > 0 and abs(orfps - ofps) / ofps < 0.005
        check("fps (VFR input -> CFR output)", cfr, "out avg %.3f r %.3f" % (ofps, orfps))
    else:
        exp = ifps * a.interp
        check("fps", exp > 0 and abs(ofps - exp) / exp < 0.003, "in %.3f x %d -> out %.3f (expected %.3f)" % (ifps, a.interp, ofps, exp))

    # duration: span of the video packets (container independent; ignores TS/MKV start offsets and subtitle tails)
    i0, i1 = video_span(a.ffprobe, a.input)
    o0, o1 = video_span(a.ffprobe, a.output)
    idur, odur = i1 - i0, o1 - o0
    ofps_ = ofps if ofps > 0 else ifps
    tol = (1.0 / ifps if ifps > 0 else 0.05) + (1.0 / ofps_ if ofps_ > 0 else 0.0) * 0.5 + 0.002
    check("duration within ~1 input frame", abs(odur - idur) <= tol,
          "in %.3fs out %.3fs diff %.3fs (tol %.3fs)" % (idur, odur, odur - idur, tol))

    # A/V sync: the audio-vs-video start offset must be unchanged (<= 1 input frame)
    def first_audio_start(d):
        for st in d["streams"]:
            if st["codec_type"] == "audio" and st.get("start_time") not in (None, "N/A"):
                return float(st["start_time"])
        return None
    ia0, oa0 = first_audio_start(i), first_audio_start(o)
    if ia0 is not None and oa0 is not None:
        off_in, off_out = i0 - ia0, o0 - oa0
        # MP4 hides the AAC encoder delay (priming, one 1024-sample frame) in an edit list, while
        # MKV/TS expose it as a start offset; plain `ffmpeg -c copy` shows the same shift, so allow it.
        priming = 0.0
        for st in i["streams"]:
            if st["codec_type"] == "audio" and st.get("sample_rate"):
                priming = 1024.0 / float(st["sample_rate"])
                break
        tol = (1.0 / ifps if ifps > 0 else 0.05) + priming + 0.002
        check("A/V start offset unchanged", abs(off_out - off_in) <= tol,
              "in %+.3fs out %+.3fs (tol %.3fs)" % (off_in, off_out, tol))

    # audio / subtitles
    ia = [s for s in i["streams"] if s["codec_type"] == "audio"]
    oa = [s for s in o["streams"] if s["codec_type"] == "audio"]
    check("audio stream count", len(ia) == len(oa), "in %d out %d" % (len(ia), len(oa)))
    if len(ia) == len(oa):
        for k, (x, y) in enumerate(zip(ia, oa)):
            check("audio[%d] codec/rate copied" % k,
                  x["codec_name"] == y["codec_name"] and x.get("sample_rate") == y.get("sample_rate"),
                  "%s/%s -> %s/%s" % (x["codec_name"], x.get("sample_rate"), y["codec_name"], y.get("sample_rate")))
    isub = [s for s in i["streams"] if s["codec_type"] == "subtitle"]
    osub = [s for s in o["streams"] if s["codec_type"] == "subtitle"]
    check("subtitle stream count", len(isub) == len(osub), "in %d out %d" % (len(isub), len(osub)))

    # codec
    if a.codec:
        want = {"h264": "h264", "hevc": "hevc", "h265": "hevc", "av1": "av1"}.get(a.codec.lower(), a.codec.lower())
        check("video codec", ov["codec_name"] == want, "out %s expected %s" % (ov["codec_name"], want))

    # color tags
    for key in ("color_space", "color_primaries", "color_transfer", "color_range"):
        iv_, ov_ = iv.get(key), ov.get(key)
        if iv_ and iv_ not in ("unknown", "unspecified"):
            check("color tag %s" % key, iv_ == ov_, "in %s out %s" % (iv_, ov_))

    # bit depth
    if "10" in (iv.get("pix_fmt") or ""):
        check("10-bit preserved", "10" in (ov.get("pix_fmt") or ""), "in %s out %s" % (iv.get("pix_fmt"), ov.get("pix_fmt")), warn=True)

    return report(results)


def report(results):
    bad = 0
    for name, ok, detail, warn in results:
        tag = "PASS" if ok else ("WARN" if warn else "FAIL")
        if not ok and not warn:
            bad += 1
        print("    [%s] %s%s" % (tag, name, (" - " + detail) if (detail and not ok) or tag != "PASS" else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
