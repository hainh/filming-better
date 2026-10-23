// YUV (NV12/P010) <-> planar RGB kernels vs. double-precision CPU references.
#include <algorithm>

#include "test_common.h"

using namespace t;

namespace {

struct Coef {
    double kr, kb, yoff, yrng, coff, crng, maxc;
};

Coef coef_for(int fmt, int matrix, int range) {
    Coef c{};
    if (matrix == FB_MATRIX_BT601) { c.kr = 0.299; c.kb = 0.114; }
    else if (matrix == FB_MATRIX_BT709) { c.kr = 0.2126; c.kb = 0.0722; }
    else { c.kr = 0.2627; c.kb = 0.0593; }
    double s = fmt == FB_FMT_P010 ? 4.0 : 1.0;
    c.maxc = fmt == FB_FMT_P010 ? 1023.0 : 255.0;
    c.coff = 128.0 * s;
    if (range == FB_RANGE_FULL) { c.yoff = 0; c.yrng = c.maxc; c.crng = c.maxc; }
    else { c.yoff = 16 * s; c.yrng = 219 * s; c.crng = 224 * s; }
    return c;
}

struct Yuv {
    int w = 0, h = 0, fmt = FB_FMT_NV12;
    size_t lp = 0, cp = 0;
    std::vector<uint8_t> luma, chroma;
    int bps() const { return fmt == FB_FMT_P010 ? 2 : 1; }
    int cw() const { return (w + 1) / 2; }
    int ch() const { return (h + 1) / 2; }
    int ycode(int x, int y) const {
        const uint8_t* p = &luma[(size_t)y * lp + (size_t)x * bps()];
        return fmt == FB_FMT_P010 ? ((p[0] | (p[1] << 8)) >> 6) : p[0];
    }
    void ccode(int cx, int cy, int& u, int& v) const {
        const uint8_t* p = &chroma[(size_t)cy * cp + (size_t)cx * 2 * bps()];
        if (fmt == FB_FMT_P010) {
            u = (p[0] | (p[1] << 8)) >> 6;
            v = (p[2] | (p[3] << 8)) >> 6;
        } else {
            u = p[0];
            v = p[1];
        }
    }
    void set_y(int x, int y, int code) {
        uint8_t* p = &luma[(size_t)y * lp + (size_t)x * bps()];
        if (fmt == FB_FMT_P010) { p[0] = (code << 6) & 0xff; p[1] = (uint8_t)((code << 6) >> 8); }
        else p[0] = (uint8_t)code;
    }
    void set_c(int cx, int cy, int u, int v) {
        uint8_t* p = &chroma[(size_t)cy * cp + (size_t)cx * 2 * bps()];
        if (fmt == FB_FMT_P010) {
            p[0] = (u << 6) & 0xff; p[1] = (uint8_t)((u << 6) >> 8);
            p[2] = (v << 6) & 0xff; p[3] = (uint8_t)((v << 6) >> 8);
        } else { p[0] = (uint8_t)u; p[1] = (uint8_t)v; }
    }
};

constexpr uint8_t kGarbage = 0xCD;

Yuv make_yuv(int w, int h, int fmt, size_t extra_pitch) {
    Yuv f;
    f.w = w; f.h = h; f.fmt = fmt;
    f.lp = (size_t)w * f.bps() + extra_pitch;
    f.cp = (size_t)f.cw() * 2 * f.bps() + extra_pitch;
    f.luma.assign(f.lp * h, kGarbage);
    f.chroma.assign(f.cp * f.ch(), kGarbage);
    return f;
}

struct DevYuv {
    DBuf luma, chroma;
    fb_yuv_frame_t desc;
    explicit DevYuv(const Yuv& f) : luma(f.luma.size()), chroma(f.chroma.size()) {
        luma.up(f.luma.data(), f.luma.size());
        chroma.up(f.chroma.data(), f.chroma.size());
        desc.luma = luma.p; desc.luma_pitch = f.lp; desc.chroma = chroma.p; desc.chroma_pitch = f.cp;
        desc.width = f.w; desc.height = f.h; desc.format = f.fmt;
    }
    void download(Yuv& f) {
        luma.down(f.luma.data(), f.luma.size());
        chroma.down(f.chroma.data(), f.chroma.size());
    }
};

// Reference: bilinear (centre-sited) chroma upsample, matrix, range, clamp, edge-replicated padding.
Tensor ref_yuv_to_rgb(const Yuv& f, const Coef& c, int dw, int dh) {
    Tensor out(dw, dh);
    double kg = 1 - c.kr - c.kb;
    auto clamp01 = [](double v) { return std::min(1.0, std::max(0.0, v)); };
    for (int y = 0; y < dh; ++y)
        for (int x = 0; x < dw; ++x) {
            int sx = std::min(x, f.w - 1), sy = std::min(y, f.h - 1);
            double fx = (sx + 0.5) / 2 - 0.5, fy = (sy + 0.5) / 2 - 0.5;
            int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy);
            double ax = fx - x0, ay = fy - y0;
            auto cl = [](int v, int hi) { return std::min(std::max(v, 0), hi); };
            double u = 0, v = 0;
            for (int j = 0; j < 2; ++j)
                for (int i = 0; i < 2; ++i) {
                    int uu, vv;
                    f.ccode(cl(x0 + i, f.cw() - 1), cl(y0 + j, f.ch() - 1), uu, vv);
                    double wgt = (i ? ax : 1 - ax) * (j ? ay : 1 - ay);
                    u += wgt * uu;
                    v += wgt * vv;
                }
            double Y = (f.ycode(sx, sy) - c.yoff) / c.yrng;
            double Cb = (u - c.coff) / c.crng, Cr = (v - c.coff) / c.crng;
            double r = Y + 2 * (1 - c.kr) * Cr;
            double b = Y + 2 * (1 - c.kb) * Cb;
            double g = (Y - c.kr * r - c.kb * b) / kg;
            out.at(0, x, y) = (float)clamp01(r);
            out.at(1, x, y) = (float)clamp01(g);
            out.at(2, x, y) = (float)clamp01(b);
        }
    return out;
}

void fill_random_yuv(Yuv& f, Rng& rng, bool limited_only) {
    int maxc = f.fmt == FB_FMT_P010 ? 1023 : 255;
    int s = f.fmt == FB_FMT_P010 ? 4 : 1;
    for (int y = 0; y < f.h; ++y)
        for (int x = 0; x < f.w; ++x)
            f.set_y(x, y, limited_only ? rng.integer(16 * s, 235 * s) : rng.integer(0, maxc));
    for (int y = 0; y < f.ch(); ++y)
        for (int x = 0; x < f.cw(); ++x)
            f.set_c(x, y, limited_only ? rng.integer(16 * s, 240 * s) : rng.integer(0, maxc),
                    limited_only ? rng.integer(16 * s, 240 * s) : rng.integer(0, maxc));
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    CHECK(a.size() == b.size());
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

void run_yuv_to_rgb_case(int w, int h, int fmt, int matrix, int range, int dtype, int padw, int padh,
                         size_t extra_pitch, fb_stream* s) {
    Rng rng((uint32_t)(w * 7919 + h * 31 + fmt * 5 + matrix * 3 + range));
    Yuv f = make_yuv(w, h, fmt, extra_pitch);
    fill_random_yuv(f, rng, false);
    DevYuv d(f);
    int dw = w + padw, dh = h + padh;
    DBuf out((size_t)3 * dw * dh * esize(dtype));
    OK(fb_yuv_to_rgb(&d.desc, matrix, range, out.p, dtype, dw, dh, s));
    OK(fb_stream_sync(s));
    std::vector<float> got = download(out, (size_t)3 * dw * dh, dtype);
    Tensor ref = ref_yuv_to_rgb(f, coef_for(fmt, matrix, range), dw, dh);
    double tol = dtype == FB_DTYPE_FP16 ? 6e-4 : 3e-5;
    double m = max_abs_diff(got, ref.v);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "yuv_to_rgb %dx%d fmt=%d matrix=%d range=%d dtype=%d pad=%d,%d pitch+%zu: max diff %g (tol %g)",
                  w, h, fmt, matrix, range, dtype, padw, padh, extra_pitch, m, tol);
    CHECK_MSG(m <= tol, buf);
}

}  // namespace

TEST(color_yuv_to_rgb_matches_reference) {
    fb_stream* s = nullptr;
    OK(fb_stream_create(&s));
    const int sizes[][2] = {{64, 48}, {1366, 768}, {577, 1023}, {17, 9}, {1, 1}, {2, 3}};
    for (auto& sz : sizes)
        for (int fmt : {FB_FMT_NV12, FB_FMT_P010})
            for (int matrix : {FB_MATRIX_BT601, FB_MATRIX_BT709, FB_MATRIX_BT2020NC})
                for (int range : {FB_RANGE_LIMITED, FB_RANGE_FULL})
                    for (int dtype : {FB_DTYPE_FP32, FB_DTYPE_FP16}) {
                        // alternate padding / pitch variations to keep the matrix manageable
                        bool alt = ((sz[0] + matrix + range + dtype) & 1) != 0;
                        run_yuv_to_rgb_case(sz[0], sz[1], fmt, matrix, range, dtype, alt ? 5 : 0, alt ? 3 : 0,
                                            alt ? 192 : 0, alt ? s : nullptr);
                    }
    OK(fb_stream_destroy(s));
}

TEST(color_yuv_to_rgb_known_values) {
    // Mid-grey / white / black / pure red in limited-range BT.709 NV12 (flat colours -> chroma upsampling is exact).
    struct C { int y, u, v; float r, g, b; };
    const C cases[] = {
        {16, 128, 128, 0, 0, 0},
        {235, 128, 128, 1, 1, 1},
        {126, 128, 128, 0.5023f, 0.5023f, 0.5023f},  // (126-16)/219
        {63, 102, 240, 1.0f, 0.0f, 0.0f},            // BT.709 red
    };
    for (const C& c : cases) {
        Yuv f = make_yuv(4, 4, FB_FMT_NV12, 0);
        for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x) f.set_y(x, y, c.y);
        for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x) f.set_c(x, y, c.u, c.v);
        DevYuv d(f);
        DBuf out(3 * 16 * 4);
        OK(fb_yuv_to_rgb(&d.desc, FB_MATRIX_BT709, FB_RANGE_LIMITED, out.p, FB_DTYPE_FP32, 4, 4, nullptr));
        auto v = download(out, 48, FB_DTYPE_FP32);
        for (int i = 0; i < 16; ++i) {
            CHECK_MSG(std::fabs(v[i] - c.r) < 0.01, "R " + std::to_string(v[i]) + " vs " + std::to_string(c.r));
            CHECK_MSG(std::fabs(v[16 + i] - c.g) < 0.01, "G " + std::to_string(v[16 + i]) + " vs " + std::to_string(c.g));
            CHECK_MSG(std::fabs(v[32 + i] - c.b) < 0.01, "B " + std::to_string(v[32 + i]) + " vs " + std::to_string(c.b));
        }
    }
}

namespace {
void run_rgb_to_yuv_case(int sw, int sh, int w, int h, int fmt, int matrix, int range, int dtype, size_t extra_pitch,
                         fb_stream* s) {
    Rng rng((uint32_t)(sw * 131 + sh * 17 + fmt * 5 + matrix * 3 + range + dtype));
    Tensor img(sw, sh);
    for (auto& x : img.v) x = quant_half(rng.uni(-0.1f, 1.1f));  // exercises clamping; exact in fp16
    DBuf in(img.v.size() * esize(dtype));
    upload(in, img.v, dtype);
    Yuv f = make_yuv(w, h, fmt, extra_pitch);
    DevYuv d(f);
    OK(fb_rgb_to_yuv(in.p, dtype, sw, sh, &d.desc, matrix, range, s));
    OK(fb_stream_sync(s));
    Yuv g = f;
    d.download(g);

    Coef c = coef_for(fmt, matrix, range);
    double kg = 1 - c.kr - c.kb;
    int maxdiff = 0;
    auto clamp01 = [](double v) { return std::min(1.0, std::max(0.0, v)); };
    auto q = [&](double v) { return std::min(std::max(std::nearbyint(v), 0.0), c.maxc); };
    auto px = [&](int ch, int x, int y) {
        return clamp01(img.at(ch, std::min(x, w - 1), std::min(y, h - 1)));
    };
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            double Y = c.kr * px(0, x, y) + kg * px(1, x, y) + c.kb * px(2, x, y);
            maxdiff = std::max(maxdiff, (int)std::abs((int)q(Y * c.yrng + c.yoff) - g.ycode(x, y)));
        }
    for (int cy = 0; cy < g.ch(); ++cy)
        for (int cx = 0; cx < g.cw(); ++cx) {
            double r = 0, gg = 0, b = 0;
            for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i) {
                r += px(0, 2 * cx + i, 2 * cy + j); gg += px(1, 2 * cx + i, 2 * cy + j); b += px(2, 2 * cx + i, 2 * cy + j);
            }
            r /= 4; gg /= 4; b /= 4;
            double Y = c.kr * r + kg * gg + c.kb * b;
            double Cb = (b - Y) / (2 * (1 - c.kb)), Cr = (r - Y) / (2 * (1 - c.kr));
            int u, v;
            g.ccode(cx, cy, u, v);
            maxdiff = std::max(maxdiff, (int)std::abs((int)q(Cb * c.crng + c.coff) - u));
            maxdiff = std::max(maxdiff, (int)std::abs((int)q(Cr * c.crng + c.coff) - v));
        }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "rgb_to_yuv src %dx%d dst %dx%d fmt=%d matrix=%d range=%d dtype=%d: max code diff %d",
                  sw, sh, w, h, fmt, matrix, range, dtype, maxdiff);
    CHECK_MSG(maxdiff <= 1, buf);

    // pitch padding must not be touched
    for (int y = 0; y < h; ++y)
        for (size_t x = (size_t)w * g.bps(); x < g.lp; ++x) CHECK_MSG(g.luma[(size_t)y * g.lp + x] == kGarbage, "luma pitch padding overwritten");
    for (int y = 0; y < g.ch(); ++y)
        for (size_t x = (size_t)g.cw() * 2 * g.bps(); x < g.cp; ++x) CHECK_MSG(g.chroma[(size_t)y * g.cp + x] == kGarbage, "chroma pitch padding overwritten");
}
}  // namespace

TEST(color_rgb_to_yuv_matches_reference) {
    fb_stream* s = nullptr;
    OK(fb_stream_create(&s));
    const int sizes[][2] = {{64, 48}, {1366, 768}, {577, 1023}, {17, 9}, {1, 1}, {3, 5}};
    for (auto& sz : sizes)
        for (int fmt : {FB_FMT_NV12, FB_FMT_P010})
            for (int matrix : {FB_MATRIX_BT601, FB_MATRIX_BT709, FB_MATRIX_BT2020NC})
                for (int range : {FB_RANGE_LIMITED, FB_RANGE_FULL})
                    for (int dtype : {FB_DTYPE_FP32, FB_DTYPE_FP16}) {
                        bool alt = ((sz[1] + matrix + range + dtype) & 1) != 0;
                        run_rgb_to_yuv_case(sz[0], sz[1], sz[0], sz[1], fmt, matrix, range, dtype, alt ? 96 : 0,
                                            alt ? s : nullptr);
                    }
    // cropping: tensor larger than the frame (the pad region is dropped)
    run_rgb_to_yuv_case(640, 384, 577, 361, FB_FMT_NV12, FB_MATRIX_BT709, FB_RANGE_LIMITED, FB_DTYPE_FP16, 64, s);
    run_rgb_to_yuv_case(640, 384, 577, 361, FB_FMT_P010, FB_MATRIX_BT2020NC, FB_RANGE_LIMITED, FB_DTYPE_FP32, 0, s);
    OK(fb_stream_destroy(s));
}

namespace {
Tensor smooth_image(int w, int h) {
    Tensor img(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float fx = (float)x / std::max(1, w - 1), fy = (float)y / std::max(1, h - 1);
            float base = 0.25f + 0.5f * fx;
            img.at(0, x, y) = base + 0.15f * fy;
            img.at(1, x, y) = base - 0.10f * std::sin(3.0f * fy);
            img.at(2, x, y) = 0.30f + 0.4f * fy * fx + 0.1f * fx;
        }
    return img;
}
}  // namespace

TEST(color_roundtrip_rgb_yuv_rgb) {
    struct Case { int w, h, fmt, matrix, range, dtype; double max_tol, mean_tol; };
    const Case cases[] = {
        {1366, 768, FB_FMT_NV12, FB_MATRIX_BT709, FB_RANGE_LIMITED, FB_DTYPE_FP16, 0.012, 0.003},
        {577, 1023, FB_FMT_NV12, FB_MATRIX_BT601, FB_RANGE_FULL, FB_DTYPE_FP32, 0.012, 0.003},
        {577, 1023, FB_FMT_P010, FB_MATRIX_BT2020NC, FB_RANGE_LIMITED, FB_DTYPE_FP16, 0.006, 0.002},
        {255, 129, FB_FMT_P010, FB_MATRIX_BT709, FB_RANGE_FULL, FB_DTYPE_FP32, 0.004, 0.0015},
    };
    for (const Case& c : cases) {
        Tensor img = smooth_image(c.w, c.h);
        DBuf rgb(img.v.size() * esize(c.dtype));
        upload(rgb, img.v, c.dtype);
        Yuv f = make_yuv(c.w, c.h, c.fmt, 128);
        DevYuv d(f);
        OK(fb_rgb_to_yuv(rgb.p, c.dtype, c.w, c.h, &d.desc, c.matrix, c.range, nullptr));
        DBuf back(img.v.size() * esize(c.dtype));
        OK(fb_yuv_to_rgb(&d.desc, c.matrix, c.range, back.p, c.dtype, c.w, c.h, nullptr));
        OK(fb_stream_sync(nullptr));
        std::vector<float> got = download(back, img.v.size(), c.dtype);
        std::vector<float> want(img.v.size());
        for (size_t i = 0; i < want.size(); ++i) want[i] = quant_half(img.v[i]);  // what was actually uploaded (fp16)
        if (c.dtype == FB_DTYPE_FP32) want = img.v;
        double mx = 0, sum = 0;
        for (size_t i = 0; i < want.size(); ++i) {
            double e = std::fabs(got[i] - want[i]);
            mx = std::max(mx, e);
            sum += e;
        }
        double mean = sum / (double)want.size();
        char buf[200];
        std::snprintf(buf, sizeof(buf), "roundtrip %dx%d fmt=%d: max %g (tol %g) mean %g (tol %g)", c.w, c.h, c.fmt, mx,
                      c.max_tol, mean, c.mean_tol);
        CHECK_MSG(mx <= c.max_tol && mean <= c.mean_tol, buf);
    }
}

TEST(color_invalid_args) {
    Yuv f = make_yuv(8, 8, FB_FMT_NV12, 0);
    DevYuv d(f);
    DBuf out(3 * 8 * 8 * 2);
    CHECK_EQ_INT(fb_yuv_to_rgb(nullptr, 0, 0, out.p, FB_DTYPE_FP16, 8, 8, nullptr), FB_ERR_INVALID_ARG);
    CHECK(fb_last_error()[0] != 0);
    CHECK_EQ_INT(fb_yuv_to_rgb(&d.desc, 0, 0, out.p, FB_DTYPE_FP16, 4, 8, nullptr), FB_ERR_INVALID_ARG);  // dst too small
    CHECK_EQ_INT(fb_yuv_to_rgb(&d.desc, 7, 0, out.p, FB_DTYPE_FP16, 8, 8, nullptr), FB_ERR_INVALID_ARG);   // bad matrix
    CHECK_EQ_INT(fb_yuv_to_rgb(&d.desc, 0, 0, out.p, 9, 8, 8, nullptr), FB_ERR_UNSUPPORTED);               // bad dtype
    fb_yuv_frame_t bad = d.desc;
    bad.luma_pitch = 4;
    CHECK_EQ_INT(fb_yuv_to_rgb(&bad, 0, 0, out.p, FB_DTYPE_FP16, 8, 8, nullptr), FB_ERR_INVALID_ARG);
    bad = d.desc;
    bad.format = 5;
    CHECK_EQ_INT(fb_yuv_to_rgb(&bad, 0, 0, out.p, FB_DTYPE_FP16, 8, 8, nullptr), FB_ERR_UNSUPPORTED);
    CHECK_EQ_INT(fb_rgb_to_yuv(out.p, FB_DTYPE_FP16, 4, 4, &d.desc, 0, 0, nullptr), FB_ERR_INVALID_ARG);  // dst larger than src
}
