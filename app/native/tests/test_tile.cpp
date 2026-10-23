// pad / crop / convert / tile split / tile merge / finalize tests.
#include <algorithm>

#include "test_common.h"

using namespace t;

namespace {

Tensor random_tensor(int w, int h, uint32_t seed) {
    Rng rng(seed);
    Tensor t(w, h);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                // structure + noise so that misplaced pixels are noticed
                t.at(c, x, y) = quant_half(0.5f + 0.4f * std::sin(0.05f * x * (c + 1) + 0.03f * y) + 0.05f * rng.uni(-1, 1));
    return t;
}

Tensor ref_region(const Tensor& s, int x0, int y0, int dw, int dh) {
    Tensor o(dw, dh);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < dh; ++y)
            for (int x = 0; x < dw; ++x)
                o.at(c, x, y) = s.at(c, std::min(std::max(x0 + x, 0), s.w - 1), std::min(std::max(y0 + y, 0), s.h - 1));
    return o;
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    CHECK(a.size() == b.size());
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

Tensor nearest2x(const Tensor& s) {
    Tensor o(s.w * 2, s.h * 2);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < o.h; ++y)
            for (int x = 0; x < o.w; ++x) o.at(c, x, y) = s.at(c, x / 2, y / 2);
    return o;
}

}  // namespace

TEST(tensor_pad_crop_split_exact) {
    fb_stream* s = nullptr;
    OK(fb_stream_create(&s));
    for (int dtype : {FB_DTYPE_FP32, FB_DTYPE_FP16}) {
        Tensor img = random_tensor(577, 1023, 11);
        DBuf src(img.v.size() * esize(dtype));
        upload(src, img.v, dtype);

        // pad to multiples of 64 (edge replicate)
        int pw = 640, ph = 1024;
        DBuf padded((size_t)3 * pw * ph * esize(dtype));
        OK(fb_pad(src.p, dtype, img.w, img.h, padded.p, pw, ph, s));
        OK(fb_stream_sync(s));
        CHECK(max_diff(download(padded, (size_t)3 * pw * ph, dtype), ref_region(img, 0, 0, pw, ph).v) == 0.0);

        // crop the pad off again -> original
        DBuf back((size_t)3 * img.w * img.h * esize(dtype));
        OK(fb_crop(padded.p, dtype, pw, ph, 0, 0, back.p, img.w, img.h, s));
        OK(fb_stream_sync(s));
        CHECK(max_diff(download(back, img.v.size(), dtype), img.v) == 0.0);

        // inner crop
        DBuf inner((size_t)3 * 100 * 77 * esize(dtype));
        OK(fb_crop(src.p, dtype, img.w, img.h, 13, 500, inner.p, 100, 77, s));
        OK(fb_stream_sync(s));
        CHECK(max_diff(download(inner, (size_t)3 * 100 * 77, dtype), ref_region(img, 13, 500, 100, 77).v) == 0.0);

        // tile split incl. out-of-bounds (negative start, past the right/bottom edge)
        const int pos[][2] = {{0, 0}, {-20, -30}, {500, 900}, {300, 1000}, {-5, 800}};
        for (auto& p : pos) {
            DBuf tile((size_t)3 * 128 * 96 * esize(dtype));
            OK(fb_tile_split(src.p, dtype, img.w, img.h, p[0], p[1], tile.p, 128, 96, s));
            OK(fb_stream_sync(s));
            CHECK(max_diff(download(tile, (size_t)3 * 128 * 96, dtype), ref_region(img, p[0], p[1], 128, 96).v) == 0.0);
        }
        // crop region outside -> error
        CHECK_EQ_INT(fb_crop(src.p, dtype, img.w, img.h, 500, 0, back.p, 100, 10, s), FB_ERR_INVALID_ARG);
        CHECK(fb_last_error()[0] != 0);
        CHECK_EQ_INT(fb_pad(src.p, dtype, img.w, img.h, padded.p, 100, ph, s), FB_ERR_INVALID_ARG);
    }
    OK(fb_stream_destroy(s));
}

TEST(tensor_convert_dtype) {
    Tensor img = random_tensor(333, 201, 5);
    // fp32 (not fp16-exact values) -> fp16 -> fp32
    Rng rng(3);
    for (auto& v : img.v) v = rng.uni(-2.f, 3.f);
    DBuf a(img.v.size() * 4), b(img.v.size() * 2), c(img.v.size() * 4), d(img.v.size() * 4);
    upload(a, img.v, FB_DTYPE_FP32);
    OK(fb_convert(a.p, FB_DTYPE_FP32, b.p, FB_DTYPE_FP16, img.w, img.h, nullptr));
    OK(fb_convert(b.p, FB_DTYPE_FP16, c.p, FB_DTYPE_FP32, img.w, img.h, nullptr));
    OK(fb_convert(a.p, FB_DTYPE_FP32, d.p, FB_DTYPE_FP32, img.w, img.h, nullptr));  // same dtype: plain copy
    OK(fb_stream_sync(nullptr));
    auto half = download(b, img.v.size(), FB_DTYPE_FP16);
    auto full = download(c, img.v.size(), FB_DTYPE_FP32);
    auto copy = download(d, img.v.size(), FB_DTYPE_FP32);
    for (size_t i = 0; i < img.v.size(); ++i) {
        CHECK_MSG(half[i] == quant_half(img.v[i]), "fp32->fp16 mismatch (round to nearest even expected)");
        CHECK(full[i] == half[i]);
        CHECK(copy[i] == img.v[i]);
    }
}

TEST(memory_copy_set_2d) {
    fb_stream* s = nullptr;
    OK(fb_stream_create(&s));
    // memset
    DBuf a(1000);
    OK(fb_memset(a.p, 0xAB, 1000, s));
    std::vector<uint8_t> h(1000);
    OK(fb_stream_sync(s));
    a.down(h.data(), 1000);
    for (auto v : h) CHECK(v == 0xAB);
    // 2D copy host(tight, 100 wide) -> device(pitch 256) -> host(pitch 128)
    const size_t W = 100, H = 20;
    std::vector<uint8_t> src(W * H), out(128 * H, 0xEE);
    for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t)(i * 7);
    DBuf dev(256 * H);
    OK(fb_memset(dev.p, 0, 256 * H, s));
    OK(fb_memcpy2d(dev.p, 256, (uint64_t)(uintptr_t)src.data(), W, W, H, s));
    OK(fb_memcpy2d((uint64_t)(uintptr_t)out.data(), 128, dev.p, 256, W, H, s));
    OK(fb_stream_sync(s));
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) CHECK(out[y * 128 + x] == src[y * W + x]);
        for (size_t x = W; x < 128; ++x) CHECK(out[y * 128 + x] == 0xEE);
    }
    // device -> device
    DBuf b(1000);
    OK(fb_memcpy(b.p, a.p, 1000, s));
    OK(fb_stream_sync(s));
    std::vector<uint8_t> h2(1000);
    b.down(h2.data(), 1000);
    CHECK(h2 == h);
    OK(fb_stream_destroy(s));
}

namespace {

struct TileGrid {
    std::vector<int> xs, ys;
};

// Mimics what the Rust pipeline does: tile starts every (tile-overlap), optionally clamped so tiles stay inside
// the image (shift_last) or allowed to run past the border (then replicate padding is used).
std::vector<int> starts(int extent, int tile, int overlap, bool shift_last) {
    std::vector<int> v;
    if (extent <= tile) { v.push_back(0); return v; }
    int step = tile - overlap;
    for (int p = 0;; p += step) {
        if (p + tile >= extent) {
            v.push_back(shift_last ? extent - tile : p);
            break;
        }
        v.push_back(p);
    }
    return v;
}

// Full tile pipeline with a host-side "upscaler" (scale 1 = identity, scale 2 = nearest).
Tensor run_tiled(const Tensor& img, int dtype, int tile, int overlap, int feather, int scale, bool shift_last) {
    DBuf src(img.v.size() * esize(dtype));
    upload(src, img.v, dtype);
    int ow = img.w * scale, oh = img.h * scale;
    DBuf accum((size_t)3 * ow * oh * 4), weight((size_t)ow * oh * 4), dst((size_t)3 * ow * oh * esize(dtype));
    OK(fb_memset(accum.p, 0, accum.bytes, nullptr));
    OK(fb_memset(weight.p, 0, weight.bytes, nullptr));
    auto xs = starts(img.w, tile, overlap, shift_last), ys = starts(img.h, tile, overlap, shift_last);
    int tw = tile, th = tile;
    DBuf tin((size_t)3 * tw * th * esize(dtype));
    DBuf tout((size_t)3 * tw * scale * th * scale * esize(dtype));
    for (int y : ys)
        for (int x : xs) {
            OK(fb_tile_split(src.p, dtype, img.w, img.h, x, y, tin.p, tw, th, nullptr));
            OK(fb_stream_sync(nullptr));
            Tensor t(tw, th);
            t.v = download(tin, (size_t)3 * tw * th, dtype);
            Tensor up = scale == 2 ? nearest2x(t) : t;
            upload(tout, up.v, dtype);
            int flags = 0;
            if (x == 0) flags |= 1;
            if (y == 0) flags |= 2;
            if (x + tw >= img.w) flags |= 4;
            if (y + th >= img.h) flags |= 8;
            OK(fb_tile_merge(tout.p, dtype, tw * scale, th * scale, accum.p, weight.p, ow, oh, x * scale, y * scale,
                             feather * scale, flags, nullptr));
        }
    OK(fb_tile_finalize(accum.p, weight.p, ow, oh, dst.p, dtype, nullptr));
    OK(fb_stream_sync(nullptr));
    Tensor out(ow, oh);
    out.v = download(dst, out.v.size(), dtype);
    return out;
}

}  // namespace

TEST(tiling_identity_reconstructs_image) {
    struct Case { int w, h, tile, overlap, feather; bool shift; int dtype; };
    const Case cases[] = {
        {577, 1023, 256, 32, 16, true, FB_DTYPE_FP16},
        {577, 1023, 256, 32, 16, false, FB_DTYPE_FP32},   // last tiles run past the border (replicate)
        {1366, 300, 192, 48, 24, true, FB_DTYPE_FP16},
        {100, 80, 256, 32, 16, true, FB_DTYPE_FP32},       // image smaller than the tile
        {512, 512, 256, 32, 0, true, FB_DTYPE_FP32},       // feather 0 = plain average
        {300, 300, 128, 64, 64, false, FB_DTYPE_FP16},     // feather == overlap
    };
    for (const Case& c : cases) {
        Tensor img = random_tensor(c.w, c.h, (uint32_t)c.w);
        Tensor out = run_tiled(img, c.dtype, c.tile, c.overlap, c.feather, 1, c.shift);
        double m = max_diff(out.v, img.v);
        CHECK_MSG(m <= (c.dtype == FB_DTYPE_FP16 ? 1e-3 : 1e-5),
                  "identity tiling " + std::to_string(c.w) + "x" + std::to_string(c.h) + " max diff " + std::to_string(m));
    }
}

TEST(tiling_nearest2x_reconstructs_image) {
    const int sizes[][2] = {{577, 1023}, {200, 150}, {64, 64}};
    for (auto& sz : sizes)
        for (bool shift : {true, false}) {
            Tensor img = random_tensor(sz[0], sz[1], 77);
            Tensor out = run_tiled(img, FB_DTYPE_FP16, 128, 32, 12, 2, shift);
            Tensor want = nearest2x(img);
            double m = max_diff(out.v, want.v);
            CHECK_MSG(m <= 1e-3, "nearest2x tiling " + std::to_string(sz[0]) + "x" + std::to_string(sz[1]) + " max diff " + std::to_string(m));
        }
}

TEST(tiling_feather_blends_seam) {
    // Two overlapping tiles with different constant values (0.2 left, 0.8 right). Inside the overlap the result has
    // to ramp monotonically; outside it equals the owning tile's value.
    const int W = 200, H = 8, T = 120, ov = 40;  // tile A: [0,120), tile B: [80,200)
    DBuf accum(3 * W * H * 4), weight(W * H * 4), dst(3 * W * H * 4), ta(3 * T * H * 4), tb(3 * T * H * 4);
    OK(fb_memset(accum.p, 0, accum.bytes, nullptr));
    OK(fb_memset(weight.p, 0, weight.bytes, nullptr));
    std::vector<float> a(3 * T * H, 0.2f), b(3 * T * H, 0.8f);
    ta.up(a.data(), a.size() * 4);
    tb.up(b.data(), b.size() * 4);
    OK(fb_tile_merge(ta.p, FB_DTYPE_FP32, T, H, accum.p, weight.p, W, H, 0, 0, ov, 1 | 2 | 8, nullptr));      // left,top,bottom on border
    OK(fb_tile_merge(tb.p, FB_DTYPE_FP32, T, H, accum.p, weight.p, W, H, W - T, 0, ov, 2 | 4 | 8, nullptr));  // right,top,bottom
    OK(fb_tile_finalize(accum.p, weight.p, W, H, dst.p, FB_DTYPE_FP32, nullptr));
    OK(fb_stream_sync(nullptr));
    auto out = download(dst, 3 * W * H, FB_DTYPE_FP32);
    for (int c = 0; c < 3; ++c) {
        float prev = -1;
        for (int x = 0; x < W; ++x) {
            float v = out[(size_t)c * W * H + 3 * W + x];  // row 3
            CHECK_MSG(v >= prev - 1e-6f, "seam must be monotonic");
            prev = v;
            if (x < 80) CHECK(std::fabs(v - 0.2f) < 1e-5);
            if (x >= 120) CHECK(std::fabs(v - 0.8f) < 1e-5);
            if (x >= 80 && x < 120) CHECK(v > 0.2f - 1e-6f && v < 0.8f + 1e-6f);
        }
        // the middle of the overlap is about half way
        float mid = out[(size_t)c * W * H + 3 * W + 100];
        CHECK_MSG(std::fabs(mid - 0.5f) < 0.06f, "seam midpoint should be ~0.5, got " + std::to_string(mid));
    }
}

TEST(tiling_clipped_and_invalid) {
    // tile partially outside the output is clipped; negative offsets too
    DBuf accum(3 * 10 * 10 * 4), weight(10 * 10 * 4), dst(3 * 10 * 10 * 4), tile(3 * 8 * 8 * 4);
    OK(fb_memset(accum.p, 0, accum.bytes, nullptr));
    OK(fb_memset(weight.p, 0, weight.bytes, nullptr));
    std::vector<float> ones(3 * 64, 1.0f);
    tile.up(ones.data(), ones.size() * 4);
    OK(fb_tile_merge(tile.p, FB_DTYPE_FP32, 8, 8, accum.p, weight.p, 10, 10, 6, 6, 2, 0, nullptr));
    OK(fb_tile_merge(tile.p, FB_DTYPE_FP32, 8, 8, accum.p, weight.p, 10, 10, -6, -6, 2, 0, nullptr));
    OK(fb_tile_merge(tile.p, FB_DTYPE_FP32, 8, 8, accum.p, weight.p, 10, 10, 100, 100, 2, 0, nullptr));  // fully outside
    OK(fb_tile_finalize(accum.p, weight.p, 10, 10, dst.p, FB_DTYPE_FP32, nullptr));
    OK(fb_stream_sync(nullptr));
    auto out = download(dst, 300, FB_DTYPE_FP32);
    CHECK(std::fabs(out[7 * 10 + 7] - 1.0f) < 1e-6);  // covered
    CHECK(out[4 * 10 + 4] == 0.0f);                   // not covered -> 0 (accum 0 / eps)
    CHECK_EQ_INT(fb_tile_merge(tile.p, FB_DTYPE_FP32, 8, 8, accum.p, weight.p, 10, 10, 0, 0, -1, 0, nullptr), FB_ERR_INVALID_ARG);
    CHECK_EQ_INT(fb_tile_merge(0, FB_DTYPE_FP32, 8, 8, accum.p, weight.p, 10, 10, 0, 0, 1, 0, nullptr), FB_ERR_INVALID_ARG);
}
