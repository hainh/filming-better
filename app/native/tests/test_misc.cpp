// Device / error-handling / scene-score / threading tests.
#include <algorithm>
#include <atomic>
#include <thread>

#include "test_common.h"

using namespace t;

TEST(device_basics) {
    CHECK_EQ_INT(fb_api_version(), FB_API_VERSION);
    int n = 0;
    OK(fb_device_count(&n));
    CHECK(n >= 1);
    fb_gpu_info_t gi;
    OK(fb_gpu_info(0, &gi));
    CHECK(gi.name[0] != 0);
    CHECK(gi.cc_major >= 7);
    CHECK(gi.driver_version >= FB_MIN_DRIVER_VERSION);
    CHECK(gi.runtime_version >= 12000);
    CHECK(gi.multiprocessor_count > 0);
    CHECK(gi.total_mem > (uint64_t)1 << 28);
    CHECK_EQ_INT(fb_gpu_info(n, &gi), FB_ERR_INVALID_ARG);
    CHECK_EQ_INT(fb_gpu_info(-1, &gi), FB_ERR_INVALID_ARG);

    size_t fr = 0, tot = 0;
    OK(fb_mem_info(&fr, &tot));
    CHECK(tot > 0 && fr > 0 && fr <= tot);

    int drv = 0;
    OK(fb_init(0, &drv));  // idempotent
    CHECK(drv >= FB_MIN_DRIVER_VERSION);
    CHECK_EQ_INT(fb_init(99, &drv), FB_ERR_INVALID_ARG);
    CHECK(fb_last_error()[0] != 0);
    CHECK(drv >= FB_MIN_DRIVER_VERSION);  // filled even on failure
    OK(fb_mem_info(&fr, &tot));            // a failed re-init must not break the initialised state
}

TEST(device_not_initialized_after_shutdown) {
    OK(fb_shutdown());
    fb_devptr p = 123;
    CHECK_EQ_INT(fb_alloc(16, &p), FB_ERR_NOT_INITIALIZED);
    CHECK(p == 0);
    CHECK(std::string(fb_last_error()).find("fb_init") != std::string::npos);
    fb_stream* s = nullptr;
    CHECK_EQ_INT(fb_stream_create(&s), FB_ERR_NOT_INITIALIZED);
    size_t a, b;
    CHECK_EQ_INT(fb_mem_info(&a, &b), FB_ERR_NOT_INITIALIZED);
    OK(fb_shutdown());  // double shutdown is fine
    OK(fb_init(0, nullptr));
    OK(fb_alloc(16, &p));
    OK(fb_free(p));
}

TEST(error_state_is_thread_local_and_cleared) {
    CHECK_EQ_INT(fb_alloc(0, nullptr), FB_ERR_INVALID_ARG);
    CHECK(fb_last_error()[0] != 0);
    std::string main_err = fb_last_error();
    std::thread th([&] {
        CHECK(fb_last_error()[0] == 0);  // fresh thread has no error
        fb_devptr p;
        CHECK_EQ_INT(fb_alloc(0, &p), FB_ERR_INVALID_ARG);
        CHECK(fb_last_error()[0] != 0);
    });
    th.join();
    CHECK(main_err == fb_last_error());
    fb_devptr p = 0;
    OK(fb_alloc(64, &p));           // a successful call clears the message
    CHECK(fb_last_error()[0] == 0);
    OK(fb_free(p));
    OK(fb_free(0));                 // no-op
    OK(fb_stream_destroy(nullptr)); // no-op
    OK(fb_engine_free(nullptr));    // no-op
}

TEST(memory_out_of_memory_maps_to_code) {
    fb_devptr p = 0;
    size_t fr = 0, tot = 0;
    OK(fb_mem_info(&fr, &tot));
    int st = fb_alloc((size_t)1 << 46, &p);  // 64 TB: far beyond VRAM (and fails fast; slightly-too-big requests are slow on WDDM)
    CHECK_EQ_INT(st, FB_ERR_OUT_OF_MEMORY);
    CHECK(p == 0);
    CHECK(std::string(fb_last_error()).find("cudaMalloc") != std::string::npos);
    // the context must still be usable afterwards
    OK(fb_alloc(1 << 20, &p));
    OK(fb_free(p));
}

TEST(streams_work_across_threads) {
    // Worker threads that never called fb_init themselves must still use the chosen device & primary context.
    std::atomic<int> failures{0};
    std::vector<std::thread> ths;
    for (int i = 0; i < 4; ++i)
        ths.emplace_back([&, i] {
            try {
                fb_stream* s = nullptr;
                OK(fb_stream_create(&s));
                CHECK(fb_stream_raw(s) != nullptr);
                for (int it = 0; it < 20; ++it) {
                    Tensor img(97 + i, 61 + it);
                    for (size_t k = 0; k < img.v.size(); ++k) img.v[k] = (float)((k * 7 + i) % 100) / 100.f;
                    DBuf a(img.v.size() * 4), b(img.v.size() * 2), c(img.v.size() * 4);
                    a.up(img.v.data(), img.v.size() * 4, s);
                    OK(fb_convert(a.p, FB_DTYPE_FP32, b.p, FB_DTYPE_FP16, img.w, img.h, s));
                    OK(fb_convert(b.p, FB_DTYPE_FP16, c.p, FB_DTYPE_FP32, img.w, img.h, s));
                    std::vector<float> out(img.v.size());
                    c.down(out.data(), out.size() * 4, s);
                    for (size_t k = 0; k < out.size(); ++k) CHECK(std::fabs(out[k] - img.v[k]) < 1e-3);
                }
                OK(fb_stream_destroy(s));
            } catch (const std::exception& e) {
                std::printf("thread %d: %s\n", i, e.what());
                ++failures;
            }
        });
    for (auto& th : ths) th.join();
    CHECK_EQ_INT(failures.load(), 0);
}

namespace {
// Reference scene score (same cell boundaries and luma weights as the kernel).
double ref_scene(const Tensor& a, const Tensor& b) {
    const int W = 64, H = 36;
    double total = 0;
    for (int cy = 0; cy < H; ++cy)
        for (int cx = 0; cx < W; ++cx) {
            int x0 = (int)((long long)cx * a.w / W), x1 = (int)((long long)(cx + 1) * a.w / W);
            int y0 = (int)((long long)cy * a.h / H), y1 = (int)((long long)(cy + 1) * a.h / H);
            if (x1 <= x0) x1 = std::min(x0 + 1, a.w);
            if (y1 <= y0) y1 = std::min(y0 + 1, a.h);
            if (x1 <= x0) x0 = x1 - 1;
            if (y1 <= y0) y0 = y1 - 1;
            double sa = 0, sb = 0;
            for (int y = y0; y < y1; ++y)
                for (int x = x0; x < x1; ++x) {
                    sa += 0.299 * a.at(0, x, y) + 0.587 * a.at(1, x, y) + 0.114 * a.at(2, x, y);
                    sb += 0.299 * b.at(0, x, y) + 0.587 * b.at(1, x, y) + 0.114 * b.at(2, x, y);
                }
            double n = (double)(x1 - x0) * (y1 - y0);
            total += std::fabs(sa - sb) / n;
        }
    return total / (W * H);
}
}  // namespace

TEST(scene_score) {
    fb_stream* s = nullptr;
    OK(fb_stream_create(&s));
    struct Case { int w, h, dtype; };
    const Case cases[] = {{1920, 1080, FB_DTYPE_FP16}, {577, 1023, FB_DTYPE_FP32}, {20, 10, FB_DTYPE_FP32}, {64, 36, FB_DTYPE_FP16}, {1366, 768, FB_DTYPE_FP32}};
    for (const Case& c : cases) {
        Rng rng((uint32_t)c.w);
        Tensor a(c.w, c.h), shifted(c.w, c.h), inverted(c.w, c.h), other(c.w, c.h), noisy(c.w, c.h);
        for (int ch = 0; ch < 3; ++ch)
            for (int y = 0; y < c.h; ++y)
                for (int x = 0; x < c.w; ++x) {
                    float v = quant_half(0.5f + 0.4f * std::sin(0.02f * x + 0.5f * ch) * std::cos(0.03f * y));
                    a.at(ch, x, y) = v;
                    inverted.at(ch, x, y) = quant_half(1.f - v);
                    other.at(ch, x, y) = quant_half(rng.uni());
                    noisy.at(ch, x, y) = quant_half(std::min(1.f, std::max(0.f, v + rng.uni(-0.02f, 0.02f))));
                    shifted.at(ch, x, y) = quant_half(0.5f + 0.4f * std::sin(0.02f * (x + 1) + 0.5f * ch) * std::cos(0.03f * y));
                }
        DBuf da(a.v.size() * esize(c.dtype)), db(a.v.size() * esize(c.dtype));
        upload(da, a.v, c.dtype);
        auto score = [&](const Tensor& other_t) {
            upload(db, other_t.v, c.dtype);
            float sc = -1;
            OK(fb_scene_score(da.p, db.p, c.dtype, c.w, c.h, &sc, s));
            double want = ref_scene(a, other_t);
            CHECK_MSG(std::fabs(sc - want) < 1e-4, "scene score " + std::to_string(sc) + " vs reference " + std::to_string(want));
            return sc;
        };
        float same = score(a);
        CHECK_MSG(same == 0.f, "identical frames must score exactly 0, got " + std::to_string(same));
        float small = score(shifted);
        float noise = score(noisy);
        float inv = score(inverted);
        float oth = score(other);
        CHECK(small < 0.05f);
        CHECK(noise < 0.01f);
        CHECK_MSG(inv > 0.15f, "inverted frame should score high: " + std::to_string(inv));
        CHECK_MSG(oth > 0.05f || c.w < 64, "unrelated frame should score high: " + std::to_string(oth));
        CHECK(inv > small * 3);
        // the kernel runs in-stream and is synchronized before return: the score is already valid
    }
    // concurrent use from several threads (shared scratch is mutex protected)
    std::atomic<int> bad{0};
    std::vector<std::thread> ths;
    for (int i = 0; i < 4; ++i)
        ths.emplace_back([&, i] {
            try {
                Tensor a(160, 90), b(160, 90);
                for (size_t k = 0; k < a.v.size(); ++k) { a.v[k] = 0.1f * (float)(i + 1); b.v[k] = 0.1f * (float)(i + 1) + 0.1f; }
                DBuf da(a.v.size() * 4), db(a.v.size() * 4);
                upload(da, a.v, FB_DTYPE_FP32);
                upload(db, b.v, FB_DTYPE_FP32);
                for (int it = 0; it < 30; ++it) {
                    float sc = 0;
                    OK(fb_scene_score(da.p, db.p, FB_DTYPE_FP32, 160, 90, &sc, nullptr));
                    if (std::fabs(sc - 0.1f) > 1e-4) ++bad;
                }
            } catch (...) { ++bad; }
        });
    for (auto& th : ths) th.join();
    CHECK_EQ_INT(bad.load(), 0);
    OK(fb_stream_destroy(s));
}
