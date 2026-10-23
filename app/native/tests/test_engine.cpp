// TensorRT lean-runtime wrapper tests. Needs the engines produced by make_test_engines.py
// (directory passed in env FB_TEST_ENGINE_DIR, or compiled in as FB_TEST_ENGINE_DIR).
#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <thread>

#include "test_common.h"

#ifndef FB_TEST_ENGINE_DIR
#define FB_TEST_ENGINE_DIR "."
#endif
#ifndef FB_REAL_ENGINE_DIR
#define FB_REAL_ENGINE_DIR ""
#endif

using namespace t;

namespace {

std::string engine_dir() {
    const char* e = std::getenv("FB_TEST_ENGINE_DIR");
    return e && *e ? e : FB_TEST_ENGINE_DIR;
}
std::string engine_path(const char* name) { return engine_dir() + "\\" + name; }

struct Engine {
    fb_engine* e = nullptr;
    explicit Engine(const std::string& path) { OK(fb_engine_load(path.c_str(), &e)); }
    Engine(const Engine&) = delete;
    ~Engine() { fb_engine_free(e); }
};

std::vector<fb_tensor_desc_t> descs(fb_engine* e) {
    int n = 0;
    OK(fb_engine_io_count(e, &n));
    std::vector<fb_tensor_desc_t> v(n);
    for (int i = 0; i < n; ++i) OK(fb_engine_io_desc(e, i, &v[i]));
    return v;
}

const fb_tensor_desc_t& find(const std::vector<fb_tensor_desc_t>& d, const char* name) {
    for (auto& x : d)
        if (std::strcmp(x.name, name) == 0) return x;
    FAIL_MSG(std::string("no tensor ") + name);
}

void set_shape(fb_engine* e, const char* name, int64_t h, int64_t w) {
    int64_t dims[4] = {1, 3, h, w};
    OK(fb_engine_set_input_shape(e, name, 4, dims));
}

size_t nelem(const int64_t* d, int n) {
    size_t c = 1;
    for (int i = 0; i < n; ++i) c *= (size_t)d[i];
    return c;
}

// Runs the nearest-2x engine for one shape and checks the exact output.
void check_up2(fb_engine* e, int dtype, int h, int w, fb_stream* s) {
    Rng rng((uint32_t)(h * 1000 + w));
    Tensor in(w, h);
    for (auto& v : in.v) v = quant_half(rng.uni());
    set_shape(e, "input", h, w);
    int nd = 0;
    int64_t od[8];
    OK(fb_engine_io_shape(e, "output", &nd, od));
    CHECK_EQ_INT(nd, 4);
    CHECK(od[0] == 1 && od[1] == 3 && od[2] == 2 * h && od[3] == 2 * w);
    DBuf din(in.v.size() * esize(dtype)), dout(nelem(od, nd) * esize(dtype));
    upload(din, in.v, dtype, s);
    OK(fb_engine_set_tensor(e, "input", din.p));
    OK(fb_engine_set_tensor(e, "output", dout.p));
    OK(fb_engine_run(e, s));
    OK(fb_stream_sync(s));
    auto out = download(dout, nelem(od, nd), dtype);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < 2 * h; ++y)
            for (int x = 0; x < 2 * w; ++x) {
                float got = out[(size_t)c * 4 * h * w + (size_t)y * 2 * w + x];
                float want = in.at(c, x / 2, y / 2);
                if (got != want)
                    FAIL_MSG("up2 mismatch at c=" + std::to_string(c) + " x=" + std::to_string(x) + " y=" + std::to_string(y) +
                             " got " + std::to_string(got) + " want " + std::to_string(want));
            }
}

}  // namespace

TEST(engine_error_paths) {
    // bad library path (strict override) -> FB_ERR_TRT_LOAD, library state unchanged
    CHECK_EQ_INT(fb_trt_set_library("C:\\definitely\\not\\here\\nvinfer_lean_11.dll"), FB_ERR_TRT_LOAD);
    CHECK(fb_last_error()[0] != 0);
    int before = 0;
    OK(fb_trt_version(&before));
    CHECK(before >= 110000);
    std::printf("         TensorRT lean version %d\n", before);
    fb_engine* e = reinterpret_cast<fb_engine*>(0x1);
    CHECK_EQ_INT(fb_engine_load("C:\\definitely\\not\\here\\nothing.engine", &e), FB_ERR_IO);
    CHECK(e == nullptr);
    CHECK(fb_last_error()[0] != 0);
    CHECK(std::string(fb_last_error()).find("nothing.engine") != std::string::npos);
    CHECK_EQ_INT(fb_engine_load(nullptr, &e), FB_ERR_INVALID_ARG);
    CHECK_EQ_INT(fb_engine_load("", &e), FB_ERR_INVALID_ARG);
    CHECK_EQ_INT(fb_engine_load("x.engine", nullptr), FB_ERR_INVALID_ARG);
    // invalid UTF-8 path
    CHECK_EQ_INT(fb_engine_load("\xff\xfe.engine", &e), FB_ERR_INVALID_ARG);
    // a directory is not a readable engine file
    int st = fb_engine_load(engine_dir().c_str(), &e);
    CHECK_MSG(st == FB_ERR_IO, "directory as engine path -> " + std::to_string(st));
    // garbage file -> FB_ERR_ENGINE
    std::string junk = engine_path("junk.engine");
    {
        std::ofstream f(junk, std::ios::binary);
        std::vector<char> g(4096);
        for (size_t i = 0; i < g.size(); ++i) g[i] = (char)(i * 31 + 7);
        f.write(g.data(), (std::streamsize)g.size());
    }
    CHECK_EQ_INT(fb_engine_load(junk.c_str(), &e), FB_ERR_ENGINE);
    CHECK(e == nullptr);
    CHECK(fb_last_error()[0] != 0);
    // empty file -> FB_ERR_IO
    std::string empty = engine_path("empty.engine");
    { std::ofstream f(empty, std::ios::binary); }
    CHECK_EQ_INT(fb_engine_load(empty.c_str(), &e), FB_ERR_IO);
    std::remove(junk.c_str());
    std::remove(empty.c_str());
    // NULL engine handles
    int cnt;
    CHECK_EQ_INT(fb_engine_io_count(nullptr, &cnt), FB_ERR_INVALID_ARG);
    CHECK_EQ_INT(fb_engine_run(nullptr, nullptr), FB_ERR_INVALID_ARG);
}

TEST(engine_up2_fp16_dynamic_shapes) {
    fb_stream* s = nullptr;
    OK(fb_stream_create(&s));
    Engine eng(engine_path("up2_fp16.engine"));
    auto d = descs(eng.e);
    CHECK_EQ_INT(d.size(), 2);
    const auto& in = find(d, "input");
    const auto& out = find(d, "output");
    CHECK(in.is_input == 1 && out.is_input == 0);
    CHECK_EQ_INT(in.dtype, FB_DTYPE_FP16);
    CHECK_EQ_INT(out.dtype, FB_DTYPE_FP16);
    CHECK_EQ_INT(in.nb_dims, 4);
    CHECK(in.dims[0] == 1 && in.dims[1] == 3 && in.dims[2] == -1 && in.dims[3] == -1);
    CHECK(in.min_dims[2] == 16 && in.min_dims[3] == 16 && in.opt_dims[2] == 128 && in.max_dims[2] == 512 && in.max_dims[3] == 512);
    CHECK(in.min_dims[0] == 1 && in.max_dims[1] == 3);
    CHECK(out.dims[2] == -1);  // output dims are dynamic until the input shape is set

    // out-of-profile shape is rejected with a readable message
    int64_t bad[4] = {1, 3, 1024, 1024};
    CHECK_EQ_INT(fb_engine_set_input_shape(eng.e, "input", 4, bad), FB_ERR_ENGINE);
    CHECK(std::string(fb_last_error()).find("profile") != std::string::npos);
    int64_t bad2[3] = {3, 16, 16};
    CHECK_EQ_INT(fb_engine_set_input_shape(eng.e, "input", 3, bad2), FB_ERR_INVALID_ARG);
    CHECK_EQ_INT(fb_engine_set_input_shape(eng.e, "output", 4, bad), FB_ERR_INVALID_ARG);
    CHECK_EQ_INT(fb_engine_set_input_shape(eng.e, "nope", 4, bad), FB_ERR_INVALID_ARG);

    // run without shape / addresses -> error, not crash
    CHECK_EQ_INT(fb_engine_run(eng.e, s), FB_ERR_ENGINE);

    const int shapes[][2] = {{16, 16}, {77, 53}, {128, 128}, {33, 512}, {512, 512}, {101, 17}, {64, 64}};
    for (auto& sh : shapes) check_up2(eng.e, FB_DTYPE_FP16, sh[0], sh[1], (sh[0] & 1) ? s : nullptr);
    OK(fb_stream_destroy(s));
}

TEST(engine_up2_fp32) {
    Engine eng(engine_path("up2_fp32.engine"));
    auto d = descs(eng.e);
    CHECK_EQ_INT(find(d, "input").dtype, FB_DTYPE_FP32);
    check_up2(eng.e, FB_DTYPE_FP32, 45, 70, nullptr);
    check_up2(eng.e, FB_DTYPE_FP32, 256, 300, nullptr);
}

TEST(engine_interp_three_inputs) {
    fb_stream* s = nullptr;
    OK(fb_stream_create(&s));
    Engine eng(engine_path("interp_fp16.engine"));
    auto d = descs(eng.e);
    CHECK_EQ_INT(d.size(), 4);
    CHECK(find(d, "img0").is_input && find(d, "img1").is_input && find(d, "timestep").is_input && !find(d, "output").is_input);
    CHECK(find(d, "timestep").min_dims[0] == 1 && find(d, "timestep").max_dims[3] == 1);
    for (auto hw : {std::pair<int, int>{64, 64}, {96, 160}, {250, 130}}) {
        int h = hw.first, w = hw.second;
        Rng rng((uint32_t)(h + w));
        Tensor a(w, h), b(w, h);
        for (auto& v : a.v) v = quant_half(rng.uni());
        for (auto& v : b.v) v = quant_half(rng.uni());
        float tval = quant_half(rng.uni(0.1f, 0.9f));
        uint16_t t16 = f2h(tval);
        set_shape(eng.e, "img0", h, w);
        set_shape(eng.e, "img1", h, w);
        int64_t td[4] = {1, 1, 1, 1};
        OK(fb_engine_set_input_shape(eng.e, "timestep", 4, td));
        DBuf da(a.v.size() * 2), db(b.v.size() * 2), dt(2), dout(a.v.size() * 2);
        upload(da, a.v, FB_DTYPE_FP16, s);
        upload(db, b.v, FB_DTYPE_FP16, s);
        dt.up(&t16, 2, s);
        OK(fb_engine_set_tensor(eng.e, "img0", da.p));
        OK(fb_engine_set_tensor(eng.e, "img1", db.p));
        OK(fb_engine_set_tensor(eng.e, "timestep", dt.p));
        OK(fb_engine_set_tensor(eng.e, "output", dout.p));
        OK(fb_engine_run(eng.e, s));
        OK(fb_stream_sync(s));
        auto out = download(dout, a.v.size(), FB_DTYPE_FP16);
        double mx = 0;
        for (size_t i = 0; i < out.size(); ++i) mx = std::max(mx, (double)std::fabs(out[i] - (a.v[i] + (b.v[i] - a.v[i]) * tval)));
        CHECK_MSG(mx < 4e-3, "interp max diff " + std::to_string(mx));
    }
    OK(fb_stream_destroy(s));
}

TEST(engine_used_from_another_thread) {
    // created on this thread, used (and freed) on another one
    Engine* eng = new Engine(engine_path("up2_fp16.engine"));
    std::atomic<int> ok{0};
    std::string err;
    std::thread th([&] {
        try {
            fb_stream* s = nullptr;
            OK(fb_stream_create(&s));
            check_up2(eng->e, FB_DTYPE_FP16, 40, 24, s);
            OK(fb_stream_destroy(s));
            ok = 1;
        } catch (const std::exception& e) {
            err = e.what();
        }
    });
    th.join();
    CHECK_MSG(ok == 1, err);
    // and back on the creating thread
    check_up2(eng->e, FB_DTYPE_FP16, 32, 32, nullptr);
    std::thread th2([&] { delete eng; });
    th2.join();
}

TEST(engine_two_engines_concurrently) {
    Engine a(engine_path("up2_fp16.engine")), b(engine_path("up2_fp32.engine"));
    std::atomic<int> bad{0};
    std::thread ta([&] {
        try { for (int i = 0; i < 10; ++i) check_up2(a.e, FB_DTYPE_FP16, 30 + i, 40, nullptr); } catch (...) { ++bad; }
    });
    std::thread tb([&] {
        try { for (int i = 0; i < 10; ++i) check_up2(b.e, FB_DTYPE_FP32, 50, 20 + i, nullptr); } catch (...) { ++bad; }
    });
    ta.join();
    tb.join();
    CHECK_EQ_INT(bad.load(), 0);
}

TEST(engine_real_engines_if_present) {
    // Engines built by models/ (optional): loads them, checks descriptors, runs one tile.
    const char* names[] = {"span_x2.ampere+.engine", "animevideov3_x4.ampere+.engine"};
    std::string dir = FB_REAL_ENGINE_DIR;
    if (const char* e = std::getenv("FB_TEST_REAL_ENGINE_DIR")) dir = e;
    if (dir.empty()) { std::printf("         (no real engine dir configured, skipped)\n"); return; }
    fb_gpu_info_t gi;
    OK(fb_gpu_info(0, &gi));
    if (gi.cc_major < 8) { std::printf("         (GPU is not Ampere+, skipped)\n"); return; }
    int ran = 0;
    for (const char* n : names) {
        std::string p = dir + "\\" + n;
        if (!std::ifstream(p, std::ios::binary).good()) continue;
        Engine eng(p);
        auto d = descs(eng.e);
        const auto& in = find(d, "input");
        const auto& out = find(d, "output");
        CHECK(in.nb_dims == 4 && out.nb_dims == 4);
        std::printf("         %s: dtype=%d min=[%lld,%lld] opt=[%lld,%lld] max=[%lld,%lld]\n", n, in.dtype,
                    (long long)in.min_dims[2], (long long)in.min_dims[3], (long long)in.opt_dims[2],
                    (long long)in.opt_dims[3], (long long)in.max_dims[2], (long long)in.max_dims[3]);
        int h = (int)std::max<int64_t>(in.min_dims[2], 96), w = (int)std::max<int64_t>(in.min_dims[3], 128);
        set_shape(eng.e, "input", h, w);
        int nd;
        int64_t od[8];
        OK(fb_engine_io_shape(eng.e, "output", &nd, od));
        CHECK(od[2] % h == 0 && od[3] % w == 0);
        Rng rng(1);
        Tensor img(w, h);
        for (auto& v : img.v) v = quant_half(rng.uni());
        DBuf din((size_t)3 * h * w * esize(in.dtype)), dout(nelem(od, nd) * esize(out.dtype));
        upload(din, img.v, in.dtype);
        OK(fb_engine_set_tensor(eng.e, "input", din.p));
        OK(fb_engine_set_tensor(eng.e, "output", dout.p));
        OK(fb_engine_run(eng.e, nullptr));
        OK(fb_stream_sync(nullptr));
        auto o = download(dout, nelem(od, nd), out.dtype);
        for (float v : o) CHECK_MSG(std::isfinite(v), "non-finite engine output");
        ++ran;
    }
    if (!ran) std::printf("         (no real engines found in %s, skipped)\n", dir.c_str());
}

namespace {
size_t private_bytes() {
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc));
    return pmc.PrivateUsage;
}
DWORD handle_count() {
    DWORD n = 0;
    GetProcessHandleCount(GetCurrentProcess(), &n);
    return n;
}
}  // namespace

TEST(leak_check_alloc_stream_engine_loops) {
    const std::string path = engine_path("up2_fp16.engine");
    // warm-up: first use initialises lazily loaded CUDA/TRT modules and allocator pools
    for (int i = 0; i < 3; ++i) {
        fb_devptr p;
        OK(fb_alloc(64 << 20, &p));
        OK(fb_free(p));
        Engine e(path);
        check_up2(e.e, FB_DTYPE_FP16, 64, 64, nullptr);
    }
    OK(fb_stream_sync(nullptr));
    size_t free0 = 0, tot = 0;
    OK(fb_mem_info(&free0, &tot));
    size_t priv0 = private_bytes();
    DWORD h0 = handle_count();
    for (int i = 0; i < 100; ++i) {
        fb_devptr p = 0;
        OK(fb_alloc((size_t)(8 + i % 16) << 20, &p));
        fb_stream* s = nullptr;
        OK(fb_stream_create(&s));
        OK(fb_memset(p, 1, 1 << 20, s));
        OK(fb_stream_sync(s));
        OK(fb_stream_destroy(s));
        OK(fb_free(p));
    }
    size_t free1 = 0;
    OK(fb_mem_info(&free1, &tot));
    long long dalloc = (long long)free0 - (long long)free1;
    std::printf("         alloc/stream loop: VRAM delta %lld KB\n", dalloc / 1024);
    CHECK_MSG(std::llabs(dalloc) < (8 << 20), "VRAM leak in alloc/free loop: " + std::to_string(dalloc) + " bytes");

    for (int i = 0; i < 100; ++i) {
        Engine e(path);
        if (i % 10 == 0) check_up2(e.e, FB_DTYPE_FP16, 48 + i % 7, 48, nullptr);
    }
    OK(fb_stream_sync(nullptr));
    size_t free2 = 0;
    OK(fb_mem_info(&free2, &tot));
    long long dengine = (long long)free1 - (long long)free2;
    size_t priv1 = private_bytes();
    DWORD h1 = handle_count();
    std::printf("         engine loop: VRAM delta %lld KB, host private delta %lld KB, handles %lu -> %lu\n",
                dengine / 1024, ((long long)priv1 - (long long)priv0) / 1024, (unsigned long)h0, (unsigned long)h1);
    CHECK_MSG(std::llabs(dengine) < (16 << 20), "VRAM leak in engine load/free loop: " + std::to_string(dengine) + " bytes");
    CHECK_MSG((long long)priv1 - (long long)priv0 < (64 << 20), "host memory growth in loops");
    CHECK_MSG(h1 < h0 + 20, "handle leak");
}
