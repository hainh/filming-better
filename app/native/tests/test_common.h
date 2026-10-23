// Tiny test harness + helpers for the fb_native tests (host-only code: talks to the DLL through fb_native.h).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "fb_native.h"

namespace t {

struct TestCase {
    const char* name;
    std::function<void()> fn;
};
inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}
struct Registrar {
    Registrar(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};
struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

#define T_CAT2(a, b) a##b
#define T_CAT(a, b) T_CAT2(a, b)
#define TEST(name)                                                         \
    static void name();                                                    \
    static ::t::Registrar T_CAT(reg_, name)(#name, name);                  \
    static void name()

#define FAIL_MSG(msg)                                                                          \
    do {                                                                                       \
        char _b[1024];                                                                         \
        std::snprintf(_b, sizeof(_b), "%s:%d: %s", __FILE__, __LINE__, std::string(msg).c_str()); \
        throw ::t::Failure(_b);                                                                \
    } while (0)
#define CHECK(cond) \
    do { if (!(cond)) FAIL_MSG(std::string("CHECK failed: ") + #cond); } while (0)
#define CHECK_MSG(cond, msg) \
    do { if (!(cond)) FAIL_MSG(std::string("CHECK failed: ") + #cond + " -- " + (msg)); } while (0)
#define CHECK_EQ_INT(a, b)                                                                        \
    do {                                                                                          \
        long long _a = (long long)(a), _b2 = (long long)(b);                                      \
        if (_a != _b2) FAIL_MSG(std::string(#a " == " #b " failed: ") + std::to_string(_a) + " vs " + std::to_string(_b2)); \
    } while (0)
// status must be FB_OK; otherwise report fb_last_error()
#define OK(call)                                                                                  \
    do {                                                                                          \
        int _s = (call);                                                                          \
        if (_s != FB_OK) FAIL_MSG(std::string(#call " returned ") + std::to_string(_s) + ": " + fb_last_error()); \
    } while (0)

// ---- half <-> float (round to nearest even) ----
inline uint16_t f2h(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0));
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1);
        uint32_t mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1))) ++half;
        return (uint16_t)(sign | half);
    }
    uint32_t h = (uint32_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
    uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
    return (uint16_t)h;
}
inline float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ffu;
    uint32_t x;
    if (exp == 0) {
        if (man == 0) {
            x = sign;
        } else {
            exp = 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3ffu;
            x = sign | ((exp + 127 - 15) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7f800000u | (man << 13);
    } else {
        x = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// ---- RAII device buffer ----
struct DBuf {
    fb_devptr p = 0;
    size_t bytes = 0;
    explicit DBuf(size_t b) : bytes(b) { OK(fb_alloc(b, &p)); }
    DBuf(const DBuf&) = delete;
    DBuf& operator=(const DBuf&) = delete;
    ~DBuf() { fb_free(p); }
    void up(const void* host, size_t n, fb_stream* s = nullptr) {
        OK(fb_memcpy(p, (uint64_t)(uintptr_t)host, n, s));
        OK(fb_stream_sync(s));
    }
    void down(void* host, size_t n, fb_stream* s = nullptr) const {
        OK(fb_memcpy((uint64_t)(uintptr_t)host, p, n, s));
        OK(fb_stream_sync(s));
    }
};

// A planar RGB tensor on the host, always stored as float; (de)serialised to FP16/FP32 on transfer.
struct Tensor {
    int w = 0, h = 0;
    std::vector<float> v;  // 3*w*h
    Tensor() = default;
    Tensor(int w_, int h_) : w(w_), h(h_), v((size_t)3 * w_ * h_) {}
    float& at(int c, int x, int y) { return v[(size_t)c * w * h + (size_t)y * w + x]; }
    float at(int c, int x, int y) const { return v[(size_t)c * w * h + (size_t)y * w + x]; }
};

inline size_t esize(int dtype) { return dtype == FB_DTYPE_FP16 ? 2 : 4; }

inline void upload(DBuf& d, const std::vector<float>& v, int dtype, fb_stream* s = nullptr) {
    if (dtype == FB_DTYPE_FP32) {
        d.up(v.data(), v.size() * 4, s);
    } else {
        std::vector<uint16_t> h(v.size());
        for (size_t i = 0; i < v.size(); ++i) h[i] = f2h(v[i]);
        d.up(h.data(), h.size() * 2, s);
    }
}
inline std::vector<float> download(const DBuf& d, size_t count, int dtype, fb_stream* s = nullptr) {
    std::vector<float> out(count);
    if (dtype == FB_DTYPE_FP32) {
        d.down(out.data(), count * 4, s);
    } else {
        std::vector<uint16_t> h(count);
        d.down(h.data(), count * 2, s);
        for (size_t i = 0; i < count; ++i) out[i] = h2f(h[i]);
    }
    return out;
}

// Values already representable in half (so FP16 round trips are exact).
inline float quant_half(float f) { return h2f(f2h(f)); }

struct Rng {
    std::mt19937 g;
    explicit Rng(uint32_t seed) : g(seed) {}
    float uni(float a = 0.f, float b = 1.f) { return std::uniform_real_distribution<float>(a, b)(g); }
    int integer(int a, int b) { return std::uniform_int_distribution<int>(a, b)(g); }
};

}  // namespace t
