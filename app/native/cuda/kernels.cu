// CUDA kernels for fb_native: colour conversion, pad/crop/tile, tile merge, scene score.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <mutex>

#include "kernels.h"

namespace fb::k {

// ------------------------------------------------------------------------------------------
// Element load/store helpers
// ------------------------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float ld(const T* p);
template <>
__device__ __forceinline__ float ld<float>(const float* p) {
    return *p;
}
template <>
__device__ __forceinline__ float ld<__half>(const __half* p) {
    return __half2float(*p);
}
template <typename T>
__device__ __forceinline__ void st(T* p, float v);
template <>
__device__ __forceinline__ void st<float>(float* p, float v) {
    *p = v;
}
template <>
__device__ __forceinline__ void st<__half>(__half* p, float v) {
    *p = __float2half_rn(v);
}

__device__ __forceinline__ float sat(float v) { return fminf(fmaxf(v, 0.f), 1.f); }

// ------------------------------------------------------------------------------------------
// Colour
// ------------------------------------------------------------------------------------------
// Sample fetch. Luma/chroma are addressed in bytes so any pitch / base alignment works.
template <bool P010>
__device__ __forceinline__ float luma_code(const uint8_t* base, size_t pitch, int x, int y) {
    const uint8_t* p = base + (size_t)y * pitch;
    if (P010) {
        const uint8_t* q = p + (size_t)x * 2;
        unsigned v = (unsigned)q[0] | ((unsigned)q[1] << 8);
        return (float)(v >> 6);
    }
    return (float)p[x];
}
template <bool P010>
__device__ __forceinline__ void chroma_code(const uint8_t* base, size_t pitch, int cx, int cy, float& u, float& v) {
    const uint8_t* p = base + (size_t)cy * pitch;
    if (P010) {
        const uint8_t* q = p + (size_t)cx * 4;
        u = (float)((((unsigned)q[0]) | ((unsigned)q[1] << 8)) >> 6);
        v = (float)((((unsigned)q[2]) | ((unsigned)q[3] << 8)) >> 6);
    } else {
        const uint8_t* q = p + (size_t)cx * 2;
        u = (float)q[0];
        v = (float)q[1];
    }
}

// YUV 4:2:0 (centre-sited chroma, bilinear upsample) -> planar RGB. Output pixels outside the visible area
// replicate the nearest visible pixel (edge replicate of the converted result).
template <typename T, bool P010>
__global__ void k_yuv_to_rgb(const uint8_t* __restrict__ luma, size_t lpitch, const uint8_t* __restrict__ chroma,
                             size_t cpitch, int w, int h, ColorCoef c, T* __restrict__ dst, int dw, int dh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    int sx = min(x, w - 1);
    int sy = min(y, h - 1);
    int cw = (w + 1) >> 1, ch = (h + 1) >> 1;

    float fx = (sx + 0.5f) * 0.5f - 0.5f;
    float fy = (sy + 0.5f) * 0.5f - 0.5f;
    float flx = floorf(fx), fly = floorf(fy);
    float ax = fx - flx, ay = fy - fly;
    int x0 = min(max((int)flx, 0), cw - 1), x1 = min(max((int)flx + 1, 0), cw - 1);
    int y0 = min(max((int)fly, 0), ch - 1), y1 = min(max((int)fly + 1, 0), ch - 1);

    float u00, v00, u10, v10, u01, v01, u11, v11;
    chroma_code<P010>(chroma, cpitch, x0, y0, u00, v00);
    chroma_code<P010>(chroma, cpitch, x1, y0, u10, v10);
    chroma_code<P010>(chroma, cpitch, x0, y1, u01, v01);
    chroma_code<P010>(chroma, cpitch, x1, y1, u11, v11);
    float u = (u00 * (1.f - ax) + u10 * ax) * (1.f - ay) + (u01 * (1.f - ax) + u11 * ax) * ay;
    float v = (v00 * (1.f - ax) + v10 * ax) * (1.f - ay) + (v01 * (1.f - ax) + v11 * ax) * ay;

    float Y = (luma_code<P010>(luma, lpitch, sx, sy) - c.y_off) / c.y_rng;
    float Cb = (u - c.c_off) / c.c_rng;
    float Cr = (v - c.c_off) / c.c_rng;
    float kg = 1.f - c.kr - c.kb;
    float r = Y + 2.f * (1.f - c.kr) * Cr;
    float b = Y + 2.f * (1.f - c.kb) * Cb;
    float g = Y - (2.f * c.kr * (1.f - c.kr) / kg) * Cr - (2.f * c.kb * (1.f - c.kb) / kg) * Cb;

    size_t plane = (size_t)dw * dh;
    size_t o = (size_t)y * dw + x;
    st<T>(dst + o, sat(r));
    st<T>(dst + plane + o, sat(g));
    st<T>(dst + 2 * plane + o, sat(b));
}

template <typename T, bool P010>
__device__ __forceinline__ void put_luma(uint8_t* base, size_t pitch, int x, int y, float code) {
    uint8_t* p = base + (size_t)y * pitch;
    if (P010) {
        unsigned q = ((unsigned)code) << 6;
        uint8_t* d = p + (size_t)x * 2;
        d[0] = (uint8_t)(q & 0xff);
        d[1] = (uint8_t)(q >> 8);
    } else {
        p[x] = (uint8_t)code;
    }
}

// One thread per chroma sample: converts the 2x2 luma block and averages its RGB for the chroma.
template <typename T, bool P010>
__global__ void k_rgb_to_yuv(const T* __restrict__ src, int sw, int sh, uint8_t* __restrict__ luma, size_t lpitch,
                             uint8_t* __restrict__ chroma, size_t cpitch, int w, int h, ColorCoef c) {
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;
    int cw = (w + 1) >> 1, ch = (h + 1) >> 1;
    if (cx >= cw || cy >= ch) return;
    size_t plane = (size_t)sw * sh;
    float kg = 1.f - c.kr - c.kb;
    float sr = 0.f, sg = 0.f, sb = 0.f;
#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
#pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            int px = 2 * cx + dx, py = 2 * cy + dy;
            int qx = min(px, w - 1), qy = min(py, h - 1);  // replicate for odd sizes
            size_t o = (size_t)qy * sw + qx;
            float r = sat(ld<T>(src + o));
            float g = sat(ld<T>(src + plane + o));
            float b = sat(ld<T>(src + 2 * plane + o));
            sr += r;
            sg += g;
            sb += b;
            if (px < w && py < h) {
                float Y = c.kr * r + kg * g + c.kb * b;
                float code = fminf(fmaxf(rintf(Y * c.y_rng + c.y_off), 0.f), c.max_code);
                put_luma<T, P010>(luma, lpitch, px, py, code);
            }
        }
    }
    sr *= 0.25f;
    sg *= 0.25f;
    sb *= 0.25f;
    float Y = c.kr * sr + kg * sg + c.kb * sb;
    float Cb = (sb - Y) / (2.f * (1.f - c.kb));
    float Cr = (sr - Y) / (2.f * (1.f - c.kr));
    float ucode = fminf(fmaxf(rintf(Cb * c.c_rng + c.c_off), 0.f), c.max_code);
    float vcode = fminf(fmaxf(rintf(Cr * c.c_rng + c.c_off), 0.f), c.max_code);
    uint8_t* d = chroma + (size_t)cy * cpitch;
    if (P010) {
        unsigned uq = ((unsigned)ucode) << 6, vq = ((unsigned)vcode) << 6;
        d += (size_t)cx * 4;
        d[0] = (uint8_t)(uq & 0xff);
        d[1] = (uint8_t)(uq >> 8);
        d[2] = (uint8_t)(vq & 0xff);
        d[3] = (uint8_t)(vq >> 8);
    } else {
        d += (size_t)cx * 2;
        d[0] = (uint8_t)ucode;
        d[1] = (uint8_t)vcode;
    }
}

// ------------------------------------------------------------------------------------------
// pad / crop / tile split (all "clamped region copy")
// ------------------------------------------------------------------------------------------
template <typename E>
__global__ void k_region_copy(const E* __restrict__ src, int sw, int sh, int x0, int y0, E* __restrict__ dst, int dw,
                              int dh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    int c = blockIdx.z;
    int sx = min(max(x0 + x, 0), sw - 1);
    int sy = min(max(y0 + y, 0), sh - 1);
    dst[(size_t)c * dw * dh + (size_t)y * dw + x] = src[(size_t)c * sw * sh + (size_t)sy * sw + sx];
}

template <typename TS, typename TD>
__global__ void k_convert(const TS* __restrict__ src, TD* __restrict__ dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) st<TD>(dst + i, ld<TS>(src + i));
}

// ------------------------------------------------------------------------------------------
// tile merge / finalize
// ------------------------------------------------------------------------------------------
__device__ __forceinline__ float ramp_weight(int t, int n, int feather, bool lo_border, bool hi_border) {
    float w = 1.f;
    if (feather > 0) {
        if (!lo_border && t < feather) w = fminf(w, (t + 0.5f) / (float)feather);
        int r = n - 1 - t;
        if (!hi_border && r < feather) w = fminf(w, (r + 0.5f) / (float)feather);
    }
    return w;
}

template <typename T>
__global__ void k_tile_merge(const T* __restrict__ tile, int tw, int th, float* __restrict__ accum,
                             float* __restrict__ weight, int ow, int oh, int dx, int dy, int feather, int flags) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tw || ty >= th) return;
    int ox = dx + tx, oy = dy + ty;
    if (ox < 0 || oy < 0 || ox >= ow || oy >= oh) return;
    float wx = ramp_weight(tx, tw, feather, flags & 1, flags & 4);
    float wy = ramp_weight(ty, th, feather, flags & 2, flags & 8);
    float w = wx * wy;
    size_t tp = (size_t)tw * th, op = (size_t)ow * oh;
    size_t ti = (size_t)ty * tw + tx, oi = (size_t)oy * ow + ox;
#pragma unroll
    for (int c = 0; c < 3; ++c) accum[c * op + oi] += ld<T>(tile + c * tp + ti) * w;
    weight[oi] += w;
}

template <typename T>
__global__ void k_tile_finalize(const float* __restrict__ accum, const float* __restrict__ weight, int ow, int oh,
                                T* __restrict__ dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ow || y >= oh) return;
    size_t op = (size_t)ow * oh, i = (size_t)y * ow + x;
    float inv = 1.f / fmaxf(weight[i], 1e-6f);
#pragma unroll
    for (int c = 0; c < 3; ++c) st<T>(dst + c * op + i, accum[c * op + i] * inv);
}

// ------------------------------------------------------------------------------------------
// scene score
// ------------------------------------------------------------------------------------------
constexpr int kSceneW = 64, kSceneH = 36, kSceneCells = kSceneW * kSceneH;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

template <typename T>
__global__ void k_scene_cells(const T* __restrict__ a, const T* __restrict__ b, int w, int h,
                              float* __restrict__ cell_diff) {
    int cxi = blockIdx.x, cyi = blockIdx.y;
    int x0 = (int)(((long long)cxi * w) / kSceneW), x1 = (int)(((long long)(cxi + 1) * w) / kSceneW);
    int y0 = (int)(((long long)cyi * h) / kSceneH), y1 = (int)(((long long)(cyi + 1) * h) / kSceneH);
    if (x1 <= x0) x1 = min(x0 + 1, w);
    if (y1 <= y0) y1 = min(y0 + 1, h);
    if (x1 <= x0) x0 = x1 - 1;
    if (y1 <= y0) y0 = y1 - 1;
    int bw = x1 - x0, bh = y1 - y0;
    int n = bw * bh;
    size_t plane = (size_t)w * h;
    float sa = 0.f, sb = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int px = x0 + i % bw, py = y0 + i / bw;
        size_t o = (size_t)py * w + px;
        sa += 0.299f * ld<T>(a + o) + 0.587f * ld<T>(a + plane + o) + 0.114f * ld<T>(a + 2 * plane + o);
        sb += 0.299f * ld<T>(b + o) + 0.587f * ld<T>(b + plane + o) + 0.114f * ld<T>(b + 2 * plane + o);
    }
    sa = warp_sum(sa);
    sb = warp_sum(sb);
    __shared__ float sha[32], shb[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) {
        sha[wid] = sa;
        shb[wid] = sb;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ta = 0.f, tb = 0.f;
        for (int i = 0; i < (int)(blockDim.x >> 5); ++i) {
            ta += sha[i];
            tb += shb[i];
        }
        cell_diff[cyi * kSceneW + cxi] = fabsf(ta - tb) / (float)n;
    }
}

__global__ void k_scene_reduce(const float* __restrict__ cell_diff, float* __restrict__ out) {
    float s = 0.f;
    for (int i = threadIdx.x; i < kSceneCells; i += blockDim.x) s += cell_diff[i];
    s = warp_sum(s);
    __shared__ float sh[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sh[wid] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < (int)(blockDim.x >> 5); ++i) t += sh[i];
        *out = t / (float)kSceneCells;
    }
}

// ------------------------------------------------------------------------------------------
// Host launchers
// ------------------------------------------------------------------------------------------
static inline dim3 grid2d(int w, int h, dim3 blk, int z = 1) {
    return dim3((w + blk.x - 1) / blk.x, (h + blk.y - 1) / blk.y, z);
}
static const dim3 kBlk(32, 8);

ColorCoef make_coef(int format, int matrix, int range) {
    ColorCoef c{};
    switch (matrix) {
        case FB_MATRIX_BT601: c.kr = 0.299f; c.kb = 0.114f; break;
        case FB_MATRIX_BT709: c.kr = 0.2126f; c.kb = 0.0722f; break;
        default: c.kr = 0.2627f; c.kb = 0.0593f; break;  // BT.2020 non-constant luminance
    }
    c.p010 = (format == FB_FMT_P010);
    float s = c.p010 ? 4.f : 1.f;  // 10-bit codes are 4x the 8-bit ones
    c.max_code = c.p010 ? 1023.f : 255.f;
    if (range == FB_RANGE_FULL) {
        c.y_off = 0.f;
        c.y_rng = c.max_code;
        c.c_off = 128.f * s;
        c.c_rng = c.max_code;
    } else {
        c.y_off = 16.f * s;
        c.y_rng = 219.f * s;
        c.c_off = 128.f * s;
        c.c_rng = 224.f * s;
    }
    return c;
}

cudaError_t yuv_to_rgb(const fb_yuv_frame_t& f, const ColorCoef& c, void* dst, int dtype, int dw, int dh,
                       cudaStream_t s) {
    dim3 g = grid2d(dw, dh, kBlk);
    auto* l = reinterpret_cast<const uint8_t*>(f.luma);
    auto* ch = reinterpret_cast<const uint8_t*>(f.chroma);
#define FB_LAUNCH(T, P)                                                                                       \
    k_yuv_to_rgb<T, P><<<g, kBlk, 0, s>>>(l, f.luma_pitch, ch, f.chroma_pitch, f.width, f.height, c,           \
                                           static_cast<T*>(dst), dw, dh)
    if (dtype == FB_DTYPE_FP16) {
        if (c.p010) FB_LAUNCH(__half, true); else FB_LAUNCH(__half, false);
    } else {
        if (c.p010) FB_LAUNCH(float, true); else FB_LAUNCH(float, false);
    }
#undef FB_LAUNCH
    return cudaGetLastError();
}

cudaError_t rgb_to_yuv(const void* src, int dtype, int sw, int sh, const fb_yuv_frame_t& f, const ColorCoef& c,
                       cudaStream_t s) {
    int cw = (f.width + 1) / 2, ch = (f.height + 1) / 2;
    dim3 g = grid2d(cw, ch, kBlk);
    auto* l = reinterpret_cast<uint8_t*>(f.luma);
    auto* cp = reinterpret_cast<uint8_t*>(f.chroma);
#define FB_LAUNCH(T, P)                                                                                  \
    k_rgb_to_yuv<T, P><<<g, kBlk, 0, s>>>(static_cast<const T*>(src), sw, sh, l, f.luma_pitch, cp,       \
                                            f.chroma_pitch, f.width, f.height, c)
    if (dtype == FB_DTYPE_FP16) {
        if (c.p010) FB_LAUNCH(__half, true); else FB_LAUNCH(__half, false);
    } else {
        if (c.p010) FB_LAUNCH(float, true); else FB_LAUNCH(float, false);
    }
#undef FB_LAUNCH
    return cudaGetLastError();
}

cudaError_t region_copy(const void* src, int dtype, int sw, int sh, int x, int y, void* dst, int dw, int dh,
                        cudaStream_t s) {
    dim3 g = grid2d(dw, dh, kBlk, 3);
    if (dtype == FB_DTYPE_FP16)
        k_region_copy<uint16_t><<<g, kBlk, 0, s>>>(static_cast<const uint16_t*>(src), sw, sh, x, y,
                                                   static_cast<uint16_t*>(dst), dw, dh);
    else
        k_region_copy<uint32_t><<<g, kBlk, 0, s>>>(static_cast<const uint32_t*>(src), sw, sh, x, y,
                                                   static_cast<uint32_t*>(dst), dw, dh);
    return cudaGetLastError();
}

cudaError_t convert(const void* src, int sdt, void* dst, int ddt, size_t n, cudaStream_t s) {
    if (sdt == ddt) {
        if (src == dst) return cudaSuccess;
        return cudaMemcpyAsync(dst, src, n * (sdt == FB_DTYPE_FP16 ? 2 : 4), cudaMemcpyDeviceToDevice, s);
    }
    int blocks = (int)std::min<size_t>((n + 255) / 256, 65535);
    if (sdt == FB_DTYPE_FP32)
        k_convert<float, __half><<<blocks, 256, 0, s>>>(static_cast<const float*>(src), static_cast<__half*>(dst), n);
    else
        k_convert<__half, float><<<blocks, 256, 0, s>>>(static_cast<const __half*>(src), static_cast<float*>(dst), n);
    return cudaGetLastError();
}

cudaError_t tile_merge(const void* tile, int dtype, int tw, int th, float* accum, float* weight, int ow, int oh, int dx,
                       int dy, int feather, int flags, cudaStream_t s) {
    dim3 g = grid2d(tw, th, kBlk);
    if (dtype == FB_DTYPE_FP16)
        k_tile_merge<__half><<<g, kBlk, 0, s>>>(static_cast<const __half*>(tile), tw, th, accum, weight, ow, oh, dx,
                                                dy, feather, flags);
    else
        k_tile_merge<float><<<g, kBlk, 0, s>>>(static_cast<const float*>(tile), tw, th, accum, weight, ow, oh, dx,
                                               dy, feather, flags);
    return cudaGetLastError();
}

cudaError_t tile_finalize(const float* accum, const float* weight, int ow, int oh, void* dst, int dtype,
                          cudaStream_t s) {
    dim3 g = grid2d(ow, oh, kBlk);
    if (dtype == FB_DTYPE_FP16)
        k_tile_finalize<__half><<<g, kBlk, 0, s>>>(accum, weight, ow, oh, static_cast<__half*>(dst));
    else
        k_tile_finalize<float><<<g, kBlk, 0, s>>>(accum, weight, ow, oh, static_cast<float*>(dst));
    return cudaGetLastError();
}

// Scratch for the scene score (cell diffs + result). Guarded by a mutex that is held until the stream has been
// synchronized, so concurrent callers cannot clobber each other's partial results.
static std::mutex g_scene_mu;
static float* g_scene_scratch = nullptr;  // kSceneCells + 1 floats (device)

cudaError_t scene_score(const void* a, const void* b, int dtype, int w, int h, float* out, cudaStream_t s) {
    std::lock_guard<std::mutex> lk(g_scene_mu);
    if (!g_scene_scratch) {
        cudaError_t e = cudaMalloc(&g_scene_scratch, (kSceneCells + 1) * sizeof(float));
        if (e != cudaSuccess) {
            g_scene_scratch = nullptr;
            return e;
        }
    }
    float* cells = g_scene_scratch;
    float* res = g_scene_scratch + kSceneCells;
    dim3 g(kSceneW, kSceneH);
    if (dtype == FB_DTYPE_FP16)
        k_scene_cells<__half><<<g, 128, 0, s>>>(static_cast<const __half*>(a), static_cast<const __half*>(b), w, h,
                                                cells);
    else
        k_scene_cells<float><<<g, 128, 0, s>>>(static_cast<const float*>(a), static_cast<const float*>(b), w, h, cells);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    k_scene_reduce<<<1, 256, 0, s>>>(cells, res);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(out, res, sizeof(float), cudaMemcpyDeviceToHost, s);
    if (e != cudaSuccess) return e;
    return cudaStreamSynchronize(s);
}

void scene_release_scratch() noexcept {
    std::lock_guard<std::mutex> lk(g_scene_mu);
    if (g_scene_scratch) {
        cudaFree(g_scene_scratch);
        g_scene_scratch = nullptr;
    }
}

}  // namespace fb::k
