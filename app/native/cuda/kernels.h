// Host-callable launchers for the CUDA kernels (implemented in kernels.cu). Internal header.
// All launchers are asynchronous on `stream` and return the launch error (cudaGetLastError()).
#pragma once
#include <cuda_runtime_api.h>

#include "fb_native.h"

namespace fb::k {

// Quantization / matrix description for one (format, matrix, range) combination.
struct ColorCoef {
    float kr, kb;        // luma coefficients of the matrix
    float y_off, y_rng;  // code = norm * y_rng + y_off  (limited 8-bit: 16, 219; full: 0, 255)
    float c_off, c_rng;  // chroma likewise (limited 8-bit: 128, 224; full: 128, 255)
    float max_code;      // 255 or 1023
    int p010;            // 1 = 16-bit samples with 10 MSBs significant
};

ColorCoef make_coef(int format, int matrix, int range);

cudaError_t yuv_to_rgb(const fb_yuv_frame_t& src, const ColorCoef& c, void* dst, int dst_dtype, int dst_w, int dst_h,
                       cudaStream_t s);
cudaError_t rgb_to_yuv(const void* src, int src_dtype, int src_w, int src_h, const fb_yuv_frame_t& dst,
                       const ColorCoef& c, cudaStream_t s);

// dst(dw x dh) = src region starting at (x, y), coordinates clamped to the source (edge replicate).
// Used for pad, crop and tile_split. 3 planes.
cudaError_t region_copy(const void* src, int dtype, int sw, int sh, int x, int y, void* dst, int dw, int dh,
                        cudaStream_t s);

cudaError_t convert(const void* src, int src_dtype, void* dst, int dst_dtype, size_t count, cudaStream_t s);

cudaError_t tile_merge(const void* tile, int tile_dtype, int tw, int th, float* accum, float* weight, int out_w,
                       int out_h, int dst_x, int dst_y, int feather, int border_flags, cudaStream_t s);
cudaError_t tile_finalize(const float* accum, const float* weight, int out_w, int out_h, void* dst, int dst_dtype,
                          cudaStream_t s);

// Mean absolute difference of the 64x36 area-downscaled luma of a and b. Synchronizes `s`.
cudaError_t scene_score(const void* a, const void* b, int dtype, int w, int h, float* out_score, cudaStream_t s);
void scene_release_scratch() noexcept;

}  // namespace fb::k
