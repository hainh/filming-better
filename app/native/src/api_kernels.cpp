// Entry points for colour conversion / tiling / scene-score kernels: argument validation + launch.
#include "common.h"
#include "kernels.h"

namespace fb {

void kernels_release_scratch() noexcept { k::scene_release_scratch(); }

namespace {

void check_dtype(int dtype, const char* what) {
    if (dtype != FB_DTYPE_FP32 && dtype != FB_DTYPE_FP16)
        fail(FB_ERR_UNSUPPORTED, std::string(what) + ": unsupported dtype " + std::to_string(dtype));
}

void check_color(int matrix, int range) {
    if (matrix < FB_MATRIX_BT601 || matrix > FB_MATRIX_BT2020NC)
        fail(FB_ERR_INVALID_ARG, "invalid matrix " + std::to_string(matrix));
    if (range != FB_RANGE_LIMITED && range != FB_RANGE_FULL)
        fail(FB_ERR_INVALID_ARG, "invalid range " + std::to_string(range));
}

void check_yuv(const fb_yuv_frame_t* f, const char* what) {
    require(f != nullptr, "yuv frame is NULL");
    require(f->luma != 0 && f->chroma != 0, "yuv frame: NULL plane pointer");
    require(f->width > 0 && f->height > 0, "yuv frame: width/height must be > 0");
    if (f->format != FB_FMT_NV12 && f->format != FB_FMT_P010)
        fail(FB_ERR_UNSUPPORTED, std::string(what) + ": unsupported pixel format " + std::to_string(f->format));
    size_t bps = f->format == FB_FMT_P010 ? 2 : 1;
    require(f->luma_pitch >= (size_t)f->width * bps, "yuv frame: luma_pitch < width * bytes_per_sample");
    require(f->chroma_pitch >= (size_t)((f->width + 1) / 2) * 2 * bps,
            "yuv frame: chroma_pitch < ceil(width/2) * 2 * bytes_per_sample");
}

// Region of >0 size, dimensions within int range such that element counts fit comfortably.
void check_dims(int w, int h, const char* what) {
    if (w <= 0 || h <= 0) fail(FB_ERR_INVALID_ARG, std::string(what) + ": width/height must be > 0");
}

void post_launch(cudaError_t e, const char* what) { check_cuda(e, what); }

}  // namespace
}  // namespace fb

using namespace fb;

extern "C" {

FB_API int fb_yuv_to_rgb(const fb_yuv_frame_t* src, int matrix, int range, fb_devptr dst, int dst_dtype, int dst_w,
                         int dst_h, fb_stream* s) {
    return guard([&] {
        check_yuv(src, "fb_yuv_to_rgb");
        check_color(matrix, range);
        check_dtype(dst_dtype, "fb_yuv_to_rgb");
        require(dst != 0, "dst is NULL");
        require(dst_w >= src->width && dst_h >= src->height, "dst size must be >= source size");
        require_ready();
        auto coef = k::make_coef(src->format, matrix, range);
        post_launch(k::yuv_to_rgb(*src, coef, reinterpret_cast<void*>(dst), dst_dtype, dst_w, dst_h, raw_stream(s)),
                    "yuv_to_rgb kernel");
    });
}

FB_API int fb_rgb_to_yuv(fb_devptr src, int src_dtype, int src_w, int src_h, const fb_yuv_frame_t* dst, int matrix,
                         int range, fb_stream* s) {
    return guard([&] {
        check_yuv(dst, "fb_rgb_to_yuv");
        check_color(matrix, range);
        check_dtype(src_dtype, "fb_rgb_to_yuv");
        require(src != 0, "src is NULL");
        require(dst->width <= src_w && dst->height <= src_h, "dst size must be <= source tensor size");
        require_ready();
        auto coef = k::make_coef(dst->format, matrix, range);
        post_launch(k::rgb_to_yuv(reinterpret_cast<const void*>(src), src_dtype, src_w, src_h, *dst, coef,
                                  raw_stream(s)),
                    "rgb_to_yuv kernel");
    });
}

FB_API int fb_pad(fb_devptr src, int dtype, int w, int h, fb_devptr dst, int dst_w, int dst_h, fb_stream* s) {
    return guard([&] {
        check_dtype(dtype, "fb_pad");
        check_dims(w, h, "fb_pad");
        require(src != 0 && dst != 0, "NULL pointer");
        require(dst_w >= w && dst_h >= h, "fb_pad: dst must be >= src");
        require_ready();
        post_launch(k::region_copy(reinterpret_cast<const void*>(src), dtype, w, h, 0, 0,
                                   reinterpret_cast<void*>(dst), dst_w, dst_h, raw_stream(s)),
                    "pad kernel");
    });
}

FB_API int fb_crop(fb_devptr src, int dtype, int src_w, int src_h, int x, int y, fb_devptr dst, int dst_w, int dst_h,
                   fb_stream* s) {
    return guard([&] {
        check_dtype(dtype, "fb_crop");
        check_dims(src_w, src_h, "fb_crop");
        check_dims(dst_w, dst_h, "fb_crop");
        require(src != 0 && dst != 0, "NULL pointer");
        require(x >= 0 && y >= 0 && (int64_t)x + dst_w <= src_w && (int64_t)y + dst_h <= src_h,
                "fb_crop: region outside source");
        require_ready();
        post_launch(k::region_copy(reinterpret_cast<const void*>(src), dtype, src_w, src_h, x, y,
                                   reinterpret_cast<void*>(dst), dst_w, dst_h, raw_stream(s)),
                    "crop kernel");
    });
}

FB_API int fb_convert(fb_devptr src, int src_dtype, fb_devptr dst, int dst_dtype, int w, int h, fb_stream* s) {
    return guard([&] {
        check_dtype(src_dtype, "fb_convert");
        check_dtype(dst_dtype, "fb_convert");
        check_dims(w, h, "fb_convert");
        require(src != 0 && dst != 0, "NULL pointer");
        require_ready();
        post_launch(k::convert(reinterpret_cast<const void*>(src), src_dtype, reinterpret_cast<void*>(dst), dst_dtype,
                               (size_t)3 * w * h, raw_stream(s)),
                    "convert");
    });
}

FB_API int fb_tile_split(fb_devptr src, int dtype, int src_w, int src_h, int x, int y, fb_devptr tile, int tile_w,
                         int tile_h, fb_stream* s) {
    return guard([&] {
        check_dtype(dtype, "fb_tile_split");
        check_dims(src_w, src_h, "fb_tile_split");
        check_dims(tile_w, tile_h, "fb_tile_split");
        require(src != 0 && tile != 0, "NULL pointer");
        require_ready();
        post_launch(k::region_copy(reinterpret_cast<const void*>(src), dtype, src_w, src_h, x, y,
                                   reinterpret_cast<void*>(tile), tile_w, tile_h, raw_stream(s)),
                    "tile_split kernel");
    });
}

FB_API int fb_tile_merge(fb_devptr tile, int tile_dtype, int tile_w, int tile_h, fb_devptr accum, fb_devptr weight,
                         int out_w, int out_h, int dst_x, int dst_y, int feather, int border_flags, fb_stream* s) {
    return guard([&] {
        check_dtype(tile_dtype, "fb_tile_merge");
        check_dims(tile_w, tile_h, "fb_tile_merge");
        check_dims(out_w, out_h, "fb_tile_merge");
        require(tile != 0 && accum != 0 && weight != 0, "NULL pointer");
        require(feather >= 0, "feather must be >= 0");
        require_ready();
        post_launch(k::tile_merge(reinterpret_cast<const void*>(tile), tile_dtype, tile_w, tile_h,
                                  reinterpret_cast<float*>(accum), reinterpret_cast<float*>(weight), out_w, out_h,
                                  dst_x, dst_y, feather, border_flags, raw_stream(s)),
                    "tile_merge kernel");
    });
}

FB_API int fb_tile_finalize(fb_devptr accum, fb_devptr weight, int out_w, int out_h, fb_devptr dst, int dst_dtype,
                            fb_stream* s) {
    return guard([&] {
        check_dtype(dst_dtype, "fb_tile_finalize");
        check_dims(out_w, out_h, "fb_tile_finalize");
        require(accum != 0 && weight != 0 && dst != 0, "NULL pointer");
        require_ready();
        post_launch(k::tile_finalize(reinterpret_cast<const float*>(accum), reinterpret_cast<const float*>(weight),
                                     out_w, out_h, reinterpret_cast<void*>(dst), dst_dtype, raw_stream(s)),
                    "tile_finalize kernel");
    });
}

FB_API int fb_scene_score(fb_devptr a, fb_devptr b, int dtype, int w, int h, float* out_score, fb_stream* s) {
    return guard([&] {
        require(out_score != nullptr, "out_score is NULL");
        *out_score = 0.f;
        check_dtype(dtype, "fb_scene_score");
        check_dims(w, h, "fb_scene_score");
        require(a != 0 && b != 0, "NULL pointer");
        require_ready();
        post_launch(k::scene_score(reinterpret_cast<const void*>(a), reinterpret_cast<const void*>(b), dtype, w, h,
                                   out_score, raw_stream(s)),
                    "scene_score");
    });
}

}  // extern "C"
