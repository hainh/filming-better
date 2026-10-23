/*
 * fb_native.h — the ONLY boundary between the Rust app and the C++/CUDA core.
 *
 * Rules:
 *  - Plain C types only. No exceptions cross this boundary: every function is
 *    wrapped in try/catch and returns an int status (FB_OK == 0).
 *  - On error, fb_last_error() returns a thread-local, human-readable message
 *    (valid until the next fb_* call on the same thread).
 *  - Ownership: whoever allocates frees. FFmpeg frames belong to Rust; buffers
 *    from fb_alloc and engines/streams belong to fb_native.
 *  - All CUDA work uses the device's PRIMARY context (runtime API), so it is
 *    shared with FFmpeg's AV_CUDA_USE_PRIMARY_CONTEXT hw device.
 *  - Device pointers are passed as fb_devptr (== CUdeviceptr, 64-bit).
 *  - "Tensor" = planar RGB, batch 1, layout CHW contiguous: 3 planes of w*h
 *    elements, R then G then B, values nominally in [0,1].
 */
#ifndef FB_NATIVE_H
#define FB_NATIVE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(FB_NATIVE_BUILD)
#    define FB_API __declspec(dllexport)
#  else
#    define FB_API __declspec(dllimport)
#  endif
#else
#  define FB_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define FB_API_VERSION 1
/* Minimum CUDA driver version as reported by cuDriverGetVersion / cudaDriverGetVersion
 * (1000*major + 10*minor). App is built against CUDA 12.9 runtime -> 12.8+ driver needed
 * (12080 is the floor so RTX 50 / sm_120 works). */
#define FB_MIN_DRIVER_VERSION 12080

typedef uint64_t fb_devptr;
typedef struct fb_stream fb_stream; /* opaque: wraps a cudaStream_t */
typedef struct fb_engine fb_engine; /* opaque: TRT runtime + engine + execution context */

/* ---- status codes (Rust maps these to NativeError / exit codes) ---- */
enum {
    FB_OK = 0,
    FB_ERR_INVALID_ARG = 1,
    FB_ERR_NO_GPU = 2,          /* no CUDA-capable NVIDIA GPU / nvcuda.dll missing */
    FB_ERR_DRIVER_TOO_OLD = 3,  /* driver < FB_MIN_DRIVER_VERSION */
    FB_ERR_OUT_OF_MEMORY = 4,   /* VRAM exhausted */
    FB_ERR_CUDA = 5,            /* any other CUDA error */
    FB_ERR_TRT_LOAD = 6,        /* TensorRT lean runtime DLL could not be loaded */
    FB_ERR_ENGINE = 7,          /* engine deserialize / shape / enqueue failure */
    FB_ERR_NOT_INITIALIZED = 8, /* fb_init not called */
    FB_ERR_UNSUPPORTED = 9,     /* unsupported format / dtype / arch */
    FB_ERR_IO = 10,             /* file read error (engine file) */
    FB_ERR_INTERNAL = 11
};

/* ---- enums ---- */
enum { FB_DTYPE_FP32 = 0, FB_DTYPE_FP16 = 1 };
enum { FB_FMT_NV12 = 0, FB_FMT_P010 = 1 }; /* P010: 16-bit little endian, 10 significant MSBs */
enum { FB_MATRIX_BT601 = 0, FB_MATRIX_BT709 = 1, FB_MATRIX_BT2020NC = 2 };
enum { FB_RANGE_LIMITED = 0, FB_RANGE_FULL = 1 };

/* ---- structs ---- */
typedef struct fb_gpu_info_t {
    char name[256];
    int device_index;
    int cc_major;
    int cc_minor;
    int driver_version;   /* e.g. 13010 */
    int runtime_version;  /* CUDA runtime fb_native was built with, e.g. 12090 */
    int multiprocessor_count;
    uint64_t total_mem;   /* bytes */
} fb_gpu_info_t;

/* A semi-planar YUV 4:2:0 image in VRAM (exactly how FFmpeg CUDA frames look:
 * AVFrame.data[0]/linesize[0] = luma, data[1]/linesize[1] = interleaved UV). */
typedef struct fb_yuv_frame_t {
    fb_devptr luma;
    size_t luma_pitch;   /* bytes */
    fb_devptr chroma;
    size_t chroma_pitch; /* bytes */
    int width;           /* visible luma width  */
    int height;          /* visible luma height */
    int format;          /* FB_FMT_* */
} fb_yuv_frame_t;

typedef struct fb_tensor_desc_t {
    char name[128];
    int is_input;        /* 1 = input, 0 = output */
    int dtype;           /* FB_DTYPE_*; FB_ERR_UNSUPPORTED from load if engine uses something else */
    int nb_dims;
    int64_t dims[8];     /* -1 for dynamic dims */
    /* optimization profile 0 (inputs only; zeros for outputs) */
    int64_t min_dims[8];
    int64_t opt_dims[8];
    int64_t max_dims[8];
} fb_tensor_desc_t;

/* ======================= device ======================= */
FB_API int fb_api_version(void);
FB_API const char* fb_last_error(void);

/* Checks driver (FB_ERR_DRIVER_TOO_OLD if too old; *out_driver_version is always filled
 * when a driver exists, 0 otherwise), selects the device, activates its primary context.
 * Safe to call more than once. */
FB_API int fb_init(int device_index, int* out_driver_version);
FB_API int fb_shutdown(void);
FB_API int fb_device_count(int* out_count);
FB_API int fb_gpu_info(int device_index, fb_gpu_info_t* out);
FB_API int fb_mem_info(size_t* out_free, size_t* out_total); /* cudaMemGetInfo, current device */

/* ======================= streams & memory ======================= */
FB_API int fb_stream_create(fb_stream** out);
FB_API int fb_stream_destroy(fb_stream* s);          /* NULL is a no-op */
FB_API int fb_stream_sync(fb_stream* s);              /* NULL = legacy default stream */
FB_API void* fb_stream_raw(fb_stream* s);             /* cudaStream_t, for diagnostics */

FB_API int fb_alloc(size_t bytes, fb_devptr* out);
FB_API int fb_free(fb_devptr p);                       /* 0 is a no-op */
/* Copies use cudaMemcpyDefault (UVA): src/dst may be host or device addresses.
 * Host addresses are passed as uint64 (cast from pointer). Async on stream s (NULL ok).
 * Pageable host memory makes the copy effectively synchronous - that is fine. */
FB_API int fb_memcpy(uint64_t dst, uint64_t src, size_t bytes, fb_stream* s);
FB_API int fb_memcpy2d(uint64_t dst, size_t dst_pitch, uint64_t src, size_t src_pitch,
                       size_t width_bytes, size_t height, fb_stream* s);
FB_API int fb_memset(fb_devptr dst, int value, size_t bytes, fb_stream* s);

/* ======================= color / layout kernels ======================= */
/* YUV(NV12/P010) -> RGB tensor of size dst_w x dst_h (dst_w >= src->width, dst_h >= src->height).
 * The image goes to the top-left; the pad region (right/bottom) is filled by edge replication. */
FB_API int fb_yuv_to_rgb(const fb_yuv_frame_t* src, int matrix, int range,
                         fb_devptr dst, int dst_dtype, int dst_w, int dst_h, fb_stream* s);

/* RGB tensor (src_w x src_h) -> YUV frame. Writes dst->width x dst->height taken from the
 * top-left of the tensor (i.e. crops pad). Values are clamped to [0,1] first.
 * dst->width <= src_w, dst->height <= src_h. Chroma is the 2x2 average. */
FB_API int fb_rgb_to_yuv(fb_devptr src, int src_dtype, int src_w, int src_h,
                         const fb_yuv_frame_t* dst, int matrix, int range, fb_stream* s);

/* Pad tensor (w x h) into (dst_w x dst_h) with edge replication (top-left anchored). */
FB_API int fb_pad(fb_devptr src, int dtype, int w, int h,
                  fb_devptr dst, int dst_w, int dst_h, fb_stream* s);
/* Copy region [x,x+dst_w) x [y,y+dst_h) of src into dst (crop). Region must be inside src. */
FB_API int fb_crop(fb_devptr src, int dtype, int src_w, int src_h, int x, int y,
                   fb_devptr dst, int dst_w, int dst_h, fb_stream* s);
/* Convert tensor dtype (same dims). */
FB_API int fb_convert(fb_devptr src, int src_dtype, fb_devptr dst, int dst_dtype,
                      int w, int h, fb_stream* s);

/* ======================= tiling ======================= */
/* Extract tile [x,x+tile_w) x [y,y+tile_h) from src (may extend past the edges -> edge replicate). */
FB_API int fb_tile_split(fb_devptr src, int dtype, int src_w, int src_h, int x, int y,
                         fb_devptr tile, int tile_w, int tile_h, fb_stream* s);
/* Accumulate an (already upscaled) tile into FP32 accumulators:
 *   accum: 3 x out_w x out_h FP32, weight: out_w x out_h FP32 (both zeroed by caller first).
 * The tile is placed at (dst_x, dst_y) in output coords; pixels outside the output are ignored.
 * Weight ramps linearly from ~0 to 1 over `feather` px at each tile edge that is NOT on the
 * image border (pass edge flags bitmask: 1=left,2=top,4=right,8=bottom on image border). */
FB_API int fb_tile_merge(fb_devptr tile, int tile_dtype, int tile_w, int tile_h,
                         fb_devptr accum, fb_devptr weight, int out_w, int out_h,
                         int dst_x, int dst_y, int feather, int border_flags, fb_stream* s);
/* dst = accum / max(weight, eps), converted to dst_dtype. */
FB_API int fb_tile_finalize(fb_devptr accum, fb_devptr weight, int out_w, int out_h,
                            fb_devptr dst, int dst_dtype, fb_stream* s);

/* ======================= scene detection ======================= */
/* Downscale both tensors to 64x36 luma (area average) and return mean absolute difference
 * in [0,1] (0 = identical). Synchronizes stream s before returning. */
FB_API int fb_scene_score(fb_devptr a, fb_devptr b, int dtype, int w, int h,
                          float* out_score, fb_stream* s);

/* ======================= TensorRT (lean runtime) ======================= */
/* Optionally point at nvinfer_lean DLL explicitly (UTF-8 path). Otherwise fb_engine_load
 * tries, in order: env FB_TRT_LEAN_DLL, the directory of fb_native.dll, then the default
 * DLL search path. Returns FB_ERR_TRT_LOAD on failure. */
FB_API int fb_trt_set_library(const char* dll_path_utf8);
FB_API int fb_trt_version(int* out_version); /* e.g. 110400 for 11.4.0 */

FB_API int fb_engine_load(const char* path_utf8, fb_engine** out);
FB_API int fb_engine_free(fb_engine* e);      /* NULL is a no-op */
FB_API int fb_engine_io_count(fb_engine* e, int* out_count);
FB_API int fb_engine_io_desc(fb_engine* e, int index, fb_tensor_desc_t* out);
FB_API int fb_engine_set_input_shape(fb_engine* e, const char* name, int nb_dims, const int64_t* dims);
/* Current (resolved) shape of any tensor after input shapes are set. */
FB_API int fb_engine_io_shape(fb_engine* e, const char* name, int* out_nb_dims, int64_t* out_dims /*[8]*/);
FB_API int fb_engine_set_tensor(fb_engine* e, const char* name, fb_devptr ptr);
FB_API int fb_engine_run(fb_engine* e, fb_stream* s); /* enqueueV3, async */

#ifdef __cplusplus
}
#endif
#endif /* FB_NATIVE_H */
