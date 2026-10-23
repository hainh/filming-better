// Device, stream and memory entry points.
#include <atomic>
#include <cstring>
#include <mutex>

#include "common.h"

namespace fb {

namespace {
std::mutex g_init_mutex;
std::atomic<bool> g_initialized{false};
std::atomic<int> g_device{0};
std::atomic<unsigned> g_generation{0};
}  // namespace

bool is_initialized() { return g_initialized.load(std::memory_order_acquire); }
int current_device_index() { return g_device.load(); }

void require_ready() {
    if (!is_initialized()) fail(FB_ERR_NOT_INITIALIZED, "fb_init has not been called (or fb_shutdown was called)");
    // Runtime-API "current device" is thread-local: make sure every calling thread (Rust worker threads
    // start with device 0) uses the device chosen in fb_init. Cached per thread to keep this cheap.
    thread_local unsigned tl_gen = 0;
    thread_local int tl_dev = -1;
    unsigned gen = g_generation.load(std::memory_order_acquire);
    int dev = g_device.load();
    if (tl_gen != gen || tl_dev != dev) {
        check_cuda(cudaSetDevice(dev), "cudaSetDevice");
        tl_gen = gen;
        tl_dev = dev;
    }
}

static void fill_gpu_info(int idx, fb_gpu_info_t* out) {
    cudaDeviceProp p{};
    check_cuda(cudaGetDeviceProperties(&p, idx), "cudaGetDeviceProperties");
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->name, p.name, sizeof(out->name) - 1);
    out->device_index = idx;
    out->cc_major = p.major;
    out->cc_minor = p.minor;
    int drv = 0, rt = 0;
    (void)cudaDriverGetVersion(&drv);
    (void)cudaRuntimeGetVersion(&rt);
    out->driver_version = drv;
    out->runtime_version = rt;
    out->multiprocessor_count = p.multiProcessorCount;
    out->total_mem = static_cast<uint64_t>(p.totalGlobalMem);
}

}  // namespace fb

using namespace fb;

extern "C" {

FB_API int fb_init(int device_index, int* out_driver_version) {
    return guard([&] {
        if (out_driver_version) *out_driver_version = 0;
        std::lock_guard<std::mutex> lk(g_init_mutex);
        int drv = 0;
        cudaError_t e = cudaDriverGetVersion(&drv);
        if (e != cudaSuccess) {
            (void)cudaGetLastError();
            drv = 0;
        }
        if (out_driver_version) *out_driver_version = drv;
        if (drv == 0) fail(FB_ERR_NO_GPU, "no NVIDIA driver found (nvcuda.dll is missing or not functional)");
        if (drv < FB_MIN_DRIVER_VERSION) {
            fail(FB_ERR_DRIVER_TOO_OLD, "NVIDIA driver supports CUDA " + std::to_string(drv / 1000) + "." +
                                            std::to_string((drv % 1000) / 10) + ", but CUDA " +
                                            std::to_string(FB_MIN_DRIVER_VERSION / 1000) + "." +
                                            std::to_string((FB_MIN_DRIVER_VERSION % 1000) / 10) +
                                            " or newer is required; please update the NVIDIA driver");
        }
        int count = 0;
        e = cudaGetDeviceCount(&count);
        if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver || (e == cudaSuccess && count == 0)) {
            (void)cudaGetLastError();
            fail(FB_ERR_NO_GPU, "no CUDA-capable NVIDIA GPU found");
        }
        check_cuda(e, "cudaGetDeviceCount");
        if (device_index < 0 || device_index >= count) {
            fail(FB_ERR_INVALID_ARG, "device_index " + std::to_string(device_index) + " out of range (0.." +
                                         std::to_string(count - 1) + ")");
        }
        check_cuda(cudaSetDevice(device_index), "cudaSetDevice");
        // Force primary-context creation now so failures surface here and not mid-pipeline.
        check_cuda(cudaFree(nullptr), "cudaFree(0) [primary context init]");
        cudaDeviceProp p{};
        check_cuda(cudaGetDeviceProperties(&p, device_index), "cudaGetDeviceProperties");
        if (p.major < 7 || (p.major == 7 && p.minor < 5)) {
            fail(FB_ERR_UNSUPPORTED, std::string("GPU ") + p.name + " (sm_" + std::to_string(p.major) +
                                         std::to_string(p.minor) + ") is older than Turing (sm_75)");
        }
        g_device.store(device_index);
        g_generation.fetch_add(1, std::memory_order_acq_rel);
        g_initialized.store(true, std::memory_order_release);
    });
}

FB_API int fb_shutdown(void) {
    return guard([&] {
        std::lock_guard<std::mutex> lk(g_init_mutex);
        if (!is_initialized()) return;
        require_ready();
        // Wait for outstanding work, then drop our scratch. The primary context is deliberately NOT reset
        // (cudaDeviceReset): FFmpeg's hw device shares it and may still be alive.
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        kernels_release_scratch();
        g_initialized.store(false, std::memory_order_release);
    });
}

FB_API int fb_device_count(int* out_count) {
    return guard([&] {
        require(out_count != nullptr, "out_count is NULL");
        *out_count = 0;
        int n = 0;
        cudaError_t e = cudaGetDeviceCount(&n);
        if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver) {
            (void)cudaGetLastError();
            return;  // 0 devices, not an error
        }
        check_cuda(e, "cudaGetDeviceCount");
        *out_count = n;
    });
}

FB_API int fb_gpu_info(int device_index, fb_gpu_info_t* out) {
    return guard([&] {
        require(out != nullptr, "out is NULL");
        int n = 0;
        cudaError_t e = cudaGetDeviceCount(&n);
        if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver) {
            (void)cudaGetLastError();
            fail(FB_ERR_NO_GPU, "no CUDA-capable NVIDIA GPU found");
        }
        check_cuda(e, "cudaGetDeviceCount");
        if (device_index < 0 || device_index >= n) fail(FB_ERR_INVALID_ARG, "device_index out of range");
        fill_gpu_info(device_index, out);
    });
}

FB_API int fb_mem_info(size_t* out_free, size_t* out_total) {
    return guard([&] {
        require(out_free && out_total, "out_free/out_total is NULL");
        require_ready();
        check_cuda(cudaMemGetInfo(out_free, out_total), "cudaMemGetInfo");
    });
}

/* ---- streams ---- */
FB_API int fb_stream_create(fb_stream** out) {
    return guard([&] {
        require(out != nullptr, "out is NULL");
        *out = nullptr;
        require_ready();
        cudaStream_t s = nullptr;
        // Non-blocking: must not implicitly synchronize with the legacy default stream (FFmpeg uses it).
        check_cuda(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate");
        *out = new fb_stream{s};
    });
}

FB_API int fb_stream_destroy(fb_stream* s) {
    return guard([&] {
        if (!s) return;
        require_ready();
        cudaError_t e = cudaStreamDestroy(s->s);  // pending work completes asynchronously
        delete s;
        check_cuda(e, "cudaStreamDestroy");
    });
}

FB_API int fb_stream_sync(fb_stream* s) {
    return guard([&] {
        require_ready();
        check_cuda(cudaStreamSynchronize(raw_stream(s)), "cudaStreamSynchronize");
    });
}

FB_API void* fb_stream_raw(fb_stream* s) { return s ? static_cast<void*>(s->s) : nullptr; }

/* ---- memory ---- */
FB_API int fb_alloc(size_t bytes, fb_devptr* out) {
    return guard([&] {
        require(out != nullptr, "out is NULL");
        *out = 0;
        require(bytes > 0, "fb_alloc: bytes must be > 0");
        require_ready();
        void* p = nullptr;
        cudaError_t e = cudaMalloc(&p, bytes);
        if (e != cudaSuccess) {
            (void)cudaGetLastError();
            fail(map_cuda_error(e), "cudaMalloc(" + std::to_string(bytes) + " bytes) failed: " + cudaGetErrorString(e));
        }
        *out = reinterpret_cast<fb_devptr>(p);
    });
}

FB_API int fb_free(fb_devptr p) {
    return guard([&] {
        if (p == 0) return;
        require_ready();
        check_cuda(cudaFree(reinterpret_cast<void*>(p)), "cudaFree");
    });
}

FB_API int fb_memcpy(uint64_t dst, uint64_t src, size_t bytes, fb_stream* s) {
    return guard([&] {
        if (bytes == 0) return;
        require(dst != 0 && src != 0, "fb_memcpy: NULL pointer");
        require_ready();
        check_cuda(cudaMemcpyAsync(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), bytes,
                                   cudaMemcpyDefault, raw_stream(s)),
                   "cudaMemcpyAsync");
    });
}

FB_API int fb_memcpy2d(uint64_t dst, size_t dst_pitch, uint64_t src, size_t src_pitch, size_t width_bytes,
                       size_t height, fb_stream* s) {
    return guard([&] {
        if (width_bytes == 0 || height == 0) return;
        require(dst != 0 && src != 0, "fb_memcpy2d: NULL pointer");
        require(dst_pitch >= width_bytes && src_pitch >= width_bytes, "fb_memcpy2d: pitch < width_bytes");
        require_ready();
        check_cuda(cudaMemcpy2DAsync(reinterpret_cast<void*>(dst), dst_pitch, reinterpret_cast<const void*>(src),
                                     src_pitch, width_bytes, height, cudaMemcpyDefault, raw_stream(s)),
                   "cudaMemcpy2DAsync");
    });
}

FB_API int fb_memset(fb_devptr dst, int value, size_t bytes, fb_stream* s) {
    return guard([&] {
        if (bytes == 0) return;
        require(dst != 0, "fb_memset: NULL pointer");
        require_ready();
        check_cuda(cudaMemsetAsync(reinterpret_cast<void*>(dst), value, bytes, raw_stream(s)), "cudaMemsetAsync");
    });
}

}  // extern "C"
