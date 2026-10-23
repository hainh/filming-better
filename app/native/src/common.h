// Internal helpers shared by all fb_native translation units (not part of the public API).
#pragma once
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <utility>

#include "fb_native.h"

namespace fb {

// Exception used internally only; never crosses the C ABI.
struct Error : std::exception {
    int code;
    std::string msg;
    Error(int c, std::string m) : code(c), msg(std::move(m)) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

[[noreturn]] void fail(int code, const std::string& msg);

// Thread-local last-error storage.
void set_last_error(const std::string& msg);
const char* last_error();

// Maps a CUDA error to an fb status code (OOM -> FB_ERR_OUT_OF_MEMORY, no device -> FB_ERR_NO_GPU, ...).
int map_cuda_error(cudaError_t e);
// Throws fb::Error if e != cudaSuccess. `what` names the failing operation.
void check_cuda(cudaError_t e, const char* what);
// Checks cudaGetLastError() after a kernel launch.
void check_launch(const char* what);

// Requires fb_init() to have succeeded, and makes the initialized device current on this thread
// (the runtime API's "current device" is per-thread; Rust worker threads start on device 0).
void require_ready();
bool is_initialized();
int current_device_index();

// Runs f(), converting every exception into a status code + last-error message.
template <class F>
int guard(F&& f) noexcept {
    try {
        set_last_error(std::string());
        f();
        return FB_OK;
    } catch (const Error& e) {
        set_last_error(e.msg);
        return e.code;
    } catch (const std::bad_alloc&) {
        set_last_error("host out of memory");
        return FB_ERR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        set_last_error(std::string("internal error: ") + e.what());
        return FB_ERR_INTERNAL;
    } catch (...) {
        set_last_error("internal error: unknown exception");
        return FB_ERR_INTERNAL;
    }
}

inline void require(bool cond, const char* msg) {
    if (!cond) fail(FB_ERR_INVALID_ARG, msg);
}


inline size_t dtype_size(int dtype) {
    if (dtype == FB_DTYPE_FP32) return 4;
    if (dtype == FB_DTYPE_FP16) return 2;
    fail(FB_ERR_UNSUPPORTED, "unsupported dtype " + std::to_string(dtype));
}

// Release scratch allocations owned by the kernel layer (called from fb_shutdown).
void kernels_release_scratch() noexcept;

}  // namespace fb

struct fb_stream {
    cudaStream_t s;
};

namespace fb {
inline cudaStream_t raw_stream(fb_stream* s) { return s ? s->s : cudaStream_t(0); }
}  // namespace fb
