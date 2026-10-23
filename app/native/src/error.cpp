#include <string>

#include "common.h"

namespace fb {

static thread_local std::string tl_error;

void set_last_error(const std::string& msg) { tl_error = msg; }
const char* last_error() { return tl_error.c_str(); }

void fail(int code, const std::string& msg) { throw Error(code, msg); }

int map_cuda_error(cudaError_t e) {
    switch (e) {
        case cudaSuccess: return FB_OK;
        case cudaErrorMemoryAllocation: return FB_ERR_OUT_OF_MEMORY;
        case cudaErrorNoDevice:
        case cudaErrorInsufficientDriver:
        case cudaErrorNoKernelImageForDevice:
            return e == cudaErrorNoKernelImageForDevice ? FB_ERR_UNSUPPORTED : FB_ERR_NO_GPU;
        case cudaErrorInvalidValue: return FB_ERR_CUDA;
        default: return FB_ERR_CUDA;
    }
}

void check_cuda(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return;
    (void)cudaGetLastError();  // clear non-sticky error state
    std::string m = std::string(what) + ": " + cudaGetErrorName(e) + " (" + cudaGetErrorString(e) + ")";
    fail(map_cuda_error(e), m);
}

void check_launch(const char* what) { check_cuda(cudaGetLastError(), what); }

}  // namespace fb

extern "C" {
FB_API const char* fb_last_error(void) { return fb::last_error(); }
FB_API int fb_api_version(void) { return FB_API_VERSION; }
}
