// TensorRT lean-runtime loader + logger (internal). No import library is used: nvinfer_lean_<major>.dll is
// LoadLibrary'd and createInferRuntime_INTERNAL / getInferLibVersion are resolved with GetProcAddress.
#pragma once
#include <NvInferRuntime.h>

#include <cstdint>
#include <string>

namespace fb::trt {

struct Lib {
    void* module = nullptr;                               // HMODULE
    void* (*create_runtime)(void* logger, int32_t version) = nullptr;
    int32_t (*lib_version)() = nullptr;
    std::string path_utf8;                                // where it was loaded from
};

// Strict explicit override (fb_trt_set_library). Loads immediately; throws fb::Error(FB_ERR_TRT_LOAD) on failure.
void set_library(const char* path_utf8);
// Loads (once) following the documented search order and returns the loaded library; throws FB_ERR_TRT_LOAD.
const Lib& load_library();

// Application logger passed to TensorRT. Errors are captured per thread; verbosity via env FB_TRT_VERBOSE
// (unset/0: silent, 1: warnings+errors to stderr, 2: everything).
nvinfer1::ILogger& logger();
void clear_captured();                 // forget captured messages of this thread
std::string captured();                // errors/warnings captured on this thread since clear_captured()

std::wstring utf8_to_wide(const char* s);  // throws FB_ERR_INVALID_ARG on invalid UTF-8
std::string wide_to_utf8(const wchar_t* s);

}  // namespace fb::trt
