// fb_engine_* / fb_trt_* entry points (TensorRT lean runtime).
//
// Threading: an fb_engine has no internal locking and no thread affinity. It may be created on one thread and used
// on another, but never by two threads at the same time (external synchronization required).
#include <NvInferRuntime.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "trt_runtime.h"

struct fb_engine {
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* ctx = nullptr;
    std::vector<fb_tensor_desc_t> descs;
    std::vector<char> addr_set;  // per IO tensor: setTensorAddress called
    ~fb_engine() {
        // Order matters: context -> engine -> runtime. Virtual destructors dispatch into nvinfer_lean.
        delete ctx;
        delete engine;
        delete runtime;
    }
};

using namespace fb;

namespace {

int to_fb_dtype(nvinfer1::DataType t, const char* name) {
    if (t == nvinfer1::DataType::kFLOAT) return FB_DTYPE_FP32;
    if (t == nvinfer1::DataType::kHALF) return FB_DTYPE_FP16;
    fail(FB_ERR_UNSUPPORTED,
         std::string("engine tensor '") + name + "' uses unsupported dtype " + std::to_string(static_cast<int>(t)) +
             " (only FP32/FP16 supported)");
}

std::string with_trt_log(const std::string& msg) {
    std::string c = trt::captured();
    return c.empty() ? msg : msg + " [TensorRT: " + c + "]";
}

void read_file(const char* path_utf8, std::vector<char>& out) {
    std::wstring wpath = trt::utf8_to_wide(path_utf8);
    FILE* f = _wfopen(wpath.c_str(), L"rb");
    if (!f) {
        char buf[128];
        strerror_s(buf, sizeof(buf), errno);
        fail(FB_ERR_IO, std::string("cannot open engine file '") + path_utf8 + "': " + buf);
    }
    struct Closer {
        FILE* f;
        ~Closer() { std::fclose(f); }
    } closer{f};
    if (_fseeki64(f, 0, SEEK_END) != 0) fail(FB_ERR_IO, std::string("cannot seek in engine file '") + path_utf8 + "'");
    long long sz = _ftelli64(f);
    if (sz <= 0) fail(FB_ERR_IO, std::string("engine file '") + path_utf8 + "' is empty or unreadable");
    _fseeki64(f, 0, SEEK_SET);
    out.resize(static_cast<size_t>(sz));
    size_t got = 0;
    while (got < out.size()) {
        size_t n = std::fread(out.data() + got, 1, out.size() - got, f);
        if (n == 0) fail(FB_ERR_IO, std::string("short read on engine file '") + path_utf8 + "'");
        got += n;
    }
}

fb_engine* checked(fb_engine* e) {
    require(e != nullptr, "engine is NULL");
    return e;
}

int find_tensor(const fb_engine* e, const char* name) {
    require(name != nullptr, "tensor name is NULL");
    for (size_t i = 0; i < e->descs.size(); ++i)
        if (std::strcmp(e->descs[i].name, name) == 0) return static_cast<int>(i);
    fail(FB_ERR_INVALID_ARG, std::string("engine has no tensor named '") + name + "'");
}

void fill_dims(int64_t* dst, const nvinfer1::Dims& d) {
    for (int i = 0; i < 8; ++i) dst[i] = (i < d.nbDims) ? d.d[i] : 0;
}

void build_descs(fb_engine* e) {
    using namespace nvinfer1;
    int n = e->engine->getNbIOTensors();
    int nprof = e->engine->getNbOptimizationProfiles();
    for (int i = 0; i < n; ++i) {
        const char* name = e->engine->getIOTensorName(i);
        require(name && std::strlen(name) < sizeof(fb_tensor_desc_t::name), "engine tensor name too long");
        fb_tensor_desc_t d;
        std::memset(&d, 0, sizeof(d));
        std::strncpy(d.name, name, sizeof(d.name) - 1);
        d.is_input = e->engine->getTensorIOMode(name) == TensorIOMode::kINPUT ? 1 : 0;
        d.dtype = to_fb_dtype(e->engine->getTensorDataType(name), name);
        Dims shape = e->engine->getTensorShape(name);
        if (shape.nbDims < 0 || shape.nbDims > 8) fail(FB_ERR_ENGINE, std::string("bad rank for tensor ") + name);
        d.nb_dims = shape.nbDims;
        fill_dims(d.dims, shape);
        if (d.is_input) {
            if (nprof > 0) {
                fill_dims(d.min_dims, e->engine->getProfileShape(name, 0, OptProfileSelector::kMIN));
                fill_dims(d.opt_dims, e->engine->getProfileShape(name, 0, OptProfileSelector::kOPT));
                fill_dims(d.max_dims, e->engine->getProfileShape(name, 0, OptProfileSelector::kMAX));
            } else {
                fill_dims(d.min_dims, shape);
                fill_dims(d.opt_dims, shape);
                fill_dims(d.max_dims, shape);
            }
        }
        e->descs.push_back(d);
    }
    e->addr_set.assign(e->descs.size(), 0);
}

}  // namespace

extern "C" {

FB_API int fb_trt_set_library(const char* dll_path_utf8) {
    return guard([&] { trt::set_library(dll_path_utf8); });
}

FB_API int fb_trt_version(int* out_version) {
    return guard([&] {
        require(out_version != nullptr, "out_version is NULL");
        *out_version = 0;
        const trt::Lib& lib = trt::load_library();
        *out_version = lib.lib_version();
    });
}

FB_API int fb_engine_load(const char* path_utf8, fb_engine** out) {
    return guard([&] {
        require(out != nullptr, "out is NULL");
        *out = nullptr;
        require(path_utf8 != nullptr && *path_utf8 != 0, "engine path is NULL or empty");
        require_ready();
        std::vector<char> blob;
        read_file(path_utf8, blob);
        const trt::Lib& lib = trt::load_library();
        trt::clear_captured();

        std::unique_ptr<fb_engine> e(new fb_engine());
        e->runtime = static_cast<nvinfer1::IRuntime*>(
            lib.create_runtime(static_cast<void*>(&trt::logger()), NV_TENSORRT_VERSION));
        if (!e->runtime) fail(FB_ERR_TRT_LOAD, with_trt_log("createInferRuntime failed"));
        // Engines are built with EXCLUDE_LEAN_RUNTIME (pure data, no embedded host code): host code stays disabled.
        e->engine = e->runtime->deserializeCudaEngine(blob.data(), blob.size());
        if (!e->engine)
            fail(FB_ERR_ENGINE, with_trt_log(std::string("cannot deserialize engine '") + path_utf8 + "'"));
        build_descs(e.get());
        e->ctx = e->engine->createExecutionContext();
        if (!e->ctx) fail(FB_ERR_ENGINE, with_trt_log("createExecutionContext failed"));
        *out = e.release();
    });
}

FB_API int fb_engine_free(fb_engine* e) {
    return guard([&] {
        if (!e) return;
        // The destructors release GPU memory: make sure the right device is current on this thread.
        if (is_initialized()) require_ready();
        delete e;
    });
}

FB_API int fb_engine_io_count(fb_engine* e, int* out_count) {
    return guard([&] {
        require(out_count != nullptr, "out_count is NULL");
        *out_count = static_cast<int>(checked(e)->descs.size());
    });
}

FB_API int fb_engine_io_desc(fb_engine* e, int index, fb_tensor_desc_t* out) {
    return guard([&] {
        require(out != nullptr, "out is NULL");
        checked(e);
        require(index >= 0 && index < static_cast<int>(e->descs.size()), "io index out of range");
        *out = e->descs[static_cast<size_t>(index)];
    });
}

FB_API int fb_engine_set_input_shape(fb_engine* e, const char* name, int nb_dims, const int64_t* dims) {
    return guard([&] {
        checked(e);
        require(dims != nullptr && nb_dims >= 1 && nb_dims <= 8, "invalid dims");
        int idx = find_tensor(e, name);
        const fb_tensor_desc_t& d = e->descs[static_cast<size_t>(idx)];
        require(d.is_input != 0, "set_input_shape: tensor is not an input");
        require(nb_dims == d.nb_dims, "set_input_shape: rank mismatch");
        for (int i = 0; i < nb_dims; ++i) {
            if (dims[i] < d.min_dims[i] || dims[i] > d.max_dims[i]) {
                fail(FB_ERR_ENGINE, std::string("input '") + name + "' dim " + std::to_string(i) + " = " +
                                        std::to_string(dims[i]) + " is outside the engine profile [" +
                                        std::to_string(d.min_dims[i]) + ", " + std::to_string(d.max_dims[i]) + "]");
            }
        }
        require_ready();
        trt::clear_captured();
        nvinfer1::Dims nd;
        nd.nbDims = nb_dims;
        for (int i = 0; i < 8; ++i) nd.d[i] = i < nb_dims ? dims[i] : 0;
        if (!e->ctx->setInputShape(name, nd))
            fail(FB_ERR_ENGINE, with_trt_log(std::string("setInputShape failed for '") + name + "'"));
    });
}

FB_API int fb_engine_io_shape(fb_engine* e, const char* name, int* out_nb_dims, int64_t* out_dims) {
    return guard([&] {
        require(out_nb_dims != nullptr && out_dims != nullptr, "NULL output");
        checked(e);
        find_tensor(e, name);
        require_ready();
        nvinfer1::Dims d = e->ctx->getTensorShape(name);
        if (d.nbDims < 0) fail(FB_ERR_ENGINE, std::string("getTensorShape failed for '") + name + "'");
        *out_nb_dims = d.nbDims;
        for (int i = 0; i < 8; ++i) out_dims[i] = i < d.nbDims ? d.d[i] : 0;
    });
}

FB_API int fb_engine_set_tensor(fb_engine* e, const char* name, fb_devptr ptr) {
    return guard([&] {
        checked(e);
        int idx = find_tensor(e, name);
        require(ptr != 0, "tensor address is NULL");
        require_ready();
        trt::clear_captured();
        if (!e->ctx->setTensorAddress(name, reinterpret_cast<void*>(ptr)))
            fail(FB_ERR_ENGINE, with_trt_log(std::string("setTensorAddress failed for '") + name + "'"));
        e->addr_set[static_cast<size_t>(idx)] = 1;
    });
}

FB_API int fb_engine_run(fb_engine* e, fb_stream* s) {
    return guard([&] {
        checked(e);
        require_ready();
        for (size_t i = 0; i < e->descs.size(); ++i) {
            if (!e->addr_set[i])
                fail(FB_ERR_ENGINE, std::string("tensor address not set for '") + e->descs[i].name + "'");
        }
        trt::clear_captured();
        if (!e->ctx->allInputDimensionsSpecified())
            fail(FB_ERR_ENGINE, "fb_engine_run: not all input shapes were set (fb_engine_set_input_shape)");
        if (!e->ctx->enqueueV3(raw_stream(s)))
            fail(FB_ERR_ENGINE, with_trt_log("enqueueV3 failed"));
    });
}

}  // extern "C"
