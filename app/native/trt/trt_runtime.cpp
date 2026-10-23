#include "trt_runtime.h"

#include <NvInferVersion.h>
#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "common.h"

namespace fb::trt {

// ---------------------------------------------------------------------------------------------
// UTF-8 <-> UTF-16
// ---------------------------------------------------------------------------------------------
std::wstring utf8_to_wide(const char* s) {
    if (!s) return {};
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, nullptr, 0);
    if (n <= 0) fail(FB_ERR_INVALID_ARG, "path is not valid UTF-8");
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, w.data(), n);
    w.resize(static_cast<size_t>(n - 1));
    return w;
}

std::string wide_to_utf8(const wchar_t* s) {
    if (!s) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, s, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 0) return {};
    std::string o(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, s, -1, o.data(), n, nullptr, nullptr);
    o.resize(static_cast<size_t>(n - 1));
    return o;
}

// ---------------------------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------------------------
namespace {
thread_local std::string tl_captured;

class FbLogger final : public nvinfer1::ILogger {
public:
    FbLogger() {
        const char* v = std::getenv("FB_TRT_VERBOSE");
        verbosity_ = v ? std::atoi(v) : 0;
    }
    void log(Severity sev, const char* msg) noexcept override {
        try {
            if (sev == Severity::kINTERNAL_ERROR || sev == Severity::kERROR || sev == Severity::kWARNING) {
                if (!tl_captured.empty()) tl_captured += " | ";
                tl_captured += msg ? msg : "";
                if (tl_captured.size() > 4096) tl_captured.resize(4096);
            }
            bool print = (verbosity_ >= 2) || (verbosity_ >= 1 && sev <= Severity::kINFO);
            if (print) std::fprintf(stderr, "[TRT %d] %s\n", static_cast<int>(sev), msg ? msg : "");
        } catch (...) {
        }
    }

private:
    int verbosity_ = 0;
};
}  // namespace

nvinfer1::ILogger& logger() {
    static FbLogger l;
    return l;
}
void clear_captured() { tl_captured.clear(); }
std::string captured() { return tl_captured; }

// ---------------------------------------------------------------------------------------------
// Library loading
// ---------------------------------------------------------------------------------------------
namespace {
std::mutex g_lib_mu;
Lib g_lib;
bool g_loaded = false;
std::wstring g_override;  // explicit path (fb_trt_set_library)

std::wstring lean_dll_name() { return L"nvinfer_lean_" + std::to_wstring(NV_TENSORRT_MAJOR) + L".dll"; }

std::wstring own_module_dir() {
    HMODULE h = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&own_module_dir), &h))
        return {};
    std::vector<wchar_t> buf(32768);
    DWORD n = GetModuleFileNameW(h, buf.data(), static_cast<DWORD>(buf.size()));
    if (n == 0 || n >= buf.size()) return {};
    std::wstring p(buf.data(), n);
    size_t slash = p.find_last_of(L"\\/");
    return slash == std::wstring::npos ? std::wstring() : p.substr(0, slash);
}

std::string win_error_text(DWORD code) {
    wchar_t* msg = nullptr;
    DWORD n = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                             nullptr, code, 0, reinterpret_cast<LPWSTR>(&msg), 0, nullptr);
    std::string s = "error " + std::to_string(code);
    if (n && msg) {
        std::string t = wide_to_utf8(msg);
        while (!t.empty() && (t.back() == '\n' || t.back() == '\r' || t.back() == ' ')) t.pop_back();
        s += " (" + t + ")";
    }
    if (msg) LocalFree(msg);
    return s;
}

bool is_absolute(const std::wstring& p) {
    return p.size() > 2 && ((p[1] == L':' && (p[2] == L'\\' || p[2] == L'/')) || (p[0] == L'\\' && p[1] == L'\\'));
}

// Try to load `path`; on success fill g_lib. Appends a diagnostic to `err` on failure.
bool try_load(const std::wstring& path, std::string& err) {
    // Absolute path: ALTERED_SEARCH_PATH makes the DLL's own directory (and then the standard search incl. PATH)
    // the place where its dependencies are resolved.
    DWORD flags = is_absolute(path) ? LOAD_WITH_ALTERED_SEARCH_PATH : 0;
    HMODULE m = LoadLibraryExW(path.c_str(), nullptr, flags);
    if (!m) {
        err += "\n  " + wide_to_utf8(path.c_str()) + ": " + win_error_text(GetLastError());
        return false;
    }
    auto create = reinterpret_cast<void* (*)(void*, int32_t)>(GetProcAddress(m, "createInferRuntime_INTERNAL"));
    auto ver = reinterpret_cast<int32_t (*)()>(GetProcAddress(m, "getInferLibVersion"));
    if (!create || !ver) {
        err += "\n  " + wide_to_utf8(path.c_str()) + ": loaded, but createInferRuntime_INTERNAL/getInferLibVersion not exported";
        FreeLibrary(m);
        return false;
    }
    int32_t v = ver();
    if (v / 10000 != NV_TENSORRT_MAJOR) {
        err += "\n  " + wide_to_utf8(path.c_str()) + ": TensorRT version " + std::to_string(v) +
               " does not match the headers fb_native was built with (major " + std::to_string(NV_TENSORRT_MAJOR) + ")";
        FreeLibrary(m);
        return false;
    }
    g_lib.module = m;
    g_lib.create_runtime = create;
    g_lib.lib_version = ver;
    wchar_t full[MAX_PATH * 4];
    DWORD n = GetModuleFileNameW(m, full, static_cast<DWORD>(sizeof(full) / sizeof(full[0])));
    g_lib.path_utf8 = n ? wide_to_utf8(std::wstring(full, n).c_str()) : wide_to_utf8(path.c_str());
    g_loaded = true;
    return true;
}
}  // namespace

void set_library(const char* path_utf8) {
    std::lock_guard<std::mutex> lk(g_lib_mu);
    require(path_utf8 && *path_utf8, "dll_path is NULL or empty");
    std::wstring w = utf8_to_wide(path_utf8);
    if (g_loaded) {
        std::string cur = g_lib.path_utf8;
        if (_stricmp(cur.c_str(), path_utf8) == 0) return;
        fail(FB_ERR_TRT_LOAD, "TensorRT lean runtime already loaded from " + cur);
    }
    std::string err;
    if (!try_load(w, err)) fail(FB_ERR_TRT_LOAD, "cannot load TensorRT lean runtime:" + err);
    g_override = w;
}

const Lib& load_library() {
    std::lock_guard<std::mutex> lk(g_lib_mu);
    if (g_loaded) return g_lib;
    std::string err;
    // 1. explicit override was handled eagerly in set_library (g_loaded). 2. env FB_TRT_LEAN_DLL
    if (const wchar_t* env = _wgetenv(L"FB_TRT_LEAN_DLL"); env && *env) {
        if (try_load(env, err)) return g_lib;
    }
    // 3. directory of fb_native.dll
    std::wstring dir = own_module_dir();
    if (!dir.empty()) {
        if (try_load(dir + L"\\" + lean_dll_name(), err)) return g_lib;
    }
    // 4. default DLL search path
    if (try_load(lean_dll_name(), err)) return g_lib;
    fail(FB_ERR_TRT_LOAD, "cannot load " + wide_to_utf8(lean_dll_name().c_str()) +
                              " (set FB_TRT_LEAN_DLL or place it next to fb_native.dll). Tried:" + err);
}

}  // namespace fb::trt
