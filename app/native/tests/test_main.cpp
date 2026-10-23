#include <chrono>
#include <cstdlib>

#include "test_common.h"

int main(int argc, char** argv) {
    const char* filter = argc > 1 ? argv[1] : "";
    int drv = 0;
    int st = fb_init(0, &drv);
    if (st != FB_OK) {
        std::printf("fb_init failed (%d): %s\n", st, fb_last_error());
        return 2;
    }
    fb_gpu_info_t info;
    if (fb_gpu_info(0, &info) == FB_OK)
        std::printf("GPU: %s  sm_%d%d  driver %d  runtime %d  %.1f GB\n", info.name, info.cc_major, info.cc_minor,
                    info.driver_version, info.runtime_version, info.total_mem / 1073741824.0);

    int run = 0, failed = 0, skipped = 0;
    for (auto& tc : t::registry()) {
        if (*filter && std::string(tc.name).find(filter) == std::string::npos) continue;
        auto t0 = std::chrono::steady_clock::now();
        std::printf("[ RUN  ] %s\n", tc.name);
        std::fflush(stdout);
        try {
            tc.fn();
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("[  OK  ] %s (%.0f ms)\n", tc.name, ms);
        } catch (const t::Failure& f) {
            std::printf("[ FAIL ] %s\n         %s\n", tc.name, f.what());
            ++failed;
        } catch (const std::exception& e) {
            std::printf("[ FAIL ] %s\n         exception: %s\n", tc.name, e.what());
            ++failed;
        }
        ++run;
    }
    (void)skipped;
    std::printf("%d test(s) run, %d failed\n", run, failed);
    fb_shutdown();
    if (run == 0) {
        std::printf("no tests matched filter '%s'\n", filter);
        return 3;
    }
    return failed ? 1 : 0;
}
