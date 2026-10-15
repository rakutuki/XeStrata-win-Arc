// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/platform/memory_test.cpp - plan v0.3 P0.1: lock_resident on a 256 MiB region (CPU only, no GPU, no model).
#include "strata/platform/memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/resource.h>


int main() {
    const uint64_t bytes = 256ull << 20;
    void* p = std::malloc(bytes);
    if (p == nullptr) return 2;
    std::memset(p, 1, bytes);
    const strata::platform::LockResult r = strata::platform::lock_resident(p, bytes);
    std::printf("lock_resident: ok=%d locked=%llu MiB (%s)\n", (int) r.ok, (unsigned long long) (r.locked_bytes >> 20),
                r.note.c_str());
    // a refusal under a memlock limit below the region (8 MiB by default on Ubuntu, without CAP_IPC_LOCK) is the
    // machine's setting, not a fault: skipped.  With the limit high enough, a refusal still fails.
    rlimit lim{};
    if (!r.ok && getrlimit(RLIMIT_MEMLOCK, &lim) == 0 && lim.rlim_cur != RLIM_INFINITY && lim.rlim_cur < bytes) {
        std::printf("platform_memory_test: skipped (ulimit -l is %llu KiB, below the %llu MiB region)\n",
                    (unsigned long long) (lim.rlim_cur >> 10), (unsigned long long) (bytes >> 20));
        std::free(p);
        return 77;
    }
    int fail = !(r.ok && r.locked_bytes == bytes);
    strata::platform::unlock_resident(p, bytes);
    std::free(p);
    std::printf("platform_memory_test: %s\n", fail ? "FAILED" : "OK");
    return fail;
}
