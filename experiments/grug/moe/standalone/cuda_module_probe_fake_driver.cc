// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace {

std::atomic<unsigned int> fat_index{0};
std::atomic<unsigned int> data_index{0};
std::atomic<unsigned int> barrier_arrivals{0};

int configured_result(const char* variable, std::atomic<unsigned int>& index) {
    const char* values = std::getenv(variable);
    if (values == nullptr) {
        return 0;
    }
    unsigned int target = index.fetch_add(1);
    const char* cursor = values;
    for (unsigned int current = 0; current < target && *cursor != '\0'; ++current) {
        const char* comma = std::strchr(cursor, ',');
        if (comma == nullptr) {
            break;
        }
        cursor = comma + 1;
    }
    return std::atoi(cursor);
}

void record_call(const char* api, int result) {
    const char* path = std::getenv("FAKE_CUDA_LOG");
    if (path == nullptr) {
        return;
    }
    const int file = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (file < 0) {
        return;
    }
    char line[128];
    const int size = std::snprintf(line, sizeof(line), "%s:%d\n", api, result);
    if (size > 0) {
        const ssize_t ignored = write(file, line, static_cast<std::size_t>(size));
        (void)ignored;
    }
    close(file);
}

}  // namespace

extern "C" int cuModuleLoadFatBinary(void** module, const void*) {
    const int result = configured_result("FAKE_CUDA_FAT_RESULTS", fat_index);
    if (std::getenv("FAKE_CUDA_BARRIER") != nullptr && barrier_arrivals.fetch_add(1) < 2) {
        while (barrier_arrivals.load() < 2) {
            std::this_thread::yield();
        }
    }
    const char* delay_us = std::getenv("FAKE_CUDA_DELAY_US");
    if (delay_us != nullptr) {
        std::this_thread::sleep_for(std::chrono::microseconds(std::strtoul(delay_us, nullptr, 10)));
    }
    record_call("fat", result);
    if (result == 0) {
        *module = reinterpret_cast<void*>(0xF00);
    }
    return result;
}

extern "C" int cuModuleLoadData(void** module, const void*) {
    const int result = configured_result("FAKE_CUDA_DATA_RESULTS", data_index);
    record_call("data", result);
    if (result == 0) {
        *module = reinterpret_cast<void*>(0xD00);
    }
    return result;
}

extern "C" int cuModuleLoadDataEx(void** module, const void* image, unsigned int, int*, void**) {
    return cuModuleLoadData(module, image);
}

extern "C" int cuLibraryLoadData(void** library, const void* image, int*, void**, unsigned int, int*, void**, unsigned int) {
    return cuModuleLoadData(library, image);
}

extern "C" int cuModuleUnload(void* module) {
    return module == reinterpret_cast<void*>(0xF00) || module == reinterpret_cast<void*>(0xD00) ? 0 : 1;
}

extern "C" int cuCtxSynchronize() {
    const int result = std::getenv("FAKE_CUDA_SYNC_RESULT") == nullptr ? 0 : std::atoi(std::getenv("FAKE_CUDA_SYNC_RESULT"));
    record_call("sync", result);
    return result;
}

extern "C" int cuCtxGetCurrent(void** context) {
    *context = reinterpret_cast<void*>(0xC0);
    return 0;
}

extern "C" int cuCtxGetDevice(int* device) {
    *device = 0;
    return 0;
}

extern "C" int cuMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
    *free_bytes = 80ULL << 30;
    *total_bytes = 192ULL << 30;
    return 0;
}

extern "C" int cuMemAlloc(std::uint64_t* pointer, std::size_t) {
    *pointer = 0xA110C;
    record_call("alloc", 0);
    return 0;
}

extern "C" int cuMemFree(std::uint64_t) {
    record_call("free", 0);
    return 0;
}
