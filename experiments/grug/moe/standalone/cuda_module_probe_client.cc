// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <dlfcn.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace {

using ModuleLoad = int (*)(void**, const void*);
using ModuleUnload = int (*)(void*);

std::array<unsigned char, 128> test_image() {
    std::array<unsigned char, 128> image{};
    image[0] = 0x7f;
    image[1] = 'E';
    image[2] = 'L';
    image[3] = 'F';
    image[4] = 2;
    image[5] = 1;
    const std::uint16_t header_size = 64;
    const std::uint64_t section_offset = std::getenv("CLIENT_INVALID_ELF") == nullptr ? 64 : (2ULL << 30);
    const std::uint16_t section_entry_size = 64;
    const std::uint16_t section_count = 1;
    std::memcpy(image.data() + 40, &section_offset, sizeof(section_offset));
    std::memcpy(image.data() + 52, &header_size, sizeof(header_size));
    std::memcpy(image.data() + 58, &section_entry_size, sizeof(section_entry_size));
    std::memcpy(image.data() + 60, &section_count, sizeof(section_count));
    return image;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s LIBCUDA\n", argv[0]);
        return 2;
    }
    void* handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        std::fprintf(stderr, "dlopen: %s\n", dlerror());
        return 3;
    }
    auto load = reinterpret_cast<ModuleLoad>(dlsym(handle, "cuModuleLoadFatBinary"));
    auto unload = reinterpret_cast<ModuleUnload>(dlsym(handle, "cuModuleUnload"));
    if (load == nullptr || unload == nullptr) {
        std::fprintf(stderr, "dlsym: %s\n", dlerror());
        return 4;
    }
    const auto image = test_image();
    if (std::getenv("CLIENT_THREADS") != nullptr) {
        std::array<void*, 2> modules{};
        std::array<int, 2> results{};
        std::array<std::thread, 2> threads;
        for (std::size_t index = 0; index < threads.size(); ++index) {
            threads[index] = std::thread([&, index] { results[index] = load(&modules[index], image.data()); });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        int successes = 0;
        for (std::size_t index = 0; index < modules.size(); ++index) {
            if (results[index] == 0 && unload(modules[index]) == 0) {
                ++successes;
            }
        }
        std::printf("{\"success_count\":%d}\n", successes);
        dlclose(handle);
        return successes == 2 ? 0 : 5;
    }
    void* module = nullptr;
    const void* image_pointer = std::getenv("CLIENT_NULL_IMAGE") == nullptr ? image.data() : nullptr;
    const int module_result = load(&module, image_pointer);
    const int unload_result = module_result == 0 ? unload(module) : -1;
    std::printf(
        "{\"module_result\":%d,\"unload_result\":%d,\"module\":%llu}\n",
        module_result,
        unload_result,
        static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(module))
    );
    dlclose(handle);
    return module_result == 0 && unload_result == 0 ? 0 : 5;
}
