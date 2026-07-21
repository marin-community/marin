// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

using CUresult = int;
using CUmodule = void*;
using CUlibrary = void*;
using CUcontext = void*;
using CUdeviceptr = std::uint64_t;
using Dlsym = void* (*)(void*, const char*);
using ModuleLoad = CUresult (*)(CUmodule*, const void*);
using ModuleLoadDataEx = CUresult (*)(CUmodule*, const void*, unsigned int, int*, void**);
using LibraryLoadData = CUresult (*)(CUlibrary*, const void*, int*, void**, unsigned int, int*, void**, unsigned int);
using CtxSynchronize = CUresult (*)();
using CtxGetCurrent = CUresult (*)(CUcontext*);
using CtxGetDevice = CUresult (*)(int*);
using MemGetInfo = CUresult (*)(std::size_t*, std::size_t*);
using MemAlloc = CUresult (*)(CUdeviceptr*, std::size_t);
using MemFree = CUresult (*)(CUdeviceptr);

constexpr CUresult kInvalidValue = 1;
constexpr std::size_t kMaximumElfSize = 1ULL << 30;
constexpr char kFatBinarySymbol[] = "cuModuleLoadFatBinary";
constexpr char kDataSymbol[] = "cuModuleLoadData";
constexpr char kDataExSymbol[] = "cuModuleLoadDataEx";
constexpr char kLibraryDataSymbol[] = "cuLibraryLoadData";

std::atomic<ModuleLoad> original_fat_binary{nullptr};
std::atomic<ModuleLoad> original_data{nullptr};
std::atomic<ModuleLoadDataEx> original_data_ex{nullptr};
std::atomic<LibraryLoadData> original_library_data{nullptr};
std::atomic<CtxSynchronize> context_synchronize{nullptr};
std::atomic<CtxGetCurrent> context_get_current{nullptr};
std::atomic<CtxGetDevice> context_get_device{nullptr};
std::atomic<MemGetInfo> memory_get_info{nullptr};
std::atomic<MemAlloc> memory_alloc{nullptr};
std::atomic<MemFree> memory_free{nullptr};
std::atomic<std::uint64_t> next_sequence{1};
std::atomic<unsigned int> in_flight_loads{0};
thread_local bool resolving_symbol = false;

struct ElfIdentity {
    enum class Kind { kUnknown, kInvalidElf64, kElf64 };
    Kind kind = Kind::kUnknown;
    std::size_t size = 0;
    std::string sha256;
};

struct Attempt {
    const char* name;
    CUresult result;
};

std::uint16_t read_u16(const unsigned char* bytes) {
    std::uint16_t value;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

std::uint32_t read_u32(const unsigned char* bytes) {
    std::uint32_t value;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

std::uint64_t read_u64(const unsigned char* bytes) {
    std::uint64_t value;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

bool bounded_end(std::uint64_t offset, std::uint64_t count, std::uint64_t element_size, std::size_t* end) {
    if (count != 0 && element_size > (kMaximumElfSize - offset) / count) {
        return false;
    }
    const std::uint64_t candidate = offset + count * element_size;
    if (candidate > kMaximumElfSize) {
        return false;
    }
    *end = static_cast<std::size_t>(candidate);
    return true;
}

std::uint32_t rotate_right(std::uint32_t value, unsigned int amount) {
    return (value >> amount) | (value << (32 - amount));
}

std::string sha256(const unsigned char* data, std::size_t size) {
    constexpr std::array<std::uint32_t, 64> round_constants = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    };
    std::array<std::uint32_t, 8> hash = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };
    const std::size_t padded_size = ((size + 9 + 63) / 64) * 64;
    std::vector<unsigned char> padded(padded_size, 0);
    std::memcpy(padded.data(), data, size);
    padded[size] = 0x80;
    const std::uint64_t bit_size = static_cast<std::uint64_t>(size) * 8;
    for (unsigned int index = 0; index < 8; ++index) {
        padded[padded_size - 1 - index] = static_cast<unsigned char>(bit_size >> (index * 8));
    }
    for (std::size_t block = 0; block < padded_size; block += 64) {
        std::array<std::uint32_t, 64> words{};
        for (unsigned int index = 0; index < 16; ++index) {
            const unsigned char* word = padded.data() + block + index * 4;
            words[index] = static_cast<std::uint32_t>(word[0]) << 24 | static_cast<std::uint32_t>(word[1]) << 16 |
                           static_cast<std::uint32_t>(word[2]) << 8 | word[3];
        }
        for (unsigned int index = 16; index < 64; ++index) {
            const std::uint32_t sigma0 = rotate_right(words[index - 15], 7) ^ rotate_right(words[index - 15], 18) ^
                                         (words[index - 15] >> 3);
            const std::uint32_t sigma1 = rotate_right(words[index - 2], 17) ^ rotate_right(words[index - 2], 19) ^
                                         (words[index - 2] >> 10);
            words[index] = words[index - 16] + sigma0 + words[index - 7] + sigma1;
        }
        auto [a, b, c, d, e, f, g, h] = hash;
        for (unsigned int index = 0; index < 64; ++index) {
            const std::uint32_t choice = (e & f) ^ (~e & g);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const std::uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const std::uint32_t temporary1 = h + sum1 + choice + round_constants[index] + words[index];
            const std::uint32_t temporary2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        hash[0] += a;
        hash[1] += b;
        hash[2] += c;
        hash[3] += d;
        hash[4] += e;
        hash[5] += f;
        hash[6] += g;
        hash[7] += h;
    }
    char output[65];
    for (unsigned int index = 0; index < hash.size(); ++index) {
        std::snprintf(output + index * 8, 9, "%08x", hash[index]);
    }
    output[64] = '\0';
    return output;
}

ElfIdentity elf_identity(const void* image) {
    ElfIdentity identity;
    if (image == nullptr) {
        return identity;
    }
    const auto* bytes = static_cast<const unsigned char*>(image);
    if (std::memcmp(bytes, "\x7f" "ELF", 4) != 0) {
        return identity;
    }
    identity.kind = ElfIdentity::Kind::kInvalidElf64;
    if (bytes[4] != 2 || bytes[5] != 1) {
        return identity;
    }
    const std::uint16_t header_size = read_u16(bytes + 52);
    const std::uint64_t program_offset = read_u64(bytes + 32);
    const std::uint16_t program_entry_size = read_u16(bytes + 54);
    const std::uint16_t program_count = read_u16(bytes + 56);
    const std::uint64_t section_offset = read_u64(bytes + 40);
    const std::uint16_t section_entry_size = read_u16(bytes + 58);
    const std::uint16_t section_count = read_u16(bytes + 60);
    std::size_t program_table_end;
    std::size_t section_table_end;
    if (header_size < 64 ||
        !bounded_end(program_offset, program_count, program_entry_size, &program_table_end) ||
        !bounded_end(section_offset, section_count, section_entry_size, &section_table_end) ||
        (program_count != 0 && program_entry_size < 56) || (section_count != 0 && section_entry_size < 64)) {
        return identity;
    }
    std::size_t span = std::max({static_cast<std::size_t>(header_size), program_table_end, section_table_end});
    for (std::uint16_t index = 0; index < program_count; ++index) {
        const unsigned char* entry = bytes + program_offset + static_cast<std::uint64_t>(index) * program_entry_size;
        std::size_t file_end;
        if (!bounded_end(read_u64(entry + 8), 1, read_u64(entry + 32), &file_end)) {
            return identity;
        }
        span = std::max(span, file_end);
    }
    for (std::uint16_t index = 0; index < section_count; ++index) {
        const unsigned char* entry = bytes + section_offset + static_cast<std::uint64_t>(index) * section_entry_size;
        if (read_u32(entry + 4) == 8) {
            continue;
        }
        std::size_t file_end;
        if (!bounded_end(read_u64(entry + 24), 1, read_u64(entry + 32), &file_end)) {
            return identity;
        }
        span = std::max(span, file_end);
    }
    if (span < 64 || span > kMaximumElfSize) {
        return identity;
    }
    identity.kind = ElfIdentity::Kind::kElf64;
    identity.size = span;
    identity.sha256 = sha256(bytes, span);
    return identity;
}

Dlsym real_dlsym() {
    static auto function = [] {
        void* resolved = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        if (resolved == nullptr) {
            resolved = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.17");
        }
        return reinterpret_cast<Dlsym>(resolved);
    }();
    if (function == nullptr) {
        _exit(126);
    }
    return function;
}

bool probe_required() {
    const char* value = std::getenv("MARIN_CUDA_MODULE_PROBE_REQUIRED");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

std::uint64_t timestamp_ns() {
    timespec value{};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000 + value.tv_nsec;
}

void log_event(const std::string& event) {
    const char* log_dir = std::getenv("MARIN_CUDA_MODULE_PROBE_LOG_DIR");
    if (log_dir == nullptr) {
        if (probe_required()) {
            _exit(125);
        }
        return;
    }
    const char* task = std::getenv("IRIS_TASK_INDEX");
    if (task == nullptr) {
        task = "local";
    }
    char path[4096];
    const int path_size = std::snprintf(
        path, sizeof(path), "%s/probe-%s-%ld.ndjson", log_dir, task, static_cast<long>(getpid())
    );
    if (path_size < 0 || static_cast<std::size_t>(path_size) >= sizeof(path)) {
        if (probe_required()) {
            _exit(125);
        }
        return;
    }
    const int file = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (file < 0) {
        if (probe_required()) {
            _exit(125);
        }
        return;
    }
    std::string line = event;
    line.push_back('\n');
    const ssize_t written = write(file, line.data(), line.size());
    close(file);
    if (written != static_cast<ssize_t>(line.size()) && probe_required()) {
        _exit(125);
    }
}

void capture_cubin(const void* image, const ElfIdentity& identity) {
    const char* capture = std::getenv("MARIN_CUDA_MODULE_PROBE_CAPTURE_CUBIN");
    const char* task = std::getenv("IRIS_TASK_INDEX");
    const char* log_dir = std::getenv("MARIN_CUDA_MODULE_PROBE_LOG_DIR");
    if (capture == nullptr || std::strcmp(capture, "1") != 0 || task == nullptr || std::strcmp(task, "0") != 0 ||
        log_dir == nullptr || identity.kind != ElfIdentity::Kind::kElf64) {
        return;
    }
    char path[4096];
    const int path_size = std::snprintf(path, sizeof(path), "%s/%s.cubin", log_dir, identity.sha256.c_str());
    if (path_size < 0 || static_cast<std::size_t>(path_size) >= sizeof(path)) {
        if (probe_required()) {
            _exit(124);
        }
        return;
    }
    const int file = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (file < 0) {
        if (errno != EEXIST && probe_required()) {
            _exit(124);
        }
        return;
    }
    const auto* bytes = static_cast<const unsigned char*>(image);
    std::size_t offset = 0;
    while (offset < identity.size) {
        const ssize_t written = write(file, bytes + offset, identity.size - offset);
        if (written <= 0) {
            close(file);
            if (probe_required()) {
                _exit(124);
            }
            return;
        }
        offset += static_cast<std::size_t>(written);
    }
    close(file);
}

const char* requested_profile() {
    const char* profile = std::getenv("MARIN_CUDA_MODULE_PROBE_PROFILE");
    return profile == nullptr ? "trace" : profile;
}

std::string effective_profile() {
    const char* requested = requested_profile();
    if (std::strcmp(requested, "trace_sync_split") != 0) {
        return requested;
    }
    const char* task = std::getenv("IRIS_TASK_INDEX");
    return task != nullptr && std::strtol(task, nullptr, 10) % 2 != 0 ? "sync" : "trace";
}

void resolve_related_symbols(void* handle, Dlsym lookup) {
    original_data.store(reinterpret_cast<ModuleLoad>(lookup(handle, kDataSymbol)), std::memory_order_release);
    original_data_ex.store(reinterpret_cast<ModuleLoadDataEx>(lookup(handle, kDataExSymbol)), std::memory_order_release);
    original_library_data.store(reinterpret_cast<LibraryLoadData>(lookup(handle, kLibraryDataSymbol)), std::memory_order_release);
    context_synchronize.store(reinterpret_cast<CtxSynchronize>(lookup(handle, "cuCtxSynchronize")), std::memory_order_release);
    context_get_current.store(reinterpret_cast<CtxGetCurrent>(lookup(handle, "cuCtxGetCurrent")), std::memory_order_release);
    context_get_device.store(reinterpret_cast<CtxGetDevice>(lookup(handle, "cuCtxGetDevice")), std::memory_order_release);
    memory_get_info.store(reinterpret_cast<MemGetInfo>(lookup(handle, "cuMemGetInfo")), std::memory_order_release);
    memory_alloc.store(reinterpret_cast<MemAlloc>(lookup(handle, "cuMemAlloc")), std::memory_order_release);
    memory_free.store(reinterpret_cast<MemFree>(lookup(handle, "cuMemFree")), std::memory_order_release);
}

std::string attempts_json(const std::vector<Attempt>& attempts) {
    std::string output = "[";
    for (std::size_t index = 0; index < attempts.size(); ++index) {
        if (index != 0) {
            output += ',';
        }
        output += "{\"name\":\"" + std::string(attempts[index].name) + "\",\"result\":" +
                  std::to_string(attempts[index].result) + '}';
    }
    output += ']';
    return output;
}

void append_post_load_telemetry(std::string& event) {
    CUcontext context = nullptr;
    int device = -1;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const auto get_context = context_get_current.load(std::memory_order_acquire);
    const auto get_device = context_get_device.load(std::memory_order_acquire);
    const auto get_memory = memory_get_info.load(std::memory_order_acquire);
    const CUresult context_result = get_context == nullptr ? kInvalidValue : get_context(&context);
    const CUresult device_result = get_device == nullptr ? kInvalidValue : get_device(&device);
    const CUresult memory_result = get_memory == nullptr ? kInvalidValue : get_memory(&free_bytes, &total_bytes);
    event += ",\"telemetry\":{\"context_result\":" + std::to_string(context_result) +
             ",\"context\":" + std::to_string(reinterpret_cast<std::uintptr_t>(context)) +
             ",\"device_result\":" + std::to_string(device_result) + ",\"device\":" + std::to_string(device) +
             ",\"memory_result\":" + std::to_string(memory_result) + ",\"free_bytes\":" +
             std::to_string(free_bytes) + ",\"total_bytes\":" + std::to_string(total_bytes) + '}';
}

void* aligned_copy(const void* image, std::size_t size) {
    void* copy = nullptr;
    if (posix_memalign(&copy, 4096, size) != 0) {
        return nullptr;
    }
    std::memcpy(copy, image, size);
    return copy;
}

}  // namespace

extern "C" CUresult cuModuleLoadFatBinary(CUmodule* module, const void* image) {
    const ModuleLoad fat_binary = original_fat_binary.load(std::memory_order_acquire);
    if (fat_binary == nullptr) {
        return kInvalidValue;
    }
    const std::uint64_t sequence = next_sequence.fetch_add(1);
    const unsigned int in_flight = in_flight_loads.fetch_add(1) + 1;
    const ElfIdentity identity = elf_identity(image);
    capture_cubin(image, identity);
    const std::string profile = effective_profile();
    const char* kind = identity.kind == ElfIdentity::Kind::kElf64
                           ? "elf64"
                           : (identity.kind == ElfIdentity::Kind::kInvalidElf64 ? "invalid_elf64" : "unknown");
    std::string enter = "{\"event\":\"load_enter\",\"sequence\":" + std::to_string(sequence) +
                        ",\"timestamp_ns\":" + std::to_string(timestamp_ns()) +
                        ",\"api\":\"cuModuleLoadFatBinary\",\"requested_profile\":\"" + requested_profile() +
                        "\",\"effective_profile\":\"" + profile + "\",\"input_kind\":\"" + kind +
                        "\",\"address_mod_4096\":" +
                        std::to_string(reinterpret_cast<std::uintptr_t>(image) % 4096) +
                        ",\"in_flight\":" + std::to_string(in_flight);
    if (identity.kind == ElfIdentity::Kind::kElf64) {
        enter += ",\"size\":" + std::to_string(identity.size) + ",\"sha256\":\"" + identity.sha256 + '"';
    }
    enter += '}';
    log_event(enter);

    std::vector<Attempt> attempts;
    CUresult result = kInvalidValue;
    CUresult pre_sync_result = 0;
    void* copy = nullptr;
    CUdeviceptr reserve = 0;
    const bool pressure = profile == "pressure";
    if (pressure) {
        const char* reserve_value = std::getenv("MARIN_CUDA_MODULE_PROBE_RESERVE_BYTES");
        const std::size_t reserve_bytes = reserve_value == nullptr ? 0 : std::strtoull(reserve_value, nullptr, 10);
        const auto allocate = memory_alloc.load(std::memory_order_acquire);
        if (reserve_bytes != 0 && allocate != nullptr) {
            allocate(&reserve, reserve_bytes);
        }
    }
    if (profile == "sync") {
        const auto synchronize = context_synchronize.load(std::memory_order_acquire);
        pre_sync_result = synchronize == nullptr ? kInvalidValue : synchronize();
        if (pre_sync_result != 0) {
            result = pre_sync_result;
        }
    }
    if (profile != "sync" || pre_sync_result == 0) {
        const ModuleLoad data = original_data.load(std::memory_order_acquire);
        if (profile == "data_direct" && identity.kind == ElfIdentity::Kind::kElf64 && data != nullptr) {
            result = data(module, image);
            attempts.push_back({"data_direct", result});
        } else {
            result = fat_binary(module, image);
            attempts.push_back({"original", result});
            if (result != 0 && identity.kind == ElfIdentity::Kind::kElf64) {
                result = fat_binary(module, image);
                attempts.push_back({"same_pointer", result});
            }
            if (result != 0 && identity.kind == ElfIdentity::Kind::kElf64) {
                copy = aligned_copy(image, identity.size);
                if (copy != nullptr) {
                    result = fat_binary(module, copy);
                    attempts.push_back({"owned_copy", result});
                }
            }
            if (result != 0 && copy != nullptr && data != nullptr) {
                result = data(module, copy);
                attempts.push_back({"owned_copy_data", result});
            }
        }
    }
    if (pressure && result != 0 && reserve != 0) {
        const auto release = memory_free.load(std::memory_order_acquire);
        if (release != nullptr) {
            release(reserve);
            reserve = 0;
        }
        const ModuleLoad data = original_data.load(std::memory_order_acquire);
        if (copy != nullptr && data != nullptr) {
            result = data(module, copy);
            attempts.push_back({"post_release_data", result});
        }
    }
    if (reserve != 0) {
        const auto release = memory_free.load(std::memory_order_acquire);
        if (release != nullptr) {
            release(reserve);
        }
    }

    std::string exit = "{\"event\":\"load_exit\",\"sequence\":" + std::to_string(sequence) +
                       ",\"timestamp_ns\":" + std::to_string(timestamp_ns()) + ",\"result\":" +
                       std::to_string(result) + ",\"pre_sync_result\":" + std::to_string(pre_sync_result) +
                       ",\"attempts\":" + attempts_json(attempts);
    append_post_load_telemetry(exit);
    exit += '}';
    log_event(exit);
    std::free(copy);
    in_flight_loads.fetch_sub(1);
    return result;
}

extern "C" CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    const ModuleLoad original = original_data.load(std::memory_order_acquire);
    return original == nullptr ? kInvalidValue : original(module, image);
}

extern "C" CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int option_count, int* options, void** values) {
    const ModuleLoadDataEx original = original_data_ex.load(std::memory_order_acquire);
    return original == nullptr ? kInvalidValue : original(module, image, option_count, options, values);
}

extern "C" CUresult cuLibraryLoadData(
    CUlibrary* library,
    const void* image,
    int* jit_options,
    void** jit_values,
    unsigned int jit_count,
    int* library_options,
    void** library_values,
    unsigned int library_count
) {
    const LibraryLoadData original = original_library_data.load(std::memory_order_acquire);
    return original == nullptr ? kInvalidValue
                               : original(
                                     library,
                                     image,
                                     jit_options,
                                     jit_values,
                                     jit_count,
                                     library_options,
                                     library_values,
                                     library_count
                                 );
}

extern "C" void* dlsym(void* handle, const char* symbol) {
    Dlsym lookup = real_dlsym();
    if (resolving_symbol) {
        return lookup(handle, symbol);
    }
    resolving_symbol = true;
    void* original = lookup(handle, symbol);
    void* replacement = original;
    if (original != nullptr && std::strcmp(symbol, kFatBinarySymbol) == 0) {
        original_fat_binary.store(reinterpret_cast<ModuleLoad>(original), std::memory_order_release);
        resolve_related_symbols(handle, lookup);
        replacement = reinterpret_cast<void*>(&cuModuleLoadFatBinary);
    } else if (original != nullptr && std::strcmp(symbol, kDataSymbol) == 0) {
        original_data.store(reinterpret_cast<ModuleLoad>(original), std::memory_order_release);
        replacement = reinterpret_cast<void*>(&cuModuleLoadData);
    } else if (original != nullptr && std::strcmp(symbol, kDataExSymbol) == 0) {
        original_data_ex.store(reinterpret_cast<ModuleLoadDataEx>(original), std::memory_order_release);
        replacement = reinterpret_cast<void*>(&cuModuleLoadDataEx);
    } else if (original != nullptr && std::strcmp(symbol, kLibraryDataSymbol) == 0) {
        original_library_data.store(reinterpret_cast<LibraryLoadData>(original), std::memory_order_release);
        replacement = reinterpret_cast<void*>(&cuLibraryLoadData);
    }
    resolving_symbol = false;
    if (replacement != original) {
        log_event(
            "{\"event\":\"symbol_redirect\",\"symbol\":\"" + std::string(symbol) +
            "\",\"timestamp_ns\":" + std::to_string(timestamp_ns()) + ",\"pid\":" +
            std::to_string(static_cast<long>(getpid())) + '}'
        );
    }
    return replacement;
}
