// Copyright The Levanter Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#if defined(DISABLE_SM90_FEATURES)

#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream) \
  cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}

#define LAUNCH_KERNEL(config, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))

#define SET_SHARED_MEMORY_FOR_TMA(kernel) void()

#else

#define SET_SHARED_MEMORY_FOR_TMA(kernel) \
  do { \
    EP_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == \
                   cudaSuccess); \
    cfg.dynamicSmemBytes = smem_size; \
  } while (false)

#endif
