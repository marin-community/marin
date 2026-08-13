// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "shuttle/Runtime/GpuFfi.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
#include "gtest/gtest.h"

namespace {

struct Observation {
  int64_t calls = 0;
  std::string target;
  int apiVersion = -1;
  XLA_FFI_Handler_Bundle handlers{};
};

Observation observation;

PJRT_Error *registerWithFakePjrt(PJRT_Gpu_Register_Custom_Call_Args *args) {
  ++observation.calls;
  observation.target.assign(args->function_name, args->function_name_size);
  observation.apiVersion = args->api_version;
  observation.handlers = {
      reinterpret_cast<XLA_FFI_Handler *>(args->handler_instantiate),
      reinterpret_cast<XLA_FFI_Handler *>(args->handler_prepare),
      reinterpret_cast<XLA_FFI_Handler *>(args->handler_initialize),
      reinterpret_cast<XLA_FFI_Handler *>(args->handler_execute),
  };
  return nullptr;
}

TEST(GpuFfiRegistrationTest, RoutesExportedBundleThroughTypedPjrtApiOne) {
  XLA_FFI_Handler_Bundle bundle =
      mlir::shuttle::gpuExecutableBundleFfiHandlerBundle();
  PJRT_Gpu_Register_Custom_Call_Args args{};
  args.struct_size = PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE;
  args.function_name = "shuttle.gpu.executable_bundle.v1";
  args.function_name_size = std::char_traits<char>::length(args.function_name);
  args.api_version = 1;
  args.handler_instantiate = bundle.instantiate;
  args.handler_prepare = bundle.prepare;
  args.handler_initialize = bundle.initialize;
  args.handler_execute = bundle.execute;

  observation = {};
  ASSERT_EQ(registerWithFakePjrt(&args), nullptr);
  EXPECT_EQ(observation.calls, 1);
  EXPECT_EQ(observation.target, "shuttle.gpu.executable_bundle.v1");
  EXPECT_EQ(observation.apiVersion, 1);
  EXPECT_EQ(observation.handlers.instantiate, bundle.instantiate);
  EXPECT_EQ(observation.handlers.prepare, bundle.prepare);
  EXPECT_EQ(observation.handlers.initialize, bundle.initialize);
  EXPECT_EQ(observation.handlers.execute, bundle.execute);
}

} // namespace
