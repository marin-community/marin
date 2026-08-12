// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Transforms/XlaRegistration.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "shuttle/Runtime/CpuBytecode.h"
#include "xla/ffi/ffi.h"
#include "xla/pjrt/stablehlo_module_transform.h"
#include "llvm/Support/ErrorHandling.h"

namespace {

namespace ffi = xla::ffi;

bool hasDimensions(ffi::AnyBuffer::Dimensions actual,
                   std::initializer_list<int64_t> expected) {
  return actual.size() == expected.size() &&
         std::equal(actual.begin(), actual.end(), expected.begin());
}

struct ShuttleCpuExecutableState {
  std::vector<uint8_t> transportBytes;
  std::shared_ptr<const mlir::shuttle::CpuExecutable> executable;
};

absl::StatusOr<std::unique_ptr<ShuttleCpuExecutableState>>
InstantiateShuttleCpuExecutable(int64_t transportSchemaVersion,
                                absl::string_view bundleBytes,
                                int64_t bundleSize,
                                absl::string_view bundleSha256) {
  if (transportSchemaVersion != 1 || bundleSize < 0 ||
      static_cast<size_t>(bundleSize) != bundleBytes.size()) {
    return absl::InvalidArgumentError(
        "invalid Shuttle CPU executable transport metadata");
  }
  std::vector<uint8_t> owned(bundleBytes.begin(), bundleBytes.end());
  if (mlir::shuttle::cpuExecutableBundleDigest(owned) != bundleSha256) {
    return absl::InvalidArgumentError(
        "Shuttle CPU executable transport digest mismatch");
  }
  absl::StatusOr<std::shared_ptr<const mlir::shuttle::CpuExecutable>> loaded =
      mlir::shuttle::CpuExecutable::Load(owned);
  if (!loaded.ok()) {
    return loaded.status();
  }
  return std::make_unique<ShuttleCpuExecutableState>(
      ShuttleCpuExecutableState{std::move(owned), std::move(*loaded)});
}

using Bf16R2 = ffi::Buffer<xla::PrimitiveType::BF16, 2>;
using Bf16R1 = ffi::Buffer<xla::PrimitiveType::BF16, 1>;
using Bf16R2Result = ffi::Result<Bf16R2>;

absl::Status ExecuteShuttleCpuExecutable(Bf16R2 input, Bf16R1 scale,
                                         Bf16R2Result output,
                                         ShuttleCpuExecutableState *state,
                                         int64_t transportSchemaVersion,
                                         absl::string_view bundleBytes,
                                         int64_t bundleSize,
                                         absl::string_view bundleSha256) {
  if (state == nullptr || transportSchemaVersion != 1 || bundleSize < 0 ||
      static_cast<size_t>(bundleSize) != bundleBytes.size() ||
      state->transportBytes !=
          std::vector<uint8_t>(bundleBytes.begin(), bundleBytes.end()) ||
      mlir::shuttle::cpuExecutableBundleDigest(state->transportBytes) !=
          bundleSha256 ||
      !hasDimensions(input.dimensions(), {7, 13}) ||
      !hasDimensions(scale.dimensions(), {13}) ||
      !hasDimensions(output->dimensions(), {7, 13})) {
    return absl::InvalidArgumentError(
        "typed FFI call does not match the Shuttle executable ABI");
  }
  llvm::SmallVector<mlir::shuttle::CpuExternalBuffer> buffers{
      {0, llvm::MutableArrayRef<uint8_t>(
              reinterpret_cast<uint8_t *>(input.untyped_data()),
              input.size_bytes())},
      {1, llvm::MutableArrayRef<uint8_t>(
              reinterpret_cast<uint8_t *>(scale.untyped_data()),
              scale.size_bytes())},
      {20, llvm::MutableArrayRef<uint8_t>(
               reinterpret_cast<uint8_t *>(output->untyped_data()),
               output->size_bytes())}};
  return state->executable->Execute(buffers);
}

XLA_FFI_DEFINE_HANDLER(kInstantiateShuttleCpuExecutable,
                       InstantiateShuttleCpuExecutable,
                       ffi::Ffi::BindInstantiate()
                           .Attr<int64_t>("transport_schema_version")
                           .Attr<absl::string_view>("bundle_bytes")
                           .Attr<int64_t>("bundle_size")
                           .Attr<absl::string_view>("bundle_sha256"));

XLA_FFI_DEFINE_HANDLER(kExecuteShuttleCpuExecutable,
                       ExecuteShuttleCpuExecutable,
                       ffi::Ffi::Bind()
                           .Arg<Bf16R2>()
                           .Arg<Bf16R1>()
                           .Ret<Bf16R2>()
                           .Ctx<ffi::State<ShuttleCpuExecutableState>>()
                           .Attr<int64_t>("transport_schema_version")
                           .Attr<absl::string_view>("bundle_bytes")
                           .Attr<int64_t>("bundle_size")
                           .Attr<absl::string_view>("bundle_sha256"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         mlir::shuttle::kCpuExecutableBundleFfiTarget, "Host",
                         {/*instantiate=*/kInstantiateShuttleCpuExecutable,
                          /*prepare=*/nullptr,
                          /*initialize=*/nullptr,
                          /*execute=*/kExecuteShuttleCpuExecutable});

[[maybe_unused]] const bool kShuttleXlaTransformRegistered = [] {
  absl::Status status =
      xla::StablehloModuleTransformRegistry::Global().Register(
          "shuttle",
          [](mlir::ModuleOp module, absl::string_view serializedOptions) {
            return mlir::shuttle::runShuttleXlaTransform(module,
                                                         serializedOptions);
          });
  if (!status.ok()) {
    std::string diagnostic =
        absl::StrCat("failed to register the Shuttle XLA StableHLO transform: ",
                     status.message());
    llvm::report_fatal_error(diagnostic.c_str());
  }
  return true;
}();

} // namespace
