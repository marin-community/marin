// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Transforms/XlaRegistration.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinTypes.h"
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
  std::string transportDigest;
  std::shared_ptr<const mlir::shuttle::CpuExecutable> executable;
  int64_t inputSlot;
  int64_t scaleSlot;
  int64_t outputSlot;
};

bool bindingMatches(const mlir::shuttle::CpuExternalBinding &binding,
                    mlir::shuttle::ExecutableBindingKind kind, int64_t index,
                    std::initializer_list<int64_t> shape) {
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(binding.tensorType);
  if (!tensor || !tensor.getElementType().isBF16() ||
      tensor.getRank() != static_cast<int64_t>(shape.size()) ||
      !std::equal(tensor.getShape().begin(), tensor.getShape().end(),
                  shape.begin(), shape.end())) {
    return false;
  }
  uint64_t elements = 1;
  for (int64_t extent : shape) {
    elements *= extent;
  }
  return binding.kind == kind && binding.index == index &&
         binding.requiredBytes == static_cast<int64_t>(elements * 2) &&
         binding.alignment == 2;
}

absl::StatusOr<std::unique_ptr<ShuttleCpuExecutableState>>
InstantiateShuttleCpuExecutable(int64_t transportSchemaVersion,
                                absl::string_view bundleBytes,
                                int64_t bundleSize,
                                absl::string_view bundleSha256) {
  try {
    if (transportSchemaVersion != 1 || bundleSize < 0 ||
        static_cast<size_t>(bundleSize) != bundleBytes.size()) {
      return absl::InvalidArgumentError(
          "invalid Shuttle CPU executable transport metadata");
    }
    std::vector<uint8_t> owned(bundleBytes.begin(), bundleBytes.end());
    std::string digest = mlir::shuttle::cpuExecutableBundleDigest(owned);
    if (digest != bundleSha256) {
      return absl::InvalidArgumentError(
          "Shuttle CPU executable transport digest mismatch");
    }
    absl::StatusOr<std::shared_ptr<const mlir::shuttle::CpuExecutable>> loaded =
        mlir::shuttle::CpuExecutable::Load(owned);
    if (!loaded.ok()) {
      return loaded.status();
    }
    llvm::ArrayRef<mlir::shuttle::CpuExternalBinding> bindings =
        (*loaded)->externalBindings();
    if (bindings.size() != 3 ||
        !bindingMatches(bindings[0],
                        mlir::shuttle::ExecutableBindingKind::Operand, 0,
                        {7, 13}) ||
        !bindingMatches(bindings[1],
                        mlir::shuttle::ExecutableBindingKind::Operand, 1,
                        {13}) ||
        !bindingMatches(bindings[2],
                        mlir::shuttle::ExecutableBindingKind::Result, 0,
                        {7, 13})) {
      return absl::InvalidArgumentError(
          "Shuttle CPU executable external bindings do not match the typed "
          "FFI contract");
    }
    return std::make_unique<ShuttleCpuExecutableState>(
        ShuttleCpuExecutableState{std::move(owned), std::move(digest),
                                  std::move(*loaded), bindings[0].slotOrdinal,
                                  bindings[1].slotOrdinal,
                                  bindings[2].slotOrdinal});
  } catch (const std::bad_alloc &) {
    return absl::ResourceExhaustedError(
        "insufficient memory to instantiate Shuttle CPU executable");
  } catch (const std::length_error &) {
    return absl::InvalidArgumentError(
        "invalid oversized Shuttle CPU executable transport");
  } catch (...) {
    return absl::InternalError(
        "unexpected failure while instantiating Shuttle CPU executable");
  }
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
      state->transportBytes.size() != bundleBytes.size() ||
      std::memcmp(state->transportBytes.data(), bundleBytes.data(),
                  bundleBytes.size()) != 0 ||
      state->transportDigest != bundleSha256 ||
      !hasDimensions(input.dimensions(), {7, 13}) ||
      !hasDimensions(scale.dimensions(), {13}) ||
      !hasDimensions(output->dimensions(), {7, 13})) {
    return absl::InvalidArgumentError(
        "typed FFI call does not match the Shuttle executable ABI");
  }
  llvm::SmallVector<mlir::shuttle::CpuExternalBuffer> buffers{
      {state->inputSlot, llvm::MutableArrayRef<uint8_t>(
                             reinterpret_cast<uint8_t *>(input.untyped_data()),
                             input.size_bytes())},
      {state->scaleSlot, llvm::MutableArrayRef<uint8_t>(
                             reinterpret_cast<uint8_t *>(scale.untyped_data()),
                             scale.size_bytes())},
      {state->outputSlot,
       llvm::MutableArrayRef<uint8_t>(
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
