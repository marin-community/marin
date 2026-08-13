// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Transforms/XlaRegistration.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
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
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

namespace {

namespace ffi = xla::ffi;

struct ShuttleCpuExecutableState {
  std::vector<uint8_t> transportBytes;
  std::string transportDigest;
  std::shared_ptr<const mlir::shuttle::CpuExecutable> executable;
  std::vector<mlir::shuttle::CpuExternalBinding> operands;
  std::vector<mlir::shuttle::CpuExternalBinding> results;
};

absl::Status bindingMatches(const mlir::shuttle::CpuExternalBinding &binding,
                            const ffi::AnyBuffer &buffer, bool requireData) {
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(binding.tensorType);
  const xla::PrimitiveType expectedType =
      tensor && tensor.getElementType().isBF16() ? xla::PrimitiveType::BF16
      : tensor && tensor.getElementType().isF32()
          ? xla::PrimitiveType::F32
          : xla::PrimitiveType::PRIMITIVE_TYPE_INVALID;
  if (!tensor || expectedType == xla::PrimitiveType::PRIMITIVE_TYPE_INVALID ||
      buffer.element_type() != expectedType ||
      buffer.dimensions().size() != static_cast<size_t>(tensor.getRank()) ||
      !std::equal(buffer.dimensions().begin(), buffer.dimensions().end(),
                  tensor.getShape().begin(), tensor.getShape().end()) ||
      buffer.size_bytes() != static_cast<size_t>(binding.requiredBytes)) {
    return absl::InvalidArgumentError(
        "Shuttle CPU executable external bindings do not match the typed "
        "FFI contract");
  }
  if (requireData &&
      (buffer.untyped_data() == nullptr || binding.alignment <= 0 ||
       reinterpret_cast<uintptr_t>(buffer.untyped_data()) %
               static_cast<uint64_t>(binding.alignment) !=
           0)) {
    return absl::InvalidArgumentError(
        "typed FFI buffer does not satisfy Shuttle alignment");
  }
  return absl::OkStatus();
}

absl::Status
validateProjection(llvm::ArrayRef<mlir::shuttle::CpuExternalBinding> bindings,
                   std::vector<mlir::shuttle::CpuExternalBinding> &operands,
                   std::vector<mlir::shuttle::CpuExternalBinding> &results) {
  for (const mlir::shuttle::CpuExternalBinding &binding : bindings) {
    if (binding.index < 0 ||
        binding.index > static_cast<int64_t>(bindings.size())) {
      return absl::InvalidArgumentError(
          "Shuttle CPU executable has an invalid external binding index");
    }
    std::vector<mlir::shuttle::CpuExternalBinding> *projection = nullptr;
    if (binding.kind == mlir::shuttle::ExecutableBindingKind::Operand) {
      projection = &operands;
    } else if (binding.kind == mlir::shuttle::ExecutableBindingKind::Result) {
      projection = &results;
    } else {
      return absl::InvalidArgumentError(
          "Shuttle CPU executable external binding projection is invalid");
    }
    if (static_cast<size_t>(binding.index) != projection->size()) {
      return absl::InvalidArgumentError(
          "Shuttle CPU executable external binding projection is invalid");
    }
    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(binding.tensorType);
    if (!tensor ||
        (!tensor.getElementType().isBF16() &&
         !tensor.getElementType().isF32()) ||
        !tensor.hasStaticShape() || binding.requiredBytes <= 0 ||
        binding.alignment <= 0) {
      return absl::InvalidArgumentError(
          "Shuttle CPU executable external binding type is invalid");
    }
    const uint64_t elementBytes = tensor.getElementType().isBF16() ? 2 : 4;
    uint64_t expectedBytes = elementBytes;
    for (int64_t extent : tensor.getShape()) {
      if (extent <= 0 || expectedBytes > std::numeric_limits<uint64_t>::max() /
                                             static_cast<uint64_t>(extent)) {
        return absl::InvalidArgumentError(
            "Shuttle CPU executable external binding type is invalid");
      }
      expectedBytes *= static_cast<uint64_t>(extent);
    }
    if (expectedBytes != static_cast<uint64_t>(binding.requiredBytes)) {
      return absl::InvalidArgumentError(
          "Shuttle CPU executable external binding byte extent is invalid");
    }
    if (binding.alignment != static_cast<int64_t>(elementBytes)) {
      return absl::InvalidArgumentError(
          "Shuttle CPU executable external binding alignment is invalid");
    }
    projection->push_back(binding);
  }

  auto isSmallMatrix = [](const mlir::shuttle::CpuExternalBinding &binding) {
    auto tensor = mlir::cast<mlir::RankedTensorType>(binding.tensorType);
    return tensor.getElementType().isBF16() && tensor.getRank() == 2 &&
           tensor.getDimSize(0) == 7 && tensor.getDimSize(1) == 13;
  };
  auto isSmallVector = [](const mlir::shuttle::CpuExternalBinding &binding) {
    auto tensor = mlir::cast<mlir::RankedTensorType>(binding.tensorType);
    return tensor.getElementType().isBF16() && tensor.getRank() == 1 &&
           tensor.getDimSize(0) == 13;
  };
  auto isLargeMatrix = [](const mlir::shuttle::CpuExternalBinding &binding) {
    auto tensor = mlir::cast<mlir::RankedTensorType>(binding.tensorType);
    return tensor.getElementType().isBF16() && tensor.getRank() == 2 &&
           tensor.getDimSize(0) == 2048 && tensor.getDimSize(1) == 4096;
  };
  auto isLargeVector = [](const mlir::shuttle::CpuExternalBinding &binding) {
    auto tensor = mlir::cast<mlir::RankedTensorType>(binding.tensorType);
    return tensor.getElementType().isBF16() && tensor.getRank() == 1 &&
           tensor.getDimSize(0) == 4096;
  };
  const bool smallTwoInputOneResult =
      operands.size() == 2 && results.size() == 1 &&
      isSmallMatrix(operands[0]) && isSmallVector(operands[1]) &&
      isSmallMatrix(results[0]);
  const bool largeTwoInputOneResult =
      operands.size() == 2 && results.size() == 1 &&
      isLargeMatrix(operands[0]) && isLargeVector(operands[1]) &&
      isLargeMatrix(results[0]);
  const bool twoInputOneResult =
      smallTwoInputOneResult || largeTwoInputOneResult;
  const bool threeInputTwoResult =
      operands.size() == 3 && results.size() == 2 &&
      isSmallMatrix(operands[0]) && isSmallVector(operands[1]) &&
      isSmallMatrix(operands[2]) && isSmallVector(results[0]) &&
      isSmallMatrix(results[1]);
  const bool threeInputThreeResult =
      operands.size() == 3 && results.size() == 3 &&
      isSmallMatrix(operands[0]) && isSmallVector(operands[1]) &&
      isSmallMatrix(operands[2]) && isSmallMatrix(results[0]) &&
      isSmallVector(results[1]) && isSmallMatrix(results[2]);
  if (!twoInputOneResult && !threeInputTwoResult && !threeInputThreeResult) {
    return absl::InvalidArgumentError(
        "Shuttle CPU executable external bindings do not match the closed "
        "typed FFI projection");
  }
  return absl::OkStatus();
}

absl::Status validateViews(
    ffi::RemainingArgs arguments, ffi::RemainingRets results,
    llvm::ArrayRef<mlir::shuttle::CpuExternalBinding> operands,
    llvm::ArrayRef<mlir::shuttle::CpuExternalBinding> outputs, bool requireData,
    llvm::SmallVectorImpl<mlir::shuttle::CpuExternalBuffer> *buffers) {
  if (operands.size() != arguments.size() || outputs.size() != results.size()) {
    return absl::InvalidArgumentError(
        "Shuttle CPU executable external bindings do not match the typed "
        "FFI contract");
  }
  if (buffers != nullptr) {
    buffers->clear();
    buffers->reserve(operands.size() + outputs.size());
  }
  for (size_t index = 0; index < operands.size(); ++index) {
    absl::StatusOr<ffi::AnyBuffer> buffer =
        arguments.get<ffi::AnyBuffer>(index);
    if (!buffer.ok()) {
      return buffer.status();
    }
    if (absl::Status status =
            bindingMatches(operands[index], *buffer, requireData);
        !status.ok()) {
      return status;
    }
    if (buffers != nullptr) {
      buffers->push_back(
          {operands[index].slotOrdinal,
           llvm::MutableArrayRef<uint8_t>(
               reinterpret_cast<uint8_t *>(buffer->untyped_data()),
               operands[index].requiredBytes)});
    }
  }
  for (size_t index = 0; index < outputs.size(); ++index) {
    absl::StatusOr<ffi::Result<ffi::AnyBuffer>> buffer =
        results.get<ffi::AnyBuffer>(index);
    if (!buffer.ok()) {
      return buffer.status();
    }
    if (absl::Status status =
            bindingMatches(outputs[index], **buffer, requireData);
        !status.ok()) {
      return status;
    }
    if (buffers != nullptr) {
      buffers->push_back(
          {outputs[index].slotOrdinal,
           llvm::MutableArrayRef<uint8_t>(
               reinterpret_cast<uint8_t *>((*buffer)->untyped_data()),
               outputs[index].requiredBytes)});
    }
  }
  if (buffers != nullptr) {
    for (auto [index, buffer] : llvm::enumerate(*buffers)) {
      const uintptr_t begin = reinterpret_cast<uintptr_t>(buffer.bytes.data());
      if (begin > std::numeric_limits<uintptr_t>::max() - buffer.bytes.size()) {
        return absl::InvalidArgumentError(
            "typed FFI buffer range overflows the address space");
      }
      const uintptr_t end = begin + buffer.bytes.size();
      for (const mlir::shuttle::CpuExternalBuffer &prior :
           llvm::ArrayRef(*buffers).take_front(index)) {
        const uintptr_t priorBegin =
            reinterpret_cast<uintptr_t>(prior.bytes.data());
        const uintptr_t priorEnd = priorBegin + prior.bytes.size();
        if (begin < priorEnd && priorBegin < end) {
          return absl::InvalidArgumentError(
              "typed FFI external buffers must not alias");
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ShuttleCpuExecutableState>>
InstantiateShuttleCpuExecutable(int64_t transportSchemaVersion,
                                absl::string_view bundleBytes,
                                int64_t bundleSize,
                                absl::string_view bundleSha256) {
  try {
    if (transportSchemaVersion != 1 || bundleSize < 0 ||
        static_cast<size_t>(bundleSize) != bundleBytes.size() ||
        bundleBytes.size() > mlir::shuttle::kMaximumCpuTransportBytes) {
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
    std::vector<mlir::shuttle::CpuExternalBinding> operands;
    std::vector<mlir::shuttle::CpuExternalBinding> results;
    if (absl::Status status = validateProjection(bindings, operands, results);
        !status.ok()) {
      return status;
    }
    return std::make_unique<ShuttleCpuExecutableState>(
        ShuttleCpuExecutableState{std::move(owned), std::move(digest),
                                  std::move(*loaded), std::move(operands),
                                  std::move(results)});
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

absl::Status ExecuteShuttleCpuExecutable(ffi::RemainingArgs arguments,
                                         ffi::RemainingRets results,
                                         ShuttleCpuExecutableState *state,
                                         int64_t transportSchemaVersion,
                                         absl::string_view bundleBytes,
                                         int64_t bundleSize,
                                         absl::string_view bundleSha256) {
  try {
    if (state == nullptr || transportSchemaVersion != 1 || bundleSize < 0 ||
        static_cast<size_t>(bundleSize) != bundleBytes.size() ||
        state->transportBytes.size() != bundleBytes.size() ||
        std::memcmp(state->transportBytes.data(), bundleBytes.data(),
                    bundleBytes.size()) != 0 ||
        state->transportDigest != bundleSha256) {
      return absl::InvalidArgumentError(
          "typed FFI call does not match the Shuttle executable ABI");
    }
    llvm::SmallVector<mlir::shuttle::CpuExternalBuffer> buffers;
    if (absl::Status status =
            validateViews(arguments, results, state->operands, state->results,
                          /*requireData=*/true, &buffers);
        !status.ok()) {
      return status;
    }
    return state->executable->Execute(buffers);
  } catch (const std::bad_alloc &) {
    return absl::ResourceExhaustedError(
        "insufficient memory to execute Shuttle CPU executable");
  } catch (const std::length_error &) {
    return absl::InvalidArgumentError(
        "invalid oversized Shuttle CPU executable invocation");
  } catch (...) {
    return absl::InternalError(
        "unexpected failure while executing Shuttle CPU executable");
  }
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
                           .RemainingArgs()
                           .RemainingRets()
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
