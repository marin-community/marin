// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Transforms/XlaRegistration.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/stablehlo_module_transform.h"
#include "llvm/Support/ErrorHandling.h"

namespace {

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
