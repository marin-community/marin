// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_TRANSFORMS_XLAREGISTRATION_H_
#define SHUTTLE_TRANSFORMS_XLAREGISTRATION_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "shuttle/Transforms/Observer.h"

namespace mlir::shuttle {

// Parses the canonical, closed-schema value transported by
// xla_shuttle_options into the native pipeline configuration.
absl::StatusOr<ShuttlePipelineOptions>
parseShuttleXlaOptions(absl::string_view serializedOptions);

// Composite transform callback used by the XLA registry adapter. XLA owns
// transactional cloning; Shuttle owns its option schema and exact pipeline.
LogicalResult runShuttleXlaTransform(ModuleOp module,
                                     const ShuttlePipelineOptions &options);

// Parses serialized compiler options and runs the exact shared Shuttle
// pipeline. A failure is returned to XLA and never falls back silently.
absl::Status runShuttleXlaTransform(ModuleOp module,
                                    absl::string_view serializedOptions);

} // namespace mlir::shuttle

#endif // SHUTTLE_TRANSFORMS_XLAREGISTRATION_H_
