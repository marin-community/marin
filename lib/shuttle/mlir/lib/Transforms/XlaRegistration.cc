// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Transforms/XlaRegistration.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/Pass/PassManager.h"
#include "shuttle/IR/ShuttleDialect.h"
#include "shuttle/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::shuttle {
namespace {

constexpr int64_t kSchemaVersion = 1;
constexpr int64_t kPipelineAbiVersion = 7;
constexpr int64_t kMaximumNativeInteger = 2147483647;
constexpr size_t kMaximumTensorRank = 8;
constexpr size_t kMaximumClusterRank = 3;

void writeCanonicalJson(const llvm::json::Value &value,
                        llvm::raw_ostream &output) {
  if (const auto *object = value.getAsObject()) {
    std::vector<llvm::StringRef> keys;
    keys.reserve(object->size());
    for (const auto &property : *object) {
      keys.push_back(property.first);
    }
    llvm::sort(keys);

    output << '{';
    bool first = true;
    for (llvm::StringRef key : keys) {
      if (!first) {
        output << ',';
      }
      first = false;
      output << llvm::json::Value(key) << ':';
      writeCanonicalJson(*object->get(key), output);
    }
    output << '}';
    return;
  }
  if (const auto *array = value.getAsArray()) {
    output << '[';
    bool first = true;
    for (const llvm::json::Value &element : *array) {
      if (!first) {
        output << ',';
      }
      first = false;
      writeCanonicalJson(element, output);
    }
    output << ']';
    return;
  }
  output << value;
}

std::string canonicalJson(const llvm::json::Value &value) {
  std::string result;
  llvm::raw_string_ostream output(result);
  writeCanonicalJson(value, output);
  output.flush();
  return result;
}

absl::Status invalidOptions(absl::string_view message) {
  return absl::InvalidArgumentError(
      absl::StrCat("invalid xla_shuttle_options: ", message));
}

absl::Status requireExactFields(const llvm::json::Object &object,
                                llvm::ArrayRef<llvm::StringRef> fields,
                                absl::string_view objectName) {
  for (const auto &property : object) {
    if (!llvm::is_contained(fields, llvm::StringRef(property.first))) {
      return invalidOptions(absl::StrCat(objectName, " has unknown field '",
                                         property.first.str(), "'"));
    }
  }
  for (llvm::StringRef field : fields) {
    if (object.get(field) == nullptr) {
      return invalidOptions(absl::StrCat(
          objectName, " is missing required field '", field.str(), "'"));
    }
  }
  return absl::OkStatus();
}

absl::Status requirePositiveInteger(const llvm::json::Object &object,
                                    llvm::StringRef field) {
  std::optional<int64_t> value = object.getInteger(field);
  if (!value || *value <= 0 || *value > kMaximumNativeInteger) {
    return invalidOptions(absl::StrCat("tuning field '", field.str(),
                                       "' must be an integer between 1 and ",
                                       kMaximumNativeInteger));
  }
  return absl::OkStatus();
}

absl::Status requireBoundedShape(const llvm::json::Object &object,
                                 llvm::StringRef field, size_t maximumRank) {
  const llvm::json::Array *values = object.getArray(field);
  if (values == nullptr || values->size() > maximumRank) {
    return invalidOptions(absl::StrCat("tuning field '", field.str(),
                                       "' must be an array with at most ",
                                       maximumRank, " entries"));
  }
  for (const llvm::json::Value &element : *values) {
    std::optional<int64_t> value = element.getAsInteger();
    if (!value || *value <= 0 || *value > kMaximumNativeInteger) {
      return invalidOptions(absl::StrCat(
          "tuning field '", field.str(),
          "' entries must be integers between 1 and ", kMaximumNativeInteger));
    }
  }
  return absl::OkStatus();
}

} // namespace

absl::StatusOr<ShuttlePipelineOptions>
parseShuttleXlaOptions(absl::string_view serializedOptions) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(
      llvm::StringRef(serializedOptions.data(), serializedOptions.size()));
  if (!parsed) {
    return invalidOptions(llvm::toString(parsed.takeError()));
  }
  const llvm::json::Object *object = parsed->getAsObject();
  if (object == nullptr) {
    return invalidOptions("expected a JSON object");
  }

  const llvm::StringRef rootFields[] = {"execution_mode", "numerics",
                                        "pipeline_abi_version",
                                        "schema_version", "tuning"};
  if (absl::Status status = requireExactFields(*object, rootFields, "root");
      !status.ok()) {
    return status;
  }
  if (object->getInteger("schema_version") != kSchemaVersion) {
    return invalidOptions("field 'schema_version' must be integer 1");
  }
  if (object->getInteger("pipeline_abi_version") != kPipelineAbiVersion) {
    return invalidOptions("field 'pipeline_abi_version' must be integer 7");
  }

  std::optional<llvm::StringRef> numerics = object->getString("numerics");
  ShuttlePipelineOptions options;
  std::optional<llvm::StringRef> executionMode =
      object->getString("execution_mode");
  if (executionMode && *executionMode == "stablehlo_round_trip") {
    options.executionMode = ExecutionMode::StablehloRoundTrip;
  } else if (executionMode && *executionMode == "cpu_executable_bundle") {
    options.executionMode = ExecutionMode::CpuExecutableBundle;
  } else {
    return invalidOptions("field 'execution_mode' must be "
                          "'stablehlo_round_trip' or 'cpu_executable_bundle'");
  }
  if (numerics && *numerics == "source_ordered") {
    options.numerics = NumericalPolicy::SourceOrdered;
  } else if (numerics && *numerics == "fast") {
    options.numerics = NumericalPolicy::Fast;
  } else {
    return invalidOptions(
        "field 'numerics' must be 'source_ordered' or 'fast'");
  }
  if (options.executionMode == ExecutionMode::CpuExecutableBundle &&
      options.numerics != NumericalPolicy::SourceOrdered) {
    return invalidOptions(
        "cpu_executable_bundle requires source_ordered numerics");
  }

  const llvm::json::Value *tuningValue = object->get("tuning");
  const llvm::json::Object *tuning = tuningValue->getAsObject();
  if (tuning == nullptr) {
    return invalidOptions("field 'tuning' must be an object");
  }
  const llvm::StringRef tuningFields[] = {"cluster_shape", "materialization",
                                          "maximum_candidates",
                                          "pipeline_stages", "tile_sizes"};
  if (absl::Status status = requireExactFields(*tuning, tuningFields, "tuning");
      !status.ok()) {
    return status;
  }
  if (absl::Status status =
          requireBoundedShape(*tuning, "tile_sizes", kMaximumTensorRank);
      !status.ok()) {
    return status;
  }
  if (absl::Status status =
          requireBoundedShape(*tuning, "cluster_shape", kMaximumClusterRank);
      !status.ok()) {
    return status;
  }
  if (absl::Status status = requirePositiveInteger(*tuning, "pipeline_stages");
      !status.ok()) {
    return status;
  }
  if (absl::Status status =
          requirePositiveInteger(*tuning, "maximum_candidates");
      !status.ok()) {
    return status;
  }
  std::optional<llvm::StringRef> materialization =
      tuning->getString("materialization");
  if (!materialization ||
      (*materialization != "automatic" && *materialization != "prefer_fusion" &&
       *materialization != "prefer_materialization")) {
    return invalidOptions(
        "tuning field 'materialization' has an unsupported value");
  }

  options.canonicalOptions = canonicalJson(*parsed);
  if (serializedOptions != options.canonicalOptions) {
    return invalidOptions(absl::StrCat("value is not canonical; expected ",
                                       options.canonicalOptions));
  }
  options.canonicalTuning = canonicalJson(*tuningValue);
  return options;
}

LogicalResult runShuttleXlaTransform(ModuleOp module,
                                     const ShuttlePipelineOptions &options) {
  module.getContext()->getOrLoadDialect<ShuttleDialect>();
  PassManager manager(module.getContext(), ModuleOp::getOperationName());
  buildShuttleStablehloPipeline(manager, options);
  return manager.run(module);
}

absl::Status runShuttleXlaTransform(ModuleOp module,
                                    absl::string_view serializedOptions) {
  absl::StatusOr<ShuttlePipelineOptions> options =
      parseShuttleXlaOptions(serializedOptions);
  if (!options.ok()) {
    return options.status();
  }
  if (failed(runShuttleXlaTransform(module, *options))) {
    return absl::InternalError("Shuttle XLA pipeline failed");
  }
  return absl::OkStatus();
}

} // namespace mlir::shuttle
