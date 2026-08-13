// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_TRANSFORMS_OBSERVER_H_
#define SHUTTLE_TRANSFORMS_OBSERVER_H_

#include <cstdint>
#include <memory>
#include <string>

#include "shuttle/IR/ShuttleAttrs.h"

namespace mlir::shuttle {

enum class ExecutionMode {
  StablehloRoundTrip,
  CpuExecutableBundle,
};

struct ShuttlePipelineOptions {
  NumericalPolicy numerics = NumericalPolicy::SourceOrdered;
  ExecutionMode executionMode = ExecutionMode::StablehloRoundTrip;
  std::string canonicalOptions =
      R"json({"execution_mode":"stablehlo_round_trip","numerics":"source_ordered","pipeline_abi_version":9,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  std::string canonicalTuning =
      R"json({"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";
};

struct ShuttlePipelineIdentity {
  std::string policy;
  std::string policyDigest;
  std::string tuningDigest;

  friend bool operator==(const ShuttlePipelineIdentity &left,
                         const ShuttlePipelineIdentity &right) {
    return left.policy == right.policy &&
           left.policyDigest == right.policyDigest &&
           left.tuningDigest == right.tuningDigest;
  }
  friend bool operator!=(const ShuttlePipelineIdentity &left,
                         const ShuttlePipelineIdentity &right) {
    return !(left == right);
  }
};

ShuttlePipelineIdentity
shuttlePipelineIdentity(const ShuttlePipelineOptions &options);

enum class ShuttlePipelinePhase {
  AlgebraCoverage,
  LoweredCoverage,
  FinalErasure,
  Failure,
};

namespace detail {
class ShuttleObserverInvocation;
}

class ShuttlePipelineEvent {
public:
  uint64_t invocationId() const { return invocationIdValue; }
  ShuttlePipelinePhase phase() const { return phaseValue; }
  const ShuttlePipelineIdentity &identity() const { return identityValue; }
  const std::string &regionMembership() const { return regionMembershipValue; }
  const std::string &coverageManifest() const { return coverageManifestValue; }
  const std::string &unsupportedFingerprint() const {
    return unsupportedFingerprintValue;
  }
  const std::string &normalizedModuleFingerprint() const {
    return normalizedModuleFingerprintValue;
  }
  bool noShuttleSemantics() const { return noShuttleSemanticsValue; }
  const std::string &failurePass() const { return failurePassValue; }

private:
  friend class detail::ShuttleObserverInvocation;

  ShuttlePipelineEvent(uint64_t invocationId, ShuttlePipelinePhase phase,
                       ShuttlePipelineIdentity identity,
                       std::string regionMembership,
                       std::string coverageManifest,
                       std::string unsupportedFingerprint,
                       std::string normalizedModuleFingerprint,
                       bool noShuttleSemantics, std::string failurePass);

  uint64_t invocationIdValue;
  ShuttlePipelinePhase phaseValue;
  ShuttlePipelineIdentity identityValue;
  std::string regionMembershipValue;
  std::string coverageManifestValue;
  std::string unsupportedFingerprintValue;
  std::string normalizedModuleFingerprintValue;
  bool noShuttleSemanticsValue;
  std::string failurePassValue;
};

class ShuttlePipelineObserver {
public:
  virtual ~ShuttlePipelineObserver() = default;
  virtual void observe(const ShuttlePipelineEvent &event) const = 0;
};

inline constexpr char kShuttleObserverReentrantTeardownDiagnostic[] =
    "Shuttle observer subscription captured by the current invocation cannot "
    "be destroyed or move-assigned from an observer callback";

// A subscription is intentionally separate from ShuttlePipelineOptions and
// shuttlePipelineIdentity. Installing an observer cannot alter compilation or
// executable-cache identity. Destruction removes the observer from future
// invocations and waits for invocations that captured the subscription.
//
// Precondition: an observer callback must not destroy or move-assign any
// subscription captured by its current invocation. This includes another
// subscription for the same observer. The precondition is checked in every
// build and terminates with kShuttleObserverReentrantTeardownDiagnostic instead
// of waiting on the current invocation.
class ShuttleObserverSubscription {
public:
  ShuttleObserverSubscription();
  ShuttleObserverSubscription(ShuttleObserverSubscription &&) noexcept;
  ShuttleObserverSubscription &
  operator=(ShuttleObserverSubscription &&) noexcept;
  ~ShuttleObserverSubscription();

  ShuttleObserverSubscription(const ShuttleObserverSubscription &) = delete;
  ShuttleObserverSubscription &
  operator=(const ShuttleObserverSubscription &) = delete;

  explicit operator bool() const { return implementation != nullptr; }

private:
  class Impl;
  explicit ShuttleObserverSubscription(std::unique_ptr<Impl> implementation);
  friend ShuttleObserverSubscription subscribeShuttlePipelineObserver(
      std::shared_ptr<const ShuttlePipelineObserver> observer);

  std::unique_ptr<Impl> implementation;
};

[[nodiscard]] ShuttleObserverSubscription subscribeShuttlePipelineObserver(
    std::shared_ptr<const ShuttlePipelineObserver> observer);

} // namespace mlir::shuttle

#endif // SHUTTLE_TRANSFORMS_OBSERVER_H_
