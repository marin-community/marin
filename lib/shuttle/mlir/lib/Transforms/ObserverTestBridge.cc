// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Testing/ObserverTestBridge.h"

#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "shuttle/Transforms/Observer.h"

namespace mlir::shuttle::testing {
namespace {

const char *phaseName(ShuttlePipelinePhase phase) {
  switch (phase) {
  case ShuttlePipelinePhase::AlgebraCoverage:
    return "algebra_coverage";
  case ShuttlePipelinePhase::LoweredCoverage:
    return "lowered_coverage";
  case ShuttlePipelinePhase::FinalErasure:
    return "final_erasure";
  case ShuttlePipelinePhase::Failure:
    return "failure";
  }
  return "unknown";
}

class RecordingObserver final : public ShuttlePipelineObserver {
public:
  void observe(const ShuttlePipelineEvent &event) const final {
    std::lock_guard<std::mutex> lock(mutex);
    const ShuttlePipelineIdentity &identity = event.identity();
    events.push_back(ShuttleObserverTestEvent{
        event.invocationId(), phaseName(event.phase()), identity.policy,
        identity.policyDigest, identity.tuningDigest,
        event.regionMembership(), event.coverageManifest(),
        event.unsupportedFingerprint(), event.normalizedModuleFingerprint(),
        event.noShuttleSemantics(), event.failurePass()});
  }

  std::vector<ShuttleObserverTestEvent> snapshot() const {
    std::lock_guard<std::mutex> lock(mutex);
    return events;
  }

private:
  mutable std::mutex mutex;
  mutable std::vector<ShuttleObserverTestEvent> events;
};

} // namespace

class ShuttleObserverTestCapture::Impl {
public:
  Impl() : observer(std::make_shared<RecordingObserver>()),
           subscription(subscribeShuttlePipelineObserver(observer)) {}

  void close() {
    std::optional<ShuttleObserverSubscription> closing;
    {
      std::lock_guard<std::mutex> lock(mutex);
      closing.swap(subscription);
    }
    closing.reset();
  }

  std::vector<ShuttleObserverTestEvent> snapshot() const {
    return observer->snapshot();
  }

private:
  std::mutex mutex;
  std::shared_ptr<RecordingObserver> observer;
  std::optional<ShuttleObserverSubscription> subscription;
};

ShuttleObserverTestCapture::ShuttleObserverTestCapture()
    : implementation(std::make_unique<Impl>()) {}

ShuttleObserverTestCapture::~ShuttleObserverTestCapture() { close(); }

void ShuttleObserverTestCapture::close() { implementation->close(); }

std::vector<ShuttleObserverTestEvent>
ShuttleObserverTestCapture::snapshot() const {
  return implementation->snapshot();
}

std::unique_ptr<ShuttleObserverTestCapture>
subscribeShuttleObserverForTesting() {
  return std::make_unique<ShuttleObserverTestCapture>();
}

} // namespace mlir::shuttle::testing
