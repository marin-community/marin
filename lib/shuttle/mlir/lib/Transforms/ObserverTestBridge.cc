// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Testing/ObserverTestBridge.h"

#include <condition_variable>
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
    std::unique_lock<std::mutex> lock(mutex);
    const ShuttlePipelineIdentity &identity = event.identity();
    events.push_back(ShuttleObserverTestEvent{
        event.invocationId(), phaseName(event.phase()), identity.policy,
        identity.policyDigest, identity.tuningDigest,
        event.regionMembership(), event.coverageManifest(),
        event.unsupportedFingerprint(), event.normalizedModuleFingerprint(),
        event.noShuttleSemantics(), event.failurePass()});
    if (blockNextCallback) {
      blockNextCallback = false;
      callbackBlocked = true;
      condition.notify_all();
      condition.wait(lock, [&] { return callbackReleaseAllowed; });
    }
  }

  std::vector<ShuttleObserverTestEvent> snapshot() const {
    std::lock_guard<std::mutex> lock(mutex);
    return events;
  }

  void blockNextForTesting() {
    std::lock_guard<std::mutex> lock(mutex);
    blockNextCallback = true;
    callbackBlocked = false;
    callbackReleaseAllowed = false;
  }

  void waitForBlockedCallbackForTesting() const {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return callbackBlocked; });
  }

  void releaseBlockedCallbackForTesting() {
    std::lock_guard<std::mutex> lock(mutex);
    callbackReleaseAllowed = true;
    condition.notify_all();
  }

private:
  mutable std::mutex mutex;
  mutable std::condition_variable condition;
  mutable std::vector<ShuttleObserverTestEvent> events;
  mutable bool blockNextCallback = false;
  mutable bool callbackBlocked = false;
  mutable bool callbackReleaseAllowed = false;
};

} // namespace

class ShuttleObserverTestCapture::Impl {
public:
  Impl() : observer(std::make_shared<RecordingObserver>()),
           subscription(subscribeShuttlePipelineObserver(observer)) {}

  void close() {
    std::optional<ShuttleObserverSubscription> closing;
    std::unique_lock<std::mutex> lock(mutex);
    ++closeCallers;
    condition.notify_all();
    if (closeState == CloseState::Closed) {
      --closeCallers;
      return;
    }
    if (closeState == CloseState::Closing) {
      condition.wait(lock, [&] { return closeState == CloseState::Closed; });
      --closeCallers;
      return;
    }
    closeState = CloseState::Closing;
    closing.swap(subscription);
    lock.unlock();
    closing.reset();
    lock.lock();
    closeState = CloseState::Closed;
    condition.notify_all();
    --closeCallers;
  }

  std::vector<ShuttleObserverTestEvent> snapshot() const {
    return observer->snapshot();
  }

  void blockNextCallbackForTesting() { observer->blockNextForTesting(); }

  void waitForBlockedCallbackForTesting() const {
    observer->waitForBlockedCallbackForTesting();
  }

  void waitForCloseCallersForTesting(std::size_t count) const {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return closeCallers >= count; });
  }

  void releaseBlockedCallbackForTesting() {
    observer->releaseBlockedCallbackForTesting();
  }

private:
  enum class CloseState { Open, Closing, Closed };

  mutable std::mutex mutex;
  mutable std::condition_variable condition;
  std::shared_ptr<RecordingObserver> observer;
  std::optional<ShuttleObserverSubscription> subscription;
  CloseState closeState = CloseState::Open;
  std::size_t closeCallers = 0;
};

ShuttleObserverTestCapture::ShuttleObserverTestCapture()
    : implementation(std::make_unique<Impl>()) {}

ShuttleObserverTestCapture::~ShuttleObserverTestCapture() { close(); }

void ShuttleObserverTestCapture::close() { implementation->close(); }

std::vector<ShuttleObserverTestEvent>
ShuttleObserverTestCapture::snapshot() const {
  return implementation->snapshot();
}

void ShuttleObserverTestCapture::blockNextCallbackForTesting() {
  implementation->blockNextCallbackForTesting();
}

void ShuttleObserverTestCapture::waitForBlockedCallbackForTesting() const {
  implementation->waitForBlockedCallbackForTesting();
}

void ShuttleObserverTestCapture::waitForCloseCallersForTesting(
    std::size_t count) const {
  implementation->waitForCloseCallersForTesting(count);
}

void ShuttleObserverTestCapture::releaseBlockedCallbackForTesting() {
  implementation->releaseBlockedCallbackForTesting();
}

std::unique_ptr<ShuttleObserverTestCapture>
subscribeShuttleObserverForTesting() {
  return std::make_unique<ShuttleObserverTestCapture>();
}

} // namespace mlir::shuttle::testing
