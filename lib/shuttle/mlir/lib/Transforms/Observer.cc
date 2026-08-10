// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Transforms/Observer.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "ObserverInternal.h"
#include "ObserverTestInternal.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SHA256.h"

namespace mlir::shuttle {
namespace {

std::string sha256(llvm::StringRef value) {
  llvm::SHA256 digest;
  digest.update(value);
  return llvm::toHex(digest.final(), true);
}

llvm::StringRef policyName(NumericalPolicy numerics) {
  return numerics == NumericalPolicy::SourceOrdered ? "source_ordered" : "fast";
}

struct TeardownWaitHookState {
  std::mutex mutex;
  detail::ShuttleObserverTeardownWaitHook hook = nullptr;
  void *context = nullptr;
};

TeardownWaitHookState &teardownWaitHookState() {
  static auto *state = new TeardownWaitHookState();
  return *state;
}

void notifyTeardownWaitForTesting(void *subscriptionState) {
  TeardownWaitHookState &state = teardownWaitHookState();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.hook != nullptr) {
    state.hook(state.context, subscriptionState);
  }
}

class ObserverSubscriptionState;

struct CapturedSubscriptionCallback {
  const ObserverSubscriptionState *state;
  const CapturedSubscriptionCallback *previous;
};

thread_local const CapturedSubscriptionCallback *capturedSubscriptionCallback =
    nullptr;

bool isCapturedByCurrentCallback(const ObserverSubscriptionState *state) {
  for (const CapturedSubscriptionCallback *callback =
           capturedSubscriptionCallback;
       callback != nullptr; callback = callback->previous) {
    if (callback->state == state) {
      return true;
    }
  }
  return false;
}

class ScopedCapturedSubscriptions {
public:
  explicit ScopedCapturedSubscriptions(
      const std::vector<
          std::pair<std::shared_ptr<ObserverSubscriptionState>,
                    std::shared_ptr<const ShuttlePipelineObserver>>>
          &subscriptions)
      : previous(capturedSubscriptionCallback) {
    callbacks.reserve(subscriptions.size());
    for (const auto &subscription : subscriptions) {
      callbacks.push_back(CapturedSubscriptionCallback{
          subscription.first.get(), capturedSubscriptionCallback});
      capturedSubscriptionCallback = &callbacks.back();
    }
  }

  ~ScopedCapturedSubscriptions() { capturedSubscriptionCallback = previous; }

private:
  const CapturedSubscriptionCallback *previous;
  std::vector<CapturedSubscriptionCallback> callbacks;
};

class ObserverSubscriptionState {
public:
  explicit ObserverSubscriptionState(
      std::shared_ptr<const ShuttlePipelineObserver> observer)
      : observer(std::move(observer)) {}

  std::shared_ptr<const ShuttlePipelineObserver> retain() {
    std::lock_guard<std::mutex> lock(mutex);
    if (!subscribed) {
      return {};
    }
    ++activeInvocations;
    return observer;
  }

  void release() {
    std::lock_guard<std::mutex> lock(mutex);
    assert(activeInvocations > 0);
    --activeInvocations;
    if (activeInvocations == 0) {
      condition.notify_all();
    }
  }

  void removeAndWait();
  bool confirmTeardownWaitForTesting();

private:
  std::mutex mutex;
  std::condition_variable condition;
  std::shared_ptr<const ShuttlePipelineObserver> observer;
  uint64_t activeInvocations = 0;
  bool subscribed = true;
};

void ObserverSubscriptionState::removeAndWait() {
  if (isCapturedByCurrentCallback(this)) {
    llvm::report_fatal_error(kShuttleObserverReentrantTeardownDiagnostic,
                             false);
  }
  std::unique_lock<std::mutex> lock(mutex);
  subscribed = false;
  if (activeInvocations != 0) {
    notifyTeardownWaitForTesting(this);
  }
  condition.wait(lock, [&] { return activeInvocations == 0; });
  observer.reset();
}

bool ObserverSubscriptionState::confirmTeardownWaitForTesting() {
  std::lock_guard<std::mutex> lock(mutex);
  return !subscribed && activeInvocations != 0;
}

class ObserverRegistry {
public:
  std::shared_ptr<ObserverSubscriptionState>
  add(std::shared_ptr<const ShuttlePipelineObserver> observer) {
    auto state =
        std::make_shared<ObserverSubscriptionState>(std::move(observer));
    std::lock_guard<std::mutex> lock(mutex);
    subscriptions.push_back(state);
    return state;
  }

  void remove(const std::shared_ptr<ObserverSubscriptionState> &state) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      subscriptions.erase(
          std::remove(subscriptions.begin(), subscriptions.end(), state),
          subscriptions.end());
    }
    state->removeAndWait();
  }

  std::vector<std::pair<std::shared_ptr<ObserverSubscriptionState>,
                        std::shared_ptr<const ShuttlePipelineObserver>>>
  snapshot() {
    std::vector<std::pair<std::shared_ptr<ObserverSubscriptionState>,
                          std::shared_ptr<const ShuttlePipelineObserver>>>
        result;
    std::lock_guard<std::mutex> lock(mutex);
    result.reserve(subscriptions.size());
    for (const auto &state : subscriptions) {
      if (auto observer = state->retain()) {
        result.emplace_back(state, std::move(observer));
      }
    }
    return result;
  }

private:
  std::mutex mutex;
  std::vector<std::shared_ptr<ObserverSubscriptionState>> subscriptions;
};

ObserverRegistry &observerRegistry() {
  static auto *registry = new ObserverRegistry();
  return *registry;
}

std::atomic<uint64_t> nextInvocationId{0};

} // namespace

void detail::setShuttleObserverTeardownWaitHookForTesting(
    detail::ShuttleObserverTeardownWaitHook hook, void *context) {
  TeardownWaitHookState &state = teardownWaitHookState();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.hook = hook;
  state.context = context;
}

bool detail::confirmShuttleObserverTeardownWaitForTesting(
    void *subscriptionState) {
  return static_cast<ObserverSubscriptionState *>(subscriptionState)
      ->confirmTeardownWaitForTesting();
}

ShuttlePipelineIdentity
shuttlePipelineIdentity(const ShuttlePipelineOptions &options) {
  std::string tuningDigest = sha256(options.canonicalTuning);
  std::string policy = policyName(options.numerics).str();
  return ShuttlePipelineIdentity{policy, sha256(options.canonicalOptions),
                                 std::move(tuningDigest)};
}

ShuttlePipelineEvent::ShuttlePipelineEvent(
    uint64_t invocationId, ShuttlePipelinePhase phase,
    ShuttlePipelineIdentity identity, std::string regionMembership,
    std::string coverageManifest, std::string unsupportedFingerprint,
    std::string normalizedModuleFingerprint, bool noShuttleSemantics,
    std::string failurePass)
    : invocationIdValue(invocationId), phaseValue(phase),
      identityValue(std::move(identity)),
      regionMembershipValue(std::move(regionMembership)),
      coverageManifestValue(std::move(coverageManifest)),
      unsupportedFingerprintValue(std::move(unsupportedFingerprint)),
      normalizedModuleFingerprintValue(std::move(normalizedModuleFingerprint)),
      noShuttleSemanticsValue(noShuttleSemantics),
      failurePassValue(std::move(failurePass)) {}

class ShuttleObserverSubscription::Impl {
public:
  explicit Impl(std::shared_ptr<ObserverSubscriptionState> state)
      : state(std::move(state)) {}
  ~Impl() { observerRegistry().remove(state); }

private:
  std::shared_ptr<ObserverSubscriptionState> state;
};

ShuttleObserverSubscription::ShuttleObserverSubscription() = default;
ShuttleObserverSubscription::ShuttleObserverSubscription(
    ShuttleObserverSubscription &&) noexcept = default;
ShuttleObserverSubscription &ShuttleObserverSubscription::operator=(
    ShuttleObserverSubscription &&) noexcept = default;
ShuttleObserverSubscription::~ShuttleObserverSubscription() = default;

ShuttleObserverSubscription::ShuttleObserverSubscription(
    std::unique_ptr<Impl> implementation)
    : implementation(std::move(implementation)) {}

ShuttleObserverSubscription subscribeShuttlePipelineObserver(
    std::shared_ptr<const ShuttlePipelineObserver> observer) {
  assert(observer && "a Shuttle observer subscription requires an observer");
  auto state = observerRegistry().add(std::move(observer));
  return ShuttleObserverSubscription(
      std::make_unique<ShuttleObserverSubscription::Impl>(std::move(state)));
}

namespace detail {

class ShuttleObserverInvocation::Impl {
public:
  Impl(uint64_t invocationId, ShuttlePipelineIdentity identity,
       std::vector<std::pair<std::shared_ptr<ObserverSubscriptionState>,
                             std::shared_ptr<const ShuttlePipelineObserver>>>
           subscriptions,
       std::shared_ptr<const ShuttlePipelineObserver> directObserver)
      : invocationId(invocationId), identity(std::move(identity)),
        subscriptions(std::move(subscriptions)),
        directObserver(std::move(directObserver)) {}

  ~Impl() {
    directObserver.reset();
    for (auto &subscription : subscriptions) {
      subscription.second.reset();
    }
    for (const auto &subscription : subscriptions) {
      subscription.first->release();
    }
  }

  uint64_t invocationId;
  ShuttlePipelineIdentity identity;
  std::vector<std::pair<std::shared_ptr<ObserverSubscriptionState>,
                        std::shared_ptr<const ShuttlePipelineObserver>>>
      subscriptions;
  std::shared_ptr<const ShuttlePipelineObserver> directObserver;
};

ShuttleObserverInvocation::ShuttleObserverInvocation(
    std::unique_ptr<Impl> implementation)
    : implementation(std::move(implementation)) {}
ShuttleObserverInvocation::~ShuttleObserverInvocation() = default;

void ShuttleObserverInvocation::emit(ShuttlePipelinePhase phase,
                                     std::string regionMembership,
                                     std::string coverageManifest,
                                     std::string unsupportedFingerprint,
                                     std::string normalizedModuleFingerprint,
                                     bool noShuttleSemantics,
                                     std::string failurePass) const {
  ShuttlePipelineEvent event(
      implementation->invocationId, phase, implementation->identity,
      std::move(regionMembership), std::move(coverageManifest),
      std::move(unsupportedFingerprint), std::move(normalizedModuleFingerprint),
      noShuttleSemantics, std::move(failurePass));
  ScopedCapturedSubscriptions captured(implementation->subscriptions);
  for (const auto &subscription : implementation->subscriptions) {
    subscription.second->observe(event);
  }
  if (implementation->directObserver) {
    implementation->directObserver->observe(event);
  }
}

std::shared_ptr<ShuttleObserverInvocation> beginShuttleObserverInvocation(
    ShuttlePipelineIdentity identity,
    std::shared_ptr<const ShuttlePipelineObserver> directObserver) {
  uint64_t invocationId =
      nextInvocationId.fetch_add(1, std::memory_order_relaxed);
  return std::shared_ptr<ShuttleObserverInvocation>(
      new ShuttleObserverInvocation(
          std::make_unique<ShuttleObserverInvocation::Impl>(
              invocationId, std::move(identity), observerRegistry().snapshot(),
              std::move(directObserver))));
}

} // namespace detail
} // namespace mlir::shuttle
