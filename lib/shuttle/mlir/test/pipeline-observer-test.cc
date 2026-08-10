// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "lib/Transforms/ObserverTestInternal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "shuttle/IR/ShuttleDialect.h"
#include "shuttle/Transforms/Passes.h"
#include "shuttle/Transforms/XlaRegistration.h"
#include "stablehlo/dialect/Register.h"
#include "llvm/Support/raw_ostream.h"

namespace {

using Event = mlir::shuttle::ShuttlePipelineEvent;
using EventsByInvocation = std::map<uint64_t, std::vector<Event>>;

constexpr llvm::StringLiteral kProgram = R"mlir(
module @left {
  func.func @first(%arg0: tensor<7xf32>) -> tensor<7xf32> {
    %0 = stablehlo.tanh %arg0 : tensor<7xf32>
    %1 = stablehlo.negate %0 : tensor<7xf32>
    return %1 : tensor<7xf32>
  }
}
)mlir";

constexpr llvm::StringLiteral kRenamedProgram = R"mlir(
module @renamed {
  func.func @second(%input: tensor<7xf32>) -> tensor<7xf32> {
    %mapped = stablehlo.tanh %input : tensor<7xf32>
    %result = stablehlo.negate %mapped : tensor<7xf32>
    return %result : tensor<7xf32>
  }
}
)mlir";

constexpr llvm::StringLiteral kFailingProgram = R"mlir(
module @failure {
  func.func @main(%arg0: tensor<7xf32>) -> tensor<7xf32> {
    %condition = arith.constant true
    %0 = scf.if %condition -> tensor<7xf32> {
      scf.yield %arg0 : tensor<7xf32>
    } else {
      scf.yield %arg0 : tensor<7xf32>
    }
    return %0 : tensor<7xf32>
  }
}
)mlir";

class RecordingObserver final : public mlir::shuttle::ShuttlePipelineObserver {
public:
  void observe(const Event &event) const final {
    std::lock_guard<std::mutex> lock(mutex);
    events[event.invocationId()].push_back(event);
  }

  EventsByInvocation snapshot() const {
    std::lock_guard<std::mutex> lock(mutex);
    return events;
  }

private:
  mutable std::mutex mutex;
  mutable EventsByInvocation events;
};

class BlockingObserver final : public mlir::shuttle::ShuttlePipelineObserver {
public:
  void observe(const Event &event) const final {
    std::unique_lock<std::mutex> lock(mutex);
    if (teardownReturned) {
      callbackAfterTeardownReturn = true;
    }
    phases.push_back(event.phase());
    if (event.phase() == mlir::shuttle::ShuttlePipelinePhase::AlgebraCoverage) {
      entered = true;
      condition.notify_all();
      condition.wait(lock, [&] { return callbackReleaseAllowed; });
      callbackCompleted = true;
      condition.notify_all();
    }
  }

  void waitForAlgebraCoverage() const {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return entered; });
  }

  void markTeardownWaiting(void *subscriptionState) const {
    std::lock_guard<std::mutex> lock(mutex);
    teardownSubscriptionState = subscriptionState;
    teardownWaiting = true;
    condition.notify_all();
  }

  void *waitForTeardownWaiting() const {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return teardownWaiting; });
    return teardownSubscriptionState;
  }

  bool hasTeardownReturned() const {
    std::lock_guard<std::mutex> lock(mutex);
    return teardownReturned;
  }

  void releaseCallback() const {
    std::lock_guard<std::mutex> lock(mutex);
    callbackReleaseAllowed = true;
    condition.notify_all();
  }

  void markTeardownReturned() const {
    std::lock_guard<std::mutex> lock(mutex);
    teardownReturnedAfterCallbackCompletion = callbackCompleted;
    teardownReturned = true;
    condition.notify_all();
  }

  bool returnedAfterCallbackCompletion() const {
    std::lock_guard<std::mutex> lock(mutex);
    return teardownReturnedAfterCallbackCompletion;
  }

  bool callbackBeganAfterTeardownReturn() const {
    std::lock_guard<std::mutex> lock(mutex);
    return callbackAfterTeardownReturn;
  }

  std::vector<mlir::shuttle::ShuttlePipelinePhase> snapshot() const {
    std::lock_guard<std::mutex> lock(mutex);
    return phases;
  }

private:
  mutable std::mutex mutex;
  mutable std::condition_variable condition;
  mutable std::vector<mlir::shuttle::ShuttlePipelinePhase> phases;
  mutable bool entered = false;
  mutable bool teardownWaiting = false;
  mutable bool callbackReleaseAllowed = false;
  mutable bool callbackCompleted = false;
  mutable bool teardownReturned = false;
  mutable bool teardownReturnedAfterCallbackCompletion = false;
  mutable bool callbackAfterTeardownReturn = false;
  mutable void *teardownSubscriptionState = nullptr;
};

void notifyTeardownWait(void *context, void *subscriptionState) {
  static_cast<BlockingObserver *>(context)->markTeardownWaiting(
      subscriptionState);
}

class ScopedTeardownWaitHook {
public:
  explicit ScopedTeardownWaitHook(BlockingObserver *observer) {
    mlir::shuttle::detail::setShuttleObserverTeardownWaitHookForTesting(
        notifyTeardownWait, observer);
  }

  ~ScopedTeardownWaitHook() {
    mlir::shuttle::detail::setShuttleObserverTeardownWaitHookForTesting(
        nullptr, nullptr);
  }
};

enum class SelfTeardownAction {
  Destroy,
  MoveAssign,
};

enum class ReentrantTeardownCase {
  DestroyCurrent,
  MoveAssignCurrent,
  DestroyCapturedSibling,
  DestroyCapturedFromDirectObserver,
};

class SelfTeardownObserver final
    : public mlir::shuttle::ShuttlePipelineObserver {
public:
  SelfTeardownObserver(
      std::optional<mlir::shuttle::ShuttleObserverSubscription> *subscription,
      SelfTeardownAction action)
      : subscription(subscription), action(action) {}

  void observe(const Event &) const final {
    if (action == SelfTeardownAction::Destroy) {
      subscription->reset();
      return;
    }
    subscription->value() = mlir::shuttle::ShuttleObserverSubscription{};
  }

private:
  std::optional<mlir::shuttle::ShuttleObserverSubscription> *subscription;
  SelfTeardownAction action;
};

struct PipelineResult {
  bool succeeded;
  std::string normalizedFingerprint;
};

PipelineResult
runPipeline(llvm::StringRef source,
            const mlir::shuttle::ShuttlePipelineOptions &options,
            std::shared_ptr<const mlir::shuttle::ShuttlePipelineObserver>
                directObserver = {},
            bool throughXlaCallback = false) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect, mlir::shuttle::ShuttleDialect>();
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module) {
    llvm::errs() << "failed to parse observer fixture\n";
    return {false, {}};
  }
  mlir::LogicalResult result = mlir::failure();
  if (throughXlaCallback) {
    if (directObserver) {
      llvm::errs() << "XLA callback fixture cannot use a direct observer\n";
      return {false, {}};
    }
    result = mlir::shuttle::runShuttleXlaTransform(*module, options);
  } else {
    mlir::PassManager manager(&context);
    mlir::shuttle::buildShuttleStablehloPipeline(manager, options,
                                                 std::move(directObserver));
    result = manager.run(*module);
  }
  if (mlir::failed(result)) {
    return {false, {}};
  }
  return {true, mlir::shuttle::normalizedStablehloFingerprint(*module)};
}

bool runSelfTeardown(SelfTeardownAction action) {
  std::optional<mlir::shuttle::ShuttleObserverSubscription> subscription;
  auto observer = std::make_shared<SelfTeardownObserver>(&subscription, action);
  subscription.emplace(
      mlir::shuttle::subscribeShuttlePipelineObserver(observer));
  return runPipeline(kProgram, {}).succeeded;
}

bool runCrossSubscriptionTeardown() {
  std::optional<mlir::shuttle::ShuttleObserverSubscription> firstSubscription;
  std::optional<mlir::shuttle::ShuttleObserverSubscription> secondSubscription;
  auto observer = std::make_shared<SelfTeardownObserver>(
      &secondSubscription, SelfTeardownAction::Destroy);
  firstSubscription.emplace(
      mlir::shuttle::subscribeShuttlePipelineObserver(observer));
  secondSubscription.emplace(
      mlir::shuttle::subscribeShuttlePipelineObserver(observer));
  return runPipeline(kProgram, {}).succeeded;
}

bool runDirectObserverCapturedTeardown() {
  std::optional<mlir::shuttle::ShuttleObserverSubscription> subscription;
  auto registeredObserver = std::make_shared<RecordingObserver>();
  subscription.emplace(
      mlir::shuttle::subscribeShuttlePipelineObserver(registeredObserver));
  auto directObserver = std::make_shared<SelfTeardownObserver>(
      &subscription, SelfTeardownAction::Destroy);
  return runPipeline(kProgram, {}, directObserver).succeeded;
}

bool expectReentrantTeardownDeath(ReentrantTeardownCase teardownCase) {
  int diagnostics[2];
  if (pipe(diagnostics) != 0) {
    llvm::errs() << "failed to create death-test diagnostic pipe\n";
    return false;
  }
  pid_t process = fork();
  if (process == -1) {
    close(diagnostics[0]);
    close(diagnostics[1]);
    llvm::errs() << "failed to fork observer death test\n";
    return false;
  }
  if (process == 0) {
    close(diagnostics[0]);
    if (dup2(diagnostics[1], STDERR_FILENO) == -1) {
      _exit(125);
    }
    close(diagnostics[1]);
    alarm(5);
    bool succeeded;
    if (teardownCase ==
        ReentrantTeardownCase::DestroyCapturedFromDirectObserver) {
      succeeded = runDirectObserverCapturedTeardown();
    } else if (teardownCase == ReentrantTeardownCase::DestroyCapturedSibling) {
      succeeded = runCrossSubscriptionTeardown();
    } else {
      SelfTeardownAction action =
          teardownCase == ReentrantTeardownCase::DestroyCurrent
              ? SelfTeardownAction::Destroy
              : SelfTeardownAction::MoveAssign;
      succeeded = runSelfTeardown(action);
    }
    _exit(succeeded ? 0 : 1);
  }

  close(diagnostics[1]);
  std::string output;
  std::array<char, 256> buffer;
  while (true) {
    ssize_t count = read(diagnostics[0], buffer.data(), buffer.size());
    if (count > 0) {
      output.append(buffer.data(), static_cast<size_t>(count));
      continue;
    }
    if (count == -1 && errno == EINTR) {
      continue;
    }
    break;
  }
  close(diagnostics[0]);

  int status = 0;
  while (waitpid(process, &status, 0) == -1 && errno == EINTR) {
  }
  bool terminatedFatally = (WIFEXITED(status) && WEXITSTATUS(status) != 0) ||
                           (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
  if (!terminatedFatally ||
      output.find(mlir::shuttle::kShuttleObserverReentrantTeardownDiagnostic) ==
          std::string::npos) {
    llvm::errs()
        << "reentrant teardown did not fail with the stable diagnostic\n"
        << output;
    return false;
  }
  return true;
}

bool checkSuccessfulEvents(
    const std::vector<Event> &events,
    const mlir::shuttle::ShuttlePipelineIdentity &identity) {
  using mlir::shuttle::ShuttlePipelinePhase;
  if (events.size() != 3 ||
      events[0].phase() != ShuttlePipelinePhase::AlgebraCoverage ||
      events[1].phase() != ShuttlePipelinePhase::LoweredCoverage ||
      events[2].phase() != ShuttlePipelinePhase::FinalErasure) {
    llvm::errs() << "observer phases are incomplete or out of order\n";
    return false;
  }
  for (const Event &event : events) {
    if (event.invocationId() != events[0].invocationId() ||
        event.identity() != identity || !event.failurePass().empty()) {
      llvm::errs() << "observer invocation identity changed between phases\n";
      return false;
    }
  }
  if (events[0].regionMembership().empty() ||
      events[0].regionMembership() != events[1].regionMembership() ||
      events[0].coverageManifest().empty() ||
      events[1].coverageManifest().empty() ||
      events[0].coverageManifest().find(identity.policyDigest) ==
          std::string::npos ||
      events[0].coverageManifest().find(identity.tuningDigest) ==
          std::string::npos ||
      events[0].unsupportedFingerprint().empty() ||
      events[0].unsupportedFingerprint() !=
          events[1].unsupportedFingerprint() ||
      !events[0].normalizedModuleFingerprint().empty() ||
      !events[1].normalizedModuleFingerprint().empty() ||
      events[0].noShuttleSemantics() || events[1].noShuttleSemantics() ||
      !events[2].regionMembership().empty() ||
      !events[2].coverageManifest().empty() ||
      !events[2].unsupportedFingerprint().empty() ||
      !events[2].noShuttleSemantics() ||
      events[2].normalizedModuleFingerprint().empty()) {
    llvm::errs() << "observer snapshots violate provenance lifetime\n";
    return false;
  }
  return true;
}

bool checkSuccessAndCacheNeutrality() {
  auto observer = std::make_shared<RecordingObserver>();
  mlir::shuttle::ShuttlePipelineOptions sourceOrdered;
  mlir::shuttle::ShuttlePipelineOptions fast;
  fast.numerics = mlir::shuttle::NumericalPolicy::Fast;
  mlir::shuttle::ShuttlePipelineOptions tuned = fast;
  tuned.canonicalTuning = R"json({"tile":2})json";
  const auto sourceIdentity =
      mlir::shuttle::shuttlePipelineIdentity(sourceOrdered);
  const auto fastIdentity = mlir::shuttle::shuttlePipelineIdentity(fast);
  const auto tunedIdentity = mlir::shuttle::shuttlePipelineIdentity(tuned);

  std::array<PipelineResult, 4> results;
  {
    auto subscription =
        mlir::shuttle::subscribeShuttlePipelineObserver(observer);
    results = {runPipeline(kProgram, sourceOrdered),
               runPipeline(kRenamedProgram, sourceOrdered),
               runPipeline(kProgram, fast),
               runPipeline(kProgram, tuned, {}, true)};
  }
  for (const PipelineResult &result : results) {
    if (!result.succeeded) {
      llvm::errs() << "successful observer fixture failed\n";
      return false;
    }
  }

  EventsByInvocation invocations = observer->snapshot();
  if (invocations.size() != results.size()) {
    llvm::errs() << "observer did not produce one record per invocation\n";
    return false;
  }
  auto iterator = invocations.begin();
  const auto &sourceEvents = iterator++->second;
  const auto &renamedEvents = iterator++->second;
  const auto &fastEvents = iterator++->second;
  const auto &tunedEvents = iterator->second;
  if (!checkSuccessfulEvents(sourceEvents, sourceIdentity) ||
      !checkSuccessfulEvents(renamedEvents, sourceIdentity) ||
      !checkSuccessfulEvents(fastEvents, fastIdentity) ||
      !checkSuccessfulEvents(tunedEvents, tunedIdentity)) {
    return false;
  }
  if (sourceEvents[0].regionMembership() !=
          renamedEvents[0].regionMembership() ||
      sourceEvents[0].unsupportedFingerprint() !=
          renamedEvents[0].unsupportedFingerprint() ||
      results[0].normalizedFingerprint != results[1].normalizedFingerprint ||
      sourceEvents[2].normalizedModuleFingerprint() !=
          results[0].normalizedFingerprint ||
      renamedEvents[2].normalizedModuleFingerprint() !=
          results[1].normalizedFingerprint ||
      fastEvents[2].normalizedModuleFingerprint() !=
          results[2].normalizedFingerprint ||
      tunedEvents[2].normalizedModuleFingerprint() !=
          results[3].normalizedFingerprint ||
      sourceIdentity == fastIdentity || fastIdentity == tunedIdentity ||
      results[0].normalizedFingerprint != results[2].normalizedFingerprint) {
    llvm::errs() << "observer identity depends on names or merges policies\n";
    return false;
  }

  const auto identityWithoutObserver =
      mlir::shuttle::shuttlePipelineIdentity(sourceOrdered);
  PipelineResult unobserved = runPipeline(kProgram, sourceOrdered);
  if (!unobserved.succeeded || identityWithoutObserver != sourceIdentity ||
      unobserved.normalizedFingerprint != results[0].normalizedFingerprint ||
      observer->snapshot().size() != invocations.size()) {
    llvm::errs() << "observer subscription changed compilation identity\n";
    return false;
  }
  return true;
}

bool checkDirectObserver() {
  auto observer = std::make_shared<RecordingObserver>();
  PipelineResult result = runPipeline(kProgram, {}, observer);
  EventsByInvocation invocations = observer->snapshot();
  if (!result.succeeded || invocations.size() != 1 ||
      !checkSuccessfulEvents(invocations.begin()->second,
                             mlir::shuttle::shuttlePipelineIdentity({}))) {
    llvm::errs() << "direct observer was mistaken for a scoped subscription\n";
    return false;
  }
  return true;
}

bool checkFailureEvent() {
  auto observer = std::make_shared<RecordingObserver>();
  {
    auto subscription =
        mlir::shuttle::subscribeShuttlePipelineObserver(observer);
    if (runPipeline(kFailingProgram, {}, {}, true).succeeded) {
      llvm::errs() << "failing observer fixture unexpectedly succeeded\n";
      return false;
    }
  }
  EventsByInvocation invocations = observer->snapshot();
  if (invocations.size() != 1 || invocations.begin()->second.size() != 1) {
    llvm::errs() << "failed pipeline did not emit one terminal record\n";
    return false;
  }
  const Event &failure = invocations.begin()->second.front();
  if (failure.phase() != mlir::shuttle::ShuttlePipelinePhase::Failure ||
      failure.identity() != mlir::shuttle::shuttlePipelineIdentity({}) ||
      failure.failurePass() != "shuttle-form-structural-regions" ||
      failure.noShuttleSemantics() ||
      !failure.normalizedModuleFingerprint().empty()) {
    llvm::errs() << "failure record does not identify terminal pass failure\n";
    return false;
  }
  if (!runPipeline(kProgram, {}).succeeded ||
      observer->snapshot().size() != invocations.size()) {
    llvm::errs() << "observer received callbacks after scope teardown\n";
    return false;
  }
  return true;
}

bool checkConcurrentInvocations() {
  constexpr unsigned kThreads = 8;
  auto observer = std::make_shared<RecordingObserver>();
  auto subscription = mlir::shuttle::subscribeShuttlePipelineObserver(observer);
  std::atomic<unsigned> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (unsigned index = 0; index < kThreads; ++index) {
    threads.emplace_back([&, index] {
      mlir::shuttle::ShuttlePipelineOptions options;
      options.numerics = index % 2 == 0
                             ? mlir::shuttle::NumericalPolicy::SourceOrdered
                             : mlir::shuttle::NumericalPolicy::Fast;
      if (!runPipeline(kProgram, options).succeeded) {
        failures.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (std::thread &thread : threads) {
    thread.join();
  }
  if (failures.load(std::memory_order_relaxed) != 0) {
    llvm::errs() << "concurrent observer pipeline failed\n";
    return false;
  }

  EventsByInvocation invocations = observer->snapshot();
  if (invocations.size() != kThreads) {
    llvm::errs() << "concurrent invocations reused observer IDs\n";
    return false;
  }
  unsigned sourceOrdered = 0;
  unsigned fast = 0;
  for (const auto &[invocationId, events] : invocations) {
    (void)invocationId;
    if (events.empty()) {
      return false;
    }
    const auto &identity = events.front().identity();
    if (!checkSuccessfulEvents(events, identity)) {
      return false;
    }
    if (identity.policy == "source_ordered") {
      ++sourceOrdered;
    } else if (identity.policy == "fast") {
      ++fast;
    }
  }
  if (sourceOrdered != kThreads / 2 || fast != kThreads / 2) {
    llvm::errs() << "concurrent observer records crossed policy identity\n";
    return false;
  }
  return true;
}

bool checkScopedTeardownWaitsForCapturedInvocation() {
  auto observer = std::make_shared<BlockingObserver>();
  ScopedTeardownWaitHook waitHook(observer.get());
  auto subscription = mlir::shuttle::subscribeShuttlePipelineObserver(observer);
  std::atomic<bool> pipelineSucceeded{false};
  std::thread pipeline([&] {
    pipelineSucceeded.store(runPipeline(kProgram, {}).succeeded,
                            std::memory_order_release);
  });
  observer->waitForAlgebraCoverage();
  auto lateObserver = std::make_shared<RecordingObserver>();
  auto lateSubscription =
      mlir::shuttle::subscribeShuttlePipelineObserver(lateObserver);
  std::thread teardown(
      [subscription = std::move(subscription), observer]() mutable {
        subscription = {};
        observer->markTeardownReturned();
      });
  void *teardownSubscriptionState = observer->waitForTeardownWaiting();
  bool teardownIsWaiting =
      mlir::shuttle::detail::confirmShuttleObserverTeardownWaitForTesting(
          teardownSubscriptionState);
  if (!teardownIsWaiting) {
    llvm::errs() << "teardown did not enter the captured-invocation wait\n";
    observer->releaseCallback();
    pipeline.join();
    teardown.join();
    return false;
  }
  if (observer->hasTeardownReturned()) {
    llvm::errs() << "teardown returned while a captured callback was blocked\n";
    observer->releaseCallback();
    pipeline.join();
    teardown.join();
    return false;
  }
  observer->releaseCallback();
  pipeline.join();
  teardown.join();
  if (!pipelineSucceeded.load(std::memory_order_acquire)) {
    llvm::errs() << "pipeline failed during scoped observer teardown\n";
    return false;
  }
  if (!observer->returnedAfterCallbackCompletion()) {
    llvm::errs()
        << "teardown returned before the captured callback completed\n";
    return false;
  }
  if (observer->callbackBeganAfterTeardownReturn()) {
    llvm::errs() << "observer callback began after teardown returned\n";
    return false;
  }
  const auto phases = observer->snapshot();
  if (phases != std::vector<mlir::shuttle::ShuttlePipelinePhase>{
                    mlir::shuttle::ShuttlePipelinePhase::AlgebraCoverage,
                    mlir::shuttle::ShuttlePipelinePhase::LoweredCoverage,
                    mlir::shuttle::ShuttlePipelinePhase::FinalErasure}) {
    llvm::errs() << "teardown dropped callbacks captured at invocation start\n";
    return false;
  }
  if (!lateObserver->snapshot().empty()) {
    llvm::errs() << "late subscription entered an active invocation\n";
    return false;
  }
  if (!runPipeline(kProgram, {}).succeeded || observer->snapshot() != phases ||
      lateObserver->snapshot().size() != 1) {
    llvm::errs() << "teardown allowed callbacks from a later invocation\n";
    return false;
  }
  return true;
}

} // namespace

int main() {
  if (!expectReentrantTeardownDeath(ReentrantTeardownCase::DestroyCurrent) ||
      !expectReentrantTeardownDeath(ReentrantTeardownCase::MoveAssignCurrent) ||
      !expectReentrantTeardownDeath(
          ReentrantTeardownCase::DestroyCapturedSibling) ||
      !expectReentrantTeardownDeath(
          ReentrantTeardownCase::DestroyCapturedFromDirectObserver) ||
      !checkDirectObserver() || !checkSuccessAndCacheNeutrality() ||
      !checkFailureEvent() || !checkConcurrentInvocations() ||
      !checkScopedTeardownWaitsForCapturedInvocation()) {
    return 1;
  }
  return 0;
}
