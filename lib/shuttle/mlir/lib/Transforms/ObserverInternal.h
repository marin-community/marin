// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_TRANSFORMS_OBSERVERINTERNAL_H_
#define SHUTTLE_TRANSFORMS_OBSERVERINTERNAL_H_

#include <memory>
#include <string>

#include "shuttle/Transforms/Observer.h"

namespace mlir::shuttle::detail {

class ShuttleObserverInvocation {
public:
  ~ShuttleObserverInvocation();

  void emit(ShuttlePipelinePhase phase, std::string regionMembership,
            std::string coverageManifest, std::string unsupportedFingerprint,
            std::string normalizedModuleFingerprint, bool noShuttleSemantics,
            std::string failurePass = {}) const;

private:
  class Impl;
  explicit ShuttleObserverInvocation(std::unique_ptr<Impl> implementation);
  friend std::shared_ptr<ShuttleObserverInvocation>
  beginShuttleObserverInvocation(
      ShuttlePipelineIdentity identity,
      std::shared_ptr<const ShuttlePipelineObserver> directObserver);

  std::unique_ptr<Impl> implementation;
};

std::shared_ptr<ShuttleObserverInvocation> beginShuttleObserverInvocation(
    ShuttlePipelineIdentity identity,
    std::shared_ptr<const ShuttlePipelineObserver> directObserver);

} // namespace mlir::shuttle::detail

#endif // SHUTTLE_TRANSFORMS_OBSERVERINTERNAL_H_
