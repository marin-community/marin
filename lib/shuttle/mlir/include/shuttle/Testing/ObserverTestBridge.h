// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_TESTING_OBSERVERTESTBRIDGE_H_
#define SHUTTLE_TESTING_OBSERVERTESTBRIDGE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlir::shuttle::testing {

// Immutable copy of one native observer event for CPU jaxlib acceptance tests.
struct ShuttleObserverTestEvent {
  uint64_t invocationId;
  std::string phase;
  std::string policy;
  std::string policyDigest;
  std::string tuningDigest;
  std::string regionMembership;
  std::string coverageManifest;
  std::string unsupportedFingerprint;
  std::string normalizedModuleFingerprint;
  bool noShuttleSemantics;
  std::string failurePass;
};

// A test-only scoped capture. Its subscription is separate from pipeline
// options and cache identity. Closing is idempotent and retains copied records.
class ShuttleObserverTestCapture {
public:
  ShuttleObserverTestCapture();
  ~ShuttleObserverTestCapture();

  ShuttleObserverTestCapture(const ShuttleObserverTestCapture &) = delete;
  ShuttleObserverTestCapture &
  operator=(const ShuttleObserverTestCapture &) = delete;

  void close();
  std::vector<ShuttleObserverTestEvent> snapshot() const;

private:
  class Impl;
  std::unique_ptr<Impl> implementation;
};

std::unique_ptr<ShuttleObserverTestCapture>
subscribeShuttleObserverForTesting();

} // namespace mlir::shuttle::testing

#endif // SHUTTLE_TESTING_OBSERVERTESTBRIDGE_H_
