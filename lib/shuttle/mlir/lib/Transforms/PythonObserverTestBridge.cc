// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Testing/PythonObserverTestBridge.h"

#include <cstddef>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/stl/unique_ptr.h"
#include "shuttle/Testing/ObserverTestBridge.h"

namespace nb = nanobind;

namespace mlir::shuttle::testing {
namespace {

nb::tuple pythonRecord(const ShuttleObserverTestEvent &event) {
  nb::tuple record(11);
  record[0] = nb::int_(event.invocationId);
  record[1] = nb::str(event.phase.c_str());
  record[2] = nb::str(event.policy.c_str());
  record[3] = nb::str(event.policyDigest.c_str());
  record[4] = nb::str(event.tuningDigest.c_str());
  record[5] = nb::str(event.regionMembership.c_str());
  record[6] = nb::str(event.coverageManifest.c_str());
  record[7] = nb::str(event.unsupportedFingerprint.c_str());
  record[8] = nb::str(event.normalizedModuleFingerprint.c_str());
  record[9] = nb::bool_(event.noShuttleSemantics);
  record[10] = nb::str(event.failurePass.c_str());
  return record;
}

nb::tuple pythonEvents(const ShuttleObserverTestCapture &capture) {
  // snapshot() releases the native mutex before Python objects are allocated.
  std::vector<ShuttleObserverTestEvent> events = capture.snapshot();
  nb::tuple records(events.size());
  for (std::size_t index = 0; index < events.size(); ++index) {
    records[index] = pythonRecord(events[index]);
  }
  return records;
}

} // namespace

void registerShuttleObserverTestBindings(nb::module_ &module) {
  nb::module_ bridge = module.def_submodule(
      "_shuttle_test_observer",
      "Test-only native Shuttle observer bridge; absent from normal builds.");
  nb::class_<ShuttleObserverTestCapture>(bridge, "Capture")
      .def("snapshot", &pythonEvents)
      .def("close", &ShuttleObserverTestCapture::close,
           nb::call_guard<nb::gil_scoped_release>())
      .def("__enter__",
           [](ShuttleObserverTestCapture &capture)
               -> ShuttleObserverTestCapture & { return capture; },
           nb::rv_policy::reference_internal)
      .def("__exit__",
           [](ShuttleObserverTestCapture &capture, nb::handle, nb::handle,
              nb::handle) {
             nb::gil_scoped_release release;
             capture.close();
             return false;
           });
  bridge.def("subscribe", &subscribeShuttleObserverForTesting);
}

} // namespace mlir::shuttle::testing
