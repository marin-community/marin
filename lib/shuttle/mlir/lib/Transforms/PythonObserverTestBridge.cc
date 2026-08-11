// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Testing/PythonObserverTestBridge.h"

#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/stl/unique_ptr.h"
#include "shuttle/Testing/ObserverTestBridge.h"

namespace nb = nanobind;

namespace mlir::shuttle::testing {
namespace {

nb::tuple pythonRecord(const ShuttleObserverTestEvent &event) {
  nb::list record;
  record.append(nb::int_(event.invocationId));
  record.append(nb::str(event.phase.c_str()));
  record.append(nb::str(event.policy.c_str()));
  record.append(nb::str(event.policyDigest.c_str()));
  record.append(nb::str(event.tuningDigest.c_str()));
  record.append(nb::str(event.regionMembership.c_str()));
  record.append(nb::str(event.coverageManifest.c_str()));
  record.append(nb::str(event.unsupportedFingerprint.c_str()));
  record.append(nb::str(event.normalizedModuleFingerprint.c_str()));
  record.append(nb::bool_(event.noShuttleSemantics));
  record.append(nb::str(event.failurePass.c_str()));
  return nb::tuple(record);
}

nb::tuple pythonEvents(const ShuttleObserverTestCapture &capture) {
  // snapshot() releases the native mutex before Python objects are allocated.
  std::vector<ShuttleObserverTestEvent> events = capture.snapshot();
  nb::list records;
  for (const ShuttleObserverTestEvent &event : events) {
    records.append(pythonRecord(event));
  }
  return nb::tuple(records);
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
           },
           nb::arg("exception_type").none(),
           nb::arg("exception_value").none(),
           nb::arg("traceback").none());
  bridge.def("subscribe", &subscribeShuttleObserverForTesting);
}

} // namespace mlir::shuttle::testing
