// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_TESTING_PYTHONOBSERVERTESTBRIDGE_H_
#define SHUTTLE_TESTING_PYTHONOBSERVERTESTBRIDGE_H_

namespace nanobind {
class module_;
}

namespace mlir::shuttle::testing {

void registerShuttleObserverTestBindings(nanobind::module_ &module);

} // namespace mlir::shuttle::testing

#endif // SHUTTLE_TESTING_PYTHONOBSERVERTESTBRIDGE_H_
