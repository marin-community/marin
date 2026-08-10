// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_TRANSFORMS_OBSERVERTESTINTERNAL_H_
#define SHUTTLE_TRANSFORMS_OBSERVERTESTINTERNAL_H_

namespace mlir::shuttle::detail {

using ShuttleObserverTeardownWaitHook = void (*)(void *context,
                                                 void *subscriptionState);

void setShuttleObserverTeardownWaitHookForTesting(
    ShuttleObserverTeardownWaitHook hook, void *context);

bool confirmShuttleObserverTeardownWaitForTesting(void *subscriptionState);

} // namespace mlir::shuttle::detail

#endif // SHUTTLE_TRANSFORMS_OBSERVERTESTINTERNAL_H_
