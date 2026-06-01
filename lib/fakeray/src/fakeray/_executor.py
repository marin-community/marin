# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Fray actor that executes one remote call.

A single long-lived actor per pool slot. ``run`` receives a cloudpickled
``(callable, args, kwargs)`` payload where each argument has already been
dereferenced by the scheduler (no ObjectRefs cross the wire), invokes it, and
returns the (cloudpickled) result.

Keeping the actor generic — it just runs whatever callable it is handed — is
what keeps fakeray smallpond-agnostic. smallpond's ``exec_task`` body is an
ordinary function as far as the actor is concerned.
"""

from __future__ import annotations

import logging

import cloudpickle

logger = logging.getLogger(__name__)


class FakeRayExecutor:
    """Stateless executor actor. One ``run`` call == one remote task.

    Resources (cpu/ram) are set on the actor at pool-creation time; this class
    holds no per-task resource logic in v1 (slot-based scheduling).
    """

    def __init__(self) -> None:
        # Each actor is a separate process (a separate Iris job replica) and did
        # NOT inherit the driver's sys.modules. Re-install the shim here so that
        # `import ray` inside an unpickled task payload binds to fakeray rather
        # than failing (the bundle ships no real `ray`).
        import fakeray

        fakeray.install()
        logger.info("FakeRayExecutor actor started (ray shim installed)")

    def run(self, payload: bytes) -> bytes:
        """Execute a cloudpickled (fn, args, kwargs) and return pickled result.

        Args and kwargs are concrete values (the scheduler resolved any
        ObjectRefs before dispatch). Exceptions propagate to the caller's
        ActorFuture, which the scheduler turns into a failed ObjectRef.
        """
        fn, args, kwargs = cloudpickle.loads(payload)
        result = fn(*args, **kwargs)
        return cloudpickle.dumps(result)
