# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fake trainer entrypoints for exercising the supervisor without a GPU.

Imported by ``levanter.recovery._child`` in tests via its module+qualname, so it
must be a real installed module (not a test file). It imports no JAX.
"""

from __future__ import annotations

import logging
import os
import sys
import time

from levanter.recovery.detection import touch_heartbeat
from levanter.recovery.faults import InjectedStickyFault
from levanter.recovery.supervisor import ENV_ATTEMPT, ENV_HEARTBEAT_PATH


logger = logging.getLogger(__name__)

ENV_BEHAVIOR = "SELFTEST_BEHAVIOR"


def fake_trainer(config: dict) -> None:
    """Step ``config['steps']`` times, injecting ``SELFTEST_BEHAVIOR`` once.

    The behaviour fires only on the first attempt unless ``always`` is set, so a
    supervised restart runs clean and completes — mirroring a real trainer that
    resumes from the snapshot after a one-off fault.
    """
    behavior = os.environ.get(ENV_BEHAVIOR, "complete")
    attempt = int(os.environ.get(ENV_ATTEMPT, "1"))
    heartbeat = os.environ[ENV_HEARTBEAT_PATH]
    steps = int(config["steps"])
    fault_step = int(config.get("fault_step", -1))
    always = bool(config.get("always", False))

    for step in range(steps):
        touch_heartbeat(heartbeat, step)
        if step == fault_step and (attempt == 1 or always):
            _inject(behavior)
        time.sleep(float(config.get("step_seconds", 0.01)))
    touch_heartbeat(heartbeat, steps)


def _inject(behavior: str) -> None:
    if behavior == "sticky":
        raise InjectedStickyFault("selftest: CUDA_ERROR_ILLEGAL_INSTRUCTION: an illegal instruction was encountered")
    if behavior == "hang":
        logger.warning("selftest: freezing (heartbeat stops); deadman should fire")
        while True:
            time.sleep(3600)
    if behavior == "crash":
        os.abort()  # SIGABRT -> negative returncode -> FaultClass.CRASH
    if behavior == "hard":
        sys.exit(7)  # non-sentinel nonzero -> FaultClass.HARD, not restarted
    if behavior == "complete":
        return
    raise ValueError(f"unknown selftest behavior: {behavior}")
