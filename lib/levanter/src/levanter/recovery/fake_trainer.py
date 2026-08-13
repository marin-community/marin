# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fake trainer entrypoint for exercising the supervisor without a GPU.

Spawned like a real trainer by ``levanter.recovery.child`` via its module and
qualname, so it must be a real installed module, not a test file. It makes no JAX
device calls, so the supervisor's whole recovery state machine runs on CPU.
"""

from __future__ import annotations

import logging
import os
import resource
import sys
import time

from levanter.recovery.detection import touch_heartbeat
from levanter.recovery.faults import STICKY_CUDA_ERROR, InjectedStickyFault
from levanter.recovery.supervisor import ENV_ATTEMPT, ENV_HEARTBEAT_PATH


logger = logging.getLogger(__name__)

ENV_BEHAVIOR = "LEVANTER_FAKE_TRAINER_BEHAVIOR"


def fake_trainer(config: dict) -> None:
    """Step ``config['steps']`` times, injecting the configured behaviour once.

    The behaviour fires only on the first attempt unless ``always`` is set, so a
    supervised restart runs clean and completes, mirroring a real trainer that
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
        raise InjectedStickyFault(f"fake trainer: {STICKY_CUDA_ERROR}: an illegal instruction was encountered")
    if behavior == "hang":
        logger.warning("fake trainer: freezing (heartbeat stops); deadman should fire")
        while True:
            time.sleep(3600)
    if behavior == "crash":
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.abort()  # SIGABRT -> negative returncode -> FaultClass.CRASH
    if behavior == "hard":
        sys.exit(7)  # non-sentinel nonzero -> FaultClass.HARD, not restarted
    if behavior == "complete":
        return
    raise ValueError(f"unknown fake-trainer behavior: {behavior}")
