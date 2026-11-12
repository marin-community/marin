# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resource retries preserve unrelated dimensions and stop at the policy caps."""

import pytest

from taskcompendium.pipeline.resource_calibration import (
    CALIBRATION_CAP,
    INITIAL_REQUIREMENTS,
    FailureSignal,
    ResourceDimension,
    ResourceUsage,
    failure_signal,
    next_requirements,
)


@pytest.mark.parametrize(
    "exception,log,usage,expected",
    [
        ({"exception_type": "AgentTimeoutError"}, "", ResourceUsage(), FailureSignal.AGENT_TIMEOUT),
        ({"exception_type": "VerifierTimeoutError"}, "", ResourceUsage(), FailureSignal.VERIFIER_TIMEOUT),
        ({"exception_type": "AgentDiskFullError"}, "", ResourceUsage(), FailureSignal.DISK_FULL),
        (None, "write failed: No space left on device", ResourceUsage(), FailureSignal.DISK_FULL),
        (None, "ENOSPC", ResourceUsage(), FailureSignal.DISK_FULL),
        (None, "container state: OOMKilled", ResourceUsage(), FailureSignal.OOM),
        (None, "", ResourceUsage(oom_kills=1), FailureSignal.OOM),
        (None, "Killed; process exited 137", ResourceUsage(), FailureSignal.NON_RESOURCE),
        (
            {"exception_type": "RuntimeError", "exception_message": "bad credentials"},
            "",
            ResourceUsage(),
            FailureSignal.NON_RESOURCE,
        ),
    ],
)
def test_failure_signals_distinguish_resource_pressure_from_other_failures(exception, log, usage, expected):
    assert failure_signal(exception, log, usage) is expected


@pytest.mark.parametrize(
    "signal,usage,changes",
    [
        (FailureSignal.OOM, ResourceUsage(), {"memory_mb": 4096}),
        (FailureSignal.DISK_FULL, ResourceUsage(), {"storage_mb": 4096}),
        (FailureSignal.AGENT_TIMEOUT, ResourceUsage(), {"agent_timeout_sec": 600}),
        (
            FailureSignal.AGENT_TIMEOUT,
            ResourceUsage(cpu_seconds=295, elapsed_seconds=300),
            {"agent_timeout_sec": 600, "cpus": 2},
        ),
        (FailureSignal.AGENT_TIMEOUT, ResourceUsage(cpu_seconds=20, elapsed_seconds=300), {"agent_timeout_sec": 600}),
        (FailureSignal.VERIFIER_TIMEOUT, ResourceUsage(), {"verifier_timeout_sec": 120}),
    ],
)
def test_retry_doubles_only_the_failed_dimension(signal, usage, changes):
    updated = next_requirements(INITIAL_REQUIREMENTS, signal, usage)
    assert updated == INITIAL_REQUIREMENTS.model_copy(update=changes)


def test_mixed_failures_reach_caps_without_growing_unrelated_resources():
    shape = INITIAL_REQUIREMENTS
    for _ in range(5):
        shape = next_requirements(shape, FailureSignal.OOM, ResourceUsage())
        assert shape is not None
    assert shape.memory_mb == 65536
    assert shape.cpus == 1 and shape.storage_mb == 2048
    assert next_requirements(shape, FailureSignal.OOM, ResourceUsage()) is None
    assert next_requirements(shape, FailureSignal.NON_RESOURCE, ResourceUsage()) is None
    near_cap = CALIBRATION_CAP.model_copy(update={"agent_timeout_sec": 9600})
    assert next_requirements(near_cap, FailureSignal.AGENT_TIMEOUT, ResourceUsage()) == CALIBRATION_CAP
    assert (
        next_requirements(
            CALIBRATION_CAP, FailureSignal.AGENT_TIMEOUT, ResourceUsage(cpu_seconds=4800, elapsed_seconds=300)
        )
        is None
    )


def test_partial_source_declarations_stay_fixed_during_missing_resource_calibration():
    shape = INITIAL_REQUIREMENTS.model_copy(update={"cpus": 2, "agent_timeout_sec": 900})
    fixed = frozenset({ResourceDimension.CPUS, ResourceDimension.AGENT_TIMEOUT})
    assert next_requirements(shape, FailureSignal.OOM, ResourceUsage(), fixed) == shape.model_copy(
        update={"memory_mb": 4096}
    )
    assert (
        next_requirements(
            shape, FailureSignal.AGENT_TIMEOUT, ResourceUsage(cpu_seconds=1800, elapsed_seconds=900), fixed
        )
        is None
    )
