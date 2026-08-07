# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from levanter.recovery.detection import (
    ABORT_COLLECTIVES_ON_FAILURE_ENV,
    USE_TFRT_GPU_CLIENT_ENV,
    DetectionConfig,
    classify_exception,
    recovery_xla_env,
    touch_heartbeat,
)
from levanter.recovery.types import FaultClass


def _parse_xla_flags(xla_flags: str) -> dict[str, str]:
    """Parse an ``XLA_FLAGS`` string into a ``--name -> value`` mapping."""
    parsed: dict[str, str] = {}
    for part in xla_flags.split():
        name, _, value = part.partition("=")
        parsed[name] = value
    return parsed


def test_recovery_xla_env_adds_deadman_progress_and_nccl_init_flags():
    env = recovery_xla_env(DetectionConfig(execution_terminate_timeout_seconds=240.0), base_env={})

    flags = _parse_xla_flags(env["XLA_FLAGS"])
    assert flags["--xla_gpu_execution_terminate_timeout"] == "240s"
    assert flags["--xla_gpu_execution_progress_tracking"] == "8"
    assert flags["--xla_gpu_nccl_termination_timeout_seconds"] == "120"


def test_recovery_xla_env_preserves_a_preset_deadman_and_does_not_duplicate_it():
    base_env = {"XLA_FLAGS": "--xla_gpu_execution_terminate_timeout=999s"}

    env = recovery_xla_env(DetectionConfig(execution_terminate_timeout_seconds=240.0), base_env=base_env)

    rendered = env["XLA_FLAGS"]
    assert rendered.count("--xla_gpu_execution_terminate_timeout") == 1
    flags = _parse_xla_flags(rendered)
    assert flags["--xla_gpu_execution_terminate_timeout"] == "999s"
    # the other flags are still merged in alongside the preserved one
    assert flags["--xla_gpu_execution_progress_tracking"] == "8"


def test_recovery_xla_env_omits_recoverability_recipe_when_disabled():
    env = recovery_xla_env(DetectionConfig(enable_recoverability=False), base_env={})

    assert ABORT_COLLECTIVES_ON_FAILURE_ENV not in env
    assert USE_TFRT_GPU_CLIENT_ENV not in env
    flags = _parse_xla_flags(env["XLA_FLAGS"])
    assert "--xla_gpu_nccl_blocking_communicators" not in flags
    assert "--xla_gpu_nccl_async_execution" not in flags


def test_recovery_xla_env_adds_full_recoverability_recipe_when_enabled():
    env = recovery_xla_env(DetectionConfig(enable_recoverability=True), base_env={})

    assert env[ABORT_COLLECTIVES_ON_FAILURE_ENV] == "1"
    assert env[USE_TFRT_GPU_CLIENT_ENV] == "1"
    flags = _parse_xla_flags(env["XLA_FLAGS"])
    assert flags["--xla_gpu_nccl_blocking_communicators"] == "false"
    assert flags["--xla_gpu_nccl_async_execution"] == "true"


def test_recovery_xla_env_does_not_mutate_base_env():
    base_env = {"XLA_FLAGS": "--some_existing_flag=1"}
    snapshot = dict(base_env)

    env = recovery_xla_env(DetectionConfig(enable_recoverability=True), base_env=base_env)

    assert base_env == snapshot
    assert env is not base_env


@pytest.mark.parametrize(
    "message",
    [
        "CUDA_ERROR_ILLEGAL_INSTRUCTION: an illegal instruction was encountered",
        "GPU 3 hit an uncorrectable ECC error",
        "an illegal memory access was encountered",
    ],
)
def test_classify_exception_flags_poisoned_context_as_sticky(message):
    assert classify_exception(RuntimeError(message)) is FaultClass.STICKY


@pytest.mark.parametrize(
    "message",
    [
        "DEADLINE_EXCEEDED: rendezvous timed out",
        "Terminating jobs stuck in NCCL collective",
        "collective all-reduce did not complete before timeout",
    ],
)
def test_classify_exception_flags_timeouts_as_stall(message):
    assert classify_exception(RuntimeError(message)) is FaultClass.STALL


def test_classify_exception_defaults_to_hard_for_application_errors():
    assert classify_exception(ValueError("shape mismatch in the data loader")) is FaultClass.HARD


def test_classify_exception_unwraps_a_sticky_cause_behind_a_generic_outer():
    try:
        try:
            raise ValueError("device fault: CUDA_ERROR_ILLEGAL_INSTRUCTION observed")
        except ValueError as inner:
            raise RuntimeError("training step failed") from inner
    except RuntimeError as exc:
        assert classify_exception(exc) is FaultClass.STICKY


def test_touch_heartbeat_records_step_and_recent_mtime(tmp_path):
    heartbeat = tmp_path / "heartbeat"

    before = time.time()
    touch_heartbeat(heartbeat, 42)
    after = time.time()

    step_str, timestamp_str = heartbeat.read_text().split()
    assert int(step_str) == 42
    assert before <= float(timestamp_str) <= after
    # the supervisor deadman keys off file mtime, so it must reflect the write time
    assert before - 1 <= heartbeat.stat().st_mtime <= after + 1


def test_touch_heartbeat_overwrites_with_the_latest_step(tmp_path):
    heartbeat = tmp_path / "heartbeat"

    touch_heartbeat(heartbeat, 5)
    touch_heartbeat(heartbeat, 6)

    step_str, _ = heartbeat.read_text().split()
    assert int(step_str) == 6
