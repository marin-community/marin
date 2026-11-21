"""Resolve config-defined environment for Ray workers."""

import os
from collections.abc import Mapping

from loguru import logger
from omegaconf import DictConfig

from marinskyrl.environment_contract import (
    EnvVarManager,
    EnvVarScope,
    FR_DUMP_TEMP_FILE_ENV,
    NCCL_DEBUG_INFO_TEMP_FILE_ENV,
    nccl_diagnostics_environment,
)


# Default worker-NCCL-collective timeout. Raised to 1800s (30 min) because the long MoE weight-sync
# gather + first-step forward on 80B routinely exceeds the old 600s watchdog and
# SIGABRTs the gang. ``trainer.distributed.worker_collective_timeout_seconds``
# is the runtime authority; this constant supplies lower-level API defaults.
DEFAULT_WORKER_NCCL_TIMEOUT_IN_S = 1800
DEFAULT_NCCL_MONITOR_HEARTBEAT_TIMEOUT = 300


def validate_worker_collective_timeout_seconds(value: int | None = None) -> int:
    """Validate a configured worker collective timeout."""
    timeout = DEFAULT_WORKER_NCCL_TIMEOUT_IN_S if value is None else int(value)
    if timeout <= 0:
        raise ValueError("trainer.distributed.worker_collective_timeout_seconds must be positive")
    return timeout


def get_nccl_monitor_heartbeat_timeout(value: int | None = None) -> int:
    """Validate the requested NCCL monitor heartbeat deadline."""
    timeout = DEFAULT_NCCL_MONITOR_HEARTBEAT_TIMEOUT if value is None else value
    if timeout <= 0:
        raise ValueError("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC must be positive")
    return timeout


def worker_nccl_environment(
    base_environment: Mapping[str, str], *, collective_timeout_seconds: int | None = None
) -> dict[str, str]:
    """Resolve worker collective deadlines, diagnostics, and dump destinations."""

    dump_path = (
        base_environment.get(FR_DUMP_TEMP_FILE_ENV)
        or base_environment.get(NCCL_DEBUG_INFO_TEMP_FILE_ENV)
        or os.environ.get(FR_DUMP_TEMP_FILE_ENV)
        or os.environ.get(NCCL_DEBUG_INFO_TEMP_FILE_ENV)
        or "/tmp/nccl_fr_rank"
    )
    collective_timeout_seconds = validate_worker_collective_timeout_seconds(collective_timeout_seconds)
    heartbeat_timeout_value = base_environment.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC") or os.environ.get(
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"
    )
    requested_heartbeat_timeout_seconds = get_nccl_monitor_heartbeat_timeout(
        None if heartbeat_timeout_value is None else int(heartbeat_timeout_value)
    )
    heartbeat_timeout_seconds = min(requested_heartbeat_timeout_seconds, collective_timeout_seconds)
    if heartbeat_timeout_seconds != requested_heartbeat_timeout_seconds:
        logger.warning(
            "Capping TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC from {} to the {}-second collective timeout",
            requested_heartbeat_timeout_seconds,
            collective_timeout_seconds,
        )

    environment = {
        FR_DUMP_TEMP_FILE_ENV: dump_path,
        NCCL_DEBUG_INFO_TEMP_FILE_ENV: dump_path,
    }
    environment.update(nccl_diagnostics_environment(heartbeat_timeout_seconds=heartbeat_timeout_seconds))
    return environment


def configured_ray_worker_environment(cfg: DictConfig) -> dict[str, str]:
    """Resolve config-defined Ray worker settings before probing hardware."""
    environment = EnvVarManager.from_config(cfg).environment_for(EnvVarScope.RAY_WORKER)
    # The collective timeout caps the debug preset's heartbeat deadline.
    environment.update(
        worker_nccl_environment(
            base_environment=environment,
            collective_timeout_seconds=int(cfg.trainer.distributed.worker_collective_timeout_seconds),
        )
    )
    return environment
