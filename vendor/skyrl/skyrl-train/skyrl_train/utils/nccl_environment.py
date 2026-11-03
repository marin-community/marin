import os
from collections.abc import Mapping

from loguru import logger

from skyrl_train.nccl_diagnostics import nccl_diagnostics_environment
from skyrl_train.env_vars import FR_DUMP_TEMP_FILE_ENV, NCCL_DEBUG_INFO_TEMP_FILE_ENV
from skyrl_train.utils.constants import (
    get_nccl_monitor_heartbeat_timeout,
    validate_worker_collective_timeout_seconds,
)


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
