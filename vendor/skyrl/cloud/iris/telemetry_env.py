"""Resolve telemetry settings from the current Iris task."""

from __future__ import annotations

import logging
import os
import sys

from connectrpc.errors import ConnectError
from iris.client.client import get_iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME, TELEMETRY_ENDPOINT_PATH

from marinskyrl.environment_contract import (
    EXECUTION_UID_ENV,
    RUN_ID_ENV,
    TELEMETRY_ENDPOINT_ENV,
    TRAINING_TYPE_ENV,
    TrainingType,
)

logger = logging.getLogger(__name__)


def telemetry_environment(*, run_id: str | None = None, training_type: TrainingType | None = None) -> dict[str, str]:
    """Return the telemetry variables for this task, or nothing outside a cluster.

    Args:
        run_id: Experiment identity.
        training_type: The trainer the run uses; SKYRL_TRAINING_TYPE supplies it when this is None.
    """
    job_info = get_job_info()
    ctx = get_iris_ctx()
    if job_info is None or ctx is None or ctx.client is None:
        return {}
    if not job_info.attempt_uid:
        logger.warning("Iris did not provide an attempt uid; MarinSkyRL telemetry stays inert")
        return {}
    try:
        endpoint = ctx.client.resolve_endpoint(LOG_SERVER_ENDPOINT_NAME).rstrip("/") + TELEMETRY_ENDPOINT_PATH
    except (ConnectError, ConnectionError, TimeoutError):
        logger.warning("could not resolve the MarinSkyRL telemetry environment", exc_info=True)
        return {}
    resolved = {
        TELEMETRY_ENDPOINT_ENV: endpoint,
        RUN_ID_ENV: run_id or os.environ.get(RUN_ID_ENV) or str(job_info.job_id),
        EXECUTION_UID_ENV: os.environ.get(EXECUTION_UID_ENV) or f"iris:{job_info.attempt_uid}",
    }
    if value := training_type or os.environ.get(TRAINING_TYPE_ENV):
        resolved[TRAINING_TYPE_ENV] = TrainingType(value).value
    return resolved


def _main(argv: list[str]) -> None:
    if len(argv) >= 3 and argv[1] == "--":
        os.execvpe(argv[2], argv[2:], {**os.environ, **telemetry_environment()})
    raise SystemExit(f"usage: {argv[0]} -- COMMAND [ARG ...]")


if __name__ == "__main__":
    _main(sys.argv)
