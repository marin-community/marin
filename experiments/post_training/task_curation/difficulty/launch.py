# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared regional CPU-job envelope for the importer and controller."""

from rigging.timing import Duration
from shellbox.backends.daytona.connection import DEFAULT_DAYTONA_API_URL

JOB_TIMEOUT = Duration.from_hours(72)
DAYTONA_API_URL = DEFAULT_DAYTONA_API_URL
DAYTONA_TARGET = "us"
DATA_ADMISSION_COMMAND = (
    "python",
    "-m",
    "shellbox.backends.daytona.usage_cli",
    "--api-key-env",
    "DAYTONA_DATA_API_KEY",
    "--limits",
    "data",
    "--target",
    DAYTONA_TARGET,
    "--headroom",
    "0.2",
)


def cpu_job_command(name: str) -> list[str]:
    return [
        "uv",
        "run",
        "--no-sync",
        "iris",
        "--cluster=marin",
        "job",
        "run",
        "--job-name",
        name,
        "--target-cluster=cw-rno2a",
        "--cpu",
        "4",
        "--memory",
        "8GB",
        "--disk",
        "10GB",
        "--enable-extra-resources",
        "--priority",
        "interactive",
        "--timeout",
        str(int(JOB_TIMEOUT.to_seconds())),
        "--max-retries",
        "0",
        "--no-wait",
    ]
