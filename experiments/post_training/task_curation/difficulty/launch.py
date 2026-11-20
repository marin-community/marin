# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared regional CPU-job envelope for the importer and controller."""

from rigging.timing import Duration

JOB_TIMEOUT = Duration.from_hours(72)
DAYTONA_API_URL = "https://app.daytona.io/api"
DAYTONA_TARGET = "us"


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
