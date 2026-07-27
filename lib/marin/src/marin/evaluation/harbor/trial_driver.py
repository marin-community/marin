# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one Harbor job in an isolated environment; invoked as a subprocess, never imported.

Harbor (and its Daytona SDK) carry pre-release transitive pins that do not fit the marin lock, so
Harbor is treated as an external tool: :func:`marin.evaluation.harbor.runner.run_harbor` runs
this script under ``uv run --no-project --with harbor --with daytona --prerelease=allow``, which
builds an ephemeral environment with Harbor and Daytona. This file imports the stdlib-only typed
adapter from the Marin source tree, but none of Marin's runtime dependencies, so it loads cleanly in
that project-less environment.

It reads a JSON config (path in ``argv[1]``), runs the Harbor job against the served model's proxy
URL, and lets Harbor write its native ``result.json`` tree under ``jobs_dir``. The caller reads those
trial files back and normalizes them into the shared eval contract.
"""

import asyncio
import json
import sys
from pathlib import Path

from harbor.job import Job
from harbor.models.job.config import JobConfig

from marin.evaluation.harbor.driver_config import HarborDriverConfig, native_job_config


async def _run(config: dict) -> None:
    driver_config = HarborDriverConfig.from_dict(config)
    job = await Job.create(JobConfig.model_validate(native_job_config(driver_config)))
    await job.run()


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text())
    asyncio.run(_run(config))


if __name__ == "__main__":
    main()
