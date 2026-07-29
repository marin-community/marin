# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate or run one Harbor job in an isolated environment.

Harbor (and its Daytona SDK) carry pre-release transitive pins that do not fit the marin lock, so
Harbor is treated as an external tool. This script runs under
``uv run --no-project --with <pinned-harbor> --with daytona --prerelease=allow``, which builds an
ephemeral environment with Harbor and Daytona.

The ``validate`` command parses YAML or JSON with the pinned ``JobConfig`` and emits canonical JSON.
The ``run`` command parses the launcher's adapted JSON, executes the job, and lets Harbor write its
``result.json`` tree under ``jobs_dir``.
"""

import asyncio
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml
from harbor.job import Job
from harbor_config import JobConfig, canonical_json
from pydantic import ValidationError


def _document(path: Path) -> Mapping:
    if path.suffix in {".yaml", ".yml"}:
        document = yaml.safe_load(path.read_text())
    elif path.suffix == ".json":
        document = json.loads(path.read_text())
    else:
        raise ValueError(f"unsupported Harbor config file format: {path.suffix}")
    if not isinstance(document, Mapping):
        raise ValueError("Harbor config must contain a mapping")
    return document


def _job_config(path: Path) -> JobConfig:
    document = dict(_document(path))
    document.setdefault("job_name", path.stem)
    try:
        return JobConfig.model_validate(document, extra="forbid")
    except ValidationError as exc:
        print(exc.json(include_url=False, include_input=False), file=sys.stderr)
        raise SystemExit(2) from exc


async def _run(config: JobConfig) -> None:
    job = await Job.create(config)
    await job.run()


def main() -> None:
    command = sys.argv[1]
    config = _job_config(Path(sys.argv[2]))
    if command == "validate":
        sys.stdout.buffer.write(canonical_json(config))
        return
    if command != "run":
        raise ValueError(f"unknown command {command!r}")
    asyncio.run(_run(config))


if __name__ == "__main__":
    main()
