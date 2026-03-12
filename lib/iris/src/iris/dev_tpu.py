# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class GcpNodeRef:
    """Resolved GCP node target for SSH/SCP operations."""

    kind: str
    name: str
    zone: str
    project: str
    tpu_worker_id: int = 0


@dataclass(frozen=True)
class DevTpuWorker:
    """One allocated worker backing a dev TPU session."""

    task_id: str
    worker_id: str
    worker_address: str
    host: str
    node: GcpNodeRef


@dataclass(frozen=True)
class DevTpuState:
    """Persisted local state for an active dev TPU session."""

    session_name: str
    config_file: str
    job_id: str
    tpu_type: str
    workers: list[DevTpuWorker]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> DevTpuState:
        data = json.loads(raw)
        workers = [
            DevTpuWorker(
                task_id=worker["task_id"],
                worker_id=worker["worker_id"],
                worker_address=worker["worker_address"],
                host=worker["host"],
                node=GcpNodeRef(**worker["node"]),
            )
            for worker in data["workers"]
        ]
        return cls(
            session_name=data["session_name"],
            config_file=data["config_file"],
            job_id=data["job_id"],
            tpu_type=data["tpu_type"],
            workers=workers,
        )

    @staticmethod
    def state_file_path(base_dir: Path, session_name: str) -> Path:
        return base_dir / f"{session_name}.json"


def parse_worker_host(worker_address: str) -> str:
    """Extract a host from an Iris worker address."""
    if not worker_address:
        raise ValueError("worker address must not be empty")

    if "://" in worker_address:
        parsed = urlsplit(worker_address)
    else:
        parsed = urlsplit(f"//{worker_address}")

    host = parsed.hostname
    if not host:
        raise ValueError(f"worker address does not include a host: {worker_address}")

    return host
