# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from iris.rpc import job_pb2, query_pb2


class TpuNameLookup(Protocol):
    def __call__(self, name: str, project: str, *, zone: str = "-") -> tuple[str, str] | None: ...


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
class WorkerResolutionMetadata:
    address: str
    metadata: job_pb2.WorkerMetadata


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


def parse_tpu_worker_id(raw_worker_id: str) -> int:
    """Parse the TPU worker id stored in Iris worker metadata."""
    if not raw_worker_id:
        return 0
    try:
        return int(raw_worker_id)
    except ValueError as exc:
        raise ValueError(f"invalid TPU worker id in worker metadata: {raw_worker_id!r}") from exc


def resolve_node_ref_from_worker_metadata(
    metadata: job_pb2.WorkerMetadata,
    project: str,
    *,
    find_tpu_by_name: TpuNameLookup | None = None,
) -> GcpNodeRef | None:
    """Resolve a GCP SSH target from Iris worker metadata."""
    if metadata.tpu_name:
        zone = metadata.gce_zone
        if not zone and find_tpu_by_name is not None:
            tpu_match = find_tpu_by_name(metadata.tpu_name, project, zone="-")
            if tpu_match:
                _name, zone = tpu_match
        if zone:
            return GcpNodeRef(
                kind="tpu",
                name=metadata.tpu_name,
                zone=zone,
                project=project,
                tpu_worker_id=parse_tpu_worker_id(metadata.tpu_worker_id),
            )

    if metadata.gce_instance_name and metadata.gce_zone:
        return GcpNodeRef(
            kind="vm",
            name=metadata.gce_instance_name,
            zone=metadata.gce_zone,
            project=project,
        )

    return None


def worker_address_lookup_values(worker_address: str) -> list[str]:
    """Return worker address forms used by different Iris controller versions."""
    if not worker_address:
        return []
    values = [worker_address]
    if "://" in worker_address:
        parsed = urlsplit(worker_address)
    else:
        parsed = urlsplit(f"//{worker_address}")
    if parsed.netloc:
        values.append(parsed.netloc)
    return list(dict.fromkeys(values))


def worker_resolution_metadata_from_response(response: query_pb2.RawQueryResponse) -> WorkerResolutionMetadata | None:
    """Decode the worker metadata row shape selected by the dev TPU script."""
    if not response.rows:
        return None

    columns = {column.name: index for index, column in enumerate(response.columns)}
    row = json.loads(response.rows[0])
    metadata = job_pb2.WorkerMetadata(
        ip_address=row[columns["md_ip_address"]] or "",
        tpu_name=row[columns["md_tpu_name"]] or "",
        tpu_worker_id=row[columns["md_tpu_worker_id"]] or "",
        gce_instance_name=row[columns["md_gce_instance_name"]] or "",
        gce_zone=row[columns["md_gce_zone"]] or "",
    )
    return WorkerResolutionMetadata(address=row[columns["address"]] or "", metadata=metadata)
