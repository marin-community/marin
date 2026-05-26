# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Output data structures for the distributed inference library.

Each shard's output is a JSONL[.gz] file at
``gs://marin-{results_region}/{job_name}/{run_id}/outputs/shard-NNNNNNNN.jsonl.gz``.
The file's existence is itself the done-marker; there is no separate manifest.

`InferenceResult` provides lazy iteration over those files for callers running
in ``results_region``. Downstream ExecutorStep callers should run there to
avoid cross-region read egress.
"""
from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import fsspec


@dataclass(frozen=True)
class ResponseRecord:
    """A single inference output."""

    id: str
    shard: int
    response: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "shard": self.shard, "response": self.response}
        if self.extra:
            out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResponseRecord:
        extra = {k: v for k, v in data.items() if k not in {"id", "shard", "response"}}
        return cls(id=data["id"], shard=int(data["shard"]), response=data["response"], extra=extra)


@dataclass(frozen=True)
class InferenceResult:
    """Lazy handle over a completed inference run's outputs.

    Attributes:
        results_uri: GCS prefix containing the shard output files.
        results_region: Canonical region of ``results_uri``. Callers running
            in this region pay no egress when iterating.
        missing_shards: Shard indices for which no output file exists. Non-empty
            indicates a partial run (one or more regions failed and no other
            region completed the shard).
    """

    results_uri: str
    results_region: str
    missing_shards: tuple[int, ...] = ()

    @property
    def is_complete(self) -> bool:
        return not self.missing_shards

    def iter_records(self) -> Iterator[ResponseRecord]:
        """Yield every ResponseRecord from every shard output file."""
        for path in self.list_output_files():
            yield from _read_shard_file(path)

    def to_list(self) -> list[ResponseRecord]:
        return list(self.iter_records())

    def list_output_files(self) -> list[str]:
        """Return sorted shard output file URIs."""
        fs, _ = fsspec.core.url_to_fs(self.results_uri)
        pattern = f"{self.results_uri.rstrip('/')}/shard-*.jsonl.gz"
        matches = fs.glob(pattern)
        protocol = self.results_uri.split("://", 1)[0] + "://" if "://" in self.results_uri else ""
        return sorted(p if "://" in p else f"{protocol}{p}" for p in matches)


def _read_shard_file(path: str) -> Iterator[ResponseRecord]:
    with fsspec.open(path, "rb") as f:
        with gzip.open(f, "rt", encoding="utf-8") as gz:
            for line in gz:
                line = line.strip()
                if not line:
                    continue
                yield ResponseRecord.from_dict(json.loads(line))


def shard_output_path(results_uri: str, shard_idx: int) -> str:
    """Canonical output path for a shard. Used both by the writer and the reader."""
    return f"{results_uri.rstrip('/')}/shard-{shard_idx:08d}.jsonl.gz"


def shard_idx_from_output_path(path: str) -> int:
    """Reverse of `shard_output_path` — used by the missing-shards checker."""
    basename = path.rsplit("/", 1)[-1]
    if not basename.startswith("shard-"):
        raise ValueError(f"Not a shard output path: {path!r}")
    digits = basename.removeprefix("shard-").split(".", 1)[0]
    return int(digits)


def compute_missing_shards(results_uri: str, total_shards: int) -> tuple[int, ...]:
    """Return the sorted tuple of shard indices for which no output exists.

    Lists ``results_uri`` once and compares to ``range(total_shards)``.
    """
    fs, _ = fsspec.core.url_to_fs(results_uri)
    pattern = f"{results_uri.rstrip('/')}/shard-*.jsonl.gz"
    found_indices: set[int] = set()
    for path in fs.glob(pattern):
        try:
            found_indices.add(shard_idx_from_output_path(path))
        except ValueError:
            continue
    return tuple(idx for idx in range(total_shards) if idx not in found_indices)
