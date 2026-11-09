# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The snapshot ``sample.py`` writes and ``load.py`` reads: every mixture source with its sampled documents."""

import io
import json
from dataclasses import asdict, dataclass
from enum import StrEnum

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

SOURCES_FILE = "sources.json"
SAMPLES_FILE = "samples.parquet"


class SourceKind(StrEnum):
    SFT = "sft"
    REPLAY = "replay"


@dataclass(frozen=True)
class MixSource:
    """One token cache of the mixture and the share of training positions it was scheduled for."""

    name: str
    kind: SourceKind
    family: str
    share: float
    cache_path: str


@dataclass(frozen=True)
class Sample:
    """One whole document, decoded; ``text`` is clipped when the document is long."""

    doc_index: int
    tokens: int
    chars: int
    text: str


@dataclass(frozen=True)
class SampledSource:
    source: MixSource
    documents: int  # documents in the cache
    samples: tuple[Sample, ...]


def write_snapshot(root: StoragePath, seed: int, sampled: list[SampledSource]) -> int:
    """Write the snapshot below ``root`` and return the size of its samples file in bytes."""
    source_rows = [{**asdict(entry.source), "documents": entry.documents} for entry in sampled]
    sample_rows = [{"source": entry.source.name, **asdict(sample)} for entry in sampled for sample in entry.samples]
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(sample_rows), buffer, compression="zstd")
    (root / SAMPLES_FILE).write_bytes(buffer.getvalue())
    (root / SOURCES_FILE).write_text(json.dumps({"seed": seed, "sources": source_rows}, indent=1))
    return buffer.tell()


def read_snapshot(root: StoragePath) -> list[SampledSource]:
    samples: dict[str, list[Sample]] = {}
    for row in pq.read_table(io.BytesIO((root / SAMPLES_FILE).read_bytes())).to_pylist():
        samples.setdefault(row.pop("source"), []).append(Sample(**row))
    sampled = []
    for row in json.loads((root / SOURCES_FILE).read_text())["sources"]:
        documents = row.pop("documents")
        source = MixSource(**{**row, "kind": SourceKind(row["kind"])})
        sampled.append(SampledSource(source, documents, tuple(samples.pop(source.name, ()))))
    if samples:
        raise ValueError(f"Samples name sources missing from {SOURCES_FILE}: {sorted(samples)[:5]}")
    return sampled
