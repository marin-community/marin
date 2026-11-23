# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the bounded chat mixes for Mochi offline distillation."""

from dataclasses import dataclass
from itertools import islice

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.lazy import OUT, ArtifactStep, apply
from rigging.filesystem.storage_path import StoragePath, prefix_join

_VERSION = "2026.11.22"
_NEMOTRON = "gs://marin-us-central2/normalized-chat/nemotron_sft_v3"


@dataclass(frozen=True, kw_only=True)
class MixSource:
    name: str
    path: str
    version: str
    smoke_rows_per_shard: int
    scaled_rows_per_shard: int


_SOURCES = (
    MixSource(
        name="nemotron-tool-calling",
        path=prefix_join(_NEMOTRON, "agentic_v2/tool_calling_bc0791c0"),
        version="2026.10.18",
        smoke_rows_per_shard=256,
        scaled_rows_per_shard=2_560,
    ),
    MixSource(
        name="nemotron-instruction",
        path=prefix_join(_NEMOTRON, "instruction_following_chat_v2/reasoning_off_e4fcbd89"),
        version="2026.10.23",
        smoke_rows_per_shard=256,
        scaled_rows_per_shard=1_536,
    ),
    MixSource(
        name="agenttrove",
        path="s3://marin-us-east-02a/marin/normalized-chat/agenttrove_8ada38cb",
        version="2026.11.22",
        smoke_rows_per_shard=128,
        scaled_rows_per_shard=384,
    ),
)


def _write_mix(*, output: str, sources: tuple[tuple[str, str, int], ...], max_shards: int = 4) -> None:
    from dupekit import hash_xxh3_128  # noqa: PLC0415 - Datakit extra
    from marin.datakit.chat_normalize import CHAT_SCHEMA  # noqa: PLC0415 - Datakit extra
    from marin.datakit.download.rollout_transforms import load_parquet_batched  # noqa: PLC0415 - Datakit extra

    root = StoragePath(output)
    for name, source, rows in sources:
        shards = sorted((StoragePath(source) / "outputs/main" / "*.parquet").glob(), key=str)
        splits: dict[str, list[dict]] = {"train": [], "validation": []}
        for shard in shards[:max_shards]:
            for row in islice(load_parquet_batched(str(shard)), rows):
                user = next(message for message in row["messages"] if message["role"] == "user")
                prompt = "\n".join(part["text"] for part in user["content"])
                split = "validation" if hash_xxh3_128(prompt.encode()) % 20 == 0 else "train"
                splits[split].append(row)
        for split, examples in splits.items():
            if not examples:
                raise ValueError(f"No {split} rows from {source}")
            directory = root / split
            directory.mkdirs()
            with (directory / f"{name}.parquet").open("wb") as stream:
                pq.write_table(pa.Table.from_pylist(examples, schema=CHAT_SCHEMA), stream)


def _source_inputs(rows_per_shard: tuple[int, ...]) -> tuple[tuple[str, ArtifactStep, int], ...]:
    """Bind each source name, normalized artifact and per-shard row limit."""
    return tuple(
        (
            source.name,
            ArtifactStep.adopt(f"dspark/mix-inputs/{source.name}", source.version, source.path),
            rows,
        )
        for source, rows in zip(_SOURCES, rows_per_shard, strict=True)
    )


def smoke_mix() -> ArtifactStep:
    """Bind archived Datakit inputs to the same fixed smoke split on every run."""
    return apply(
        "dspark/mixes/mochi-agentic-smoke",
        _write_mix,
        version=_VERSION,
        output=OUT,
        sources=_source_inputs(tuple(source.smoke_rows_per_shard for source in _SOURCES)),
    )


def scaled_mix() -> ArtifactStep:
    """Extend the same frozen sources and prompt-group split to about 50M training tokens."""
    return apply(
        "dspark/mixes/mochi-agentic-50m",
        _write_mix,
        version="2026.11.23",
        output=OUT,
        sources=_source_inputs(tuple(source.scaled_rows_per_shard for source in _SOURCES)),
        max_shards=13,
    )
