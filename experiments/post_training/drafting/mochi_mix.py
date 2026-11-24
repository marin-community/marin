# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample and interleave the chat mixes for Mochi offline distillation."""

from collections.abc import Iterator
from dataclasses import dataclass
from heapq import merge
from itertools import batched
from operator import itemgetter
from random import Random

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.lazy import OUT, ArtifactStep, apply
from rigging.filesystem.storage_path import StoragePath, prefix_join

_VERSION = "2026.11.24"
_SEED = 20261124
_HOLDOUT_MODULUS = 20
_OUTPUT_BATCH_ROWS = 256
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


def _sample_source(source: str, rows_per_shard: int, max_shards: int, rng: Random) -> list[dict]:
    shards = sorted((StoragePath(source) / "outputs/main" / "*.parquet").glob(), key=str)
    examples = []
    for shard in rng.sample(shards, min(max_shards, len(shards))):
        with shard.open("rb") as stream:
            parquet = pq.ParquetFile(stream)
            groups = list(range(parquet.num_row_groups))
            rng.shuffle(groups)
            remaining = rows_per_shard
            for group in groups:
                table = parquet.read_row_group(group)
                count = min(remaining, table.num_rows)
                indices = rng.sample(range(table.num_rows), count)
                examples.extend(table.take(indices).to_pylist())
                remaining -= count
                if not remaining:
                    break
    rng.shuffle(examples)
    return examples


def _distributed_rows(rows: list[dict]) -> Iterator[tuple[float, dict]]:
    for index, row in enumerate(rows):
        yield (index + 0.5) / len(rows), row


def _write_mix(*, output: str, sources: tuple[tuple[str, str, int], ...], seed: int, max_shards: int = 4) -> None:
    from dupekit import hash_xxh3_128  # noqa: PLC0415 - Datakit extra
    from marin.datakit.chat_normalize import CHAT_SCHEMA  # noqa: PLC0415 - Datakit extra

    root = StoragePath(output)
    components: dict[str, list[list[dict]]] = {"train": [], "validation": []}
    for name, source, rows in sources:
        splits: dict[str, list[dict]] = {split: [] for split in components}
        for row in _sample_source(source, rows, max_shards, Random(f"{seed}:{name}")):
            user = next(message for message in row["messages"] if message["role"] == "user")
            prompt = "\n".join(part["text"] for part in user["content"])
            split = "validation" if hash_xxh3_128(prompt.encode()) % _HOLDOUT_MODULUS == 0 else "train"
            splits[split].append(row)
        for split, examples in splits.items():
            if not examples:
                raise ValueError(f"No {split} rows from {source}")
            components[split].append(examples)

    for split, source_rows in components.items():
        directory = root / split
        directory.mkdirs()
        # Place each source's rows evenly across the split, including when a shard is too short to fill its quota.
        mixed = merge(*map(_distributed_rows, source_rows), key=itemgetter(0))
        with (directory / "mix.parquet").open("wb") as stream, pq.ParquetWriter(stream, CHAT_SCHEMA) as writer:
            for batch in batched((row for _, row in mixed), _OUTPUT_BATCH_ROWS):
                writer.write_table(pa.Table.from_pylist(batch, schema=CHAT_SCHEMA))


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
        seed=_SEED,
    )


def scaled_mix() -> ArtifactStep:
    """Extend the same frozen sources and prompt-group split to about 50M training tokens."""
    return apply(
        "dspark/mixes/mochi-agentic-50m",
        _write_mix,
        version=_VERSION,
        output=OUT,
        sources=_source_inputs(tuple(source.scaled_rows_per_shard for source in _SOURCES)),
        seed=_SEED,
        max_shards=13,
    )
