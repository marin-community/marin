# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze a bounded chat mix for the Mochi DSpark qualification."""

from itertools import islice

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.lazy import OUT, ArtifactStep, apply
from rigging.filesystem.storage_path import StoragePath

_VERSION = "2026.11.22"
_NEMOTRON = "gs://marin-us-central2/normalized-chat/nemotron_sft_v3"
_SOURCES = (
    ("nemotron-tool-calling", f"{_NEMOTRON}/agentic_v2/tool_calling_bc0791c0", "2026.10.18", 256),
    ("nemotron-instruction", f"{_NEMOTRON}/instruction_following_chat_v2/reasoning_off_e4fcbd89", "2026.10.23", 256),
    (
        "agenttrove",
        "s3://marin-us-east-02a/marin/normalized-chat/agenttrove_8ada38cb",
        "2026.11.22",
        128,
    ),
)


def _write_smoke_mix(*, output: str, sources: tuple[tuple[str, str, int], ...]) -> None:
    from dupekit import hash_xxh3_128  # noqa: PLC0415 - Datakit extra
    from marin.datakit.chat_normalize import CHAT_SCHEMA  # noqa: PLC0415 - Datakit extra
    from marin.datakit.download.rollout_transforms import load_parquet_batched  # noqa: PLC0415 - Datakit extra

    root = StoragePath(output)
    for name, source, rows in sources:
        shards = sorted((StoragePath(source) / "outputs/main" / "*.parquet").glob(), key=str)
        splits: dict[str, list[dict]] = {"train": [], "validation": []}
        for shard in shards[:4]:
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


def smoke_mix() -> ArtifactStep:
    """Bind archived Datakit inputs to the same fixed smoke split on every run."""

    return apply(
        "dspark/mixes/mochi-agentic-smoke",
        _write_smoke_mix,
        version=_VERSION,
        output=OUT,
        sources=tuple(
            (name, ArtifactStep.adopt(f"dspark/mix-inputs/{name}", version, path), rows)
            for name, path, version, rows in _SOURCES
        ),
    )
