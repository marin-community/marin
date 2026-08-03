# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from extend_arctic_teacher import (  # noqa: E402
    BASE_MANIFEST_METADATA_KEY,
    EXPANDED_MANIFEST_METADATA_KEY,
    MANIFEST_METADATA_KEY,
    RUNG_METADATA_KEY,
    TEACHER_ID,
    TEACHER_ID_METADATA_KEY,
    TEACHER_REVISION,
    TEACHER_REVISION_METADATA_KEY,
    assigned_sources,
    checked_prefix_embeddings,
    prefix_metadata,
    validate_teacher_metadata,
)


def alignment_table(hashes: list[str]) -> pa.Table:
    return pa.table(
        {
            "raw_sha256": hashes,
            "eval_rank": list(range(len(hashes))),
            "train_rank": [-1] * len(hashes),
        }
    )


def teacher_table(hashes: list[str]) -> pa.Table:
    table = alignment_table(hashes)
    values = np.arange(len(hashes) * 256, dtype=np.uint8).reshape(len(hashes), 256)
    embedding = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), 256)
    return table.append_column("embedding", embedding)


def test_checked_prefix_embeddings_returns_exact_prefix() -> None:
    hashes = ["a", "b", "c"]
    expanded = alignment_table([*hashes, "d"])

    values = checked_prefix_embeddings(expanded, alignment_table(hashes), teacher_table(hashes))

    assert values.shape == (3, 256)
    assert values.dtype == np.uint8


def test_checked_prefix_embeddings_rejects_changed_prefix() -> None:
    with pytest.raises(ValueError, match="expanded prefix differs"):
        checked_prefix_embeddings(
            alignment_table(["a", "changed"]), alignment_table(["a", "b"]), teacher_table(["a", "b"])
        )


def test_validate_teacher_metadata_requires_exact_inputs() -> None:
    table = teacher_table(["a"]).replace_schema_metadata(
        {
            MANIFEST_METADATA_KEY: b"base-sha",
            TEACHER_ID_METADATA_KEY: TEACHER_ID.encode(),
            TEACHER_REVISION_METADATA_KEY: TEACHER_REVISION.encode(),
        }
    )

    manifest = {"sha256": "base-sha"}
    validate_teacher_metadata(table, prefix_metadata(manifest, None))
    with pytest.raises(ValueError, match="metadata differs"):
        validate_teacher_metadata(table, prefix_metadata({"sha256": "different-sha"}, None))


def test_prefix_metadata_identifies_expanded_rung() -> None:
    manifest = {"sha256": "10m-sha", "base_manifest_sha256": "base-sha"}

    metadata = prefix_metadata(manifest, "10m")

    assert metadata == {
        EXPANDED_MANIFEST_METADATA_KEY: b"10m-sha",
        BASE_MANIFEST_METADATA_KEY: b"base-sha",
        RUNG_METADATA_KEY: b"10m",
        TEACHER_ID_METADATA_KEY: TEACHER_ID.encode(),
        TEACHER_REVISION_METADATA_KEY: TEACHER_REVISION.encode(),
    }


def test_assigned_sources_is_complete_and_balanced() -> None:
    manifest = {
        "sources": {
            "a": {"counts": {"train_10m": 100}},
            "b": {"counts": {"train_10m": 90}},
            "c": {"counts": {"train_10m": 80}},
            "d": {"counts": {"train_10m": 70}},
        }
    }

    shards = [assigned_sources(manifest, "10m", index, 2) for index in range(2)]
    totals = [sum(manifest["sources"][source]["counts"]["train_10m"] for source in shard) for shard in shards]

    assert sorted(source for shard in shards for source in shard) == ["a", "b", "c", "d"]
    assert max(totals) - min(totals) <= 20
