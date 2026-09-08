# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["jax==0.11.0", "numpy==2.3.5"]
# ///

"""Materialize the frozen StarCoder support and holdout identities for review-v8."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from levanter.data.dataset import AsyncDataset
from levanter.data.text.datasets import DEFAULT_LM_DATA_SHUFFLE, _stable_simulated_epoch_subset_key

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/starcoder_wsd80_gradient_conflict_design_20260811"
OUTPUT_JSON = OUTPUT_DIR / "support_partition_audit.json"
OUTPUT_CSV = OUTPUT_DIR / "support_position_histogram.csv"

AUDIT_VERSION = "2026-08-11-support-v1"
COMPONENT_NAME = "dolma/starcoder"
SOURCE_TOKEN_COUNT = 216_567_300_822
TOKENS_PER_SEQUENCE = 2_048
SOURCE_SEQUENCE_COUNT = SOURCE_TOKEN_COUNT // TOKENS_PER_SEQUENCE
HOLDOUT_SEQUENCE_COUNT = 4_096
HOLDOUT_SEED = 2_026_081_102
SUPPORT_POOL_SEED = 2_026_081_101
SUPPORT_SEQUENCE_COUNT = 136_704
SUPPORT_STARTS = {"m100a": 0, "m100b": SUPPORT_SEQUENCE_COUNT}
POSITION_BIN_COUNT = 49

EXPECTED_SUPPORT_DIGESTS = {
    "m100a": "f44bf12ef0da5f401689655cca9ce16c0ca30097e5e9e84123cc62cc7e8d7cd7",
    "m100b": "9dee546086dcd39111c0fad824696d5cbc895bae714d3bca86ba16d8e7ef415c",
}
EXPECTED_SORTED_HOLDOUT_DIGEST = "b40f0c563f181fc8a113728a2223f0ec583aa15b4e74ee2628fc9c76be3f62f9"
EXPECTED_DISTINCT_BLOCK_COUNTS = {"m100a": 1_027, "m100b": 1_026}
EXPECTED_SHARED_BLOCK_COUNT = 512


class _IndexDataset(AsyncDataset[int]):
    """Finite index-only stand-in for the token cache."""

    def __init__(self, length: int):
        self.length = length

    async def async_len(self) -> int:
        return self.length

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[int]:
        return [int(index) for index in indices]


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(values.astype(np.uint64).tobytes()).hexdigest()


def _support_summary(name: str, values: np.ndarray) -> tuple[dict[str, object], list[dict[str, object]]]:
    physical_blocks = np.unique(values // DEFAULT_LM_DATA_SHUFFLE.io_block_size)
    bin_counts, bin_edges = np.histogram(values, bins=POSITION_BIN_COUNT, range=(0, SOURCE_SEQUENCE_COUNT))
    expected_bin_count = len(values) / POSITION_BIN_COUNT
    relative_deviation = np.abs(bin_counts - expected_bin_count) / expected_bin_count
    histogram = [
        {
            "support_id": name,
            "position_bin": index,
            "source_sequence_start": int(bin_edges[index]),
            "source_sequence_end": int(bin_edges[index + 1]),
            "sequence_count": int(bin_counts[index]),
            "relative_deviation_from_equal_bin_count": float(relative_deviation[index]),
        }
        for index in range(POSITION_BIN_COUNT)
    ]
    return (
        {
            "sequence_count": len(values),
            "ordered_sequence_sha256": _digest(values),
            "minimum_source_sequence": int(values.min()),
            "maximum_source_sequence": int(values.max()),
            "distinct_physical_blocks": len(physical_blocks),
            "maximum_equal_bin_relative_deviation": float(relative_deviation.max()),
        },
        histogram,
    )


async def materialize() -> tuple[dict[str, object], list[dict[str, object]]]:
    shuffle = DEFAULT_LM_DATA_SHUFFLE
    base = _IndexDataset(SOURCE_SEQUENCE_COUNT)
    retained, holdout = base.random_holdout_split(
        HOLDOUT_SEQUENCE_COUNT,
        key=_stable_simulated_epoch_subset_key(COMPONENT_NAME, "train_holdout", HOLDOUT_SEED),
        perm_type="feistel",
    )
    support_pool = retained.block_shuffle(
        io_block_size=shuffle.io_block_size,
        window_blocks=shuffle.window_blocks,
        key=_stable_simulated_epoch_subset_key(COMPONENT_NAME, "train", SUPPORT_POOL_SEED),
        perm_type=shuffle.perm_type,
    )

    supports: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, object]] = {}
    histograms: list[dict[str, object]] = []
    for support_id, start in SUPPORT_STARTS.items():
        dataset = support_pool.slice_dataset(start_index=start, end_index=start + SUPPORT_SEQUENCE_COUNT)
        values = np.asarray(await dataset.get_batch(range(SUPPORT_SEQUENCE_COUNT)), dtype=np.int64)
        summary, histogram = _support_summary(support_id, values)
        supports[support_id] = values
        summaries[support_id] = summary
        histograms.extend(histogram)

    holdout_values = np.asarray(await holdout.get_batch(range(HOLDOUT_SEQUENCE_COUNT)), dtype=np.int64)
    partition_state = await retained.partition.state()
    sorted_holdout = partition_state.sorted_holdout_indices.astype(np.int64)
    sequence_overlap = np.intersect1d(supports["m100a"], supports["m100b"])
    block_overlap = np.intersect1d(
        np.unique(supports["m100a"] // shuffle.io_block_size),
        np.unique(supports["m100b"] // shuffle.io_block_size),
    )
    holdout_overlap = {
        support_id: int(np.intersect1d(values, sorted_holdout).size) for support_id, values in supports.items()
    }

    payload: dict[str, object] = {
        "audit_version": AUDIT_VERSION,
        "component_name": COMPONENT_NAME,
        "source_token_count": SOURCE_TOKEN_COUNT,
        "tokens_per_sequence": TOKENS_PER_SEQUENCE,
        "source_sequence_count": SOURCE_SEQUENCE_COUNT,
        "discarded_trailing_token_count": SOURCE_TOKEN_COUNT % TOKENS_PER_SEQUENCE,
        "holdout": {
            "selection": "first outputs of a seeded Feistel permutation",
            "retained_view": "rank-paired sparse tail swaps",
            "sequence_count": HOLDOUT_SEQUENCE_COUNT,
            "seed": HOLDOUT_SEED,
            "ordered_sequence_sha256": _digest(holdout_values),
            "sorted_sequence_sha256": _digest(sorted_holdout),
            "replacement_sequence_count": len(partition_state.retained_replacement_indices),
            "replacement_tail_block_count": len(
                np.unique(partition_state.retained_replacement_sources // shuffle.io_block_size)
            ),
        },
        "support_pool": {
            "selection": "seeded block shuffle of the retained view followed by adjacent slices",
            "seed": SUPPORT_POOL_SEED,
            "sequence_count_per_support": SUPPORT_SEQUENCE_COUNT,
            "io_block_size": shuffle.io_block_size,
            "window_blocks": shuffle.window_blocks,
            "permutation_type": shuffle.perm_type,
        },
        "supports": summaries,
        "cross_support": {
            "shared_sequence_count": int(sequence_overlap.size),
            "shared_physical_block_count": int(block_overlap.size),
            "holdout_overlap_sequence_count": holdout_overlap,
        },
        "interpretation": {
            "identifiable": "exact physical sequence identities and source-position coverage",
            "not_identifiable": (
                "per-sequence programming-language or repository composition; the packed token cache does not retain "
                "those labels"
            ),
            "inferential_unit": (
                "paired training seed; neither sequences nor physical blocks are treated as independent scientific "
                "replicates"
            ),
        },
    }

    assert {name: row["ordered_sequence_sha256"] for name, row in summaries.items()} == EXPECTED_SUPPORT_DIGESTS
    assert {name: row["distinct_physical_blocks"] for name, row in summaries.items()} == (EXPECTED_DISTINCT_BLOCK_COUNTS)
    assert _digest(sorted_holdout) == EXPECTED_SORTED_HOLDOUT_DIGEST
    assert sequence_overlap.size == 0
    assert block_overlap.size == EXPECTED_SHARED_BLOCK_COUNT
    assert all(count == 0 for count in holdout_overlap.values())
    return payload, histograms


def main() -> None:
    payload, histograms = asyncio.run(materialize())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(histograms[0]))
        writer.writeheader()
        writer.writerows(histograms)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
