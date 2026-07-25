# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Open deterministic, hash-verified batches of semantic-review pairs."""

import argparse
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import pyarrow.parquet as pq
from pydantic import BaseModel
from rigging.filesystem import StoragePath

from experiments.datakit.scripts.dedup_ab_machine_labels import DedupMachineLabelsData, decision_for_pair

DECISION_FIELDS = (
    "review_key",
    "variant",
    "member_source_main_dir",
    "member_basename",
    "member_id",
    "canonical_source_main_dir",
    "canonical_basename",
    "canonical_id",
    "raw_sha256",
    "canonical_raw_sha256",
    "label",
    "method",
    "basis",
    "needs_semantic_review",
)


class SemanticBatchData(BaseModel):
    """One resumable slice of a semantic-review decision shard."""

    version: str = "v1"
    machine_labels_path: str
    decision_file: str
    decision_file_index: int
    semantic_offset: int
    next_semantic_offset: int
    total_semantic_in_file: int
    cases: list[dict[str, Any]]


def _records(path: str) -> list[dict[str, Any]]:
    with StoragePath(path).open("rb") as handle:
        return pq.ParquetFile(handle).read().to_pylist()


def _requested_pair_rows(path: str, row_indices: set[int]) -> dict[int, dict[str, Any]]:
    if any(index < 0 for index in row_indices):
        raise ValueError(f"Negative pair row index requested from {path}")
    if not row_indices:
        return {}

    result: dict[int, dict[str, Any]] = {}
    remaining = set(row_indices)
    with StoragePath(path).open("rb") as handle:
        parquet_file = pq.ParquetFile(handle)
        row_start = 0
        for row_group_index in range(parquet_file.num_row_groups):
            row_count = parquet_file.metadata.row_group(row_group_index).num_rows
            group_indices = sorted(index for index in remaining if row_start <= index < row_start + row_count)
            if group_indices:
                rows = parquet_file.read_row_group(row_group_index).to_pylist()
                for index in group_indices:
                    result[index] = rows[index - row_start]
                    remaining.remove(index)
            row_start += row_count
            if not remaining:
                break
    if remaining:
        raise IndexError(f"Pair row indices outside {path}: {sorted(remaining)}")
    return result


def _verified_case(decision: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    expected = decision_for_pair(pair)
    mismatches = [field for field in DECISION_FIELDS if decision.get(field) != expected.get(field)]
    if mismatches:
        raise AssertionError(f"Machine decision differs from referenced pair {decision['review_key']}: {mismatches}")
    if not decision["needs_semantic_review"]:
        raise AssertionError(f"Machine-resolved pair entered semantic batch: {decision['review_key']}")
    return {
        "review_key": decision["review_key"],
        "variant": decision["variant"],
        "pair_path": decision["pair_path"],
        "pair_row_index": decision["pair_row_index"],
        "member_source_main_dir": decision["member_source_main_dir"],
        "member_basename": decision["member_basename"],
        "member_id": decision["member_id"],
        "canonical_source_main_dir": decision["canonical_source_main_dir"],
        "canonical_basename": decision["canonical_basename"],
        "canonical_id": decision["canonical_id"],
        "raw_sha256": decision["raw_sha256"],
        "canonical_raw_sha256": decision["canonical_raw_sha256"],
        "evidence_class": pair["evidence_class"],
        "cross_source": pair["cross_source"],
        "raw_chars": pair["raw_chars"],
        "canonical_raw_chars": pair["canonical_raw_chars"],
        "length_ratio": pair["length_ratio"],
        "member_is_longer": pair["member_is_longer"],
        "member_text_truncated_for_minhash": pair["member_text_truncated_for_minhash"],
        "canonical_text_truncated_for_minhash": pair["canonical_text_truncated_for_minhash"],
        "exact_clean_text": pair["exact_clean_text"],
        "member_clean_text_contained": pair["member_clean_text_contained"],
        "char_5gram_jaccard": pair["char_5gram_jaccard"],
        "char_5gram_canonical_containment": pair["char_5gram_canonical_containment"],
        "char_5gram_member_containment": pair["char_5gram_member_containment"],
        "word_5gram_jaccard": pair["word_5gram_jaccard"],
        "word_5gram_canonical_containment": pair["word_5gram_canonical_containment"],
        "word_5gram_member_containment": pair["word_5gram_member_containment"],
        "baseline_shared_buckets": pair["baseline_shared_buckets"],
        "treatment_shared_buckets": pair["treatment_shared_buckets"],
        "member_text": pair["member_text"],
        "canonical_text": pair["canonical_text"],
    }


def load_semantic_cases(
    decisions: Iterable[dict[str, Any]],
    *,
    semantic_offset: int,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    """Load one deterministic slice and revalidate every referenced full-text row."""
    if semantic_offset < 0:
        raise ValueError(f"semantic_offset must be non-negative, got {semantic_offset}")
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    semantic = [decision for decision in decisions if decision["needs_semantic_review"]]
    selected = semantic[semantic_offset : semantic_offset + limit]
    requested: dict[str, set[int]] = defaultdict(set)
    for decision in selected:
        path = decision["pair_path"]
        row_index = int(decision["pair_row_index"])
        if row_index in requested[path]:
            raise AssertionError(f"Duplicate pair row reference: {path}:{row_index}")
        requested[path].add(row_index)

    pairs_by_path = {path: _requested_pair_rows(path, indices) for path, indices in requested.items()}
    cases = [
        _verified_case(decision, pairs_by_path[decision["pair_path"]][int(decision["pair_row_index"])])
        for decision in selected
    ]
    return cases, len(semantic)


def build_batch(
    *,
    machine_labels_path: str,
    decision_file_index: int,
    semantic_offset: int,
    limit: int,
) -> SemanticBatchData:
    """Build one independently resumable semantic-review batch."""
    machine = DedupMachineLabelsData.model_validate_json(StoragePath(machine_labels_path).read_text())
    decision_files = sorted(str(path) for path in StoragePath(f"{machine.decisions_dir.rstrip('/')}/*.parquet").glob())
    if not decision_files:
        raise FileNotFoundError(f"No machine decision files under {machine.decisions_dir}")
    if not 0 <= decision_file_index < len(decision_files):
        raise IndexError(f"Decision file index {decision_file_index} outside 0..{len(decision_files) - 1}")
    decision_file = decision_files[decision_file_index]
    cases, total_semantic = load_semantic_cases(
        _records(decision_file),
        semantic_offset=semantic_offset,
        limit=limit,
    )
    return SemanticBatchData(
        machine_labels_path=machine_labels_path,
        decision_file=decision_file,
        decision_file_index=decision_file_index,
        semantic_offset=semantic_offset,
        next_semantic_offset=semantic_offset + len(cases),
        total_semantic_in_file=total_semantic,
        cases=cases,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-labels", required=True)
    parser.add_argument("--decision-file-index", type=int, required=True)
    parser.add_argument("--semantic-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    batch = build_batch(
        machine_labels_path=args.machine_labels,
        decision_file_index=args.decision_file_index,
        semantic_offset=args.semantic_offset,
        limit=args.limit,
    )
    StoragePath(args.output).write_text(batch.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
