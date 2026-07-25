# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the complete full-text workload routed to semantic adjudication."""

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Any

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from pydantic import BaseModel
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.datakit.scripts.dedup_ab_machine_labels import DedupMachineLabelsData, decision_for_pair
from experiments.datakit.scripts.dedup_ab_materialize import DedupReviewData
from experiments.datakit.scripts.dedup_ab_semantic_judge import MAX_DIRECT_CHARS, chunk_review_units
from experiments.datakit.scripts.dedup_ab_summarize import _fraction_bin, _length_bin

SUM_FIELDS = (
    "pairs",
    "semantic_pairs",
    "semantic_raw_chars",
    "semantic_review_units",
    "minimum_model_requests",
    "maximum_model_requests",
    "direct_pairs",
    "chunked_pairs",
)
MAX_FIELDS = ("max_member_raw_chars", "max_canonical_raw_chars", "max_combined_raw_chars")


class SemanticWorkloadData(BaseModel):
    """Exact aggregate size and composition of the semantic-review queue."""

    version: str = "v2"
    review_path: str
    machine_labels_path: str
    summaries_dir: str
    summary: dict[str, Any]


def summarize_pairs(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Revalidate pair routing and aggregate semantic-review workload."""
    counts: Counter[str] = Counter()
    pairs = 0
    semantic_pairs = 0
    semantic_raw_chars = 0
    semantic_review_units = 0
    direct_pairs = 0
    chunked_pairs = 0
    max_member_raw_chars = 0
    max_canonical_raw_chars = 0
    max_combined_raw_chars = 0
    for pair in records:
        pairs += 1
        member_raw_chars = len(pair["member_text"])
        canonical_raw_chars = len(pair["canonical_text"])
        if member_raw_chars != pair["raw_chars"]:
            raise AssertionError(f"Member length changed for {pair['review_key']}")
        if canonical_raw_chars != pair["canonical_raw_chars"]:
            raise AssertionError(f"Canonical length changed for {pair['review_key']}")

        decision = decision_for_pair(pair)
        if not decision["needs_semantic_review"]:
            continue

        variant = pair["variant"]
        combined_raw_chars = member_raw_chars + canonical_raw_chars
        if combined_raw_chars <= MAX_DIRECT_CHARS:
            review_units = 1
            direct_pairs += 1
            counts[f"{variant}/direct_pairs"] += 1
        else:
            review_units = len(chunk_review_units(pair))
            chunked_pairs += 1
            counts[f"{variant}/chunked_pairs"] += 1
        semantic_pairs += 1
        semantic_raw_chars += combined_raw_chars
        semantic_review_units += review_units
        max_member_raw_chars = max(max_member_raw_chars, member_raw_chars)
        max_canonical_raw_chars = max(max_canonical_raw_chars, canonical_raw_chars)
        max_combined_raw_chars = max(max_combined_raw_chars, combined_raw_chars)
        counts[f"{variant}/pairs"] += 1
        counts[f"{variant}/raw_chars"] += combined_raw_chars
        counts[f"{variant}/review_units"] += review_units
        counts[f"{variant}/combined_raw_chars/{_length_bin(combined_raw_chars)}"] += 1
        counts[f"{variant}/word_5gram_jaccard/{_fraction_bin(float(pair['word_5gram_jaccard']))}"] += 1
        counts[
            f"{variant}/word_5gram_canonical_containment/"
            f"{_fraction_bin(float(pair['word_5gram_canonical_containment']))}"
        ] += 1
        counts[
            f"{variant}/word_5gram_member_containment/" f"{_fraction_bin(float(pair['word_5gram_member_containment']))}"
        ] += 1
        counts[f"{variant}/length_ratio/{_fraction_bin(float(pair['length_ratio']))}"] += 1
        counts[f"{variant}/cross_source/{bool(pair['cross_source'])}"] += 1
        counts[f"{variant}/exact_clean_text/{bool(pair['exact_clean_text'])}"] += 1
        counts[f"{variant}/member_clean_text_contained/{bool(pair['member_clean_text_contained'])}"] += 1
        truncated = bool(pair["member_text_truncated_for_minhash"] or pair["canonical_text_truncated_for_minhash"])
        counts[f"{variant}/either_text_truncated_for_minhash/{truncated}"] += 1

    return {
        "pairs": pairs,
        "semantic_pairs": semantic_pairs,
        "semantic_raw_chars": semantic_raw_chars,
        "semantic_review_units": semantic_review_units,
        "minimum_model_requests": semantic_review_units * 2,
        "maximum_model_requests": semantic_review_units * 3,
        "direct_pairs": direct_pairs,
        "chunked_pairs": chunked_pairs,
        "max_member_raw_chars": max_member_raw_chars,
        "max_canonical_raw_chars": max_canonical_raw_chars,
        "max_combined_raw_chars": max_combined_raw_chars,
        "counts": dict(sorted(counts.items())),
    }


def merge_summaries(summaries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Merge independently computed file summaries."""
    result: dict[str, Any] = {field: 0 for field in (*SUM_FIELDS, *MAX_FIELDS)}
    counts: Counter[str] = Counter()
    for summary in summaries:
        for field in SUM_FIELDS:
            result[field] += int(summary[field])
        for field in MAX_FIELDS:
            result[field] = max(result[field], int(summary[field]))
        counts.update({key: int(value) for key, value in summary["counts"].items()})
    result["counts"] = dict(sorted(counts.items()))
    return result


def _file_summary(entry: dict[str, str]) -> dict[str, str]:
    def records() -> Iterator[dict[str, Any]]:
        with StoragePath(entry["path"]).open("rb") as handle:
            for batch in pq.ParquetFile(handle).iter_batches(batch_size=16):
                yield from batch.to_pylist()

    return {"summary_json": json.dumps(summarize_pairs(records()), separators=(",", ":"), sort_keys=True)}


def _summary_records(directory: str) -> Iterator[dict[str, Any]]:
    paths = sorted(str(path) for path in StoragePath(f"{directory.rstrip('/')}/*.parquet").glob())
    if not paths:
        raise FileNotFoundError(f"No semantic workload summaries under {directory}")
    for path in paths:
        with StoragePath(path).open("rb") as handle:
            for value in pq.ParquetFile(handle).read(columns=["summary_json"])["summary_json"].to_pylist():
                yield json.loads(value)


def measure_semantic_workload(
    *,
    review_path: str,
    machine_labels_path: str,
    output_path: str,
    max_workers: int,
) -> SemanticWorkloadData:
    """Hash-check and size every pair routed to semantic review."""
    review = DedupReviewData.model_validate_json(StoragePath(review_path).read_text())
    machine = DedupMachineLabelsData.model_validate_json(StoragePath(machine_labels_path).read_text())
    if machine.review_path != review_path:
        raise AssertionError(
            f"Machine labels refer to a different review corpus: {machine.review_path} != {review_path}"
        )
    pair_files = sorted(str(path) for path in StoragePath(f"{review.pairs_dir.rstrip('/')}/*.parquet").glob())
    if not pair_files:
        raise FileNotFoundError(f"No review pairs under {review.pairs_dir}")

    resources = ResourceConfig(cpu=2, ram="24g", disk="20g", preemptible=False)
    coordinator = ResourceConfig(cpu=4, ram="16g", disk="20g", preemptible=False)
    context = ZephyrContext(
        name="dedup-ab-semantic-workload",
        max_workers=max_workers,
        resources=resources,
        coordinator_resources=coordinator,
    )
    summaries_dir = f"{output_path.rstrip('/')}/summaries"
    pipeline = (
        Dataset.from_list([{"path": path} for path in pair_files])
        .map(_file_summary)
        .write_parquet(f"{summaries_dir}/part-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    context.execute(pipeline, verbose=True)
    summary = merge_summaries(_summary_records(summaries_dir))

    expected_pairs = int(review.counters.get("audit/materialize/pairs", 0))
    expected_machine_pairs = int(machine.counters.get("machine_labels/pairs", 0))
    if summary["pairs"] != expected_pairs or summary["pairs"] != expected_machine_pairs:
        raise AssertionError(
            f"Pair coverage mismatch: workload={summary['pairs']}, "
            f"review={expected_pairs}, machine={expected_machine_pairs}"
        )
    for variant in ("baseline", "treatment"):
        expected_semantic = int(machine.counters.get(f"machine_labels/{variant}/semantic", 0))
        actual_semantic = int(summary["counts"].get(f"{variant}/pairs", 0))
        if actual_semantic != expected_semantic:
            raise AssertionError(
                f"{variant} semantic coverage mismatch: workload={actual_semantic}, machine={expected_semantic}"
            )
    expected_semantic = sum(
        int(machine.counters.get(f"machine_labels/{variant}/semantic", 0)) for variant in ("baseline", "treatment")
    )
    if summary["semantic_pairs"] != expected_semantic:
        raise AssertionError(
            f"Semantic coverage mismatch: workload={summary['semantic_pairs']}, machine={expected_semantic}"
        )
    return SemanticWorkloadData(
        review_path=review_path,
        machine_labels_path=machine_labels_path,
        summaries_dir=summaries_dir,
        summary=summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True)
    parser.add_argument("--machine-labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-workers", type=int, default=128)
    args = parser.parse_args()

    result = measure_semantic_workload(
        review_path=args.review,
        machine_labels_path=args.machine_labels,
        output_path=args.output,
        max_workers=args.max_workers,
    )
    StoragePath(f"{args.output.rstrip('/')}/workload.json").write_text(result.model_dump_json(indent=2))
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    configure_logging()
    main()
