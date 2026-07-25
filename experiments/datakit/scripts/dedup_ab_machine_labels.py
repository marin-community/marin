# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hash-verify every full-text pair and route it to a safe adjudication method."""

import argparse
import hashlib
import json
from collections.abc import Iterator
from typing import Any

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from pydantic import BaseModel
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.datakit.scripts.dedup_ab_materialize import DedupReviewData

MAX_LOW_OVERLAP_JACCARD = 0.05
MAX_LOW_OVERLAP_CONTAINMENT = 0.15


class DedupMachineLabelsData(BaseModel):
    """Paths and exact counters for the first exhaustive adjudication pass."""

    version: str = "v1"
    review_path: str
    pairs_dir: str
    decisions_dir: str
    counters: dict[str, int | float]


def decision_for_pair(pair: dict[str, Any]) -> dict[str, Any]:
    """Hash-verify one pair and return a machine decision or semantic-review route."""
    member_text = pair["member_text"]
    canonical_text = pair["canonical_text"]
    member_sha256 = hashlib.sha256(member_text.encode()).hexdigest()
    canonical_sha256 = hashlib.sha256(canonical_text.encode()).hexdigest()
    if member_sha256 != pair["raw_sha256"]:
        raise AssertionError(f"Member text hash changed for {pair['review_key']}")
    if canonical_sha256 != pair["canonical_raw_sha256"]:
        raise AssertionError(f"Canonical text hash changed for {pair['review_key']}")

    exact_raw_text = member_sha256 == canonical_sha256
    if exact_raw_text != pair["exact_raw_text"]:
        raise AssertionError(f"Raw identity flag differs from complete texts for {pair['review_key']}")

    if exact_raw_text:
        label = "true_duplicate"
        method = "raw_identity"
        basis = f"Complete raw texts have identical SHA-256 {member_sha256}."
        needs_semantic_review = False
    else:
        word_jaccard = float(pair["word_5gram_jaccard"])
        canonical_containment = float(pair["word_5gram_canonical_containment"])
        member_containment = float(pair["word_5gram_member_containment"])
        texts_truncated = bool(pair["member_text_truncated_for_minhash"] or pair["canonical_text_truncated_for_minhash"])
        strong_false_positive = (
            pair["evidence_class"] == "strong_false_positive"
            and word_jaccard <= MAX_LOW_OVERLAP_JACCARD
            and max(canonical_containment, member_containment) <= MAX_LOW_OVERLAP_CONTAINMENT
        )
        if strong_false_positive and not texts_truncated:
            label = "false_positive"
            method = "low_overlap"
            basis = (
                "Complete texts have negligible ordered word-5-gram overlap: "
                f"Jaccard={word_jaccard:.8f}, canonical containment={canonical_containment:.8f}, "
                f"member containment={member_containment:.8f}."
            )
            needs_semantic_review = False
        else:
            label = ""
            method = ""
            basis = "Requires full-text semantic review."
            needs_semantic_review = True

    return {
        "review_key": pair["review_key"],
        "variant": pair["variant"],
        "member_source_main_dir": pair["member_source_main_dir"],
        "member_basename": pair["member_basename"],
        "member_id": pair["member_id"],
        "canonical_source_main_dir": pair["canonical_source_main_dir"],
        "canonical_basename": pair["canonical_basename"],
        "canonical_id": pair["canonical_id"],
        "raw_sha256": member_sha256,
        "canonical_raw_sha256": canonical_sha256,
        "label": label,
        "method": method,
        "basis": basis,
        "needs_semantic_review": needs_semantic_review,
    }


def _pair_decisions(entry: dict[str, str]) -> Iterator[dict[str, Any]]:
    with StoragePath(entry["path"]).open("rb") as handle:
        parquet_file = pq.ParquetFile(handle)
        for batch in parquet_file.iter_batches(batch_size=16):
            for pair in batch.to_pylist():
                decision = decision_for_pair(pair)
                variant = decision["variant"]
                route = "semantic" if decision["needs_semantic_review"] else decision["method"]
                counters.pipeline.update_counter("machine_labels/pairs", 1)
                counters.pipeline.update_counter(f"machine_labels/{variant}/pairs", 1)
                counters.pipeline.update_counter(f"machine_labels/{variant}/{route}", 1)
                yield decision


def generate_machine_labels(
    *,
    review_path: str,
    output_path: str,
    max_workers: int,
) -> DedupMachineLabelsData:
    """Process every materialized pair and persist compact adjudication decisions."""
    review = DedupReviewData.model_validate_json(StoragePath(review_path).read_text())
    pair_files = sorted(str(path) for path in StoragePath(f"{review.pairs_dir.rstrip('/')}/*.parquet").glob())
    if not pair_files:
        raise FileNotFoundError(f"No review pairs under {review.pairs_dir}")

    resources = ResourceConfig(cpu=2, ram="24g", disk="20g", preemptible=False)
    coordinator = ResourceConfig(cpu=4, ram="16g", disk="20g", preemptible=False)
    decisions_dir = f"{output_path.rstrip('/')}/decisions"
    context = ZephyrContext(
        name="dedup-ab-machine-labels",
        max_workers=max_workers,
        resources=resources,
        coordinator_resources=coordinator,
    )
    pipeline = (
        Dataset.from_list([{"path": path} for path in pair_files])
        .flat_map(_pair_decisions)
        .write_parquet(f"{decisions_dir}/part-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    outcome = context.execute(pipeline, verbose=True)
    pairs = int(outcome.counters.get("machine_labels/pairs", 0))
    expected_pairs = int(review.counters.get("audit/materialize/pairs", 0))
    if pairs != expected_pairs:
        raise AssertionError(f"Machine-label coverage mismatch: {pairs}/{expected_pairs}")
    return DedupMachineLabelsData(
        review_path=review_path,
        pairs_dir=review.pairs_dir,
        decisions_dir=decisions_dir,
        counters=dict(outcome.counters),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-workers", type=int, default=128)
    args = parser.parse_args()
    configure_logging()

    result = generate_machine_labels(
        review_path=args.review,
        output_path=args.output,
        max_workers=args.max_workers,
    )
    StoragePath(f"{args.output.rstrip('/')}/machine-labels.json").write_text(
        json.dumps(result.model_dump(), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
