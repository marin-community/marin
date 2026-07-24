# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize every audited drop with its complete member and canonical text.

Score rows retain stable normalized-shard references. This pipeline groups all
requests for one normalized shard, reads that shard once, verifies the raw-text
hashes recorded by the audit, and emits one full-text row per dropped pair.
The result is the exhaustive semantic-review corpus; it is not a sample.
"""

import argparse
import bisect
import hashlib
import json
from collections.abc import Iterator
from itertools import pairwise
from typing import Any

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from pydantic import BaseModel
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext


class DedupReviewData(BaseModel):
    """Paths and exact counters for an exhaustive full-text review corpus."""

    version: str = "v2"
    scores_dir: str
    pairs_dir: str
    counters: dict[str, int | float]


def _review_key(score: dict[str, Any]) -> str:
    return json.dumps(
        [
            score["variant"],
            score["source_main_dir"],
            score["basename"],
            score["id"],
            score["canonical_source_main_dir"],
            score["canonical_basename"],
            score["canonical_id"],
        ],
        separators=(",", ":"),
    )


def _score_requests(entry: dict[str, str]) -> Iterator[dict[str, str]]:
    with StoragePath(entry["path"]).open("rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches():
            for score in batch.to_pylist():
                if score["role"] != "drop":
                    continue
                review_key = _review_key(score)
                score_json = json.dumps(score, separators=(",", ":"), sort_keys=True)
                counters.pipeline.update_counter("audit/materialize/drop_requests", 1)
                yield {
                    "normalized_path": f"{score['source_main_dir'].rstrip('/')}/{score['basename']}",
                    "review_key": review_key,
                    "side": "member",
                    "doc_id": score["id"],
                    "expected_sha256": score["raw_sha256"],
                    "score_json": score_json,
                }
                yield {
                    "normalized_path": f"{score['canonical_source_main_dir'].rstrip('/')}/{score['canonical_basename']}",
                    "review_key": review_key,
                    "side": "canonical",
                    "doc_id": score["canonical_id"],
                    "expected_sha256": score["canonical_raw_sha256"],
                    "score_json": "",
                }


def _join_requested_texts(normalized_path: str, requests: Iterator[dict[str, str]]) -> Iterator[dict[str, str]]:
    with StoragePath(normalized_path).open("rb") as handle:
        table = pq.ParquetFile(handle).read(columns=["id", "text"])
    ids = table["id"].to_pylist()
    if any(left >= right for left, right in pairwise(ids)):
        raise AssertionError(f"Normalized IDs are not strictly sorted in {normalized_path}")

    for request in requests:
        index = bisect.bisect_left(ids, request["doc_id"])
        if index == len(ids) or ids[index] != request["doc_id"]:
            raise AssertionError(f"Review document {request['doc_id']} is absent from {normalized_path}")
        text = table["text"][index].as_py()
        actual_sha256 = hashlib.sha256(text.encode()).hexdigest()
        if actual_sha256 != request["expected_sha256"]:
            raise AssertionError(
                f"Review document hash changed for {request['doc_id']} in {normalized_path}: "
                f"actual={actual_sha256}, expected={request['expected_sha256']}"
            )
        counters.pipeline.update_counter("audit/materialize/texts", 1)
        counters.pipeline.update_counter("audit/materialize/raw_chars", len(text))
        yield {
            "review_key": request["review_key"],
            "side": request["side"],
            "score_json": request["score_json"],
            "text": text,
        }


def _pair_texts(review_key: str, records: Iterator[dict[str, str]]) -> dict[str, Any]:
    by_side: dict[str, dict[str, str]] = {}
    for record in records:
        side = record["side"]
        if side in by_side:
            raise AssertionError(f"Duplicate {side} text for {review_key}")
        by_side[side] = record
    if by_side.keys() != {"member", "canonical"}:
        raise AssertionError(f"Incomplete text pair for {review_key}: sides={sorted(by_side)}")

    score = json.loads(by_side["member"]["score_json"])
    if _review_key(score) != review_key:
        raise AssertionError(f"Review key changed while materializing {review_key}")
    member_text = by_side["member"]["text"]
    canonical_text = by_side["canonical"]["text"]
    if hashlib.sha256(member_text.encode()).hexdigest() != score["raw_sha256"]:
        raise AssertionError(f"Member hash changed while pairing {review_key}")
    if hashlib.sha256(canonical_text.encode()).hexdigest() != score["canonical_raw_sha256"]:
        raise AssertionError(f"Canonical hash changed while pairing {review_key}")

    counters.pipeline.update_counter("audit/materialize/pairs", 1)
    return {
        "review_key": review_key,
        "variant": score["variant"],
        "member_source_main_dir": score["source_main_dir"],
        "member_basename": score["basename"],
        "member_id": score["id"],
        "canonical_source_main_dir": score["canonical_source_main_dir"],
        "canonical_basename": score["canonical_basename"],
        "canonical_id": score["canonical_id"],
        "evidence_class": score["evidence_class"],
        "cross_source": score["cross_source"],
        "raw_chars": score["raw_chars"],
        "canonical_raw_chars": score["canonical_raw_chars"],
        "clean_chars": score["clean_chars"],
        "canonical_clean_chars": score["canonical_clean_chars"],
        "length_ratio": score["length_ratio"],
        "member_is_longer": score["member_is_longer"],
        "member_text_truncated_for_minhash": score["member_text_truncated_for_minhash"],
        "canonical_text_truncated_for_minhash": score["canonical_text_truncated_for_minhash"],
        "exact_raw_text": score["exact_raw_text"],
        "exact_clean_text": score["exact_clean_text"],
        "member_clean_text_contained": score["member_clean_text_contained"],
        "char_5gram_jaccard": score["char_5gram_jaccard"],
        "char_5gram_canonical_containment": score["char_5gram_canonical_containment"],
        "char_5gram_member_containment": score["char_5gram_member_containment"],
        "word_5gram_jaccard": score["word_5gram_jaccard"],
        "word_5gram_canonical_containment": score["word_5gram_canonical_containment"],
        "word_5gram_member_containment": score["word_5gram_member_containment"],
        "baseline_shared_buckets": score["baseline_shared_buckets"],
        "treatment_shared_buckets": score["treatment_shared_buckets"],
        "raw_sha256": score["raw_sha256"],
        "canonical_raw_sha256": score["canonical_raw_sha256"],
        "member_text": member_text,
        "canonical_text": canonical_text,
    }


def materialize(
    *,
    scores_dir: str,
    output_path: str,
    max_workers: int,
) -> DedupReviewData:
    """Build and hash-verify the complete full-text review corpus."""
    score_files = sorted(str(path) for path in StoragePath(f"{scores_dir.rstrip('/')}/*.parquet").glob())
    if not score_files:
        raise FileNotFoundError(f"No score Parquet files under {scores_dir}")

    resources = ResourceConfig(cpu=2, ram="24g", disk="20g", preemptible=False)
    coordinator = ResourceConfig(cpu=4, ram="16g", disk="20g", preemptible=False)
    context = ZephyrContext(
        name="dedup-ab-materialize-review",
        max_workers=max_workers,
        resources=resources,
        coordinator_resources=coordinator,
    )
    pairs_dir = f"{output_path.rstrip('/')}/pairs"
    pipeline = (
        Dataset.from_list([{"path": path} for path in score_files])
        .flat_map(_score_requests)
        .group_by(
            key=lambda request: request["normalized_path"],
            sort_by=lambda request: f"{request['doc_id']}|{request['review_key']}|{request['side']}",
            reducer=_join_requested_texts,
            num_output_shards=max_workers,
        )
        .group_by(
            key=lambda record: record["review_key"],
            reducer=_pair_texts,
            num_output_shards=max_workers,
        )
        .write_parquet(f"{pairs_dir}/part-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    outcome = context.execute(pipeline, verbose=True)
    drops = int(outcome.counters.get("audit/materialize/drop_requests", 0))
    texts = int(outcome.counters.get("audit/materialize/texts", 0))
    pairs = int(outcome.counters.get("audit/materialize/pairs", 0))
    if texts != 2 * drops or pairs != drops:
        raise AssertionError(f"Review corpus accounting mismatch: drops={drops}, texts={texts}, pairs={pairs}")
    return DedupReviewData(
        scores_dir=scores_dir,
        pairs_dir=pairs_dir,
        counters=dict(outcome.counters),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-workers", type=int, default=128)
    args = parser.parse_args()

    result = materialize(
        scores_dir=args.scores_dir,
        output_path=args.output,
        max_workers=args.max_workers,
    )
    StoragePath(f"{args.output.rstrip('/')}/review.json").write_text(result.model_dump_json(indent=2))


if __name__ == "__main__":
    configure_logging()
    main()
