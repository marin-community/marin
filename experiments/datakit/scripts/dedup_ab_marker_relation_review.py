# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize changed capped/converged marker relationships for semantic review."""

import argparse
import bisect
import hashlib
import json
from collections.abc import Iterable
from functools import cache
from itertools import pairwise
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit.scripts.dedup_ab_audit import (
    TEXT_CAP_CHARS,
    _artifact_result,
    _clean_texts,
    _evidence_class,
    _marker_map,
    _set_metrics,
    _shards,
    _shingles,
)
from experiments.datakit.scripts.dedup_ab_machine_labels import DedupMachineLabelsData, decision_for_pair
from experiments.datakit.scripts.dedup_ab_marker_diff import MarkerDiffData


class MarkerRelationReviewData(BaseModel):
    """Exact paths and counts for changed marker-relationship review pairs."""

    version: str = "v2"
    marker_diff_path: str
    pairs_path: str
    pairs_sha256: str
    decisions_path: str
    decisions_sha256: str
    machine_labels_path: str
    differences: int
    behavior_changes: int
    metadata_only_changes: int
    relations: int
    canonical_locations: dict[str, dict[str, dict[str, str]]]
    orphan_cluster_ids: dict[str, list[str]]


def canonical_locations(
    records: Iterable[dict[str, Any]],
    cluster_ids: set[str],
) -> dict[str, dict[str, str]]:
    """Find exactly one canonical occurrence for every requested cluster."""
    found: dict[str, dict[str, str]] = {}
    for record in records:
        marker = record["marker"]
        cluster_id = marker["dup_cluster_id"]
        if cluster_id not in cluster_ids or not marker["is_cluster_canonical"]:
            continue
        location = {
            "source_main_dir": record["source_main_dir"],
            "basename": record["basename"],
            "id": record["id"],
        }
        prior = found.get(cluster_id)
        if prior is not None:
            raise AssertionError(f"Cluster {cluster_id} has multiple canonicals: {prior}, {location}")
        found[cluster_id] = location
    missing = cluster_ids - found.keys()
    if missing:
        raise AssertionError(f"Canonical markers are absent for clusters: {sorted(missing)}")
    return found


def canonical_inventory(
    records: Iterable[dict[str, Any]],
    cluster_ids: set[str],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Find canonical occurrences while preserving nonconverged orphan labels."""
    found: dict[str, dict[str, str]] = {}
    for record in records:
        marker = record["marker"]
        cluster_id = marker["dup_cluster_id"]
        if cluster_id not in cluster_ids or not marker["is_cluster_canonical"]:
            continue
        location = {
            "source_main_dir": record["source_main_dir"],
            "basename": record["basename"],
            "id": record["id"],
        }
        prior = found.get(cluster_id)
        if prior is not None:
            raise AssertionError(f"Cluster {cluster_id} has multiple canonicals: {prior}, {location}")
        found[cluster_id] = location
    return found, sorted(cluster_ids - found.keys())


def _kept_by_datakit(difference: dict[str, Any], prefix: str) -> bool:
    """Return the exact KEEP_DOC result for one marker arm."""
    return difference[f"{prefix}_is_canonical"] is not False


def _marker_records(dedup: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for source_main_dir in sorted(dedup["sources"]):
        attr_dir = dedup["sources"][source_main_dir]["attr_dir"]
        for basename, path in sorted(_shards(attr_dir).items()):
            for doc_id, marker in _marker_map(path).items():
                yield {
                    "source_main_dir": source_main_dir,
                    "basename": basename,
                    "id": doc_id,
                    "marker": marker,
                }


def _difference_rows(marker_diff: MarkerDiffData) -> list[dict[str, Any]]:
    paths = sorted(str(path) for path in StoragePath(f"{marker_diff.differences_dir.rstrip('/')}/*.parquet").glob())
    if not paths:
        raise FileNotFoundError(f"No marker differences under {marker_diff.differences_dir}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        with StoragePath(path).open("rb") as handle:
            rows.extend(pq.ParquetFile(handle).read().to_pylist())
    expected = int(marker_diff.counters.get("marker_diff/differences", 0))
    if len(rows) != expected:
        raise AssertionError(f"Marker-difference coverage mismatch: {len(rows)}/{expected}")
    return sorted(rows, key=lambda row: (row["source_main_dir"], row["basename"], row["id"]))


@cache
def _document_text(source_main_dir: str, basename: str, doc_id: str) -> str:
    path = f"{source_main_dir.rstrip('/')}/{basename}"
    with StoragePath(path).open("rb") as handle:
        table = pq.ParquetFile(handle).read(columns=["id", "text"])
    ids = table["id"].to_pylist()
    if any(left >= right for left, right in pairwise(ids)):
        raise AssertionError(f"Normalized IDs are not strictly sorted in {path}")
    index = bisect.bisect_left(ids, doc_id)
    if index == len(ids) or ids[index] != doc_id:
        raise AssertionError(f"Document {doc_id} is absent from {path}")
    return table["text"][index].as_py()


def relationship_pair(
    *,
    difference: dict[str, Any],
    variant: str,
    cluster_id: str,
    canonical: dict[str, str],
    member_text: str,
    canonical_text: str,
) -> dict[str, Any]:
    """Build one complete directional pair with exact lexical evidence."""
    member_clean, canonical_clean = _clean_texts([member_text, canonical_text])
    member_char = _shingles(member_clean, "char")
    canonical_char = _shingles(canonical_clean, "char")
    char_jaccard, canonical_char_containment, member_char_containment = _set_metrics(
        canonical_char,
        member_char,
    )
    member_word = _shingles(member_clean, "word")
    canonical_word = _shingles(canonical_clean, "word")
    word_jaccard, canonical_word_containment, member_word_containment = _set_metrics(
        canonical_word,
        member_word,
    )
    member_sha256 = hashlib.sha256(member_text.encode()).hexdigest()
    canonical_sha256 = hashlib.sha256(canonical_text.encode()).hexdigest()
    max_chars = max(len(member_clean), len(canonical_clean))
    metric_evidence_class = _evidence_class(
        exact_raw_text=member_sha256 == canonical_sha256,
        word_jaccard=word_jaccard,
        canonical_word_containment=canonical_word_containment,
        member_word_containment=member_word_containment,
    )
    review_key = json.dumps(
        [
            variant,
            difference["source_main_dir"],
            difference["basename"],
            difference["id"],
            canonical["source_main_dir"],
            canonical["basename"],
            canonical["id"],
        ],
        separators=(",", ":"),
    )
    return {
        "review_key": review_key,
        "variant": variant,
        "relationship_cluster_id": cluster_id,
        "change_kind": difference["change_kind"],
        "capped_cluster_id": difference["capped_cluster_id"],
        "converged_cluster_id": difference["converged_cluster_id"],
        "member_source_main_dir": difference["source_main_dir"],
        "member_basename": difference["basename"],
        "member_id": difference["id"],
        "canonical_source_main_dir": canonical["source_main_dir"],
        "canonical_basename": canonical["basename"],
        "canonical_id": canonical["id"],
        # These targeted graph changes are deliberately routed through semantic
        # review even when the lexical metrics alone look decisive.
        "evidence_class": "ambiguous",
        "metric_evidence_class": metric_evidence_class,
        "cross_source": difference["source_main_dir"] != canonical["source_main_dir"],
        "raw_chars": len(member_text),
        "canonical_raw_chars": len(canonical_text),
        "clean_chars": len(member_clean),
        "canonical_clean_chars": len(canonical_clean),
        "length_ratio": min(len(member_clean), len(canonical_clean)) / max_chars if max_chars else 1.0,
        "member_is_longer": len(member_text) > len(canonical_text),
        "member_text_truncated_for_minhash": len(member_text) > TEXT_CAP_CHARS,
        "canonical_text_truncated_for_minhash": len(canonical_text) > TEXT_CAP_CHARS,
        "exact_raw_text": member_sha256 == canonical_sha256,
        "exact_clean_text": member_clean == canonical_clean,
        "member_clean_text_contained": member_clean in canonical_clean,
        "char_5gram_jaccard": char_jaccard,
        "char_5gram_canonical_containment": canonical_char_containment,
        "char_5gram_member_containment": member_char_containment,
        "word_5gram_jaccard": word_jaccard,
        "word_5gram_canonical_containment": canonical_word_containment,
        "word_5gram_member_containment": member_word_containment,
        "baseline_shared_buckets": -1,
        "treatment_shared_buckets": -1,
        "raw_sha256": member_sha256,
        "canonical_raw_sha256": canonical_sha256,
        "member_text": member_text,
        "canonical_text": canonical_text,
    }


def _parquet_bytes(rows: list[dict[str, Any]]) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows), sink, compression="zstd")
    return sink.getvalue().to_pybytes()


def _write_verified(path: str, rows: list[dict[str, Any]]) -> str:
    payload = _parquet_bytes(rows)
    target = StoragePath(path)
    target.parent.mkdirs()
    target.write_bytes(payload)
    persisted = target.read_bytes()
    digest = hashlib.sha256(persisted).hexdigest()
    if persisted != payload:
        raise AssertionError(f"Persisted bytes differ for {path}")
    with pa.BufferReader(persisted) as handle:
        persisted_rows = pq.ParquetFile(handle).read().to_pylist()
    if persisted_rows != rows:
        raise AssertionError(f"Persisted rows differ for {path}")
    return digest


def materialize_relation_review(
    *,
    marker_diff_path: str,
    output_path: str,
) -> MarkerRelationReviewData:
    """Separate output changes from inert metadata changes and review changed drops."""
    marker_diff = MarkerDiffData.model_validate_json(StoragePath(marker_diff_path).read_text())
    differences = _difference_rows(marker_diff)
    behavior_changes = [
        row for row in differences if _kept_by_datakit(row, "capped") != _kept_by_datakit(row, "converged")
    ]

    arms = {
        "baseline_cap50": (marker_diff.capped_dedup, "capped"),
        "baseline_converged": (marker_diff.converged_dedup, "converged"),
    }
    canonical_by_arm: dict[str, dict[str, dict[str, str]]] = {}
    orphan_by_arm: dict[str, list[str]] = {}
    for variant, (dedup_path, prefix) in arms.items():
        dedup = _artifact_result(dedup_path)
        cluster_ids = {row[f"{prefix}_cluster_id"] for row in differences if row[f"{prefix}_cluster_id"] is not None}
        canonical_by_arm[variant], orphan_by_arm[variant] = canonical_inventory(
            _marker_records(dedup),
            cluster_ids,
        )

    pairs: list[dict[str, Any]] = []
    for difference in behavior_changes:
        member_text = _document_text(
            difference["source_main_dir"],
            difference["basename"],
            difference["id"],
        )
        for variant, (_, prefix) in arms.items():
            if _kept_by_datakit(difference, prefix):
                continue
            cluster_id = difference[f"{prefix}_cluster_id"]
            canonical = canonical_by_arm[variant].get(cluster_id)
            if canonical is None:
                raise AssertionError(f"Behavior-changing drop references orphan cluster {cluster_id} in {variant}")
            canonical_text = _document_text(
                canonical["source_main_dir"],
                canonical["basename"],
                canonical["id"],
            )
            pairs.append(
                relationship_pair(
                    difference=difference,
                    variant=variant,
                    cluster_id=cluster_id,
                    canonical=canonical,
                    member_text=member_text,
                    canonical_text=canonical_text,
                )
            )

    pairs_path = f"{output_path.rstrip('/')}/pairs/part-00000-of-00001.parquet"
    pairs_sha256 = _write_verified(pairs_path, pairs)
    decisions = []
    for row_index, pair in enumerate(pairs):
        decision = decision_for_pair(pair)
        if not decision["needs_semantic_review"]:
            raise AssertionError(f"Changed relation bypassed semantic review: {pair['review_key']}")
        decisions.append(
            {
                **decision,
                "pair_path": pairs_path,
                "pair_row_index": row_index,
            }
        )
    decisions_path = f"{output_path.rstrip('/')}/decisions/part-00000-of-00001.parquet"
    decisions_sha256 = _write_verified(decisions_path, decisions)

    machine_labels_path = f"{output_path.rstrip('/')}/machine-labels.json"
    pairs_by_variant = {variant: sum(pair["variant"] == variant for pair in pairs) for variant in arms}
    counters: dict[str, int | float] = {
        "machine_labels/pairs": len(decisions),
        **{
            f"machine_labels/{variant}/{kind}": pairs_by_variant[variant]
            for variant in arms
            for kind in ("pairs", "semantic")
        },
    }
    relation_review_path = f"{output_path.rstrip('/')}/relation-review.json"
    machine = DedupMachineLabelsData(
        review_path=relation_review_path,
        pairs_dir=f"{output_path.rstrip('/')}/pairs",
        decisions_dir=f"{output_path.rstrip('/')}/decisions",
        counters=counters,
    )
    StoragePath(machine_labels_path).write_text(machine.model_dump_json(indent=2))
    result = MarkerRelationReviewData(
        marker_diff_path=marker_diff_path,
        pairs_path=pairs_path,
        pairs_sha256=pairs_sha256,
        decisions_path=decisions_path,
        decisions_sha256=decisions_sha256,
        machine_labels_path=machine_labels_path,
        differences=len(differences),
        behavior_changes=len(behavior_changes),
        metadata_only_changes=len(differences) - len(behavior_changes),
        relations=len(pairs),
        canonical_locations=canonical_by_arm,
        orphan_cluster_ids=orphan_by_arm,
    )
    StoragePath(relation_review_path).write_text(result.model_dump_json(indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker-diff", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configure_logging()

    result = materialize_relation_review(
        marker_diff_path=args.marker_diff,
        output_path=args.output,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
