# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finalize and verify exhaustive fuzzy-dedup A/B adjudication.

The first distributed join binds every full-text pair to its machine decision
and, when required, exactly one semantic decision. The second join proves that
every score marker is covered: drops appear once as labeled members and
canonicals are referenced by at least one labeled member. Full texts are
rehashed before the compact records enter either shuffle.
"""

import argparse
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

from experiments.datakit.scripts.dedup_ab_audit import DedupAuditData
from experiments.datakit.scripts.dedup_ab_machine_labels import DedupMachineLabelsData, decision_for_pair
from experiments.datakit.scripts.dedup_ab_materialize import DedupReviewData, _review_key

LABELS = frozenset({"false_positive", "true_duplicate"})
MACHINE_METHODS = frozenset({"raw_identity", "low_overlap"})


class DedupFinalReviewData(BaseModel):
    """Paths and exact counters for a completely verified adjudication."""

    version: str = "v1"
    audit_path: str
    review_path: str
    machine_labels_path: str
    semantic_decisions_dir: str
    labels_dir: str
    coverage_dir: str
    counters: dict[str, int | float]


def _parquet_records(path: str) -> Iterator[dict[str, Any]]:
    with StoragePath(path).open("rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches(batch_size=16):
            yield from batch.to_pylist()


def _variant_occurrence_key(variant: str, source_main_dir: str, basename: str, doc_id: str) -> str:
    return json.dumps([variant, source_main_dir, basename, doc_id], separators=(",", ":"))


def _compact_verified_pair(pair: dict[str, Any]) -> dict[str, Any]:
    expected = decision_for_pair(pair)
    return {
        **expected,
        "expected_label": expected["label"],
        "expected_method": expected["method"],
        "expected_basis": expected["basis"],
    }


def _adjudication_input_records(entry: dict[str, str]) -> Iterator[dict[str, str]]:
    for row_index, record in enumerate(_parquet_records(entry["path"])):
        if entry["kind"] == "pair":
            payload = {
                **_compact_verified_pair(record),
                "pair_path": entry["path"],
                "pair_row_index": row_index,
            }
        else:
            payload = record
        review_key = payload["review_key"]
        yield {
            "review_key": review_key,
            "kind": entry["kind"],
            "payload_json": json.dumps(payload, separators=(",", ":"), sort_keys=True),
        }


def _one_record(by_kind: dict[str, list[dict[str, Any]]], kind: str, review_key: str) -> dict[str, Any]:
    records = by_kind.get(kind, [])
    if len(records) != 1:
        raise AssertionError(f"Expected one {kind} record for {review_key}, found {len(records)}")
    return records[0]


def _matching_identity_fields(left: dict[str, Any], right: dict[str, Any], review_key: str) -> None:
    fields = (
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
        "pair_path",
        "pair_row_index",
    )
    mismatches = [field for field in fields if left.get(field) != right.get(field)]
    if mismatches:
        raise AssertionError(f"Identity mismatch for {review_key}: {mismatches}")


def final_decision(review_key: str, records: Iterator[dict[str, str]]) -> dict[str, Any]:
    """Return one verified final decision for a full-text pair."""
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        kind = record["kind"]
        by_kind.setdefault(kind, []).append(json.loads(record["payload_json"]))
    unknown_kinds = by_kind.keys() - {"pair", "machine", "semantic"}
    if unknown_kinds:
        raise AssertionError(f"Unknown adjudication inputs for {review_key}: {sorted(unknown_kinds)}")

    pair = _one_record(by_kind, "pair", review_key)
    machine = _one_record(by_kind, "machine", review_key)
    _matching_identity_fields(pair, machine, review_key)
    for field in ("label", "method", "basis", "needs_semantic_review"):
        if machine.get(field) != pair.get(f"expected_{field}", pair.get(field)):
            raise AssertionError(f"Machine {field} differs from verified full-text evidence for {review_key}")

    semantic_records = by_kind.get("semantic", [])
    if machine["needs_semantic_review"]:
        if len(semantic_records) != 1:
            raise AssertionError(f"Expected one semantic record for {review_key}, found {len(semantic_records)}")
        decision = semantic_records[0]
        _matching_identity_fields(pair, decision, review_key)
        if decision.get("method") != "semantic":
            raise AssertionError(f"Semantic decision has method {decision.get('method')!r} for {review_key}")
        if decision.get("label") not in LABELS:
            raise AssertionError(f"Semantic decision has invalid label {decision.get('label')!r} for {review_key}")
        if not str(decision.get("basis", "")).strip():
            raise AssertionError(f"Semantic decision has no basis for {review_key}")
    else:
        if semantic_records:
            raise AssertionError(f"Unexpected semantic record for machine-resolved pair {review_key}")
        decision = machine
        if decision["method"] not in MACHINE_METHODS or decision["label"] not in LABELS:
            raise AssertionError(f"Invalid machine decision for {review_key}")

    variant = pair["variant"]
    label = decision["label"]
    method = decision["method"]
    counters.pipeline.update_counter("finalize/labels/pairs", 1)
    counters.pipeline.update_counter(f"finalize/labels/{variant}/pairs", 1)
    counters.pipeline.update_counter(f"finalize/labels/{variant}/{label}", 1)
    counters.pipeline.update_counter(f"finalize/labels/{variant}/{method}", 1)
    if machine["needs_semantic_review"]:
        counters.pipeline.update_counter("finalize/labels/semantic_required", 1)

    return {
        "review_key": review_key,
        "variant": variant,
        "member_source_main_dir": pair["member_source_main_dir"],
        "member_basename": pair["member_basename"],
        "member_id": pair["member_id"],
        "canonical_source_main_dir": pair["canonical_source_main_dir"],
        "canonical_basename": pair["canonical_basename"],
        "canonical_id": pair["canonical_id"],
        "raw_sha256": pair["raw_sha256"],
        "canonical_raw_sha256": pair["canonical_raw_sha256"],
        "label": label,
        "method": method,
        "basis": decision["basis"],
        "member_occurrence_key": _variant_occurrence_key(
            variant,
            pair["member_source_main_dir"],
            pair["member_basename"],
            pair["member_id"],
        ),
        "canonical_occurrence_key": _variant_occurrence_key(
            variant,
            pair["canonical_source_main_dir"],
            pair["canonical_basename"],
            pair["canonical_id"],
        ),
    }


def _score_review_key(score: dict[str, Any]) -> str:
    return _review_key(score) if score["role"] == "drop" else ""


def _coverage_input_records(entry: dict[str, str]) -> Iterator[dict[str, str]]:
    for record in _parquet_records(entry["path"]):
        if entry["kind"] == "score":
            key = _variant_occurrence_key(
                record["variant"],
                record["source_main_dir"],
                record["basename"],
                record["id"],
            )
            yield {
                "occurrence_key": key,
                "kind": "score",
                "review_key": _score_review_key(record),
                "role": record["role"],
            }
            continue

        yield {
            "occurrence_key": record["member_occurrence_key"],
            "kind": "member",
            "review_key": record["review_key"],
            "role": "",
        }
        yield {
            "occurrence_key": record["canonical_occurrence_key"],
            "kind": "canonical",
            "review_key": record["review_key"],
            "role": "",
        }


def validate_occurrence_coverage(occurrence_key: str, records: Iterator[dict[str, str]]) -> dict[str, Any]:
    """Validate exact label coverage for one scored marker occurrence."""
    scores: list[dict[str, str]] = []
    member_labels: list[str] = []
    canonical_labels: list[str] = []
    for record in records:
        if record["kind"] == "score":
            scores.append(record)
        elif record["kind"] == "member":
            member_labels.append(record["review_key"])
        elif record["kind"] == "canonical":
            canonical_labels.append(record["review_key"])
        else:
            raise AssertionError(f"Unknown occurrence input {record['kind']!r} for {occurrence_key}")

    if len(scores) != 1:
        raise AssertionError(f"Expected one score for {occurrence_key}, found {len(scores)}")
    score = scores[0]
    if score["role"] == "drop":
        if member_labels != [score["review_key"]] or canonical_labels:
            raise AssertionError(
                f"Drop coverage mismatch for {occurrence_key}: "
                f"members={member_labels}, canonicals={canonical_labels}"
            )
    elif score["role"] == "canonical":
        if member_labels or not canonical_labels:
            raise AssertionError(
                f"Canonical coverage mismatch for {occurrence_key}: "
                f"members={member_labels}, canonicals={canonical_labels}"
            )
    else:
        raise AssertionError(f"Unknown score role {score['role']!r} for {occurrence_key}")

    variant = json.loads(occurrence_key)[0]
    counters.pipeline.update_counter("finalize/coverage/markers", 1)
    counters.pipeline.update_counter(f"finalize/coverage/{variant}/markers", 1)
    counters.pipeline.update_counter(f"finalize/coverage/{variant}/{score['role']}", 1)
    counters.pipeline.update_counter("finalize/coverage/canonical_references", len(canonical_labels))
    return {
        "occurrence_key": occurrence_key,
        "variant": variant,
        "role": score["role"],
        "label_references": len(member_labels) + len(canonical_labels),
    }


def _paths(directory: str, kind: str, *, required: bool = True) -> list[dict[str, str]]:
    paths = sorted(str(path) for path in StoragePath(f"{directory.rstrip('/')}/*.parquet").glob())
    if required and not paths:
        raise FileNotFoundError(f"No {kind} Parquet files under {directory}")
    return [{"kind": kind, "path": path} for path in paths]


def _expected_counts(audit: DedupAuditData, variant: str) -> tuple[int, int]:
    markers = int(audit.counters.get(f"audit/markers/{variant}", 0))
    drops = int(audit.counters.get(f"audit/drops/{variant}", 0))
    return markers, drops


def _validate_counters(
    audit: DedupAuditData,
    review: DedupReviewData,
    machine: DedupMachineLabelsData,
    combined: dict[str, int | float],
) -> None:
    expected_pairs = int(review.counters.get("audit/materialize/pairs", 0))
    machine_pairs = int(machine.counters.get("machine_labels/pairs", 0))
    finalized_pairs = int(combined.get("finalize/labels/pairs", 0))
    if machine_pairs != expected_pairs or finalized_pairs != expected_pairs:
        raise AssertionError(
            f"Final pair coverage mismatch: machine={machine_pairs}, finalized={finalized_pairs}, "
            f"expected={expected_pairs}"
        )

    expected_semantic = sum(
        int(machine.counters.get(f"machine_labels/{variant}/semantic", 0)) for variant in ("baseline", "treatment")
    )
    actual_semantic = int(combined.get("finalize/labels/semantic_required", 0))
    if actual_semantic != expected_semantic:
        raise AssertionError(f"Semantic coverage mismatch: {actual_semantic}/{expected_semantic}")

    expected_markers = 0
    for variant in ("baseline", "treatment"):
        markers, drops = _expected_counts(audit, variant)
        expected_markers += markers
        actual_markers = int(combined.get(f"finalize/coverage/{variant}/markers", 0))
        actual_drops = int(combined.get(f"finalize/coverage/{variant}/drop", 0))
        actual_pairs = int(combined.get(f"finalize/labels/{variant}/pairs", 0))
        if actual_markers != markers or actual_drops != drops or actual_pairs != drops:
            raise AssertionError(
                f"{variant} final coverage mismatch: markers={actual_markers}/{markers}, "
                f"drops={actual_drops}/{drops}, pairs={actual_pairs}/{drops}"
            )

    actual_markers = int(combined.get("finalize/coverage/markers", 0))
    canonical_references = int(combined.get("finalize/coverage/canonical_references", 0))
    if actual_markers != expected_markers or canonical_references != expected_pairs:
        raise AssertionError(
            f"Global final coverage mismatch: markers={actual_markers}/{expected_markers}, "
            f"canonical references={canonical_references}/{expected_pairs}"
        )


def finalize(
    *,
    audit_path: str,
    review_path: str,
    machine_labels_path: str,
    semantic_decisions_dir: str,
    output_path: str,
    max_workers: int,
) -> DedupFinalReviewData:
    """Verify every pair decision and every scored marker using distributed joins."""
    audit = DedupAuditData.model_validate_json(StoragePath(audit_path).read_text())
    review = DedupReviewData.model_validate_json(StoragePath(review_path).read_text())
    machine = DedupMachineLabelsData.model_validate_json(StoragePath(machine_labels_path).read_text())
    if review.scores_dir != audit.scores_dir:
        raise ValueError(f"Review scores differ from audit: {review.scores_dir} != {audit.scores_dir}")
    if machine.review_path != review_path or machine.pairs_dir != review.pairs_dir:
        raise ValueError("Machine-label metadata does not identify the supplied review artifact")

    labels_dir = f"{output_path.rstrip('/')}/labels"
    label_entries = [
        *_paths(review.pairs_dir, "pair"),
        *_paths(machine.decisions_dir, "machine"),
        *_paths(semantic_decisions_dir, "semantic", required=False),
    ]
    resources = ResourceConfig(cpu=2, ram="24g", disk="20g", preemptible=False)
    coordinator = ResourceConfig(cpu=4, ram="16g", disk="20g", preemptible=False)
    label_context = ZephyrContext(
        name="dedup-ab-finalize-labels",
        max_workers=max_workers,
        resources=resources,
        coordinator_resources=coordinator,
    )
    label_pipeline = (
        Dataset.from_list(label_entries)
        .flat_map(_adjudication_input_records)
        .group_by(
            key=lambda record: record["review_key"],
            sort_by=lambda record: record["kind"],
            reducer=final_decision,
            num_output_shards=max_workers,
        )
        .write_parquet(f"{labels_dir}/part-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    label_outcome = label_context.execute(label_pipeline, verbose=True)

    coverage_dir = f"{output_path.rstrip('/')}/coverage"
    coverage_entries = [
        *_paths(audit.scores_dir, "score"),
        *_paths(labels_dir, "label"),
    ]
    coverage_context = ZephyrContext(
        name="dedup-ab-finalize-coverage",
        max_workers=max_workers,
        resources=resources,
        coordinator_resources=coordinator,
    )
    coverage_pipeline = (
        Dataset.from_list(coverage_entries)
        .flat_map(_coverage_input_records)
        .group_by(
            key=lambda record: record["occurrence_key"],
            sort_by=lambda record: f"{record['kind']}|{record['review_key']}",
            reducer=validate_occurrence_coverage,
            num_output_shards=max_workers,
        )
        .write_parquet(f"{coverage_dir}/part-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    coverage_outcome = coverage_context.execute(coverage_pipeline, verbose=True)

    combined = {
        key: value
        for outcome in (label_outcome, coverage_outcome)
        for key, value in outcome.counters.items()
        if key.startswith("finalize/")
    }
    _validate_counters(audit, review, machine, combined)
    return DedupFinalReviewData(
        audit_path=audit_path,
        review_path=review_path,
        machine_labels_path=machine_labels_path,
        semantic_decisions_dir=semantic_decisions_dir,
        labels_dir=labels_dir,
        coverage_dir=coverage_dir,
        counters=combined,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--machine-labels", required=True)
    parser.add_argument("--semantic-decisions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-workers", type=int, default=128)
    args = parser.parse_args()
    configure_logging()

    result = finalize(
        audit_path=args.audit,
        review_path=args.review,
        machine_labels_path=args.machine_labels,
        semantic_decisions_dir=args.semantic_decisions,
        output_path=args.output,
        max_workers=args.max_workers,
    )
    StoragePath(f"{args.output.rstrip('/')}/final-review.json").write_text(
        json.dumps(result.model_dump(), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
