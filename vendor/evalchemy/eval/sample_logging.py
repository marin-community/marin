"""Canonical per-example records written to FineStore.

The evaluators use a variety of internal result shapes.  This module is the
single normalization boundary: it gives custom benchmarks and lm-eval-native
tasks the same record envelope before FineStore stores them.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from eval.completion_response import CompletionText, FailedGeneration
from eval.contracts.sample_manifest import SampleCoverageError, SampleEntry, SampleManifest
from eval.contracts.sample_results import SampleMetricsError, validate_sample_metrics
from eval.lm_eval_tasks.drop.utils import DropAnswer

SAMPLE_SCHEMA_VERSION = 1
"""Version of the stable sample record envelope."""
DEFAULT_FILTER_NAME = "none"
_IFEVAL_TASKS = frozenset({"ifeval", "ifeval_ca", "ifeval_es", "leaderboard_ifeval"})
_IFEVAL_INSTRUCTION_METRICS = {
    "inst_level_strict_acc": "strict_instruction_pass",
    "inst_level_loose_acc": "loose_instruction_pass",
}


def is_scored_result(result: Any) -> bool:
    """Return whether a task completed scoring rather than returning an error."""
    return isinstance(result, Mapping) and bool(result) and "error" not in result


def canonicalize_samples(
    task_name: str,
    samples: Sequence[Mapping[str, Any]],
    sample_manifest: SampleManifest | None = None,
) -> list[dict[str, Any]]:
    """Add the stable envelope to lm-eval-compatible sample records.

    With a sample manifest, complete lm-eval filter cohorts become one record
    whose ``filter_variants`` retain every response and metric. The envelope
    makes task identity and schema version explicit, while filling fields that
    custom benchmark adapters must provide for FineStore sample rows.

    Raises:
        SampleMetricsError: If a record reached this boundary without the
            per-sample metrics its grader owes every scored sample.
    """
    manifest_entries = ()
    if sample_manifest is not None and sample_manifest.expected_sample_count:
        sample_entries, unit_entries = manifest_entries_for_task(task_name, sample_manifest)
        samples = _coalesce_lm_eval_filter_variants(samples, len(sample_entries))
        manifest_entries = _entries_for_count(task_name, len(samples), sample_entries, unit_entries)

    canonical = [
        canonicalize_sample(task_name, sample, index, manifest_entries[index] if manifest_entries else None)
        for index, sample in enumerate(samples)
    ]
    validate_sample_metrics(task_name, canonical)
    return canonical


def canonicalize_sample(
    task_name: str, sample: Mapping[str, Any], index: int, entry: SampleEntry | None = None
) -> dict[str, Any]:
    """Add the stable envelope and optional manifest identity to one scored record."""
    record = dict(sample)
    if task_name.lower() in _IFEVAL_TASKS:
        _normalize_ifeval_instruction_metrics(record)
    record["schema_version"] = SAMPLE_SCHEMA_VERSION
    record["task_name"] = task_name
    record.setdefault("doc_id", index)
    record.setdefault("doc", {})
    record.setdefault("target", "")
    record.setdefault("arguments", [])
    record.setdefault("resps", [])
    record.setdefault("filtered_resps", [])
    record.setdefault("filter", DEFAULT_FILTER_NAME)
    if entry is not None:
        record["sample_id"] = entry.sample_id
        record["source_id"] = entry.source_id
        record["sample_ordinal"] = entry.ordinal
        record["sample_namespace"] = entry.namespace
        record["sample_shard"] = entry.shard
        record["sample_repeat"] = entry.repeat
    completion_artifacts = _completion_artifacts(record["resps"])
    if completion_artifacts is not None:
        record["completion_responses"] = completion_artifacts
        failure_category = _completion_failure_category(completion_artifacts)
        if failure_category is not None:
            record["failure_category"] = failure_category
    drop_extractions = _drop_extractions(record["filtered_resps"])
    if drop_extractions is not None:
        record["drop_extractions"] = drop_extractions
    return record


def manifest_entries_for_task(
    task_name: str, sample_manifest: SampleManifest
) -> tuple[tuple[SampleEntry, ...], tuple[SampleEntry, ...]]:
    """Return sample and request-unit identities for one task namespace."""
    sample_entries = sample_manifest.sample_entries(namespace=task_name)
    unit_entries = sample_manifest.unit_entries(namespace=task_name)
    if not sample_entries and sample_manifest.task_name == task_name:
        sample_entries = sample_manifest.sample_entries()
        unit_entries = sample_manifest.unit_entries()
    return sample_entries, unit_entries


def _entries_for_count(
    task_name: str,
    count: int,
    sample_entries: tuple[SampleEntry, ...],
    unit_entries: tuple[SampleEntry, ...],
) -> tuple[SampleEntry, ...]:
    entries = sample_entries
    if count != len(entries) and count == len(unit_entries):
        entries = unit_entries
    if count != len(entries):
        raise SampleCoverageError(
            f"{task_name}: sample logger received {count} records but the manifest "
            f"contains {len(entries)} samples"
        )
    return entries


def _normalize_ifeval_instruction_metrics(record: dict[str, Any]) -> None:
    """Keep lm-eval's instruction verdicts while making each metric scalar."""
    for metric, detail in _IFEVAL_INSTRUCTION_METRICS.items():
        values = record.get(metric)
        if not isinstance(values, list):
            continue
        if not values or any(not isinstance(value, bool) for value in values):
            raise SampleMetricsError(f"IFEval per-instruction metric {metric!r} must be a non-empty list of booleans")
        record[detail] = values
        record[metric] = sum(values) / len(values)


def _coalesce_lm_eval_filter_variants(
    samples: Sequence[Mapping[str, Any]], expected_sample_count: int
) -> list[Mapping[str, Any]]:
    """Coalesce complete lm-eval filter cohorts, leaving other record sets unchanged."""
    records = list(samples)
    if len(records) <= expected_sample_count:
        return records

    by_doc_id: dict[str | int, list[Mapping[str, Any]]] = {}
    for record in records:
        doc_id = record.get("doc_id")
        filter_name = record.get("filter")
        if not isinstance(doc_id, (str, int)) or not isinstance(filter_name, str) or filter_name == DEFAULT_FILTER_NAME:
            return records
        by_doc_id.setdefault(doc_id, []).append(record)

    if len(by_doc_id) != expected_sample_count:
        return records

    expected_filters = {record["filter"] for record in next(iter(by_doc_id.values()))}
    if any(
        len(variants) != len(expected_filters) or {record["filter"] for record in variants} != expected_filters
        for variants in by_doc_id.values()
    ):
        return records

    coalesced: list[Mapping[str, Any]] = []
    for variants in by_doc_id.values():
        primary = dict(variants[0])
        primary["filter_variants"] = [
            {
                "filter": variant["filter"],
                "filtered_resps": variant.get("filtered_resps", []),
                "metrics": {str(name): variant.get(name) for name in variant.get("metrics", [])},
            }
            for variant in variants
        ]
        coalesced.append(primary)
    return coalesced


def _completion_artifacts(value: Any) -> Any | None:
    """Mirror scorer-response nesting with audit data for normalized chat output."""
    if isinstance(value, (CompletionText, FailedGeneration)):
        return value.artifact()
    if isinstance(value, Sequence) and not isinstance(value, str):
        artifacts = [_completion_artifacts(item) for item in value]
        return artifacts if any(artifact is not None for artifact in artifacts) else None
    return None


def _completion_failure_category(value: Any) -> str | None:
    if isinstance(value, Mapping):
        category = value.get("failure_category")
        return category if isinstance(category, str) else None
    if isinstance(value, Sequence) and not isinstance(value, str):
        for item in value:
            if category := _completion_failure_category(item):
                return category
    return None


def _drop_extractions(value: Any) -> Any | None:
    """Mirror filtered-response nesting with DROP extraction classifications."""
    if isinstance(value, DropAnswer):
        return {"classification": value.classification}
    if isinstance(value, Sequence) and not isinstance(value, str):
        extractions = [_drop_extractions(item) for item in value]
        return extractions if any(extraction is not None for extraction in extractions) else None
    return None


def without_embedded_samples(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return aggregate metrics without the custom benchmark's raw examples."""
    aggregate = dict(result)
    aggregate.pop("examples", None)
    return aggregate
