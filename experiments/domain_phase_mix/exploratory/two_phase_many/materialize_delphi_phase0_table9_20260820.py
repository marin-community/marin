# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec[gcs]"]
# ///
"""Materialize phase-0 Table-9 results into fitting-ready tables.

The native evaluator writes one durable JSON record per boundary checkpoint.
This script joins those records to the frozen canonical policy manifest and
emits deterministic wide and long tables. Strict mode is the default: no table
is published unless all 280 policies have exactly one complete result.

Usage::

    uv run python experiments/domain_phase_mix/exploratory/two_phase_many/\
        materialize_delphi_phase0_table9_20260820.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import posixpath
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import fsspec
from marin.evaluation.olmo_base_eval.aggregate import assemble_table9, table9_macro
from marin.evaluation.olmo_base_eval.components import (
    MMLU_BUCKETS,
    leaf_components,
    mmlu_subjects,
    scored_tasks,
    table9_components,
)
from marin.evaluation.olmo_base_eval.metrics import (
    TABLE9_MACRO_ALIAS_KEY,
    TABLE9_MACRO_KEY,
    sc_mmlu_subject_key,
    sc_task_key,
    table9_component_key,
)

EXPERIMENT_ROOT = "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/" "delphi_3e18_phase0_prefix_replay_20260820"
DEFAULT_OUTPUT_DIR = f"{EXPERIMENT_ROOT}/materialized_table9"
EXPECTED_ROWS = 280
EXPECTED_PREFIX_TRAIN_STEPS = 2400
EXPECTED_PREFIX_TRAIN_TOKENS = 1_258_291_200
EXPECTED_PREFIX_HF_STEP = 2399
EXPECTED_PANEL = "delphi_3e18_280row_phase0_prefix_replay"
EXPECTED_TEMPORAL_POSITION = "phase_0_boundary"
EXPECTED_SCALE = "3e18"


@dataclass(frozen=True)
class MaterializedTables:
    """CSV-ready records and coverage metadata for one materialization."""

    fit_matrix: list[dict[str, Any]]
    metrics_wide: list[dict[str, Any]]
    policy_registry: list[dict[str, Any]]
    components_long: list[dict[str, Any]]
    tasks_long: list[dict[str, Any]]
    coverage: dict[str, Any]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: str) -> tuple[Any, bytes]:
    with fsspec.open(path, "rb") as handle:
        payload = handle.read()
    return json.loads(payload), payload


def _glob_urls(pattern: str) -> list[str]:
    fs, fs_pattern = fsspec.core.url_to_fs(pattern)
    return sorted(fs.unstrip_protocol(path) for path in fs.glob(fs_pattern))


def _select_full_manifest(experiment_root: str, expected_rows: int) -> tuple[str, list[dict[str, Any]], str]:
    manifest_paths = _glob_urls(f"{experiment_root.rstrip('/')}/manifest-*/source_run_specs.json")
    candidates: list[tuple[str, list[dict[str, Any]], str]] = []
    observed_sizes: dict[str, int] = {}
    for path in manifest_paths:
        value, payload = _read_json(path)
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError(f"Manifest {path} is not a list of objects")
        observed_sizes[path] = len(value)
        if len(value) == expected_rows:
            candidates.append((path, value, _sha256(payload)))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one {expected_rows}-row manifest, found {len(candidates)}; "
            f"manifest sizes={observed_sizes}"
        )
    return candidates[0]


def _load_full_manifest(path: str, expected_rows: int) -> tuple[str, list[dict[str, Any]], str]:
    value, payload = _read_json(path)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Manifest {path} is not a list of objects")
    if len(value) != expected_rows:
        raise ValueError(f"Expected {expected_rows} source rows in {path}, found {len(value)}")
    return path, value, _sha256(payload)


def _require_finite_mapping(value: Any, *, name: str, expected_keys: set[str]) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    keys = set(value)
    if keys != expected_keys:
        raise ValueError(
            f"{name} schema differs: missing={sorted(expected_keys - keys)}, extra={sorted(keys - expected_keys)}"
        )
    result = {key: float(value[key]) for key in sorted(keys)}
    invalid = [key for key, score in result.items() if not math.isfinite(score)]
    if invalid:
        raise ValueError(f"{name} contains non-finite values: {invalid}")
    return result


def _validate_source_specs(source_specs: Sequence[Mapping[str, Any]], expected_rows: int) -> list[str]:
    if len(source_specs) != expected_rows:
        raise ValueError(f"Expected {expected_rows} source rows, found {len(source_specs)}")
    run_orders = [int(row["run_order"]) for row in source_specs]
    if run_orders != list(range(expected_rows)):
        raise ValueError("Source rows are not in canonical contiguous run_order")

    source_names = [str(row["source_run_name"]) for row in source_specs]
    swarm_names = [str(row["run_name"]) for row in source_specs]
    if len(set(source_names)) != expected_rows or len(set(swarm_names)) != expected_rows:
        raise ValueError("Source manifest contains duplicate source or swarm run names")

    domains = sorted(source_specs[0]["phase_weights"]["phase_0"])
    if len(domains) != 39:
        raise ValueError(f"Expected 39 mixture buckets, found {len(domains)}")
    for row in source_specs:
        phase_weights = row["phase_weights"]
        if set(phase_weights) != {"phase_0", "phase_1"}:
            raise ValueError(f"Unexpected phase names for {row['source_run_name']}")
        for phase in ("phase_0", "phase_1"):
            weights = phase_weights[phase]
            if set(weights) != set(domains):
                raise ValueError(f"Bucket schema differs for {row['source_run_name']} {phase}")
            if not math.isclose(sum(float(weights[d]) for d in domains), 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"Weights do not sum to one for {row['source_run_name']} {phase}")
    return domains


def _validate_result(
    result: Mapping[str, Any],
    *,
    result_path: str,
    source_spec: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    source_name = str(source_spec["source_run_name"])
    swarm_name = str(source_spec["run_name"])
    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Result {result_path} has no provenance object")
    expected_provenance = {
        "panel": EXPECTED_PANEL,
        "scale": EXPECTED_SCALE,
        "temporal_position": EXPECTED_TEMPORAL_POSITION,
        "source_run_name": source_name,
        "swarm_run_name": swarm_name,
        "panel_source": str(source_spec["panel_source"]),
    }
    mismatches = {
        key: (provenance.get(key), expected)
        for key, expected in expected_provenance.items()
        if provenance.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Result provenance mismatch for {source_name}: {mismatches}")
    if result.get("name") != f"t9_boundary_{swarm_name}":
        raise ValueError(f"Result name mismatch for {source_name}: {result.get('name')!r}")
    checkpoint_path = str(result.get("checkpoint_path", ""))
    if not checkpoint_path.endswith(f"/hf/step-{EXPECTED_PREFIX_HF_STEP}"):
        raise ValueError(f"Result {source_name} used the wrong checkpoint: {checkpoint_path}")

    components = _require_finite_mapping(
        result.get("table9_components"),
        name=f"{source_name} table9_components",
        expected_keys=set(table9_components()),
    )
    tasks = _require_finite_mapping(
        result.get("task_bpb"),
        name=f"{source_name} task_bpb",
        expected_keys=set(scored_tasks()),
    )
    derived_components = assemble_table9(
        {task: tasks[task] for task in leaf_components()},
        {subject: tasks[subject] for subject in mmlu_subjects()},
    )
    inconsistent_components = [
        component
        for component in table9_components()
        if not math.isclose(components[component], derived_components[component], rel_tol=0.0, abs_tol=1e-12)
    ]
    if inconsistent_components:
        raise ValueError(f"Table-9 components disagree with raw tasks for {source_name}: {inconsistent_components}")
    reported_macro = float(result.get("table9_macro_bpb"))
    recomputed_macro = table9_macro(components)
    if not math.isclose(reported_macro, recomputed_macro, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Table-9 macro mismatch for {source_name}: reported={reported_macro}, recomputed={recomputed_macro}"
        )
    return components, tasks


def _result_source_name(result: Mapping[str, Any], path: str) -> str:
    provenance = result.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("source_run_name"):
        raise ValueError(f"Result {path} has no provenance.source_run_name")
    return str(provenance["source_run_name"])


def _fit_component_column(component: str) -> str:
    if component in MMLU_BUCKETS:
        return component
    return sc_task_key(component)


def materialize_tables(
    source_specs: Sequence[Mapping[str, Any]],
    result_records: Sequence[Mapping[str, Any]],
    result_paths: Sequence[str],
    *,
    expected_rows: int,
    allow_incomplete: bool,
    source_manifest_path: str,
    source_manifest_sha256: str,
) -> MaterializedTables:
    """Validate and join evaluator records to the canonical source policies."""
    if len(result_records) != len(result_paths):
        raise ValueError("Each result record must have one source path")
    domains = _validate_source_specs(source_specs, expected_rows)
    source_by_name = {str(row["source_run_name"]): row for row in source_specs}

    results_by_name: dict[str, tuple[Mapping[str, Any], str]] = {}
    duplicates: dict[str, list[str]] = {}
    unexpected: list[str] = []
    for result, path in zip(result_records, result_paths, strict=True):
        source_name = _result_source_name(result, path)
        if source_name not in source_by_name:
            unexpected.append(source_name)
            continue
        if source_name in results_by_name:
            duplicates.setdefault(source_name, [results_by_name[source_name][1]]).append(path)
            continue
        results_by_name[source_name] = (result, path)
    if unexpected:
        raise ValueError(f"Results contain unexpected source runs: {sorted(unexpected)}")
    if duplicates:
        raise ValueError(f"Duplicate evaluator results: {duplicates}")

    missing = [str(row["source_run_name"]) for row in source_specs if str(row["source_run_name"]) not in results_by_name]
    if missing and not allow_incomplete:
        raise ValueError(f"Missing {len(missing)}/{expected_rows} evaluator results: {missing[:20]}")

    fit_matrix: list[dict[str, Any]] = []
    metrics_wide: list[dict[str, Any]] = []
    policy_registry: list[dict[str, Any]] = []
    components_long: list[dict[str, Any]] = []
    tasks_long: list[dict[str, Any]] = []
    subject_names = set(mmlu_subjects())

    for source_spec in source_specs:
        source_name = str(source_spec["source_run_name"])
        registry_row: dict[str, Any] = {
            "run_order": int(source_spec["run_order"]),
            "run_name": str(source_spec["run_name"]),
            "source_run_name": source_name,
            "source_experiment": str(source_spec["source_experiment"]),
            "panel_source": str(source_spec["panel_source"]),
            "data_seed": int(source_spec["data_seed"]),
            "trainer_seed": int(source_spec["trainer_seed"]),
            "phase_0_fraction": float(source_spec["phase_0_fraction"]),
            "phase_1_fraction": float(source_spec["phase_1_fraction"]),
            "temporal_position": EXPECTED_TEMPORAL_POSITION,
            "prefix_train_steps": EXPECTED_PREFIX_TRAIN_STEPS,
            "prefix_train_tokens": EXPECTED_PREFIX_TRAIN_TOKENS,
            "prefix_hf_step": EXPECTED_PREFIX_HF_STEP,
        }
        for domain in domains:
            registry_row[f"phase_0_{domain}"] = float(source_spec["phase_weights"]["phase_0"][domain])
            registry_row[f"planned_phase_1_{domain}"] = float(source_spec["phase_weights"]["phase_1"][domain])
        policy_registry.append(registry_row)

        if source_name not in results_by_name:
            continue
        result, result_path = results_by_name[source_name]
        components, tasks = _validate_result(
            result,
            result_path=result_path,
            source_spec=source_spec,
        )
        macro = float(result["table9_macro_bpb"])
        common = {
            "run_order": int(source_spec["run_order"]),
            "run_name": str(source_spec["run_name"]),
            "source_run_name": source_name,
            "source_experiment": str(source_spec["source_experiment"]),
            "panel_source": str(source_spec["panel_source"]),
            "data_seed": int(source_spec["data_seed"]),
            "trainer_seed": int(source_spec["trainer_seed"]),
            "temporal_position": EXPECTED_TEMPORAL_POSITION,
            "prefix_train_steps": EXPECTED_PREFIX_TRAIN_STEPS,
            "prefix_train_tokens": EXPECTED_PREFIX_TRAIN_TOKENS,
            "prefix_hf_step": EXPECTED_PREFIX_HF_STEP,
            "table9_result_path": result_path,
        }
        fit_row = dict(common)
        for domain in domains:
            fit_row[f"phase_0_{domain}"] = float(source_spec["phase_weights"]["phase_0"][domain])
        for component in table9_components():
            fit_row[_fit_component_column(component)] = components[component]
        fit_row["table9_macro_bpb"] = macro
        fit_matrix.append(fit_row)

        metrics_row = dict(fit_row)
        for component in table9_components():
            metrics_row[table9_component_key(component)] = components[component]
        metrics_row[TABLE9_MACRO_KEY] = macro
        metrics_row[TABLE9_MACRO_ALIAS_KEY] = macro
        for task in scored_tasks():
            task_key = sc_mmlu_subject_key(task) if task in subject_names else sc_task_key(task)
            metrics_row[task_key] = tasks[task]
        metrics_wide.append(metrics_row)

        for component in table9_components():
            components_long.append(
                {
                    **common,
                    "component": component,
                    "metric_key": table9_component_key(component),
                    "bpb": components[component],
                }
            )
        for task in scored_tasks():
            task_key = sc_mmlu_subject_key(task) if task in subject_names else sc_task_key(task)
            tasks_long.append({**common, "task": task, "metric_key": task_key, "bpb": tasks[task]})

    coverage = {
        "source_manifest_path": source_manifest_path,
        "source_manifest_sha256": source_manifest_sha256,
        "expected_rows": expected_rows,
        "completed_rows": len(fit_matrix),
        "missing_rows": missing,
        "allow_incomplete": allow_incomplete,
        "bucket_count": len(domains),
        "table9_component_count": len(table9_components()),
        "raw_task_count": len(scored_tasks()),
        "temporal_position": EXPECTED_TEMPORAL_POSITION,
        "prefix_train_steps": EXPECTED_PREFIX_TRAIN_STEPS,
        "prefix_train_tokens": EXPECTED_PREFIX_TRAIN_TOKENS,
        "prefix_hf_step": EXPECTED_PREFIX_HF_STEP,
    }
    return MaterializedTables(
        fit_matrix=fit_matrix,
        metrics_wide=metrics_wide,
        policy_registry=policy_registry,
        components_long=components_long,
        tasks_long=tasks_long,
        coverage=coverage,
    )


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        return b""
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("CSV rows have inconsistent field order")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _write_atomic(output_dir: str, filename: str, payload: bytes) -> str:
    fs, output_path = fsspec.core.url_to_fs(output_dir)
    fs.makedirs(output_path, exist_ok=True)
    destination = posixpath.join(output_path, filename)
    temporary = f"{destination}.tmp-{uuid.uuid4().hex}"
    with fs.open(temporary, "wb") as handle:
        handle.write(payload)
    if fs.exists(destination):
        fs.rm(destination)
    fs.mv(temporary, destination)
    return fs.unstrip_protocol(destination)


def write_tables(tables: MaterializedTables, output_dir: str) -> dict[str, str]:
    """Write all tables, publishing coverage last as the completion marker."""
    payloads = {
        "prefix_fit_matrix.csv": _csv_bytes(tables.fit_matrix),
        "prefix_table9_metrics_wide.csv": _csv_bytes(tables.metrics_wide),
        "prefix_policy_registry.csv": _csv_bytes(tables.policy_registry),
        "table9_components_long.csv": _csv_bytes(tables.components_long),
        "table9_tasks_long.csv": _csv_bytes(tables.tasks_long),
    }
    coverage = dict(tables.coverage)
    coverage["artifacts"] = {
        filename: {"sha256": _sha256(payload), "bytes": len(payload)} for filename, payload in payloads.items()
    }
    written = {filename: _write_atomic(output_dir, filename, payload) for filename, payload in payloads.items()}
    coverage_payload = (json.dumps(coverage, indent=2, sort_keys=True) + "\n").encode()
    written["coverage.json"] = _write_atomic(output_dir, "coverage.json", coverage_payload)
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", default=EXPERIMENT_ROOT)
    parser.add_argument("--source-manifest", help="Explicit full source_run_specs.json; otherwise discover it")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.source_manifest:
        manifest_path, source_specs, manifest_sha256 = _load_full_manifest(args.source_manifest, EXPECTED_ROWS)
    else:
        manifest_path, source_specs, manifest_sha256 = _select_full_manifest(args.experiment_root, EXPECTED_ROWS)
    result_paths = _glob_urls(f"{args.experiment_root.rstrip('/')}/**/olmo_base_eval_table9_results.json")
    result_records = []
    for path in result_paths:
        value, _ = _read_json(path)
        if not isinstance(value, dict):
            raise ValueError(f"Result {path} is not an object")
        result_records.append(value)
    tables = materialize_tables(
        source_specs,
        result_records,
        result_paths,
        expected_rows=EXPECTED_ROWS,
        allow_incomplete=args.allow_incomplete,
        source_manifest_path=manifest_path,
        source_manifest_sha256=manifest_sha256,
    )
    written = write_tables(tables, args.output_dir)
    print(
        json.dumps(
            {
                "completed_rows": tables.coverage["completed_rows"],
                "expected_rows": tables.coverage["expected_rows"],
                "output_dir": args.output_dir,
                "written": written,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
