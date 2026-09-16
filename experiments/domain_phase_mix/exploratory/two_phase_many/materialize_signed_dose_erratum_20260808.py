# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Correct stale metric summaries without changing the frozen SUR-073 evaluation.

The original v2 materializer preferred finite W&B summaries over exact persisted
final-step metrics. Four summaries in the complete 60M panel are stale. This
script creates a versioned erratum artifact using one source rule uniformly:
checkpoint ``eval_metrics.jsonl`` for Uncheatable and native Table-9 result JSON
as a per-row identity and value audit for the byte-preserved Table-9 targets. It
then calls the unchanged frozen evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import gcsfs
import pandas as pd
import wandb

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    evaluate_intervention_identified_signed_dose_potential_20260731 as evaluator,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "intervention_identified_signed_dose_potential_20260731"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "intervention_identified_signed_dose_potential_20260808_erratum"
SOURCE_DIAGNOSIS_NAME = "source_diagnosis.json"
PREOUTCOME_SUPERSESSION_NAME = "preoutcome_protocol_supersession_v2.json"
ERRATUM_PROTOCOL_NAME = "erratum_protocol_v2.json"
SUPERSEDED_ERRATUM_PROTOCOL_NAME = "erratum_protocol.json"
SUPERSEDED_ERRATUM_PROTOCOL_SHA256 = "f8101d138014b993e60a9dbaf28f90f3ffc4e4c84e3a80ecd961399629cd7fe2"
SOURCE_DIAGNOSIS_SHA256 = "4265d696e53a3b97417f471e355b13ea4a3bcd1c7fe33139c582447b9d794dcd"
EXPECTED_ROWS = 277
EXPECTED_STEP = 4576
EXPECTED_TABLE9_COMPONENTS = 51
EXPECTED_TABLE9_REPRESENTATION_DIFFERENCES = 36
VALUE_TOLERANCE = 1e-10
EXPECTED_STALE_RUNS = {
    "p240_d34_m0",
    "p247_d34_m32",
    "p251_d35_m2",
    "p255_d35_m32",
}
TABLE9_GLOB = "marin-us-east5/evaluation/olmo_base_eval_table9/t9_p*/olmo_base_eval_table9_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "materialize", "evaluate", "report"), required=True)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel-dir", type=Path, default=evaluator.PANEL_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=180)
    parser.add_argument("--expect-erratum-sha256")
    parser.add_argument("--expect-materialization-sha256")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_if_absent_or_equal(path: Path, content: str) -> None:
    if path.exists():
        if path.read_bytes() != content.encode():
            raise ValueError(f"Frozen erratum artifact differs: {path}")
        return
    path.write_bytes(content.encode())


def erratum_payload(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    source_observations = source_dir / "observations_60m.csv"
    source_protocol = source_dir / "protocol.json"
    source_evaluation_protocol = source_dir / "evaluation_protocol.json"
    source_selected = source_dir / "selected_60m.json"
    source_diagnosis = output_dir / SOURCE_DIAGNOSIS_NAME
    preoutcome_supersession = output_dir / PREOUTCOME_SUPERSESSION_NAME
    superseded_protocol = output_dir / SUPERSEDED_ERRATUM_PROTOCOL_NAME
    for path in (
        source_observations,
        source_protocol,
        source_evaluation_protocol,
        source_selected,
        source_diagnosis,
        preoutcome_supersession,
        superseded_protocol,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256(superseded_protocol) != SUPERSEDED_ERRATUM_PROTOCOL_SHA256:
        raise ValueError("Superseded v1 erratum protocol differs from its external Fieldbook anchor")
    if sha256(source_diagnosis) != SOURCE_DIAGNOSIS_SHA256:
        raise ValueError("Restored source diagnosis differs from its v1-anchored SHA-256")

    payload: dict[str, Any] = {
        "version": "signed-dose-potential-source-erratum-v2",
        "candidate_id": evaluator.model.CANDIDATE_ID,
        "scope": "metric-source correction only; no scientific procedure changes",
        "registered_output_dir": str(output_dir.resolve()),
        "source_artifacts": {
            "observations_60m_sha256": sha256(source_observations),
            "protocol_sha256": sha256(source_protocol),
            "evaluation_protocol_sha256": sha256(source_evaluation_protocol),
            "selected_60m_sha256": sha256(source_selected),
            "frozen_evaluator_sha256": sha256(Path(evaluator.__file__)),
            "erratum_materializer_sha256": sha256(Path(__file__)),
            "source_diagnosis_sha256": sha256(source_diagnosis),
            "preoutcome_supersession_sha256": sha256(preoutcome_supersession),
            "superseded_erratum_protocol_sha256": sha256(superseded_protocol),
        },
        "authoritative_sources": {
            "uncheatable": f"exact persisted step-{EXPECTED_STEP} checkpoints/eval_metrics.jsonl",
            "table9": (
                "frozen v2 values retained after exact per-row checkpoint and provenance verification against "
                "native GCS olmo_base_eval_table9_results.json"
            ),
        },
        "uniformity": {
            "uncheatable": (
                f"replace all {EXPECTED_ROWS} rows from the same exact-final-step source rule; require exactly "
                "the four preregistered substantive corrections above tolerance and report any additional "
                "last-bit representation differences"
            ),
            "table9": (
                f"retain all {EXPECTED_ROWS} frozen values after native-result verification within "
                f"absolute tolerance {VALUE_TOLERANCE}"
            ),
            "table9_components_per_row": EXPECTED_TABLE9_COMPONENTS,
            "value_tolerance": VALUE_TOLERANCE,
        },
        "known_trigger": {
            "stale_finite_wandb_summaries": sorted(EXPECTED_STALE_RUNS),
            "source_census": "271 matching finite summaries + 2 pre-existing persisted values + 4 stale summaries",
            "table9_gcs_audit": "277 unique run names, 277 artifacts, 51 components each, zero source disagreement",
            "table9_last_bit_representation_differences": EXPECTED_TABLE9_REPRESENTATION_DIFFERENCES,
        },
        "preoutcome_supersession": {
            "record_file": PREOUTCOME_SUPERSESSION_NAME,
            "source_diagnosis_sha256_v1_anchored": SOURCE_DIAGNOSIS_SHA256,
            "superseded_protocol_file": SUPERSEDED_ERRATUM_PROTOCOL_NAME,
            "superseded_protocol_sha256": SUPERSEDED_ERRATUM_PROTOCOL_SHA256,
            "reason": (
                "The first materialization stopped before any corrected artifact write or model evaluation because "
                "36 native Table-9 JSON floats differed from their frozen CSV representations only in the last bit. "
                "V2 preserves the frozen target bytes and treats native GCS as a per-row identity and value audit."
            ),
            "scientific_outcomes_seen": False,
        },
        "frozen_unchanged": [
            "candidate equations",
            "shape and ridge grids",
            "fold assignments",
            "bootstrap procedure and seeds",
            "x32 holdout",
            "acceptance thresholds",
            "selection and optimization procedure",
        ],
        "preservation": (
            "the 20260731 v2 directory remains immutable and is superseded only for source-corrected results"
        ),
        "binding_outcome_rule": (
            "corrected gate result is binding; failure does not permit another directory, threshold change, "
            "or candidate-grid extension"
        ),
        "anticipated_direction": (
            "removing two gross shared gating outliers is expected to lower candidate and linear RMSE; the prior "
            "relative-RMSE gate already passed at 0.763, while curvature activity and fold-mode fraction may worsen"
        ),
    }
    payload["erratum_protocol_sha256"] = canonical_hash(payload)
    return payload


def prepare(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("protocol.json", "evaluation_protocol.json"):
        write_if_absent_or_equal(output_dir / name, (source_dir / name).read_text())
    payload = erratum_payload(source_dir, output_dir)
    write_if_absent_or_equal(output_dir / ERRATUM_PROTOCOL_NAME, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def verify_erratum(source_dir: Path, output_dir: Path, expected_file_sha256: str) -> dict[str, Any]:
    path = output_dir / ERRATUM_PROTOCOL_NAME
    if not path.exists():
        raise FileNotFoundError(f"Freeze the erratum before reading corrected outcomes: {path}")
    if sha256(path) != expected_file_sha256:
        raise ValueError("Erratum protocol differs from the externally anchored Fieldbook SHA-256")
    frozen = json.loads(path.read_text())
    current = erratum_payload(source_dir, output_dir)
    if frozen != current:
        raise ValueError("Erratum source, rules, or frozen evaluator changed after preparation")
    for name in ("protocol.json", "evaluation_protocol.json"):
        if (output_dir / name).read_bytes() != (source_dir / name).read_bytes():
            raise ValueError(f"Copied frozen artifact differs: {name}")
    return frozen


def persisted_metric_history(path: str) -> list[tuple[int, float]]:
    filesystem, inner_path = evaluator.fsspec.core.url_to_fs(path)
    values = []
    with filesystem.open(inner_path, "rt") as handle:
        for line in handle:
            payload = json.loads(line)
            try:
                value = float(payload.get(evaluator.UNCHEATABLE_METRIC, math.nan))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append((int(payload.get("step", -1)), value))
    return values


def exact_training_values(
    observations: pd.DataFrame,
    manifest: pd.DataFrame,
    timeout: int,
) -> tuple[dict[str, tuple[float, str]], list[dict[str, Any]]]:
    api = wandb.Api(timeout=timeout)
    runs = list(
        api.runs(
            evaluator.TRAIN_PROJECT,
            filters=evaluator.SCALE_CONFIGS["60m"]["train_filter"],
            per_page=500,
        )
    )
    observation_by_name = observations.set_index("run_name", verify_integrity=True)
    outcomes: dict[str, tuple[float, str]] = {}
    retry_audit = []
    for spec in manifest.itertuples(index=False):
        run_name = str(spec.run_name)
        expected_step = int(spec.expected_checkpoint_step)
        candidates = []
        for run in runs:
            if not evaluator.run_name_matches("60m", run, run_name):
                continue
            persisted = evaluator.persisted_training_metric(run, evaluator.UNCHEATABLE_METRIC, expected_step)
            if persisted is not None:
                candidates.append((run, *persisted))
        if not candidates:
            raise ValueError(f"{run_name}: no persisted exact-final-step outcome")
        values = [value for _, value, _ in candidates]
        if max(values) - min(values) > VALUE_TOLERANCE:
            raise ValueError(f"{run_name}: persisted retry outcomes disagree: {values}")

        run, value, source_path = max(candidates, key=lambda item: str(item[0].created_at))
        frozen_run_id = str(observation_by_name.loc[run_name, "training_wandb_id"])
        if str(run.id) != frozen_run_id:
            raise ValueError(f"{run_name}: selected run changed from {frozen_run_id} to {run.id}")
        outcomes[run_name] = (value, source_path)

        old_value = float(observation_by_name.loc[run_name, "uncheatable_bpb"])
        old_source = str(observation_by_name.loc[run_name, "training_metric_source"])
        summary_value = evaluator.finite_summary(run, evaluator.UNCHEATABLE_METRIC)
        category = "preexisting_persisted"
        matching_earlier_steps: list[int] = []
        if summary_value is not None:
            category = "matching_finite_summary"
            if abs(summary_value - value) > VALUE_TOLERANCE:
                category = "stale_finite_summary"
                history = persisted_metric_history(source_path)
                matching_earlier_steps = [
                    step
                    for step, history_value in history
                    if step < expected_step and abs(history_value - summary_value) <= VALUE_TOLERANCE
                ]
                if len(matching_earlier_steps) != 1:
                    raise ValueError(
                        f"{run_name}: stale summary does not identify exactly one earlier persisted step: "
                        f"{matching_earlier_steps}"
                    )
        if old_source == "wandb_summary" and summary_value is None:
            raise ValueError(f"{run_name}: v2 records a W&B summary that is now absent or non-finite")
        recorded_value = summary_value if old_source == "wandb_summary" else value
        if recorded_value is None or abs(old_value - recorded_value) > VALUE_TOLERANCE:
            raise ValueError(f"{run_name}: v2 value does not match its recorded source")

        retry_audit.append(
            {
                "run_name": run_name,
                "selected_training_wandb_id": str(run.id),
                "frozen_training_wandb_id": frozen_run_id,
                "selected_created_at": str(run.created_at),
                "candidate_count": len(candidates),
                "candidate_ids": json.dumps([str(candidate.id) for candidate, _, _ in candidates]),
                "candidate_values": json.dumps(values),
                "source_category": category,
                "old_source": old_source,
                "old_value": old_value,
                "finite_summary_value": summary_value,
                "exact_final_value": value,
                "matching_earlier_steps": json.dumps(matching_earlier_steps),
                "exact_source_path": source_path,
            }
        )
    return outcomes, retry_audit


def exact_table9_values(
    observations: pd.DataFrame,
    manifest: pd.DataFrame,
    training: dict[str, tuple[float, str]],
) -> dict[str, tuple[float, str]]:
    filesystem = gcsfs.GCSFileSystem()
    observation_by_name = observations.set_index("run_name", verify_integrity=True)
    manifest_by_name = manifest.set_index("run_name", verify_integrity=True)
    candidates: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for path in filesystem.glob(TABLE9_GLOB):
        with filesystem.open(path, "rt") as handle:
            payload = json.load(handle)
        provenance = dict(payload.get("provenance") or {})
        if provenance.get("scale") != "60m" or provenance.get("stage") != "full":
            continue
        components = payload.get("table9_components") or []
        if len(components) != EXPECTED_TABLE9_COMPONENTS:
            raise ValueError(f"{path}: expected {EXPECTED_TABLE9_COMPONENTS} Table-9 components")
        value = float(payload["table9_macro_bpb"])
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite Table-9 value")
        run_name = str(provenance.get("run_name") or "")
        if run_name not in observation_by_name.index:
            raise ValueError(f"{path}: unknown frozen run name {run_name!r}")
        exact_training_path = training[run_name][1]
        suffix = "/checkpoints/eval_metrics.jsonl"
        if not exact_training_path.endswith(suffix):
            raise ValueError(f"{run_name}: unexpected persisted training-metric path {exact_training_path!r}")
        expected_checkpoint_path = exact_training_path.removesuffix(suffix) + f"/hf/step-{EXPECTED_STEP}"
        if payload.get("checkpoint_path") != expected_checkpoint_path:
            raise ValueError(
                f"{run_name}: native Table-9 checkpoint differs from the frozen training identity: "
                f"{payload.get('checkpoint_path')!r} != {expected_checkpoint_path!r}"
            )
        expected_provenance = {
            "point_id": str(manifest_by_name.loc[run_name, "point_id"]),
            "seed_block": str(manifest_by_name.loc[run_name, "seed_block"]),
            "replicate_index": int(manifest_by_name.loc[run_name, "replicate_index"]),
            "trainer_seed": int(manifest_by_name.loc[run_name, "trainer_seed"]),
            "data_seed": int(manifest_by_name.loc[run_name, "data_seed"]),
            "simulated_epoch_subset_seed": int(manifest_by_name.loc[run_name, "simulated_epoch_subset_seed"]),
        }
        for key, expected in expected_provenance.items():
            if provenance.get(key) != expected:
                raise ValueError(
                    f"{run_name}: native Table-9 provenance {key}={provenance.get(key)!r}, expected {expected!r}"
                )
        candidates[run_name].append((value, f"gs://{path}"))

    if len(candidates) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} unique Table-9 rows, found {len(candidates)}")
    selected: dict[str, tuple[float, str]] = {}
    for run_name, attempts in candidates.items():
        values = [value for value, _ in attempts]
        if max(values) - min(values) > VALUE_TOLERANCE:
            raise ValueError(f"{run_name}: native Table-9 retries disagree: {values}")
        if len(attempts) != 1:
            raise ValueError(f"{run_name}: expected one canonical native Table-9 artifact, found {len(attempts)}")
        selected[run_name] = attempts[0]
    return selected


def materialize(
    source_dir: Path,
    output_dir: Path,
    panel_dir: Path,
    timeout: int,
    expected_erratum_sha256: str,
) -> None:
    protocol = verify_erratum(source_dir, output_dir, expected_erratum_sha256)
    evaluator.verify_evaluation_protocol(output_dir, panel_dir)
    source_path = source_dir / "observations_60m.csv"
    with source_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        source_fieldnames = list(reader.fieldnames or ())
        source_rows = list(reader)
    observations = pd.read_csv(
        source_path,
        dtype={"training_wandb_id": str, "table9_wandb_id": str},
    )
    if len(observations) != EXPECTED_ROWS or observations["run_name"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_ROWS} unique source observations")
    manifest = pd.read_csv(panel_dir / "60m" / "run_manifest.csv")
    if len(manifest) != EXPECTED_ROWS or manifest["run_name"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_ROWS} unique frozen manifest rows")
    if set(manifest["run_name"]) != set(observations["run_name"]):
        raise ValueError("Frozen manifest and v2 observations differ in run identity")
    if not (manifest["expected_checkpoint_step"] == EXPECTED_STEP).all():
        raise ValueError(f"Frozen manifest must target exact final step {EXPECTED_STEP}")
    recorded_steps = observations.set_index("run_name")["expected_checkpoint_step"]
    manifest_steps = manifest.set_index("run_name")["expected_checkpoint_step"]
    if not recorded_steps.equals(manifest_steps.loc[recorded_steps.index]):
        raise ValueError("v2 observations do not reproduce frozen-manifest checkpoint steps")

    training, retry_audit_rows = exact_training_values(observations, manifest, timeout)
    table9 = exact_table9_values(observations, manifest, training)
    corrected_rows = []
    audit_rows = []
    for row in source_rows:
        run_name = str(row["run_name"])
        uncheatable, uncheatable_path = training[run_name]
        table9_value, table9_path = table9[run_name]
        old_uncheatable = float(row["uncheatable_bpb"])
        old_table9 = float(row["table9_macro_bpb"])
        uncheatable_delta = uncheatable - old_uncheatable
        table9_delta = table9_value - old_table9
        corrected_row = dict(row)
        corrected_row["uncheatable_bpb"] = repr(uncheatable)
        corrected_row.update(
            {
                "training_metric_source": "gcs_checkpoint_eval_metrics_authoritative_erratum",
                "training_metric_source_path": uncheatable_path,
                "table9_metric_source": "v2_value_verified_against_native_gcs_erratum",
                "table9_metric_source_path": table9_path,
                "uncheatable_bpb_original_v2": row["uncheatable_bpb"],
                "table9_macro_bpb_original_v2": row["table9_macro_bpb"],
                "uncheatable_bpb_correction": repr(uncheatable_delta),
                "table9_macro_bpb_correction": repr(table9_delta),
            }
        )
        corrected_rows.append(corrected_row)
        audit_rows.append(
            {
                "run_name": run_name,
                "point_id": row["point_id"],
                "old_uncheatable_bpb": old_uncheatable,
                "exact_uncheatable_bpb": uncheatable,
                "uncheatable_correction": uncheatable_delta,
                "old_table9_macro_bpb": old_table9,
                "exact_table9_macro_bpb": table9_value,
                "table9_correction": table9_delta,
                "uncheatable_source_path": uncheatable_path,
                "table9_source_path": table9_path,
            }
        )

    audit = pd.DataFrame(audit_rows)
    retry_audit = pd.DataFrame(retry_audit_rows)
    stale = set(audit.loc[audit["uncheatable_correction"].abs() > VALUE_TOLERANCE, "run_name"])
    if stale != EXPECTED_STALE_RUNS:
        raise ValueError(f"Stale Uncheatable set changed: expected {sorted(EXPECTED_STALE_RUNS)}, found {sorted(stale)}")
    if (audit["table9_correction"].abs() > VALUE_TOLERANCE).any():
        raise ValueError("Native Table-9 differs from the audited v2 values")
    table9_representation_differences = int(audit["table9_correction"].ne(0.0).sum())
    if table9_representation_differences != EXPECTED_TABLE9_REPRESENTATION_DIFFERENCES:
        raise ValueError(
            "Table-9 representation-difference count changed: "
            f"expected {EXPECTED_TABLE9_REPRESENTATION_DIFFERENCES}, found {table9_representation_differences}"
        )

    census = retry_audit["source_category"].value_counts().to_dict()
    expected_census = {
        "matching_finite_summary": 271,
        "preexisting_persisted": 2,
        "stale_finite_summary": 4,
    }
    if census != expected_census:
        raise ValueError(f"Metric-source census changed: expected {expected_census}, found {census}")

    extra_fields = [
        "table9_metric_source",
        "table9_metric_source_path",
        "uncheatable_bpb_original_v2",
        "table9_macro_bpb_original_v2",
        "uncheatable_bpb_correction",
        "table9_macro_bpb_correction",
    ]
    output_fields = source_fieldnames + [field for field in extra_fields if field not in source_fieldnames]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=output_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(corrected_rows)
    corrected_csv = buffer.getvalue()

    allowed_shared_changes = {
        "uncheatable_bpb",
        "training_metric_source",
        "training_metric_source_path",
    }
    changed_shared_cells: dict[str, list[str]] = defaultdict(list)
    for old_row, new_row in zip(source_rows, corrected_rows, strict=True):
        if old_row["run_name"] != new_row["run_name"]:
            raise ValueError("Corrected row ordering changed")
        for field in source_fieldnames:
            if old_row[field] != new_row[field]:
                changed_shared_cells[field].append(old_row["run_name"])
    unexpected_changes = set(changed_shared_cells) - allowed_shared_changes
    if unexpected_changes:
        raise ValueError(f"Unexpected shared-column changes: {sorted(unexpected_changes)}")
    changed_uncheatable_cells = set(changed_shared_cells["uncheatable_bpb"])
    if not EXPECTED_STALE_RUNS.issubset(changed_uncheatable_cells):
        raise ValueError("A preregistered substantive Uncheatable correction did not change its frozen cell")
    uncheatable_representation_differences = changed_uncheatable_cells - EXPECTED_STALE_RUNS
    audit_by_run = audit.set_index("run_name", verify_integrity=True)
    if uncheatable_representation_differences:
        representation_deltas = audit_by_run.loc[
            sorted(uncheatable_representation_differences), "uncheatable_correction"
        ].abs()
        if (representation_deltas > VALUE_TOLERANCE).any():
            raise ValueError("Unexpected substantive Uncheatable correction outside the preregistered four rows")
        uncheatable_representation_max = float(representation_deltas.max())
    else:
        uncheatable_representation_max = 0.0

    corrected = pd.read_csv(io.StringIO(corrected_csv))
    if len(corrected) != EXPECTED_ROWS:
        raise ValueError("Corrected observations changed row count")
    audit_csv = audit.to_csv(index=False)
    retry_audit_csv = retry_audit.to_csv(index=False)
    summary = {
        "erratum_protocol_sha256": protocol["erratum_protocol_sha256"],
        "erratum_protocol_file_sha256": expected_erratum_sha256,
        "rows": len(corrected),
        "uncheatable_corrected_rows": len(stale),
        "uncheatable_corrected_run_names": sorted(stale),
        "uncheatable_max_abs_correction": float(audit["uncheatable_correction"].abs().max()),
        "uncheatable_representation_difference_rows": len(uncheatable_representation_differences),
        "uncheatable_max_abs_representation_difference": uncheatable_representation_max,
        "source_census": census,
        "changed_shared_cells": dict(changed_shared_cells),
        "table9_corrected_rows": 0,
        "table9_native_representation_difference_rows": table9_representation_differences,
        "table9_native_max_abs_representation_difference": float(audit["table9_correction"].abs().max()),
        "observations_60m_sha256": sha256_text(corrected_csv),
        "source_correction_audit_60m_sha256": sha256_text(audit_csv),
        "training_retry_audit_60m_sha256": sha256_text(retry_audit_csv),
    }
    summary_json = json.dumps(summary, indent=2, sort_keys=True) + "\n"

    # No corrected artifact is durable until every identity and bit-difference check above passes.
    write_if_absent_or_equal(output_dir / "observations_60m.csv", corrected_csv)
    write_if_absent_or_equal(output_dir / "source_correction_audit_60m.csv", audit_csv)
    write_if_absent_or_equal(output_dir / "training_retry_audit_60m.csv", retry_audit_csv)
    write_if_absent_or_equal(
        output_dir / "materialization_summary.json",
        summary_json,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def verify_materialization(
    output_dir: Path,
    expected_erratum_sha256: str,
    expected_materialization_sha256: str,
) -> dict[str, Any]:
    summary_path = output_dir / "materialization_summary.json"
    if sha256(summary_path) != expected_materialization_sha256:
        raise ValueError("Materialization summary differs from the externally anchored Fieldbook SHA-256")
    summary = json.loads((output_dir / "materialization_summary.json").read_text())
    if summary["erratum_protocol_file_sha256"] != expected_erratum_sha256:
        raise ValueError("Materialization was not produced from the externally anchored erratum protocol")
    artifact_hashes = {
        "observations_60m_sha256": "observations_60m.csv",
        "source_correction_audit_60m_sha256": "source_correction_audit_60m.csv",
        "training_retry_audit_60m_sha256": "training_retry_audit_60m.csv",
    }
    for key, name in artifact_hashes.items():
        if summary[key] != sha256(output_dir / name):
            raise ValueError(f"Materialized artifact changed after anchoring: {name}")
    return summary


def evaluate(
    source_dir: Path,
    output_dir: Path,
    panel_dir: Path,
    expected_erratum_sha256: str,
    expected_materialization_sha256: str,
) -> None:
    verify_erratum(source_dir, output_dir, expected_erratum_sha256)
    verify_materialization(output_dir, expected_erratum_sha256, expected_materialization_sha256)
    evaluator.evaluate_60m(output_dir, panel_dir)


def metric_comparison(old: dict[str, Any], new: dict[str, Any], target: str) -> dict[str, Any]:
    old_target = old["targets"][target]
    new_target = new["targets"][target]
    return {
        "old_config": old_target["selected_config"],
        "new_config": new_target["selected_config"],
        "old_oof_rmse": old_target["candidate_metrics"]["rmse"],
        "new_oof_rmse": new_target["candidate_metrics"]["rmse"],
        "old_relative_rmse": old_target["relative_rmse"],
        "new_relative_rmse": new_target["relative_rmse"],
        "old_spearman": old_target["candidate_metrics"]["spearman"],
        "new_spearman": new_target["candidate_metrics"]["spearman"],
        "old_calibration_slope": old_target["candidate_metrics"]["observed_on_predicted_slope"],
        "new_calibration_slope": new_target["candidate_metrics"]["observed_on_predicted_slope"],
        "new_x32_metrics": new_target["x32_extrapolation_metrics"],
        "passed": new_target["passed_60m_gate"],
    }


def intervention_extremes(observations: pd.DataFrame, target: str) -> dict[str, Any]:
    anchor = float(observations.loc[observations["point_kind"].eq("proportional_anchor"), target].iloc[0])
    rows = observations.loc[
        observations["point_kind"].eq("focal_bucket_dose") & observations["epoch_multiplier"].le(16)
    ].copy()
    rows["gain_vs_anchor"] = anchor - rows[target]
    columns = ["run_name", "focal_domain", "epoch_multiplier", target, "gain_vs_anchor"]
    return {
        "anchor": anchor,
        "best": rows.nlargest(5, "gain_vs_anchor")[columns].to_dict(orient="records"),
        "worst": rows.nsmallest(5, "gain_vs_anchor")[columns].to_dict(orient="records"),
    }


def report(
    source_dir: Path,
    output_dir: Path,
    expected_erratum_sha256: str,
    expected_materialization_sha256: str,
) -> None:
    protocol = verify_erratum(source_dir, output_dir, expected_erratum_sha256)
    materialization = verify_materialization(
        output_dir,
        expected_erratum_sha256,
        expected_materialization_sha256,
    )
    old = json.loads((source_dir / "selected_60m.json").read_text())
    new = json.loads((output_dir / "selected_60m.json").read_text())
    if new["observation_sha256"] != materialization["observations_60m_sha256"]:
        raise ValueError("Frozen evaluator output does not reference the anchored corrected observations")
    old_table9_hash = canonical_hash(old["targets"]["table9"])
    new_table9_hash = canonical_hash(new["targets"]["table9"])
    if old_table9_hash != new_table9_hash:
        raise ValueError("Table-9 evaluation block changed despite bit-identical Table-9 inputs")
    observations = pd.read_csv(output_dir / "observations_60m.csv")
    retry_audit = pd.read_csv(output_dir / "training_retry_audit_60m.csv")
    stale_steps = []
    for row in retry_audit.loc[retry_audit["source_category"].eq("stale_finite_summary")].itertuples():
        matching_steps = json.loads(str(row.matching_earlier_steps))
        if len(matching_steps) != 1:
            raise ValueError(f"{row.run_name}: stale-step audit is not unique: {matching_steps}")
        stale_steps.append(f"{row.run_name} at step {matching_steps[0]}")
    payload = {
        "candidate_id": evaluator.model.CANDIDATE_ID,
        "erratum_protocol_sha256": protocol["erratum_protocol_sha256"],
        "erratum_protocol_file_sha256": expected_erratum_sha256,
        "materialization_summary_file_sha256": expected_materialization_sha256,
        "passed_both_targets": new["passed_both_targets"],
        "table9_block_bit_identical": True,
        "table9_block_sha256": new_table9_hash,
        "targets": {target: metric_comparison(old, new, target) for target in ("uncheatable", "table9")},
        "direct_interventions_through_x16": {
            "uncheatable": intervention_extremes(observations, "uncheatable_bpb"),
            "table9": intervention_extremes(observations, "table9_macro_bpb"),
        },
    }
    write_if_absent_or_equal(output_dir / "erratum_results.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# SUR-073 60M source-correction erratum",
        "",
        "The frozen v2 model form, selection procedure, folds, hyperparameters, x32 holdout, and gates are unchanged.",
        (
            "All 277 Uncheatable values now come from exact final-step checkpoint metrics; "
            "the 277 frozen Table-9 values are retained after per-row verification against native GCS results."
        ),
        "The original v2 artifact is preserved. Four stale finite W&B summaries triggered this versioned correction.",
        (
            f"Fieldbook anchors: erratum protocol `{expected_erratum_sha256}`; "
            f"materialization summary `{expected_materialization_sha256}`."
        ),
        "",
        f"**Frozen 60M gate: {'PASS' if new['passed_both_targets'] else 'FAIL'} on both targets.**",
        "",
        "| Target | Selected form | OOF RMSE | Linear RMSE | Ratio | Spearman | Calibration slope | x32 RMSE | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for target in ("uncheatable", "table9"):
        value = new["targets"][target]
        config = value["selected_config"]
        lines.append(
            f"| {target} | q={config['generator_order']}, {config['curvature_mode']}, ridge={config['ridge']} "
            f"| {value['candidate_metrics']['rmse']:.6f} | {value['linear_metrics']['rmse']:.6f} "
            f"| {value['relative_rmse']:.3f} | {value['candidate_metrics']['spearman']:.3f} "
            f"| {value['candidate_metrics']['observed_on_predicted_slope']:.3f} "
            f"| {value['x32_extrapolation_metrics']['rmse']:.6f} "
            f"| {'PASS' if value['passed_60m_gate'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            (
                "The gate establishes that nonlinear dose curvature is identified better than the signed-linear "
                "ablation on the complete 60M intervention panel. It does not establish cross-scale transfer or a "
                "supported raw 300M optimum; x32 remains a held-out, right-censored extrapolation stress test."
            ),
            "",
            "## Metric-source audit",
            "",
            (
                "- Table-9: 277 unique native artifacts, 51 components each, no missing rows, no retry "
                "disagreement, and no difference above 1e-10 from v2. The frozen CSV is byte-preserved; "
                f"{materialization['table9_native_representation_difference_rows']} rows differ from native JSON "
                "only at floating-point serialization precision "
                f"(maximum {materialization['table9_native_max_abs_representation_difference']:.3e})."
            ),
            (
                "- Uncheatable: 271 finite summaries matched the final checkpoint, two rows already used the "
                "persisted fallback, and p240, p247, p251, and p255 contained stale finite summaries."
            ),
            (
                "- Rewriting every Uncheatable value from the exact persisted source changed "
                f"{materialization['uncheatable_representation_difference_rows']} additional CSV cells only at "
                "floating-point serialization precision "
                f"(maximum {materialization['uncheatable_max_abs_representation_difference']:.3e})."
            ),
            (
                "- Each stale summary equals an earlier persisted evaluation exactly: "
                f"{', '.join(stale_steps)}; all are corrected from exact step {EXPECTED_STEP}."
            ),
            "- The complete corrected Table-9 evaluation block is bit-identical to v2.",
            (
                "- Before evaluation, removing the two gross gating outliers was expected to lower both candidate "
                "and linear RMSE. The prior relative-RMSE gate already passed at 0.763; curvature activity and "
                "fold-mode stability remained capable of moving against the candidate."
            ),
            "- No outcome was used to alter the candidate grid, fold design, bootstrap, gate, or optimizer.",
            "",
        ]
    )
    write_if_absent_or_equal(output_dir / "report.md", "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare(args.source_dir, args.output_dir)
        return
    if not args.expect_erratum_sha256:
        raise ValueError("--expect-erratum-sha256 is required after prepare")
    if args.mode == "materialize":
        materialize(
            args.source_dir,
            args.output_dir,
            args.panel_dir,
            args.wandb_timeout,
            args.expect_erratum_sha256,
        )
        return
    if not args.expect_materialization_sha256:
        raise ValueError("--expect-materialization-sha256 is required for evaluate and report")
    if args.mode == "evaluate":
        evaluate(
            args.source_dir,
            args.output_dir,
            args.panel_dir,
            args.expect_erratum_sha256,
            args.expect_materialization_sha256,
        )
        return
    report(
        args.source_dir,
        args.output_dir,
        args.expect_erratum_sha256,
        args.expect_materialization_sha256,
    )


if __name__ == "__main__":
    main()
