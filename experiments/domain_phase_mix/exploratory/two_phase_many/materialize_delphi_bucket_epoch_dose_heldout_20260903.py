# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec", "gcsfs", "numpy", "pandas", "wandb>=0.21"]
# ///
"""Freeze currently available Delphi epoch-dose outcomes for heldout modeling.

All Uncheatable payloads must contain the complete seven-component inventory.
Finished runs may use validated W&B summaries. Non-finished runs must instead
use the exact final training step from their small, region-local
``eval_metrics.jsonl`` because W&B can retain a complete but stale summary
after preemption. Table-9 payloads are included only when all 51 components are
present and reconstruct the reported macro average.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    prepare_single_phase_heldout_benchmark_20260902 as heldout,
)

REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
PANEL_DIR = REFERENCE_OUTPUTS / "bucket_epoch_dose_response_20260729" / "full" / "delphi_3e18"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "bucket_epoch_dose_response_20260729" / "recovery" / "delphi_3e18_20260902"
RUN_MANIFEST = PANEL_DIR / "run_manifest.csv"
PHASE_WEIGHTS = PANEL_DIR / "phase_weights.csv"
TABLE9_METADATA = heldout.TABLE9_METADATA

TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
TRAIN_FILTER = {"tags": "bucket-epoch-dose-response"}
EVAL_GROUPS = (
    "olmo_base_eval_table9_bucket_epoch_dose_delphi_full_20260729",
    "olmo_base_eval_table9_bucket_epoch_dose_delphi_3e18_full_20260729",
)
EXPECTED_ROWS = 277
RETRY_TOLERANCE = 1e-10


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _run_name_matches(run: Any, run_name: str) -> bool:
    return run_name in set(run.tags or ()) or str(run.name).startswith(f"{run_name}-")


def _component_aggregate(target: str, values: np.ndarray) -> float:
    if target == "table9":
        return float(values.mean())
    return float(values @ heldout.DELPHI_UNCHEATABLE_WEIGHTS)


def _validated_payload(
    *,
    target: str,
    aggregate: object,
    component_values: list[object],
    identity: str,
) -> tuple[float, np.ndarray]:
    aggregate_value = _finite(aggregate)
    values = np.asarray([_finite(value) for value in component_values], dtype=float)
    if aggregate_value is None or not np.isfinite(values).all():
        raise ValueError(f"{identity}: incomplete {target} endpoint payload")
    reconstructed = _component_aggregate(target, values)
    error = abs(reconstructed - aggregate_value)
    if error > heldout.AGGREGATE_TOLERANCE:
        raise ValueError(f"{identity}: {target} components differ from aggregate by {error:.3g}")
    return aggregate_value, values


def _summary_uncheatable(run: Any) -> tuple[float, np.ndarray] | None:
    summary = dict(run.summary)
    keys = (heldout.UNCHEATABLE_AGGREGATE, *heldout.UNCHEATABLE_COMPONENTS)
    if any(key not in summary for key in keys):
        return None
    return _validated_payload(
        target="uncheatable",
        aggregate=summary[heldout.UNCHEATABLE_AGGREGATE],
        component_values=[summary[key] for key in heldout.UNCHEATABLE_COMPONENTS],
        identity=str(run.id),
    )


def _persisted_uncheatable(run: Any, expected_step: int) -> tuple[float, np.ndarray, str] | None:
    trainer = dict(run.config.get("trainer") or {})
    checkpoint_path = str(dict(trainer.get("checkpointer") or {}).get("base_path") or "")
    if not checkpoint_path.startswith("gs://marin-us-east5/"):
        return None
    metrics_uri = f"{checkpoint_path.rstrip('/')}/eval_metrics.jsonl"
    filesystem, path = fsspec.core.url_to_fs(metrics_uri)
    if not filesystem.exists(path):
        return None

    payloads: list[tuple[float, np.ndarray]] = []
    with filesystem.open(path, "rt") as handle:
        for line in handle:
            payload = json.loads(line)
            if int(payload.get("step", -1)) != expected_step:
                continue
            try:
                payloads.append(
                    _validated_payload(
                        target="uncheatable",
                        aggregate=payload.get(heldout.UNCHEATABLE_AGGREGATE),
                        component_values=[payload.get(key) for key in heldout.UNCHEATABLE_COMPONENTS],
                        identity=f"{run.id}@step-{expected_step}",
                    )
                )
            except ValueError:
                continue
    if not payloads:
        return None
    reference_aggregate, reference_values = payloads[0]
    for aggregate, values in payloads[1:]:
        if abs(aggregate - reference_aggregate) > RETRY_TOLERANCE or not np.allclose(
            values, reference_values, atol=RETRY_TOLERANCE, rtol=0.0
        ):
            raise ValueError(f"{run.id}: exact-final-step Uncheatable payloads disagree")
    return reference_aggregate, reference_values, metrics_uri


def _uncheatable_candidate(run: Any, expected_step: int) -> dict[str, object] | None:
    if str(run.state).lower() != "finished":
        persisted = _persisted_uncheatable(run, expected_step)
        if persisted is None:
            return None
        aggregate, values, source_uri = persisted
        provenance = "gcs_exact_final_step"
    else:
        summary_payload = _summary_uncheatable(run)
        if summary_payload is not None:
            aggregate, values = summary_payload
            provenance = "wandb_summary_validated"
            source_uri = ""
        else:
            persisted = _persisted_uncheatable(run, expected_step)
            if persisted is None:
                return None
            aggregate, values, source_uri = persisted
            provenance = "gcs_exact_final_step"
    return {
        "run": run,
        "aggregate": aggregate,
        "values": values,
        "provenance": provenance,
        "source_uri": source_uri,
    }


def _selected_uncheatable(runs: list[Any], run_name: str, expected_step: int) -> dict[str, object]:
    candidates = [
        candidate
        for run in runs
        if _run_name_matches(run, run_name)
        if (candidate := _uncheatable_candidate(run, expected_step)) is not None
    ]
    if not candidates:
        raise ValueError(f"{run_name}: no complete Uncheatable endpoint")
    reference = candidates[0]
    for candidate in candidates[1:]:
        if abs(float(candidate["aggregate"]) - float(reference["aggregate"])) > RETRY_TOLERANCE or not np.allclose(
            candidate["values"], reference["values"], atol=RETRY_TOLERANCE, rtol=0.0
        ):
            raise ValueError(f"{run_name}: complete training attempts disagree")
    return max(candidates, key=lambda candidate: str(candidate["run"].created_at))


def _table9_candidate(run: Any) -> dict[str, object] | None:
    provenance = dict(run.config.get("provenance") or {})
    run_name = str(provenance.get("run_name") or "")
    if not run_name:
        return None
    summary = dict(run.summary)
    summary_keys = heldout.table9_summary_keys()
    if any(summary_keys[component] not in summary for component in heldout.table9_components()):
        return None
    values = np.asarray([summary[summary_keys[component]] for component in heldout.table9_components()], dtype=float)
    aggregate = next((_finite(summary.get(key)) for key in heldout.TABLE9_AGGREGATE_KEYS if key in summary), None)
    if aggregate is None:
        aggregate = float(values.mean())
    aggregate, values = _validated_payload(
        target="table9",
        aggregate=aggregate,
        component_values=list(values),
        identity=str(run.id),
    )
    return {"run_name": run_name, "run": run, "aggregate": aggregate, "values": values}


def _selected_table9(runs: list[Any]) -> dict[str, dict[str, object]]:
    candidates: dict[str, list[dict[str, object]]] = {}
    for run in runs:
        candidate = _table9_candidate(run)
        if candidate is not None:
            candidates.setdefault(str(candidate["run_name"]), []).append(candidate)

    selected: dict[str, dict[str, object]] = {}
    for run_name, attempts in candidates.items():
        reference = attempts[0]
        for attempt in attempts[1:]:
            if abs(float(attempt["aggregate"]) - float(reference["aggregate"])) > RETRY_TOLERANCE or not np.allclose(
                attempt["values"], reference["values"], atol=RETRY_TOLERANCE, rtol=0.0
            ):
                raise ValueError(f"{run_name}: complete Table-9 attempts disagree")
        selected[run_name] = max(attempts, key=lambda attempt: str(attempt["run"].created_at))
    return selected


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def materialize(output_dir: Path, timeout: int) -> dict[str, Any]:
    manifest = pd.read_csv(RUN_MANIFEST).sort_values("run_order")
    if len(manifest) != EXPECTED_ROWS or manifest["run_name"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_ROWS} unique Delphi epoch-dose runs")

    api = wandb.Api(timeout=timeout)
    training_runs = list(api.runs(TRAIN_PROJECT, filters=TRAIN_FILTER, per_page=500))
    evaluation_runs: list[Any] = []
    for group in EVAL_GROUPS:
        evaluation_runs.extend(api.runs(EVAL_PROJECT, filters={"group": group}, per_page=500))
    table9_by_name = _selected_table9(evaluation_runs)

    result_rows: list[dict[str, object]] = []
    uncheatable_rows: list[dict[str, object]] = []
    table9_rows: list[dict[str, object]] = []
    for spec in manifest.itertuples(index=False):
        run_name = str(spec.run_name)
        uncheatable = _selected_uncheatable(training_runs, run_name, int(spec.expected_checkpoint_step))
        training_run = uncheatable["run"]
        for position, (component, value) in enumerate(
            zip(heldout.UNCHEATABLE_COMPONENTS, uncheatable["values"], strict=True)
        ):
            uncheatable_rows.append(
                {
                    "run_order": int(spec.run_order),
                    "run_name": run_name,
                    "component_position": position,
                    "component": component,
                    "bpb": float(value),
                    "provenance": str(uncheatable["provenance"]),
                }
            )

        evaluation = table9_by_name.get(run_name)
        if evaluation is None:
            eval_id = ""
            eval_url = ""
            eval_state = ""
            table9_macro = np.nan
            table9_count = 0
        else:
            eval_run = evaluation["run"]
            eval_id = str(eval_run.id)
            eval_url = str(eval_run.url)
            eval_state = str(eval_run.state)
            table9_macro = float(evaluation["aggregate"])
            table9_count = len(heldout.table9_components())
            for position, (component, value) in enumerate(
                zip(heldout.table9_components(), evaluation["values"], strict=True)
            ):
                table9_rows.append(
                    {
                        "run_order": int(spec.run_order),
                        "run_name": run_name,
                        "component_position": position,
                        "component": component,
                        "bpb": float(value),
                        "provenance": "wandb_summary_validated",
                    }
                )

        result_rows.append(
            {
                "run_order": int(spec.run_order),
                "run_name": run_name,
                "training_wandb_run_id": str(training_run.id),
                "training_wandb_url": str(training_run.url),
                "training_wandb_state": str(training_run.state),
                "uncheatable_bpb": float(uncheatable["aggregate"]),
                "uncheatable_provenance": str(uncheatable["provenance"]),
                "uncheatable_source_uri": str(uncheatable["source_uri"]),
                "uncheatable_component_count": len(heldout.UNCHEATABLE_COMPONENTS),
                "table9_eval_run_id": eval_id,
                "table9_eval_url": eval_url,
                "table9_eval_state": eval_state,
                "table9_macro_bpb": table9_macro,
                "table9_component_count": table9_count,
            }
        )

    results = pd.DataFrame(result_rows)
    uncheatable_components = pd.DataFrame(uncheatable_rows)
    table9_components = pd.DataFrame(table9_rows)
    if len(results) != EXPECTED_ROWS or not results["uncheatable_component_count"].eq(7).all():
        raise ValueError("Uncheatable materialization is incomplete")
    complete_table9 = results["table9_component_count"].eq(51)
    if not results.loc[complete_table9, "table9_macro_bpb"].notna().all():
        raise ValueError("A complete Table-9 row lacks its aggregate")

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(results, output_dir / "heldout_results.csv")
    _atomic_csv(uncheatable_components, output_dir / "uncheatable_components.csv")
    _atomic_csv(table9_components, output_dir / "table9_components.csv")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "observed_at": datetime.now(UTC).isoformat(),
        "counts": {
            "runs": len(results),
            "uncheatable_complete": int(results["uncheatable_component_count"].eq(7).sum()),
            "uncheatable_from_wandb": int(results["uncheatable_provenance"].eq("wandb_summary_validated").sum()),
            "uncheatable_from_gcs": int(results["uncheatable_provenance"].eq("gcs_exact_final_step").sum()),
            "table9_complete": int(complete_table9.sum()),
            "table9_missing": int((~complete_table9).sum()),
        },
        "sources": {
            "training_project": TRAIN_PROJECT,
            "training_filter": TRAIN_FILTER,
            "evaluation_project": EVAL_PROJECT,
            "evaluation_groups": list(EVAL_GROUPS),
            "run_manifest": str(RUN_MANIFEST.relative_to(REPO_ROOT)),
            "phase_weights": str(PHASE_WEIGHTS.relative_to(REPO_ROOT)),
        },
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): _file_sha256(path)
            for path in (RUN_MANIFEST, PHASE_WEIGHTS, TABLE9_METADATA)
        },
    }
    (output_dir / "heldout_materialization_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=180)
    args = parser.parse_args()
    print(json.dumps(materialize(args.output_dir, args.wandb_timeout)["counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
