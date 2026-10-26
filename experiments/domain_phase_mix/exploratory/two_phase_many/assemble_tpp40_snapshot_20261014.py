# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["fsspec==2026.1.0", "numpy==2.3.5", "pandas==2.2.2"]
# ///
"""Assemble downloaded TPP40 receipts into local modeling tables without fitting."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from collect_tpp40_surrogate_snapshot_20260909 import digest, write_json

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "reference_outputs/tpp40_complete_20261014"
ENDPOINT_STEP = 27335
PHASE0_STEP = 21855
ROLES = (("endpoint", ENDPOINT_STEP), ("phase0", PHASE0_STEP))


def numeric_metrics(raw: dict[str, Any]) -> dict[str, float]:
    """Retain every finite scalar numeric value in a native evaluation record."""
    result = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite native metric {key}: {value}")
        if key != "step":
            result[key] = float(value)
    return result


def write_table(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> pd.DataFrame:
    """Write a deterministic CSV atomically; missing measurements remain blank."""
    frame = pd.DataFrame(rows, columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)
    return frame


def read_verified_raw(directory: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Recheck downloaded bytes against the collector receipt before assembly."""
    receipt = json.loads((directory / "receipt.json").read_text())
    files = {}
    for name, record in receipt["files"].items():
        content = (directory / name).read_bytes()
        if digest(content) != record["sha256"] or len(content) != record["bytes"]:
            raise ValueError(f"Downloaded artifact changed: {directory / name}")
        files[name] = content
    return receipt, files


def policy_rows(
    spec: dict[str, Any], receipt: dict[str, Any], buckets: pd.DataFrame
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Expose nominal and block-aligned aggregate coordinates on the frozen bucket axis."""
    names = buckets.bucket.tolist()
    phases = spec["phase_weights"]
    if set(phases["phase_0"]) != set(names) or set(phases["phase_1"]) != set(names):
        raise ValueError(f"Bucket axis mismatch for order {spec['run_order']}")
    w0 = np.array([phases["phase_0"][name] for name in names], dtype=float)
    w1 = np.array([phases["phase_1"][name] for name in names], dtype=float)
    for weights in (w0, w1):
        if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1, atol=1e-12):
            raise ValueError(f"Invalid phase weights for order {spec['run_order']}")
    alpha = float(spec["phase_0_fraction"])
    alpha_realized = (PHASE0_STEP + 1) / int(spec["train_steps"])
    aggregate = alpha * w0 + (1 - alpha) * w1
    aggregate_realized = alpha_realized * w0 + (1 - alpha_realized) * w1
    base = {
        "order": int(spec["run_order"]),
        "run_name": spec["run_name"],
        "source_run": spec["source_run_name"],
        "source": spec["panel_source"],
        "tied": bool(np.array_equal(w0, w1)),
        "tied_tolerance_1e12": bool(np.max(abs(w0 - w1)) < 1e-12),
        "maximum_phase_weight_difference": float(np.max(abs(w0 - w1))),
        "region": receipt["region"],
        "training_root": receipt["training_root"],
        "trainer_seed": spec["trainer_seed"],
        "data_seed": spec["data_seed"],
        "simulated_epoch_subset_seed": spec["simulated_epoch_subset_seed"],
        "alpha_nominal": alpha,
        "alpha_realized": alpha_realized,
        "phase0_checkpoint_step": PHASE0_STEP,
        "endpoint_checkpoint_step": ENDPOINT_STEP,
        "train_steps": spec["train_steps"],
        "realized_train_tokens": spec["realized_train_tokens"],
        "simulated_epoch_target_budget": spec["simulated_epoch_target_budget"],
        "total_trainable_params": spec["total_trainable_params"],
        "non_embedding_params": spec["non_embedding_params"],
    }
    weights = [
        {
            "order": base["order"],
            "bucket_index": i,
            "bucket": name,
            "phase0_weight": w0[i],
            "phase1_weight": w1[i],
            "alpha_nominal": alpha,
            "alpha_realized": alpha_realized,
            "aggregate_weight_nominal": aggregate[i],
            "aggregate_weight_realized": aggregate_realized[i],
            "contrast_phase1_minus_phase0": w1[i] - w0[i],
            "pool_tokens": buckets.pool_tokens.iloc[i],
            "exposure_epochs_nominal": (
                aggregate[i] * spec["simulated_epoch_target_budget"] / buckets.pool_tokens.iloc[i]
            ),
        }
        for i, name in enumerate(names)
    ]
    return base, weights


def endpoint_and_history(content: bytes, order: int) -> tuple[dict[str, float], list[dict[str, Any]], int]:
    """Preserve every logged numeric value and merge agreeing final scientific records."""
    endpoint: dict[str, float] = {}
    history = []
    final_count = 0
    for record_index, line in enumerate(content.decode().splitlines()):
        if not line.strip():
            continue
        raw = json.loads(line)
        metrics = numeric_metrics(raw)
        step = raw.get("step")
        for metric, value in metrics.items():
            history.append(
                {"order": order, "record_index": record_index, "step": step, "metric": metric, "value": value}
            )
        if step == ENDPOINT_STEP:
            final_count += 1
            for metric, value in metrics.items():
                scientific = metric.startswith("eval/") and metric.endswith(("bpb", "loss"))
                if scientific and metric in endpoint and endpoint[metric] != value:
                    raise ValueError(f"Conflicting final metric: order={order}, {metric}")
                endpoint[metric] = value
    return endpoint, history, final_count


def table9_rows(
    result: dict[str, Any], base: dict[str, Any], role: str, step: int, component_weights: dict[str, float]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Retain component, task, instance-count, and macro values with explicit checkpoint roles."""
    components = result.get("table9_components", {})
    tasks = result.get("task_bpb", {})
    counts = result.get("num_instances", {})
    if set(components) - set(component_weights):
        raise ValueError(f"Unregistered Table-9 components for order={base['order']}, {role}")
    prefix = {"order": base["order"], "checkpoint_role": role, "checkpoint_step": step}
    component_rows: list[dict[str, Any]] = [
        {
            **prefix,
            "component": name,
            "weight": weight,
            "bpb": components.get(name),
            "available": name in components,
        }
        for name, weight in component_weights.items()
    ]
    macro = result.get("table9_macro_bpb")
    component_rows.append(
        {**prefix, "component": "table9_macro", "weight": None, "bpb": macro, "available": macro is not None}
    )
    task_rows = [
        {
            **prefix,
            "task": task,
            "bpb": tasks.get(task),
            "num_instances": counts.get(task),
            "available": task in tasks,
        }
        for task in sorted(set(tasks) | set(counts))
    ]
    values = [row["bpb"] for row in component_rows + task_rows if row["available"]]
    if any(not isinstance(value, int | float) or not math.isfinite(value) for value in values):
        raise ValueError(f"Invalid Table-9 score for order={base['order']}, {role}")
    complete = len(components) == len(component_weights)
    reconstructed = sum(component_weights[name] * components[name] for name in component_weights) if complete else None
    discrepancy = abs(macro - reconstructed) if macro is not None and reconstructed is not None else None
    if discrepancy is not None and discrepancy > 1e-12:
        raise ValueError(f"Table-9 macro reconstruction disagreement: {base['order']}, {role}, {discrepancy}")
    summary = {
        "table9_macro_bpb": macro,
        "table9_frozen_weight_bpb": reconstructed,
        "table9_component_count": len(components),
        "table9_task_count": len(tasks),
        "table9_instance_count": sum(counts.values()) if counts else None,
        "table9_macro_absolute_discrepancy": discrepancy,
    }
    return component_rows, task_rows, summary


def objective_row(
    base: dict[str, Any],
    role: str,
    step: int,
    metrics: dict[str, float],
    weights: dict[str, float],
    table9: dict[str, Any],
) -> dict[str, Any]:
    """Calculate the frozen seven-component objective without substituting missing values."""
    present = [name for name in weights if name in metrics]
    uncheatable = sum(metrics[name] * weights[name] for name in weights) if len(present) == len(weights) else None
    native = metrics.get("eval/uncheatable_eval/bpb")
    return {
        "order": base["order"],
        "checkpoint_role": role,
        "checkpoint_step": step,
        "uncheatable_bpb": uncheatable,
        "uncheatable_native_bpb": native,
        "uncheatable_component_count": len(present),
        "uncheatable_native_absolute_discrepancy": (
            abs(uncheatable - native) if uncheatable is not None and native is not None else None
        ),
        **table9,
    }


def assemble(output: Path) -> dict[str, Any]:
    """Build modeling CSVs and coverage from immutable local collector receipts."""
    inputs = output / "inputs"
    specs = sorted(json.loads((inputs / "run_specs.json").read_text()), key=lambda row: row["run_order"])
    buckets = pd.read_csv(inputs / "buckets.csv")
    objectives = pd.read_csv(inputs / "objectives.csv")
    if len({spec["run_order"] for spec in specs}) != len(specs) or buckets.bucket.duplicated().any():
        raise ValueError("Duplicate run or bucket keys")
    objective_weights = {
        name: dict(zip(group.component, group.weight, strict=True)) for name, group in objectives.groupby("objective")
    }
    uncheatable_weights = objective_weights["uncheatable"]
    component_weights = {
        name.removeprefix("olmo_base_eval/easy_bpb/").removesuffix("/bpb"): weight
        for name, weight in objective_weights["table9"].items()
    }
    if len(uncheatable_weights) != 7 or len(component_weights) != 51:
        raise ValueError("Expected frozen seven-component Uncheatable and 51-component Table-9 objectives")
    for name, weights in objective_weights.items():
        if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
            raise ValueError(f"Invalid frozen objective weight: {name}")
    tables: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in (
            "outcomes",
            "coverage",
            "policies",
            "policy_weights",
            "native_history",
            "phase0_native",
            "measured_objectives",
            "endpoint_table9_components",
            "phase0_table9_components",
            "endpoint_table9_tasks",
            "phase0_table9_tasks",
        )
    }
    raw_hashes = {}
    final_record_counts = []
    for spec in specs:
        order = int(spec["run_order"])
        directory = output / "raw" / f"{order:03d}"
        receipt, files = read_verified_raw(directory)
        if receipt["order"] != order:
            raise ValueError(f"Receipt order mismatch: {directory}")
        raw_hashes[f"raw/{order:03d}/receipt.json"] = digest((directory / "receipt.json").read_bytes())
        base, weights = policy_rows(spec, receipt, buckets)
        tables["policies"].append(base)
        tables["policy_weights"].extend(weights)
        final, history, final_count = endpoint_and_history(files["eval_metrics.jsonl"], order)
        final_record_counts.append(final_count)
        tables["native_history"].extend(history)
        completed = files["training_status.txt"].decode().strip() == "SUCCESS" and bool(final)
        tables["outcomes"].append(
            {
                **base,
                "completed": completed,
                "checkpoint_role": "endpoint",
                "checkpoint_step": ENDPOINT_STEP,
                "step": ENDPOINT_STEP,
                **final,
            }
        )
        boundary = json.loads(files["phase0_validation.json"]) if "phase0_validation.json" in files else {}
        boundary_metrics = numeric_metrics(boundary.get("metrics", {}))
        tables["phase0_native"].append(
            {**base, "checkpoint_role": "phase0", "checkpoint_step": PHASE0_STEP, **boundary_metrics}
        )
        coverage = {
            "order": order,
            "run_name": base["run_name"],
            "source_run": base["source_run"],
            "source": base["source"],
            "tied": base["tied"],
            "region": base["region"],
            "completed": completed,
            "endpoint_native_metric_count": len(final),
            "phase0_native_metric_count": len(boundary_metrics),
            "endpoint_native_final_record_count": final_count,
        }
        for role, step in ROLES:
            filename = f"{role}_table9.json"
            result = json.loads(files[filename]) if filename in files else {}
            components, tasks, summary = table9_rows(result, base, role, step, component_weights)
            tables[f"{role}_table9_components"].extend(components)
            tables[f"{role}_table9_tasks"].extend(tasks)
            metrics = final if role == "endpoint" else boundary_metrics
            objective = objective_row(base, role, step, metrics, uncheatable_weights, summary)
            tables["measured_objectives"].append(objective)
            for key in ("uncheatable_component_count", "table9_component_count", "table9_task_count"):
                coverage[f"{role}_{key}"] = objective[key]
        tables["coverage"].append(coverage)

    task_names = sorted({row["task"] for role, _ in ROLES for row in tables[f"{role}_table9_tasks"]})
    for role, step in ROLES:
        task_table = tables[f"{role}_table9_tasks"]
        present = {(row["order"], row["task"]) for row in task_table}
        task_table.extend(
            {
                "order": spec["run_order"],
                "checkpoint_role": role,
                "checkpoint_step": step,
                "task": task,
                "bpb": None,
                "num_instances": None,
                "available": False,
            }
            for spec in specs
            for task in task_names
            if (spec["run_order"], task) not in present
        )
        task_table.sort(key=lambda row: (row["order"], row["task"]))
    paths = {}
    frames = {}
    for name, rows in tables.items():
        path = inputs / f"{name}.csv"
        frames[name] = write_table(path, rows)
        paths[name] = str(path.relative_to(output))
    axis_path = inputs / "bucket_axis.csv"
    write_table(axis_path, [{"bucket_index": i, **row} for i, row in enumerate(buckets.to_dict("records"))])
    paths["bucket_axis"] = str(axis_path.relative_to(output))
    metric_coverage = []
    for name, role in (("outcomes", "endpoint"), ("phase0_native", "phase0")):
        frame = frames[name]
        metric_coverage.extend(
            {
                "checkpoint_role": role,
                "metric": column,
                "present": int(frame[column].notna().sum()),
                "missing": int(frame[column].isna().sum()),
            }
            for column in frame.columns
            if column.startswith("eval/")
        )
    metric_path = inputs / "native_metric_coverage.csv"
    write_table(metric_path, metric_coverage)
    paths["native_metric_coverage"] = str(metric_path.relative_to(output))
    coverage_frame = frames["coverage"]
    objective_frame = frames["measured_objectives"]
    coverage = {
        "designed": len(specs),
        "completed": int(coverage_frame.completed.sum()),
        "completed_tied": int((coverage_frame.completed & coverage_frame.tied).sum()),
        "completed_asymmetric": int((coverage_frame.completed & ~coverage_frame.tied).sum()),
        "bucket_count": len(buckets),
        "frozen_objective_weight_sums": {name: sum(weights.values()) for name, weights in objective_weights.items()},
        "table9_task_axis_count": len(task_names),
        "native_history_numeric_values": len(tables["native_history"]),
        "endpoint_native_final_record_count_range": [min(final_record_counts), max(final_record_counts)],
        "by_source": (
            coverage_frame.groupby(["source", "tied"])
            .agg(designed=("order", "size"), completed=("completed", "sum"))
            .reset_index()
            .to_dict("records")
        ),
        "by_region": coverage_frame.region.value_counts().to_dict(),
        "by_checkpoint_role": {
            role: {
                "uncheatable_complete": int((group.uncheatable_component_count == 7).sum()),
                "table9_components_complete": int((group.table9_component_count == 51).sum()),
                "table9_tasks_complete": int((group.table9_task_count == 104).sum()),
                "uncheatable_native_max_absolute_discrepancy": (
                    float(group.uncheatable_native_absolute_discrepancy.max())
                    if group.uncheatable_native_absolute_discrepancy.notna().any()
                    else None
                ),
                "table9_macro_max_absolute_discrepancy": (
                    float(group.table9_macro_absolute_discrepancy.max())
                    if group.table9_macro_absolute_discrepancy.notna().any()
                    else None
                ),
            }
            for role, group in objective_frame.groupby("checkpoint_role")
        },
    }
    result = {
        "coverage": coverage,
        "table_paths": paths,
        "table_sha256": {name: digest((output / path).read_bytes()) for name, path in paths.items()},
        "raw_receipt_sha256": raw_hashes,
        "assembly_source_sha256": digest(Path(__file__).read_bytes()),
        "geometry": {
            "nominal_alpha": "phase_0_fraction from frozen run specification",
            "realized_alpha": "(phase0 checkpoint step + 1) / train_steps, after block alignment",
            "aggregate": "alpha * phase0_weight + (1 - alpha) * phase1_weight; both alphas retained",
            "exposure_epochs": "nominal aggregate * simulated_epoch_target_budget / pool_tokens",
            "tied": "exact equality of both phase weight vectors; tolerance diagnostic stored separately",
        },
        "missing_data": "Blank measurements and explicit available/count columns; no imputation",
        "checkpoint_semantics": "phase0 is measured boundary state; endpoint is final outcome",
    }
    write_json(output / "assembly.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = assemble(args.output)
    print(json.dumps(result["coverage"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
