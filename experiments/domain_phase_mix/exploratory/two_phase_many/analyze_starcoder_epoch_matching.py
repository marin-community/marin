# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9"]
# ///
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare completed, frozen StarCoder proxy curves on their shared measured grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Literal

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from experiments.domain_phase_mix import starcoder_epoch_matching as design_module

ARMS = ("unmatched", "matched")
MEASUREMENT_COLUMNS = ("run_name", "step", "metric", "value", "design_sha256", "config_fingerprint", "status")


@dataclass(frozen=True)
class Measurement:
    run_name: str
    step: int
    metric: str
    value: float
    design_sha256: str
    config_fingerprint: str
    status: str


@dataclass(frozen=True)
class CurvePoint:
    arm: str
    starcoder_weight: float
    trainer_seed: int
    observed_bpb: float
    source_run_name: str
    source_kind: str
    config_fingerprint: str
    is_alias: bool


def read_measurements(path: Path) -> tuple[Measurement, ...]:
    """Read one provenance-bearing final measurement per newly trained run."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(MEASUREMENT_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Measurement CSV is missing columns: {sorted(missing)}")
        return tuple(
            Measurement(
                row["run_name"],
                int(row["step"]),
                row["metric"],
                float(row["value"]),
                row["design_sha256"],
                row["config_fingerprint"],
                row["status"],
            )
            for row in reader
        )


def read_plan(path: Path) -> dict:
    """Read the launcher's expected training identities without importing training code."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Launch plan must be a JSON object")
    return payload


def _expected_fingerprints(design: design_module.ExperimentDesign, plan: dict, stage: str) -> dict[str, str]:
    if plan.get("design_sha256") != design.design_sha256:
        raise ValueError("Launch plan design hash mismatch")
    if plan.get("stage") != stage:
        raise ValueError("Launch plan stage mismatch")
    if plan.get("metric") != design.primary_metric:
        raise ValueError("Launch plan metric mismatch")
    planned = {run.run_name: run for run in design_module.select_runs(design, stage)}
    if plan.get("new_training_runs") != len(planned):
        raise ValueError("Launch plan run count mismatch")
    rows = plan.get("runs")
    if not isinstance(rows, list):
        raise ValueError("Launch plan has no run records")
    fingerprints = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Launch plan run must be a JSON object")
        name = row.get("run_name")
        if name not in planned:
            raise ValueError(f"Launch plan includes a run outside stage {stage}: {name}")
        if name in fingerprints:
            raise ValueError(f"Duplicate launch plan run: {name}")
        for field, expected in asdict(planned[name]).items():
            if row.get(field) != expected:
                raise ValueError(f"Launch plan run configuration mismatch: {name}/{field}")
        fingerprint = row.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError(f"Launch plan is missing a config fingerprint: {name}")
        fingerprints[name] = fingerprint
    if set(fingerprints) != set(planned):
        raise ValueError("Launch plan is missing required stage runs")
    return fingerprints


def _validated_measurements(
    design: design_module.ExperimentDesign, measurements: tuple[Measurement, ...], stage: str, plan: dict
) -> dict[str, Measurement]:
    expected_fingerprints = _expected_fingerprints(design, plan, stage)
    planned = {run.run_name: run for run in design_module.select_runs(design, stage)}
    records = {}
    for record in measurements:
        if record.run_name in records:
            raise ValueError(f"Duplicate measurement: {record.run_name}")
        if record.run_name not in planned:
            raise ValueError(f"Measurement is outside the planned stage: {record.run_name}")
        run = planned[record.run_name]
        if record.design_sha256 != design.design_sha256:
            raise ValueError(f"Measurement design hash mismatch: {record.run_name}")
        if record.step != run.total_steps - 1:
            raise ValueError(f"Measurement is not the final step: {record.run_name}")
        if record.metric != design.primary_metric:
            raise ValueError(f"Measurement metric mismatch: {record.run_name}")
        if record.status != "succeeded":
            raise ValueError(f"Measurement run did not succeed: {record.run_name}")
        if not math.isfinite(record.value):
            raise ValueError(f"Nonfinite measurement: {record.run_name}")
        if record.config_fingerprint != expected_fingerprints[record.run_name]:
            raise ValueError(f"Measurement config fingerprint mismatch: {record.run_name}")
        records[record.run_name] = record
    required = {run.run_name for run in design_module.select_runs(design, stage)}
    missing = required - set(records)
    if missing:
        raise ValueError(f"Stage {stage} has {len(missing)} missing final measurements: {sorted(missing)}")
    return records


def _curve_summary(points: tuple[CurvePoint, ...], arm: str) -> list[dict]:
    groups = defaultdict(list)
    for point in points:
        if point.arm == arm:
            groups[point.starcoder_weight].append(point)
    return [
        {
            "starcoder_weight": weight,
            "mean_bpb": mean(point.observed_bpb for point in group),
            "trainer_seed_sd": stdev(point.observed_bpb for point in group) if len(group) > 1 else None,
            "n_measurements": len(group),
            "trainer_seeds": sorted(point.trainer_seed for point in group),
        }
        for weight, group in sorted(groups.items())
    ]


def _minimum(curve: list[dict]) -> tuple[float, list[float]]:
    minimum = min(row["mean_bpb"] for row in curve)
    ties = [row["starcoder_weight"] for row in curve if row["mean_bpb"] == minimum]
    return min(ties), ties


def analyze(
    design: design_module.ExperimentDesign, measurements: tuple[Measurement, ...], stage: str, plan: dict
) -> tuple[dict, tuple[CurvePoint, ...]]:
    """Report measured-grid selection only after every run in the stage succeeds."""
    records = _validated_measurements(design, measurements, stage, plan)
    requests = design_module.select_runs(design, stage)
    proxy_runs = [run for run in requests if run.arm in ARMS]
    grid = sorted({run.starcoder_weight for run in proxy_runs})
    seeds = sorted({run.trainer_seed for run in proxy_runs})
    runs_by_coordinate = {(run.arm, run.starcoder_weight, run.trainer_seed): run for run in proxy_runs}
    if len(runs_by_coordinate) != len(proxy_runs):
        raise ValueError("Frozen design has duplicate proxy coordinates")
    points = []
    for arm in ARMS:
        for weight in grid:
            for seed in seeds:
                is_alias = arm == "matched" and weight == 0.0
                source_arm = "unmatched" if is_alias else arm
                key = (source_arm, weight, seed)
                if key not in runs_by_coordinate:
                    raise ValueError(f"Frozen stage lacks a common-grid coordinate: {key}")
                run = runs_by_coordinate[key]
                record = records[run.run_name]
                points.append(
                    CurvePoint(arm, weight, seed, record.value, run.run_name, "new", record.config_fingerprint, is_alias)
                )
    historical = {}
    for observation in design.target_observations:
        if not observation.reusable:
            continue
        if observation.starcoder_weight in historical:
            raise ValueError("Duplicate reusable historical target coordinate")
        historical[observation.starcoder_weight] = observation
    targets = {run.starcoder_weight: run for run in requests if run.arm == "target"}
    if len(targets) != sum(run.arm == "target" for run in requests):
        raise ValueError("Duplicate newly trained target coordinate")
    for weight in grid:
        if weight in targets:
            run = targets[weight]
            record = records[run.run_name]
            point = CurvePoint(
                "target",
                weight,
                run.trainer_seed,
                record.value,
                run.run_name,
                "new",
                record.config_fingerprint,
                False,
            )
        elif weight in historical and weight < 1.0:
            observation = historical[weight]
            point = CurvePoint(
                "target",
                weight,
                design_module.REFERENCE_SEED,
                observation.observed_bpb,
                observation.source_run_name,
                "historical",
                "",
                False,
            )
        else:
            raise ValueError(f"No eligible target measurement at mixture weight {weight}")
        if not math.isfinite(point.observed_bpb):
            raise ValueError(f"Nonfinite historical target measurement at mixture weight {weight}")
        points.append(point)
    points = tuple(points)
    curves = {arm: _curve_summary(points, arm) for arm in (*ARMS, "target")}
    target_values = {row["starcoder_weight"]: row["mean_bpb"] for row in curves["target"]}
    target_best_weight, target_ties = _minimum(curves["target"])
    target_minimum = target_values[target_best_weight]
    selections: dict[str, dict] = {}
    for arm in ARMS:
        selected, ties = _minimum(curves[arm])
        selections[arm] = {
            "selected_weight": selected,
            "absolute_weight_displacement_from_target_minimum": abs(selected - target_best_weight),
            "tied_weights": ties,
            "proxy_mean_bpb": min(row["mean_bpb"] for row in curves[arm]),
            "target_bpb_at_selected_weight": target_values[selected],
            "target_grid_regret": target_values[selected] - target_minimum,
        }
    report = {
        "design_sha256": design.design_sha256,
        "launch_plan_sha256": design_module.canonical_sha256(plan),
        "stage": stage,
        "metric": design.primary_metric,
        "complete": True,
        "grid": grid,
        "proxy_trainer_seeds": seeds,
        "new_measurements_used": len(requests),
        "proxy_measurements_used": len(proxy_runs),
        "proxy_curve_points": len(proxy_runs) + len(seeds),
        "shared_zero_weight_sources": [runs_by_coordinate[("unmatched", 0.0, seed)].run_name for seed in seeds],
        "target_trainer_seed": design_module.REFERENCE_SEED,
        "target_minimum": {"weight": target_best_weight, "tied_weights": target_ties, "bpb": target_minimum},
        "selection": selections,
        "delta_target_regret": (
            selections["matched"]["target_grid_regret"] - selections["unmatched"]["target_grid_regret"]
        ),
        "curves": curves,
        "interpretation": {
            "selection_grid": {
                "pilot": "pilot grid only",
                "refinement": "adaptive refinement grid",
                "primary": "full prespecified grid",
                "replicated": "full prespecified grid",
            }[stage],
            "target_regret": "descriptive, conditional on the fixed target data and trainer seed",
            "delta_target_regret": "matched minus unmatched; negative values favor matching",
            "weight_displacement": "absolute distance from the lowest-weight exact target minimum",
            "replication": "proxy trainer seeds share the same fixed data pool",
            "zero_weight": "matched and unmatched reuse one web-only run per trainer seed",
            "tie_rule": "exact measured ties are reported; the lowest mixture weight is selected",
            "fingerprints": (
                "all input fingerprints match the checked launch plan; the collector verifies artifact records"
            ),
            "statistical_superiority_test": False,
        },
    }
    return report, points


def write_outputs(report: dict, points: tuple[CurvePoint, ...], output_dir: Path) -> None:
    """Write aggregate selection results and the individual measured curve points."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output_dir / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(CurvePoint.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(point) for point in points)


def plot_curves(
    report: dict,
    points: tuple[CurvePoint, ...],
    output_paths: tuple[Path, ...],
    *,
    mode: Literal["raw", "excess"] = "raw",
) -> Figure:
    """Plot raw BPB or BPB above each arm's observed mean-curve minimum."""
    if mode not in ("raw", "excess"):
        raise ValueError(f"Unknown curve plot mode: {mode}")
    figure = Figure(figsize=(6.8, 3.8), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    colors = {"target": "#333333", "unmatched": "#0072B2", "matched": "#D55E00"}
    labels = {"target": "Target", "unmatched": "Unmatched proxy", "matched": "Epoch-matched proxy"}
    minimum_sizes = {"target": 220, "unmatched": 140, "matched": 70} if mode == "excess" else dict.fromkeys(labels, 100)
    for arm in ("target", *ARMS):
        curve = report["curves"][arm]
        offset = min(row["mean_bpb"] for row in curve) if mode == "excess" else 0.0
        axis.plot(
            [row["starcoder_weight"] for row in curve],
            [row["mean_bpb"] - offset for row in curve],
            marker="o",
            markersize=4,
            linewidth=1.2,
            color=colors[arm],
            label=labels[arm],
        )
        if len(report["proxy_trainer_seeds"]) > 1 and arm != "target":
            samples = [point for point in points if point.arm == arm]
            axis.scatter(
                [point.starcoder_weight for point in samples],
                [point.observed_bpb - offset for point in samples],
                s=10,
                color=colors[arm],
                alpha=0.35,
            )
        weight, _ = _minimum(curve)
        value = next(row["mean_bpb"] for row in curve if row["starcoder_weight"] == weight)
        axis.scatter(
            [weight], [value - offset], marker="*", s=minimum_sizes[arm], color=colors[arm], edgecolor="white", zorder=5
        )
    ylabel = "BPB above observed grid minimum" if mode == "excess" else "Programming-languages BPB"
    axis.set(xlabel="StarCoder mixture fraction", ylabel=ylabel, xlim=(-0.02, 1.02))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.2)
    axis.legend(loc="best", framealpha=0.95)
    axis.set_title(f"{report['stage'].capitalize()} measured grid; stars mark grid minima", fontsize=10)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=design_module.DESIGN_PATH)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True, help="Launch plan with the expected config fingerprints")
    parser.add_argument("--stage", choices=design_module.STAGES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-pdf", type=Path)
    parser.add_argument("--plot-png", type=Path)
    parser.add_argument("--plot-excess-pdf", type=Path, help="Plot BPB above each arm's observed grid minimum")
    parser.add_argument("--plot-excess-png", type=Path, help="Plot BPB above each arm's observed grid minimum")
    args = parser.parse_args()
    design = design_module.load_design(args.design)
    report, points = analyze(design, read_measurements(args.measurements), args.stage, read_plan(args.plan))
    write_outputs(report, points, args.output_dir)
    paths = tuple(path for path in (args.plot_pdf, args.plot_png) if path is not None)
    if paths:
        plot_curves(report, points, paths)
    excess_paths = tuple(path for path in (args.plot_excess_pdf, args.plot_excess_png) if path is not None)
    if excess_paths:
        plot_curves(report, points, excess_paths, mode="excess")


if __name__ == "__main__":
    main()
