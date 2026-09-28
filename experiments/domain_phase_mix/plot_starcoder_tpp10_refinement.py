# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot completed refinement endpoints while retaining both archived run plans."""

import argparse
import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import wandb
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from marin.training.training import LevanterCheckpoint

from experiments.domain_phase_mix import analyze_starcoder_tpp10 as analysis
from experiments.domain_phase_mix import starcoder_tpp10 as experiment
from experiments.domain_phase_mix.evaluate_starcoder_tpp10_uncheatable import checkpoint_metadata
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

REFINEMENT_SHA256 = "99ca724bc3bde33e80afb07373e47ddc0d7e44f22e15cd41da3daa3de2f4ef40"
REFINEMENT_GRID = (40, 55, 60, 65, 80)
COMPLETE_GRID = (0, 10, 30, 40, 50, 55, 60, 65, 70, 80, 90, 100)
POPULATION_COUNTS_SHA256 = "7c7258e80208b230394cbee4ce1f41436d6c248d5d02e4ad86d24d05b5c32f4a"
METRIC_DEFINITION = "scored_byte_bpb_from_token_loss_v1"
SCHEMA2_RECONSTRUCTION_TOLERANCE = 5e-5


def normalize_final_metrics(
    runs: list[dict], recorded_values: dict[str, float], manifest_path: Path, population: dict
) -> tuple[dict[str, float], list[dict]]:
    """Reconstruct scored-byte BPB from hash-verified final token-average losses.

    The archived raw values remain provenance only: an absent schema identifies
    the audited legacy evaluator, while schema 2 must agree with reconstruction.
    """
    records = json.loads(manifest_path.read_text())
    expected = {r["run_name"]: r for r in runs}
    if len(expected) != len(runs) or set(recorded_values) != set(expected):
        raise ValueError("Normalization requires each verified run exactly once")
    if len(records) != len(expected) or {r["run_name"] for r in records} != set(expected):
        raise ValueError("Metric manifest does not match the verified run set")
    metric = experiment.PRIMARY_METRIC
    loss_metric = metric.removesuffix("bpb") + "loss"
    factor = population["tokens"] / (population["bytes"] * math.log(2))
    values, provenance = {}, []
    for record in records:
        name = record["run_name"]
        request = expected[name]
        if any(record.get(key) != value for key, value in request.items()):
            raise ValueError(f"Metric manifest identity differs from the frozen plan: {name}")
        source = Path(record["source"])
        if file_sha256(source) != record["source_sha256"]:
            raise ValueError(f"Final metric source hash differs: {name}")
        events = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        final = [event for event in events if event.get("step") == request["total_steps"] - 1 and metric in event]
        if not final or len(final) != record["final_records"]:
            raise ValueError(f"Missing or changed final-step metric records: {name}")
        for event in final:
            if loss_metric not in event or not math.isfinite(event[loss_metric]) or event[loss_metric] < 0:
                raise ValueError(f"Missing or invalid final token loss: {name}")
            if any(
                event.get(key) != record["final"].get(key) for key in (metric, loss_metric, "eval/bpb_schema_version")
            ):
                raise ValueError(f"Final metrics disagree with the audited manifest: {name}")
            if event[metric] != recorded_values[name]:
                raise ValueError(f"Raw BPB disagrees with the original verified measurement: {name}")
            schema = event.get("eval/bpb_schema_version")
            if schema is not None and schema != 2:
                raise ValueError(f"Unknown BPB schema {schema}: {name}")
            value = event[loss_metric] * factor
            if schema == 2 and abs(value - event[metric]) > SCHEMA2_RECONSTRUCTION_TOLERANCE:
                raise ValueError(f"Schema-2 BPB disagrees with the audited population: {name}")
        values[name] = value
        provenance.append(
            {
                **request,
                "step": request["total_steps"] - 1,
                "reported_bpb": recorded_values[name],
                "reported_schema": schema,
                "token_average_loss": final[0][loss_metric],
                "normalized_bpb": value,
                "source": str(source.resolve()),
                "source_sha256": record["source_sha256"],
            }
        )
    return values, provenance


def collect_endpoint(plan: dict, request: dict) -> dict:
    """Verify a completed artifact; return its state if training is incomplete."""
    path = request["output_path"]
    status = StatusFile(path, worker_id="tpp10-refinement-plot").status
    row = {**request, "artifact_status": status, "plan_sha256": plan["plan_sha256"]}
    if status != STATUS_SUCCESS:
        return row
    checkpoint = checkpoint_metadata(request)
    with fsspec.open(path + "/verified_runtime.json", "rt") as handle:
        runtime = json.load(handle)
    expected_runtime = {
        "design_sha256": plan["design_sha256"],
        "versions": plan["runtime_versions"],
        "code_sha256": plan["code_sha256"],
    }
    if runtime != expected_runtime:
        raise ValueError(f"Runtime differs from the archived plan: {request['run_name']}")
    step = request["total_steps"] - 1
    values = []
    with fsspec.open(LevanterCheckpoint(path=path).checkpoint_dir + "/eval_metrics.jsonl", "rt") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if item.get("step") == step and plan["primary_metric"] in item:
                    values.append(float(item[plan["primary_metric"]]))
    if not values or not all(math.isfinite(v) and v == values[0] for v in values):
        raise ValueError(f"Missing or conflicting final metric: {request['run_name']}")
    run = wandb.Api(timeout=30).run(f"marin-community/marin/{request['run_name']}")
    if run.state != "finished" or abs(float(run.summary[plan["primary_metric"]]) - values[0]) > 1e-10:
        raise ValueError(f"W&B disagrees with the final artifact: {request['run_name']}")
    return {
        **row,
        "value": values[0],
        "step": step,
        "metric": plan["primary_metric"],
        "checkpoint": checkpoint,
        "runtime_verified": True,
        "wandb_state": run.state,
    }


def summarize(pilot: dict, refinement: dict, pilot_values: dict[str, float], endpoints: list[dict]) -> dict:
    """Keep per-curve coverage explicit; never impute a missing endpoint or seed."""
    if len(endpoints) != len(refinement["runs"]):
        raise ValueError("Snapshot does not cover the refinement plan")
    for request, row in zip(refinement["runs"], endpoints, strict=True):
        if any(row[key] != value for key, value in request.items()):
            raise ValueError(f"Snapshot identity differs: {request['run_name']}")
        if row["artifact_status"] == STATUS_SUCCESS:
            if not row["runtime_verified"] or row["wandb_state"] != "finished" or not math.isfinite(row["value"]):
                raise ValueError(f"Unverified snapshot endpoint: {request['run_name']}")
    completed = [row for row in endpoints if row["artifact_status"] == STATUS_SUCCESS]
    values = {**pilot_values, **{row["run_name"]: row["value"] for row in completed}}
    index = {
        (r["arm"], r["subset_seed"], r["trainer_seed"], r["percent"]): r["run_name"]
        for r in pilot["runs"] + refinement["runs"]
    }
    grid = sorted({r["percent"] for r in pilot["runs"] + refinement["runs"]})
    curves: list[dict] = []
    for arm, subset in [("target", None), ("unmatched", None)] + [("matched", d) for d in experiment.SUBSET_SEEDS]:
        seeds = experiment.TRAINER_SEEDS[:1] if arm == "target" else experiment.TRAINER_SEEDS
        points = []
        for percent in grid:
            names = [
                (
                    index[("unmatched", None, s, 0)]
                    if arm == "matched" and percent == 0
                    else index[(arm, subset, s, percent)]
                )
                for s in seeds
            ]
            if not all(name in values for name in names):
                continue
            y = [values[name] for name in names]
            points.append({"percent": percent, "value": float(np.mean(y)), "seed_values": y, "run_names": names})
        selected = min(points, key=lambda r: (r["value"], r["percent"]))
        curves.append({"arm": arm, "subset_seed": subset, "points": points, "selected_percent": selected["percent"]})
    used = {name for curve in curves for point in curve["points"] for name in point["run_names"]}
    result = {
        "pilot_plan_sha256": pilot["plan_sha256"],
        "refinement_plan_sha256": refinement["plan_sha256"],
        "refinement_complete": len(completed),
        "refinement_planned": len(refinement["runs"]),
        "curves": curves,
        "pilot_common_grid_analysis": analysis.analyze(pilot, pilot_values),
        "missing_run_names": [r["run_name"] for r in endpoints if r["artifact_status"] != STATUS_SUCCESS],
        "verified_but_unplotted": sorted(set(values) - used),
        "scope": (
            "Per-curve observed minima on available grids. No refined target regret while target grid is incomplete."
        ),
    }
    if len(completed) == len(refinement["runs"]):
        runs = pilot["runs"] + refinement["runs"]
        if len(pilot["runs"]) != 57 or len(completed) != 45 or len({r["run_name"] for r in runs}) != 102:
            raise ValueError("Complete refinement analysis requires the 57 pilot and 45 distinct refinement artifacts")
        if grid != list(COMPLETE_GRID) or any([p["percent"] for p in c["points"]] != grid for c in curves):
            raise ValueError("Complete refinement analysis requires every curve on the twelve-coordinate common grid")
        if result["missing_run_names"] or result["verified_but_unplotted"]:
            raise ValueError("Complete refinement analysis cannot omit a verified artifact")
        result["complete_common_grid_analysis"] = {
            "pilot_plan_sha256": pilot["plan_sha256"],
            "refinement_plan_sha256": refinement["plan_sha256"],
            "stage": "pilot_plus_refinement",
            "verified_artifact_count": len(runs),
            **analysis.analyze_common_grid(runs, values, grid),
        }
        result["scope"] = (
            "Measured selections on the complete twelve-coordinate common grid from 57 pilot and 45 refinement "
            "artifacts. Trainer seeds are averaged within each subset; target regrets share one target trainer seed."
        )
    return result


def plot(result: dict, output: Path) -> None:
    """Preserve the pilot palette and show measured points with straight segments."""
    definition = result.get("metric_definition", {})
    if definition.get("id") != METRIC_DEFINITION or definition.get("schema_version") != 2:
        raise ValueError("Refusing to plot BPB without a verified common metric definition")
    colors = {"target": "#333333", "unmatched": "#0072B2", "matched": "#D55E00"}
    labels = {"target": "Target", "unmatched": "Unmatched proxy", "matched": "Matched subsets (3)"}
    with plt.rc_context({"text.usetex": False}):
        figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
        for axis, excess in zip(axes, (False, True), strict=True):
            for i, curve in enumerate(result["curves"]):
                x = np.array([p["percent"] for p in curve["points"]]) / 100
                y = np.array([p["value"] for p in curve["points"]])
                if excess:
                    y = y - y.min()
                arm = curve["arm"]
                matched = arm == "matched"
                axis.plot(
                    x,
                    y,
                    marker="." if matched else "o",
                    markersize=4 if matched else 3,
                    color=colors[arm],
                    alpha=0.65 if matched else 1,
                    linewidth=1.2 if matched else 1.5,
                    label=labels[arm] if i < 3 else None,
                )
                j = int(np.argmin(y))
                axis.plot(x[j], y[j], "*", markersize=8 if matched else 10, color=colors[arm])
            axis.set(
                xlabel="StarCoder token fraction, p",
                ylabel="BPB above own observed minimum" if excess else "Programming-languages BPB",
            )
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(alpha=0.2)
        axes[1].legend(fontsize=8)
        figure.savefig(output / "curves.pdf")
        figure.savefig(output / "curves.png", dpi=180)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-plan", type=Path, required=True)
    parser.add_argument("--pilot-metrics", type=Path, required=True)
    parser.add_argument("--refinement-plan", type=Path, required=True)
    parser.add_argument(
        "--metric-records", type=Path, required=True, help="Audited final JSONL manifest with source hashes"
    )
    parser.add_argument("--population-counts", type=Path, required=True, help="Frozen scored-token and byte population")
    parser.add_argument("--output", type=Path, required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--refresh", action="store_true", help="Recheck raw endpoints against GCS and W&B")
    source_group.add_argument("--snapshot", type=Path, help="Existing verified endpoint snapshot; read without changes")
    args = parser.parse_args()
    pilot = json.loads(args.pilot_plan.read_text())
    refinement = json.loads(args.refinement_plan.read_text())
    experiment.validate_plan(pilot)
    experiment.validate_plan(refinement)
    if refinement["plan_sha256"] != REFINEMENT_SHA256 or refinement["pilot_plan_sha256"] != pilot["plan_sha256"]:
        raise ValueError("Expected the archived pilot and approved five-coordinate refinement")
    if sorted({r["percent"] for r in refinement["runs"]}) != list(REFINEMENT_GRID) or len(refinement["runs"]) != 45:
        raise ValueError("Unexpected refinement grid or run count")
    for key in ("design_sha256", "primary_metric", "runtime_versions", "code_sha256"):
        if pilot[key] != refinement[key]:
            raise ValueError(f"Pilot and refinement disagree on {key}")
    pilot_values = analysis.verified_measurements(pilot, args.pilot_metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.snapshot if args.snapshot is not None else args.output / "verified_endpoints.json"
    if args.refresh:
        with ThreadPoolExecutor(max_workers=4) as pool:
            endpoints = list(pool.map(lambda request: collect_endpoint(refinement, request), refinement["runs"]))
        snapshot = {
            "checked_at": datetime.now(UTC).isoformat(),
            "plan_sha256": refinement["plan_sha256"],
            "rows": endpoints,
        }
        snapshot_path.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot["plan_sha256"] != refinement["plan_sha256"] or len(snapshot["rows"]) != len(refinement["runs"]):
        raise ValueError("Snapshot does not match the refinement plan")
    population_counts = json.loads(args.population_counts.read_text())
    if canonical_sha256(population_counts) != POPULATION_COUNTS_SHA256:
        raise ValueError("Population counts differ from the audited frozen evaluation population")
    population = population_counts[experiment.PRIMARY_METRIC.removeprefix("eval/").removesuffix("/bpb")]
    completed = [row for row in snapshot["rows"] if row["artifact_status"] == STATUS_SUCCESS]
    raw_values = {**pilot_values, **{row["run_name"]: row["value"] for row in completed}}
    completed_names = {row["run_name"] for row in completed}
    runs = pilot["runs"] + [row for row in refinement["runs"] if row["run_name"] in completed_names]
    normalized, provenance = normalize_final_metrics(runs, raw_values, args.metric_records, population)
    normalized_pilot = {name: normalized[name] for name in pilot_values}
    normalized_endpoints = [
        {**row, "value": normalized[row["run_name"]]} if row["artifact_status"] == STATUS_SUCCESS else row
        for row in snapshot["rows"]
    ]
    result = summarize(pilot, refinement, normalized_pilot, normalized_endpoints)
    result["metric_definition"] = {
        "id": METRIC_DEFINITION,
        "schema_version": 2,
        "metric": experiment.PRIMARY_METRIC,
        "formula": "token_average_loss * scored_tokens / (scored_bytes * ln(2))",
        "population": population,
        "population_counts_sha256": POPULATION_COUNTS_SHA256,
        "total_records": len(provenance),
        "reported_schema_counts": {
            "legacy_unversioned": sum(row["reported_schema"] is None for row in provenance),
            "2": sum(row["reported_schema"] == 2 for row in provenance),
        },
        "normalization_source_sha256": {
            str(args.metric_records.resolve()): file_sha256(args.metric_records),
            str(args.population_counts.resolve()): file_sha256(args.population_counts),
        },
        "schema2_reconstruction_tolerance_bpb": SCHEMA2_RECONSTRUCTION_TOLERANCE,
    }
    provenance_path = args.output / "metric_provenance.json"
    provenance_path.write_text(
        json.dumps({"metric_definition": result["metric_definition"], "rows": provenance}, indent=2) + "\n"
    )
    result["sources_sha256"] = {
        str(path.resolve()): file_sha256(path)
        for path in (
            args.pilot_plan,
            args.pilot_metrics,
            args.refinement_plan,
            snapshot_path,
            args.metric_records,
            args.population_counts,
            provenance_path,
            Path(__file__),
            Path(analysis.__file__),
        )
    }
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (args.output / "plotted_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("arm", "subset_seed", "percent", "value", "seed_count"))
        writer.writeheader()
        for curve in result["curves"]:
            for point in curve["points"]:
                writer.writerow(
                    {
                        "arm": curve["arm"],
                        "subset_seed": curve["subset_seed"],
                        "percent": point["percent"],
                        "value": point["value"],
                        "seed_count": len(point["seed_values"]),
                    }
                )
    plot(result, args.output)
    print(
        json.dumps(
            {
                "refinement_complete": result["refinement_complete"],
                "minima": [(c["arm"], c["subset_seed"], c["selected_percent"]) for c in result["curves"]],
                "missing": result["missing_run_names"],
                "unplotted": result["verified_but_unplotted"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
