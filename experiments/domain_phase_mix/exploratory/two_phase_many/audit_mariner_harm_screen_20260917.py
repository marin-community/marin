# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["joblib", "numpy", "pandas", "scipy"]
# ///

"""Independently audit saved harm-screen shards without repeating grid fits.

Run in the repository's pinned environment with ``uv run --offline --no-sync
python -m experiments.domain_phase_mix.exploratory.two_phase_many.
audit_mariner_harm_screen_20260917 --tier screen``. The audit shares input and
split definitions, but does not use the runner's collection, aggregation, metric,
or CV routines. Head replays solve only the saved shape/ridge, not a shape grid.
"""

import argparse
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    screen_mariner_harm_forms_20260917 as runner,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

PREDICTION_KEYS = ["model", "panel", "target", "repeat", "fold", "row_index"]
GROUP_KEYS = ["model", "panel", "target", "repeat"]
METRICS = ("rmse", "mae", "spearman", "regret_at_1", "regret_at_top_k", "selection_optimism", "n")


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def assert_close(actual: np.ndarray | float, expected: np.ndarray | float, label: str) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10, equal_nan=True, err_msg=label)


def audit_sources(output: Path, destination: Path, manifest: dict) -> dict:
    """Verify frozen identities and capture omitted loader sources separately."""
    for filename, expected in manifest["sources"].items():
        path = Path(filename)
        require(checksum(path) == expected, f"Current source differs from registration: {path}")
        snapshot = output / "source_snapshot" / path.name
        require(checksum(snapshot) == expected, f"Snapshot differs from registration: {snapshot}")
    for name, expected in manifest["dependencies"].items():
        require(importlib.metadata.version(name) == expected, f"Dependency version changed: {name}")
    panels = tuple(bench.load_panel(name) for name in bench.THIRTY_NINE_BUCKET_PANELS)
    require(
        bench.manifest_hash(bench.split_manifest(panels, 1)) == manifest["split_hash"],
        "Realized split manifest changed",
    )
    for panel in panels:
        require(panel.input_hashes == manifest["panel_inputs"][panel.name], f"Inputs changed: {panel.name}")
    supplemental = destination / "supplemental_source_snapshot"
    supplemental.mkdir(exist_ok=True)
    captured = datetime.datetime.now(datetime.UTC).isoformat()
    registered = (output / "screen_manifest.json").stat().st_mtime
    records = []
    for module in (bench.olmix_benchmark, bench.starcoder_curves):
        path = Path(module.__file__).resolve()
        target = supplemental / path.name
        source_hash = checksum(path)
        if target.exists():
            require(checksum(target) == source_hash, f"Supplemental loader changed after audit capture: {path}")
        else:
            shutil.copyfile(path, target)
        records.append(
            {
                "path": str(path),
                "sha256": source_hash,
                "mtime_utc": datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.UTC).isoformat(),
                "mtime_precedes_registration": path.stat().st_mtime <= registered,
                "captured_utc": captured,
            }
        )
    supplemental_manifest = destination / "supplemental_sources.json"
    if not supplemental_manifest.exists():
        supplemental_manifest.write_text(json.dumps(records, indent=2) + "\n")
    return {"registered_sources": len(manifest["sources"]), "supplemental_loaders": records}


def component_model(
    task: bench.FitTask, panel: bench.BenchPanel
) -> tuple[models.Features, models.FittedFloorModel | models.GridModel]:
    entry = registry.ENTRY_BY_ID[task.model_id]
    features = dataclasses.replace(registry.apply_transform(panel.features, entry), component=task.component)
    return features, entry.build(features)


def check_selected_grid(task: bench.FitTask, panel: bench.BenchPanel, saved: dict) -> None:
    features, model = component_model(task, panel)
    grid = model.base if isinstance(model, models.FittedFloorModel) else model
    require(isinstance(grid, models.GridModel) and not grid.refine, "Audit expects unrefined fixed-basis grids")
    require(not isinstance(model, models.FittedFloorModel) or model.joint_rounds == 0, "Joint rounds unsupported")
    shapes = grid.candidate_shapes(features)
    table = saved["cv_table"]
    require(table.shape == (len(shapes), len(grid.ridge_grid)), f"Wrong CV grid dimensions: {task}")
    require(np.isfinite(table).any(), f"No finite CV score: {task}")
    shape_index, ridge_index = np.unravel_index(np.argmin(table), table.shape)
    require(json.loads(str(saved["shape_json"])) == shapes[shape_index], f"Saved shape is not CV argmin: {task}")
    require(float(saved["ridge"]) == grid.ridge_grid[ridge_index], f"Saved ridge is not CV argmin: {task}")
    diagnostics = json.loads(str(saved["diagnostics_json"]))
    selection_score = diagnostics.get("selection_inner_cv_rmse", diagnostics["inner_cv_rmse"])
    assert_close(selection_score, table[shape_index, ridge_index], f"CV table/diagnostic mismatch: {task}")


def score_saved_head(
    design: models.Design, response: np.ndarray, ridge: float, spec: models.HeadSpec, inner: models.InnerFolds
) -> float:
    errors = []
    for train, test in inner:
        train_design = models.Design(design.values[train], design.ridge, design.names)
        head = models.fit_head(train_design, response[train], ridge, spec)
        errors.append(models.predict_head(head, design.values[test], spec) - response[test])
    return float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))


def replay_selected(task: bench.FitTask, panel: bench.BenchPanel, split: bench.Split, saved: dict) -> dict:
    """Rebuild the saved basis and head; recompute both prior-floor and selected-floor CV."""
    features, model = component_model(task, panel)
    grid = model.base if isinstance(model, models.FittedFloorModel) else model
    shape = json.loads(str(saved["shape_json"]))
    ridge = float(saved["ridge"])
    diagnostics = json.loads(str(saved["diagnostics_json"]))
    design = grid.design(features, shape)
    response = panel.group(task.target).outcomes[:, task.component_index]
    initial_spec = grid.head_for(shape)
    initial_cv = score_saved_head(design, response, ridge, initial_spec, split.inner)
    expected_initial = diagnostics.get("selection_inner_cv_rmse", diagnostics["inner_cv_rmse"])
    assert_close(initial_cv, expected_initial, f"Replayed selected-grid CV: {task}")
    final_spec = dataclasses.replace(initial_spec, link=models.LinkKind(diagnostics["link"]))
    if isinstance(model, models.FittedFloorModel):
        kappa = float(diagnostics["kappa"])
        final_spec = dataclasses.replace(final_spec, floor_kappa_prior=kappa, floor_kappa_bounds=(kappa, kappa))
    final_cv = score_saved_head(design, response, ridge, final_spec, split.inner)
    assert_close(final_cv, diagnostics["inner_cv_rmse"], f"Replayed final CV: {task}")
    head = models.fit_head(
        models.Design(design.values[split.train], design.ridge, design.names), response[split.train], ridge, final_spec
    )
    prediction = models.predict_head(head, design.values[split.test], final_spec)
    train_prediction = models.predict_head(head, design.values[split.train], final_spec)
    assert_close(prediction, saved["prediction"], f"Replayed test predictions: {task}")
    assert_close(train_prediction, saved["train_prediction"], f"Replayed train predictions: {task}")
    return dataclasses.asdict(task) | {
        "selected_grid_cv_rmse": initial_cv,
        "final_cv_rmse": final_cv,
        "max_prediction_difference": float(np.max(np.abs(prediction - saved["prediction"]))),
        "heads_refitted": 2 * len(split.inner) + 1,
    }


def read_predictions(tasks: list[bench.FitTask], output: Path, manifest: dict, replays_per_model: int) -> tuple:
    rows, replays, incomplete = [], [], []
    replay_counts: dict[str, int] = {}
    completed = 0
    for task in tasks:
        path = bench.shard_path(output, task)
        if not path.exists():
            incomplete.append(dataclasses.asdict(task) | {"status": "missing"})
            continue
        with np.load(path, allow_pickle=False) as archive:
            saved = {key: archive[key] for key in archive.files}
        if str(saved["status"]) != "ok":
            incomplete.append(dataclasses.asdict(task) | {"status": str(saved["status"])})
            continue
        panel = bench.load_panel(task.panel)
        split = next(
            s for s in bench.panel_splits(panel, task.repeat + 1) if (s.repeat, s.fold) == (task.repeat, task.fold)
        )
        expected_protocol = bench.task_protocol_hashes(
            task, registry.ENTRY_BY_ID[task.model_id], manifest["split_hash"], panel
        )[0]
        require(str(saved["protocol_hash"]) == expected_protocol, f"Wrong primary protocol: {path}")
        require(
            str(saved["model_id"]) == task.model_id and str(saved["component"]) == task.component,
            f"Wrong identity: {path}",
        )
        require(int(saved["component_index"]) == task.component_index, f"Wrong component index: {path}")
        require(np.array_equal(saved["test"], split.test), f"Wrong test rows: {path}")
        require(int(saved["train_rows"]) == len(split.train), f"Wrong train row count: {path}")
        require(
            saved["prediction"].shape == split.test.shape and np.isfinite(saved["prediction"]).all(),
            f"Invalid predictions: {path}",
        )
        response = panel.group(task.target).outcomes[:, task.component_index]
        require(np.array_equal(saved["observed"], response[split.test]), f"Wrong observed outcomes: {path}")
        check_selected_grid(task, panel, saved)
        if replay_counts.get(task.model_id, 0) < replays_per_model:
            replays.append(replay_selected(task, panel, split, saved))
            replay_counts[task.model_id] = replay_counts.get(task.model_id, 0) + 1
        completed += 1
        for local, row_index in enumerate(split.test):
            rows.append(
                {
                    "model": task.model_id,
                    "panel": task.panel,
                    "target": task.target,
                    "repeat": task.repeat,
                    "fold": task.fold,
                    "row_index": int(row_index),
                    "component_index": task.component_index,
                    "component": task.component,
                    "observed": float(saved["observed"][local]),
                    "prediction": float(saved["prediction"][local]),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        require(not frame.duplicated([*PREDICTION_KEYS, "component_index"]).any(), "Duplicate component identities")
    return frame, completed, incomplete, replays


def reconstruct_aggregates(predictions: pd.DataFrame, tasks: list[bench.FitTask]) -> pd.DataFrame:
    """Sum all objective components only for groups with exact, complete expected coverage."""
    aggregate_rows = []
    if predictions.empty:
        return pd.DataFrame()
    for identity, frame in predictions.groupby(GROUP_KEYS, sort=False):
        _model_id, panel_name, target, repeat = identity
        panel = bench.load_panel(panel_name)
        group = panel.group(target)
        folds = {task.fold for task in tasks if (task.model_id, task.panel, task.target, task.repeat) == identity}
        expected_rows = sorted(
            int(row)
            for split in bench.panel_splits(panel, repeat + 1)
            if split.repeat == repeat and split.fold in folds
            for row in split.test
        )
        expected = {(row, component) for row in expected_rows for component in range(len(group.components))}
        actual = set(frame[["row_index", "component_index"]].itertuples(index=False, name=None))
        if actual != expected:
            continue
        for row, component_rows in frame.groupby("row_index", sort=True):
            require(component_rows["fold"].nunique() == 1, f"Components use different folds: {identity}, {row}")
            ordered = component_rows.sort_values("component_index")
            prediction = sum(
                float(value) * float(weight)
                for value, weight in zip(ordered["prediction"], group.aggregation_weights, strict=True)
            )
            reconstructed_observed = sum(
                float(value) * float(weight)
                for value, weight in zip(ordered["observed"], group.aggregation_weights, strict=True)
            )
            # Frozen micro-BPB weights reconstruct the evaluator labels only to the
            # documented loader tolerance; retain the published labels for regret.
            observed = float(group.aggregate[int(row)])
            reconstruction_error = reconstructed_observed - observed
            require(
                abs(reconstruction_error) <= bench.canonical.RECONSTRUCTION_TOLERANCE,
                f"Objective observed aggregation differs from its frozen rule: {identity}",
            )
            aggregate_rows.append(
                dict(zip(GROUP_KEYS, identity, strict=True))
                | {
                    "fold": int(ordered["fold"].iloc[0]),
                    "row_index": int(row),
                    "observed": observed,
                    "prediction": prediction,
                    "component_reconstruction_error": reconstruction_error,
                }
            )
    return pd.DataFrame(aggregate_rows)


def independent_metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for identity, group in frame.groupby(keys, sort=False):
        observed, predicted = group["observed"].to_numpy(), group["prediction"].to_numpy()
        order = np.argsort(predicted, kind="stable")
        correlation = (
            float("nan")
            if len(group) < 3 or np.ptp(observed) <= 1e-12 or np.ptp(predicted) <= 1e-12
            else float(stats.spearmanr(observed, predicted).statistic)
        )
        rows.append(
            dict(zip(keys, identity, strict=True))
            | {
                "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
                "mae": float(np.mean(np.abs(predicted - observed))),
                "spearman": correlation,
                "regret_at_1": float(observed[order[0]] - observed.min()),
                "regret_at_top_k": float(observed[order[:5]].min() - observed.min()),
                "selection_optimism": float(observed[order[0]] - predicted[order[0]]),
                "n": len(group),
            }
        )
    return pd.DataFrame(rows)


def compare_csv(path: Path, reconstructed: pd.DataFrame, keys: list[str], columns: list[str]) -> bool:
    if not path.exists():
        return False
    reported = pd.read_csv(path)
    reported = reported[reported["model"].isin(reconstructed["model"].unique())].set_index(keys).sort_index()
    expected = reconstructed.set_index(keys).sort_index()
    require(reported.index.equals(expected.index), f"Report identities differ: {path}")
    assert_close(reported[columns].to_numpy(), expected[columns].to_numpy(), f"Report values: {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=runner.DEFAULT_OUTPUT)
    parser.add_argument("--tier", choices=("smoke", "screen", "certify"), required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--replays-per-model", type=int, default=1)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    require(args.replays_per_model >= 0, "Replay count must be nonnegative")
    manifest = json.loads((args.output / "screen_manifest.json").read_text())
    ids = tuple(args.models or [spec["model_id"] for spec in manifest["candidates"]])
    require(len(ids) == len(set(ids)), "Model IDs must be unique")
    runner.register_candidates()
    destination = args.output / f"independent_audit_{args.tier}"
    destination.mkdir(exist_ok=True)
    sources = audit_sources(args.output, destination, manifest)
    plan = dataclasses.replace(bench.tier_plan(args.tier, panels=bench.THIRTY_NINE_BUCKET_PANELS), curves=())
    tasks = bench.plan_tasks(plan, ids)
    predictions, completed, incomplete, replays = read_predictions(tasks, args.output, manifest, args.replays_per_model)
    aggregate = reconstruct_aggregates(predictions, tasks)
    comparisons = {}
    report = args.output / f"report_{args.tier}"
    if not incomplete and not predictions.empty:
        comparisons["component_predictions"] = compare_csv(
            report / "component_predictions.csv",
            predictions,
            [*PREDICTION_KEYS, "component_index"],
            ["observed", "prediction"],
        )
        for suffix, keys in (
            ("metrics", [*GROUP_KEYS, "component"]),
            ("fold_metrics", [*GROUP_KEYS, "component", "fold"]),
        ):
            values = independent_metrics(predictions, keys)
            values.to_csv(destination / f"component_{suffix}.csv", index=False)
            comparisons[f"component_{suffix}"] = compare_csv(
                report / f"component_{suffix}.csv", values, keys, list(METRICS)
            )
    if not aggregate.empty:
        aggregate.to_csv(destination / "aggregate_predictions.csv", index=False)
        if not incomplete:
            comparisons["aggregate_predictions"] = compare_csv(
                report / "aggregate_predictions.csv", aggregate, PREDICTION_KEYS, ["observed", "prediction"]
            )
        for suffix, keys in (("metrics", GROUP_KEYS), ("fold_metrics", [*GROUP_KEYS, "fold"])):
            values = independent_metrics(aggregate, keys)
            values.to_csv(destination / f"aggregate_{suffix}.csv", index=False)
            if not incomplete:
                comparisons[f"aggregate_{suffix}"] = compare_csv(
                    report / f"aggregate_{suffix}.csv", values, keys, list(METRICS)
                )
    checks = {
        "audited_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "audit_source_sha256": checksum(Path(__file__)),
        "models": ids,
        "tier": args.tier,
        "expected_fits": len(tasks),
        "completed_fits": completed,
        "incomplete": incomplete,
        "complete_exact_coverage": not incomplete,
        "component_prediction_rows": len(predictions),
        "aggregate_prediction_rows": len(aggregate),
        "selected_grid_argmins_verified": completed,
        "sampled_selected_head_replays": replays,
        "report_comparisons": comparisons,
        "source_checks": sources,
        "scope": (
            "Archived OOF menu diagnostics; screen anchors do not reconstruct either full objective. "
            "Head replay is conditional on saved shape/ridge/kappa, not an independent full hyperparameter search."
        ),
    }
    (destination / "audit.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: checks[key]
                for key in (
                    "expected_fits",
                    "completed_fits",
                    "component_prediction_rows",
                    "aggregate_prediction_rows",
                    "complete_exact_coverage",
                    "report_comparisons",
                )
            }
        ),
        flush=True,
    )
    require(args.allow_incomplete or not incomplete, f"Incomplete screen: {len(incomplete)} missing or failed fits")


if __name__ == "__main__":
    main()
