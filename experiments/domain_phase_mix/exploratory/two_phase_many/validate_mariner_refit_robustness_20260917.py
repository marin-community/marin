# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["joblib", "numpy", "pandas", "scipy"]
# ///

"""Frozen repeated-CV and regularization sensitivity for three MARINER harms.

Run as a module in the pinned repository environment. Saved final heads can be
loaded with ``load_fitted(task_head_path(output, task))`` and passed to the
matching registry model's predict method. No training jobs are submitted.
"""

import argparse
import dataclasses
import functools
import importlib
import importlib.metadata
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    audit_mariner_harm_screen_20260917 as audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_harm_screen_models_20260917 as candidates,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    screen_mariner_harm_forms_20260917 as screen,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "reference_outputs/mariner_local_validation_20260917/refits"
DISCOVERY = HERE / "reference_outputs/mariner_harm_screen_20260917/fixed_basis"
MODEL_IDS = ("hs17_mariner", "hs17_log1", "hs17_bounded_exp_c1")
REPEATS = 3
SENSITIVITIES = ("rms", "broad_ridge")
BROAD_RIDGE = (0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
KEYS = ["model", "panel", "target", "repeat", "fold", "row_index"]


def build_sensitivity(spec: candidates.CandidateSpec, setting: str, features: models.Features):
    """Change one regularization choice while preserving nested floor fitting."""
    base = candidates.build_model(spec, features)
    if not isinstance(base, models.FittedFloorModel) or not isinstance(base.base, models.GridModel):
        raise TypeError("Repeated validation requires a fitted-floor grid model")
    grid = base.base
    if setting == "rms":
        grid = dataclasses.replace(grid, head=dataclasses.replace(grid.head, scale_columns=True, scale_rule="rms"))
    elif setting == "broad_ridge":
        grid = dataclasses.replace(grid, ridge_grid=BROAD_RIDGE)
    else:
        raise ValueError(f"Unknown sensitivity: {setting}")
    return dataclasses.replace(base, base=dataclasses.replace(grid, model_id=f"{spec.model_id}__{setting}"))


def register_models() -> None:
    screen.register_candidates()
    specs = {spec.model_id: spec for spec in candidates.candidate_specs()}
    for model_id in MODEL_IDS:
        for setting in SENSITIVITIES:
            registry.ENTRY_BY_ID[f"{model_id}__{setting}"] = dataclasses.replace(
                registry.ENTRY_BY_ID[model_id],
                model_id=f"{model_id}__{setting}",
                parent=model_id,
                build=functools.partial(build_sensitivity, specs[model_id], setting),
                note=f"{setting}; one change to regularization, fixed candidate equation",
            )


def task_head_path(output: Path, task: bench.FitTask) -> Path:
    return output / "heads" / bench.shard_path(output, task).relative_to(output / "shards")


def load_fitted(path: Path) -> models.Fitted:
    """Restore a portable numeric final head without executing a pickle."""
    saved = bench.load_shard(path)
    if saved is None:
        raise FileNotFoundError(path)
    head = models.FittedHead(
        intercept=float(saved["head_intercept"]),
        coefficients=saved["head_coefficients"],
        floor=float(saved["head_floor"]),
        active=int(saved["head_active"]),
        cap=float(saved["head_cap"]),
        kappa=float(saved["head_kappa"]),
        converged=bool(saved["head_converged"]),
    )
    return models.Fitted(
        shape=json.loads(str(saved["shape_json"])),
        ridge=float(saved["ridge"]),
        head=head,
        diagnostics=json.loads(str(saved["diagnostics_json"])),
    )


def reconstruct_head(task: bench.FitTask, output: Path, saved: dict) -> None:
    """Recover the selected final head and verify both training and OOF predictions."""
    panel = bench.load_panel(task.panel)
    features, model = audit.component_model(task, panel)
    if not isinstance(model, models.FittedFloorModel) or not isinstance(model.base, models.GridModel):
        raise TypeError("Expected fitted-floor grid")
    split = next(s for s in bench.panel_splits(panel, task.repeat + 1) if (s.repeat, s.fold) == (task.repeat, task.fold))
    path = task_head_path(output, task)
    previous = bench.load_shard(path)
    if previous is not None and str(previous["prediction_shard_sha256"]) == screen.digest(
        bench.shard_path(output, task)
    ):
        fitted = load_fitted(path)
    else:
        shape = json.loads(str(saved["shape_json"]))
        diagnostics = json.loads(str(saved["diagnostics_json"]))
        spec = model._spec(shape, models.LinkKind(diagnostics["link"]), float(diagnostics["kappa"]))
        design = model.base.design(features, shape)
        response = panel.group(task.target).outcomes[:, task.component_index]
        head = models.fit_head(
            models.Design(design.values[split.train], design.ridge, design.names),
            response[split.train],
            float(saved["ridge"]),
            spec,
        )
        fitted = models.Fitted(shape, float(saved["ridge"]), head, diagnostics)
        payload = {f"head_{key}": value for key, value in dataclasses.asdict(head).items()}
        payload.update(
            shape_json=saved["shape_json"],
            ridge=saved["ridge"],
            diagnostics_json=saved["diagnostics_json"],
            prediction_shard_sha256=screen.digest(bench.shard_path(output, task)),
            train=split.train,
            test=split.test,
        )
        bench.atomic_save(path, payload)
    np.testing.assert_allclose(model.predict(fitted, features, split.test), saved["prediction"], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        model.predict(fitted, features, split.train), saved["train_prediction"], rtol=1e-9, atol=1e-10
    )


def source_paths() -> tuple[Path, ...]:
    return (*screen.source_paths(), Path(__file__), Path(audit.__file__), Path(bench.olmix_benchmark.__file__))


def prepare(output: Path) -> dict:
    """Freeze all choices and identities before inspecting the new scores."""
    old = json.loads((DISCOVERY / "screen_manifest.json").read_text())
    for path, digest in old["sources"].items():
        if screen.digest(Path(path)) != digest:
            raise ValueError(f"Discovery source changed; reuse is forbidden: {path}")
    panels = tuple(bench.load_panel(name) for name in bench.THIRTY_NINE_BUCKET_PANELS)
    if {p.name: p.input_hashes for p in panels} != old["panel_inputs"]:
        raise ValueError("Discovery panel inputs changed")
    if bench.manifest_hash(bench.split_manifest(panels, 1)) != old["split_hash"]:
        raise ValueError("Discovery repeat-zero partition changed")
    current = {
        "version": 1,
        "models": MODEL_IDS,
        "repeats": REPEATS,
        "sources": {str(path.resolve()): screen.digest(path) for path in source_paths()},
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "joblib")},
        "panel_inputs": {p.name: p.input_hashes for p in panels},
        "split_hash": bench.manifest_hash(bench.split_manifest(panels, REPEATS)),
        "old_manifest_hash": screen.digest(DISCOVERY / "screen_manifest.json"),
        "main": "All 58 components, three panels, five grouped folds, repeats 0,1,2; reuse verified repeat zero",
        "sensitivity": {
            "components": bench._anchor_components(),
            "repeats": [0],
            "settings": {"rms": "scale_columns=True only", "broad_ridge": BROAD_RIDGE},
            "baseline": "unscaled columns, original ridge grid",
        },
        "interpretation": (
            "Three paired partition results are descriptive dispersion, not independent-fold confidence intervals"
        ),
    }
    current = json.loads(json.dumps(current))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != current:
            raise ValueError("Validation source/input/configuration changed; use a new identity")
        return current
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    for source in source_paths():
        shutil.copyfile(source, snapshots / source.name)
    bench.split_manifest(panels, REPEATS).to_csv(output / "splits.csv", index=False)
    return current


@functools.cache
def initialize_worker(output: Path) -> dict:
    manifest = json.loads((output / "protocol.json").read_text())
    for path, digest in manifest["sources"].items():
        if screen.digest(Path(path)) != digest:
            raise ValueError(f"Source changed during validation: {path}")
    register_models()
    return manifest


def fit_task(task: bench.FitTask, output: Path) -> str:
    manifest = initialize_worker(output)
    panel = bench.load_panel(task.panel)
    split = next(s for s in bench.panel_splits(panel, task.repeat + 1) if (s.repeat, s.fold) == (task.repeat, task.fold))
    path = bench.shard_path(output, task)
    hashes = bench.task_protocol_hashes(task, registry.ENTRY_BY_ID[task.model_id], manifest["split_hash"], panel, output)
    reused = False
    if task.repeat == 0 and task.model_id in MODEL_IDS and not path.exists():
        source = bench.shard_path(DISCOVERY, task)
        if not bench.valid_shard(source, hashes, task.component, split.test):
            raise ValueError(f"Repeat-zero source is invalid: {source}")
        saved = bench.load_shard(source)
        if saved is None or str(saved["status"]) != "ok" or not np.isfinite(saved["prediction"]).all():
            raise ValueError(f"Repeat-zero source failed: {source}")
        bench.atomic_save(path, saved)
        reused = True
    result = bench.fit_one(task, output, manifest["split_hash"])
    saved = bench.load_shard(path)
    if saved is None or str(saved["status"]) != "ok":
        raise RuntimeError(f"Fit failed: {path}")
    audit.check_selected_grid(task, panel, saved)
    reconstruct_head(task, output, saved)
    return "reused" if reused else result


def planned_tasks(phase: str) -> list[bench.FitTask]:
    plan = dataclasses.replace(
        bench.tier_plan("certify", panels=bench.THIRTY_NINE_BUCKET_PANELS), repeats=REPEATS, curves=()
    )
    main = bench.plan_tasks(plan, MODEL_IDS)
    anchors = dataclasses.replace(bench.tier_plan("screen", panels=bench.THIRTY_NINE_BUCKET_PANELS), curves=())
    sensitivity = bench.plan_tasks(
        anchors, tuple(f"{model}__{setting}" for setting in SENSITIVITIES for model in MODEL_IDS)
    )
    result = main if phase == "main" else sensitivity if phase == "sensitivity" else main + sensitivity
    result.sort(key=lambda t: (t.repeat, t.fold, t.component_index, t.panel, t.target, t.model_id))
    return result


def independent_aggregate(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by explicit objective weights, separately from benchmark helpers."""
    rows = []
    for (model, panel_name, target, repeat, fold), frame in predictions.groupby(KEYS[:-1], sort=False):
        panel = bench.load_panel(panel_name)
        group = panel.group(target)
        if frame.component.nunique() != len(group.components):
            continue
        for row_index, row in frame.groupby("row_index", sort=False):
            ordered = row.set_index("component").loc[list(group.components)]
            rows.append(
                dict(
                    zip(KEYS, (model, panel_name, target, repeat, fold, row_index), strict=True),
                    prediction=float(np.dot(group.aggregation_weights, ordered.prediction)),
                    observed=float(group.aggregate[int(row_index)]),
                    component_sum_observed=float(np.dot(group.aggregation_weights, ordered.observed)),
                )
            )
    return pd.DataFrame(rows)


def paired_dispersion(aggregate: pd.DataFrame, destination: Path) -> pd.DataFrame:
    records = []
    for keys, frame in aggregate.groupby(["model", "panel", "target", "repeat"]):
        regrets = []
        for _, fold in frame.groupby("fold"):
            selected = fold.loc[fold.prediction.idxmin()]
            regrets.append(float(selected.observed - fold.observed.min()))
        records.append(
            dict(
                zip(["model", "panel", "target", "repeat"], keys, strict=True),
                spearman=float(stats.spearmanr(frame.observed, frame.prediction).statistic),
                rmse=float(np.sqrt(np.mean((frame.prediction - frame.observed) ** 2))),
                mean_fold_regret=float(np.mean(regrets)),
            )
        )
    table = pd.DataFrame(records)
    keys = ["panel", "target", "repeat"]
    baseline = table[table.model.eq(MODEL_IDS[0])].set_index(keys)
    paired = table.join(baseline[["spearman", "rmse", "mean_fold_regret"]].add_prefix("baseline_"), on=keys)
    for metric in ("spearman", "rmse", "mean_fold_regret"):
        paired[f"delta_{metric}"] = paired[metric] - paired[f"baseline_{metric}"]
    paired.to_csv(destination / "paired_partition_metrics.csv", index=False)
    paired.groupby(["model", "panel", "target"])[["delta_spearman", "delta_rmse", "delta_mean_fold_regret"]].agg(
        ["mean", "min", "max"]
    ).to_csv(destination / "partition_dispersion.csv")
    return paired


def report(tasks: list[bench.FitTask], output: Path, manifest: dict, phase: str) -> None:
    destination = output / f"report_{phase}"
    destination.mkdir(exist_ok=True)
    predictions, fits = bench.collect_shards(tasks, output, manifest["split_hash"])
    checks = screen.verify_coverage(tasks, predictions, fits)
    fits.to_csv(destination / "fits.csv", index=False)
    predictions.to_csv(destination / "component_predictions.csv", index=False)
    component_keys = ["model", "panel", "target", "component", "repeat"]
    cm = bench.grouped_metrics(predictions, component_keys)
    cm.to_csv(destination / "component_metrics.csv", index=False)
    baseline = cm[cm.model.eq(MODEL_IDS[0])].set_index(component_keys[1:])
    decomposition = cm.join(baseline[["spearman", "rmse"]].add_prefix("baseline_"), on=component_keys[1:])
    decomposition["delta_spearman"] = decomposition.spearman - decomposition.baseline_spearman
    decomposition["delta_rmse"] = decomposition.rmse - decomposition.baseline_rmse
    decomposition.to_csv(destination / "component_decomposition.csv", index=False)
    aggregate = bench.aggregate_predictions(predictions)
    independent = independent_aggregate(predictions)
    if len(aggregate):
        actual = aggregate.set_index(KEYS).sort_index()[["prediction", "observed"]]
        expected = independent.set_index(KEYS).sort_index()[["prediction", "observed"]]
        pd.testing.assert_index_equal(actual.index, expected.index)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        aggregate.to_csv(destination / "aggregate_predictions.csv", index=False)
        paired = paired_dispersion(independent, destination)
        checks["aggregate_predictions"] = len(aggregate)
        checks["independent_aggregation"] = "passed"
        checks["max_component_sum_vs_stored_observation"] = float(
            np.max(np.abs(independent.component_sum_observed - independent.observed))
        )
        text = [
            "# Repeated partition and regularization validation",
            "",
            "Three predeclared grouped outer partitions, with shape, ridge and floor fitted on each outer "
            "training set using its inner folds.",
            "Repeat-zero predictions are source/input/split verified reuse from the discovery screen. "
            "All saved final heads reproduce training and out-of-fold predictions.",
            "Partition ranges below describe sensitivity to split assignment; the partitions share observations "
            "and are not independent replications. No iid-fold confidence interval is claimed.",
            "",
            paired.round(6).to_markdown(index=False),
            "",
            "The main comparison uses the same archived panels used to shortlist the forms. This is development "
            "robustness, not independent confirmation or performance of a newly trained optimized mixture.",
        ]
        (destination / "README.md").write_text("\n".join(text) + "\n")
    checks["heads"] = sum(task_head_path(output, task).is_file() for task in tasks)
    (destination / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"report": str(destination), **checks}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("main", "sensitivity", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four single-thread CPU workers")
    register_models()
    manifest = prepare(args.output)
    tasks = planned_tasks(args.phase)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    print(json.dumps({"tasks": len(tasks), "phase": args.phase, "output": str(args.output)}), flush=True)
    if args.prepare_only:
        return
    if not args.report_only:
        module = importlib.import_module(
            "experiments.domain_phase_mix.exploratory.two_phase_many.validate_mariner_refit_robustness_20260917"
        )
        counts: dict[str, int] = {}
        started = time.monotonic()
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            results = joblib.Parallel(n_jobs=args.workers, return_as="generator_unordered")(
                joblib.delayed(module.fit_task)(task, args.output) for task in tasks
            )
            for index, result in enumerate(results, 1):
                counts[result] = counts.get(result, 0) + 1
                if index % 30 == 0 or index == len(tasks):
                    print(
                        json.dumps(
                            {
                                "completed": index,
                                "total": len(tasks),
                                "elapsed": round(time.monotonic() - started),
                                **counts,
                            }
                        ),
                        flush=True,
                    )
    report(tasks, args.output, manifest, args.phase)


if __name__ == "__main__":
    main()
