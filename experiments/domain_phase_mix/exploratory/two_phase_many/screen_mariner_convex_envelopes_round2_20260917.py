# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["joblib", "numpy", "pandas", "scipy"]
# ///

"""Separate archived-data screen preserving the onset and units of MARINER harm."""

import argparse
import dataclasses
import functools
import importlib
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_convex_envelopes_round2_20260917 as candidates,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    screen_mariner_convex_20260917 as convex_screen,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    screen_mariner_harm_forms_20260917 as previous,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE / "reference_outputs/mariner_local_validation_20260917"
OUTPUT = ROOT / "convex_envelopes_round2_v2"
BASELINE = ROOT / "convex"


def register_candidates() -> None:
    parent = registry.ENTRY_BY_ID[registry.FROZEN_ID]
    for spec in candidates.candidate_specs():
        registry.ENTRY_BY_ID[spec.model_id] = dataclasses.replace(
            parent,
            model_id=spec.model_id,
            role="ablation",
            parent=registry.FROZEN_ID,
            build=functools.partial(candidates.build_model, spec),
            note=spec.equation,
        )


def verify_analytic_families() -> dict:
    """Check ordering, convexity, derivatives, tangent roots and frozen control parity."""
    panels = [bench.load_panel(name) for name in bench.THIRTY_NINE_BUCKET_PANELS]
    bounds = sorted({0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0, *(float(x) for p in panels for x in p.features.inventory)})
    thresholds = (-3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    fd_error = 0.0
    root_error = 0.0
    regimes = {"tangent_at_bound": 0, "interior_join": 0, "line_from_zero": 0}
    count = 0
    for threshold in thresholds:
        inflection = candidates.harm_inflection(threshold)
        for maximum in bounds:
            join, slope = candidates.envelope_join(threshold, maximum)
            if join == maximum:
                regimes["tangent_at_bound"] += 1
            elif join == 0:
                regimes["line_from_zero"] += 1
            else:
                regimes["interior_join"] += 1
                residual = abs(slope - float(candidates.harm_derivative(np.asarray(join), threshold)))
                root_error = max(root_error, residual)
                assert residual < 1e-10
            points = np.unique(
                np.r_[np.linspace(0, maximum, 151), np.geomspace(1e-5, max(maximum, inflection) * 4, 181)]
            )
            epochs = points[:, None]
            original = candidates.original_harm(epochs, threshold)
            for spec in candidates.candidate_specs():
                harm, derivative = candidates.harm_and_derivative(spec.model_id, epochs, threshold, (maximum,))
                assert np.isfinite(harm).all() and np.isfinite(derivative).all()
                assert abs(float(harm[0, 0])) < 1e-12
                assert derivative.min() >= -1e-12 and np.diff(derivative[:, 0]).min() >= -1e-10
                if spec.model_id == "ce2_majorant":
                    assert np.min(harm - original) >= -1e-10
                else:
                    assert np.max((harm - original)[points <= maximum]) <= 1e-10
                    endpoint, _ = candidates.harm_and_derivative(
                        spec.model_id, np.asarray([[maximum]]), threshold, (maximum,)
                    )
                    assert abs(float(endpoint[0, 0] - candidates.original_harm(np.asarray(maximum), threshold))) < 1e-9
                positive = epochs[epochs[:, 0] > 1e-3]
                step = np.minimum(positive / 4, 1e-6 * np.maximum(1.0, positive))
                plus = candidates.harm_and_derivative(spec.model_id, positive + step, threshold, (maximum,))[0]
                minus = candidates.harm_and_derivative(spec.model_id, positive - step, threshold, (maximum,))[0]
                finite_difference = (plus - minus) / (2 * step)
                analytic = candidates.harm_and_derivative(spec.model_id, positive, threshold, (maximum,))[1]
                error = float(np.max(np.abs(finite_difference - analytic)))
                fd_error = max(fd_error, error)
                assert error < 2e-7, (spec.model_id, threshold, maximum, join, error)
                count += 1
    control_error = 0.0
    for panel in panels:
        frozen = registry.ENTRY_BY_ID[registry.FROZEN_ID].build(panel.features)
        for shape in frozen.base.shapes:
            expected = candidates.original.log_harm_design(panel.features, shape, (("harm_power", 2.0),))
            actual = frozen.base.design(panel.features, shape)
            centered_actual = actual.values - actual.values[0]
            centered_expected = expected.values - expected.values[0]
            control_error = max(control_error, float(np.max(np.abs(centered_actual - centered_expected))))
            np.testing.assert_allclose(centered_actual, centered_expected, rtol=0, atol=1e-12)
        for spec in candidates.candidate_specs():
            model = candidates.build_model(spec, panel.features)
            shape = model.base.shapes[0]
            query = dataclasses.replace(panel.features, inventory=panel.features.inventory * 2, label="inventory_probe")
            np.testing.assert_allclose(
                model.base.builder(query, shape).values, model.base.builder(panel.features, shape).values
            )
            assert model.base.shapes == frozen.base.shapes
            assert model.base.ridge_grid == frozen.base.ridge_grid
            assert json.dumps(dataclasses.asdict(model.base.head), sort_keys=True) == json.dumps(
                dataclasses.asdict(frozen.base.head), sort_keys=True
            )
    return {
        "harm_cases": count,
        "thresholds": thresholds,
        "bounds": len(bounds),
        "regimes": regimes,
        "maximum_derivative_absolute_error": fd_error,
        "maximum_tangent_root_residual": root_error,
        "nonconvex_control_centered_design_error": control_error,
        "fixed_inventory_query_invariance": True,
        "canonical_shape_ridge_and_head_preserved": True,
    }


def prepare(output: Path) -> dict:
    baseline = json.loads((BASELINE / "screen_manifest.json").read_text())
    for filename, checksum in baseline["sources"].items():
        if previous.digest(Path(filename)) != checksum:
            raise ValueError(f"Canonical baseline source changed: {filename}")
    panels = tuple(bench.load_panel(name) for name in bench.THIRTY_NINE_BUCKET_PANELS)
    split_hash = bench.manifest_hash(bench.split_manifest(panels, 1))
    assert split_hash == baseline["split_hash"]
    assert {panel.name: panel.input_hashes for panel in panels} == baseline["panel_inputs"]
    sources = tuple(dict.fromkeys((*convex_screen.source_paths(), Path(__file__), Path(candidates.__file__))))
    manifest = {
        "version": 1,
        "sources": {str(path.resolve()): previous.digest(path) for path in sources},
        "split_hash": split_hash,
        "panel_inputs": {panel.name: panel.input_hashes for panel in panels},
        "physical_inventory": {panel.name: panel.features.inventory.tolist() for panel in panels},
        "candidates": [dataclasses.asdict(spec) for spec in candidates.candidate_specs()],
        "baseline_manifest_sha256": previous.digest(BASELINE / "screen_manifest.json"),
        "baseline_prediction_sha256": previous.digest(BASELINE / "report_screen/component_predictions.csv"),
        "advancement": (
            "At most one: lowest balanced anchor mean-fold regret, then highest balanced Spearman; "
            "complete objectives before banks."
        ),
        "regularization": (
            "Canonical original-harm units, unscaled nonnegative ridge, shared shape grid and fitted-floor procedure."
        ),
        "scope": "Two new forms, 90 anchor fits each. No bank outcomes select parameters or advancement.",
    }
    current = json.loads(json.dumps(manifest))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "screen_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != current:
            raise ValueError("Registered protocol changed; use a fresh identity.")
        return current
    validation = verify_analytic_families()
    path.write_text(json.dumps(current, indent=2) + "\n")
    (output / "analytic_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for source in sources:
        shutil.copyfile(source, snapshot / source.name)
    return current


@functools.cache
def initialize_worker(output: Path) -> None:
    manifest = json.loads((output / "screen_manifest.json").read_text())
    for filename, checksum in manifest["sources"].items():
        if previous.digest(Path(filename)) != checksum:
            raise ValueError(f"Registered source changed: {filename}")
    register_candidates()


def fit_task(task: bench.FitTask, output: Path, split_hash: str) -> str:
    initialize_worker(output)
    status = bench.fit_one(task, output, split_hash)
    path = bench.shard_path(output, task)
    saved = bench.load_shard(path)
    if saved is not None and str(saved["status"]) == "ok" and "vertex_logits" not in saved:
        saved.update(convex_screen.augment_diagnostics(task, saved))
        bench.atomic_save(path, saved)
    return status


def report(tasks: list[bench.FitTask], output: Path, split_hash: str, tier: str) -> None:
    previous.report(tasks, output, split_hash, tier)
    if tier != "screen":
        return
    destination = output / f"report_{tier}"
    predictions = pd.read_csv(destination / "component_predictions.csv")
    baseline = pd.read_csv(BASELINE / "report_screen/component_predictions.csv")
    baseline = baseline[baseline.model.eq("cv17_mariner")].copy()
    keys = ["panel", "target", "component", "repeat", "fold", "row_index"]
    for _, rows in predictions.groupby("model"):
        joined = rows.merge(baseline, on=keys, suffixes=("", "_baseline"), validate="one_to_one")
        assert len(joined) == len(rows) == len(baseline)
        np.testing.assert_allclose(joined.observed, joined.observed_baseline, rtol=0, atol=1e-12)
    frames = []
    unit_keys = ["model", "panel", "target", "component", "repeat"]
    for frame in (predictions, baseline):
        components = bench.grouped_metrics(frame, unit_keys)
        folds = bench.grouped_metrics(frame, [*unit_keys, "fold"])
        regret = folds.groupby(unit_keys, as_index=False).regret_at_1.mean()
        frames.append(components.merge(regret, on=unit_keys, suffixes=("_pooled", "")))
    metrics = pd.concat(frames)
    panel_metrics = metrics.groupby(["model", "panel", "target"], as_index=False)[
        ["spearman", "rmse", "regret_at_1"]
    ].mean()
    balanced = panel_metrics.groupby("model", as_index=False)[["spearman", "rmse", "regret_at_1"]].mean()
    reference = balanced[balanced.model.eq("cv17_mariner")].iloc[0]
    for metric in ("spearman", "rmse", "regret_at_1"):
        balanced[f"delta_{metric}"] = balanced[metric] - reference[metric]
    balanced.to_csv(destination / "balanced_anchor_summary.csv", index=False)
    panel_metrics.to_csv(destination / "panel_anchor_summary.csv", index=False)
    print(balanced.to_string(index=False), flush=True)
    winner = (
        balanced[balanced.model.str.startswith("ce2_")]
        .sort_values(["regret_at_1", "spearman"], ascending=[True, False])
        .iloc[0]
    )
    (output / "advancement.json").write_text(
        json.dumps(
            {
                "model_id": winner.model,
                "rule": "balanced anchor mean-fold regret, then Spearman",
                "next_gate": "complete grouped objectives before banks",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--tier", choices=("screen", "certify"), default="screen")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four single-threaded CPU workers.")
    register_candidates()
    manifest = prepare(args.output)
    if args.prepare_only:
        print(json.dumps({"registered": str(args.output), "validation": "passed"}), flush=True)
        return
    ids = tuple(args.models or [spec.model_id for spec in candidates.candidate_specs()])
    plan = dataclasses.replace(bench.tier_plan(args.tier, panels=bench.THIRTY_NINE_BUCKET_PANELS), curves=())
    tasks = bench.plan_tasks(plan, ids)
    tasks.sort(key=lambda task: (task.fold, task.component_index, task.panel, task.target, task.model_id))
    if not args.report_only:
        worker = importlib.import_module(
            "experiments.domain_phase_mix.exploratory.two_phase_many.screen_mariner_convex_envelopes_round2_20260917"
        )
        started = time.monotonic()
        counts: dict[str, int] = {}
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            results = joblib.Parallel(n_jobs=args.workers, return_as="generator_unordered")(
                joblib.delayed(worker.fit_task)(task, args.output, manifest["split_hash"]) for task in tasks
            )
            for index, status in enumerate(results, 1):
                counts[status] = counts.get(status, 0) + 1
                if index % 12 == 0 or index == len(tasks) or status == "failed":
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
    report(tasks, args.output, manifest["split_hash"], args.tier)


if __name__ == "__main__":
    main()
