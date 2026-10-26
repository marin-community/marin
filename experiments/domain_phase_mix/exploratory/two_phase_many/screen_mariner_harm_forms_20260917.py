# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resumable archived-data screen of registered MARINER harm hypotheses."""

import argparse
import dataclasses
import functools
import hashlib
import importlib
import importlib.metadata
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
    mariner_harm_screen_models_20260917 as candidates,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "reference_outputs/mariner_harm_screen_20260917/fixed_basis"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def source_paths() -> tuple[Path, ...]:
    root = next(path for path in HERE.parents if (path / "uv.lock").is_file())
    modules = (Path(__file__), Path(candidates.__file__), Path(models.__file__), Path(registry.__file__))
    helpers = tuple(HERE.glob("benchmark_single_phase*20260902.py"))
    return (*modules, *helpers, registry.FLOOR_ANCHORS_FILE, root / "uv.lock")


def protocol() -> dict:
    panels = tuple(bench.load_panel(name) for name in bench.THIRTY_NINE_BUCKET_PANELS)
    return {
        "version": 1,
        "sources": {str(path.resolve()): digest(path) for path in source_paths()},
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "joblib")},
        "split_hash": bench.manifest_hash(bench.split_manifest(panels, 1)),
        "panel_inputs": {panel.name: panel.input_hashes for panel in panels},
        "candidates": [
            {key: value for key, value in dataclasses.asdict(spec).items() if key != "design_function"}
            for spec in candidates.candidate_specs()
        ],
    }


def prepare(output: Path) -> dict:
    current = json.loads(json.dumps(protocol()))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "screen_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if current != previous:
            raise ValueError("Screen source, input, or dependency identity changed; use a new output identity.")
        return previous
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source in source_paths():
        shutil.copyfile(source, snapshot / source.name)
    return current


@functools.cache
def initialize_worker(output: Path) -> None:
    expected = json.loads((output / "screen_manifest.json").read_text())
    for path, checksum in expected["sources"].items():
        if digest(Path(path)) != checksum:
            raise ValueError(f"Source changed since screen registration: {path}")
    register_candidates()


def fit_task(task: bench.FitTask, output: Path, split_hash: str) -> str:
    initialize_worker(output)
    path = bench.shard_path(output, task)
    saved = bench.load_shard(path)
    if saved is not None and (str(saved.get("status", "")) != "ok" or not np.isfinite(saved["prediction"]).all()):
        path.rename(path.with_suffix(f".failed-{time.time_ns()}.npz"))
    return bench.fit_one(task, output, split_hash)


def verify_coverage(tasks: list[bench.FitTask], predictions: pd.DataFrame, fits: pd.DataFrame) -> dict:
    if len(fits) != len(tasks) or not fits["status"].eq("ok").all():
        raise ValueError(f"Incomplete fits: {fits['status'].value_counts().to_dict()}")
    keys = ["model", "panel", "target", "component_index", "repeat", "fold", "row_index"]
    if predictions.duplicated(keys).any():
        raise ValueError("Duplicate out-of-fold prediction identities")
    expected = set()
    for task in tasks:
        panel = bench.load_panel(task.panel)
        split = next(
            s for s in bench.panel_splits(panel, task.repeat + 1) if s.repeat == task.repeat and s.fold == task.fold
        )
        expected.update(
            (task.model_id, task.panel, task.target, task.component_index, task.repeat, task.fold, int(row))
            for row in split.test
        )
    actual = set(predictions[keys].itertuples(index=False, name=None))
    if expected != actual or not np.isfinite(predictions[["observed", "prediction"]].to_numpy()).all():
        raise ValueError(f"Prediction coverage mismatch: expected {len(expected)}, actual {len(actual)}")
    return {"fits": len(tasks), "predictions": len(actual), "unique": True, "exact_coverage": True, "all_finite": True}


def report(tasks: list[bench.FitTask], output: Path, split_hash: str, tier: str) -> None:
    register_candidates()
    predictions, fits = bench.collect_shards(tasks, output, split_hash)
    destination = output / f"report_{tier}"
    destination.mkdir(exist_ok=True)
    fits.to_csv(destination / "fits.csv", index=False)
    checks = verify_coverage(tasks, predictions, fits)
    predictions.to_csv(destination / "component_predictions.csv", index=False)
    component_keys = ["model", "panel", "target", "component", "repeat"]
    components = bench.grouped_metrics(predictions, component_keys)
    components.to_csv(destination / "component_metrics.csv", index=False)
    bench.grouped_metrics(predictions, [*component_keys, "fold"]).to_csv(
        destination / "component_fold_metrics.csv", index=False
    )
    aggregate = bench.aggregate_predictions(predictions)
    if not aggregate.empty:
        aggregate.to_csv(destination / "aggregate_predictions.csv", index=False)
        group_keys = ["model", "role", "parent", "panel", "target", "repeat"]
        pooled = bench.grouped_metrics(aggregate, group_keys)
        folds = bench.grouped_metrics(aggregate, [*group_keys, "fold"])
        pooled.to_csv(destination / "aggregate_metrics.csv", index=False)
        folds.to_csv(destination / "aggregate_fold_metrics.csv", index=False)
        bench.aggregate_summary(pooled, folds).to_csv(destination / "aggregate_summary.csv", index=False)
    checks["aggregate_prediction_rows"] = len(aggregate)
    (destination / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"report": str(destination), **checks}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tier", choices=("smoke", "screen", "certify"), default="screen")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    register_candidates()
    manifest = prepare(args.output)
    ids = tuple(args.models or [spec.model_id for spec in candidates.candidate_specs()])
    plan = dataclasses.replace(bench.tier_plan(args.tier, panels=bench.THIRTY_NINE_BUCKET_PANELS), curves=())
    tasks = bench.plan_tasks(plan, ids)
    # Interleave variants so progress covers hypotheses instead of completing a family first.
    tasks.sort(key=lambda task: (task.fold, task.component_index, task.panel, task.target, task.model_id))
    if not args.report_only:
        worker_module = importlib.import_module(
            "experiments.domain_phase_mix.exploratory.two_phase_many.screen_mariner_harm_forms_20260917"
        )
        counts: dict[str, int] = {}
        started = time.monotonic()
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            results = joblib.Parallel(n_jobs=args.workers, return_as="generator_unordered")(
                joblib.delayed(worker_module.fit_task)(task, args.output, manifest["split_hash"]) for task in tasks
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
