# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["joblib", "numpy", "pandas", "scipy"]
# ///

"""Frozen archived screen of alternate convex response geometries."""

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

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_single_phase_observatory_20260902 as bench
from experiments.domain_phase_mix.exploratory.two_phase_many import mariner_convex_round2_models_20260917 as candidates
from experiments.domain_phase_mix.exploratory.two_phase_many import screen_mariner_convex_20260917 as prior
from experiments.domain_phase_mix.exploratory.two_phase_many import single_phase_observatory_models_20260902 as models
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "reference_outputs/mariner_local_validation_20260917/convex_round2"


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


def prepare(output: Path) -> dict:
    protocol = prior.protocol()
    protocol["sources"].update(
        {str(path.resolve()): prior.previous.digest(path) for path in (Path(__file__), Path(candidates.__file__))}
    )
    protocol["candidates"] = [dataclasses.asdict(spec) for spec in candidates.candidate_specs()]
    protocol["advancement"] = (
        "At most two new convex forms, selected by balanced anchor within-fold regret then "
        "correlation before bank inspection. No geometry selected on bank labels."
    )
    protocol["regularization"] = (
        "Canonical unscaled nonnegative amplitude ridge, original ridge/rate/power grids. "
        "Raw-epoch units32. Shape grid retuned for every variant. Raw link projects loss "
        "below zero, with identical projection in CV. Softplus uses training-only "
        "nonnegative floor and A=max(anchor-floor,1e-9); nested floor search has canonical "
        "bounds and flat-profile rule. Different links imply different residual/penalty coordinates."
    )
    protocol["novelty"] = (
        "Prior raw-loss probes fixed geometry and used exp-benefit plus raw harm; raw additive and "
        "softplus links here retune geometry. Alternate benefit asymptotics and PSD/hinge interactions "
        "were not in prior harm-basis/width screens."
    )
    protocol = json.loads(json.dumps(protocol))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "screen_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Round2 source or protocol identity changed")
        return protocol
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    snap = output / "source_snapshot"
    snap.mkdir()
    for path in protocol["sources"]:
        shutil.copyfile(path, snap / Path(path).name)
    return protocol


@functools.cache
def initialize_worker(output: Path) -> None:
    manifest = json.loads((output / "screen_manifest.json").read_text())
    for path, expected in manifest["sources"].items():
        if prior.previous.digest(Path(path)) != expected:
            raise ValueError(f"Source changed: {path}")
    register_candidates()


def load_fitted(saved: dict) -> models.Fitted:
    """Reconstruct a portable numeric fitted head from an augmented fit shard."""
    linear = models.FittedHead(
        intercept=float(saved["head_intercept"]),
        coefficients=saved["head_coefficients"],
        floor=float(saved["head_linear_floor"]),
        active=int(saved["head_active"]),
        cap=float(saved["head_cap"]),
        kappa=float(saved["head_linear_kappa"]),
        converged=bool(saved["head_converged"]),
    )
    if str(saved["head_kind"]) == "linked":
        head = candidates.LinkedHead(
            linear,
            float(saved["head_floor"]),
            float(saved["head_scale"]),
            float(saved["head_kappa"]),
            str(saved["head_link"]),
        )
    else:
        head = linear
    return models.Fitted(
        json.loads(str(saved["shape_json"])), float(saved["ridge"]), head, json.loads(str(saved["diagnostics_json"]))
    )


def augment(task: bench.FitTask, saved: dict) -> dict:
    panel = bench.load_panel(task.panel)
    split = next(s for s in bench.panel_splits(panel, task.repeat + 1) if (s.repeat, s.fold) == (task.repeat, task.fold))
    features = dataclasses.replace(panel.features, component=task.component)
    response = panel.group(task.target).outcomes[:, task.component_index]
    model = registry.ENTRY_BY_ID[task.model_id].build(features)
    shape = json.loads(str(saved["shape_json"]))
    diagnostics = json.loads(str(saved["diagnostics_json"]))
    ridge = float(saved["ridge"])
    matrix = model.base.design(features, shape)
    shapes = model.base.candidate_shapes(features)
    i, j = np.unravel_index(int(np.argmin(saved["cv_table"])), saved["cv_table"].shape)
    if shape != dict(shapes[i]) or ridge != model.base.ridge_grid[j]:
        raise ValueError(f"Wrong selected grid argmin: {task}")
    if isinstance(model, candidates.LinkedConvexModel):
        head = model.selected_head(features, response, split.train, shape, ridge, float(diagnostics["kappa"]))
        linear = head.linear
        kind = "linked"
        scale = head.scale
        floor = head.floor
        kappa = head.kappa
        link = head.link
        final_cv = candidates.linked_cv(
            matrix, response, ridge, model.base.head_for(shape), model.link, kappa, split.inner
        )
        np.testing.assert_allclose(final_cv, diagnostics["inner_cv_rmse"], rtol=1e-9, atol=1e-10)
    elif isinstance(model, models.FittedFloorModel):
        spec = dataclasses.replace(
            model.base.head_for(shape),
            link=models.LinkKind(diagnostics["link"]),
            floor_kappa_prior=float(diagnostics["kappa"]),
        )
        linear = models.fit_head(
            models.Design(matrix.values[split.train], matrix.ridge, matrix.names), response[split.train], ridge, spec
        )
        head = linear
        kind = "standard"
        scale = 1.0
        floor = head.floor
        kappa = head.kappa
        link = "exp"
    else:
        raise TypeError("Unsupported round2 model")
    fitted = models.Fitted(shape, ridge, head, diagnostics)
    for rows, key in ((split.train, "train_prediction"), (split.test, "prediction")):
        np.testing.assert_allclose(model.predict(fitted, features, rows), saved[key], rtol=1e-9, atol=1e-10)
    vertices = dataclasses.replace(
        features,
        weights=np.eye(features.buckets),
        exposures=np.diag(features.inventory),
        label=features.label + "|vertices",
    )
    vertex_design = model.base.design(vertices, shape)
    eta = vertex_design.values @ linear.coefficients + linear.intercept
    result = {
        "head_kind": kind,
        "head_intercept": linear.intercept,
        "head_coefficients": linear.coefficients,
        "head_linear_floor": linear.floor,
        "head_active": linear.active,
        "head_cap": linear.cap,
        "head_linear_kappa": linear.kappa,
        "head_converged": linear.converged,
        "head_floor": floor,
        "head_scale": scale,
        "head_kappa": kappa,
        "head_link": link,
        "vertex_logits": eta,
        "upper_clip_active": bool(link == "exp" and np.any(eta > min(30.0, linear.cap))),
        "training_column_rms": np.sqrt(np.mean(matrix.values[split.train] ** 2, axis=0)),
        "design_names": np.asarray(matrix.names),
        "design_ridge": matrix.ridge,
    }
    combined = saved | result
    loaded = load_fitted(combined)
    np.testing.assert_allclose(model.predict(loaded, features, split.test), saved["prediction"], rtol=1e-9, atol=1e-10)
    return result


def fit_task(task: bench.FitTask, output: Path, split_hash: str) -> str:
    initialize_worker(output)
    status = bench.fit_one(task, output, split_hash)
    path = bench.shard_path(output, task)
    saved = bench.load_shard(path)
    if saved is not None and str(saved["status"]) == "ok" and "head_kind" not in saved:
        saved.update(augment(task, saved))
        bench.atomic_save(path, saved)
    return status


def report(tasks: list[bench.FitTask], output: Path, split_hash: str, tier: str) -> None:
    register_candidates()
    prior.previous.report(tasks, output, split_hash, tier)
    diagnostics = []
    for task in tasks:
        saved = bench.load_shard(bench.shard_path(output, task))
        if saved is None or str(saved["status"]) != "ok":
            raise ValueError(f"Incomplete fit: {task}")
        diagnostics.append(
            dataclasses.asdict(task)
            | {
                "columns": len(saved["head_coefficients"]),
                "upper_clip_active": bool(saved["upper_clip_active"]),
                "max_vertex_logit": float(saved["vertex_logits"].max()),
                "prediction_min": float(saved["prediction"].min()),
                "prediction_max": float(saved["prediction"].max()),
                "min_training_column_rms": float(saved["training_column_rms"].min()),
            }
        )
    pd.DataFrame(diagnostics).to_csv(output / f"report_{tier}" / "head_diagnostics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--tier", choices=("screen", "certify"), default="screen")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four CPU workers")
    register_candidates()
    manifest = prepare(args.output)
    if args.register_only:
        return
    ids = tuple(args.models or [s.model_id for s in candidates.candidate_specs()])
    plan = dataclasses.replace(bench.tier_plan(args.tier, panels=bench.THIRTY_NINE_BUCKET_PANELS), curves=())
    tasks = bench.plan_tasks(plan, ids)
    tasks.sort(key=lambda t: (t.fold, t.component_index, t.panel, t.target, t.model_id))
    if not args.report_only:
        module = importlib.import_module(
            "experiments.domain_phase_mix.exploratory.two_phase_many.screen_mariner_convex_round2_20260917"
        )
        started = time.monotonic()
        counts = {}
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            results = joblib.Parallel(n_jobs=args.workers, return_as="generator_unordered")(
                joblib.delayed(module.fit_task)(t, args.output, manifest["split_hash"]) for t in tasks
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
