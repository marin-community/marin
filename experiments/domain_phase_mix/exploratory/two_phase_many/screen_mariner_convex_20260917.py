# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["joblib", "numpy", "pandas", "scipy"]
# ///

"""Resumable CPU screen of convex-in-mixture MARINER candidates."""

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

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_convex_candidates_20260917 as candidates,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    screen_mariner_harm_forms_20260917 as previous,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "reference_outputs/mariner_local_validation_20260917/convex"


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
    return (
        *previous.source_paths(),
        Path(__file__),
        Path(candidates.__file__),
        Path(bench.olmix_benchmark.__file__),
        Path(bench.starcoder_curves.__file__),
    )


def protocol() -> dict:
    panels = tuple(bench.load_panel(name) for name in bench.THIRTY_NINE_BUCKET_PANELS)
    return {
        "version": 1,
        "sources": {str(path.resolve()): previous.digest(path) for path in source_paths()},
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "joblib")},
        "split_hash": bench.manifest_hash(bench.split_manifest(panels, 1)),
        "panel_inputs": {panel.name: panel.input_hashes for panel in panels},
        "candidates": [
            {key: value for key, value in dataclasses.asdict(spec).items() if key != "design_function"}
            for spec in candidates.candidate_specs()
        ],
        "advancement": (
            "At most two contrasting convex forms; prioritize balanced within-fold selection regret, then Spearman. "
            "Advance best even if below MARINER; no training."
        ),
        "regularization": (
            "Canonical unscaled nonnegative ridge; new raw-epoch columns use a fixed 32-epoch unit. "
            "Units, curvature, width, and basis count affect the functional prior; "
            "they are not isolated causal interventions."
        ),
        "convexity": (
            "Analytic uncapped family proof. Every saved convex fit is checked on all 39 simplex vertices "
            "against the implementation upper-logit clip; active cases must use log-domain or uncapped optimization."
        ),
    }


def prepare(output: Path) -> dict:
    current = json.loads(json.dumps(protocol()))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "screen_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != current:
            raise ValueError("Registered sources, inputs, or protocol changed; use a fresh output identity.")
        return current
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
        if previous.digest(Path(path)) != checksum:
            raise ValueError(f"Source changed since preregistration: {path}")
    register_candidates()


def augment_diagnostics(task: bench.FitTask, saved: dict) -> dict:
    """Replay the selected head and certify upper clipping over the full simplex."""
    panel = bench.load_panel(task.panel)
    split = next(
        s for s in bench.panel_splits(panel, task.repeat + 1) if s.repeat == task.repeat and s.fold == task.fold
    )
    features = dataclasses.replace(panel.features, component=task.component)
    model = registry.ENTRY_BY_ID[task.model_id].build(features)
    if not isinstance(model, models.FittedFloorModel):
        raise TypeError("Expected fitted-floor model")
    shape = json.loads(str(saved["shape_json"]))
    diagnostics = json.loads(str(saved["diagnostics_json"]))
    spec = dataclasses.replace(
        model.base.head_for(shape),
        link=models.LinkKind(diagnostics["link"]),
        floor_kappa_prior=float(diagnostics["kappa"]),
    )
    design = model.base.design(features, shape)
    response = panel.group(task.target).outcomes[:, task.component_index]
    head = models.fit_head(
        models.Design(design.values[split.train], design.ridge, design.names),
        response[split.train],
        float(saved["ridge"]),
        spec,
    )
    reconstructed = models.predict_head(head, design.values[split.test], spec)
    np.testing.assert_allclose(reconstructed, saved["prediction"], rtol=1e-9, atol=1e-10)
    vertices = dataclasses.replace(
        features,
        weights=np.eye(features.buckets),
        exposures=np.diag(features.inventory),
        label=features.label + "|vertices",
    )
    vertex_design = model.base.design(vertices, shape)
    logits = vertex_design.values @ head.coefficients + head.intercept
    rms = np.sqrt(np.mean(design.values[split.train] ** 2, axis=0))
    upper = min(models.LINK_CLIP, head.cap)
    return {
        "head_coefficients": head.coefficients,
        "head_intercept": head.intercept,
        "head_floor": head.floor,
        "head_cap": head.cap,
        "vertex_logits": logits,
        "upper_clip_threshold": upper,
        "upper_clip_active_at_vertices": bool(np.any(logits > upper)),
        "training_column_rms": rms,
        "design_names": np.asarray(design.names),
        "design_ridge": design.ridge,
        "head_scale_columns": spec.scale_columns,
        "head_replay_max_error": float(np.max(np.abs(reconstructed - saved["prediction"]))),
    }


def fit_task(task: bench.FitTask, output: Path, split_hash: str) -> str:
    initialize_worker(output)
    path = bench.shard_path(output, task)
    saved = bench.load_shard(path)
    if saved is not None and (str(saved.get("status", "")) != "ok" or not np.isfinite(saved["prediction"]).all()):
        path.rename(path.with_suffix(f".failed-{time.time_ns()}.npz"))
    status = bench.fit_one(task, output, split_hash)
    saved = bench.load_shard(path)
    if saved is not None and str(saved["status"]) == "ok" and "vertex_logits" not in saved:
        saved.update(augment_diagnostics(task, saved))
        bench.atomic_save(path, saved)
    return status


def report(tasks: list[bench.FitTask], output: Path, split_hash: str, tier: str) -> None:
    register_candidates()
    previous.report(tasks, output, split_hash, tier)
    rows = []
    for task in tasks:
        saved = bench.load_shard(bench.shard_path(output, task))
        if saved is None or str(saved["status"]) != "ok":
            raise ValueError(f"Incomplete fit {task}")
        rms = saved["training_column_rms"]
        rows.append(
            dataclasses.asdict(task)
            | {
                "columns": len(rms),
                "zero_columns": int(np.sum(rms == 0)),
                "column_rms_min_nonzero": float(np.min(rms[rms > 0])),
                "column_rms_max": float(np.max(rms)),
                "column_rms_median": float(np.median(rms)),
                "max_vertex_logit": float(np.max(saved["vertex_logits"])),
                "upper_clip_active": bool(saved["upper_clip_active_at_vertices"]),
                "head_replay_max_error": float(saved["head_replay_max_error"]),
            }
        )
    pd.DataFrame(rows).to_csv(output / f"report_{tier}" / "basis_and_convexity_diagnostics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tier", choices=("smoke", "screen", "certify"), default="screen")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four CPU workers.")
    register_candidates()
    manifest = prepare(args.output)
    ids = tuple(args.models or [spec.model_id for spec in candidates.candidate_specs()])
    plan = dataclasses.replace(bench.tier_plan(args.tier, panels=bench.THIRTY_NINE_BUCKET_PANELS), curves=())
    tasks = bench.plan_tasks(plan, ids)
    tasks.sort(key=lambda task: (task.fold, task.component_index, task.panel, task.target, task.model_id))
    if not args.report_only:
        worker_module = importlib.import_module(
            "experiments.domain_phase_mix.exploratory.two_phase_many.screen_mariner_convex_20260917"
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
