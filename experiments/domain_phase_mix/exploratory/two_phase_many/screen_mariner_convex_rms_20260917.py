# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["joblib", "numpy", "pandas", "scipy"]
# ///

"""Registered RMS-column sensitivity for two frozen convex finalists."""

import argparse
import dataclasses
import functools
import importlib
import json
import shutil
import time
from pathlib import Path

import joblib

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_convex_candidates_20260917 as candidates,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    screen_mariner_convex_20260917 as screen,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

MODEL_IDS = ("cv17_smooth_d1", "cv17_huber_d8")
DEFAULT_OUTPUT = screen.DEFAULT_OUTPUT / "sensitivity"


def build_rms(spec: candidates.CandidateSpec, features: models.Features) -> models.FittedFloorModel:
    """Change only column scaling, preserving basis, shape grid, ridge grid, and floor search."""
    model = candidates.build_model(spec, features)
    grid = model.base
    if not isinstance(grid, models.GridModel):
        raise TypeError("Expected grid model")
    return dataclasses.replace(
        model,
        base=dataclasses.replace(
            grid,
            model_id=f"{spec.model_id}__rms",
            head=dataclasses.replace(grid.head, scale_columns=True, scale_rule="rms"),
        ),
    )


def register_candidates() -> None:
    screen.register_candidates()
    for spec in candidates.candidate_specs():
        if spec.model_id in MODEL_IDS:
            registry.ENTRY_BY_ID[f"{spec.model_id}__rms"] = dataclasses.replace(
                registry.ENTRY_BY_ID[spec.model_id],
                model_id=f"{spec.model_id}__rms",
                parent=spec.model_id,
                build=functools.partial(build_rms, spec),
                note="RMS column scaling only; frozen convex equation",
            )


def prepare(output: Path) -> dict:
    protocol = screen.protocol()
    protocol["sources"][str(Path(__file__).resolve())] = screen.previous.digest(Path(__file__))
    protocol["candidates"] = [
        dict(spec, model_id=f"{spec['model_id']}__rms", head_scale_columns=True, head_scale_rule="rms")
        for spec in protocol["candidates"]
        if spec["model_id"] in MODEL_IDS
    ]
    protocol["advancement"] = "None: sensitivity of frozen finalists; no further candidate advancement."
    protocol["regularization"] = (
        "Only scale_columns=True, scale_rule=rms. Canonical shape grid, ridge grid, floor selection, and equations "
        "unchanged. Compare with unscaled convex anchors and hs17_mariner__rms anchors in ../refits."
    )
    protocol = json.loads(json.dumps(protocol))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "screen_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("RMS sensitivity identity changed")
        return protocol
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    for path in protocol["sources"]:
        shutil.copyfile(path, snapshots / Path(path).name)
    return protocol


@functools.cache
def initialize_worker(output: Path) -> None:
    protocol = json.loads((output / "screen_manifest.json").read_text())
    for path, expected in protocol["sources"].items():
        if screen.previous.digest(Path(path)) != expected:
            raise ValueError(f"Source changed during RMS sensitivity: {path}")
    register_candidates()


def fit_task(task: bench.FitTask, output: Path, split_hash: str) -> str:
    initialize_worker(output)
    result = bench.fit_one(task, output, split_hash)
    path = bench.shard_path(output, task)
    saved = bench.load_shard(path)
    if saved is not None and str(saved["status"]) == "ok" and "vertex_logits" not in saved:
        saved.update(screen.augment_diagnostics(task, saved))
        bench.atomic_save(path, saved)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four CPU workers")
    register_candidates()
    manifest = prepare(args.output)
    if args.register_only:
        return
    plan = dataclasses.replace(bench.tier_plan("screen", panels=bench.THIRTY_NINE_BUCKET_PANELS), curves=())
    tasks = bench.plan_tasks(plan, tuple(f"{model}__rms" for model in MODEL_IDS))
    if not args.report_only:
        module = importlib.import_module(
            "experiments.domain_phase_mix.exploratory.two_phase_many.screen_mariner_convex_rms_20260917"
        )
        started = time.monotonic()
        counts: dict[str, int] = {}
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            results = joblib.Parallel(n_jobs=args.workers, return_as="generator_unordered")(
                joblib.delayed(module.fit_task)(task, args.output, manifest["split_hash"]) for task in tasks
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
    screen.report(tasks, args.output, manifest["split_hash"], "screen")


if __name__ == "__main__":
    main()
