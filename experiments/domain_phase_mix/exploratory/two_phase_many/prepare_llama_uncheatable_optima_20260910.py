# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0", "scikit-learn==1.7.2",
#   "cvxpy==1.7.5", "fsspec==2026.1.0", "gcsfs==2026.1.0", "plotly==6.5.1",
#   "tabulate==0.9.0", "threadpoolctl==3.6.0", "matplotlib==3.10.8",
# ]
# ///
"""Prepare uncapped Llama MARINER and HPR aggregate-replacement proposals offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import benchmark_single_phase_observatory_20260902 as observatory
import benchmark_two_phase_link_controls_20260907 as controls
import numpy as np
import pandas as pd
import prepare_two_phase_link_transfer_20260907 as preparation
from fit_two_phase_link_spines_20260907 import load_module, write_json_atomic
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
REFERENCE = BASE / "reference_outputs"
OUTPUT = REFERENCE / "llama_uncheatable_optima_20260910"
STANDALONE = BASE.parents[4] / "mixture-selection" / "mixture_selection.py"
ANCHORS = REFERENCE / "delphi_floor_anchors_20260907/anchors.csv"
SCALES = ("60m", "300m")
SEED = 20260911


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> None:
    destination = OUTPUT / "inputs"
    manifest = destination / "manifest.json"
    if manifest.exists():
        identity = json.loads(manifest.read_text())
        for name, digest in identity["frozen_sha256"].items():
            assert sha(destination / name) == digest, name
        print("reuse frozen inputs", flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(STANDALONE, destination / "mariner.py")
    module = load_module(destination / "mariner.py", "llama_mariner")
    anchors = pd.read_csv(ANCHORS)
    source_hashes = {str(STANDALONE): sha(STANDALONE), str(ANCHORS): sha(ANCHORS)}
    metadata = {}
    for scale in SCALES:
        panel = observatory.load_panel(f"{scale}_39bucket")
        group = panel.group("uncheatable")
        out = destination / scale
        out.mkdir(exist_ok=True)
        calibration = np.asarray(["baseline_proportional" in r for r in panel.runs])
        assert calibration.sum() == 1
        swarm = module.Swarm(
            panel.runs,
            panel.buckets,
            panel.features.weights,
            panel.features.inventory,
            pd.DataFrame(group.outcomes, columns=group.components),
            calibration,
        )
        labels = np.full(len(panel.runs), -1, dtype=int)
        for fold, (_, test) in enumerate(module.final_inner_folds(swarm, None)):
            labels[test] = fold
        weights = pd.DataFrame(swarm.weights, columns=swarm.buckets)
        weights.insert(0, "run", swarm.runs)
        weights["calibration"] = calibration
        weights.to_csv(out / "swarm_weights.csv", index=False)
        outcomes = swarm.outcomes.copy()
        outcomes.insert(0, "run", swarm.runs)
        outcomes.to_csv(out / "swarm_outcomes.csv", index=False)
        pd.DataFrame({"bucket": swarm.buckets, "epochs_per_unit_weight": swarm.inventory}).to_csv(
            out / "buckets.csv", index=False
        )
        pd.DataFrame(
            {"objective": "uncheatable", "component": group.components, "weight": group.aggregation_weights}
        ).to_csv(out / "objectives.csv", index=False)
        anchor = anchors[anchors.panel.eq(panel.name) & anchors.target.eq("uncheatable")].copy()
        assert set(anchor.component) == set(group.components) and len(anchor) == 7
        anchor["objective"] = "uncheatable"
        anchor.to_csv(out / "anchors.csv", index=False)
        pd.DataFrame({"row": np.arange(len(labels)), "run": swarm.runs, "inner_fold": labels}).to_csv(
            out / "folds.csv", index=False
        )
        source_hashes.update({str(controls.REPO_ROOT / p): h for p, h in panel.input_hashes.items()})
        metadata[scale] = {
            "single_phase_rows": len(panel.runs),
            "components": list(group.components),
            "calibration_rows": int(calibration.sum()),
            "inner_fold_sizes": np.bincount(labels[labels >= 0]).tolist(),
            "anchor_note": "Each component uses this panel's proportional outcome and the panel's aggregate repeat SD.",
        }
    write_json_atomic(
        manifest,
        {
            "sources_sha256": source_hashes,
            "panels": metadata,
            "frozen_sha256": {
                str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob("*")) if p.is_file()
            },
        },
    )


def single_inputs(scale: str) -> tuple[Any, Any, Any, dict, tuple]:
    module = load_module(OUTPUT / "inputs/mariner.py", "llama_mariner")
    folder = OUTPUT / "inputs" / scale
    swarm = module.read_swarm(folder / "swarm_weights.csv", folder / "swarm_outcomes.csv", folder / "buckets.csv")
    objective = module.read_objectives(folder / "objectives.csv")["uncheatable"]
    anchors = module.read_anchors(folder / "anchors.csv", "uncheatable")
    labels = pd.read_csv(folder / "folds.csv").inner_fold.to_numpy()
    folds = tuple((np.flatnonzero(labels != k), np.flatnonzero(labels == k)) for k in range(3))
    return module, swarm, objective, anchors, folds


def fit_component(job: tuple[str, int]) -> str:
    scale, index = job
    path = OUTPUT / "fits" / scale / f"component_{index}.json"
    if path.exists():
        return f"reuse {scale}/{index}"
    module, swarm, objective, anchors, folds = single_inputs(scale)
    component = objective.components[index]
    started = time.monotonic()
    with threadpool_limits(limits=1):
        fit = module.fit_task(swarm, component, anchors[component], folds)
    write_json_atomic(path, fit.to_json())
    return f"fitted {scale}/{index}: {time.monotonic() - started:.1f} seconds"


def fit_single() -> None:
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fit_component, [(s, i) for s in SCALES for i in range(7)]):
            print(result, flush=True)
    for scale in SCALES:
        module, swarm, objective, _, _ = single_inputs(scale)
        tasks = tuple(
            module.TaskFit.from_json(json.loads((OUTPUT / "fits" / scale / f"component_{i}.json").read_text()))
            for i in range(7)
        )
        fit = module.ObjectiveFit(objective.name, swarm.buckets, swarm.inventory, objective.weights, tasks)
        write_json_atomic(OUTPUT / "fits" / scale / "mariner.json", fit.to_json())


def fit_hpr_small() -> None:
    destination = OUTPUT / "fits/60m/hpr"
    if (destination / "complete.json").exists():
        print("reuse Llama 160M HPR", flush=True)
        return
    single = pd.read_csv(observatory.swarm39.SIXTY_M / "fit_single_phase.csv")
    two = pd.read_csv(observatory.swarm39.SIXTY_M / "fit_two_phase.csv")
    _, swarm, objective, _, _ = single_inputs("60m")
    assert tuple(single.run_name) == swarm.runs
    buckets = swarm.buckets
    phase = np.stack([two[[f"phase_{p}_{b}" for b in buckets]].to_numpy(float) for p in (0, 1)], axis=1)
    weights = np.concatenate([np.repeat(swarm.weights[:, None, :], 2, axis=1), phase])
    response = np.concatenate([swarm.outcomes.to_numpy() @ objective.weights, two.uncheatable_bpb.to_numpy()])
    groups = np.concatenate([single.paired_run_name.to_numpy(), two.run_name.to_numpy()])
    runs = np.concatenate([single.run_name.to_numpy(str), two.run_name.to_numpy(str)])
    frame = pd.DataFrame({"run_name": runs, "phase_correspondence_key": groups})
    calibration = groups == "baseline_proportional"
    assert calibration.sum() == 2
    # The Llama design and its paired single-phase policies use nominal 80/20.
    # Use its own geometry, not the Delphi runtime fraction reused by an old loader.
    inventory = swarm.inventory
    aggregate = 0.8 * weights[:, 0] + 0.2 * weights[:, 1]
    for group in np.unique(groups):
        assert np.max(np.ptp(aggregate[groups == group], axis=0)) < 1e-9, group
    folds = preparation.neighborhood_splits(groups, aggregate, calibration, np.arange(len(frame)), SEED)
    data = controls.baseline.pooled.Dataset(
        "llama160_uncheatable", frame, response, weights, 0.8 * inventory, 0.2 * inventory, list(buckets)
    )
    family_index = observatory.swarm39._exposure("delphi_3e18_two_phase_fit")[3]
    started = time.monotonic()
    with threadpool_limits(limits=1):
        fitted = controls.fit_control("hierarchical_phase_replay", data, folds, "uncheatable", family_index)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "model.pkl").write_bytes(pickle.dumps(fitted.model))
    write_json_atomic(destination / "selection.json", fitted.selection)
    np.savez_compressed(
        destination / "inputs.npz",
        weights=weights,
        observed=response,
        groups=groups.astype(str),
        c0=0.8 * inventory,
        c1=0.2 * inventory,
    )
    frame.to_csv(destination / "rows.csv", index=False)
    write_json_atomic(
        destination / "complete.json",
        {
            "rows": len(frame),
            "elapsed_seconds": time.monotonic() - started,
            "sources_sha256": {
                str(observatory.swarm39.SIXTY_M / f): sha(observatory.swarm39.SIXTY_M / f)
                for f in ("fit_single_phase.csv", "fit_two_phase.csv")
            },
            "sha256": {p.name: sha(p) for p in sorted(destination.iterdir()) if p.is_file()},
        },
    )
    print(f"fitted Llama 160M HPR: {len(frame)} rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "single", "hpr"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "single":
        fit_single()
    else:
        fit_hpr_small()


if __name__ == "__main__":
    main()
