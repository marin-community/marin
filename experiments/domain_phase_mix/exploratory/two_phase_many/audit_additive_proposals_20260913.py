# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Refit the additive ablation and check whether historical training can be reused.

Run with the repository environment:
uv run --no-sync python -m experiments.domain_phase_mix.exploratory.two_phase_many.audit_additive_proposals_20260913

Uses the existing comparator fitter and reference proposer unchanged. Outputs are
local, resumable, and contain no training submission. Pickles are this script's
own fit cache; the manifest rejects reuse after source, data, or fold changes.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_THREAD_LIMIT"] = "1"

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_comparator_proposals_20260909 as comparator,
)

MODEL_ID = "weibull_softplus_unscaled"
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "reference_outputs" / "additive_proposal_audit_20260913"
HISTORICAL = ROOT / "reference_outputs" / "delphi_one_phase_weibull_softplus_epoch_cap_sweep_20260902"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    panel = comparator.bench.load_panel(comparator.PANEL)
    inner = comparator.bench.heldout_inner_folds(panel)
    anchor_rows = comparator.bench.calibration_rows(panel)
    assert panel.rows == 280 and len(panel.buckets) == 39
    for train, validation in inner:
        assert set(anchor_rows).issubset(train)
        assert not set(anchor_rows).intersection(validation)
    comparator.reference_swarm_in_panel_order(panel)
    sources = [
        Path(__file__),
        Path(comparator.__file__),
        Path(comparator.bench.__file__),
        Path(comparator.models.__file__),
        Path(comparator.registry.__file__),
        Path(comparator.ms.__file__),
        HISTORICAL / "candidate_weights.csv",
        comparator.ms.DATA / "reference_policies.csv",
    ]
    manifest = {
        "model_id": MODEL_ID,
        "panel": comparator.PANEL,
        "panel_input_hashes": panel.input_hashes,
        "fit_protocol": comparator.bench.fit_protocol_core(),
        "source_hashes": {str(path): digest(path) for path in sources},
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "scikit-learn")},
        "buckets": list(panel.buckets),
        "runs": list(panel.runs),
        "anchor_rows": np.asarray(anchor_rows).tolist(),
        "inner_folds": [{"train": a.tolist(), "validation": b.tolist()} for a, b in inner],
        "epoch_cap": None,
        "kl": 0.0,
        "optimizer_start_seed": comparator.ms.START_SEED,
        "optimizer_swarm_starts": comparator.ms.PANEL_STARTS,
        "runtime_block_size": comparator.ms.MIXTURE_BLOCK_SIZE,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Inputs changed: use a new output directory rather than stale fitted artifacts")
    else:
        write_json(manifest_path, manifest)

    historical = pd.read_csv(HISTORICAL / "candidate_weights.csv")
    for target in comparator.TARGET_LABELS:
        completed = output / f"{target}_summary.json"
        if completed.exists():
            print(f"{target}: already complete", flush=True)
            continue
        group = panel.group(target)
        fits = []
        fit_dir = output / "fits" / target
        fit_dir.mkdir(parents=True, exist_ok=True)
        for index, component in enumerate(group.components):
            cache = fit_dir / f"{index:02d}.pickle"
            if cache.exists():
                with cache.open("rb") as handle:
                    fitted = pickle.load(handle)
            else:
                fitted = comparator.fit_component(panel, MODEL_ID, target, index, inner)
                temporary = cache.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    pickle.dump(fitted, handle)
                temporary.replace(cache)
            fits.append(fitted)
            print(f"{target}: {index + 1}/{len(group.components)} {component}", flush=True)
        surrogate = comparator.ObservatorySurrogate(panel, MODEL_ID, target, fits)
        objective = comparator.SurrogateObjective(panel.features.inventory, surrogate.predict)
        continuous, starts = comparator.ms.continuous_optimum(objective, panel.features.weights)
        counts, runtime_record = comparator.ms.runtime_policy(objective, continuous)
        runtime = counts / comparator.ms.MIXTURE_BLOCK_SIZE
        assert int(counts.sum()) == comparator.ms.MIXTURE_BLOCK_SIZE
        assert np.all(counts >= 0)
        assert np.isfinite(surrogate.predict(runtime[None])).all()
        pd.DataFrame(
            {
                "bucket": panel.buckets,
                "continuous": continuous,
                "runtime_count": counts,
                "weight": runtime,
                "epochs": runtime * panel.features.inventory,
            }
        ).to_csv(output / f"{target}_weights.csv", index=False)

        comparisons = []
        for candidate_id, rows in historical[historical.target == target].groupby("candidate_id"):
            indexed = rows.set_index("domain").loc[list(panel.buckets)]
            old_counts = indexed.runtime_count.to_numpy(int)
            old_weights = old_counts / comparator.ms.MIXTURE_BLOCK_SIZE
            assert np.array_equal(old_weights, indexed.weight.to_numpy(float))
            comparisons.append(
                {
                    "candidate_id": candidate_id,
                    "exact_runtime_match": bool(np.array_equal(counts, old_counts)),
                    "tv_distance": float(np.abs(counts - old_counts).sum() / (2 * comparator.ms.MIXTURE_BLOCK_SIZE)),
                    "changed_buckets": int(np.count_nonzero(counts != old_counts)),
                    "runtime_blocks_reassigned": int(np.abs(counts - old_counts).sum() // 2),
                    "new_model_prediction": float(surrogate.predict(old_weights[None])[0]),
                }
            )
        pd.DataFrame(comparisons).to_csv(output / f"{target}_historical_comparison.csv", index=False)
        mariner = comparator.reference_policy(comparator.MARINER_POLICY[target], panel.buckets)
        natural = 1 / panel.features.inventory
        natural /= natural.sum()
        fit_records = [
            {
                "component": str(component),
                "aggregation_weight": float(weight),
                "shape": fitted.shape,
                "ridge": fitted.ridge,
                "inner_cv_rmse": float(fitted.diagnostics["inner_cv_rmse"]),
            }
            for component, weight, fitted in zip(group.components, group.aggregation_weights, fits, strict=True)
        ]
        write_json(output / f"{target}_fits.json", fit_records)
        summary = {
            "target": target,
            "model_id": MODEL_ID,
            "prediction_runtime": float(surrogate.predict(runtime[None])[0]),
            "prediction_at_mariner": float(surrogate.predict(mariner[None])[0]),
            "tv_to_mariner": float(np.abs(runtime - mariner).sum() / 2),
            "tv_to_continuous": float(np.abs(runtime - continuous).sum() / 2),
            **comparator.summarize(runtime, panel.features.inventory, natural),
            "optimizer_starts": starts,
            "runtime_optimization": runtime_record,
            "historical_comparisons": comparisons,
            "reusable_historical_candidates": [row["candidate_id"] for row in comparisons if row["exact_runtime_match"]],
        }
        write_json(completed, summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
