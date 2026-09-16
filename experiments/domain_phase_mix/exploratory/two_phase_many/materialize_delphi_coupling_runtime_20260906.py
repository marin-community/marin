# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "tabulate", "threadpoolctl", "plotly"]
# ///
"""Put the nine frozen coupled-WSPU validation policies on the runtime grid.

This uses the existing cap-aware rounding and one-count exchange refinement.
The coupled objective is not separable, so no global integer-optimality claim
is made. Only local frozen predictions and mixture artifacts are read.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_coupling_validation_20260906 as continuous,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_one_phase_surrogate_challengers_20260831 as grid,
)

prior = continuous.prior
BLOCK_SIZE = grid.MIXTURE_BLOCK_SIZE
DEFAULT_OUTPUT = continuous.DEFAULT_OUTPUT.parent / "runtime_materialization"
OLD_CANDIDATES = prior.REFERENCE / "delphi_one_phase_weibull_softplus_epoch_cap_sweep_20260902" / "candidate_weights.csv"
OLD_CANDIDATES_SHA256 = "6eb8fb151b1966330b1501f2e3a6e37812f44294803b8d5d46b054f6cdc928f0"
STRENGTH_TAGS = {0.25: "025", 0.5: "05", 1.0: "1"}
TARGET_TAGS = {"uncheatable": "u", "table9": "t9"}


def candidate_id(target: str, cap: int, kappa: float) -> str:
    return f"cwspu_{TARGET_TAGS[target]}_k{STRENGTH_TAGS[kappa]}_cap{cap:02d}"


def predictor(base: prior.Incumbent, kappa: float, weights: np.ndarray) -> np.ndarray:
    return continuous.evaluate(base, kappa, weights).prediction


def runtime_policy(
    model: prior.Incumbent, kappa: float, weights: np.ndarray, inventory: np.ndarray, cap: int
) -> tuple[np.ndarray, dict]:
    """Round within the cap and refine until no improving one-count transfer remains."""
    maximum = np.floor(np.minimum(1.0, cap / inventory) * BLOCK_SIZE + 1e-12).astype(np.int64)
    if int(maximum.sum()) < BLOCK_SIZE:
        raise ValueError("The cap is infeasible on the runtime grid")
    if max(abs(weights.sum() - 1), -weights.min(), (weights * inventory - cap).max()) > 1e-10:
        raise ValueError("The frozen continuous policy violates its simplex or cap")
    predict = functools.partial(predictor, model, kappa)
    initial = grid.prefix_materializer.constrained_counts(weights, maximum)
    counts, steps = grid.refine_runtime_counts(predict, initial, maximum)
    runtime = counts / BLOCK_SIZE
    if int(counts.sum()) != BLOCK_SIZE or counts.min() < 0 or np.any(counts > maximum):
        raise ValueError("Invalid refined runtime allocation")
    if not np.array_equal(grid.prefix_materializer.runtime_counts(runtime), counts):
        raise ValueError("Runtime allocation does not survive the realizer")
    value = continuous.evaluate(model, kappa, runtime[None])
    if value.nonpositive_factors.sum() or np.any(value.atomic <= 0) or not np.isfinite(value.atomic).all():
        raise ValueError("Runtime allocation has an invalid coupled prediction")
    continuous_prediction = float(predict(weights[None])[0])
    initial_prediction = float(predict((initial / BLOCK_SIZE)[None])[0])
    runtime_prediction = float(value.prediction[0])
    if runtime_prediction > initial_prediction + grid.REFINE_TOLERANCE:
        raise ValueError("Runtime refinement increased its objective")
    return counts, {
        "continuous_prediction": continuous_prediction,
        "initial_grid_prediction": initial_prediction,
        "runtime_prediction": runtime_prediction,
        "runtime_minus_continuous_prediction": runtime_prediction - continuous_prediction,
        "continuous_to_runtime_tv": float(np.abs(runtime - weights).sum() / 2),
        "exchange_steps": steps,
        "minimum_factor": value.minimum_factor,
        "minimum_atomic_prediction": float(value.atomic.min()),
        "max_materialized_epoch": float((runtime * inventory).max()),
        "q95_materialized_epoch": float(np.sort(runtime * inventory)[round(0.95 * (len(inventory) - 1))]),
    }


def freeze(source: Path, reference: Path, followup: Path, previous: Path, output: Path) -> dict:
    source_protocol = json.loads((source / "protocol.json").read_text())
    for group in ("inputs", "sources"):
        for name, digest in source_protocol[group].items():
            if prior.policy.benchmark.sha256(prior.REPO_ROOT / name) != digest:
                raise ValueError(f"Frozen continuous dependency changed: {name}")
    if prior.policy.benchmark.sha256(OLD_CANDIDATES) != OLD_CANDIDATES_SHA256:
        raise ValueError("The old kappa0 comparator candidate table changed")
    inputs = [source / "policies.json", source / "protocol.json", previous / "selected_policies.csv", OLD_CANDIDATES]
    inputs.extend(
        followup / "incumbent_coupling" / "reconstructed_heads" / target / "fold_-1.npz" for target in prior.TARGETS
    )
    inputs.append(reference / "inputs" / "panel.npz")
    sources = [Path(__file__).resolve(), Path(grid.__file__), Path(grid.prefix_materializer.__file__)]
    record = {
        "version": 1,
        "cases": [{"target": target, "cap": cap, "kappa": k} for target, cap in continuous.CASES for k in STRENGTH_TAGS],
        "mixture_block_size": BLOCK_SIZE,
        "kl_coefficient": 0.0,
        "quantization": "Existing constrained_counts, then existing coupled-objective one-count exchange refinement",
        "optimality": "No improving one-count transfer above the fixed tolerance; global optimum not certified",
        "refine_tolerance": grid.REFINE_TOLERANCE,
        "max_exchange_steps": grid.MAX_EXCHANGE_STEPS,
        "kappa0_comparator_check": (
            "Apply the same procedure to saved kappa0 continuous policies; compare old exact-DP counts"
        ),
        "read_boundary": (
            "Frozen surrogate heads, coordinates/inventory, continuous policies, and old mixture counts only"
        ),
        "inputs": {str(path.relative_to(prior.REPO_ROOT)): prior.policy.benchmark.sha256(path) for path in inputs},
        "sources": {str(path.relative_to(prior.REPO_ROOT)): prior.policy.benchmark.sha256(path) for path in sources},
    }
    destination = output / "protocol.json"
    if destination.exists():
        if json.loads(destination.read_text()) != record:
            raise ValueError("Runtime materialization inputs or code changed; use a new output directory")
    else:
        prior.policy.atomic_json(destination, record)
        for path in sources:
            copy = output / "source_snapshot" / path.relative_to(prior.REPO_ROOT)
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(path.read_bytes())
    return record


def materialize(source: Path, reference: Path, followup: Path, previous: Path, output: Path) -> None:
    """Write the loader-compatible launch CSV and exact policy provenance."""
    data = prior.read_coordinates(reference)
    models = prior.load_models(data, followup)
    buckets = tuple(map(str, data["buckets"]))
    inventory = data["inventory"]
    policies = json.loads((source / "policies.json").read_text())
    expected = [(target, cap, k) for target, cap in continuous.CASES for k in STRENGTH_TAGS]
    if [(row["target"], row["cap"], row["kappa"]) for row in policies] != expected:
        raise ValueError("Continuous policies do not contain the exact ordered nine requested cells")
    weights_rows, summaries = [], []
    for row in policies:
        target, cap, kappa = row["target"], row["cap"], row["kappa"]
        if row["kl_coefficient"] != 0 or set(row["weights"]) != set(buckets):
            raise ValueError("Continuous policy has a different objective or runtime bucket set")
        if hashlib.sha256(json.dumps(row["weights"], sort_keys=True).encode()).hexdigest() != row["weights_sha256"]:
            raise ValueError("Continuous policy weights changed")
        weights = np.asarray([row["weights"][bucket] for bucket in buckets])
        counts, diagnostics = runtime_policy(models[target, 1], kappa, weights, inventory, cap)
        if abs(diagnostics["continuous_prediction"] - row["surrogate_bpb"]) > prior.PARITY_TOLERANCE:
            raise ValueError("Frozen continuous prediction parity failed")
        name = candidate_id(target, cap, kappa)
        mapping = {bucket: int(count) for bucket, count in zip(buckets, counts, strict=True)}
        runtime_weights = {bucket: count / BLOCK_SIZE for bucket, count in mapping.items()}
        summary = {
            "candidate_id": name,
            "policy_id": row["policy_id"],
            "target": target,
            "epoch_cap": cap,
            "kappa": kappa,
            "kl_coefficient": 0.0,
            "target_flops": 3e18,
            "measurement_status": "unknown_not_run",
            "continuous_weights_sha256": row["weights_sha256"],
            "runtime_counts_sha256": hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest(),
            "runtime_weights_sha256": hashlib.sha256(json.dumps(runtime_weights, sort_keys=True).encode()).hexdigest(),
            **diagnostics,
            "runtime_counts": mapping,
            "runtime_weights": runtime_weights,
        }
        summaries.append(summary)
        for bucket, scale in zip(buckets, inventory, strict=True):
            weights_rows.append(
                {
                    "candidate_id": name,
                    "target": target,
                    "target_label": "Uncheatable" if target == "uncheatable" else "Table-9 macro",
                    "epoch_cap": cap,
                    "kappa": kappa,
                    "domain": bucket,
                    "runtime_count": mapping[bucket],
                    "weight": runtime_weights[bucket],
                    "materialized_epochs": float(scale * runtime_weights[bucket]),
                }
            )
    coordinates = [tuple(row["runtime_counts"][b] for b in buckets) for row in summaries]
    if len(set(coordinates)) != len(expected):
        raise ValueError("Requested policies alias on the runtime grid; reconcile release identities before launch")
    pd.DataFrame(weights_rows).to_csv(output / "candidate_weights.csv", index=False)
    prior.policy.atomic_json(output / "candidate_mapping.json", summaries)
    pd.DataFrame(
        [{k: v for k, v in row.items() if k not in {"runtime_counts", "runtime_weights"}} for row in summaries]
    ).to_csv(output / "candidate_mapping.csv", index=False)

    previous_rows = pd.read_csv(previous / "selected_policies.csv")
    old_rows = pd.read_csv(OLD_CANDIDATES)
    comparators = []
    for target, cap in continuous.CASES:
        selected = previous_rows.loc[
            (previous_rows.target == target)
            & (previous_rows.cap == cap)
            & (previous_rows.kappa == 0)
            & (previous_rows.kl_coefficient == 0)
        ]
        if len(selected) != 1:
            raise ValueError("Missing or ambiguous saved kappa0 comparator")
        weights = np.asarray([selected.iloc[0][f"weight::{bucket}"] for bucket in buckets], dtype=float)
        counts, diagnostics = runtime_policy(models[target, 0], 0.0, weights, inventory, cap)
        old = old_rows.loc[(old_rows.target == target) & (old_rows.epoch_cap == cap)].set_index("domain")
        old_counts = old.loc[list(buckets), "runtime_count"].to_numpy(dtype=np.int64)
        comparators.append(
            {
                "target": target,
                "epoch_cap": cap,
                "old_candidate_id": str(old.candidate_id.iloc[0]),
                "exact_counts_match": bool(np.array_equal(counts, old_counts)),
                "old_to_refined_kappa0_tv": float(np.abs(counts - old_counts).sum() / (2 * BLOCK_SIZE)),
                "old_to_continuous_kappa0_tv": float(np.abs(old_counts / BLOCK_SIZE - weights).sum() / 2),
                "old_kappa0_prediction": float(predictor(models[target, 0], 0, (old_counts / BLOCK_SIZE)[None])[0]),
                **diagnostics,
                "refined_kappa0_counts": {bucket: int(count) for bucket, count in zip(buckets, counts, strict=True)},
            }
        )
    prior.policy.atomic_json(output / "kappa0_comparator_parity.json", comparators)
    prior.policy.atomic_json(
        output / "summary.json",
        {
            "candidate_ids": [row["candidate_id"] for row in summaries],
            "candidate_weights_sha256": prior.policy.benchmark.sha256(output / "candidate_weights.csv"),
            "runtime_distinct_candidates": len(coordinates),
            "maximum_continuous_to_runtime_tv": max(row["continuous_to_runtime_tv"] for row in summaries),
            "maximum_runtime_prediction_penalty": max(row["runtime_minus_continuous_prediction"] for row in summaries),
            "all_kappa0_comparator_counts_match": all(row["exact_counts_match"] for row in comparators),
            "no_jobs_launched": True,
        },
    )
    artifacts = {
        str(path.relative_to(output)): prior.policy.benchmark.sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    prior.policy.atomic_json(output / "artifact_manifest.json", artifacts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=continuous.DEFAULT_OUTPUT)
    parser.add_argument("--reference-dir", type=Path, default=prior.DEFAULT_REFERENCE)
    parser.add_argument("--followup-dir", type=Path, default=prior.DEFAULT_FOLLOWUP)
    parser.add_argument("--previous-dir", type=Path, default=prior.DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = [
        path.resolve()
        for path in (args.source_dir, args.reference_dir, args.followup_dir, args.previous_dir, args.output_dir)
    ]
    freeze(*paths)
    with threadpool_limits(limits=1):
        materialize(*paths)
    print((paths[-1] / "summary.json").read_text())


if __name__ == "__main__":
    main()
