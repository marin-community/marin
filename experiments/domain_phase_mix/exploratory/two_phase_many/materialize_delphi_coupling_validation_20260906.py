# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "tabulate", "threadpoolctl"]
# ///
"""Materialize the nine fixed coupled-WSPU policies for prospective validation.

This command only optimizes saved surrogates locally. It never refits models,
reads bank labels or live results, or launches training/evaluation jobs.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

from experiments.domain_phase_mix.exploratory.two_phase_many import optimize_delphi_incumbent_coupling_20260906 as prior

policy = prior.policy
CASES = (("uncheatable", 6), ("table9", 6), ("table9", 8))
STRENGTHS = ((0.25, "0p25"), (0.5, "0p5"), (1.0, "1"))
DEFAULT_OUTPUT = prior.REFERENCE / "delphi_coupling_validation_3e18_20260906" / "offline_materialization"


def evaluate(base: prior.Incumbent, kappa: float, weights: np.ndarray) -> prior.Evaluation:
    deltas = base.heads.bucket_effects(weights) - base.anchor_effects[None]
    atomic, counts = prior.coupling.coupled_values(base.anchor_atomic, deltas, kappa)
    minimum_factor = float((1 + kappa * deltas / base.anchor_atomic[None, :, None]).min())
    return prior.Evaluation(atomic @ base.aggregation, atomic, counts, minimum_factor)


def tracked_prediction(base: prior.Incumbent, kappa: float, tracking: dict, weights: np.ndarray) -> np.ndarray:
    value = evaluate(base, kappa, weights)
    tracking["query_rows"] += len(weights)
    tracking["minimum_factor"] = min(tracking["minimum_factor"], value.minimum_factor)
    tracking["minimum_atomic_prediction"] = min(tracking["minimum_atomic_prediction"], float(value.atomic.min()))
    tracking["nonpositive_factors"] += int(value.nonpositive_factors.sum())
    tracking["nonpositive_atomic_predictions"] += int((value.atomic <= 0).sum())
    return value.prediction


def existing_restart(previous: Path, target: str, cap: int, start: int) -> Path:
    return previous / "restart_shards" / target / "kappa_1" / f"cap{cap}_kl0_start{start}.npz"


def freeze(reference: Path, followup: Path, previous: Path, output: Path) -> tuple[dict, str]:
    old = json.loads((previous / "protocol.json").read_text())
    expected = {"method": "SLSQP", "numerical_derivatives_for_both_models": True, **policy.OPTIMIZER_OPTIONS}
    if old["optimizer"] != expected or old["near_optimal_objective_tolerance"] != prior.NEAR_OPTIMAL_TOLERANCE:
        raise ValueError("The original optimizer or restart selection has changed")
    sources = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and str(filename).endswith(".py") and Path(filename).is_relative_to(prior.REPO_ROOT / "experiments"):
            sources.add(Path(filename).resolve())
    source_hashes = {str(path.relative_to(prior.REPO_ROOT)): policy.benchmark.sha256(path) for path in sorted(sources)}
    for name, digest in old["sources"].items():
        if source_hashes[name] != digest:
            raise ValueError(f"An imported frozen predecessor source changed: {name}")
    inputs = [
        reference / "inputs" / "panel.npz",
        followup / "continuous_policies" / "starts.npz",
        previous / "protocol.json",
    ]
    inputs.extend(
        followup / "incumbent_coupling" / "reconstructed_heads" / target / "fold_-1.npz" for target in prior.TARGETS
    )
    inputs.extend(existing_restart(previous, target, cap, start) for target, cap in CASES for start in range(5))
    record = {
        "version": 1,
        "scope": "Frozen nine-policy prospective 3e18 validation materialization; no jobs launched by this producer",
        "evidence_status": "Chosen after retrospective development, frozen before these nine prospective outcomes",
        "cases": [{"target": target, "cap": cap, "kappa": kappa} for target, cap in CASES for kappa, _ in STRENGTHS],
        "target_budget": 3e18,
        "kl_coefficient": 0.0,
        "objective": "Taskwise fixed coupled-WSPU predictions aggregated in the original target's BPB units",
        "law": "A * (1 + (product_b(1 + kappa * delta_b / A) - 1) / kappa)",
        "anchor": "Original canonical280 training-mean mixture, separately for each saved atomic head",
        "cap_definition": "0 <= weight_b <= min(1, cap / frozen_inventory_b); sum(weights) = 1",
        "starts": "Exactly the original five saved starts per cap; no additional or outcome-derived initialization",
        "optimizer": expected,
        "selection": "Minimum objective among converged feasible restarts; fail if a policy has none",
        "near_optimal_objective_tolerance": prior.NEAR_OPTIMAL_TOLERANCE,
        "reuse": (
            "All 15 kappa1 restart endpoints reused only after protocol, prediction, KL, and start-objective parity"
        ),
        "tracking": (
            "New solves track every objective callback; reused kappa1 retains the original full-trajectory diagnostics"
        ),
        "invalid_values": "Nonpositive factors and predictions are counted and retained; nonfinite predictions raise",
        "read_boundary": (
            "Only panel coordinates/inventory/aggregation arrays, saved heads/starts, and prior optimizer outputs"
        ),
        "forbidden": ["bank labels", "running ladder", "prospective outcomes", "surrogate refits", "job launches"],
        "inputs": {str(path.relative_to(prior.REPO_ROOT)): policy.benchmark.sha256(path) for path in inputs},
        "sources": source_hashes,
        "previous_protocol_hash": hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest(),
        "parity_tolerance": prior.PARITY_TOLERANCE,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    path = output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError("Frozen validation protocol changed; use a new output directory")
    else:
        policy.atomic_json(path, record)
        for source in sorted(sources):
            destination = output / "source_snapshot" / source.relative_to(prior.REPO_ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    return record, digest


def restart(
    base: prior.Incumbent,
    kappa: float,
    start: np.ndarray,
    natural: np.ndarray,
    upper: np.ndarray,
    previous: Path,
    destination: Path,
    protocol: dict,
    digest: str,
    cap: int,
    index: int,
) -> tuple[dict, np.ndarray]:
    if destination.exists():
        saved = policy.benchmark.read_npz(destination)
        if str(saved["protocol_hash"]) != digest:
            raise ValueError(f"Stale restart: {destination}")
        return json.loads(str(saved["diagnostics_json"])), saved["weights"]
    tracking = {
        "query_rows": 0,
        "minimum_factor": float("inf"),
        "minimum_atomic_prediction": float("inf"),
        "nonpositive_factors": 0,
        "nonpositive_atomic_predictions": 0,
    }
    predict = functools.partial(tracked_prediction, base, kappa, tracking)
    if kappa == 1:
        saved = policy.benchmark.read_npz(existing_restart(previous, base.target, cap, index))
        if str(saved["protocol_hash"]) != protocol["previous_protocol_hash"]:
            raise ValueError("Reused kappa1 restart has a different original protocol")
        weights, row = saved["weights"], json.loads(str(saved["diagnostics_json"]))
        predictions = predict(np.stack([weights, start]))
        checked = {
            "surrogate_bpb": float(predictions[0]),
            "objective": float(predictions[0]),
            "start_objective": float(predictions[1]),
            "kl_divergence": policy.categorical_kl(weights, natural),
        }
        error = max(abs(value - row[key]) for key, value in checked.items())
        if error > prior.PARITY_TOLERANCE:
            raise ValueError(f"Kappa1 prediction/objective parity failed: {error}")
        row.update({"origin": "reused_frozen_kappa1", "reuse_max_absolute_error": error})
    else:
        weights, row = policy.optimize_start(predict, start, natural, upper, 0.0)
        row.update({"origin": "new_fixed_strength_solve", "reuse_max_absolute_error": None})
        row.update({f"evaluated_{name}": value for name, value in tracking.items()})
    selected = evaluate(base, kappa, weights[None])
    row.update(
        {
            "target": base.target,
            "cap": cap,
            "kappa": kappa,
            "kl_coefficient": 0.0,
            "start": index,
            "chosen_minimum_factor": selected.minimum_factor,
            "chosen_nonpositive_factors": int(selected.nonpositive_factors.sum()),
            "chosen_nonpositive_atomic_predictions": int((selected.atomic <= 0).sum()),
            "chosen_minimum_atomic_prediction": float(selected.atomic.min()),
        }
    )
    if max(abs(weights.sum() - 1), -weights.min(), (weights - upper).max()) > 1e-10:
        raise ValueError("Materialized restart weights violate their frozen cap/simplex")
    policy.harness.atomic_save(
        destination, {"weights": weights, "diagnostics_json": json.dumps(row), "protocol_hash": digest}
    )
    return row, weights


def materialize(reference: Path, followup: Path, previous: Path, output: Path, protocol: dict, digest: str) -> None:
    data = prior.read_coordinates(reference)
    models = prior.load_models(data, followup)
    starts = policy.benchmark.read_npz(followup / "continuous_policies" / "starts.npz")
    natural = 1 / data["inventory"]
    natural /= natural.sum()
    records, diagnostics, long_weights = [], [], []
    for target, cap in CASES:
        initial, upper = starts[f"cap_{cap}"], np.minimum(1, cap / data["inventory"])
        if (
            initial.shape != (5, 39)
            or max(abs(initial.sum(axis=1) - 1).max(), -initial.min(), (initial - upper).max()) > 1e-10
        ):
            raise ValueError("Saved starts must be the original five feasible 39-bucket mixtures")
        for kappa, tag in STRENGTHS:
            name = f"wspu_coupled_{target}_cap{cap}_kappa{tag}_kl0"
            rows = [
                restart(
                    models[target, 1],
                    kappa,
                    start,
                    natural,
                    upper,
                    previous,
                    output / "restart_shards" / name / f"start{index}.npz",
                    protocol,
                    digest,
                    cap,
                    index,
                )
                for index, start in enumerate(initial)
            ]
            best, weights, spread = prior.choose_restart(rows)
            if not best["success"]:
                raise ValueError(f"No converged feasible policy for {name}")
            mapping = {str(bucket): float(weight) for bucket, weight in zip(data["buckets"], weights, strict=True)}
            record = {
                "policy_id": name,
                "target_budget": 3e18,
                **best,
                **spread,
                **policy.support_distances(data["weights"], weights),
                "max_exposure": float((weights * data["inventory"]).max()),
                "weights": mapping,
                "weights_sha256": hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest(),
                "protocol_hash": digest,
                "measurement_status": "unknown_not_run",
            }
            records.append(record)
            diagnostics.extend({"policy_id": name, **row} for row, _ in rows)
            long_weights.extend(
                {"policy_id": name, "bucket": bucket, "weight": weight} for bucket, weight in mapping.items()
            )
            policy.atomic_json(output / "policies" / f"{name}.json", record)
            print(
                json.dumps(
                    {key: record[key] for key in ("policy_id", "surrogate_bpb", "successful_restarts", "weights_sha256")}
                ),
                flush=True,
            )
    policy.atomic_json(output / "policies.json", records)
    for name, rows in (
        ("policies", [{key: value for key, value in row.items() if key != "weights"} for row in records]),
        ("restart_diagnostics", diagnostics),
        ("weights", long_weights),
    ):
        temporary = output / f".{name}.csv.tmp"
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(output / f"{name}.csv")
    policy.atomic_json(
        output / "summary.json",
        {
            "policies": len(records),
            "restarts": len(diagnostics),
            "successful_restarts": sum(row["success"] for row in diagnostics),
            "new_strength_restarts": sum(row["origin"] == "new_fixed_strength_solve" for row in diagnostics),
            "reused_kappa1_restarts": sum(row["origin"] == "reused_frozen_kappa1" for row in diagnostics),
            "maximum_kappa1_reuse_error": max(
                row["reuse_max_absolute_error"] for row in diagnostics if row["kappa"] == 1
            ),
            "selected_nonpositive_factors": sum(row["chosen_nonpositive_factors"] for row in records),
            "selected_nonpositive_atomic_predictions": sum(
                row["chosen_nonpositive_atomic_predictions"] for row in records
            ),
            "unknown_measured_outcomes": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=prior.DEFAULT_REFERENCE)
    parser.add_argument("--followup-dir", type=Path, default=prior.DEFAULT_FOLLOWUP)
    parser.add_argument("--previous-dir", type=Path, default=prior.DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("freeze", "materialize"), default="materialize")
    args = parser.parse_args()
    reference, followup, previous, output = (
        path.resolve() for path in (args.reference_dir, args.followup_dir, args.previous_dir, args.output_dir)
    )
    protocol, digest = freeze(reference, followup, previous, output)
    if args.stage == "materialize":
        with threadpool_limits(limits=1):
            materialize(reference, followup, previous, output, protocol, digest)


if __name__ == "__main__":
    main()
