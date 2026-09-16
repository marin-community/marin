# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "tabulate", "threadpoolctl"]
# ///
"""Optimize fixed coupled WSPU offline using the original policy grid and starts.

Read only frozen heads, canonical coordinates, and earlier policy artifacts.
The sealed prior bundle is never modified. No surrogate is refitted, no bank
labels are read, and no proposed policy has a measured loss.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

from experiments.domain_phase_mix.exploratory.two_phase_many import audit_delphi_wspu_coupling_20260906 as coupling
from experiments.domain_phase_mix.exploratory.two_phase_many import optimize_delphi_matched_policies_20260906 as policy

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
REFERENCE = SCRIPT_DIR / "reference_outputs"
DEFAULT_REFERENCE = REFERENCE / "delphi_offline_selection_20260906"
DEFAULT_FOLLOWUP = REFERENCE / "delphi_coupling_followup_20260906"
DEFAULT_OUTPUT = REFERENCE / "delphi_incumbent_coupled_optima_20260906"
TARGETS = ("uncheatable", "table9")
NEAR_OPTIMAL_TOLERANCE = 1e-6
PARITY_TOLERANCE = 1e-10
KAPPAS = (0, 1)


@dataclasses.dataclass(frozen=True)
class Evaluation:
    prediction: np.ndarray
    atomic: np.ndarray
    nonpositive_factors: np.ndarray
    minimum_factor: float


@dataclasses.dataclass(frozen=True)
class Incumbent:
    target: str
    kappa: int
    heads: coupling.AdditiveHeads
    aggregation: np.ndarray
    anchor_effects: np.ndarray
    anchor_atomic: np.ndarray

    def evaluate(self, weights: np.ndarray) -> Evaluation:
        effects = self.heads.bucket_effects(weights)
        deltas = effects - self.anchor_effects[None]
        if self.kappa == 0:
            atomic = self.heads.intercept[None] + effects.sum(axis=-1)
            counts = np.zeros(atomic.shape, dtype=int)
            minimum_factor = 1.0
        else:
            atomic, counts = coupling.coupled_values(self.anchor_atomic, deltas, self.kappa)
            minimum_factor = float((1 + self.kappa * deltas / self.anchor_atomic[None, :, None]).min())
        return Evaluation(atomic @ self.aggregation, atomic, counts, minimum_factor)

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.evaluate(weights).prediction


@dataclasses.dataclass
class TrackedPrediction:
    model: Incumbent
    calls: int = 0
    query_rows: int = 0
    nonpositive_factor_evaluations: int = 0
    nonpositive_atomic_evaluations: int = 0
    nonpositive_macro_evaluations: int = 0
    minimum_factor: float = float("inf")
    minimum_atomic_prediction: float = float("inf")
    minimum_macro_prediction: float = float("inf")

    def __call__(self, weights: np.ndarray) -> np.ndarray:
        result = self.model.evaluate(weights)
        self.calls += 1
        self.query_rows += len(weights)
        self.nonpositive_factor_evaluations += int(result.nonpositive_factors.sum())
        self.nonpositive_atomic_evaluations += int((result.atomic <= 0).sum())
        self.nonpositive_macro_evaluations += int((result.prediction <= 0).sum())
        self.minimum_factor = min(self.minimum_factor, result.minimum_factor)
        self.minimum_atomic_prediction = min(self.minimum_atomic_prediction, float(result.atomic.min()))
        self.minimum_macro_prediction = min(self.minimum_macro_prediction, float(result.prediction.min()))
        return result.prediction

    def diagnostics(self) -> dict:
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self) if field.name != "model"}


def read_coordinates(reference: Path) -> dict:
    with np.load(reference / "inputs" / "panel.npz", allow_pickle=False) as data:
        return {
            name: data[name]
            for name in ("weights", "inventory", "buckets", *(f"{target}_aggregation_weights" for target in TARGETS))
        }


def old_restart(followup: Path, target: str, cap: int, kl_index: int, start: int) -> Path:
    return (
        followup
        / "continuous_policies"
        / "policy_shards"
        / "weibull_softplus_unscaled"
        / target
        / f"cap{cap}_kl{kl_index}_start{start}.npz"
    )


def protocol_files(reference: Path, followup: Path) -> list[Path]:
    paths = [
        reference / "inputs" / "panel.npz",
        followup / "continuous_policies" / "protocol.json",
        followup / "continuous_policies" / "starts.npz",
        followup / "incumbent_coupling" / "protocol.json",
    ]
    paths.extend(followup / "incumbent_coupling" / "reconstructed_heads" / target / "fold_-1.npz" for target in TARGETS)
    paths.extend(
        old_restart(followup, target, cap, kl_index, start)
        for target in TARGETS
        for cap in policy.CAPS
        for kl_index in range(len(policy.KL_VALUES))
        for start in range(policy.PANEL_STARTS + 1)
    )
    return paths


def freeze_protocol(reference: Path, followup: Path, output: Path) -> tuple[dict, str]:
    previous = json.loads((followup / "continuous_policies" / "protocol.json").read_text())
    previous_probe = json.loads((followup / "incumbent_coupling" / "protocol.json").read_text())
    optimizer = {"method": "SLSQP", "numerical_derivatives_for_both_models": True, **policy.OPTIMIZER_OPTIONS}
    if (
        previous["caps"] != list(policy.CAPS)
        or previous["kl_values"] != list(policy.KL_VALUES)
        or previous["optimizer"] != optimizer
    ):
        raise ValueError("Policy grid or optimizer differs from the sealed original diagnostic")
    if previous["script_sha256"] != policy.benchmark.sha256(Path(policy.__file__)):
        raise ValueError("The shared policy optimizer changed since the sealed original diagnostic")
    if previous_probe["script_sha256"] != policy.benchmark.sha256(Path(coupling.__file__)):
        raise ValueError("The coupling formula changed since its sealed producer protocol")
    sources = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and str(filename).endswith(".py") and Path(filename).is_relative_to(REPO_ROOT / "experiments"):
            sources.add(Path(filename).resolve())
    record = {
        "version": 1,
        "scope": "offline continuous optima of frozen original WSPU versus fixed primary kappa1; no new surrogate fit",
        "evidence_status": "retrospective model-policy diagnostic; proposed measured losses are unknown",
        "targets": list(TARGETS),
        "kappas": list(KAPPAS),
        "caps": list(policy.CAPS),
        "kl_coefficients": list(policy.KL_VALUES),
        "objective": "aggregate surrogate BPB + coefficient * KL(weights || normalized inverse frozen inventory)",
        "starts": "Exactly the five saved original starts per cap; no additional starts or label-derived initialization",
        "first_cases": "cap8/KL0 for both targets, fixed before solving; then the remaining grid",
        "optimizer": optimizer,
        "reuse": "kappa0 restarts reused only after matching saved protocol hash, predictor values and objectives",
        "selection": "minimum objective among converged feasible restarts; otherwise best feasible marked unresolved",
        "near_optimal_objective_tolerance": NEAR_OPTIMAL_TOLERANCE,
        "within_model_spread": (
            "Objective spread and max TV to selected among all converged and near-optimal converged restarts"
        ),
        "tracking": "kappa1 tracks all objective callbacks; reused kappa0 can validate only starts and endpoints",
        "invalid_values": (
            "Nonpositive factors and predictions are retained and counted; nonfinite values raise, with no clipping"
        ),
        "parity_tolerance": PARITY_TOLERANCE,
        "inputs": {
            str(path.relative_to(REPO_ROOT)): policy.benchmark.sha256(path)
            for path in protocol_files(reference, followup)
        },
        "sources": {str(path.relative_to(REPO_ROOT)): policy.benchmark.sha256(path) for path in sorted(sources)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "old_policy_protocol_hash": hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest(),
        "forbidden": [
            "bank labels",
            "running ladder results",
            "new outcomes",
            "surrogate refits",
            "training/evaluation jobs",
        ],
    }
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    path = output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError("Frozen coupled-optimum protocol changed; use a new output directory")
    else:
        policy.atomic_json(path, record)
        (output / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    return record, digest


def load_models(data: dict, followup: Path) -> dict[tuple[str, int], Incumbent]:
    result = {}
    for target in TARGETS:
        saved = policy.benchmark.read_npz(
            followup / "incumbent_coupling" / "reconstructed_heads" / target / "fold_-1.npz"
        )
        if not np.array_equal(saved["train"], np.arange(280)) or len(saved["test"]):
            raise ValueError("Coupled continuous proposals must use the original canonical280 final heads")
        heads = coupling.AdditiveHeads(
            *(saved[key] for key in ("inventory", "intercept", "coefficients", "rate", "power", "threshold"))
        )
        anchor_effects = heads.bucket_effects(data["weights"].mean(axis=0)[None])[0]
        anchor_atomic = heads.intercept + anchor_effects.sum(axis=-1)
        if (anchor_atomic <= 0).any():
            raise ValueError("A frozen original WSPU anchor is nonpositive")
        for kappa in KAPPAS:
            result[target, kappa] = Incumbent(
                target, kappa, heads, data[f"{target}_aggregation_weights"], anchor_effects, anchor_atomic
            )
    return result


def choose_restart(rows: list[tuple[dict, np.ndarray]]) -> tuple[dict, np.ndarray, dict]:
    converged = [item for item in rows if item[0]["success"]]
    selected, weights = min(converged or rows, key=lambda item: (item[0]["objective"], item[0]["start"]))
    near = [item for item in converged if item[0]["objective"] <= selected["objective"] + NEAR_OPTIMAL_TOLERANCE]
    objectives = [item[0]["objective"] for item in converged]
    return (
        selected,
        weights,
        {
            "successful_restarts": len(converged),
            "total_restarts": len(rows),
            "selection_status": "converged_candidate" if converged else "unresolved_best_feasible_candidate",
            "converged_objective_spread": float(max(objectives) - min(objectives)) if objectives else None,
            "converged_max_tv_to_selected": (
                float(max(np.abs(item[1] - weights).sum() / 2 for item in converged)) if converged else None
            ),
            "near_optimal_restarts": len(near),
            "near_optimal_objective_tolerance": NEAR_OPTIMAL_TOLERANCE,
            "near_optimal_max_tv_to_selected": (
                float(max(np.abs(item[1] - weights).sum() / 2 for item in near)) if near else None
            ),
        },
    )


def compare_policies(
    first: Callable[[np.ndarray], np.ndarray],
    second: Callable[[np.ndarray], np.ndarray],
    weight0: np.ndarray,
    weight1: np.ndarray,
    natural: np.ndarray,
    kl: float,
) -> dict:
    query = np.stack([weight0, weight1])
    loss0, loss1 = first(query), second(query)
    penalty = kl * np.array([policy.categorical_kl(weights, natural) for weights in query])
    objective0, objective1 = loss0 + penalty, loss1 + penalty
    return {
        "between_model_tv": float(np.abs(weight0 - weight1).sum() / 2),
        "kappa0_bpb_at_kappa0": float(loss0[0]),
        "kappa0_bpb_at_kappa1": float(loss0[1]),
        "kappa1_bpb_at_kappa0": float(loss1[0]),
        "kappa1_bpb_at_kappa1": float(loss1[1]),
        "kappa0_objective_at_kappa0": float(objective0[0]),
        "kappa0_objective_at_kappa1": float(objective0[1]),
        "kappa1_objective_at_kappa0": float(objective1[0]),
        "kappa1_objective_at_kappa1": float(objective1[1]),
        "kappa1_own_objective_gain": float(objective1[0] - objective1[1]),
        "kappa0_own_objective_gain": float(objective0[1] - objective0[0]),
    }


def run_restart(
    model: Incumbent,
    followup: Path,
    output: Path,
    protocol: dict,
    digest: str,
    start: np.ndarray,
    natural: np.ndarray,
    upper: np.ndarray,
    cap: int,
    kl_index: int,
    start_index: int,
) -> tuple[dict, np.ndarray]:
    path = (
        output
        / "restart_shards"
        / model.target
        / f"kappa_{model.kappa}"
        / f"cap{cap}_kl{kl_index}_start{start_index}.npz"
    )
    if path.exists():
        saved = policy.benchmark.read_npz(path)
        if str(saved["protocol_hash"]) != digest:
            raise ValueError(f"Stale coupled-optimum shard: {path}")
        return json.loads(str(saved["diagnostics_json"])), saved["weights"]
    kl = policy.KL_VALUES[kl_index]
    tracked = TrackedPrediction(model)
    if model.kappa == 0:
        reference = old_restart(followup, model.target, cap, kl_index, start_index)
        saved = policy.benchmark.read_npz(reference)
        if str(saved["protocol_hash"]) != protocol["old_policy_protocol_hash"]:
            raise ValueError("The reused original restart has a different policy protocol")
        weights = saved["weights"]
        row = json.loads(str(saved["diagnostics_json"]))
        predictions = tracked(np.stack([weights, start]))
        checked = {
            "surrogate_bpb": float(predictions[0]),
            "kl_divergence": policy.categorical_kl(weights, natural),
            "objective": float(predictions[0]) + kl * policy.categorical_kl(weights, natural),
            "start_objective": float(predictions[1]) + kl * policy.categorical_kl(start, natural),
        }
        parity = max(abs(value - row[key]) for key, value in checked.items())
        if parity > PARITY_TOLERANCE:
            raise ValueError(f"Original policy predictor/objective parity failed: {reference}, {parity}")
        row.update(
            {
                "origin": "reused_original_kappa0",
                "reuse_max_absolute_error": parity,
                "tracking_scope": "start_and_endpoint_only",
                "new_solve_elapsed": 0.0,
            }
        )
    else:
        weights, row = policy.optimize_start(tracked, start, natural, upper, kl)
        row.update(
            {
                "origin": "new_kappa1_solve",
                "reuse_max_absolute_error": None,
                "tracking_scope": "all_objective_callbacks",
                "new_solve_elapsed": row["elapsed"],
            }
        )
    row.update({"target": model.target, "kappa": model.kappa, "cap": cap, "kl_coefficient": kl, "start": start_index})
    row.update({f"evaluated_{key}": value for key, value in tracked.diagnostics().items()})
    selected = model.evaluate(weights[None])
    row.update(
        {
            "chosen_minimum_factor": selected.minimum_factor,
            "chosen_nonpositive_factors": int(selected.nonpositive_factors.sum()),
            "chosen_nonpositive_atomic_predictions": int((selected.atomic <= 0).sum()),
            "chosen_minimum_atomic_prediction": float(selected.atomic.min()),
        }
    )
    policy.harness.atomic_save(path, {"weights": weights, "diagnostics_json": json.dumps(row), "protocol_hash": digest})
    return row, weights


def run(reference: Path, followup: Path, output: Path, protocol: dict, digest: str) -> None:
    data = read_coordinates(reference)
    models = load_models(data, followup)
    natural = 1 / data["inventory"]
    natural /= natural.sum()
    starts = policy.benchmark.read_npz(followup / "continuous_policies" / "starts.npz")
    cases = [(target, 8, 0) for target in TARGETS]
    cases.extend(
        (target, cap, index)
        for target in TARGETS
        for cap in policy.CAPS
        for index in range(len(policy.KL_VALUES))
        if (cap, index) != (8, 0)
    )
    selected_rows, all_restarts, comparisons, changes = [], [], [], []
    for target, cap, kl_index in cases:
        upper = np.minimum(1, cap / data["inventory"])
        initial = starts[f"cap_{cap}"]
        if (
            initial.shape != (5, 39)
            or max(abs(initial.sum(axis=1) - 1).max(), -initial.min(), (initial - upper).max()) > 1e-10
        ):
            raise ValueError("Saved common starts are not the expected five feasible 39-bucket mixtures")
        chosen = {}
        details = {}
        for kappa in KAPPAS:
            restarts = [
                run_restart(
                    models[target, kappa],
                    followup,
                    output,
                    protocol,
                    digest,
                    start,
                    natural,
                    upper,
                    cap,
                    kl_index,
                    index,
                )
                for index, start in enumerate(initial)
            ]
            all_restarts.extend(item[0] for item in restarts)
            best, weights, spread = choose_restart(restarts)
            chosen[kappa], details[kappa] = weights, spread
            record = {
                **best,
                **spread,
                **policy.support_distances(data["weights"], weights),
                "max_exposure": float((weights * data["inventory"]).max()),
                "measurement_status": "unknown_not_run",
            }
            for (prediction_target, strength), model in models.items():
                record[f"prediction::{prediction_target}::kappa{strength}"] = float(model.predict(weights[None])[0])
            record.update(
                {f"weight::{bucket}": float(weight) for bucket, weight in zip(data["buckets"], weights, strict=True)}
            )
            selected_rows.append(record)
        comparison = {
            "target": target,
            "cap": cap,
            "kl_coefficient": policy.KL_VALUES[kl_index],
            **compare_policies(
                models[target, 0].predict,
                models[target, 1].predict,
                chosen[0],
                chosen[1],
                natural,
                policy.KL_VALUES[kl_index],
            ),
        }
        for kappa, spread in details.items():
            comparison.update({f"kappa{kappa}_{key}": value for key, value in spread.items()})
        comparisons.append(comparison)
        delta = chosen[1] - chosen[0]
        for rank, bucket in enumerate(np.argsort(-np.abs(delta), kind="stable")[:10], 1):
            changes.append(
                {
                    "target": target,
                    "cap": cap,
                    "kl_coefficient": policy.KL_VALUES[kl_index],
                    "rank": rank,
                    "bucket": str(data["buckets"][bucket]),
                    "kappa0_weight": float(chosen[0][bucket]),
                    "kappa1_weight": float(chosen[1][bucket]),
                    "weight_change": float(delta[bucket]),
                    "exposure_change": float(delta[bucket] * data["inventory"][bucket]),
                }
            )
        policy.atomic_json(output / "case_summaries" / f"{target}_cap{cap}_kl{kl_index}.json", comparison)
        print(
            json.dumps(
                {
                    key: comparison[key]
                    for key in (
                        "target",
                        "cap",
                        "kl_coefficient",
                        "between_model_tv",
                        "kappa1_own_objective_gain",
                        "kappa0_own_objective_gain",
                    )
                }
            ),
            flush=True,
        )
    for name, rows in (
        ("selected_policies", selected_rows),
        ("restart_diagnostics", all_restarts),
        ("comparisons", comparisons),
        ("top_bucket_changes", changes),
    ):
        temporary = output / f".{name}.csv.tmp"
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(output / f"{name}.csv")
    policy.atomic_json(output / "selected_policies.json", selected_rows)
    policy.atomic_json(
        output / "summary.json",
        {
            "selected_policies": len(selected_rows),
            "restarts": len(all_restarts),
            "new_kappa1_solves": sum(row["origin"] == "new_kappa1_solve" for row in all_restarts),
            "reused_kappa0_restarts": sum(row["origin"] == "reused_original_kappa0" for row in all_restarts),
            "successful_restarts": sum(row["success"] for row in all_restarts),
            "unknown_measured_outcomes": True,
            "maximum_kappa0_reuse_error": max(
                row["reuse_max_absolute_error"] for row in all_restarts if row["kappa"] == 0
            ),
            "chosen_nonpositive_factors": sum(row["chosen_nonpositive_factors"] for row in selected_rows),
            "chosen_nonpositive_atomic_predictions": sum(
                row["chosen_nonpositive_atomic_predictions"] for row in selected_rows
            ),
            "minimum_selected_factor": min(row["chosen_minimum_factor"] for row in selected_rows),
            "kappa1_evaluated_nonpositive_factors": sum(
                row["evaluated_nonpositive_factor_evaluations"] for row in all_restarts if row["kappa"] == 1
            ),
            "kappa1_evaluated_nonpositive_atomic_predictions": sum(
                row["evaluated_nonpositive_atomic_evaluations"] for row in all_restarts if row["kappa"] == 1
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--followup-dir", type=Path, default=DEFAULT_FOLLOWUP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("freeze", "optimize"), default="optimize")
    args = parser.parse_args()
    reference, followup, output = args.reference_dir.resolve(), args.followup_dir.resolve(), args.output_dir.resolve()
    protocol, digest = freeze_protocol(reference, followup, output)
    if args.stage == "optimize":
        with threadpool_limits(limits=1):
            run(reference, followup, output, protocol, digest)


if __name__ == "__main__":
    main()
