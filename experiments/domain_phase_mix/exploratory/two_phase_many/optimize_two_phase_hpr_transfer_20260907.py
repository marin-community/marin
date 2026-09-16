# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0", "scikit-learn==1.7.2",
#   "cvxpy==1.7.5", "fsspec==2026.1.0", "gcsfs==2026.1.0", "plotly==6.5.1",
#   "tabulate==0.9.0", "threadpoolctl==3.6.0",
# ]
# ///
"""Audit unconstrained simplex optima of the frozen LINK2-003 replacement.

HPR uses its original 1e-12 exposure floor inside the power value; the derivative
is zero below that floor. Its TV term uses the zero subgradient at equal phase
weights. The aggregate Weibull derivative retains the earlier audit's numerical
zero safeguard. No response floor, policy support, or epoch cap is imposed.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import audit_two_phase_link_softmax_solver_20260907 as softmax_audit
import benchmark_two_phase_hpr_transfer_20260907 as transfer
import numpy as np
import optimize_two_phase_link_transfer_20260907 as raw
import pandas as pd
from scipy.special import expit

POWER_FLOOR = 1e-12
CHECK_SEED = 20260907


class Arm(StrEnum):
    AGGREGATE = "aggregate"
    HPR_TIED = "hpr_tied"
    HPR = "hpr"
    TRANSFER = "hpr_aggregate_replacement"


@dataclass(frozen=True)
class HprResponse:
    """Exact HPR blocks with redundant benefit coefficients combined."""

    intercept: float
    exponent: float
    forgetting: float
    late_multiplier: float
    threshold: float
    bucket_benefit: np.ndarray
    family_benefit: np.ndarray
    family_harm: np.ndarray
    member_harm: np.ndarray
    tv: float
    family_index: np.ndarray
    family_count: np.ndarray
    c0: np.ndarray
    c1: np.ndarray

    def power(self, exposure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        safe = np.maximum(exposure, POWER_FLOOR)
        value = safe**self.exponent
        derivative = np.where(exposure > POWER_FLOOR, self.exponent * safe ** (self.exponent - 1), 0.0)
        return value, derivative

    def harm(self, exposure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        argument = np.log1p(np.maximum(exposure, 0.0)) - self.threshold
        softplus = np.logaddexp(0.0, argument)
        return softplus**2, 2 * softplus * expit(argument) / (1 + exposure)

    def value_gradient(self, weights: np.ndarray) -> tuple[float, np.ndarray, dict[str, Any]]:
        """Evaluate the original retained-state and replay response."""
        early = self.c0 * weights[0]
        retention = np.exp(-self.forgetting * (1 - weights[1]))
        state = retention * early + self.late_multiplier * self.c1 * weights[1]
        family = np.bincount(self.family_index, weights=state, minlength=len(self.family_count))
        pb, dpb = self.power(state)
        pf, dpf = self.power(family)
        hb, dhb = self.harm(state)
        hf, dhf = self.harm(family)
        member = self.member_harm[self.family_index] / self.family_count[self.family_index]
        signed_shift = np.sign(weights[0] - weights[1])
        value = (
            self.intercept
            - self.bucket_benefit @ pb
            - self.family_benefit @ pf
            + self.family_harm @ hf
            + member @ hb
            + self.tv * np.abs(weights[0] - weights[1]).sum() / 2
        )
        ds = (
            -self.bucket_benefit * dpb
            + (-self.family_benefit * dpf + self.family_harm * dhf)[self.family_index]
            + member * dhb
        )
        gradient = np.stack(
            [
                ds * self.c0 * retention + self.tv * signed_shift / 2,
                ds * (self.forgetting * retention * early + self.late_multiplier * self.c1) - self.tv * signed_shift / 2,
            ]
        )
        return (
            float(value),
            gradient,
            {
                "hpr_power_floor_buckets": int(np.sum(state <= POWER_FLOOR)),
                "hpr_power_floor_families": int(np.sum(family <= POWER_FLOOR)),
            },
        )


def hpr_response(fitted: Any) -> HprResponse:
    """Read only blocks present in the frozen hierarchical phase replay model."""
    source = transfer.controls.baseline.hierarchical_grp
    assert fitted.config.variant is source.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY
    data = fitted.dataset
    families = {name: index for index, name in enumerate(data.family_names)}
    buckets = {name: index for index, name in enumerate(data.domains)}
    family_index = np.empty(data.m, dtype=int)
    for index, members in enumerate(data.family_members):
        family_index[members] = index
    bucket_benefit = np.zeros(data.m)
    family_benefit = np.zeros(len(families))
    family_harm = np.zeros(len(families))
    member_harm = np.zeros(len(families))
    tv = 0.0
    design = source.build_design(data, fitted.config)
    for name, coefficient in zip(design.names, fitted.coefficients, strict=True):
        prefix, _, label = name.partition(":")
        if prefix in ("singleton_signal", "bucket_excess_signal"):
            bucket_benefit[buckets[label]] += coefficient
        elif prefix == "pooled_base_signal":
            bucket_benefit[data.family_members[families[label]]] += coefficient
        elif prefix == "family_coverage_signal":
            family_benefit[families[label]] += coefficient
        elif prefix == "family_overexposure":
            family_harm[families[label]] += coefficient
        elif prefix == "family_member_replay":
            member_harm[families[label]] += coefficient
        elif prefix == "phase_shift_tv":
            tv = float(coefficient)
        else:
            raise ValueError(f"Unimplemented frozen HPR feature: {name}")
    shape = fitted.config.shape
    return HprResponse(
        fitted.intercept,
        shape.exponent,
        shape.forgetting_rate,
        shape.late_multiplier,
        shape.penalty_threshold,
        bucket_benefit,
        family_benefit,
        family_harm,
        member_harm,
        tv,
        family_index,
        np.bincount(family_index),
        data.c0,
        data.c1,
    )


@dataclass(frozen=True)
class Surface:
    """Frozen aggregate replacement and its two component controls."""

    aggregate: raw.ObjectiveModel
    hpr: HprResponse
    arm: Arm

    @property
    def buckets(self) -> int:
        return self.aggregate.buckets

    @property
    def c0(self) -> np.ndarray:
        return self.aggregate.c0

    @property
    def c1(self) -> np.ndarray:
        return self.aggregate.c1

    @property
    def phase_fraction(self) -> np.ndarray:
        return self.aggregate.phase_fraction

    def value_gradient(self, weights: np.ndarray) -> tuple[float, np.ndarray, dict[str, Any]]:
        if self.arm == Arm.AGGREGATE:
            value, gradient, diagnostic = self.aggregate.value_gradient(weights)
        elif self.arm in (Arm.HPR, Arm.HPR_TIED):
            value, gradient, diagnostic = self.hpr.value_gradient(weights)
        else:
            aggregate = self.phase_fraction * weights[0] + (1 - self.phase_fraction) * weights[1]
            tied = np.stack([aggregate, aggregate])
            value, gradient, diagnostic = self.aggregate.value_gradient(weights)
            h, dh, hdiag = self.hpr.value_gradient(weights)
            ht, dht, _ = self.hpr.value_gradient(tied)
            value += h - ht
            tied_derivative = dht.sum(axis=0)
            gradient += dh - np.stack(
                [self.phase_fraction * tied_derivative, (1 - self.phase_fraction) * tied_derivative]
            )
            diagnostic.update(hdiag, hpr_contrast=h - ht, aggregate_bpb=value - h + ht)
        floor = float(self.aggregate.aggregation @ self.aggregate.floor)
        diagnostic.update(
            negative_bpb=bool(value < 0),
            weighted_aggregate_floor=floor,
            gap_to_aggregate_floor=value - floor,
            below_aggregate_floor=bool(value < floor),
            floor_scope="The tied aggregate has this floor; HPR and its additive transplant need not preserve it.",
        )
        return value, gradient, diagnostic


def load_models(context: str, objective: str) -> tuple[raw.ObjectiveModel, Any, dict[str, str]]:
    fold = "full" if context == "final" else context
    cell = transfer.PREVIOUS / "controls/hierarchical_phase_replay" / objective / fold
    complete = json.loads((cell / "complete.json").read_text())
    paths = [
        Path(__file__),
        Path(raw.__file__),
        Path(softmax_audit.__file__),
        Path(transfer.__file__),
        Path(raw.driver.__file__),
        transfer.OUTPUT / "PROTOCOL.md",
        transfer.PREVIOUS / "inputs/panel.npz",
        transfer.PREVIOUS / "inputs/splits.npz",
        transfer.PREVIOUS / "inputs/single_phase.py",
        cell / "complete.json",
        transfer.OUTPUT / "aggregate_replacement/predictions.csv",
    ]
    for name, digest in complete["sha256"].items():
        path = cell / name
        assert transfer.file_hash(path) == digest, path
        paths.append(path)
    for name, digest in complete["protocol"]["source_sha256"].items():
        path = transfer.controls.REPO_ROOT / name
        assert transfer.file_hash(path) == digest, path
        paths.append(path)
    with (cell / "model.pkl").open("rb") as handle:
        fitted = pickle.load(handle)
    aggregate = raw.load_model(transfer.PREVIOUS, context, objective, "aggregate")
    for index in range(len(aggregate.floor)):
        paths.append(transfer.PREVIOUS / "spines" / context / f"{objective}_c{index}.json")
    return aggregate, fitted, {str(path): transfer.file_hash(path) for path in sorted(set(paths))}


def check_surface(model: Surface, fitted: Any, panel: dict, reference: np.ndarray) -> dict[str, float]:
    predictions = np.asarray([model.value_gradient(w)[0] for w in panel["weights"]])
    parity = float(np.max(np.abs(predictions - reference)))
    hpr_prediction = np.asarray([model.hpr.value_gradient(w)[0] for w in panel["weights"]])
    hpr_parity = float(np.max(np.abs(hpr_prediction - fitted.predict(panel["weights"]))))
    rng = np.random.default_rng(CHECK_SEED)
    gradient_error, tied_error = 0.0, 0.0
    for _ in range(4):
        w = rng.dirichlet(np.full(model.buckets, 3.0), size=2)
        if model.arm in (Arm.AGGREGATE, Arm.HPR_TIED):
            w[1] = w[0]
        _, gradient, _ = model.value_gradient(w)
        for _ in range(8):
            direction = rng.normal(size=w.shape)
            direction -= direction.mean(axis=1, keepdims=True)
            if model.arm in (Arm.AGGREGATE, Arm.HPR_TIED):
                direction[1] = direction[0]
            direction /= np.linalg.norm(direction)
            analytic = float(np.sum(gradient * direction))
            numeric = (
                model.value_gradient(w + 1e-6 * direction)[0] - model.value_gradient(w - 1e-6 * direction)[0]
            ) / 2e-6
            gradient_error = max(gradient_error, abs(numeric - analytic) / max(1.0, abs(analytic)))
        tied = np.stack([w[0], w[0]])
        if model.arm == Arm.TRANSFER:
            tied_error = max(tied_error, abs(model.value_gradient(tied)[0] - model.aggregate.value_gradient(tied)[0]))
    if parity > 1e-10 or hpr_parity > 1e-10 or gradient_error > 2e-6 or tied_error > 1e-12:
        raise ValueError(f"Parity check failed: {parity}, {hpr_parity}, {gradient_error}, {tied_error}")
    return {
        "prediction_parity_max_abs": parity,
        "hpr_source_parity_max_abs": hpr_parity,
        "interior_tangent_gradient_max_relative_error": gradient_error,
        "tied_restriction_max_abs": tied_error,
    }


def optimize_cell(
    output: Path,
    context: str,
    objective: str,
    model: Surface,
    checks: dict,
    identity: dict,
    panel: dict,
    train: np.ndarray,
    tied_reference: dict,
    other_tied: dict | None,
) -> dict:
    destination = output / "optima" / f"{context}_{objective}_{model.arm}.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        assert saved["input_hashes"] == identity, f"changed optimization inputs: {destination}"
        return saved
    tied = model.arm in (Arm.AGGREGATE, Arm.HPR_TIED)
    # Frozen solvers use this structural interface; their original annotation is concrete.
    solver_model: Any = model
    starts = raw.starts_for(
        solver_model, panel["weights"][train], tied, None if tied else np.asarray(tied_reference["weights"])
    )
    if other_tied is not None:
        starts.insert(1, ("other_fitted_tied_optimum", np.asarray(other_tied["weights"])))
    results = []
    for name, weights in starts:
        _, result = raw.solve_start(solver_model, weights, tied, name)
        results.append(result | {"solver": "SLSQP", "predicted_bpb": result["retained_bpb"]})
    if any(not result["success"] for result in results):
        best_slsqp = min(results, key=lambda result: result["predicted_bpb"])
        soft_starts = [*starts, ("best_slsqp_endpoint", np.asarray(best_slsqp["weights"]))]
        for name, weights in soft_starts:
            result = softmax_audit.optimize_start(solver_model, weights, tied, name)
            results.append(result | {"solver": "softmax L-BFGS-B"})
    successful = [result for result in results if result["success"]]
    best_found = min(results, key=lambda result: result["predicted_bpb"])
    selected = min(successful or results, key=lambda result: result["predicted_bpb"])
    policy = np.asarray(selected["weights"])
    alpha = float(panel["alpha"])
    pair_tv = [
        raw.policy_tv(np.asarray(left["weights"]), np.asarray(right["weights"]), alpha)
        for i, left in enumerate(successful)
        for right in successful[i + 1 :]
    ]
    observed = panel[f"{objective}_aggregate"]
    minimum_observed = float(observed[~panel["calibration_mask"]].min())
    record = {
        "context": context,
        "objective": objective,
        "arm": model.arm,
        "input_hashes": identity,
        "checks": checks,
        "predicted_bpb": selected["predicted_bpb"],
        "minimum_found_bpb": best_found["predicted_bpb"],
        "minimum_found_solver_success": best_found["success"],
        "selected_solver_success": selected["success"],
        "selected_solver": selected["solver"],
        "selected_start": selected["name"],
        "successful_starts": len(successful),
        "total_starts": len(results),
        "tied_reference_bpb": tied_reference["predicted_bpb"] if not tied else selected["predicted_bpb"],
        "predicted_gain_over_own_tied_optimum": (
            tied_reference["predicted_bpb"] - selected["predicted_bpb"] if not tied else 0.0
        ),
        "best_observed_bpb": minimum_observed,
        "predicted_improvement_over_best_observed": minimum_observed - selected["predicted_bpb"],
        "weights": policy.tolist(),
        "max_phase_bucket_weight": float(policy.max()),
        "max_total_epochs": float(np.max(model.c0 * policy[0] + model.c1 * policy[1])),
        "phase_tv": float(np.abs(policy[0] - policy[1]).sum() / 2),
        "full_design_support": raw.support_audit(policy, panel["weights"], alpha),
        "training_support": raw.support_audit(policy, panel["weights"][train], alpha),
        "successful_start_policy_tv_max": max(pair_tv, default=0.0),
        "successful_start_bpb_spread": float(np.ptp([r["predicted_bpb"] for r in successful])) if successful else None,
        "diagnostic": selected["clips"],
        "any_start_negative_bpb": any(r["clips"]["negative_bpb"] for r in results),
        "any_start_below_aggregate_floor": any(r["clips"]["below_aggregate_floor"] for r in results),
        "starts": results,
        "limitations": [
            "Local solver success does not certify a minimum or a global optimum.",
            "Finite softmax logits approximate boundaries; the source HPR power floor remains unchanged.",
            "Off-diagonal aggregate-floor violations are exposed without a response clamp.",
        ],
    }
    raw.driver.write_json(destination, record)
    print(
        json.dumps(
            {
                key: record[key]
                for key in ("context", "objective", "arm", "predicted_bpb", "successful_starts", "total_starts")
            }
        ),
        flush=True,
    )
    return record


def summaries(output: Path, records: list[dict], alpha: float) -> None:
    rows = []
    for record in records:
        row = {key: value for key, value in record.items() if not isinstance(value, (dict, list))}
        row.update({f"full_{key}": value for key, value in record["full_design_support"].items()})
        row.update(record["diagnostic"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "optima/summary.csv", index=False)
    distances = []
    for objective in raw.driver.OBJECTIVES:
        for arm in Arm:
            cells = [r for r in records if r["objective"] == objective and r["arm"] == arm]
            for i, left in enumerate(cells):
                for right in cells[i + 1 :]:
                    l, r = np.asarray(left["weights"]), np.asarray(right["weights"])
                    la, ra = alpha * l[0] + (1 - alpha) * l[1], alpha * r[0] + (1 - alpha) * r[1]
                    distances.append(
                        {
                            "objective": objective,
                            "arm": arm,
                            "left_context": left["context"],
                            "right_context": right["context"],
                            "policy_tv": raw.policy_tv(l, r, alpha),
                            "aggregate_tv": float(np.abs(la - ra).sum() / 2),
                        }
                    )
    pd.DataFrame(distances).to_csv(output / "optima/context_stability.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=transfer.OUTPUT)
    parser.add_argument("--contexts", nargs="+", choices=raw.driver.CONTEXTS, default=list(raw.driver.CONTEXTS))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    (args.output / "optima").mkdir(exist_ok=True)
    _, panel, splits = raw.driver.inputs(str(transfer.PREVIOUS))
    reference = pd.read_csv(transfer.OUTPUT / "aggregate_replacement/predictions.csv")
    records = []
    for context in args.contexts:
        for objective in raw.driver.OBJECTIVES:
            aggregate, fitted, identity = load_models(context, objective)
            hpr = hpr_response(fitted)
            train = raw.driver.training_rows(panel, splits, context)
            reference_cell = reference[(reference.context == context) & (reference.objective == objective)]
            models = {arm: Surface(aggregate, hpr, arm) for arm in Arm}
            checks = {}
            for arm, model in models.items():
                ref_arm = "hpr" if arm == Arm.HPR_TIED else arm.value
                values = reference_cell[reference_cell.model == ref_arm].sort_values("row").predicted.to_numpy()
                checks[arm] = check_surface(model, fitted, panel, values)
            if args.check_only:
                print(json.dumps({"context": context, "objective": objective, "checks": checks}), flush=True)
                continue
            source = transfer.PREVIOUS / "optima" / f"{context}_{objective}_aggregate.json"
            prior = json.loads(source.read_text())
            for path, digest in prior["input_hashes"].items():
                assert transfer.file_hash(Path(path)) == digest, path
            identity[str(source)] = transfer.file_hash(source)
            policy = np.asarray(prior["weights"])
            value, _, diagnostic = models[Arm.AGGREGATE].value_gradient(policy)
            assert abs(value - prior["predicted_bpb"]) < 1e-12
            aggregate_record = {
                "context": context,
                "objective": objective,
                "arm": Arm.AGGREGATE,
                "predicted_bpb": value,
                "weights": policy.tolist(),
                "input_hashes": identity,
                "checks": checks[Arm.AGGREGATE],
                "diagnostic": diagnostic,
                "full_design_support": prior["full_design_support"],
                "training_support": prior["training_support"],
                "source": str(source),
                "reused_frozen_optimum": True,
            }
            raw.driver.write_json(args.output / "optima" / f"{context}_{objective}_aggregate.json", aggregate_record)
            records.append(aggregate_record)
            hpr_tied = optimize_cell(
                args.output,
                context,
                objective,
                models[Arm.HPR_TIED],
                checks[Arm.HPR_TIED],
                identity,
                panel,
                train,
                aggregate_record,
                None,
            )
            records.append(hpr_tied)
            for arm in (Arm.HPR, Arm.TRANSFER):
                tied_reference, other_tied = (
                    (hpr_tied, aggregate_record) if arm == Arm.HPR else (aggregate_record, hpr_tied)
                )
                record = optimize_cell(
                    args.output,
                    context,
                    objective,
                    models[arm],
                    checks[arm],
                    identity,
                    panel,
                    train,
                    tied_reference,
                    other_tied,
                )
                records.append(record)
            summaries(args.output, records, float(panel["alpha"]))


if __name__ == "__main__":
    main()
