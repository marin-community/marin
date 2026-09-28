# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Resolve frozen two-phase surfaces by feasible pairwise mass exchange."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import optimize_two_phase_creative_20260907 as original
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from scipy.special import expit

OUTPUT = original.HERE / "reference_outputs/two_phase_refinement_20260907/optimization_resolution"
MODELS = ("CRE2-013", "CRE2-011")
FRACTIONS = np.unique(
    np.r_[0, 0.5, 1, 10.0 ** -np.array([1, 2, 4, 6, 8, 10, 12, 14]), 1 - 10.0 ** -np.array([1, 2, 4, 6, 8, 10, 12, 14])]
)
PROBE_MASSES = (1e-3, 1e-5, 1e-7, 1e-9, 1e-11)


def product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Apply zero coefficients without manufacturing zero-times-infinity NaNs."""
    with np.errstate(invalid="ignore"):
        return np.where(left == 0, 0, left * right)


def boundary_gradient(model: str, objective: str, context: str, weights: np.ndarray, tied: bool) -> np.ndarray:
    """Return exact extended gradients; unresolved infinity cancellation stays NaN."""
    if model == "CRE2-013":
        gradient = original.joint_gradient.load_gradient(model, objective, context).value_and_gradient(weights[None])[1][
            0
        ]
        return gradient.sum(axis=0, keepdims=True) if tied else gradient
    pack = original.paths.spine_pack(context, objective)
    record = original.geometry.load_predictor(original.ROOT / "phase_geometry", objective, context, model).record
    alpha = float(record["alpha"])
    aggregate = alpha * weights[0] + (1 - alpha) * weights[1]

    def eta_jacobian(mixture: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        exposure = mixture[None, None] * (pack["c0"] + pack["c1"])
        rate, power = pack["rate"], pack["power"]
        z = rate * exposure
        powered = z**power
        with np.errstate(divide="ignore", invalid="ignore"):
            db = rate * power * z ** (power - 1) * np.exp(-powered)
        argument = np.log1p(exposure) - pack["threshold"]
        dh = 2 * np.logaddexp(0, argument) * expit(argument) / (1 + exposure)
        jacobian = (-product(pack["alpha"], db) + pack["beta"] * dh) * (pack["c0"] + pack["c1"])
        eta = pack["intercept"] + np.sum(
            -pack["alpha"] * original.paths.benefit(exposure, pack) + pack["beta"] * original.paths.harm(exposure, pack),
            axis=-1,
        )
        return eta[0], jacobian[0]

    eta, ja = eta_jacobian(aggregate)
    qa = np.exp(np.clip(eta, -30, 30)) * pack["weights"] * (np.abs(eta) < 30)
    if tied:
        return np.sum(product(qa[:, None], ja), axis=0, keepdims=True)
    eta0, j0 = eta_jacobian(weights[0])
    eta1, j1 = eta_jacobian(weights[1])
    coefficient_a, coefficient_delta = qa.copy(), np.zeros_like(qa)
    if not record["config"]["zero"]:
        scale = np.asarray(record["scale"])
        centers, tied_centers = np.asarray(record["centers"]), np.asarray(record["tied_centers"])
        linear = np.asarray(record["coefficients"]) @ pack["weights"]
        matrix = np.r_[eta, eta1 - eta0][None] / scale
        reference = np.r_[eta, np.zeros_like(eta)][None] / scale
        denominator = 2 * matrix.shape[1] * record["config"]["bandwidth"] ** 2

        def derivative(point: np.ndarray, locations: np.ndarray) -> np.ndarray:
            kernel = np.exp(-cdist(point, locations, metric="sqeuclidean") / denominator) * linear
            return ((-2 / denominator) * (point * kernel.sum(axis=1, keepdims=True) - kernel @ locations))[0]

        dm = derivative(matrix, centers) - derivative(matrix, tied_centers)
        dt = -derivative(reference, centers) + derivative(reference, tied_centers)
        width = len(eta)
        coefficient_a += (dm[:width] + dt[:width]) / scale[:width]
        coefficient_delta = dm[width:] / scale[width:]
    with np.errstate(invalid="ignore"):
        ga = np.sum(product(coefficient_a[:, None], ja), axis=0)
        gradient = np.stack(
            [
                alpha * ga - np.sum(product(coefficient_delta[:, None], j0), axis=0),
                (1 - alpha) * ga + np.sum(product(coefficient_delta[:, None], j1), axis=0),
            ]
        )
    return gradient


def scalar(value: float) -> float | str:
    return float(value) if np.isfinite(value) else str(float(value))


def kkt_diagnostic(model: str, objective: str, context: str, weights: np.ndarray, tied: bool) -> dict[str, Any]:
    gradient = boundary_gradient(model, objective, context, weights, tied)
    mixture = weights[:1] if tied else weights
    fw, pairwise = [], []
    for w, g in zip(mixture, gradient, strict=True):
        active = w > 0
        with np.errstate(invalid="ignore"):
            fw.append(float(w[active] @ g[active] - np.min(g)))
            pairwise.append(float(np.max(g[active]) - np.min(g)))
    gap = float(np.sum(fw))
    return {
        "raw_simplex_fw_gap": scalar(gap),
        "raw_pairwise_gradient_gap": scalar(float(np.max(pairwise))),
        "finite_gradient": bool(np.isfinite(gradient).all()),
        "finite_fw_gap": bool(np.isfinite(gap)),
        "zero_weights": int(np.sum(mixture == 0)),
        "minimum_positive_weight": float(mixture[mixture > 0].min()),
        "gradient": [[scalar(value) for value in row] for row in gradient],
    }


def exchanged(
    weights: np.ndarray, phase: int, recipient: int, donor: int, fractions: np.ndarray, tied: bool
) -> np.ndarray:
    candidates = np.repeat(weights[None], len(fractions), axis=0)
    total = weights[phase, recipient] + weights[phase, donor]
    candidates[:, phase, recipient] = total * fractions
    candidates[:, phase, donor] = total * (1 - fractions)
    if tied:
        candidates[:, 1] = candidates[:, 0]
    return candidates


def line_minimum(
    surface: original.Surface, weights: np.ndarray, phase: int, recipient: int, donor: int, tied: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    total = float(weights[phase, recipient] + weights[phase, donor])
    if total == 0:
        value = float(surface.predict(weights[None])[0])
        return weights, {
            "phase": phase,
            "recipient": recipient,
            "donor": donor,
            "before": value,
            "after": value,
            "changed": False,
            "evaluations": 1,
        }
    fraction = float(weights[phase, recipient] / total)
    nodes = np.unique(np.r_[FRACTIONS, fraction])
    candidates = exchanged(weights, phase, recipient, donor, nodes, tied)
    values = surface.predict(candidates)
    current = float(surface.predict(weights[None])[0])
    index = int(np.argmin(values))
    chosen, chosen_value = candidates[index], float(values[index])
    evaluations = len(values) + 1
    if 0 < index < len(nodes) - 1:
        result = minimize_scalar(
            lambda x: float(surface.predict(exchanged(weights, phase, recipient, donor, np.asarray([x]), tied))[0]),
            method="bounded",
            bounds=(float(nodes[index - 1]), float(nodes[index + 1])),
            options={"xatol": 1e-14, "maxiter": 80},
        )
        evaluations += int(result.nfev)
        if result.fun < chosen_value:
            chosen, chosen_value = exchanged(weights, phase, recipient, donor, np.asarray([result.x]), tied)[0], float(
                result.fun
            )
    improved = chosen_value < current - 1e-15
    return (chosen if improved else weights), {
        "phase": phase,
        "recipient": recipient,
        "donor": donor,
        "before": current,
        "after": chosen_value if improved else current,
        "changed": improved,
        "evaluations": evaluations,
    }


def boundary_probes(surface: original.Surface, weights: np.ndarray, tied: bool) -> dict[str, Any]:
    current = float(surface.predict(weights[None])[0])
    candidates, metadata = [], []
    for phase in range(1 if tied else 2):
        donor = int(np.argmax(weights[phase]))
        for recipient in range(39):
            if recipient == donor:
                continue
            total = float(weights[phase, donor] + weights[phase, recipient])
            for requested in PROBE_MASSES:
                mass = min(requested, float(weights[phase, donor]))
                fraction = (weights[phase, recipient] + mass) / total
                candidates.append(exchanged(weights, phase, recipient, donor, np.asarray([fraction]), tied)[0])
                metadata.append(
                    {
                        "phase": phase,
                        "donor": donor,
                        "recipient": recipient,
                        "mass": mass,
                        "recipient_was_zero": bool(weights[phase, recipient] == 0),
                    }
                )
    values = surface.predict(np.asarray(candidates))
    rows = [
        record | {"improvement": current - float(value), "one_sided_slope": (float(value) - current) / record["mass"]}
        for record, value in zip(metadata, values, strict=True)
    ]
    return {
        "max_improvement": max(r["improvement"] for r in rows),
        "most_negative_slope": min(r["one_sided_slope"] for r in rows),
        "boundary_max_improvement": max((r["improvement"] for r in rows if r["recipient_was_zero"]), default=0.0),
        "probes": rows,
    }


def resolve_start(
    surface: original.Surface, model: str, objective: str, context: str, initial: np.ndarray, tied: bool
) -> dict[str, Any]:
    weights = initial.copy()
    current = float(surface.predict(weights[None])[0])
    beginning = time.monotonic()
    initial_kkt = kkt_diagnostic(model, objective, context, weights, tied)
    trace, exchanges, stalled = [], [], 0
    status = "sweep_budget_exhausted"
    for sweep in range(12):
        before = current
        for phase in range(1 if tied else 2):
            pivot = int(np.argmax(weights[phase]))
            for bucket in range(39):
                if bucket == pivot:
                    continue
                weights, record = line_minimum(surface, weights, phase, bucket, pivot, tied)
                exchanges.append(record | {"sweep": sweep})
        current = float(surface.predict(weights[None])[0])
        kkt = kkt_diagnostic(model, objective, context, weights, tied)
        improvement = before - current
        trace.append({"sweep": sweep, "bpb": current, "improvement": improvement, "kkt": kkt})
        if (
            kkt["finite_gradient"]
            and kkt["finite_fw_gap"]
            and float(kkt["raw_simplex_fw_gap"]) <= 1e-6
            and improvement <= 1e-10
        ):
            status = "finite_kkt_converged"
            break
        stalled = stalled + 1 if improvement <= 1e-12 else 0
        if stalled >= 2:
            status = "objective_stalled_without_kkt_certificate"
            break
    probes = boundary_probes(surface, weights, tied)
    assert np.max(np.abs(weights.sum(axis=1) - 1)) < 1e-12 and weights.min() >= 0
    assert current <= float(surface.predict(initial[None])[0]) + 1e-13
    return {
        "initial_bpb": float(surface.predict(initial[None])[0]),
        "bpb": current,
        "improvement": float(surface.predict(initial[None])[0]) - current,
        "weights": weights.tolist(),
        "initial_kkt": initial_kkt,
        "final_kkt": trace[-1]["kkt"],
        "status": status,
        "sweeps": len(trace),
        "trace": trace,
        "exchanges": exchanges,
        "boundary_probes": probes,
        "elapsed_seconds": time.monotonic() - beginning,
        "diagnostics": surface.diagnostics(weights[None]),
    }


def run_cell(argument: tuple[str, str, str, str]) -> dict[str, Any]:
    model, objective, context, kind = argument
    tied = kind == "tied"
    surface = original.load_surface(model, objective, context)
    old_path = original.OUTPUT / "cells" / model / objective / context / kind / "complete.json"
    old = json.loads(old_path.read_text())
    destination = OUTPUT / "cells" / model / objective / context / kind
    destination.mkdir(parents=True, exist_ok=True)
    identity = {
        str(p): original.paths.digest(p) for p in (*surface.files, Path(__file__), OUTPUT / "PROTOCOL.md", old_path)
    }
    complete = destination / "complete.json"
    if complete.exists():
        result = json.loads(complete.read_text())
        assert result["input_hashes"] == identity
        return result
    best_training = next(row for row in old["starts"] if row["name"] == "best_training")
    starts = {
        "saved_minimum": np.asarray(old["weights"]),
        "training_policy": np.asarray(best_training["original_weights"]),
    }
    outcomes = []
    for name, weights in starts.items():
        path = destination / f"{name}.json"
        if path.exists():
            fitted = json.loads(path.read_text())
            assert fitted["input_hashes"] == identity
        else:
            fitted = resolve_start(surface, model, objective, context, weights, tied) | {
                "name": name,
                "input_hashes": identity,
            }
            original.previous.write_json(path, fitted)
        outcomes.append(fitted)
        print(
            f"{model}/{objective}/{context}/{kind}/{name}: {fitted['bpb']:.10f} "
            f"improvement={fitted['improvement']:.3g} {fitted['status']}",
            flush=True,
        )
    best = min(outcomes, key=lambda r: r["bpb"])
    _, panel, _ = original.previous.inputs(str(original.paths.PREVIOUS))
    policy = np.asarray(best["weights"])
    result = {
        "model": model,
        "objective": objective,
        "context": context,
        "kind": kind,
        "input_hashes": identity,
        "old_bpb": old["predicted_bpb"],
        "resolved_bpb": best["bpb"],
        "improvement_over_original": old["predicted_bpb"] - best["bpb"],
        "weights": best["weights"],
        "selected_start": best["name"],
        "status": best["status"],
        "final_kkt": best["final_kkt"],
        "boundary_probe_max_improvement": best["boundary_probes"]["max_improvement"],
        "policy_tv_from_original": original.prior_raw.policy_tv(
            policy, np.asarray(old["weights"]), float(panel["alpha"])
        ),
        "between_start_policy_tv": original.prior_raw.policy_tv(
            np.asarray(outcomes[0]["weights"]), np.asarray(outcomes[1]["weights"]), float(panel["alpha"])
        ),
        "between_start_bpb_spread": abs(outcomes[0]["bpb"] - outcomes[1]["bpb"]),
        "outcomes": outcomes,
    }
    original.previous.write_json(complete, result)
    return result


def summarize() -> None:
    records = [json.loads(p.read_text()) for p in sorted((OUTPUT / "cells").glob("*/*/*/*/complete.json"))]
    rows, stability = [], []
    _, panel, _ = original.previous.inputs(str(original.paths.PREVIOUS))
    for r in records:
        rows.append(
            {k: v for k, v in r.items() if k not in ("input_hashes", "weights", "final_kkt", "outcomes")}
            | {k: v for k, v in r["final_kkt"].items() if k != "gradient"}
        )
        final = next(
            (
                a
                for a in records
                if a["model"] == r["model"]
                and a["objective"] == r["objective"]
                and a["kind"] == r["kind"]
                and a["context"] == "final"
            ),
            None,
        )
        if final is not None and r["context"] != "final":
            old_a = json.loads(
                (
                    original.OUTPUT / "cells" / r["model"] / r["objective"] / "final" / r["kind"] / "complete.json"
                ).read_text()
            )
            old_b = json.loads(
                (
                    original.OUTPUT / "cells" / r["model"] / r["objective"] / r["context"] / r["kind"] / "complete.json"
                ).read_text()
            )
            stability.append(
                {
                    "model": r["model"],
                    "objective": r["objective"],
                    "context": r["context"],
                    "kind": r["kind"],
                    "original_tv": original.prior_raw.policy_tv(
                        np.asarray(old_a["weights"]), np.asarray(old_b["weights"]), float(panel["alpha"])
                    ),
                    "resolved_tv": original.prior_raw.policy_tv(
                        np.asarray(final["weights"]), np.asarray(r["weights"]), float(panel["alpha"])
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT / "summary.csv", index=False)
    pd.DataFrame(stability).to_csv(OUTPUT / "fold_stability.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--contexts", nargs="+", choices=original.previous.CONTEXTS, default=list(original.previous.CONTEXTS)
    )
    parser.add_argument(
        "--objectives", nargs="+", choices=original.previous.OBJECTIVES, default=list(original.previous.OBJECTIVES)
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    tasks = [
        (m, o, c, k) for c in args.contexts for o in args.objectives for m in args.models for k in ("tied", "two_phase")
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_cell, task) for task in tasks]
        for future in as_completed(futures):
            future.result()
            summarize()


if __name__ == "__main__":
    main()
