# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Audit frozen macro heads without phase clipping or policy constraints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import macro_semantic_policy_gradient_20260907 as gradient
import numpy as np
import optimize_two_phase_creative_20260907 as original
import pandas as pd
import resolve_two_phase_optimization_20260907 as resolution
from scipy.optimize import minimize

ROOT = original.HERE / "reference_outputs/two_phase_refinement_20260907"
OUTPUT = ROOT / "macro_raw"
MODELS = ("CRE2-019", "CRE2-020")


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def diagnostic(surface: gradient.Surface, weights: np.ndarray, tied: bool) -> dict[str, Any]:
    values, log_gradient = surface.value_logit_gradient(weights[None])
    a = surface.record["alpha"] * weights[0] + (1 - surface.record["alpha"]) * weights[1]
    eta_a, _ = surface.semantic_state(a[None])
    eta0, _ = surface.semantic_state(weights[None, 0])
    eta1, _ = surface.semantic_state(weights[None, 1])
    delta = eta1[0] - eta0[0]
    coefficient = np.asarray(surface.record["coefficients"]) / np.asarray(surface.record["scale"])
    width = len(delta)
    correction = float(delta @ coefficient[:width] + delta**2 @ coefficient[width:])
    floor = float(surface.floors @ surface.weights)
    probabilities = weights[:1] if tied else weights
    jacobian = log_gradient[0].sum(axis=0, keepdims=True) if tied else log_gradient[0]
    interior = bool(np.all(probabilities > 0))
    gap = float(-np.min(jacobian / probabilities, axis=1).sum()) if interior else None
    with np.errstate(under="ignore", over="ignore"):
        ratio = float(np.exp(correction))
    return {
        "predicted_bpb": float(values[0]),
        "macro_floor": floor,
        "floor_margin": float(values[0] - floor),
        "phase_linear": correction,
        "phase_exponential_ratio": ratio,
        "positive_link_underflow": bool(surface.record["model"] == "CRE2-020" and ratio == 0),
        "aggregate_exponent_clips": int(np.sum(np.abs(eta_a) > 30)),
        "zero_phase_weights": int(np.sum(probabilities == 0)),
        "maximum_logit_gradient": float(np.max(np.abs(jacobian))),
        "strict_interior": interior,
        "interior_simplex_fw_gap": gap,
        "ordinary_boundary_kkt_certified": False,
    }


def solve(surface: gradient.Surface, start: np.ndarray, tied: bool, name: str) -> dict[str, Any]:
    initialized = 0.999 * start + 0.001 / 39
    logits = np.log(initialized[:1] if tied else initialized).ravel()
    initial_policy = original.policies(logits, tied)[0]
    best_policy, best_value = initial_policy.copy(), float(surface.predict(initial_policy[None])[0])
    history, exceptions = [], []
    beginning = time.monotonic()

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal best_policy, best_value
        policy = original.policies(x, tied)
        try:
            values, jacobian = surface.value_logit_gradient(policy)
            if not np.isfinite(values).all() or not np.isfinite(jacobian).all():
                raise FloatingPointError("Nonfinite macro objective or gradient")
        except FloatingPointError as error:
            exceptions.append({"error": str(error), "weights": policy[0].tolist()})
            raise
        value = float(values[0])
        derivative = jacobian[0].sum(axis=0) if tied else jacobian[0].ravel()
        history.append({"value": value, "maximum_logit_gradient": float(np.max(np.abs(derivative)))})
        if value < best_value:
            best_value, best_policy = value, policy[0].copy()
        return value, derivative

    try:
        fitted = minimize(
            objective,
            logits,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-7, "maxls": 30},
        )
        result = {
            "success": bool(fitted.success),
            "status": int(fitted.status),
            "message": str(fitted.message),
            "iterations": int(fitted.nit),
            "evaluations": int(fitted.nfev),
            "endpoint_value": float(fitted.fun),
        }
    except FloatingPointError as error:
        result = {
            "success": False,
            "status": "nonfinite_trial",
            "message": str(error),
            "iterations": None,
            "evaluations": len(history) + len(exceptions),
            "endpoint_value": None,
        }
    return result | {
        "name": name,
        "original_weights": start.tolist(),
        "initialized_weights": initial_policy.tolist(),
        "weights": best_policy.tolist(),
        "minimum_finite_bpb": best_value,
        "trace": history,
        "exceptions": exceptions,
        "elapsed_seconds": time.monotonic() - beginning,
        "diagnostic": diagnostic(surface, best_policy, tied),
    }


def polish(surface: gradient.Surface, weights: np.ndarray, tied: bool) -> dict[str, Any]:
    events: list[dict[str, Any]] = []

    def safe_predict(policies: np.ndarray) -> np.ndarray:
        try:
            values = surface.predict(policies)
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite macro candidate")
            return values
        except FloatingPointError:
            values = []
            for policy in policies:
                try:
                    value = float(surface.predict(policy[None])[0])
                    if not np.isfinite(value):
                        raise FloatingPointError("Nonfinite macro candidate")
                    values.append(value)
                except FloatingPointError as error:
                    events.append(
                        {
                            "error": str(error),
                            "weights": policy.tolist(),
                            "disposition": "ineligible_nonfinite_candidate",
                        }
                    )
                    values.append(np.inf)
            return np.asarray(values)

    adapter = original.Surface(safe_predict, lambda w: {}, float(surface.floors @ surface.weights), (), 0.0, None)
    trace, exchanges = [], []
    start_value = float(surface.predict(weights[None])[0])
    for sweep in range(12):
        before = float(surface.predict(weights[None])[0])
        for phase in range(1 if tied else 2):
            pivot = int(np.argmax(weights[phase]))
            for recipient in range(39):
                if recipient != pivot:
                    weights, step = resolution.line_minimum(adapter, weights, phase, recipient, pivot, tied)
                    exchanges.append(step | {"sweep": sweep})
        trace.append({"sweep": sweep, "value": float(surface.predict(weights[None])[0]), "before": before})
    assert weights.min() >= 0 and np.max(abs(weights.sum(axis=1) - 1)) < 1e-12
    return {
        "weights": weights.tolist(),
        "initial_bpb": start_value,
        "minimum_finite_bpb": float(surface.predict(weights[None])[0]),
        "trace": trace,
        "exchanges": exchanges,
        "exceptions": events,
        "boundary_probes": resolution.boundary_probes(adapter, weights, tied),
        "diagnostic": diagnostic(surface, weights, tied),
    }


def run_cell(model: str, objective: str, tied: bool, tied_result: dict[str, Any] | None) -> dict[str, Any]:
    kind = "tied" if tied else "two_phase"
    destination = OUTPUT / "cells" / model / objective / kind
    destination.mkdir(parents=True, exist_ok=True)
    surface = gradient.load_surface(ROOT / "macro_semantic", objective, "final", model)
    _, panel, _ = original.previous.inputs(str(original.paths.PREVIOUS))
    files = [
        Path(__file__),
        Path(gradient.__file__),
        Path(gradient.macro.__file__),
        Path(resolution.__file__),
        Path(original.__file__),
        OUTPUT / "PROTOCOL.md",
        ROOT / "macro_semantic/fits" / model / objective / "final/fit.json",
        original.paths.PREVIOUS / "optima" / f"final_{objective}_aggregate.json",
        *[original.paths.PREVIOUS / "inputs" / name for name in ("panel.npz", "splits.npz", "single_phase.py")],
    ]
    identity = {str(p): original.paths.digest(p) for p in files}
    complete = destination / "complete.json"
    if complete.exists():
        cached = json.loads(complete.read_text())
        assert cached["input_hashes"] == identity
        return cached
    public = gradient.macro.load_predictor(ROOT / "macro_semantic", objective, "final", model)
    parity = float(np.max(abs(surface.predict(panel["weights"]) - public.predict(panel["weights"]))))
    assert parity < 1e-10
    alpha = float(panel["alpha"])
    proportional = panel["weights"][np.flatnonzero(panel["calibration_mask"])[0]]
    aggregate = np.asarray(json.loads(files[7].read_text())["weights"])
    best = int(np.argmin(panel[f"{objective}_aggregate"]))
    rng = np.random.default_rng(20260907)
    starts = [
        ("proportional", proportional),
        ("saved_aggregate", aggregate),
        ("best_training", panel["weights"][best]),
        *[(f"dirichlet_{index}", rng.dirichlet(np.ones(39), size=2)) for index in range(2)],
    ]
    if tied:
        starts = [(name, np.repeat((alpha * w[0] + (1 - alpha) * w[1])[None], 2, axis=0)) for name, w in starts]
    else:
        assert tied_result is not None
        starts.append(("own_tied_minimum", np.asarray(tied_result["weights"])))
    results = []
    for name, start in starts:
        path = destination / f"{name}.json"
        if path.exists():
            fitted = json.loads(path.read_text())
            assert fitted["input_hashes"] == identity
        else:
            fitted = solve(surface, start, tied, name) | {"input_hashes": identity}
            original.previous.write_json(path, clean(fitted))
        results.append(fitted)
        print(
            f"{model}/{objective}/{kind}/{name}: {fitted['minimum_finite_bpb']:.8f} success={fitted['success']}",
            flush=True,
        )
    selected = min(results, key=lambda r: r["minimum_finite_bpb"])
    polished = polish(surface, np.asarray(selected["weights"]), tied)
    weights = np.asarray(polished["weights"])
    support = original.prior_raw.support_audit(weights, panel["weights"], alpha)
    epochs = panel["c0"] * weights[0] + panel["c1"] * weights[1]
    value = polished["minimum_finite_bpb"]
    public_error = abs(float(public.predict(weights[None])[0]) - value)
    assert public_error < 1e-10
    record = {
        "model": model,
        "objective": objective,
        "kind": kind,
        "input_hashes": identity,
        "observed_prediction_parity": parity,
        "raw_prediction_parity": public_error,
        "predicted_bpb": value,
        "predicted_gain_over_own_tied": 0.0 if tied_result is None else tied_result["predicted_bpb"] - value,
        "weights": weights.tolist(),
        "max_epochs": float(epochs.max()),
        "epochs": epochs.tolist(),
        "phase_tv": float(abs(weights[0] - weights[1]).sum() / 2),
        "max_phase_weight": float(weights.max()),
        "support": support,
        "diagnostic": polished["diagnostic"],
        "selected_start": selected["name"],
        "successful_starts": sum(bool(r["success"]) for r in results),
        "n_starts": len(results),
        "start_bpb_spread": float(np.ptp([r["minimum_finite_bpb"] for r in results])),
        "start_policy_tv_max": max(
            original.prior_raw.policy_tv(np.asarray(a["weights"]), np.asarray(b["weights"]), alpha)
            for i, a in enumerate(results)
            for b in results[i + 1 :]
        ),
        "polishing_improvement": selected["minimum_finite_bpb"] - value,
        "starts": results,
        "polishing": polished,
    }
    original.previous.write_json(complete, clean(record))
    return record


def main() -> None:
    records = []
    for objective in original.previous.OBJECTIVES:
        for model in MODELS:
            tied = run_cell(model, objective, True, None)
            records.extend([tied, run_cell(model, objective, False, tied)])
            rows = []
            for r in records:
                row = {
                    key: value
                    for key, value in r.items()
                    if key not in ("input_hashes", "weights", "epochs", "starts", "polishing", "diagnostic", "support")
                }
                row |= r["diagnostic"] | r["support"]
                rows.append(row)
            pd.DataFrame(rows).to_csv(OUTPUT / "summary.csv", index=False)


if __name__ == "__main__":
    main()
