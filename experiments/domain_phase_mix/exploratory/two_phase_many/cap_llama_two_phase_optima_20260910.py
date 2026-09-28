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
"""Re-optimize the frozen Llama surrogates under a total eight-epoch policy cap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import inspect_llama_uncheatable_optima_20260910 as previous
import numpy as np
import pandas as pd
from fit_two_phase_link_spines_20260907 import write_json_atomic
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

OUTPUT = previous.REFERENCE / "llama_uncheatable_cap8_20260910"
CAP = 8.0
BLOCKS = 2048
SEED = 20260910
OPTIONS = {"maxiter": 2000, "ftol": 1e-11}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def geometry(scale: str, inventory: np.ndarray) -> np.ndarray:
    """Enforce the cap under both the fitted nominal and realized schedule fractions."""
    fractions = np.asarray([0.8, previous.REALIZED_ALPHA[scale]])
    return np.stack([fractions[:, None] * inventory, (1 - fractions[:, None]) * inventory], axis=1)


def exposures(weights: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.sum(coefficients * weights[None, :, :], axis=1)


def feasible_start(weights: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    natural = 1 / coefficients[0].sum(axis=0)
    natural /= natural.sum()
    anchor = np.stack([natural, natural])
    point = np.maximum(weights, 0)
    point /= point.sum(axis=1, keepdims=True)
    actual, base = exposures(point, coefficients), exposures(anchor, coefficients)
    over = actual > CAP
    if over.any():
        fraction = float(np.min((CAP - base[over]) / (actual[over] - base[over]))) * (1 - 1e-12)
        point = anchor + fraction * (point - anchor)
    assert exposures(point, coefficients).max() <= CAP + 1e-10
    return point


def solve(model: Any, initial: np.ndarray, coefficients: np.ndarray, name: str) -> dict:
    width = model.buckets
    start = feasible_start(initial, coefficients)
    equality = np.kron(np.eye(2), np.ones((1, width)))
    cap_matrix = np.concatenate([np.hstack([np.diag(row[0]), np.diag(row[1])]) for row in coefficients]) / CAP

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, _ = model.value_gradient(flat.reshape(2, width))
        return value, gradient.ravel()

    result = minimize(
        objective,
        start.ravel(),
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * (2 * width),
        constraints=[
            {"type": "eq", "fun": lambda x: equality @ x - 1, "jac": lambda _x: equality},
            {"type": "ineq", "fun": lambda x: 1 - cap_matrix @ x, "jac": lambda _x: -cap_matrix},
        ],
        options=OPTIONS,
    )
    violation = float(
        max(np.max(abs(equality @ result.x - 1)), np.max(cap_matrix @ result.x - 1), -result.x.min(), result.x.max() - 1)
    )
    retained = start
    accepted = bool(np.isfinite(result.x).all() and violation < 1e-7)
    if accepted:
        endpoint = feasible_start(result.x.reshape(2, width), coefficients)
        if model.value_gradient(endpoint)[0] < model.value_gradient(start)[0]:
            retained = endpoint
    return {
        "name": name,
        "weights": retained.tolist(),
        "predicted_bpb": model.value_gradient(retained)[0],
        "success": bool(result.success and accepted),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "raw_violation": violation,
        "initial_bpb": model.value_gradient(start)[0],
        "max_epochs": exposures(retained, coefficients).max(),
    }


def phase_maximum(counts: np.ndarray, phase: int, coefficients: np.ndarray) -> np.ndarray:
    remaining = CAP - coefficients[:, 1 - phase] * counts[1 - phase] / BLOCKS
    upper = np.min(remaining / coefficients[:, phase], axis=0)
    return np.clip(np.floor(BLOCKS * upper + 1e-10).astype(int), 0, BLOCKS)


def round_policy(weights: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    counts = np.floor(weights * BLOCKS).astype(int)
    for phase in range(2):
        while counts[phase].sum() < BLOCKS:
            available = np.flatnonzero(counts[phase] < phase_maximum(counts, phase, coefficients))
            assert len(available), "no feasible bucket can receive the remaining sampler block"
            deficit = BLOCKS * weights[phase, available] - counts[phase, available]
            counts[phase, available[int(np.argmax(deficit))]] += 1
    assert exposures(counts / BLOCKS, coefficients).max() <= CAP + 1e-10
    return counts


def refine(
    module: Any, model: Any, hpr: Any, fit: Any, weights: np.ndarray, coefficients: np.ndarray
) -> tuple[np.ndarray, list]:
    counts = round_policy(weights, coefficients)
    moves = []
    for cycle in range(50):
        total = 0
        for phase in range(2):
            maximum = phase_maximum(counts, phase, coefficients)
            assert (counts[phase] <= maximum).all()

            def predict(rows: np.ndarray, phase: int = phase) -> np.ndarray:
                policies = np.repeat((counts / BLOCKS)[None, :, :], len(rows), axis=0)
                policies[:, phase] = rows
                return previous.batch_predict(model, hpr, fit, policies)

            counts[phase], steps = module.refine_counts(predict, counts[phase], maximum)
            total += steps
            moves.append({"cycle": cycle, "phase": phase, "moves": steps})
        if total == 0:
            assert np.all(counts.sum(axis=1) == BLOCKS)
            assert exposures(counts / BLOCKS, coefficients).max() <= CAP + 1e-10
            return counts, moves
    raise RuntimeError("capped exchange refinement did not converge")


def identity(scale: str) -> dict:
    source = previous.OUTPUT
    paths = [
        source / "MANIFEST.json",
        source / "fits" / scale / "mariner.json",
        source / "fits" / scale / "hpr/model.pkl",
        source / "proposals" / scale / "two_phase.json",
        Path(previous.__file__),
    ]
    return {
        "cap": CAP,
        "scope": "total epochs per bucket; both nominal and realized fractions",
        "frozen_source_sha256": {str(p): previous.preparation.sha(p) for p in paths},
    }


def run_scale(scale: str) -> None:
    destination = OUTPUT / scale
    destination.mkdir(parents=True, exist_ok=True)
    protocol = identity(scale)
    protocol_path = destination / "protocol.json"
    if protocol_path.exists():
        assert read(protocol_path) == protocol, "frozen inputs changed; use a different output root"
    else:
        write_json_atomic(protocol_path, protocol)
    if (destination / "complete.json").exists():
        for name, digest in read(destination / "complete.json")["sha256"].items():
            assert previous.preparation.sha(destination / name) == digest
        print(f"reuse {scale} cap-8 result", flush=True)
        return
    module, swarm, fit = previous.load_single(scale)
    model, hpr, _ = previous.surface(scale)
    one_path = destination / "one_phase.json"
    if one_path.exists():
        one = read(one_path)
    else:
        continuous, starts = module.continuous_optimum(fit, swarm.weights, cap=CAP)
        counts, diagnostic = module.runtime_policy(fit, continuous, cap=CAP)
        one = {
            "weights": (counts / BLOCKS).tolist(),
            "counts": counts.tolist(),
            "continuous_weights": continuous.tolist(),
            "starts": starts,
            **diagnostic,
        }
        write_json_atomic(one_path, one)
    pd.DataFrame(
        {
            "bucket": swarm.buckets,
            "count": one["counts"],
            "weight": one["weights"],
            "epochs": np.asarray(one["weights"]) * swarm.inventory,
        }
    ).to_csv(destination / "one_phase.csv", index=False)
    uncapped = read(previous.OUTPUT / "proposals" / scale / "two_phase.json")
    coefficients = geometry(scale, swarm.inventory)
    starts = [(r["name"], np.asarray(r["initial_weights"])) for r in uncapped["starts"] if r["solver"] == "slsqp"]
    starts += [
        ("uncapped_endpoint", np.asarray(uncapped["continuous_weights"])),
        ("capped_one_phase", np.repeat(np.asarray(one["continuous_weights"])[None, :], 2, axis=0)),
    ]
    records = []
    for name, start in starts:
        path = destination / "solver" / f"{name}.json"
        if not path.exists():
            write_json_atomic(path, solve(model, start, coefficients, name))
        records.append(read(path))
    # Tiny interior perturbations check whether zero weights or the TV cusp trapped a leading endpoint.
    rng = np.random.default_rng(SEED)
    for index, candidate in enumerate(sorted(records, key=lambda r: r["predicted_bpb"])[:4]):
        name = f"polish_{index}"
        path = destination / "solver" / f"{name}.json"
        if not path.exists():
            perturbed = 0.9999 * np.asarray(candidate["weights"]) + 0.0001 * rng.dirichlet(
                np.ones(model.buckets), size=2
            )
            write_json_atomic(path, solve(model, perturbed, coefficients, name))
        records.append(read(path))
    best = min(records, key=lambda row: row["predicted_bpb"])
    counts, moves = refine(module, model, hpr, fit, np.asarray(best["weights"]), coefficients)
    runtime = counts / BLOCKS
    prediction, _, diagnostic = model.value_gradient(runtime)
    reconstructed = float(previous.batch_predict(model, hpr, fit, runtime[None, :, :])[0])
    assert abs(prediction - reconstructed) < 1e-10
    epoch = exposures(runtime, coefficients)
    aggregate = 0.8 * runtime[0] + 0.2 * runtime[1]
    actual_terms = previous.hpr_terms(model.hpr, runtime)
    tied_terms = previous.hpr_terms(model.hpr, np.stack([aggregate, aggregate]))
    terms = {name: float((actual_terms[name] - tied_terms[name]).sum()) for name in actual_terms}
    assert abs(sum(terms.values()) - diagnostic["hpr_contrast"]) < 1e-10
    binding = [
        swarm.buckets[i]
        for i in range(model.buckets)
        if any(counts[p, i] == phase_maximum(counts, p, coefficients)[i] for p in range(2))
    ]
    two = {
        "cap": CAP,
        "counts": counts.tolist(),
        "weights": runtime.tolist(),
        "continuous_weights": best["weights"],
        "continuous_prediction": best["predicted_bpb"],
        "runtime_prediction": prediction,
        "max_nominal_epochs": float(epoch[0].max()),
        "max_realized_epochs": float(epoch[1].max()),
        "phase_tv": float(abs(runtime[0] - runtime[1]).sum() / 2),
        "gain_vs_cap8_one_phase": one["runtime_prediction"] - prediction,
        "gain_vs_uncapped_one_phase": (
            read(previous.OUTPUT / "proposals" / scale / "one_phase.json")["runtime_prediction"] - prediction
        ),
        "predicted_cap_cost": prediction - uncapped["runtime_prediction"],
        "starts": records,
        "selected_start": best["name"],
        "selected_success": best["success"],
        "exchange_moves": moves,
        "grid_binding_buckets": binding,
        "phase_terms": terms,
        "prediction_reconstruction_error": abs(prediction - reconstructed),
        **diagnostic,
    }
    write_json_atomic(destination / "two_phase.json", two)
    pd.DataFrame(
        {
            "bucket": swarm.buckets,
            "phase0_count": counts[0],
            "phase1_count": counts[1],
            "phase0_weight": runtime[0],
            "phase1_weight": runtime[1],
            "aggregate_weight": aggregate,
            "nominal_epochs": epoch[0],
            "realized_epochs": epoch[1],
        }
    ).to_csv(destination / "two_phase.csv", index=False)
    write_json_atomic(
        destination / "complete.json",
        {
            "sha256": {
                str(p.relative_to(destination)): previous.preparation.sha(p)
                for p in sorted(destination.rglob("*"))
                if p.is_file() and p.name != "complete.json"
            }
        },
    )
    print(
        f"{scale}: 1p {one['runtime_prediction']:.6f}; 2p {prediction:.6f}; "
        f"phase correction {diagnostic['hpr_contrast']:.6f}",
        flush=True,
    )


def summarize() -> None:
    rows, buckets = [], []
    for scale in previous.preparation.SCALES:
        one = read(OUTPUT / scale / "one_phase.json")
        two = read(OUTPUT / scale / "two_phase.json")
        old_one = read(previous.OUTPUT / "proposals" / scale / "one_phase.json")
        old_two = read(previous.OUTPUT / "proposals" / scale / "two_phase.json")
        rows.append(
            {
                "scale": scale,
                "uncapped_1p": old_one["runtime_prediction"],
                "cap8_1p": one["runtime_prediction"],
                "uncapped_2p": old_two["runtime_prediction"],
                "cap8_2p": two["runtime_prediction"],
                "cap8_2p_aggregate": two["aggregate_bpb"],
                "cap8_phase_correction": two["hpr_contrast"],
                "gain_vs_uncapped_1p": two["gain_vs_uncapped_one_phase"],
                "gain_vs_cap8_1p": two["gain_vs_cap8_one_phase"],
                "max_nominal_epochs": two["max_nominal_epochs"],
                "max_realized_epochs": two["max_realized_epochs"],
            }
        )
        table = pd.read_csv(OUTPUT / scale / "two_phase.csv").set_index("bucket")
        old = pd.read_csv(previous.OUTPUT / "proposals" / scale / "two_phase.csv").set_index("bucket")
        table["uncapped_phase0_weight"] = old.phase0_weight
        table["uncapped_phase1_weight"] = old.phase1_weight
        table["uncapped_aggregate"] = old.aggregate_weight
        table["scale"] = scale
        buckets.append(table.reset_index())
    pd.DataFrame(rows).to_csv(OUTPUT / "comparison.csv", index=False)
    pd.concat(buckets, ignore_index=True).to_csv(OUTPUT / "bucket_comparison.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=previous.preparation.SCALES)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.summarize:
            summarize()
            return
        assert args.scale is not None, "choose a scale"
        run_scale(args.scale)


if __name__ == "__main__":
    main()
