# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "matplotlib"]
# ///
"""Descriptive MARINER fits to the completed TPP10 pilot; no training submission."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from experiments.domain_phase_mix import analyze_starcoder_tpp10 as pilot
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

MODEL_ID = "weibull_softplus_unscaled@kappa_floor_link_flat15_nocap"
TRAINER_SEEDS = (20260910, 20260911)
SUBSET_SEEDS = (20260912, 20260913, 20260914)
SEQUENCE_LENGTH = 2048


@dataclasses.dataclass(frozen=True)
class Curve:
    name: str
    arm: str
    response: np.ndarray
    role: str


def input_curves(plan: dict, values: dict[str, float]) -> list[Curve]:
    """Average trainer seeds within subsets; preserve individual-seed fits as diagnostics."""
    index = {(r["arm"], r["subset_seed"], r["trainer_seed"], r["percent"]): values[r["run_name"]] for r in plan["runs"]}
    grid = sorted({r["percent"] for r in plan["runs"]})

    def response(arm: str, subset: int | None, seed: int) -> np.ndarray:
        return np.array(
            [
                index[("unmatched", None, seed, 0)] if arm == "matched" and p == 0 else index[(arm, subset, seed, p)]
                for p in grid
            ]
        )

    unmatched = np.array([response("unmatched", None, s) for s in TRAINER_SEEDS])
    matched = np.array([[response("matched", d, s) for s in TRAINER_SEEDS] for d in SUBSET_SEEDS])
    curves = [
        Curve("target", "target", response("target", None, TRAINER_SEEDS[0]), "target"),
        Curve("unmatched_mean", "unmatched", unmatched.mean(axis=0), "trainer_mean"),
        *[
            Curve(f"matched_subset_{d}", "matched", y.mean(axis=0), "within_subset_trainer_mean")
            for d, y in zip(SUBSET_SEEDS, matched, strict=True)
        ],
        Curve("matched_pooled", "matched", matched.mean(axis=(0, 1)), "secondary_pooled_subset_mean"),
        *[
            Curve(f"unmatched_seed_{s}", "unmatched", y, "single_trainer_seed")
            for s, y in zip(TRAINER_SEEDS, unmatched, strict=True)
        ],
        *[
            Curve(f"matched_subset_{d}_seed_{s}", "matched", y, "single_subset_and_trainer_seed")
            for d, matrix in zip(SUBSET_SEEDS, matched, strict=True)
            for s, y in zip(TRAINER_SEEDS, matrix, strict=True)
        ],
    ]
    assert all(len(c.response) == 7 for c in curves)
    return curves


def features(share: np.ndarray, arm: str, design: dict, component: str, label: str) -> models.Features:
    """Two domain inputs: the fixed web blend and StarCoder, with physical pool sizes."""
    horizon = design["models"]["target" if arm == "target" else "unmatched"]["tokens"]
    web_tokens = sum(design["web_sequences"].values()) * SEQUENCE_LENGTH
    code_tokens = design["matched_sequences" if arm == "matched" else "parent_sequences"] * SEQUENCE_LENGTH
    inventory = horizon / np.array([web_tokens, code_tokens], dtype=float)
    result = models.features_from_panel(
        np.column_stack([1 - share, share]),
        inventory,
        ("nemotron_fixed_blend", "starcoder"),
        early_fraction=None,
        label=label,
    )
    return dataclasses.replace(result, component=component)


def global_minimum(
    model: models.FittedFloorModel, fitted: models.Fitted, design: dict, curve: Curve, component: str
) -> tuple[float, float]:
    """Check both boundaries and polish every basin on a 0.0001-spaced scan."""

    def predict(x: np.ndarray) -> np.ndarray:
        query = features(x, curve.arm, design, component, f"tpp10|{curve.name}|minimum")
        return model.predict(fitted, query, np.arange(len(x)))

    grid = np.linspace(0, 1, 10001)
    y = predict(grid)
    assert np.isfinite(y).all()
    candidates = [(float(y[0]), 0.0), (float(y[-1]), 1.0)]
    basins = np.flatnonzero((y[1:-1] <= y[:-2]) & (y[1:-1] <= y[2:])) + 1
    for i in basins:
        result = minimize_scalar(
            lambda p: float(predict(np.array([p]))[0]),
            bounds=(float(grid[i - 1]), float(grid[i + 1])),
            method="bounded",
            options={"xatol": 1e-12},
        )
        assert result.success
        candidates.append((float(result.fun), float(result.x)))
    loss, share = min(candidates)
    assert loss <= float(y.min()) + 1e-10
    return share, loss


def json_value(value):
    if dataclasses.is_dataclass(value):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def fit_curve(curve: Curve, share: np.ndarray, design: dict, component: str) -> dict:
    inputs = features(share, curve.arm, design, component, f"tpp10|{curve.name}")
    entry = registry.ENTRY_BY_ID[MODEL_ID]
    model = entry.build(registry.apply_transform(inputs, entry))
    train = np.arange(len(share))
    inner = tuple((train[train != i], np.array([i])) for i in train)
    fitted = model.fit(inputs, curve.response, train, inner, seed=20260911)
    assert fitted.diagnostics["converged"]
    predicted = model.predict(fitted, inputs, train)
    minimum, loss = global_minimum(model, fitted, design, curve, component)
    dense = np.linspace(0, 1, 2001)
    query = features(dense, curve.arm, design, component, f"tpp10|{curve.name}|plot")
    dense_prediction = model.predict(fitted, query, np.arange(len(dense)))
    assert np.isfinite(predicted).all() and np.isfinite(dense_prediction).all()
    return {
        "curve": curve.name,
        "arm": curve.arm,
        "role": curve.role,
        "observed_grid_minimum_share": float(share[np.argmin(curve.response)]),
        "predicted_minimum_share": minimum,
        "predicted_minimum_bpb": loss,
        "fit_rmse_bpb": float(np.sqrt(np.mean((predicted - curve.response) ** 2))),
        "maximum_absolute_residual_bpb": float(np.max(np.abs(predicted - curve.response))),
        "shape": fitted.shape,
        "ridge": fitted.ridge,
        "head": json_value(fitted.head),
        "diagnostics": json_value(fitted.diagnostics),
        "nominal_epoch_scales": inputs.inventory.tolist(),
        "observed": curve.response.tolist(),
        "fitted_at_grid": predicted.tolist(),
        "dense_share": dense.tolist(),
        "dense_prediction": dense_prediction.tolist(),
    }


def plot(results: list[dict], share: np.ndarray, output: Path) -> None:
    rows = {r["curve"]: r for r in results}
    series = [
        ("target", "Target", "#333333"),
        ("unmatched_mean", "Unmatched proxy", "#0072B2"),
        ("matched_pooled", "Epoch-matched proxy (pooled)", "#D55E00"),
    ]
    with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
        for ax in axes:
            for key, label, color in series:
                r = rows[key]
                offset = r["predicted_minimum_bpb"]
                x, y = np.array(r["dense_share"]), np.array(r["dense_prediction"]) - offset
                ax.plot(x, y, color=color, label=f"{label}: {100*r['predicted_minimum_share']:.1f}%")
                ax.scatter(
                    share,
                    np.array(r["observed"]) - offset,
                    s=26,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
                ax.scatter(
                    [r["predicted_minimum_share"]],
                    [0],
                    marker="*",
                    s=110,
                    color=color,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=4,
                )
            ax.axhline(0, color="#999999", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.15)
            ax.set_xlabel("StarCoder token fraction")
            ax.set_ylabel("BPB above each fitted minimum")
        axes[0].set_title("MARINER fits; dots are measured losses")
        axes[0].set_xlim(-0.02, 1.02)
        axes[0].legend(fontsize=8, loc="upper right")
        axes[1].set_title("Near the minima")
        axes[1].set_xlim(0.28, 1.02)
        local = [
            v - r["predicted_minimum_bpb"]
            for key, _, _ in series
            for r in [rows[key]]
            for x, v in zip(r["dense_share"], r["dense_prediction"], strict=True)
            if x >= 0.3
        ]
        axes[1].set_ylim(min(-0.015, min(local) - 0.01), max(local) + 0.025)
        fig.savefig(output / "mariner_fits.png", dpi=180)
        fig.savefig(output / "mariner_fits.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan, design = json.loads(args.plan.read_text()), json.loads(args.design.read_text())
    assert design["design_sha256"] == plan["design_sha256"]
    values = pilot.verified_measurements(plan, args.measurements)
    pilot.analyze(plan, values)
    curves = input_curves(plan, values)
    share = np.array(sorted({r["percent"] for r in plan["runs"]})) / 100
    args.output.mkdir(parents=True, exist_ok=True)
    provenance_paths = [
        args.plan,
        args.measurements,
        args.design,
        Path(__file__),
        Path(models.__file__),
        Path(registry.__file__),
    ]
    provenance = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in provenance_paths}
    metadata = {
        "model_id": MODEL_ID,
        "plan_sha256": plan["plan_sha256"],
        "source_sha256": provenance,
        "runtime_versions": {
            "python": sys.version.split()[0],
            **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "matplotlib")},
        },
        "protocol": (
            "Full descriptive fits to seven distinct mixture means; leave-one-mixture-out shape/ridge/floor selection. "
            "Existing registry's StarCoder fallback: each training fold's median anchor, zero external noise margin. "
            "Physical nominal epochs for the fixed web blend and StarCoder. No epoch cap or KL penalty. "
            "Continuous minima are predictions; no training was run there."
        ),
        "matched_aggregation": (
            "Primary: trainer mean within each subset. Secondary pooled fit averages all three subset means. "
            "Individual-seed fits are sensitivity diagnostics, not confidence intervals."
        ),
        "grid_share": share.tolist(),
    }
    results = []
    for curve in curves:
        result = fit_curve(curve, share, design, plan["primary_metric"])
        results.append(result)
        (args.output / f"{curve.name}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in [
                        "curve",
                        "predicted_minimum_share",
                        "predicted_minimum_bpb",
                        "fit_rmse_bpb",
                        "shape",
                        "ridge",
                    ]
                }
            ),
            flush=True,
        )
    (args.output / "summary.json").write_text(
        json.dumps({**metadata, "fits": results}, indent=2, allow_nan=False) + "\n"
    )
    plot(results, share, args.output)


if __name__ == "__main__":
    main()
