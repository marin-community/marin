# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scipy>=1.11",
# ]
# ///
"""Plot the Ordinal-Ladder Quality TSJL shared model optimum vs best observed."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from experiments.domain_phase_mix.exploratory.two_phase_many import ordinal_quality_tier_surrogates as oq

ROOT = Path(__file__).resolve().parent
OUTPUT_PNG = ROOT / "ordinal_quality_tier_shared_optimum_comparison.png"
OUTPUT_JSON = ROOT / "ordinal_quality_tier_shared_optimum_summary.json"
OUTPUT_CSV = ROOT / "ordinal_quality_tier_shared_optimum_weights.csv"
N_SEARCH_POINTS = 15_000
N_RESTARTS = 4
MAXITER = 120
TOP_K = 10


@dataclass(frozen=True)
class SharedTsjlFit:
    """Fitted shared TSJL model and its candidate panel."""

    frame: pd.DataFrame
    spec: object
    beta: np.ndarray
    ladder_structure: oq.LadderStructure
    epoch_mode: str

    def predict(self, weights: np.ndarray) -> np.ndarray:
        """Predict objective values for weight tensors of shape (R, N, M)."""
        base_block = _build_base_block(self.spec, weights)
        shared_block, _ = oq.build_ordinal_ladder_shared_block(weights, self.ladder_structure)
        return oq.predict_ridge_blocks([base_block, shared_block], self.beta)


@dataclass(frozen=True)
class PredictedOptimum:
    """Predicted continuous optimum and nearest observed anchor."""

    phase_weights: np.ndarray
    predicted_objective: float
    nearest_observed_idx: int
    nearest_observed_distance: float


def _build_base_block(spec, feature_weights: np.ndarray) -> np.ndarray:
    literature_path = oq.resolve_literature_module_path()
    if literature_path is not None:
        literature = oq.load_module(literature_path, "literature_motivated_surrogates_shared_optimum")
        block, _ = literature.build_tsjl_features(
            replace(spec, weights=feature_weights),
            grouping="source15",
            kappa=oq.SOURCE15_KAPPA,
            p=oq.SOURCE15_POWER,
            tau_t=oq.SOURCE15_TAU_TOTAL,
            sigma_t=oq.SOURCE15_SIGMA_TOTAL,
            tau_m=oq.SOURCE15_TAU_SHIFT,
            sigma_m=oq.SOURCE15_SIGMA_SHIFT,
        )
        return block

    source_groups = oq.build_source15_groups(spec.domain_names)
    return oq.build_source15_tsjl_block(feature_weights, source_groups)


def fit_shared_tsjl(*, epoch_mode: str = oq.EPOCH_MODE_UNIT) -> SharedTsjlFit:
    """Fit the shared TSJL surrogate for optimum search."""
    frame, spec, feature_weights, _ = oq.load_candidate_summary_for_epoch_mode(
        oq.DEFAULT_CANDIDATE_SUMMARY,
        epoch_mode=epoch_mode,
    )
    mapping = oq.infer_binary_ladder_mapping(spec.domain_names)
    ladder_structure = oq.ladder_structure_from_mapping(mapping, spec.domain_names)
    base_block = _build_base_block(spec, feature_weights)
    shared_block, _ = oq.build_ordinal_ladder_shared_block(feature_weights, ladder_structure)
    beta = oq.fit_ridge_blocks([base_block, shared_block], spec.y, oq.FIT_LAMBDAS)
    return SharedTsjlFit(
        frame=frame,
        spec=spec,
        beta=beta,
        ladder_structure=ladder_structure,
        epoch_mode=epoch_mode,
    )


def _sample_simplex_points(rng: np.random.Generator, n_points: int, n_dims: int) -> np.ndarray:
    raw = rng.exponential(1.0, size=(n_points, n_dims))
    return raw / raw.sum(axis=1, keepdims=True)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp / exp.sum(axis=1, keepdims=True)


def _pack_weights(weights: np.ndarray) -> np.ndarray:
    return np.log(np.clip(weights, 1e-12, 1.0)).reshape(-1)


def _unpack_logits(logits: np.ndarray, n_phases: int, n_domains: int) -> np.ndarray:
    return _softmax_rows(logits.reshape(n_phases, n_domains))


def search_predicted_optimum(
    fit: SharedTsjlFit,
    *,
    seed: int = 0,
    n_points: int = N_SEARCH_POINTS,
    n_restarts: int = N_RESTARTS,
    maxiter: int = MAXITER,
) -> PredictedOptimum:
    """Search the continuous simplex for the model's predicted optimum."""
    rng = np.random.default_rng(seed)
    spec = fit.spec
    points = np.zeros((n_points, spec.N, spec.M), dtype=float)
    for phase_idx in range(spec.N):
        points[:, phase_idx, :] = _sample_simplex_points(rng, n_points, spec.M)

    sampled_predictions = fit.predict(points)
    top_ids = np.argsort(sampled_predictions)[-n_restarts:][::-1]
    starts = [points[idx] for idx in top_ids]
    starts.append(spec.weights[int(np.argmax(spec.y))])

    best_weights: np.ndarray | None = None
    best_objective = float("-inf")
    for start in starts:

        def objective(logits: np.ndarray) -> float:
            candidate = _unpack_logits(logits, spec.N, spec.M)[None, :, :]
            return -float(fit.predict(candidate)[0])

        result = minimize(
            objective,
            _pack_weights(start),
            method="L-BFGS-B",
            options={"maxiter": maxiter},
        )
        candidate = _unpack_logits(np.asarray(result.x, dtype=float), spec.N, spec.M)
        predicted = float(fit.predict(candidate[None, :, :])[0])
        if predicted > best_objective:
            best_objective = predicted
            best_weights = candidate

    if best_weights is None:
        raise RuntimeError("Failed to find a predicted optimum")

    phase_tv = 0.5 * np.abs(spec.weights - best_weights[None, :, :]).sum(axis=2)
    mean_tv = phase_tv.mean(axis=1)
    nearest_idx = int(np.argmin(mean_tv))
    return PredictedOptimum(
        phase_weights=best_weights,
        predicted_objective=best_objective,
        nearest_observed_idx=nearest_idx,
        nearest_observed_distance=float(mean_tv[nearest_idx]),
    )


def _best_observed_idx(spec) -> int:
    return int(np.argmax(spec.y))


def _weight_table(
    *,
    fit: SharedTsjlFit,
    optimum: PredictedOptimum,
    best_idx: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    best_weights = fit.spec.weights[best_idx]
    for phase_idx, phase_name in enumerate(fit.spec.phase_names):
        for domain_idx, domain_name in enumerate(fit.spec.domain_names):
            rows.append(
                {
                    "phase_name": phase_name,
                    "domain_name": domain_name,
                    "predicted_optimum_weight": float(optimum.phase_weights[phase_idx, domain_idx]),
                    "best_observed_weight": float(best_weights[phase_idx, domain_idx]),
                    "weight_delta": float(
                        optimum.phase_weights[phase_idx, domain_idx] - best_weights[phase_idx, domain_idx]
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def _selected_domains(
    weights_table: pd.DataFrame,
    *,
    phase_name: str,
    top_k: int = TOP_K,
) -> list[str]:
    phase_table = weights_table[weights_table["phase_name"] == phase_name].copy()
    phase_table["importance"] = phase_table[["predicted_optimum_weight", "best_observed_weight"]].max(axis=1)
    return list(phase_table.sort_values("importance", ascending=False).head(top_k)["domain_name"])


def plot_optimum_comparison(
    *,
    fit: SharedTsjlFit,
    optimum: PredictedOptimum,
    best_idx: int,
    output_path: Path = OUTPUT_PNG,
) -> None:
    """Render the predicted optimum vs best observed weights."""
    weights_table = _weight_table(fit=fit, optimum=optimum, best_idx=best_idx)
    best_name = str(fit.frame.iloc[best_idx]["candidate_run_name"])
    best_actual = float(fit.spec.y[best_idx])
    best_predicted = float(fit.predict(fit.spec.weights[best_idx : best_idx + 1])[0])
    nearest_idx = optimum.nearest_observed_idx
    nearest_name = str(fit.frame.iloc[nearest_idx]["candidate_run_name"])
    nearest_actual = float(fit.spec.y[nearest_idx])

    cmap = plt.get_cmap("RdYlGn_r")
    predicted_color = cmap(0.15)
    best_color = cmap(0.85)

    fig, axes = plt.subplots(1, len(fit.spec.phase_names), figsize=(16, 8), sharex=False)
    if len(fit.spec.phase_names) == 1:
        axes = [axes]

    for axis, phase_name in zip(axes, fit.spec.phase_names, strict=True):
        domains = _selected_domains(weights_table, phase_name=phase_name)
        phase_table = weights_table[weights_table["phase_name"] == phase_name].set_index("domain_name").loc[domains]
        y = np.arange(len(domains))
        axis.barh(
            y - 0.18,
            phase_table["predicted_optimum_weight"],
            height=0.34,
            color=predicted_color,
            label="Predicted optimum",
        )
        axis.barh(
            y + 0.18,
            phase_table["best_observed_weight"],
            height=0.34,
            color=best_color,
            label=f"Best observed ({best_name})",
        )
        axis.set_yticks(y)
        axis.set_yticklabels(domains, fontsize=9)
        axis.invert_yaxis()
        axis.set_xlabel("Mixture weight")
        axis.set_title(phase_name.replace("_", " ").title())
        axis.grid(axis="x", linestyle="--", alpha=0.25)
        x_limit = (
            max(
                phase_table["predicted_optimum_weight"].max(),
                phase_table["best_observed_weight"].max(),
            )
            * 1.12
        )
        axis.set_xlim(0.0, x_limit)

    axes[0].legend(loc="lower right")
    fig.suptitle("Ordinal-Ladder Quality TSJL (shared): predicted optimum vs best observed", fontsize=14, y=0.98)
    fig.text(
        0.5,
        0.02,
        (
            f"Fit target: candidate-mean lm_eval/mmlu_sl_verb_5shot/choice_logprob_norm | "
            f"Best observed actual = {best_actual:.6f} ({best_name}) | "
            f"Best observed model prediction = {best_predicted:.6f} | "
            f"Predicted optimum model prediction = {optimum.predicted_objective:.6f} | "
            f"Nearest observed to optimum = {nearest_name} "
            f"(actual {nearest_actual:.6f}, mean phase TV {optimum.nearest_observed_distance:.3f})"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    *,
    fit: SharedTsjlFit,
    optimum: PredictedOptimum,
    best_idx: int,
    summary_path: Path = OUTPUT_JSON,
    weights_csv_path: Path = OUTPUT_CSV,
) -> None:
    """Persist optimum summary JSON and the full per-domain weight table."""
    best_name = str(fit.frame.iloc[best_idx]["candidate_run_name"])
    best_actual = float(fit.spec.y[best_idx])
    best_predicted = float(fit.predict(fit.spec.weights[best_idx : best_idx + 1])[0])
    nearest_idx = optimum.nearest_observed_idx
    nearest_name = str(fit.frame.iloc[nearest_idx]["candidate_run_name"])
    nearest_actual = float(fit.spec.y[nearest_idx])
    nearest_predicted = float(fit.predict(fit.spec.weights[nearest_idx : nearest_idx + 1])[0])

    summary = {
        "model": oq.MODEL_NAME,
        "epoch_mode": fit.epoch_mode,
        "objective_metric": "choice_logprob_norm_mean",
        "best_observed_run_name": best_name,
        "best_observed_actual": best_actual,
        "best_observed_predicted": best_predicted,
        "predicted_optimum_predicted": optimum.predicted_objective,
        "predicted_gap_vs_best_observed_predicted": optimum.predicted_objective - best_predicted,
        "predicted_gap_vs_best_observed_actual": optimum.predicted_objective - best_actual,
        "nearest_observed_run_name": nearest_name,
        "nearest_observed_actual": nearest_actual,
        "nearest_observed_predicted": nearest_predicted,
        "nearest_observed_mean_phase_tv": optimum.nearest_observed_distance,
        "phase_names": list(fit.spec.phase_names),
        "domain_names": list(fit.spec.domain_names),
        "predicted_optimum_phase_weights": {
            phase_name: {
                domain_name: float(optimum.phase_weights[phase_idx, domain_idx])
                for domain_idx, domain_name in enumerate(fit.spec.domain_names)
            }
            for phase_idx, phase_name in enumerate(fit.spec.phase_names)
        },
        "best_observed_phase_weights": {
            phase_name: {
                domain_name: float(fit.spec.weights[best_idx, phase_idx, domain_idx])
                for domain_idx, domain_name in enumerate(fit.spec.domain_names)
            }
            for phase_idx, phase_name in enumerate(fit.spec.phase_names)
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _weight_table(fit=fit, optimum=optimum, best_idx=best_idx).to_csv(weights_csv_path, index=False)


def main() -> None:
    fit = fit_shared_tsjl(epoch_mode=oq.EPOCH_MODE_UNIT)
    optimum = search_predicted_optimum(fit)
    best_idx = _best_observed_idx(fit.spec)
    plot_optimum_comparison(fit=fit, optimum=optimum, best_idx=best_idx)
    write_summary(fit=fit, optimum=optimum, best_idx=best_idx)
    print(f"Wrote {OUTPUT_PNG}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
