# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy"]
# ///
"""Materialize one-phase uncheatable-optimized validation mixtures.

The one-phase comparison for issue 6609 should validate mixtures optimized for
``eval/uncheatable_eval/bpb``.  This script creates the two launchable tied-phase
mixture CSVs:

* OLMix delta=0.01, KL=0.05, aggregate repetition cap=4.
* Effective-exposure DSP, LINEAR_REG=0.01, tied one-phase KL=0.1.

Both are optimized from the same deletion-augmented 300M uncheatable fit panel.
The OLMix artifact is copied from the existing Huber-delta sweep; the DSP
artifact is refit here and constrained to ``phase_0 == phase_1`` at proposal
time.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_dsp_vs_olmix_deletion_augmented_300m as dsp_compare,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "reference_outputs" / "one_phase_uncheatable_validation_mixtures_300m_20260629"
)
DEFAULT_OLMIX_SOURCE = (
    SCRIPT_DIR
    / "reference_outputs"
    / "olmix_huber_delta_sweep_300m_20260625"
    / "delta_0p01"
    / "uncheatable_eval_bpb_single_simplex_tied_phases_rep_cap4"
    / "proposed_mixture_weights.csv"
)
DEFAULT_OLMIX_SUMMARY = DEFAULT_OLMIX_SOURCE.with_name("fit_summary.json")
OLMIX_KEY = "olmix_onephase_uncheatable_d001_kl005_cap4"
DSP_KEY = "dsp_onephase_effexp_uncheatable_kl0p1"


@dataclass(frozen=True)
class MixtureSummary:
    key: str
    model_family: str
    target_metric: str
    source: str
    fit_panel_rows: int
    n_signal_rows: int
    n_deletion_rows: int
    n_proportional_reference_rows: int
    train_rmse: float | None
    train_spearman: float | None
    oof_rmse: float | None
    oof_spearman: float | None
    fold_mean_regret_at_1: float | None
    lower_tail_optimism: float | None
    low_tail_rmse: float | None
    predicted_objective: float | None
    regularized_objective: float | None
    kl_reg: float
    linear_reg: float | None
    huber_delta: float | None
    max_simulated_epoch: float
    q95_simulated_epoch: float
    max_epoch_multiplier: float
    q95_epoch_multiplier: float
    mean_phase_tv_to_proportional: float
    optimizer_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--olmix-source", type=Path, default=DEFAULT_OLMIX_SOURCE)
    parser.add_argument("--dsp-kl-reg", type=float, default=0.1)
    parser.add_argument("--dsp-linear-reg", type=float, default=0.01)
    parser.add_argument("--dsp-maxiter", type=int, default=dsp.FIT_MAXITER)
    parser.add_argument("--dsp-coarse-top-k", type=int, default=dsp.START_TOP_K)
    parser.add_argument("--dsp-basin-hopping-iters", type=int, default=1)
    parser.add_argument("--dsp-random-starts", type=int, default=64)
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / weights.sum()


def weights_to_logits(weights: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(weights, dtype=float), 1e-12, 1.0))


def tied_weights(single_weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(single_weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError(f"Expected one-dimensional weights, got shape {weights.shape}")
    weights = weights / weights.sum()
    return np.stack([weights, weights], axis=0)


def kl_to_proportional(weights: np.ndarray, natural: np.ndarray) -> float:
    phase_weights = tied_weights(weights)
    return olmix.weighted_multiclass_kl(phase_weights, natural, olmix.PHASE_FRACTIONS)


def proposal_starts(
    model: dsp.FittedDSPModel,
    packet: dsp.PacketData,
    natural: np.ndarray,
    olmix_weights: np.ndarray,
    *,
    num_random: int,
) -> list[np.ndarray]:
    train_predictions = dsp.predict(model, packet.w)
    exposure_average = np.einsum("p,npd->nd", olmix.PHASE_FRACTIONS, packet.w)
    starts: list[np.ndarray] = [natural, olmix_weights]

    for idx in np.argsort(train_predictions)[: min(32, len(train_predictions))]:
        starts.append(exposure_average[int(idx)] / exposure_average[int(idx)].sum())
    for idx in np.argsort(packet.y)[: min(16, len(packet.y))]:
        starts.append(exposure_average[int(idx)] / exposure_average[int(idx)].sum())

    two_phase_dsp = (
        SCRIPT_DIR
        / "reference_outputs"
        / "dsp_effective_exposure_l2_kl_sweep_deletion_augmented_300m_20260625"
        / "dsp_effective_exposure_l2_0.01_kl_only_0.1"
        / "proposed_mixture_weights.csv"
    )
    if two_phase_dsp.exists():
        frame = pd.read_csv(two_phase_dsp)
        aggregate = frame["aggregate_weight"].to_numpy(dtype=float)
        starts.append(aggregate / aggregate.sum())

    rng = np.random.default_rng(0)
    natural_logits = weights_to_logits(natural)
    for scale in (0.15, 0.35, 0.75, 1.25):
        for _ in range(max(1, num_random // 4)):
            starts.append(softmax(natural_logits + rng.normal(0.0, scale, size=len(natural))))

    deduped: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for start in starts:
        normalized = np.clip(np.asarray(start, dtype=float), 1e-12, None)
        normalized /= normalized.sum()
        key = tuple(np.round(normalized, 12))
        if key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return deduped


def fit_effective_exposure_dsp(
    packet: dsp.PacketData,
    *,
    linear_reg: float,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
) -> tuple[dsp.FittedDSPModel, pd.DataFrame, np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    previous_linear_reg = dsp.LINEAR_REG
    dsp.LINEAR_REG = float(linear_reg)
    try:
        model, tuning = dsp.fit_variant(
            packet,
            dsp.VARIANTS["effective_exposure"],
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
            basin_hopping_iters=basin_hopping_iters,
        )
        train_prediction = dsp.predict(model, packet.w)
        train_rmse, train_mae, train_pearson, train_spearman = dsp_compare.regression_metrics(
            packet.y, train_prediction
        )
        oof_prediction, folds = dsp_compare.fit_dsp_oof_predictions(packet, model)
        oof_metrics = olmix.predictive_diagnostics(packet.y, oof_prediction, folds)
        train_metrics = {
            "rmse": float(train_rmse),
            "mae": float(train_mae),
            "pearson": float(train_pearson),
            "spearman": float(train_spearman),
        }
        return (
            model,
            tuning,
            train_prediction,
            oof_prediction,
            train_metrics,
            {key: float(value) for key, value in oof_metrics.items()},
        )
    finally:
        dsp.LINEAR_REG = previous_linear_reg


def optimize_tied_dsp_kl(
    model: dsp.FittedDSPModel,
    natural: np.ndarray,
    starts: list[np.ndarray],
    *,
    kl_reg: float,
) -> tuple[np.ndarray, float, float, str]:
    def objective(logits: np.ndarray) -> float:
        weights = softmax(logits)
        prediction = float(dsp.predict(model, tied_weights(weights)[None, :, :])[0])
        return prediction + float(kl_reg) * kl_to_proportional(weights, natural)

    best: Any | None = None
    for start in starts:
        result = minimize(
            objective,
            weights_to_logits(start),
            method="L-BFGS-B",
            options={"maxiter": 1200, "ftol": 1e-11, "maxls": 40},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError("Tied DSP KL optimization produced no result")
    weights = softmax(np.asarray(best.x, dtype=float))
    predicted = float(dsp.predict(model, tied_weights(weights)[None, :, :])[0])
    return weights, predicted, float(best.fun), str(best.message)


def write_weights(
    path: Path,
    domains: list[str],
    weights: np.ndarray,
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
) -> pd.DataFrame:
    phase_weights = tied_weights(weights)
    ratios = phase_weights / np.clip(np.stack([natural, natural], axis=0), 1e-12, None)
    sim_epochs = olmix.simulated_epochs(phase_weights, token_counts, target_budget=target_budget)
    frame = pd.DataFrame(
        {
            "domain": domains,
            "proportional": natural,
            "phase_0_weight": phase_weights[0],
            "phase_1_weight": phase_weights[1],
            "aggregate_weight": olmix.aggregate_phase_weights(phase_weights),
            "available_tokens": token_counts,
            "simulated_epochs": sim_epochs,
            "phase_0_epoch_multiplier": ratios[0],
            "phase_1_epoch_multiplier": ratios[1],
            "phase_0_delta": phase_weights[0] - natural,
            "phase_1_delta": phase_weights[1] - natural,
        }
    )
    frame["max_abs_delta"] = frame[["phase_0_delta", "phase_1_delta"]].abs().max(axis=1)
    frame.to_csv(path, index=False)
    return frame


def summarize_weights(frame: pd.DataFrame, natural: np.ndarray) -> dict[str, float]:
    phase_weights = frame[["phase_0_weight", "phase_1_weight"]].to_numpy(dtype=float).T
    reference = np.stack([natural, natural], axis=0)
    ratios = phase_weights / np.clip(reference, 1e-12, None)
    return {
        "max_simulated_epoch": float(frame["simulated_epochs"].max()),
        "q95_simulated_epoch": float(frame["simulated_epochs"].quantile(0.95)),
        "max_epoch_multiplier": float(np.max(ratios)),
        "q95_epoch_multiplier": float(np.quantile(ratios, 0.95)),
        "mean_phase_tv_to_proportional": float(0.5 * np.abs(phase_weights - reference).sum(axis=1).mean()),
    }


def read_olmix_result(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text())
    result = payload.get("result")
    if not isinstance(result, dict):
        return {}
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _signal, columns, domains, natural = olmix.load_raw_signal_panel()
    token_counts = olmix.load_domain_token_counts(domains)
    target_budget = olmix.load_target_budget()
    panel, metadata = olmix.build_uncheatable_panel(columns)
    packet = dsp_compare.build_dsp_packet(panel, columns, domains, token_counts, target_budget)

    if not args.olmix_source.exists():
        raise FileNotFoundError(args.olmix_source)
    olmix_output = args.output_dir / f"{OLMIX_KEY}.csv"
    shutil.copyfile(args.olmix_source, olmix_output)
    olmix_frame = pd.read_csv(olmix_output)
    olmix_single = olmix_frame["aggregate_weight"].to_numpy(dtype=float).copy()
    olmix_single /= olmix_single.sum()
    olmix_result = read_olmix_result(args.olmix_source.with_name("fit_summary.json"))

    dsp_model, tuning, train_prediction, oof_prediction, train_metrics, oof_metrics = fit_effective_exposure_dsp(
        packet,
        linear_reg=float(args.dsp_linear_reg),
        maxiter=int(args.dsp_maxiter),
        coarse_top_k=int(args.dsp_coarse_top_k),
        basin_hopping_iters=int(args.dsp_basin_hopping_iters),
    )
    tuning.to_csv(args.output_dir / "dsp_effective_exposure_tuning.csv", index=False)
    pd.DataFrame(
        {
            "run_name": panel["run_name"],
            "panel_source": panel["panel_source"],
            "observed_uncheatable_bpb": packet.y,
            "dsp_train_prediction": train_prediction,
            "dsp_oof_prediction": oof_prediction,
            "dsp_train_residual": train_prediction - packet.y,
            "dsp_oof_residual": oof_prediction - packet.y,
        }
    ).to_csv(args.output_dir / "dsp_fit_panel_predictions.csv", index=False)
    (args.output_dir / "dsp_effective_exposure_model.json").write_text(
        json.dumps(
            dsp.model_to_json(
                dsp_model,
                {
                    "target": olmix.UNCHEATABLE_TARGET,
                    "variant": "effective_exposure",
                    "linear_reg": float(args.dsp_linear_reg),
                    "deployment_constraint": "phase_0 == phase_1",
                },
            ),
            indent=2,
        )
    )

    starts = proposal_starts(
        dsp_model,
        packet,
        natural,
        olmix_single,
        num_random=int(args.dsp_random_starts),
    )
    dsp_single, dsp_predicted, dsp_regularized, dsp_status = optimize_tied_dsp_kl(
        dsp_model,
        natural,
        starts,
        kl_reg=float(args.dsp_kl_reg),
    )
    dsp_output = args.output_dir / f"{DSP_KEY}.csv"
    dsp_frame = write_weights(dsp_output, domains, dsp_single, natural, token_counts, target_budget)

    summaries = [
        MixtureSummary(
            key=OLMIX_KEY,
            model_family="OLMix",
            target_metric=olmix.UNCHEATABLE_TARGET,
            source=str(args.olmix_source),
            fit_panel_rows=int(len(panel)),
            n_signal_rows=int(panel["panel_source"].eq("qsplit_signal").sum()),
            n_deletion_rows=int(panel["panel_source"].eq("domain_deletion").sum()),
            n_proportional_reference_rows=int(metadata.get("n_proportional_reference_rows", 0)),
            train_rmse=olmix_result.get("train_rmse"),
            train_spearman=olmix_result.get("train_spearman"),
            oof_rmse=olmix_result.get("oof_rmse"),
            oof_spearman=olmix_result.get("oof_spearman"),
            fold_mean_regret_at_1=olmix_result.get("fold_mean_regret_at_1"),
            lower_tail_optimism=olmix_result.get("lower_tail_optimism"),
            low_tail_rmse=olmix_result.get("low_tail_rmse"),
            predicted_objective=olmix_result.get("predicted_objective"),
            regularized_objective=olmix_result.get("regularized_objective"),
            kl_reg=0.05,
            linear_reg=None,
            huber_delta=0.01,
            optimizer_status=str(olmix_result.get("cvxpy_status", "copied")),
            **summarize_weights(olmix_frame, natural),
        ),
        MixtureSummary(
            key=DSP_KEY,
            model_family="DSP",
            target_metric=olmix.UNCHEATABLE_TARGET,
            source="materialized_by_this_script",
            fit_panel_rows=int(len(panel)),
            n_signal_rows=int(panel["panel_source"].eq("qsplit_signal").sum()),
            n_deletion_rows=int(panel["panel_source"].eq("domain_deletion").sum()),
            n_proportional_reference_rows=int(metadata.get("n_proportional_reference_rows", 0)),
            train_rmse=float(train_metrics["rmse"]),
            train_spearman=float(train_metrics["spearman"]),
            oof_rmse=float(oof_metrics["rmse"]),
            oof_spearman=float(oof_metrics["spearman"]),
            fold_mean_regret_at_1=float(oof_metrics["fold_mean_regret_at_1"]),
            lower_tail_optimism=float(oof_metrics["lower_tail_optimism"]),
            low_tail_rmse=float(oof_metrics["low_tail_rmse"]),
            predicted_objective=float(dsp_predicted),
            regularized_objective=float(dsp_regularized),
            kl_reg=float(args.dsp_kl_reg),
            linear_reg=float(args.dsp_linear_reg),
            huber_delta=None,
            optimizer_status=dsp_status,
            **summarize_weights(dsp_frame, natural),
        ),
    ]
    summary_frame = pd.DataFrame([asdict(row) for row in summaries])
    summary_frame.to_csv(args.output_dir / "mixture_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "target_metric": olmix.UNCHEATABLE_TARGET,
                "fit_panel_rows": int(len(panel)),
                "fit_panel_metadata": metadata,
                "output_mixtures": [OLMIX_KEY, DSP_KEY],
                "dsp_start_count": len(starts),
                "summaries": [asdict(row) for row in summaries],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(summary_frame.to_string(index=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
