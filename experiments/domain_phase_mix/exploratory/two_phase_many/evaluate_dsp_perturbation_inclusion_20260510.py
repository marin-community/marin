# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "tabulate"]
# ///
"""Measure whether proportional perturbation rows improve DSP fit quality.

This is a diagnostic for deciding whether local one-at-a-time perturbations
should be included in future mixture swarms. It compares original-swarm DSP
fits against augmented fits that include the 55 proportional perturbation rows
at the same scale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_and_plot_grp_power_family_penalty_no_l2_60m_vs_300m import (
    OBJECTIVE_METRIC,
    RUN_SET_60M,
    RUN_SET_300M,
    _build_fit_frame,
    _metric_frame,
    _packet_from_frame,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
    CV_SEED,
    VARIANTS,
    DSPVariant,
    FittedDSPModel,
    _fit_linear_head,
    _fit_variant,
    _oof_predictions,
    _predict,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    PacketData,
    load_two_phase_many_packet,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dsp_perturbation_inclusion_ablation_20260510"
PERTURBATION_FAMILY_BY_SCALE = {
    "60m_1p2b": "proportional_perturbation_60m_1p2b",
    "300m_6b": "proportional_perturbation_300m_6b",
}
RUN_SET_BY_SCALE = {
    "60m_1p2b": RUN_SET_60M,
    "300m_6b": RUN_SET_300M,
}
DISPLAY_SCALE = {
    "60m_1p2b": "60M/1.2B",
    "300m_6b": "100M/6B",
}
VARIANT_NAMES = (
    "dsp_phase_benefit_penalty_nnls",
    "dsp_effective_exposure_penalty_nnls",
    "dsp_phase_benefit_saturation_penalty_nnls",
    "dsp_saturation_penalty_split_nnls",
)
N_SPLITS = 5
BOOTSTRAP_REPS = 5000


@dataclass(frozen=True)
class PanelFit:
    """A fitted model and its packet."""

    scale: str
    fit_panel: str
    variant: DSPVariant
    packet: PacketData
    model: FittedDSPModel


def _variant_by_name(name: str) -> DSPVariant:
    for variant in VARIANTS:
        if variant.name == name:
            return variant
    raise ValueError(f"Unknown DSP variant {name!r}")


def _canonical_weight_columns() -> list[str]:
    reference = load_two_phase_many_packet(target=OBJECTIVE_METRIC)
    return [
        f"{phase_name}_{domain_name}" for phase_name in ("phase_0", "phase_1") for domain_name in reference.domain_names
    ]


def _build_perturbation_fit_frame(frame: pd.DataFrame, *, scale: str) -> pd.DataFrame:
    family = PERTURBATION_FAMILY_BY_SCALE[scale]
    weight_columns = [column for column in _canonical_weight_columns() if column in frame.columns]
    id_columns = [
        column
        for column in (
            "run_id",
            "run_name",
            "scale",
            "cohort",
            "family",
            "row_kind",
            "source_experiment",
            "checkpoint_root",
            "status",
            "intervention_id",
            "intervention_type",
            "target_domain",
            "target_family",
            "tv_distance",
        )
        if column in frame.columns
    ]
    subset = frame.loc[
        frame["scale"].eq(scale)
        & frame["family"].eq(family)
        & frame["row_kind"].eq("proportional_perturbation")
        & frame[OBJECTIVE_METRIC].notna(),
        id_columns + weight_columns + [OBJECTIVE_METRIC],
    ].copy()
    subset = subset.rename(columns={OBJECTIVE_METRIC: "objective_metric"}).reset_index(drop=True)
    subset[weight_columns] = subset[weight_columns].fillna(0.0)
    if subset["run_name"].duplicated().any():
        duplicates = sorted(subset.loc[subset["run_name"].duplicated(), "run_name"].astype(str).unique())
        raise ValueError(f"Duplicate perturbation run_name rows for {scale}: {duplicates[:5]}")
    return subset


def _packet_from_fit_frame(frame: pd.DataFrame, *, name: str) -> PacketData:
    return _packet_from_frame(frame, name=name).base


def _fit_panel(scale: str, panel_name: str, frame: pd.DataFrame, variant: DSPVariant) -> PanelFit:
    packet = _packet_from_fit_frame(frame, name=f"{variant.name}_{scale}_{panel_name}")
    model, _tuning = _fit_variant(packet, variant)
    return PanelFit(scale=scale, fit_panel=panel_name, variant=variant, packet=packet, model=model)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = pred - y
    return {
        "n": float(len(y)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "spearman": float(spearmanr(y, pred).statistic),
        "pearson": float(pearsonr(y, pred).statistic),
    }


def _bootstrap_rmse_delta(
    y: np.ndarray,
    baseline_pred: np.ndarray,
    candidate_pred: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    """Return paired bootstrap CI for candidate RMSE minus baseline RMSE."""

    rng = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAP_REPS, dtype=float)
    n = len(y)
    for idx in range(BOOTSTRAP_REPS):
        sample = rng.integers(0, n, size=n)
        baseline_rmse = float(np.sqrt(np.mean((baseline_pred[sample] - y[sample]) ** 2)))
        candidate_rmse = float(np.sqrt(np.mean((candidate_pred[sample] - y[sample]) ** 2)))
        deltas[idx] = candidate_rmse - baseline_rmse
    return {
        "rmse_delta_paired_bootstrap_mean": float(np.mean(deltas)),
        "rmse_delta_paired_bootstrap_ci025": float(np.quantile(deltas, 0.025)),
        "rmse_delta_paired_bootstrap_ci975": float(np.quantile(deltas, 0.975)),
    }


def _oof_for_test_pool(
    packet: PacketData,
    model: FittedDSPModel,
    *,
    test_pool_mask: np.ndarray,
    always_train_mask: np.ndarray,
) -> np.ndarray:
    """OOF predictions for one pool, with another pool always in training."""

    test_pool_indices = np.flatnonzero(test_pool_mask)
    if len(test_pool_indices) < N_SPLITS:
        raise ValueError(f"Need at least {N_SPLITS} test-pool rows, got {len(test_pool_indices)}")
    oof = np.full_like(packet.y, fill_value=np.nan, dtype=float)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    for train_local, test_local in kf.split(test_pool_indices):
        train_idx = np.concatenate([np.flatnonzero(always_train_mask), test_pool_indices[train_local]])
        test_idx = test_pool_indices[test_local]
        fold_model = _fit_linear_head(
            packet.w[train_idx],
            packet.y[train_idx],
            packet,
            model.variant,
            model.params,
        )
        oof[test_idx] = _predict(fold_model, packet.w[test_idx], packet)
    return oof


def _evaluate_scale_variant(
    metric_frame: pd.DataFrame,
    *,
    scale: str,
    variant: DSPVariant,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    original_frame = _build_fit_frame(metric_frame, scale=scale, run_set=RUN_SET_BY_SCALE[scale])
    perturb_frame = _build_perturbation_fit_frame(metric_frame, scale=scale)
    if len(perturb_frame) != 55:
        raise ValueError(f"Expected 55 perturbation rows for {scale}, found {len(perturb_frame)}")

    original_frame = original_frame.copy()
    perturb_frame = perturb_frame.copy()
    original_frame["fit_source_panel"] = "original"
    perturb_frame["fit_source_panel"] = "perturbation"
    augmented_frame = pd.concat([original_frame, perturb_frame], ignore_index=True)
    if augmented_frame["run_name"].duplicated().any():
        duplicates = sorted(
            augmented_frame.loc[augmented_frame["run_name"].duplicated(), "run_name"].astype(str).unique()
        )
        raise ValueError(f"Duplicate run_name rows in augmented {scale}: {duplicates[:5]}")

    original_fit = _fit_panel(scale, "original_only", original_frame, variant)
    augmented_fit = _fit_panel(scale, "original_plus_perturbation", augmented_frame, variant)
    perturb_packet = _packet_from_fit_frame(perturb_frame, name=f"{variant.name}_{scale}_perturbation_external")

    original_oof = _oof_predictions(original_fit.packet, original_fit.model)
    original_external_on_perturb = _predict(original_fit.model, perturb_packet.w, perturb_packet)

    source_panel = augmented_fit.packet.frame["fit_source_panel"].astype(str).to_numpy()
    original_mask = source_panel == "original"
    perturb_mask = source_panel == "perturbation"
    augmented_oof_original = _oof_for_test_pool(
        augmented_fit.packet,
        augmented_fit.model,
        test_pool_mask=original_mask,
        always_train_mask=perturb_mask,
    )[original_mask]
    augmented_oof_perturb = _oof_for_test_pool(
        augmented_fit.packet,
        augmented_fit.model,
        test_pool_mask=perturb_mask,
        always_train_mask=original_mask,
    )[perturb_mask]
    combined_oof = _oof_predictions(augmented_fit.packet, augmented_fit.model)

    rows: list[dict[str, Any]] = []
    comparisons = (
        (
            "original_rows",
            "original_only_oof",
            original_fit.packet.y,
            original_oof,
            None,
        ),
        (
            "original_rows",
            "augmented_oof_original_rows",
            original_fit.packet.y,
            augmented_oof_original,
            original_oof,
        ),
        (
            "perturbation_rows",
            "original_only_external",
            perturb_packet.y,
            original_external_on_perturb,
            None,
        ),
        (
            "perturbation_rows",
            "augmented_oof_perturbation_rows",
            perturb_packet.y,
            augmented_oof_perturb,
            original_external_on_perturb,
        ),
        (
            "combined_rows",
            "augmented_random_oof",
            augmented_fit.packet.y,
            combined_oof,
            None,
        ),
    )
    prediction_frames = []
    for eval_panel, evaluation, y, pred, baseline_pred in comparisons:
        row: dict[str, Any] = {
            "scale": scale,
            "display_scale": DISPLAY_SCALE[scale],
            "variant": variant.name,
            "eval_panel": eval_panel,
            "evaluation": evaluation,
            "original_n": len(original_frame),
            "perturbation_n": len(perturb_frame),
        }
        row.update(_metrics(y, pred))
        if baseline_pred is not None:
            row.update(_bootstrap_rmse_delta(y, baseline_pred, pred, seed=CV_SEED))
            row["rmse_delta_vs_baseline"] = row["rmse"] - float(np.sqrt(np.mean((baseline_pred - y) ** 2)))
        rows.append(row)

        prediction_frames.append(
            pd.DataFrame(
                {
                    "scale": scale,
                    "variant": variant.name,
                    "eval_panel": eval_panel,
                    "evaluation": evaluation,
                    "actual": y,
                    "predicted": pred,
                }
            )
        )

    return rows, pd.concat(prediction_frames, ignore_index=True)


def _write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    def table(frame: pd.DataFrame) -> str:
        cols = [
            "display_scale",
            "variant",
            "eval_panel",
            "evaluation",
            "n",
            "rmse",
            "spearman",
            "rmse_delta_vs_baseline",
            "rmse_delta_paired_bootstrap_ci025",
            "rmse_delta_paired_bootstrap_ci975",
        ]
        existing = [column for column in cols if column in frame.columns]
        return frame[existing].to_markdown(index=False, floatfmt=".6f")

    canonical = summary.loc[summary["variant"].eq("dsp_phase_benefit_penalty_nnls")].copy()
    effective = summary.loc[summary["variant"].eq("dsp_effective_exposure_penalty_nnls")].copy()
    lines = [
        "# DSP perturbation inclusion ablation",
        "",
        f"Objective: `{OBJECTIVE_METRIC}`.",
        "",
        "Question: does adding the 55 proportional perturbation rows at a scale improve DSP generalization enough to justify including similar local interventions in future swarms?",
        "",
        "Evaluation design:",
        "- `original_only_oof`: standard OOF on the original swarm rows.",
        "- `original_only_external`: model fit on original rows, evaluated on perturbation rows as an external local-intervention holdout.",
        "- `augmented_oof_original_rows`: OOF on original rows, where all perturbation rows are always available in the training folds.",
        "- `augmented_oof_perturbation_rows`: OOF on perturbation rows, where all original rows are always available in the training folds.",
        "- Negative `rmse_delta_vs_baseline` means the augmented fit improved RMSE against the relevant baseline.",
        "",
        "## Canonical DSP",
        "",
        table(canonical),
        "",
        "## Effective-exposure empirical comparator",
        "",
        table(effective),
        "",
        "## Interpretation",
        "",
        "Use perturbation rows if they improve perturbation-row OOF materially without degrading original-row OOF. If they only improve local perturbation rows and do not help original-row OOF, they are still useful as targeted gradient/intervention diagnostics, but should not replace broad randomized swarm points.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_frame = _metric_frame()
    summary_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for scale in ("60m_1p2b", "300m_6b"):
        for variant_name in VARIANT_NAMES:
            variant = _variant_by_name(variant_name)
            print(f"Fitting {DISPLAY_SCALE[scale]} {variant.name}", flush=True)
            rows, predictions = _evaluate_scale_variant(metric_frame, scale=scale, variant=variant)
            summary_rows.extend(rows)
            prediction_frames.append(predictions)

    summary = pd.DataFrame.from_records(summary_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2))
    _write_report(summary, OUTPUT_DIR)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote artifacts to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
