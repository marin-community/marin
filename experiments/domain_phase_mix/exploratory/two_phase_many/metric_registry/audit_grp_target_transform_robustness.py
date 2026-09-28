# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "tabulate"]
# ///
"""Audit GRP no-L2 robustness to target scaling and transformations.

The GRP no-L2 head is a non-negative least-squares fit on a fixed nonlinear
design. Positive affine target transforms should preserve prediction rankings
and nonlinear-parameter selection. Negation is different: it reverses the
monotonic direction encoded by the non-negative head unless the metric is first
converted to the canonical lower-is-better model target.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_grp_power_family_penalty_no_l2_retune import (
    REG_FIXED,
    VARIANT_NAME,
    _no_l2_param_keys,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import build_two_phase_many_loop_config
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.fit_grp_no_l2_benchmark_aggregates import (
    DEFAULT_FAMILY_SCHEME,
    MODEL_TARGET_COLUMN,
    _expanded_start_bank,
    _generic_family_packet_from_base,
    _oof_target_metrics,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    build_penalty_calibration_surrogate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    PacketData,
)
from experiments.domain_phase_mix.static_batch_selection import build_dataset_spec_from_frame

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_MATRIX = SCRIPT_DIR / "raw_metric_matrix_300m" / "raw_metric_matrix_300m.csv"
NOISE_MATRIX = SCRIPT_DIR / "raw_metric_matrix_300m" / "noise_baseline_run00097_variable_subset_300m.csv"
OUTPUT_DIR = SCRIPT_DIR.parent / "reference_outputs" / "grp_target_transform_robustness_20260504"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
REPORT_MD = OUTPUT_DIR / "report.md"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"
CV_SEED = 0


@dataclass(frozen=True)
class MetricSpec:
    """One metric to audit."""

    slug: str
    metric_key: str
    higher_is_better: bool


@dataclass(frozen=True)
class TransformSpec:
    """One canonical-target transformation."""

    slug: str
    description: str
    transform: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray] | None
    preserves_order: bool
    preserves_snr_by_affine_rule: bool


METRICS = (
    MetricSpec(
        slug="uncheatable_bpb",
        metric_key="eval/uncheatable_eval/bpb",
        higher_is_better=False,
    ),
    MetricSpec(
        slug="mmlu_sl_choice_prob_norm",
        metric_key="lm_eval/mmlu_sl_verb_5shot/choice_prob_norm",
        higher_is_better=True,
    ),
    MetricSpec(
        slug="mmlu_sl_choice_logprob_norm",
        metric_key="lm_eval/mmlu_sl_verb_5shot/choice_logprob_norm",
        higher_is_better=True,
    ),
    MetricSpec(
        slug="agentic_success_bpb",
        metric_key="eval/agentic_coding/success_macro_bpb",
        higher_is_better=False,
    ),
)


def _rank_normal(values: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(values, method="average")
    quantiles = (ranks - 0.5) / float(len(values))
    return stats.norm.ppf(np.clip(quantiles, 1e-6, 1.0 - 1e-6))


def _rank_normal_against_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    sorted_reference = np.sort(np.asarray(reference, dtype=float))
    positions = np.searchsorted(sorted_reference, np.asarray(values, dtype=float), side="left")
    quantiles = (positions + 0.5) / float(len(sorted_reference))
    return stats.norm.ppf(np.clip(quantiles, 1e-6, 1.0 - 1e-6))


TRANSFORMS = (
    TransformSpec(
        slug="canonical",
        description="canonical lower-is-better model target",
        transform=lambda y: y,
        inverse=lambda z: z,
        preserves_order=True,
        preserves_snr_by_affine_rule=True,
    ),
    TransformSpec(
        slug="affine_large",
        description="100 * target + 7",
        transform=lambda y: 100.0 * y + 7.0,
        inverse=lambda z: (z - 7.0) / 100.0,
        preserves_order=True,
        preserves_snr_by_affine_rule=True,
    ),
    TransformSpec(
        slug="affine_small",
        description="0.01 * target - 3",
        transform=lambda y: 0.01 * y - 3.0,
        inverse=lambda z: (z + 3.0) / 0.01,
        preserves_order=True,
        preserves_snr_by_affine_rule=True,
    ),
    TransformSpec(
        slug="wrong_sign_negation",
        description="-target, intentionally reversing the GRP monotonic direction",
        transform=lambda y: -y,
        inverse=lambda z: -z,
        preserves_order=False,
        preserves_snr_by_affine_rule=True,
    ),
    TransformSpec(
        slug="rank_normal",
        description="rank-normal transform; preserves target order but changes spacings",
        transform=_rank_normal,
        inverse=None,
        preserves_order=True,
        preserves_snr_by_affine_rule=False,
    ),
)


def _canonical_target(values: np.ndarray, metric: MetricSpec) -> np.ndarray:
    return -values if metric.higher_is_better else values


def _load_signal_frame() -> pd.DataFrame:
    frame = pd.read_csv(RAW_MATRIX)
    frame = frame.loc[
        (frame["row_kind"] == "signal") & (frame["status"] == "completed") & frame["is_qsplit240_core"]
    ].copy()
    if len(frame) != 240:
        raise ValueError(f"Expected 240 qsplit-core signal rows, found {len(frame)}")
    return frame.reset_index(drop=True)


def _load_noise_frame() -> pd.DataFrame:
    frame = pd.read_csv(NOISE_MATRIX)
    frame = frame.loc[frame["row_kind"] == "noise_variable_subset"].copy()
    if len(frame) != 10:
        raise ValueError(f"Expected 10 variable-subset noise rows, found {len(frame)}")
    return frame.reset_index(drop=True)


def _packet_from_target(frame: pd.DataFrame, values: np.ndarray, name: str):
    fit_frame = frame.copy()
    fit_frame[MODEL_TARGET_COLUMN] = np.asarray(values, dtype=float)
    loop = build_two_phase_many_loop_config(objective_metric=MODEL_TARGET_COLUMN, name=name)
    dataset_spec = build_dataset_spec_from_frame(
        fit_frame,
        objective_metric=MODEL_TARGET_COLUMN,
        name=name,
        loop=loop,
    )
    base = PacketData(
        frame=fit_frame.reset_index(drop=True),
        name_col="run_name",
        y=dataset_spec.y,
        w=dataset_spec.weights,
        m=dataset_spec.M,
        c0=np.asarray(dataset_spec.epoch_multipliers[0], dtype=float),
        c1=np.asarray(dataset_spec.epoch_multipliers[1], dtype=float),
        domain_names=list(dataset_spec.domain_names),
    )
    return _generic_family_packet_from_base(base, DEFAULT_FAMILY_SCHEME)


def _oof_predictions(packet, params: dict[str, float]) -> np.ndarray:
    from sklearn.model_selection import KFold

    y = packet.base.y
    oof = np.zeros_like(y)
    splitter = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    for train_idx, test_idx in splitter.split(packet.base.w):
        model = build_penalty_calibration_surrogate(packet, params=params, variant_name=VARIANT_NAME).fit(
            packet.base.w[train_idx],
            y[train_idx],
        )
        oof[test_idx] = model.predict(packet.base.w[test_idx])
    return oof


def _choose_coarse_params(packet, start_bank: tuple[dict[str, float], ...]) -> tuple[dict[str, float], dict[str, float]]:
    rows = []
    for start_id, params in enumerate(start_bank):
        rows.append({"start_id": int(start_id), **params, **_oof_target_metrics(packet, params)})
    coarse = pd.DataFrame.from_records(rows).sort_values(
        ["objective", "target_cv_rmse", "target_cv_foldmean_regret_at_1"],
        ascending=[True, True, True],
    )
    row = coarse.iloc[0].to_dict()
    params = {key: float(row[key]) for key in _no_l2_param_keys()}
    params["reg"] = REG_FIXED
    return params, {key: float(row[key]) for key in ("objective", "target_cv_rmse", "target_cv_spearman")}


def _snr(signal_values: np.ndarray, noise_values: np.ndarray) -> float:
    signal_std = float(np.std(signal_values, ddof=1))
    noise_std = float(np.std(noise_values, ddof=1))
    if noise_std <= 0.0:
        return float("inf")
    return float((signal_std**2 - noise_std**2) / noise_std**2)


def _apply_transform_pair(
    transform: TransformSpec, signal_values: np.ndarray, noise_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if transform.slug == "rank_normal":
        return _rank_normal(signal_values), _rank_normal_against_reference(noise_values, signal_values)
    return transform.transform(signal_values), transform.transform(noise_values)


def _summarize_metric(
    signal_frame: pd.DataFrame,
    noise_frame: pd.DataFrame,
    metric: MetricSpec,
    start_bank: tuple[dict[str, float], ...],
) -> list[dict[str, object]]:
    if metric.metric_key not in signal_frame or metric.metric_key not in noise_frame:
        raise ValueError(f"Metric {metric.metric_key} missing from signal or noise matrix")
    raw_signal = signal_frame[metric.metric_key].to_numpy(dtype=float)
    raw_noise = noise_frame[metric.metric_key].to_numpy(dtype=float)
    if np.isnan(raw_signal).any() or np.isnan(raw_noise).any():
        raise ValueError(f"Metric {metric.metric_key} has missing values")

    canonical_signal = _canonical_target(raw_signal, metric)
    canonical_noise = _canonical_target(raw_noise, metric)
    canonical_packet = _packet_from_target(
        signal_frame,
        canonical_signal,
        name=f"grp_transform_audit_{metric.slug}_canonical",
    )
    canonical_params, canonical_coarse = _choose_coarse_params(canonical_packet, start_bank)
    canonical_pred = _oof_predictions(canonical_packet, canonical_params)
    canonical_best = str(signal_frame.iloc[int(np.argmin(canonical_pred))]["run_name"])

    rows: list[dict[str, object]] = []
    for transform in TRANSFORMS:
        transformed_signal, transformed_noise = _apply_transform_pair(transform, canonical_signal, canonical_noise)
        packet = _packet_from_target(
            signal_frame,
            transformed_signal,
            name=f"grp_transform_audit_{metric.slug}_{transform.slug}",
        )
        selected_params, selected_coarse = _choose_coarse_params(packet, start_bank)
        selected_pred = _oof_predictions(packet, selected_params)
        fixed_pred = _oof_predictions(packet, canonical_params)
        fixed_pred_canonical = transform.inverse(fixed_pred) if transform.inverse is not None else fixed_pred
        selected_pred_canonical = transform.inverse(selected_pred) if transform.inverse is not None else selected_pred
        rows.append(
            {
                "metric_slug": metric.slug,
                "metric_key": metric.metric_key,
                "higher_is_better": metric.higher_is_better,
                "transform_slug": transform.slug,
                "transform_description": transform.description,
                "preserves_order": transform.preserves_order,
                "preserves_snr_by_affine_rule": transform.preserves_snr_by_affine_rule,
                "canonical_snr": _snr(canonical_signal, canonical_noise),
                "transformed_snr": _snr(transformed_signal, transformed_noise),
                "snr_ratio": _snr(transformed_signal, transformed_noise) / _snr(canonical_signal, canonical_noise),
                "canonical_start_objective": canonical_coarse["objective"],
                "selected_start_objective": selected_coarse["objective"],
                "canonical_start_spearman": canonical_coarse["target_cv_spearman"],
                "selected_start_spearman": selected_coarse["target_cv_spearman"],
                "fixed_pred_spearman_vs_actual_canonical": float(
                    stats.spearmanr(canonical_signal, fixed_pred_canonical).statistic
                ),
                "selected_pred_spearman_vs_actual_canonical": float(
                    stats.spearmanr(canonical_signal, selected_pred_canonical).statistic
                ),
                "fixed_pred_spearman_vs_canonical_pred": float(
                    stats.spearmanr(canonical_pred, fixed_pred_canonical).statistic
                ),
                "selected_pred_spearman_vs_canonical_pred": float(
                    stats.spearmanr(canonical_pred, selected_pred_canonical).statistic
                ),
                "canonical_predicted_best_run": canonical_best,
                "fixed_predicted_best_run": str(signal_frame.iloc[int(np.argmin(fixed_pred))]["run_name"]),
                "selected_predicted_best_run": str(signal_frame.iloc[int(np.argmin(selected_pred))]["run_name"]),
                "selected_same_params_as_canonical": all(
                    abs(float(selected_params[key]) - float(canonical_params[key])) < 1e-12
                    for key in _no_l2_param_keys()
                ),
            }
        )
    return rows


def _write_report(summary: pd.DataFrame) -> None:
    affine = summary.loc[summary["transform_slug"].isin(["canonical", "affine_large", "affine_small"])]
    non_affine = summary.loc[~summary["transform_slug"].isin(["canonical", "affine_large", "affine_small"])]
    lines = [
        "# GRP Target Transform Robustness Audit",
        "",
        "Dataset: 300M/6B qsplit-core signal rows (`n=240`) with variable-subset run_00097 noise rows (`n=10`).",
        "",
        "GRP no-L2 uses a non-negative least-squares linear head and `reg=0`. Therefore positive affine",
        "target transforms should be exactly safe: the target can be shifted or rescaled by a positive",
        "constant without changing prediction ranks or nonlinear-parameter selection. Negation is not safe",
        "unless it is the deliberate conversion of a higher-is-better metric into the canonical lower-is-better",
        "model target.",
        "",
        "## Positive Affine Checks",
        "",
        tabulate(
            affine[
                [
                    "metric_slug",
                    "transform_slug",
                    "snr_ratio",
                    "fixed_pred_spearman_vs_canonical_pred",
                    "selected_pred_spearman_vs_canonical_pred",
                    "selected_same_params_as_canonical",
                ]
            ],
            headers="keys",
            tablefmt="github",
            floatfmt=".6f",
            showindex=False,
        ),
        "",
        "## Non-Affine / Wrong-Sign Diagnostics",
        "",
        tabulate(
            non_affine[
                [
                    "metric_slug",
                    "transform_slug",
                    "snr_ratio",
                    "fixed_pred_spearman_vs_canonical_pred",
                    "selected_pred_spearman_vs_canonical_pred",
                    "selected_pred_spearman_vs_actual_canonical",
                    "canonical_predicted_best_run",
                    "selected_predicted_best_run",
                ]
            ],
            headers="keys",
            tablefmt="github",
            floatfmt=".6f",
            showindex=False,
        ),
        "",
        "## Interpretation",
        "",
        "- Positive affine scaling, including z-scoring by a positive standard deviation, is safe for GRP no-L2.",
        "- Naive negation keeps variance SNR fixed, but it reverses the monotonic direction and breaks the "
        "NNLS head's sign prior.",
        "- Rank-normal and other monotone nonlinear transforms preserve target order but change spacings; "
        "GRP is not rank-only, so prediction rankings and selected optima can change.",
        "- SNR invariance alone is not sufficient. The target orientation and spacing relative to the "
        "GRP inductive bias matter.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signal_frame = _load_signal_frame()
    noise_frame = _load_noise_frame()
    start_bank = _expanded_start_bank(random_starts=0)
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        rows.extend(_summarize_metric(signal_frame, noise_frame, metric, start_bank))
    summary = pd.DataFrame.from_records(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    _write_report(summary)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "signal_rows": len(signal_frame),
                "noise_rows": len(noise_frame),
                "metrics": [metric.metric_key for metric in METRICS],
                "transforms": [transform.slug for transform in TRANSFORMS],
                "summary_csv": str(SUMMARY_CSV),
                "report_md": str(REPORT_MD),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
