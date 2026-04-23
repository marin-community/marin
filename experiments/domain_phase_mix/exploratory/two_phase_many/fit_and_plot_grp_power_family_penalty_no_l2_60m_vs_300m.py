# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Fit no-L2 GRP on 300M Chinchilla data and compare it against the 60M fit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_grp_power_family_penalty_no_l2_retune import (
    CV_SEED,
    _parameter_counts,
    _pack_no_l2_params,
    _start_bank,
    _unpack_no_l2_params,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.build_metric_registry import (
    METRICS_WIDE_CSV,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (
    plot_grp_vs_proportional as reference_plot,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GenericFamilyPacket,
    family_shares,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    CALIBRATION_CV_WEIGHT,
    CALIBRATION_FOLDMEAN_WEIGHT,
    CALIBRATION_TAIL_WEIGHT,
    LOWER_TAIL_FRAC,
    build_penalty_calibration_surrogate,
    compute_penalty_calibration_metrics,
    optimize_penalty_calibration_model,
    penalty_calibration_oof_metrics,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    PacketData,
    load_two_phase_many_packet,
)
from experiments.domain_phase_mix.static_batch_selection import build_dataset_spec_from_frame
from experiments.domain_phase_mix.two_phase_many_genericfamily_penalty_raw_optima_baselines import (
    genericfamily_penalty_raw_optimum_summary,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "grp_power_family_penalty_no_l2_60m_vs_300m_fit"
PLOT_PNG = SCRIPT_DIR / f"{OUTPUT_STEM}_weights.png"
WEIGHTS_CSV = SCRIPT_DIR / f"{OUTPUT_STEM}_weights.csv"
SUMMARY_CSV = SCRIPT_DIR / f"{OUTPUT_STEM}_summary.csv"
SUMMARY_JSON = SCRIPT_DIR / f"{OUTPUT_STEM}_summary.json"
REPORT_MD = SCRIPT_DIR / f"{OUTPUT_STEM}.md"

OBJECTIVE_METRIC = MANY_DOMAIN_TARGET
SIXTY_LABEL = "60M-fit no-$L_2$ GRP"
THREE_HUNDRED_LABEL = "300M-fit no-$L_2$ GRP"
RUN_SET_60M = "fit_swarm_60m_default"
RUN_SET_300M = "swarm_like_300m"
THREE_HUNDRED_OLMIX_RUN_NAME = "baseline_olmix_loglinear_uncheatable_bpb"
SPEARMAN_SPLITS = 5
START_TOP_K = 3
OPT_METHOD = "Powell"


def _rdylgn_colors() -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    cmap = plt.get_cmap("RdYlGn_r")
    return cmap(0.15), cmap(0.85)


def _weights_array(
    phase_weights: dict[str, dict[str, float]],
    domain_names: list[str],
) -> np.ndarray:
    return np.asarray(
        [
            [float(phase_weights["phase_0"][domain_name]) for domain_name in domain_names],
            [float(phase_weights["phase_1"][domain_name]) for domain_name in domain_names],
        ],
        dtype=float,
    )


def _phase_tv(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(a - b).sum())


def _group_tv(weights: np.ndarray, domain_names: list[str]) -> dict[str, float]:
    packet = _generic_family_packet_from_base(
        PacketData(
            frame=pd.DataFrame(),
            name_col="run_name",
            y=np.zeros(1, dtype=float),
            w=np.zeros((1, 2, len(domain_names)), dtype=float),
            m=len(domain_names),
            c0=np.ones(len(domain_names), dtype=float),
            c1=np.ones(len(domain_names), dtype=float),
            domain_names=domain_names,
        )
    )
    out: dict[str, float] = {}
    for phase_idx, phase_name in enumerate(("phase_0", "phase_1")):
        broad = float(weights[phase_idx, packet.family_map["broad_text"]].sum())
        tech = float(weights[phase_idx, packet.family_map["tech_code"]].sum())
        reasoning = float(weights[phase_idx, packet.family_map["reasoning"]].sum())
        out[f"{phase_name}_broad_text"] = broad
        out[f"{phase_name}_tech_code"] = tech
        out[f"{phase_name}_reasoning"] = reasoning
    return out


def _metric_frame() -> pd.DataFrame:
    return pd.read_csv(METRICS_WIDE_CSV, low_memory=False)


def _build_fit_frame(frame: pd.DataFrame, *, scale: str, run_set: str) -> pd.DataFrame:
    reference = load_two_phase_many_packet(target=OBJECTIVE_METRIC)
    canonical_weight_columns = [
        f"{phase_name}_{domain_name}" for phase_name in ("phase_0", "phase_1") for domain_name in reference.domain_names
    ]
    weight_columns = [column for column in canonical_weight_columns if column in frame.columns]
    id_columns = [
        column
        for column in (
            "run_id",
            "run_name",
            "scale",
            "cohort",
            "source_experiment",
            "checkpoint_root",
            "status",
            "is_qsplit240_core",
            "is_baseline_olmix",
            "is_baseline_stratified",
            "is_fit_swarm_60m_default",
        )
        if column in frame.columns
    ]
    subset = frame.loc[
        frame["scale"].eq(scale) & frame["cohort"].eq("signal") & frame[OBJECTIVE_METRIC].notna(),
        id_columns + weight_columns + [OBJECTIVE_METRIC],
    ].copy()
    if run_set == RUN_SET_60M:
        subset = subset.loc[subset["is_fit_swarm_60m_default"].fillna(False)].copy()
    elif run_set == RUN_SET_300M:
        mask = (
            subset["is_qsplit240_core"].fillna(False)
            | subset["is_baseline_olmix"].fillna(False)
            | subset["is_baseline_stratified"].fillna(False)
            | subset["run_name"].eq(THREE_HUNDRED_OLMIX_RUN_NAME)
        )
        subset = subset.loc[mask].copy()
    else:
        raise ValueError(f"Unknown run_set={run_set!r}")

    subset = subset.rename(columns={OBJECTIVE_METRIC: "objective_metric"}).reset_index(drop=True)
    subset[weight_columns] = subset[weight_columns].fillna(0.0)
    if subset["run_name"].duplicated().any():
        raise ValueError(f"Duplicate run_name rows in {scale}/{run_set} fit frame")
    return subset


def _generic_family_packet_from_base(base: PacketData) -> GenericFamilyPacket:
    pairs: list[tuple[int, int]] = []
    pair_topics: list[str] = []
    paired: set[int] = set()

    for idx, domain_name in enumerate(base.domain_names):
        if idx in paired:
            continue
        if domain_name.startswith("dolma3_cc/") and domain_name.endswith("_high"):
            low_name = domain_name[:-5] + "_low"
            if low_name in base.domain_names:
                low_idx = base.domain_names.index(low_name)
                pairs.append((idx, low_idx))
                pair_topics.append(domain_name[len("dolma3_cc/") : -5])
                paired.add(idx)
                paired.add(low_idx)

    singletons = [idx for idx in range(base.m) if idx not in paired]
    family_map = {"broad_text": [], "tech_code": [], "reasoning": []}
    for idx, domain_name in enumerate(base.domain_names):
        is_broad = (
            domain_name.startswith("dolma3_cc/")
            or domain_name
            in {
                "dolma3_wikipedia",
                "dolmino_common_crawl_hq",
                "dolmino_olmocr_pdfs_hq",
                "dolmino_stem_heavy_crawl",
            }
            or domain_name.endswith("synth_qa")
        )
        is_tech = any(token in domain_name for token in ("stack_edu", "synth_code", "synth_math")) or domain_name in {
            "dolma3_arxiv",
            "dolma3_finemath_3plus",
        }
        is_reasoning = domain_name in {"dolmino_synth_instruction", "dolmino_synth_thinking"}
        if is_broad:
            family_map["broad_text"].append(idx)
        if is_tech:
            family_map["tech_code"].append(idx)
        if is_reasoning:
            family_map["reasoning"].append(idx)

    return GenericFamilyPacket(
        base=base,
        pairs=pairs,
        pair_topics=pair_topics,
        singletons=singletons,
        family_map=family_map,
    )


def _packet_from_frame(frame: pd.DataFrame, *, name: str) -> GenericFamilyPacket:
    reference = load_two_phase_many_packet(target=OBJECTIVE_METRIC)
    spec = build_dataset_spec_from_frame(
        frame,
        objective_metric="objective_metric",
        name=name,
        loop=None,
    )
    base = PacketData(
        frame=frame.reset_index(drop=True),
        name_col="run_name",
        y=spec.y,
        w=spec.weights,
        m=spec.M,
        c0=np.asarray(reference.c0, dtype=float),
        c1=np.asarray(reference.c1, dtype=float),
        domain_names=list(spec.domain_names),
    )
    if base.domain_names != reference.domain_names:
        raise ValueError("Domain ordering mismatch against canonical two-phase packet")
    return _generic_family_packet_from_base(base)


def _oof_spearman(packet: GenericFamilyPacket, params: dict[str, float]) -> float:
    y = packet.base.y
    kf = KFold(n_splits=SPEARMAN_SPLITS, shuffle=True, random_state=CV_SEED)
    oof = np.zeros_like(y)
    for train_idx, test_idx in kf.split(packet.base.w):
        model = build_penalty_calibration_surrogate(packet, params=params, variant_name="power_family_penalty").fit(
            packet.base.w[train_idx],
            y[train_idx],
        )
        oof[test_idx] = model.predict(packet.base.w[test_idx])
    return float(spearmanr(y, oof).statistic)


def _fast_oof_metrics(packet: GenericFamilyPacket, params: dict[str, float]) -> dict[str, float]:
    y = packet.base.y
    kf = KFold(n_splits=SPEARMAN_SPLITS, shuffle=True, random_state=CV_SEED)
    oof = np.zeros_like(y)
    fold_regrets: list[float] = []
    for train_idx, test_idx in kf.split(packet.base.w):
        model = build_penalty_calibration_surrogate(packet, params=params, variant_name="power_family_penalty").fit(
            packet.base.w[train_idx],
            y[train_idx],
        )
        pred = model.predict(packet.base.w[test_idx])
        oof[test_idx] = pred
        fold_regrets.append(float(y[test_idx][int(np.argmin(pred))] - np.min(y[test_idx])))

    residuals = oof - y
    cv_rmse = float(np.sqrt(np.mean(residuals**2)))
    tail_count = max(5, int(np.ceil(float(LOWER_TAIL_FRAC) * float(len(y)))))
    tail_idx = np.argsort(oof)[:tail_count]
    lower_tail_optimism = float(np.mean(np.maximum(y[tail_idx] - oof[tail_idx], 0.0)))
    mean_regret = float(np.mean(fold_regrets))
    objective = (
        CALIBRATION_CV_WEIGHT * cv_rmse
        + CALIBRATION_FOLDMEAN_WEIGHT * mean_regret
        + CALIBRATION_TAIL_WEIGHT * lower_tail_optimism
    )
    return {
        "cv_rmse": cv_rmse,
        "cv_regret_at_1": float(y[int(np.argmin(oof))] - np.min(y)),
        "cv_foldmean_regret_at_1": mean_regret,
        "lower_tail_optimism": lower_tail_optimism,
        "objective": float(objective),
    }


def _fit_no_l2(packet: GenericFamilyPacket) -> dict[str, object]:
    start_bank = _start_bank()
    coarse_rows = [
        {
            "start_id": int(start_id),
            **params,
            **_fast_oof_metrics(packet, params),
        }
        for start_id, params in enumerate(start_bank)
    ]
    coarse_frame = pd.DataFrame.from_records(coarse_rows).sort_values(
        ["objective", "cv_rmse", "cv_foldmean_regret_at_1"],
        ascending=[True, True, True],
    )

    best_result: dict[str, object] | None = None
    for start_id in coarse_frame["start_id"].head(START_TOP_K).tolist():
        start = _pack_no_l2_params(start_bank[int(start_id)])
        cache: dict[tuple[float, ...], float] = {}

        def objective(z: np.ndarray, cache: dict[tuple[float, ...], float] = cache) -> float:
            key = tuple(np.round(np.asarray(z, dtype=float), 8))
            if key not in cache:
                params = _unpack_no_l2_params(z)
                metrics = _fast_oof_metrics(packet, params)
                cache[key] = float(metrics["objective"])
            return cache[key]

        result = minimize(
            objective,
            start,
            method=OPT_METHOD,
            options={"maxiter": 30, "xtol": 1e-4, "ftol": 1e-6},
        )
        params = _unpack_no_l2_params(np.asarray(result.x, dtype=float))
        tuning_metrics = penalty_calibration_oof_metrics(
            packet, params, variant_name="power_family_penalty", seed=CV_SEED
        )
        full_model = build_penalty_calibration_surrogate(
            packet,
            params=params,
            variant_name="power_family_penalty",
        ).fit(packet.base.w, packet.base.y)
        full_metrics = compute_penalty_calibration_metrics(packet, full_model, seed=CV_SEED)
        train_pred = full_model.predict(packet.base.w)
        opt_result, phase0, phase1 = optimize_penalty_calibration_model(packet, full_model, seed=CV_SEED)
        weights = np.stack([phase0, phase1], axis=0)
        record = {
            "success": bool(result.success),
            "message": str(result.message),
            "params": params,
            "tuning_metrics": tuning_metrics,
            "full_metrics": full_metrics,
            "train_spearman": float(spearmanr(packet.base.y, train_pred).statistic),
            "predicted_optimum_value": float(opt_result.fun),
            "weights": weights,
            "family_shares": family_shares(packet, weights),
            "counts": _parameter_counts(packet),
            "oof_spearman": _oof_spearman(packet, params),
        }
        if best_result is None or float(tuning_metrics["objective"]) < float(best_result["tuning_metrics"]["objective"]):
            best_result = record

    if best_result is None:
        raise RuntimeError("No 300M no-L2 fit result")
    return best_result


def _sixty_fit_record(packet_60m: GenericFamilyPacket) -> dict[str, object]:
    summary = genericfamily_penalty_raw_optimum_summary("power_family_penalty_no_l2")
    params = dict(summary.tuned_params)
    model = build_penalty_calibration_surrogate(packet_60m, params=params, variant_name="power_family_penalty").fit(
        packet_60m.base.w,
        packet_60m.base.y,
    )
    full_metrics = compute_penalty_calibration_metrics(packet_60m, model, seed=CV_SEED)
    train_pred = model.predict(packet_60m.base.w)
    return {
        "success": True,
        "message": summary.optimizer_message,
        "params": params,
        "tuning_metrics": {
            "objective": summary.tuning_objective,
            "cv_rmse": summary.tuning_cv_rmse,
            "cv_foldmean_regret_at_1": summary.tuning_cv_foldmean_regret_at_1,
            "lower_tail_optimism": summary.tuning_lower_tail_optimism,
            "cv_depopt_best8": summary.tuning_cv_depopt_best8,
            "cv_rawopt_nearest_tv": summary.tuning_cv_rawopt_nearest_tv,
        },
        "full_metrics": full_metrics,
        "train_spearman": float(spearmanr(packet_60m.base.y, train_pred).statistic),
        "predicted_optimum_value": summary.raw_predicted_optimum_value,
        "weights": _weights_array(summary.phase_weights, packet_60m.base.domain_names),
        "family_shares": summary.family_shares,
        "counts": _parameter_counts(packet_60m),
        "oof_spearman": _oof_spearman(packet_60m, params),
    }


def _plot(
    weights_60m: np.ndarray,
    weights_300m: np.ndarray,
    domain_names: list[str],
    c0: np.ndarray,
    c1: np.ndarray,
) -> None:
    color_60m, color_300m = _rdylgn_colors()
    schedules = [
        (SIXTY_LABEL, weights_60m, color_60m),
        (THREE_HUNDRED_LABEL, weights_300m, color_300m),
    ]
    canonical_non_cc_indices, canonical_cc_indices = reference_plot._grp_domain_order(domain_names, weights_60m)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(26, 22),
        gridspec_kw={"width_ratios": [1.0, 1.62], "hspace": 0.22, "wspace": 0.31},
        facecolor="white",
    )
    reference_plot._plot_non_cc_block(
        ax=axes[0, 0],
        indices=canonical_non_cc_indices,
        labels=[reference_plot._display_non_cc_label(domain_names[idx]) for idx in canonical_non_cc_indices],
        schedules=schedules,
        phase_idx=0,
        multipliers=c0,
        title="Phase 0: Non-CC Domains",
        show_legend=True,
    )
    reference_plot._plot_cc_block(
        ax=axes[0, 1],
        domain_names=domain_names,
        indices=canonical_cc_indices,
        schedules=schedules,
        phase_idx=0,
        multipliers=c0,
        title="Phase 0: CC Domains",
    )
    reference_plot._plot_non_cc_block(
        ax=axes[1, 0],
        indices=canonical_non_cc_indices,
        labels=[reference_plot._display_non_cc_label(domain_names[idx]) for idx in canonical_non_cc_indices],
        schedules=schedules,
        phase_idx=1,
        multipliers=c1,
        title="Phase 1: Non-CC Domains",
        show_legend=False,
    )
    reference_plot._plot_cc_block(
        ax=axes[1, 1],
        domain_names=domain_names,
        indices=canonical_cc_indices,
        schedules=schedules,
        phase_idx=1,
        multipliers=c1,
        title="Phase 1: CC Domains",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("No-$L_2$ GRP optimum: 60M fit vs 300M fit", fontsize=34, y=0.996, fontweight="bold")
    fig.text(
        0.5,
        0.952,
        "Same GRP surrogate family, refit on the 60M swarm panel versus the 300M Chinchilla panel",
        ha="center",
        va="center",
        fontsize=20,
        color=reference_plot.TEXT_MUTED_COLOR,
    )
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=18.5, bbox_to_anchor=(0.5, 0.928))
    fig.text(
        0.5,
        0.072,
        (
            f"Phase TV distance: phase 0 = {_phase_tv(weights_60m[0], weights_300m[0]):.3f}, "
            f"phase 1 = {_phase_tv(weights_60m[1], weights_300m[1]):.3f}"
        ),
        ha="center",
        va="center",
        fontsize=17.5,
        color="#0f172a",
        bbox={
            "boxstyle": "round,pad=0.62,rounding_size=0.18",
            "facecolor": "#f8fafc",
            "edgecolor": "#cbd5e1",
            "alpha": 0.97,
        },
    )
    fig.text(
        0.5,
        0.026,
        "Bar-end labels show effective epochs for that domain in that phase. Values below 0.01 are displayed as 0.",
        ha="center",
        va="center",
        fontsize=15,
        color=reference_plot.TEXT_MUTED_COLOR,
    )
    fig.subplots_adjust(top=0.905, left=0.14, right=0.985, bottom=0.13, hspace=0.24, wspace=0.31)
    fig.savefig(PLOT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _weight_table(
    domain_names: list[str],
    weights_60m: np.ndarray,
    weights_300m: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain_name, w60_p0, w60_p1, w300_p0, w300_p1, m0, m1 in zip(
        domain_names,
        weights_60m[0],
        weights_60m[1],
        weights_300m[0],
        weights_300m[1],
        c0,
        c1,
        strict=True,
    ):
        rows.append(
            {
                "domain": domain_name,
                "fit60_phase0_weight": float(w60_p0),
                "fit60_phase0_epochs": float(w60_p0 * m0),
                "fit60_phase1_weight": float(w60_p1),
                "fit60_phase1_epochs": float(w60_p1 * m1),
                "fit300_phase0_weight": float(w300_p0),
                "fit300_phase0_epochs": float(w300_p0 * m0),
                "fit300_phase1_weight": float(w300_p1),
                "fit300_phase1_epochs": float(w300_p1 * m1),
            }
        )
    return pd.DataFrame(rows)


def _summary_row(label: str, scale: str, run_set: str, record: dict[str, object], n_rows: int) -> dict[str, object]:
    params = record["params"]
    tuning = record["tuning_metrics"]
    full = record["full_metrics"]
    counts = record["counts"]
    weights = np.asarray(record["weights"], dtype=float)
    return {
        "label": label,
        "fit_scale": scale,
        "fit_run_set": run_set,
        "fit_row_count": int(n_rows),
        "optimizer_success": bool(record["success"]),
        "optimizer_message": str(record["message"]),
        "predicted_optimum_value": float(record["predicted_optimum_value"]),
        "train_rmse": float(full["train_rmse"]),
        "train_spearman": float(record["train_spearman"]),
        "cv_rmse": float(tuning["cv_rmse"]),
        "cv_foldmean_regret_at_1": float(tuning["cv_foldmean_regret_at_1"]),
        "cv_regret_at_1": float(tuning.get("cv_regret_at_1", np.nan)),
        "lower_tail_optimism": float(tuning["lower_tail_optimism"]),
        "cv_depopt_best8": float(tuning["cv_depopt_best8"]),
        "cv_rawopt_nearest_tv": float(tuning["cv_rawopt_nearest_tv"]),
        "oof_spearman": float(record["oof_spearman"]),
        "nearest_observed_run_name": str(full["raw_nearest_observed_run_name"]),
        "nearest_observed_value": float(full["raw_nearest_observed_value"]),
        "nearest_observed_tv": float(full["raw_nearest_observed_tv"]),
        "phase0_max_weight": float(np.max(weights[0])),
        "phase1_max_weight": float(np.max(weights[1])),
        "phase0_broad_text": float(record["family_shares"]["phase0_broad_text"]),
        "phase0_tech_code": float(record["family_shares"]["phase0_tech_code"]),
        "phase0_reasoning": float(record["family_shares"]["phase0_reasoning"]),
        "phase1_broad_text": float(record["family_shares"]["phase1_broad_text"]),
        "phase1_tech_code": float(record["family_shares"]["phase1_tech_code"]),
        "phase1_reasoning": float(record["family_shares"]["phase1_reasoning"]),
        "total_param_count": int(counts["total_param_count"]),
        **{f"param_{key}": float(value) for key, value in params.items()},
    }


def _report(frame: pd.DataFrame) -> str:
    columns = [
        "label",
        "fit_scale",
        "fit_row_count",
        "predicted_optimum_value",
        "train_rmse",
        "cv_rmse",
        "train_spearman",
        "oof_spearman",
        "cv_foldmean_regret_at_1",
        "cv_rawopt_nearest_tv",
        "phase0_broad_text",
        "phase0_tech_code",
        "phase0_reasoning",
        "phase1_broad_text",
        "phase1_tech_code",
        "phase1_reasoning",
    ]
    return frame[columns].to_markdown(index=False, floatfmt=".6f") + "\n"


def main() -> None:
    metric_frame = _metric_frame()
    fit60_frame = _build_fit_frame(metric_frame, scale="60m_1p2b", run_set=RUN_SET_60M)
    fit300_frame = _build_fit_frame(metric_frame, scale="300m_6b", run_set=RUN_SET_300M)

    packet60 = _packet_from_frame(fit60_frame, name="grp_no_l2_fit_60m")
    packet300 = _packet_from_frame(fit300_frame, name="grp_no_l2_fit_300m")

    fit60 = _sixty_fit_record(packet60)
    fit300 = _fit_no_l2(packet300)

    summary = pd.DataFrame.from_records(
        [
            _summary_row(SIXTY_LABEL, "60m_1p2b", RUN_SET_60M, fit60, len(fit60_frame)),
            _summary_row(THREE_HUNDRED_LABEL, "300m_6b", RUN_SET_300M, fit300, len(fit300_frame)),
        ]
    )
    summary.to_csv(SUMMARY_CSV, index=False)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "rows": summary.to_dict(orient="records"),
                "plot_png": str(PLOT_PNG),
                "weights_csv": str(WEIGHTS_CSV),
                "phase_tv": {
                    "phase0": _phase_tv(np.asarray(fit60["weights"])[0], np.asarray(fit300["weights"])[0]),
                    "phase1": _phase_tv(np.asarray(fit60["weights"])[1], np.asarray(fit300["weights"])[1]),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    REPORT_MD.write_text(_report(summary))

    reference = load_two_phase_many_packet(target=OBJECTIVE_METRIC)
    weights_60m = np.asarray(fit60["weights"], dtype=float)
    weights_300m = np.asarray(fit300["weights"], dtype=float)
    _plot(weights_60m, weights_300m, reference.domain_names, reference.c0, reference.c1)
    _weight_table(reference.domain_names, weights_60m, weights_300m, reference.c0, reference.c1).to_csv(
        WEIGHTS_CSV,
        index=False,
    )

    print(f"Summary: {SUMMARY_CSV}")
    print(f"Plot: {PLOT_PNG}")
    print(f"Weights: {WEIGHTS_CSV}")


if __name__ == "__main__":
    main()
