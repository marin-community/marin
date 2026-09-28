# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot signal-to-noise comparisons across the main swarm objective settings."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

plt.rcParams["text.usetex"] = False

ROOT = Path(__file__).resolve().parents[4]
EXPLORATORY_DIR = ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many"
OUTPUT_CSV = EXPLORATORY_DIR / "choice_logprob_norm_signal_to_noise.csv"
OUTPUT_PNG = EXPLORATORY_DIR / "choice_logprob_norm_signal_to_noise.png"

COMMON_SWARM_PATH = EXPLORATORY_DIR / "two_phase_many.csv"
FIXED_SUBSET_SUMMARY_PATH = EXPLORATORY_DIR / "qsplit240_fixed_subset_seedpanel_n3_mmlu_sl_verb_candidate_summary.csv"
FIXED_SUBSET_STANDARD_PANEL_RESULTS_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_qsplit240_fixed_subset_seedpanel_n3/collect_results-33de09/results.csv"
)
FIXED_SUBSET_SL_VERB_PANEL_RESULTS_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_qsplit240_fixed_subset_seedpanel_n3_mmlu_sl_verb_rerun/collect_results-2b064a/results.csv"
)

QSPLIT240_SL_VERB_RESULTS_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_qsplit240_mmlu_sl_verb_rerun/collect_results-ef2602/results.csv"
)
RUN00097_STANDARD_RESULTS_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/" "ngd3dm2_run00097_seed_study/collect_results-ca72ba/results.csv"
)
RUN00097_SL_VERB_RESULTS_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_run00097_seed_study_mmlu_sl_verb_rerun/collect_results-34269c/results.csv"
)
RUN00097_FIXED_SUBSET_STANDARD_RESULTS_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_run00097_fixed_subset_study/collect_results-287628/results.csv"
)
RUN00097_FIXED_SUBSET_SL_VERB_RESULTS_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_run00097_fixed_subset_study_mmlu_sl_verb_rerun/collect_results-76a3ed/results.csv"
)

STANDARD_BPB_METRIC = "lm_eval/mmlu_5shot/bpb"
STANDARD_METRIC = "lm_eval/mmlu_5shot/choice_logprob_norm"
SL_VERB_BPB_METRIC = "lm_eval/mmlu_sl_verb_5shot/bpb"
SL_VERB_METRIC = "lm_eval/mmlu_sl_verb_5shot/choice_logprob_norm"
UNCHEATABLE_METRIC = "eval/uncheatable_eval/bpb"

COMMON_PANEL_NAMES = {"baseline_proportional", "baseline_unimax"} | {f"run_{run_id:05d}" for run_id in range(2, 240)}

Y_LABELS = [
    "Original swarm\nmmlu_5shot bpb",
    "Original swarm\nmmlu_5shot",
    "Original swarm\nmmlu_sl_verb_5shot bpb",
    "Original swarm\nmmlu_sl_verb_5shot",
    "Fixed subset swarm\nmmlu_5shot bpb",
    "Fixed subset swarm\nmmlu_5shot",
    "Fixed subset swarm\nmmlu_sl_verb_5shot bpb",
    "Fixed subset swarm\nmmlu_sl_verb_5shot",
    "Fixed subset seedpanel\nsl_verb bpb (3-seed mean)",
    "Fixed subset seedpanel\nsl_verb (3-seed mean)",
    "Original swarm\nuncheatable_eval/bpb\n(ordinary seed noise)",
    "Original swarm\nuncheatable_eval/bpb\n(fixed-subset seed noise)",
]


def _read_gcs_csv(uri: str) -> pd.DataFrame:
    data = subprocess.check_output(["gsutil", "cat", uri], text=True)
    return pd.read_csv(io.StringIO(data))


def _load_signal_to_noise_table() -> pd.DataFrame:
    full_swarm = pd.read_csv(COMMON_SWARM_PATH)
    common_swarm = full_swarm.copy()
    sl_verb_swarm = _read_gcs_csv(QSPLIT240_SL_VERB_RESULTS_URI)
    fixed_subset_standard_panel = _read_gcs_csv(FIXED_SUBSET_STANDARD_PANEL_RESULTS_URI)
    fixed_subset_sl_verb_panel = _read_gcs_csv(FIXED_SUBSET_SL_VERB_PANEL_RESULTS_URI)
    run00097_standard = _read_gcs_csv(RUN00097_STANDARD_RESULTS_URI)
    run00097_sl_verb = _read_gcs_csv(RUN00097_SL_VERB_RESULTS_URI)
    run00097_fixed_subset_standard = _read_gcs_csv(RUN00097_FIXED_SUBSET_STANDARD_RESULTS_URI)
    run00097_fixed_subset_sl_verb = _read_gcs_csv(RUN00097_FIXED_SUBSET_SL_VERB_RESULTS_URI)
    fixed_subset_summary = pd.read_csv(FIXED_SUBSET_SUMMARY_PATH)

    common_swarm = common_swarm[common_swarm["run_name"].isin(COMMON_PANEL_NAMES)].copy()
    if len(common_swarm) != 240:
        raise ValueError(f"Expected 240 common swarm rows, got {len(common_swarm)}")

    shared_names = set(common_swarm["run_name"])
    if shared_names != set(sl_verb_swarm["run_name"]):
        raise ValueError("Original swarm and sl_verb rerun candidate sets do not match")
    if shared_names != set(fixed_subset_summary["candidate_run_name"]):
        raise ValueError("Fixed-subset seedpanel candidate set does not match original swarm")
    fixed_subset_standard_panel = fixed_subset_standard_panel[
        fixed_subset_standard_panel["candidate_run_name"].isin(common_swarm["run_name"])
    ].copy()
    fixed_subset_standard_means = (
        fixed_subset_standard_panel.groupby("candidate_run_name", as_index=False)[STANDARD_METRIC]
        .mean()
        .rename(columns={STANDARD_METRIC: "choice_logprob_norm_mean"})
    )
    fixed_subset_standard_bpb_means = (
        fixed_subset_standard_panel.groupby("candidate_run_name", as_index=False)[STANDARD_BPB_METRIC]
        .mean()
        .rename(columns={STANDARD_BPB_METRIC: "bpb_mean"})
    )
    fixed_subset_sl_verb_means = (
        fixed_subset_sl_verb_panel.groupby("candidate_run_name", as_index=False)[SL_VERB_METRIC]
        .mean()
        .rename(columns={SL_VERB_METRIC: "choice_logprob_norm_mean"})
    )
    fixed_subset_sl_verb_bpb_means = (
        fixed_subset_sl_verb_panel.groupby("candidate_run_name", as_index=False)[SL_VERB_BPB_METRIC]
        .mean()
        .rename(columns={SL_VERB_BPB_METRIC: "bpb_mean"})
    )
    fixed_subset_sl_verb_bpb_sem = (
        fixed_subset_sl_verb_panel.groupby("candidate_run_name", as_index=False)[SL_VERB_BPB_METRIC]
        .sem()
        .rename(columns={SL_VERB_BPB_METRIC: "bpb_sem"})
    )
    if len(fixed_subset_standard_means) != 238:
        raise ValueError(f"Expected 238 fixed-subset standard candidate means, got {len(fixed_subset_standard_means)}")
    if len(fixed_subset_standard_bpb_means) != 238:
        raise ValueError(
            f"Expected 238 fixed-subset standard candidate BPB means, got {len(fixed_subset_standard_bpb_means)}"
        )
    if len(fixed_subset_sl_verb_means) != 240:
        raise ValueError(f"Expected 240 fixed-subset sl_verb candidate means, got {len(fixed_subset_sl_verb_means)}")
    if len(fixed_subset_sl_verb_bpb_means) != 240:
        raise ValueError(
            f"Expected 240 fixed-subset sl_verb candidate BPB means, got {len(fixed_subset_sl_verb_bpb_means)}"
        )
    if len(fixed_subset_sl_verb_bpb_sem) != 240:
        raise ValueError(
            f"Expected 240 fixed-subset sl_verb candidate BPB SEMs, got {len(fixed_subset_sl_verb_bpb_sem)}"
        )

    run00097_standard = run00097_standard[run00097_standard["cohort"] == "seed_sweep"].copy()
    run00097_fixed_subset_standard = run00097_fixed_subset_standard[
        run00097_fixed_subset_standard["cohort"] == "seed_sweep"
    ].copy()
    run00097_standard_bpb_noise = run00097_standard[STANDARD_BPB_METRIC].dropna()
    run00097_standard_noise = run00097_standard[STANDARD_METRIC].dropna()
    run00097_sl_verb_bpb_noise = run00097_sl_verb[SL_VERB_BPB_METRIC].dropna()
    run00097_sl_verb_noise = run00097_sl_verb[SL_VERB_METRIC].dropna()
    run00097_fixed_subset_standard_bpb_noise = run00097_fixed_subset_standard[STANDARD_BPB_METRIC].dropna()
    run00097_fixed_subset_standard_noise = run00097_fixed_subset_standard[STANDARD_METRIC].dropna()
    run00097_fixed_subset_sl_verb_bpb_noise = run00097_fixed_subset_sl_verb[SL_VERB_BPB_METRIC].dropna()
    run00097_fixed_subset_sl_verb_noise = run00097_fixed_subset_sl_verb[SL_VERB_METRIC].dropna()

    rows = [
        {
            "setting": "Original Swarm / mmlu_5shot_bpb",
            "short_label": Y_LABELS[0],
            "metric": STANDARD_BPB_METRIC,
            "signal_definition": "std across 240 candidate scores",
            "noise_definition": "std across run_00097 seed-sweep scores",
            "signal_n": len(common_swarm),
            "noise_n": len(run00097_standard_bpb_noise),
            "signal_scale": common_swarm[STANDARD_BPB_METRIC].std(ddof=1),
            "noise_scale": run00097_standard_bpb_noise.std(ddof=1),
            "signal_range": common_swarm[STANDARD_BPB_METRIC].max() - common_swarm[STANDARD_BPB_METRIC].min(),
            "note": "Same original-swarm panel, BPB metric instead of choice_logprob_norm.",
        },
        {
            "setting": "Original Swarm / mmlu_5shot",
            "short_label": Y_LABELS[1],
            "metric": STANDARD_METRIC,
            "signal_definition": "std across 240 candidate scores",
            "noise_definition": "std across run_00097 seed-sweep scores",
            "signal_n": len(common_swarm),
            "noise_n": len(run00097_standard_noise),
            "signal_scale": common_swarm[STANDARD_METRIC].std(ddof=1),
            "noise_scale": run00097_standard_noise.std(ddof=1),
            "signal_range": common_swarm[STANDARD_METRIC].max() - common_swarm[STANDARD_METRIC].min(),
            "note": "Older collector artifact has 7/10 non-null seed-study values.",
        },
        {
            "setting": "Original Swarm / mmlu_sl_verb_5shot_bpb",
            "short_label": Y_LABELS[2],
            "metric": SL_VERB_BPB_METRIC,
            "signal_definition": "std across 240 candidate scores",
            "noise_definition": "std across run_00097 seed-sweep scores",
            "signal_n": len(sl_verb_swarm),
            "noise_n": len(run00097_sl_verb_bpb_noise),
            "signal_scale": sl_verb_swarm[SL_VERB_BPB_METRIC].std(ddof=1),
            "noise_scale": run00097_sl_verb_bpb_noise.std(ddof=1),
            "signal_range": sl_verb_swarm[SL_VERB_BPB_METRIC].max() - sl_verb_swarm[SL_VERB_BPB_METRIC].min(),
            "note": "Uses the full 10-seed run_00097 sl_verb rerun, BPB metric.",
        },
        {
            "setting": "Original Swarm / mmlu_sl_verb_5shot",
            "short_label": Y_LABELS[3],
            "metric": SL_VERB_METRIC,
            "signal_definition": "std across 240 candidate scores",
            "noise_definition": "std across run_00097 seed-sweep scores",
            "signal_n": len(sl_verb_swarm),
            "noise_n": len(run00097_sl_verb_noise),
            "signal_scale": sl_verb_swarm[SL_VERB_METRIC].std(ddof=1),
            "noise_scale": run00097_sl_verb_noise.std(ddof=1),
            "signal_range": sl_verb_swarm[SL_VERB_METRIC].max() - sl_verb_swarm[SL_VERB_METRIC].min(),
            "note": "Uses the full 10-seed run_00097 sl_verb rerun.",
        },
        {
            "setting": "Fixed Subset Swarm / mmlu_5shot_bpb",
            "short_label": Y_LABELS[4],
            "metric": STANDARD_BPB_METRIC,
            "signal_definition": "std across 238 candidate means",
            "noise_definition": "std across fixed-subset run_00097 seed-sweep scores",
            "signal_n": len(fixed_subset_standard_bpb_means),
            "noise_n": len(run00097_fixed_subset_standard_bpb_noise),
            "signal_scale": fixed_subset_standard_bpb_means["bpb_mean"].std(ddof=1),
            "noise_scale": run00097_fixed_subset_standard_bpb_noise.std(ddof=1),
            "signal_range": (
                fixed_subset_standard_bpb_means["bpb_mean"].max() - fixed_subset_standard_bpb_means["bpb_mean"].min()
            ),
            "note": (
                "Fixed-subset standard MMLU BPB on the available 238 sampled runs; "
                "baseline rows are missing from the historical collector."
            ),
        },
        {
            "setting": "Fixed Subset Swarm / mmlu_5shot",
            "short_label": Y_LABELS[5],
            "metric": STANDARD_METRIC,
            "signal_definition": "std across 238 candidate means",
            "noise_definition": "std across fixed-subset run_00097 seed-sweep scores",
            "signal_n": len(fixed_subset_standard_means),
            "noise_n": len(run00097_fixed_subset_standard_noise),
            "signal_scale": fixed_subset_standard_means["choice_logprob_norm_mean"].std(ddof=1),
            "noise_scale": run00097_fixed_subset_standard_noise.std(ddof=1),
            "signal_range": (
                fixed_subset_standard_means["choice_logprob_norm_mean"].max()
                - fixed_subset_standard_means["choice_logprob_norm_mean"].min()
            ),
            "note": (
                "Fixed-subset standard MMLU on the available 238 sampled runs; "
                "baseline rows are missing from the historical collector."
            ),
        },
        {
            "setting": "Fixed Subset Swarm / mmlu_sl_verb_5shot_bpb",
            "short_label": Y_LABELS[6],
            "metric": SL_VERB_BPB_METRIC,
            "signal_definition": "std across 240 candidate means",
            "noise_definition": "std across fixed-subset run_00097 seed-sweep scores",
            "signal_n": len(fixed_subset_sl_verb_bpb_means),
            "noise_n": len(run00097_fixed_subset_sl_verb_bpb_noise),
            "signal_scale": fixed_subset_sl_verb_bpb_means["bpb_mean"].std(ddof=1),
            "noise_scale": run00097_fixed_subset_sl_verb_bpb_noise.std(ddof=1),
            "signal_range": (
                fixed_subset_sl_verb_bpb_means["bpb_mean"].max() - fixed_subset_sl_verb_bpb_means["bpb_mean"].min()
            ),
            "note": "Fixed-subset sl_verb seedpanel means, BPB metric.",
        },
        {
            "setting": "Fixed Subset Swarm / mmlu_sl_verb_5shot",
            "short_label": Y_LABELS[7],
            "metric": SL_VERB_METRIC,
            "signal_definition": "std across 240 candidate means",
            "noise_definition": "std across fixed-subset run_00097 seed-sweep scores",
            "signal_n": len(fixed_subset_sl_verb_means),
            "noise_n": len(run00097_fixed_subset_sl_verb_noise),
            "signal_scale": fixed_subset_sl_verb_means["choice_logprob_norm_mean"].std(ddof=1),
            "noise_scale": run00097_fixed_subset_sl_verb_noise.std(ddof=1),
            "signal_range": (
                fixed_subset_sl_verb_means["choice_logprob_norm_mean"].max()
                - fixed_subset_sl_verb_means["choice_logprob_norm_mean"].min()
            ),
            "note": "Apples-to-apples fixed-subset noise floor from the corrected run_00097 sl_verb rerun.",
        },
        {
            "setting": "Fixed Subset Seedpanel / sl_verb_bpb (3-seed mean)",
            "short_label": Y_LABELS[8],
            "metric": SL_VERB_BPB_METRIC,
            "signal_definition": "std across 240 candidate means",
            "noise_definition": "mean SEM of the 3-seed candidate mean",
            "signal_n": len(fixed_subset_sl_verb_bpb_means),
            "noise_n": len(fixed_subset_sl_verb_bpb_sem),
            "signal_scale": fixed_subset_sl_verb_bpb_means["bpb_mean"].std(ddof=1),
            "noise_scale": fixed_subset_sl_verb_bpb_sem["bpb_sem"].mean(),
            "signal_range": (
                fixed_subset_sl_verb_bpb_means["bpb_mean"].max() - fixed_subset_sl_verb_bpb_means["bpb_mean"].min()
            ),
            "note": "Actual 3-seed mean noise scale on fixed-subset sl_verb BPB.",
        },
        {
            "setting": "Fixed Subset Seedpanel / sl_verb (3-seed mean)",
            "short_label": Y_LABELS[9],
            "metric": SL_VERB_METRIC,
            "signal_definition": "std across 240 candidate means",
            "noise_definition": "mean SEM of the 3-seed candidate mean",
            "signal_n": len(fixed_subset_summary),
            "noise_n": len(fixed_subset_summary),
            "signal_scale": fixed_subset_summary["choice_logprob_norm_mean"].std(ddof=1),
            "noise_scale": fixed_subset_summary["choice_logprob_norm_sem"].mean(),
            "signal_range": (
                fixed_subset_summary["choice_logprob_norm_mean"].max()
                - fixed_subset_summary["choice_logprob_norm_mean"].min()
            ),
            "note": "This is the actual noise scale on the fit target.",
        },
        {
            "setting": "Original Swarm / uncheatable_eval_bpb (ordinary seed noise)",
            "short_label": Y_LABELS[10],
            "metric": UNCHEATABLE_METRIC,
            "signal_definition": "std across 241 original swarm scores",
            "noise_definition": "std across run_00097 unfixed seed-sweep scores",
            "signal_n": int(full_swarm[UNCHEATABLE_METRIC].notna().sum()),
            "noise_n": len(run00097_standard[UNCHEATABLE_METRIC].dropna()),
            "signal_scale": full_swarm[UNCHEATABLE_METRIC].dropna().std(ddof=1),
            "noise_scale": run00097_standard[UNCHEATABLE_METRIC].dropna().std(ddof=1),
            "signal_range": (
                full_swarm[UNCHEATABLE_METRIC].dropna().max() - full_swarm[UNCHEATABLE_METRIC].dropna().min()
            ),
            "note": "Original one-shot swarm already has high SNR on this perplexity metric.",
        },
        {
            "setting": "Original Swarm / uncheatable_eval_bpb (fixed-subset seed noise)",
            "short_label": Y_LABELS[11],
            "metric": UNCHEATABLE_METRIC,
            "signal_definition": "std across 241 original swarm scores",
            "noise_definition": "std across run_00097 fixed-subset seed-sweep scores",
            "signal_n": int(full_swarm[UNCHEATABLE_METRIC].notna().sum()),
            "noise_n": len(run00097_fixed_subset_standard[UNCHEATABLE_METRIC].dropna()),
            "signal_scale": full_swarm[UNCHEATABLE_METRIC].dropna().std(ddof=1),
            "noise_scale": run00097_fixed_subset_standard[UNCHEATABLE_METRIC].dropna().std(ddof=1),
            "signal_range": (
                full_swarm[UNCHEATABLE_METRIC].dropna().max() - full_swarm[UNCHEATABLE_METRIC].dropna().min()
            ),
            "note": "Same original-swarm signal, with the fixed-subset run_00097 noise floor.",
        },
    ]

    frame = pd.DataFrame(rows)
    frame["signal_to_noise"] = frame["signal_scale"] / frame["noise_scale"]
    return frame


def _ratio_colors(values: np.ndarray) -> list[tuple[float, float, float, float]]:
    cmap = plt.colormaps["RdYlGn_r"]
    norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
    normalized = norm(values)
    return [cmap(1.0 - value) for value in normalized]


def _build_plot(frame: pd.DataFrame) -> None:
    ratios = frame["signal_to_noise"].to_numpy()
    colors = _ratio_colors(ratios)
    y = np.arange(len(frame))[::-1]

    fig, (ax_scales, ax_ratio) = plt.subplots(
        1,
        2,
        figsize=(14.2, 11.0),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
        constrained_layout=True,
    )

    for idx, (_, row) in enumerate(frame.iterrows()):
        y_pos = y[idx]
        color = colors[idx]
        ax_scales.hlines(y=y_pos, xmin=row["noise_scale"], xmax=row["signal_scale"], color=color, linewidth=3)
        ax_scales.scatter(row["noise_scale"], y_pos, color=color, edgecolors="black", linewidths=0.8, s=85, marker="o")
        ax_scales.scatter(row["signal_scale"], y_pos, color=color, edgecolors="black", linewidths=0.8, s=110, marker="D")

    ax_scales.set_xscale("log")
    ax_scales.set_yticks(y, frame["short_label"])
    ax_scales.set_xlabel("Metric Scale (log std units)")
    ax_scales.set_title("Signal vs Noise Scale")
    ax_scales.grid(axis="x", alpha=0.25)
    ax_scales.set_xlim(frame["noise_scale"].min() / 1.4, frame["signal_scale"].max() * 1.6)
    ax_scales.scatter([], [], color="white", edgecolors="black", s=85, marker="o", label="Noise scale")
    ax_scales.scatter([], [], color="white", edgecolors="black", s=110, marker="D", label="Signal scale")
    ax_scales.legend(loc="lower right", frameon=True)

    bars = ax_ratio.barh(y, frame["signal_to_noise"], color=colors, edgecolor="black", linewidth=0.8)
    ax_ratio.set_yticks(y, frame["short_label"])
    ax_ratio.set_xlabel("Signal / Noise")
    ax_ratio.set_title("Signal-to-Noise Ratio")
    ax_ratio.grid(axis="x", alpha=0.25)
    ax_ratio.axvline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    for bar, ratio, noise_n in zip(bars, frame["signal_to_noise"], frame["noise_n"], strict=True):
        ax_ratio.text(
            bar.get_width() + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{ratio:.2f}x",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )
        ax_ratio.text(
            0.02,
            bar.get_y() + bar.get_height() / 2,
            f"n={noise_n}",
            va="center",
            ha="left",
            fontsize=9,
            color="black",
        )

    fig.suptitle("Signal-to-Noise Across Swarm Settings and Objectives", fontsize=16)
    fig.text(
        0.5,
        -0.01,
        "MMLU rows use matched run_00097 seed-sweep noise floors; seedpanel rows use the 3-seed mean SEM. "
        "The two uncheatable rows use the original swarm signal with unfixed vs fixed-subset "
        "run_00097 BPB noise floors.",
        ha="center",
        fontsize=10,
    )

    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frame = _load_signal_to_noise_table()
    frame.to_csv(OUTPUT_CSV, index=False)
    _build_plot(frame)
    print(frame.to_string(index=False))
    print(f"\nWrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
