# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scikit-learn",
# ]
# ///
"""Marimo notebook for clean-slate aggregate metric diagnostics."""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.decomposition import FactorAnalysis
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.issue5416_aggregate import (
        _horn_factor_count as horn_factor_count,
    )
    from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.issue5416_aggregate import (
        _nonnegative_factor_projection as nonnegative_factor_projection,
    )

    return (
        FactorAnalysis,
        KFold,
        Path,
        RidgeCV,
        StandardScaler,
        VarianceThreshold,
        go,
        horn_factor_count,
        mo,
        nonnegative_factor_projection,
        np,
        pd,
        px,
    )


@app.cell
def _(Path):
    TWO_PHASE_ROOT = Path(__file__).resolve().parent
    REPO_ROOT = TWO_PHASE_ROOT.parents[3]
    MATRIX_DIR = TWO_PHASE_ROOT / "metric_registry" / "raw_metric_matrix_300m"
    SIGNAL_CSV = MATRIX_DIR / "raw_metric_matrix_300m.csv"
    VARIABLE_NOISE_CSV = MATRIX_DIR / "noise_baseline_run00097_variable_subset_300m.csv"
    SNR_ALL_CSV = TWO_PHASE_ROOT / "eval_signal_to_noise_all_metrics_300m_current.csv"
    KEEP_DROP_CSV = TWO_PHASE_ROOT / "eval_signal_to_noise_all_metrics_300m_current_keep_drop.csv"
    SNR_POINTS_CSV = REPO_ROOT / "experiments/domain_phase_mix/exploratory/paper_plots/img/metric_snr_summary_points.csv"
    MOE_DASHBOARD_DIR = TWO_PHASE_ROOT / "reference_outputs" / "grug_moe_mix_dashboard_20260517"
    MOE_LOSS_CSV = MOE_DASHBOARD_DIR / "grug_moe_mix_task_loss_like_metrics.csv"
    MOE_FITS_CSV = MOE_DASHBOARD_DIR / "grug_moe_mix_task_powerlaw_fits.csv"
    OUTPUT_DIR = TWO_PHASE_ROOT / "reference_outputs" / "aggregate_metric_clean_slate_20260518"
    DSP_CONTROLLABILITY_CSV = OUTPUT_DIR / "metric_controllability_effective_exposure_dsp.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return (
        DSP_CONTROLLABILITY_CSV,
        KEEP_DROP_CSV,
        MOE_FITS_CSV,
        MOE_LOSS_CSV,
        OUTPUT_DIR,
        SIGNAL_CSV,
        SNR_ALL_CSV,
        SNR_POINTS_CSV,
        VARIABLE_NOISE_CSV,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Clean-slate aggregate metric diagnostics

    Goal: build a new aggregate metric for data-mix optimization using the full
    current 300M/6B raw metric matrix, including David/#5005 raw-PPL evals, then
    validate the predicted optimum against the Grug-MoE scaling tracks.

    This notebook starts with diagnostics only:

    | Diagnostic | What It Measures | Why It Matters |
    | :--- | :--- | :--- |
    | 300M SNR | mixture signal divided by variable-subset noise | whether the metric is measurable in the 300M swarm |
    | MoE scaling R² | log-log power-law predictability across scales within each MoE track | whether the metric behaves smoothly under scaling |
    | SNR × R² join | metrics that are both measurable and scale-predictable | best candidates for aggregate construction |
    | raw-PPL coverage | David/#5005 evals now present in the 300M matrix | useful smooth proxies, but MoE scaling validation requires those evals on MoE tracks |

    Working rule for now: hard accuracy is not a fit target. We prefer smooth,
    oriented metrics such as BPB, loss, NLL, choice-logprob, and task-specific
    teacher-forced proxies.
    """
    )
    return


@app.cell
def _(
    KEEP_DROP_CSV,
    MOE_FITS_CSV,
    MOE_LOSS_CSV,
    SIGNAL_CSV,
    SNR_ALL_CSV,
    SNR_POINTS_CSV,
    VARIABLE_NOISE_CSV,
    pd,
):
    signal_frame = pd.read_csv(SIGNAL_CSV, low_memory=False)
    variable_noise_frame = pd.read_csv(VARIABLE_NOISE_CSV, low_memory=False)
    snr_all_metrics = pd.read_csv(SNR_ALL_CSV)
    keep_drop_table = pd.read_csv(KEEP_DROP_CSV)
    snr_points = pd.read_csv(SNR_POINTS_CSV)
    moe_loss_metrics = pd.read_csv(MOE_LOSS_CSV)
    moe_powerlaw_fits = pd.read_csv(MOE_FITS_CSV)
    completed_signal = signal_frame[
        signal_frame["status"].eq("completed") & signal_frame["row_kind"].eq("signal")
    ].copy()
    return (
        completed_signal,
        keep_drop_table,
        moe_loss_metrics,
        moe_powerlaw_fits,
        signal_frame,
        snr_all_metrics,
        snr_points,
        variable_noise_frame,
    )


@app.cell
def _(
    completed_signal,
    keep_drop_table,
    mo,
    moe_loss_metrics,
    moe_powerlaw_fits,
    signal_frame,
    snr_all_metrics,
    snr_points,
    variable_noise_frame,
):
    raw_ppl_snr_count = int(snr_points["metric"].str.startswith("raw_ppl/").sum())
    smooth_metric_count = int(
        snr_points["metric_leaf"].isin(["bpb", "loss", "nll", "choice_logprob", "choice_logprob_norm", "logprob"]).sum()
    )
    mo.md(
        f"""
    ## Loaded Data

    | Source | Rows / Metrics |
    | :--- | :--- |
    | 300M signal matrix | {len(signal_frame):,} rows, {len(completed_signal):,} completed |
    | variable-subset noise baseline | {len(variable_noise_frame):,} rows |
    | all metric SNR rows | {len(snr_all_metrics):,} |
    | SNR summary points | {len(snr_points):,} selected metric items |
    | keep/drop task summary rows | {len(keep_drop_table):,} |
    | David/#5005 raw-PPL items in SNR summary | {raw_ppl_snr_count:,} |
    | smooth/non-hard selected SNR items | {smooth_metric_count:,} |
    | MoE task loss-like observations | {len(moe_loss_metrics):,} |
    | MoE task power-law fits | {len(moe_powerlaw_fits):,} |
    """
    )
    return


@app.cell
def _(go, mo, snr_points):
    family_colors = {
        "Uncheatable BPB": "#4C78A8",
        "Paloma BPB": "#59A14F",
        "Agentic coding BPB": "#E15759",
        "Generic eval BPB": "#B07AA1",
        "MMLU": "#F28E2B",
        "MMLU SL-Verb": "#FFBE7D",
        "MMLU-Pro": "#9C755F",
        "English MCQ/cloze": "#EDC948",
        "Generation proxies": "#76B7B2",
        "Raw PPL task train": "#1F77B4",
        "Raw PPL multilingual": "#8CD17D",
        "Raw PPL structured/technical": "#B6992D",
        "Raw PPL bio/chem": "#499894",
        "Raw PPL other": "#79706E",
        "Other lm-eval": "#BAB0AC",
    }
    source_symbols = {
        "eval BPB/loss": "circle",
        "lm-eval task": "diamond",
        "custom task proxy": "square",
        "raw PPL": "triangle-up",
    }
    metric_snr_summary_plot = go.Figure()
    snr_points_for_plot = snr_points.sort_values("rank").copy()
    max_rank = int(snr_points_for_plot["rank"].max())
    for (family, source_class), group in snr_points_for_plot.groupby(["family", "source_class"], sort=False):
        customdata = group[
            [
                "item_label",
                "family",
                "source_class",
                "metric",
                "primary_metric_kind",
                "signal_mean",
                "signal_min",
                "signal_max",
                "noise_scale",
            ]
        ].to_numpy()
        metric_snr_summary_plot.add_trace(
            go.Scatter(
                x=group["rank"],
                y=group["signal_to_noise"],
                customdata=customdata,
                mode="markers",
                name=f"{family} · {source_class}",
                marker={
                    "color": family_colors.get(family, "#BAB0AC"),
                    "symbol": source_symbols.get(source_class, "circle"),
                    "size": 9 if source_class != "eval BPB/loss" else 8,
                    "line": {"color": "white", "width": 0.6},
                    "opacity": 0.86,
                },
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "family=%{customdata[1]}<br>"
                    "source=%{customdata[2]}<br>"
                    "selected metric=%{customdata[3]}<br>"
                    "metric kind=%{customdata[4]}<br>"
                    "signal mean=%{customdata[5]:.4f}<br>"
                    "signal range=%{customdata[6]:.4f}-%{customdata[7]:.4f}<br>"
                    "noise std=%{customdata[8]:.4f}<br>"
                    "SNR=%{y:.2f}<extra></extra>"
                ),
            )
        )
    for threshold, label in ((1, "SNR=1"), (2, "SNR=2"), (5, "SNR=5"), (10, "SNR=10")):
        metric_snr_summary_plot.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#33415c",
            line_width=1,
            annotation_text=label,
            annotation_position="right",
        )
    metric_snr_summary_plot.update_layout(
        title="300M metric signal-to-noise by best available metric per eval slice",
        xaxis_title="Eval/task slice rank by SNR",
        yaxis_title="Signal-to-noise ratio, variable-subset noise",
        height=760,
        margin={"l": 82, "r": 260, "t": 72, "b": 92},
        legend={"font": {"size": 12}, "orientation": "v", "x": 1.01, "y": 0.98},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    metric_snr_summary_plot.update_xaxes(range=[0, max_rank + 8], showgrid=True, gridcolor="#E7EDF6")
    metric_snr_summary_plot.update_yaxes(
        type="log",
        range=[-0.60206, 1.778151],
        tickvals=[0.3, 0.5, 1, 2, 5, 10, 20, 50],
        ticktext=["0.3", "0.5", "1", "2", "5", "10", "20", "50"],
        showgrid=True,
        gridcolor="#E7EDF6",
    )
    mo.vstack(
        [
            mo.md(
                """
            ## Current Metric SNR Summary

            This is the current paper-plot SNR view rebuilt interactively from
            `experiments/domain_phase_mix/exploratory/paper_plots/img/metric_snr_summary_points.csv`.
            It ranks the best smooth/proxy metric per eval slice and already
            includes the 40 David/#5005 raw-PPL items.
            """
            ),
            mo.ui.plotly(metric_snr_summary_plot),
        ]
    )
    return


@app.cell
def _(moe_loss_metrics, moe_powerlaw_fits, np, pd):
    task_groups = (
        moe_loss_metrics[["task_alias", "task_group", "loss_metric", "raw_metric"]]
        .drop_duplicates()
        .groupby("task_alias", as_index=False)
        .agg(
            task_group=("task_group", "first"),
            loss_metric=("loss_metric", "first"),
            raw_metric=("raw_metric", "first"),
        )
    )
    r2_task_summary = (
        moe_powerlaw_fits.merge(task_groups, on="task_alias", how="left")
        .groupby(["task_alias", "task_group", "loss_metric", "raw_metric"], dropna=False, as_index=False)
        .agg(
            mean_loglog_r2=("loglog_r2", "mean"),
            min_loglog_r2=("loglog_r2", "min"),
            median_loglog_r2=("loglog_r2", "median"),
            tracks=("track_label", "nunique"),
            mean_beta=("beta", "mean"),
            beta_std=("beta", "std"),
            n_points_min=("n_points", "min"),
        )
        .sort_values(["mean_loglog_r2", "min_loglog_r2"], ascending=False)
    )
    r2_task_summary["r2_band"] = pd.cut(
        r2_task_summary["mean_loglog_r2"],
        bins=[-np.inf, 0.0, 0.5, 0.8, 0.95, np.inf],
        labels=["bad/negative", "<0.5", "0.5-0.8", "0.8-0.95", ">=0.95"],
    )
    return (r2_task_summary,)


@app.cell
def _(px, r2_task_summary):
    r2_rank_plot = px.scatter(
        r2_task_summary.assign(rank=lambda frame: range(1, len(frame) + 1)),
        x="rank",
        y="mean_loglog_r2",
        color="task_group",
        symbol="r2_band",
        hover_data=[
            "task_alias",
            "loss_metric",
            "raw_metric",
            "min_loglog_r2",
            "tracks",
            "mean_beta",
            "beta_std",
        ],
        title="MoE scaling predictability by task: mean log-log R² across tracks",
        labels={
            "rank": "task rank by mean log-log R²",
            "mean_loglog_r2": "mean log-log R²",
            "task_group": "task group",
            "r2_band": "R² band",
        },
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    r2_rank_plot.add_hline(y=0.95, line_dash="dash", line_color="#444", annotation_text="R²=0.95")
    r2_rank_plot.add_hline(y=0.8, line_dash="dot", line_color="#777", annotation_text="R²=0.8")
    r2_rank_plot.update_layout(height=560)
    return (r2_rank_plot,)


@app.cell
def _(mo, r2_rank_plot, r2_task_summary):
    top_r2 = r2_task_summary.head(8)[
        ["task_alias", "task_group", "loss_metric", "mean_loglog_r2", "min_loglog_r2", "mean_beta"]
    ]
    low_r2 = r2_task_summary.tail(8)[
        ["task_alias", "task_group", "loss_metric", "mean_loglog_r2", "min_loglog_r2", "mean_beta"]
    ]
    mo.vstack(
        [
            mo.md(
                """
            ## R²-Based Scaling Diagnostic

            For each MoE task and track, we fit
            $\\log_{10}(\\text{loss-like metric}) = \\alpha + \\beta\\log_{10}(\\text{training FLOPs})$.
            The plotted value is the mean R² across the four MoE tracks.

            High R² means the metric scales smoothly in the MoE ladder, even if
            the 300M same-scale SNR is not exceptional. Low R² means the metric
            is noisy, non-monotone, or strongly track-specific at this scale.
            """
            ),
            mo.ui.plotly(r2_rank_plot),
            mo.md("### Most scale-predictable MoE task metrics"),
            mo.ui.table(top_r2, page_size=8),
            mo.md("### Least scale-predictable MoE task metrics"),
            mo.ui.table(low_r2, page_size=8),
        ]
    )
    return


@app.cell
def _():
    def canonical_task_alias(task_alias: str) -> str:
        alias = str(task_alias)
        if alias == "logprob_gsm8k_5shot":
            return "gsm8k_5shot_gold_solution"
        if alias == "logprob_humaneval_10shot":
            return "humaneval_10shot_canonical_solution"
        if alias == "mmlu_sl_5shot":
            return "mmlu_5shot"
        if alias == "mmlu_sl_0shot":
            return "mmlu_0shot"
        return alias

    def best_snr_by_task(snr_frame):
        frame = snr_frame.copy()
        frame["canonical_task_alias"] = frame["task"].map(canonical_task_alias)
        frame = frame.sort_values("signal_to_noise", ascending=False)
        return frame.drop_duplicates("canonical_task_alias")

    return best_snr_by_task, canonical_task_alias


@app.cell
def _(mo):
    primary_snr_threshold = mo.ui.slider(
        start=0.0,
        stop=10.0,
        step=0.25,
        value=2.0,
        label="Primary role SNR threshold",
    )
    downweight_snr_threshold = mo.ui.slider(
        start=0.0,
        stop=5.0,
        step=0.25,
        value=1.0,
        label="Downweight SNR threshold",
    )
    smoothness_threshold = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.70,
        label="Smoothness threshold",
    )
    noise_dominated_multiplier = mo.ui.slider(
        start=0.5,
        stop=5.0,
        step=0.25,
        value=2.0,
        label="Noise-dominated range multiplier",
    )
    weak_controllability_threshold = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.13,
        label="Weak controllability Spearman threshold",
    )
    strong_controllability_threshold = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.30,
        label="Strong controllability Spearman threshold",
    )
    apply_controllability_gate = mo.ui.checkbox(
        value=False,
        label="Gate roles on controllability",
    )
    mo.vstack(
        [
            mo.md(
                """
            ## Reactive Metric Table Controls

            The table below is built from measured coverage/SNR plus deterministic
            metric parsing. Changing these controls recomputes roles and all
            downstream summaries without re-reading the source CSVs.
            """
            ),
            mo.hstack(
                [
                    primary_snr_threshold,
                    downweight_snr_threshold,
                    smoothness_threshold,
                    noise_dominated_multiplier,
                    weak_controllability_threshold,
                    strong_controllability_threshold,
                    apply_controllability_gate,
                ],
                justify="start",
            ),
        ]
    )
    return (
        apply_controllability_gate,
        downweight_snr_threshold,
        noise_dominated_multiplier,
        primary_snr_threshold,
        smoothness_threshold,
        strong_controllability_threshold,
        weak_controllability_threshold,
    )


@app.cell
def _():
    smooth_metric_kinds = {
        "choice_prob_norm",
        "choice_logprob_norm",
        "bpb",
        "nll",
        "loss",
        "perplexity",
        "choice_logprob",
        "logprob",
    }
    hard_accuracy_kinds = {"acc", "acc_norm", "exact_match", "pass_at_1"}
    lower_is_better_kinds = {"bpb", "loss", "nll", "perplexity"}
    smoothness_weights = {
        "choice_prob_norm": 1.0,
        "choice_logprob_norm": 1.0,
        "bpb": 1.0,
        "nll": 1.0,
        "loss": 1.0,
        "perplexity": 0.9,
        "choice_logprob": 0.8,
        "logprob": 0.7,
    }

    def metric_source_class(metric: str) -> str:
        prefix = str(metric).split("/", maxsplit=1)[0]
        if prefix == "eval":
            return "eval BPB/loss"
        if prefix == "lm_eval":
            return "lm-eval task"
        if prefix in {"teacher_forced", "mcq_smooth"}:
            return "custom task proxy"
        if prefix == "raw_ppl":
            return "raw PPL"
        return "other"

    def metric_family(metric: str) -> str:
        metric = str(metric)
        parts = metric.split("/")
        if metric.startswith("eval/uncheatable_eval"):
            return "Uncheatable BPB"
        if metric.startswith("eval/paloma"):
            return "Paloma BPB"
        if metric.startswith("eval/agentic_coding"):
            return "Agentic coding BPB"
        if metric.startswith("eval/"):
            return "Generic eval BPB"
        if metric.startswith("raw_ppl/"):
            dataset = metric.removeprefix("raw_ppl/").rsplit("/", maxsplit=1)[0]
            if dataset.startswith("lm_eval/"):
                return "Raw PPL task train"
            if dataset.startswith("fineweb2_multilingual/"):
                return "Raw PPL multilingual"
            if dataset.startswith("bio_chem/"):
                return "Raw PPL bio/chem"
            if dataset.startswith(
                (
                    "binary_network_security/",
                    "formal_methods/",
                    "gh_archive_structured_output/",
                    "hardware_rtl/",
                    "long_tail_ppl_runnable/",
                    "package_metadata/",
                    "raw_web_markup/",
                    "runnable_long_tail/",
                )
            ):
                return "Raw PPL structured/technical"
            return "Raw PPL other"
        if len(parts) >= 2 and parts[0] == "lm_eval":
            task = parts[1]
            if task == "mmlu_sl_verb_5shot" or (task.startswith("mmlu_") and "_sl_verb_" in task):
                return "MMLU SL-Verb"
            if task == "mmlu_pro_5shot" or task.startswith("mmlu_pro_"):
                return "MMLU-Pro"
            if task == "mmlu_5shot" or task.startswith("mmlu_") or task.startswith("mmlu"):
                return "MMLU"
            if task.startswith(("gsm8k", "humaneval")):
                return "Generation proxies"
            if task.startswith(
                (
                    "arc_",
                    "boolq",
                    "copa",
                    "csqa",
                    "hellaswag",
                    "lambada",
                    "medmcqa",
                    "openbookqa",
                    "piqa",
                    "sciq",
                    "socialiqa",
                    "swag",
                    "truthfulqa",
                    "winogrande",
                    "wsc",
                )
            ):
                return "English MCQ/cloze"
            return "Other lm-eval"
        if metric.startswith(("teacher_forced/gsm8k", "teacher_forced/humaneval")):
            return "Generation proxies"
        if metric.startswith("mcq_smooth/"):
            return "English MCQ/cloze"
        return "Other lm-eval"

    def metric_item_id(metric: str, metric_kind: str) -> str:
        parts = str(metric).split("/")
        if len(parts) > 1 and parts[-1] == metric_kind:
            return "/".join(parts[:-1])
        return str(metric).rsplit("/", maxsplit=1)[0]

    def metric_orientation(metric_kind: str) -> str:
        if metric_kind in lower_is_better_kinds:
            return "minimize"
        return "maximize"

    return (
        hard_accuracy_kinds,
        metric_family,
        metric_item_id,
        metric_orientation,
        metric_source_class,
        smooth_metric_kinds,
        smoothness_weights,
    )


@app.cell
def _(keep_drop_table, pd):
    def hard_accuracy_correlation_rows(frame):
        rows = []
        for _, row in frame.iterrows():
            task = row.get("task")
            for column in frame.columns:
                if not column.startswith("acc_vs_") or not column.endswith("_metric"):
                    continue
                kind = column.removeprefix("acc_vs_").removesuffix("_metric")
                metric = row.get(column)
                if not isinstance(metric, str) or not metric:
                    continue
                pearson = row.get(f"acc_vs_{kind}_pearson")
                spearman = row.get(f"acc_vs_{kind}_spearman")
                rows.append(
                    {
                        "metric": metric,
                        "hard_accuracy_task": task,
                        "hard_accuracy_pearson": pearson,
                        "hard_accuracy_spearman": spearman,
                        "hard_accuracy_abs_spearman": abs(spearman) if pd.notna(spearman) else pd.NA,
                    }
                )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "metric",
                    "hard_accuracy_task",
                    "hard_accuracy_pearson",
                    "hard_accuracy_spearman",
                    "hard_accuracy_abs_spearman",
                ]
            )
        return pd.DataFrame(rows).drop_duplicates("metric")

    hard_accuracy_correlation_table = hard_accuracy_correlation_rows(keep_drop_table)
    return (hard_accuracy_correlation_table,)


@app.cell
def _(completed_signal, np, pd):
    phase_0_columns = sorted(column for column in completed_signal.columns if column.startswith("phase_0_"))
    phase_1_columns = sorted(column for column in completed_signal.columns if column.startswith("phase_1_"))
    reference_phase_0_column = phase_0_columns[-1]
    reference_phase_1_column = phase_1_columns[-1]
    controllability_feature_columns = phase_0_columns[:-1] + phase_1_columns[:-1]
    controllability_feature_frame = completed_signal[controllability_feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    controllability_feature_matrix = controllability_feature_frame.to_numpy(dtype=float)
    finite_feature_rows = np.isfinite(controllability_feature_matrix).all(axis=1)
    feature_rank = int(np.linalg.matrix_rank(controllability_feature_matrix[finite_feature_rows]))
    feature_diagnostic = {
        "rows": len(completed_signal),
        "phase_0_domains": len(phase_0_columns),
        "phase_1_domains": len(phase_1_columns),
        "feature_columns_after_reference_drop": len(controllability_feature_columns),
        "reference_phase_0_column": reference_phase_0_column,
        "reference_phase_1_column": reference_phase_1_column,
        "finite_feature_rows": int(finite_feature_rows.sum()),
        "raw_feature_rank": feature_rank,
    }
    return (
        controllability_feature_columns,
        controllability_feature_matrix,
        feature_diagnostic,
        finite_feature_rows,
    )


@app.cell
def _(
    KFold,
    RidgeCV,
    StandardScaler,
    VarianceThreshold,
    completed_signal,
    controllability_feature_matrix,
    finite_feature_rows,
    metric_orientation,
    np,
    pd,
    snr_all_metrics,
):
    ridge_alphas = np.logspace(-3, 2, 10)
    controllability_cv_seed = 0

    def safe_corr(left, right, *, method):
        frame = pd.DataFrame({"left": left, "right": right}).dropna()
        if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
            return np.nan
        return float(frame["left"].corr(frame["right"], method=method))

    def oof_ridge_controllability(metric, metric_kind):
        if metric not in completed_signal.columns:
            return {
                "metric": metric,
                "controllability_n": 0,
                "controllability_score": np.nan,
                "controllability_oof_spearman": np.nan,
                "controllability_oof_pearson": np.nan,
                "controllability_oof_r2": np.nan,
                "controllability_oof_rmse_z": np.nan,
                "controllability_alpha_median": np.nan,
            }
        y_raw = pd.to_numeric(completed_signal[metric], errors="coerce").to_numpy(dtype=float)
        if metric_orientation(metric_kind) == "minimize":
            y_raw = -y_raw
        mask = finite_feature_rows & np.isfinite(y_raw)
        if int(mask.sum()) < 40:
            return {
                "metric": metric,
                "controllability_n": int(mask.sum()),
                "controllability_score": np.nan,
                "controllability_oof_spearman": np.nan,
                "controllability_oof_pearson": np.nan,
                "controllability_oof_r2": np.nan,
                "controllability_oof_rmse_z": np.nan,
                "controllability_alpha_median": np.nan,
            }
        x = controllability_feature_matrix[mask]
        y = y_raw[mask]
        y_std = float(np.std(y, ddof=1))
        if not np.isfinite(y_std) or y_std <= 0:
            return {
                "metric": metric,
                "controllability_n": int(mask.sum()),
                "controllability_score": np.nan,
                "controllability_oof_spearman": np.nan,
                "controllability_oof_pearson": np.nan,
                "controllability_oof_r2": np.nan,
                "controllability_oof_rmse_z": np.nan,
                "controllability_alpha_median": np.nan,
            }
        predictions = np.full(len(y), np.nan, dtype=float)
        selected_alphas = []
        splitter = KFold(n_splits=5, shuffle=True, random_state=controllability_cv_seed)
        for train_index, test_index in splitter.split(x):
            y_train = y[train_index]
            train_mean = float(np.mean(y_train))
            train_std = float(np.std(y_train, ddof=1))
            if not np.isfinite(train_std) or train_std <= 0:
                predictions[test_index] = train_mean
                continue
            selector = VarianceThreshold(threshold=1e-10)
            x_train = selector.fit_transform(x[train_index])
            x_test = selector.transform(x[test_index])
            if x_train.shape[1] == 0:
                predictions[test_index] = train_mean
                continue
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            model = RidgeCV(alphas=ridge_alphas, cv=None)
            model.fit(x_train, (y_train - train_mean) / train_std)
            selected_alphas.append(float(model.alpha_))
            predictions[test_index] = model.predict(x_test) * train_std + train_mean
        valid = np.isfinite(predictions)
        if int(valid.sum()) < 3:
            oof_spearman = np.nan
            oof_pearson = np.nan
            oof_r2 = np.nan
            rmse_z = np.nan
        else:
            y_valid = y[valid]
            prediction_valid = predictions[valid]
            oof_spearman = safe_corr(y_valid, prediction_valid, method="spearman")
            oof_pearson = safe_corr(y_valid, prediction_valid, method="pearson")
            residual = y_valid - prediction_valid
            sse = float(np.sum(residual**2))
            sst = float(np.sum((y_valid - np.mean(y_valid)) ** 2))
            oof_r2 = np.nan if sst <= 0 else float(1.0 - sse / sst)
            rmse_z = float(np.sqrt(np.mean(residual**2)) / y_std)
        return {
            "metric": metric,
            "controllability_n": int(mask.sum()),
            "controllability_score": oof_spearman,
            "controllability_oof_spearman": oof_spearman,
            "controllability_oof_pearson": oof_pearson,
            "controllability_oof_r2": oof_r2,
            "controllability_oof_rmse_z": rmse_z,
            "controllability_alpha_median": float(np.median(selected_alphas)) if selected_alphas else np.nan,
        }

    controllability_rows = [
        oof_ridge_controllability(str(row["metric"]), str(row["primary_metric_kind"]))
        for _, row in snr_all_metrics.iterrows()
    ]
    controllability_table = pd.DataFrame(controllability_rows)
    return (controllability_table,)


@app.cell
def _(controllability_table, strong_controllability_threshold, weak_controllability_threshold):
    controllability_table_with_tags = controllability_table.copy()
    controllability_table_with_tags["controllability_tag"] = "unknown"
    controllability_score_series = controllability_table_with_tags["controllability_score"]
    controllability_table_with_tags.loc[
        controllability_score_series < weak_controllability_threshold.value, "controllability_tag"
    ] = "uncontrolled_or_noisy"
    controllability_table_with_tags.loc[
        controllability_score_series >= weak_controllability_threshold.value, "controllability_tag"
    ] = "weakly_controllable"
    controllability_table_with_tags.loc[
        controllability_score_series >= strong_controllability_threshold.value, "controllability_tag"
    ] = "controllable"
    return (controllability_table_with_tags,)


@app.cell
def _(controllability_table_with_tags, feature_diagnostic, mo):
    controllability_summary = (
        controllability_table_with_tags.groupby("controllability_tag", as_index=False)
        .agg(
            metric_count=("metric", "count"),
            median_spearman=("controllability_oof_spearman", "median"),
            median_r2=("controllability_oof_r2", "median"),
            median_rmse_z=("controllability_oof_rmse_z", "median"),
        )
        .sort_values("metric_count", ascending=False)
    )
    top_controllable = controllability_table_with_tags.sort_values(
        "controllability_score", ascending=False, na_position="last"
    ).head(20)
    bottom_controllable = controllability_table_with_tags.sort_values(
        "controllability_score", ascending=True, na_position="last"
    ).head(20)
    mo.vstack(
        [
            mo.md(
                f"""
            ## Empirical Partition Controllability

            This diagnostic estimates whether a metric is predictable from the
            controlled mixture partition. For each metric, we fit deterministic
            5-fold OOF ridge regression from phase-0 and phase-1 domain weights
            to the oriented metric. One reference domain is dropped per phase to
            remove exact simplex dependencies. The fold seed matches the
            standalone DSP comparator so rank metrics are compared on the same
            row splits.

            Feature diagnostics:

            | Quantity | Value |
            | :--- | :--- |
            | rows | {feature_diagnostic["rows"]} |
            | phase-0 domains | {feature_diagnostic["phase_0_domains"]} |
            | phase-1 domains | {feature_diagnostic["phase_1_domains"]} |
            | feature columns after reference drop | {feature_diagnostic["feature_columns_after_reference_drop"]} |
            | raw feature rank | {feature_diagnostic["raw_feature_rank"]} |
            | phase-0 reference | `{feature_diagnostic["reference_phase_0_column"]}` |
            | phase-1 reference | `{feature_diagnostic["reference_phase_1_column"]}` |
            """
            ),
            mo.md("### Controllability tag summary"),
            mo.ui.table(controllability_summary, page_size=10),
            mo.md("### Most mixture-predictable metrics"),
            mo.ui.table(top_controllable, page_size=20),
            mo.md("### Least mixture-predictable metrics"),
            mo.ui.table(bottom_controllable, page_size=20),
        ]
    )
    return


@app.cell
def _(DSP_CONTROLLABILITY_CSV, pd):
    if DSP_CONTROLLABILITY_CSV.exists():
        dsp_controllability_table = pd.read_csv(DSP_CONTROLLABILITY_CSV)
    else:
        dsp_controllability_table = pd.DataFrame()
    return (dsp_controllability_table,)


@app.cell
def _(dsp_controllability_table, mo, px):
    if dsp_controllability_table.empty:
        dsp_comparison_summary = None
        mo.md(
            """
            ## Effective-Exposure DSP Controllability

            No cached DSP controllability table is present yet. Generate it with:

            `uv run --with scipy --with scikit-learn python experiments/domain_phase_mix/exploratory/two_phase_many/fit_effective_exposure_dsp_metric_controllability_300m.py --workers 16`
            """
        )
    else:
        dsp_comparison = dsp_controllability_table.copy()
        dsp_comparison["dsp_minus_ridge_spearman"] = (
            dsp_comparison["dsp_oof_spearman"] - dsp_comparison["ridge_controllability_score"]
        )
        dsp_comparison_summary = (
            dsp_comparison.groupby("recommended_role", as_index=False)
            .agg(
                metrics=("metric", "count"),
                median_dsp_spearman=("dsp_oof_spearman", "median"),
                median_ridge_spearman=("ridge_controllability_score", "median"),
                median_dsp_minus_ridge=("dsp_minus_ridge_spearman", "median"),
                dsp_beats_ridge=("dsp_minus_ridge_spearman", lambda values: int((values > 0).sum())),
                median_dsp_r2=("dsp_oof_r2", "median"),
                median_ridge_r2=("ridge_oof_r2", "median"),
            )
            .sort_values("median_dsp_spearman", ascending=False)
        )
        dsp_scatter = px.scatter(
            dsp_comparison,
            x="ridge_controllability_score",
            y="dsp_oof_spearman",
            color="recommended_role",
            hover_name="metric",
            hover_data=["suite", "primary_metric_kind", "dsp_oof_r2", "ridge_oof_r2"],
            labels={
                "ridge_controllability_score": "OOF Spearman: ridge on mixture weights",
                "dsp_oof_spearman": "OOF Spearman: effective-exposure DSP",
            },
            title="Effective-exposure DSP vs ridge controllability",
        )
        dsp_scatter.add_shape(
            type="line",
            x0=-0.2,
            y0=-0.2,
            x1=1.0,
            y1=1.0,
            line={"color": "#64748B", "dash": "dash"},
        )
        dsp_scatter.update_layout(height=620)
        top_dsp_improvements = dsp_comparison.sort_values("dsp_minus_ridge_spearman", ascending=False).head(20)
        top_ridge_improvements = dsp_comparison.sort_values("dsp_minus_ridge_spearman", ascending=True).head(20)
        mo.vstack(
            [
                mo.md(
                    """
                    ## Effective-Exposure DSP Controllability

                    Ridge is a fast linear sanity check. Effective-exposure DSP is
                    the nonlinear, mixture-law-style comparison: it is fit outside
                    the notebook and loaded here as a cache so the UI remains
                    reactive. The target set here is the current best metric per
                    item, not all raw metric columns. That best-by-item selection
                    is based on metric role, SNR, and smoothness, not on DSP fit
                    quality.

                    Interpretation: DSP Spearman is a rank-order controllability
                    diagnostic. R² is shown separately because several noisy
                    metrics can have useful rank signal but poor absolute
                    calibration.
                    """
                ),
                mo.ui.table(dsp_comparison_summary, page_size=10),
                mo.ui.plotly(dsp_scatter),
                mo.md("### Metrics where DSP improves most over ridge"),
                mo.ui.table(top_dsp_improvements, page_size=20),
                mo.md("### Metrics where ridge improves most over DSP"),
                mo.ui.table(top_ridge_improvements, page_size=20),
            ]
        )
    return (dsp_comparison_summary,)


@app.cell
def _(
    apply_controllability_gate,
    canonical_task_alias,
    controllability_table_with_tags,
    downweight_snr_threshold,
    dsp_controllability_table,
    hard_accuracy_correlation_table,
    hard_accuracy_kinds,
    metric_family,
    metric_item_id,
    metric_orientation,
    metric_source_class,
    noise_dominated_multiplier,
    primary_snr_threshold,
    r2_task_summary,
    smooth_metric_kinds,
    smoothness_threshold,
    smoothness_weights,
    strong_controllability_threshold,
    snr_all_metrics,
):
    r2_lookup = r2_task_summary.copy()
    r2_lookup["canonical_task_alias"] = r2_lookup["task_alias"].map(canonical_task_alias)
    r2_lookup = r2_lookup[
        [
            "canonical_task_alias",
            "mean_loglog_r2",
            "min_loglog_r2",
            "tracks",
            "mean_beta",
        ]
    ].rename(
        columns={
            "mean_loglog_r2": "moe_mean_loglog_r2",
            "min_loglog_r2": "moe_min_loglog_r2",
            "tracks": "moe_track_count",
            "mean_beta": "moe_mean_beta",
        }
    )

    metric_table = snr_all_metrics.copy()
    metric_table["suite"] = metric_table["metric"].map(metric_family)
    metric_table["source_class"] = metric_table["metric"].map(metric_source_class)
    metric_table["item_id"] = [
        metric_item_id(metric, metric_kind)
        for metric, metric_kind in zip(metric_table["metric"], metric_table["primary_metric_kind"], strict=True)
    ]
    metric_table["orientation"] = metric_table["primary_metric_kind"].map(metric_orientation)
    metric_table["smoothness_score"] = metric_table["primary_metric_kind"].map(smoothness_weights).fillna(0.0)
    metric_table["is_smooth_proxy"] = metric_table["primary_metric_kind"].isin(smooth_metric_kinds)
    metric_table["is_hard_accuracy"] = metric_table["primary_metric_kind"].isin(hard_accuracy_kinds)
    metric_table["is_diagnostic_only"] = metric_table["metric"].str.contains(
        r"eval/agentic_coding/.*(?:_fail/|failed(?:_|/)|success_minus_failed)",
        regex=True,
    )
    metric_table["noise_dominated"] = metric_table["signal_range"] <= (
        noise_dominated_multiplier.value * metric_table["noise_scale"]
    )
    metric_table["measurability_tag"] = "low_snr"
    metric_table.loc[metric_table["signal_to_noise"] >= downweight_snr_threshold.value, "measurability_tag"] = (
        "marginal_snr"
    )
    metric_table.loc[metric_table["signal_to_noise"] >= primary_snr_threshold.value, "measurability_tag"] = "high_snr"
    metric_table["representability_tag"] = "pending_partition_controllability_fit"
    metric_table["canonical_task_alias"] = metric_table["task"].map(canonical_task_alias)
    metric_table = metric_table.merge(r2_lookup, on="canonical_task_alias", how="left")
    metric_table = metric_table.merge(hard_accuracy_correlation_table, on="metric", how="left")
    metric_table = metric_table.merge(controllability_table_with_tags, on="metric", how="left")
    dsp_columns = [
        "metric",
        "dsp_fit_status",
        "dsp_controllability_score",
        "dsp_oof_spearman",
        "dsp_oof_pearson",
        "dsp_oof_r2",
        "dsp_oof_rmse_z",
        "dsp_train_objective",
        "dsp_gamma",
    ]
    if not dsp_controllability_table.empty:
        metric_table = metric_table.merge(
            dsp_controllability_table[[column for column in dsp_columns if column in dsp_controllability_table.columns]],
            on="metric",
            how="left",
        )
    for column in dsp_columns[1:]:
        if column not in metric_table.columns:
            metric_table[column] = pd.NA

    task_like_families = {
        "Agentic coding BPB",
        "English MCQ/cloze",
        "Generation proxies",
        "MMLU",
        "MMLU SL-Verb",
        "MMLU-Pro",
        "Other lm-eval",
    }

    def recommended_role(row):
        if row["signal_n"] < 242 or row["noise_n"] < 10:
            return "report_only", "incomplete_signal_or_noise_coverage"
        if row["is_hard_accuracy"]:
            return "validation", "hard_accuracy_is_noisy_validation_signal"
        if row["is_diagnostic_only"]:
            return "validation", "diagnostic_metric_not_primary_target"
        if not row["is_smooth_proxy"]:
            return "report_only", "non_smooth_metric_kind"
        if row["noise_dominated"]:
            return "report_only", "signal_range_le_noise_band"
        if row["signal_to_noise"] < downweight_snr_threshold.value:
            return "report_only", "snr_below_downweight_threshold"
        if apply_controllability_gate.value and row["controllability_score"] < strong_controllability_threshold.value:
            return "downweight", "controllability_below_primary_threshold"
        if row["signal_to_noise"] < primary_snr_threshold.value or row["smoothness_score"] < smoothness_threshold.value:
            return "downweight", "marginal_snr_or_smoothness"
        if row["suite"] in task_like_families or row["source_class"] in {"lm-eval task", "custom task proxy"}:
            return "optimize", "smooth_task_proxy_high_snr"
        return "stabilizer", "smooth_broad_ppl_high_snr"

    role_rows = metric_table.apply(recommended_role, axis=1, result_type="expand")
    metric_table["recommended_role"] = role_rows[0]
    metric_table["role_reason"] = role_rows[1]
    role_weights = {
        "optimize": 1.0,
        "stabilizer": 0.35,
        "downweight": 0.25,
        "validation": 0.0,
        "report_only": 0.0,
    }
    metric_table["default_weight"] = metric_table["recommended_role"].map(role_weights).fillna(0.0)
    metric_table["role_priority"] = metric_table["recommended_role"].map(
        {"optimize": 0, "stabilizer": 1, "downweight": 2, "validation": 3, "report_only": 4}
    )
    metric_table = metric_table.sort_values(
        ["role_priority", "signal_to_noise", "smoothness_score"],
        ascending=[True, False, False],
    ).drop(columns=["role_priority"])
    return (metric_table,)


@app.cell
def _(metric_table):
    metric_role_summary = (
        metric_table.groupby(["recommended_role", "suite"], as_index=False)
        .agg(
            metric_count=("metric", "count"),
            item_count=("item_id", "nunique"),
            median_snr=("signal_to_noise", "median"),
            max_snr=("signal_to_noise", "max"),
            mean_smoothness=("smoothness_score", "mean"),
            moe_r2_coverage=("moe_mean_loglog_r2", lambda values: values.notna().mean()),
        )
        .sort_values(["recommended_role", "metric_count"], ascending=[True, False])
    )
    role_priority = {"optimize": 0, "stabilizer": 1, "downweight": 2, "validation": 3, "report_only": 4}
    metric_best_by_item = (
        metric_table.assign(_role_priority=metric_table["recommended_role"].map(role_priority))
        .sort_values(
            ["item_id", "_role_priority", "signal_to_noise", "smoothness_score"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates("item_id")
        .drop(columns=["_role_priority"])
        .assign(_role_priority=lambda frame: frame["recommended_role"].map(role_priority))
        .sort_values(["_role_priority", "signal_to_noise"], ascending=[True, False])
        .drop(columns=["_role_priority"])
    )
    return metric_best_by_item, metric_role_summary


@app.cell
def _(metric_best_by_item, metric_role_summary, metric_table, mo):
    role_counts = (
        metric_table.groupby("recommended_role", as_index=False)
        .agg(metrics=("metric", "count"), items=("item_id", "nunique"), median_snr=("signal_to_noise", "median"))
        .sort_values("metrics", ascending=False)
    )
    table_columns = [
        "metric",
        "suite",
        "source_class",
        "item_id",
        "primary_metric_kind",
        "orientation",
        "signal_to_noise",
        "measurability_tag",
        "smoothness_score",
        "representability_tag",
        "controllability_tag",
        "controllability_score",
        "controllability_oof_r2",
        "controllability_oof_rmse_z",
        "dsp_controllability_score",
        "dsp_oof_r2",
        "dsp_oof_rmse_z",
        "is_diagnostic_only",
        "moe_mean_loglog_r2",
        "hard_accuracy_spearman",
        "recommended_role",
        "role_reason",
        "default_weight",
    ]
    mo.vstack(
        [
            mo.md(
                """
            ## Reactive Metric Candidate Table

            This is the v0 aggregate-metric item inventory. Roles are computed
            from coverage, variable-subset SNR, smoothness, and deterministic
            metric ontology. Hard-accuracy correlation is shown only as a
            diagnostic and does not gate inclusion.

            `partition_controllability` and redundancy clustering are deferred:
            they require fitting mixture-to-metric models or metric clustering,
            and should be added as separate reactive cells rather than manual tags.
            """
            ),
            mo.md("### Role summary"),
            mo.ui.table(role_counts, page_size=10),
            mo.md("### Role × suite summary"),
            mo.ui.table(metric_role_summary, page_size=20),
            mo.md("### Best metric per item under current rules"),
            mo.ui.table(metric_best_by_item[table_columns], page_size=25),
            mo.md("### Full metric table"),
            mo.ui.table(metric_table[table_columns], page_size=25),
        ]
    )
    return


@app.cell
def _(mo):
    aggregate_min_snr = mo.ui.slider(
        start=0.0,
        stop=10.0,
        step=0.25,
        value=1.0,
        label="Aggregate item minimum SNR",
    )
    aggregate_min_dsp_spearman = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.30,
        label="DSP-weighted item minimum Spearman",
    )
    aggregate_min_moe_r2 = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.50,
        label="MoE R² minimum when available",
    )
    aggregate_min_suite_items = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=2,
        label="Minimum selected items per suite",
    )
    aggregate_min_suite_dsp_spearman = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.50,
        label="Minimum suite median DSP Spearman",
    )
    aggregate_task_suite_mass = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.60,
        label="Role-balanced task-suite mass",
    )
    aggregate_include_stabilizers = mo.ui.checkbox(
        value=True,
        label="Include stabilizer metrics",
    )
    aggregate_include_downweight = mo.ui.checkbox(
        value=False,
        label="Include downweight metrics",
    )
    aggregate_use_moe_r2_weight = mo.ui.checkbox(
        value=True,
        label="Use MoE R² as optional weight",
    )
    mo.vstack(
        [
            mo.md(
                """
                ## Aggregate Construction Controls

                The primary aggregate avoids DSP-Spearman weighting so that
                later DSP fit quality remains a meaningful validation signal.
                DSP-weighted variants are included as diagnostics.
                """
            ),
            mo.hstack(
                [
                    aggregate_min_snr,
                    aggregate_min_dsp_spearman,
                    aggregate_min_moe_r2,
                    aggregate_min_suite_items,
                    aggregate_min_suite_dsp_spearman,
                    aggregate_task_suite_mass,
                    aggregate_include_stabilizers,
                    aggregate_include_downweight,
                    aggregate_use_moe_r2_weight,
                ],
                justify="start",
            ),
        ]
    )
    return (
        aggregate_include_downweight,
        aggregate_include_stabilizers,
        aggregate_min_dsp_spearman,
        aggregate_min_moe_r2,
        aggregate_min_suite_dsp_spearman,
        aggregate_min_suite_items,
        aggregate_min_snr,
        aggregate_task_suite_mass,
        aggregate_use_moe_r2_weight,
    )


@app.cell
def _(
    aggregate_include_downweight,
    aggregate_include_stabilizers,
    aggregate_min_dsp_spearman,
    aggregate_min_moe_r2,
    aggregate_min_suite_dsp_spearman,
    aggregate_min_suite_items,
    aggregate_min_snr,
    aggregate_use_moe_r2_weight,
    completed_signal,
    metric_best_by_item,
    np,
    pd,
):
    aggregate_roles = {"optimize"}
    if aggregate_include_stabilizers.value:
        aggregate_roles.add("stabilizer")
    if aggregate_include_downweight.value:
        aggregate_roles.add("downweight")

    candidate_items = metric_best_by_item.copy()
    candidate_items = candidate_items[
        candidate_items["recommended_role"].isin(aggregate_roles)
        & ~candidate_items["is_hard_accuracy"].fillna(False)
        & ~candidate_items["is_diagnostic_only"].fillna(False)
        & candidate_items["is_smooth_proxy"].fillna(False)
        & (candidate_items["signal_n"] >= 242)
        & (candidate_items["noise_n"] >= 10)
        & (candidate_items["signal_to_noise"] >= aggregate_min_snr.value)
    ].copy()
    candidate_items = candidate_items[candidate_items["metric"].isin(completed_signal.columns)].copy()

    role_weight = {"optimize": 1.0, "stabilizer": 0.35, "downweight": 0.25}
    candidate_items["role_aggregate_weight"] = candidate_items["recommended_role"].map(role_weight).fillna(0.0)

    def aggregate_suite_name(row):
        metric = str(row["metric"])
        suite = str(row["suite"])
        metric_parts = metric.split("/")
        metric_leaf = metric_parts[-1]
        is_eval_suite_macro = metric.startswith("eval/") and (
            metric_leaf.startswith("macro")
            or metric_leaf.startswith("micro")
            or "_macro_" in metric_leaf
            or metric
            in {"eval/paloma/bpb", "eval/paloma/loss", "eval/uncheatable_eval/bpb", "eval/uncheatable_eval/loss"}
        )
        if metric.startswith("lm_eval/averages/") or metric.startswith("eval/macro") or is_eval_suite_macro:
            return "Meta aggregates"
        if suite in {"Generation proxies", "Raw PPL task train"}:
            return "Generative/task-train proxies"
        return suite

    candidate_items["aggregate_suite"] = candidate_items.apply(aggregate_suite_name, axis=1)
    candidate_items = candidate_items[~candidate_items["aggregate_suite"].eq("Meta aggregates")].copy()
    candidate_items["snr_factor"] = (candidate_items["signal_to_noise"] / 5.0).clip(lower=0.0, upper=1.0)
    candidate_items["dsp_factor"] = (
        candidate_items["dsp_controllability_score"].fillna(0.0).clip(lower=0.0) / 0.8
    ).clip(lower=0.0, upper=1.0)
    moe_r2 = pd.to_numeric(candidate_items["moe_mean_loglog_r2"], errors="coerce")
    if aggregate_use_moe_r2_weight.value:
        candidate_items["moe_r2_factor"] = moe_r2.clip(lower=0.0, upper=1.0).fillna(1.0)
    else:
        candidate_items["moe_r2_factor"] = 1.0
    candidate_items["passes_moe_r2_filter"] = moe_r2.isna() | (moe_r2 >= aggregate_min_moe_r2.value)
    candidate_items["passes_dsp_filter"] = (
        candidate_items["dsp_controllability_score"].fillna(-np.inf) >= aggregate_min_dsp_spearman.value
    )
    candidate_items["primary_weight_raw"] = (
        candidate_items["role_aggregate_weight"] * candidate_items["snr_factor"] * candidate_items["moe_r2_factor"]
    )
    candidate_items["dsp_weight_raw"] = candidate_items["primary_weight_raw"] * candidate_items["dsp_factor"]
    candidate_items = candidate_items[
        candidate_items["passes_moe_r2_filter"] & (candidate_items["primary_weight_raw"] > 0)
    ].copy()
    suite_filter_summary = (
        candidate_items.groupby("aggregate_suite", as_index=False)
        .agg(
            prefilter_items=("metric", "count"),
            prefilter_median_dsp_spearman=("dsp_controllability_score", "median"),
        )
        .assign(
            passes_suite_item_count=lambda frame: frame["prefilter_items"] >= aggregate_min_suite_items.value,
            passes_suite_controllability=lambda frame: frame["prefilter_median_dsp_spearman"].fillna(-np.inf)
            >= aggregate_min_suite_dsp_spearman.value,
        )
    )
    candidate_items = candidate_items.merge(
        suite_filter_summary[
            [
                "aggregate_suite",
                "prefilter_items",
                "prefilter_median_dsp_spearman",
                "passes_suite_item_count",
                "passes_suite_controllability",
            ]
        ],
        on="aggregate_suite",
        how="left",
    )
    candidate_items = candidate_items[candidate_items["passes_suite_item_count"]].copy()
    candidate_items_dsp_suite_gated = candidate_items[candidate_items["passes_suite_controllability"]].copy()

    z_columns = {}
    clipped_z_columns = {}
    rank_z_columns = {}

    def build_aggregate_item_table(source_items):
        kept_rows = []
        for _, row in source_items.iterrows():
            metric = str(row["metric"])
            values = pd.to_numeric(completed_signal[metric], errors="coerce").to_numpy(dtype=float)
            if row["orientation"] == "minimize":
                values = -values
            std = float(np.nanstd(values, ddof=1))
            if not np.isfinite(std) or std <= 1e-12:
                continue
            mean = float(np.nanmean(values))
            if metric not in z_columns:
                z_values = (values - mean) / std
                z_columns[metric] = z_values
                clipped_values = np.clip(z_values, -2.5, 2.5)
                clipped_std = float(np.nanstd(clipped_values, ddof=1))
                if np.isfinite(clipped_std) and clipped_std > 1e-12:
                    clipped_z_columns[metric] = (clipped_values - float(np.nanmean(clipped_values))) / clipped_std
                else:
                    clipped_z_columns[metric] = z_values
                rank_values = pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)
                rank_values = rank_values - float(np.nanmean(rank_values))
                rank_std = float(np.nanstd(rank_values, ddof=1))
                if np.isfinite(rank_std) and rank_std > 1e-12:
                    rank_z_columns[metric] = rank_values / rank_std
                else:
                    rank_z_columns[metric] = z_values
            kept_row = row.to_dict()
            kept_row["oriented_mean"] = mean
            kept_row["oriented_std"] = std
            kept_rows.append(kept_row)
        return pd.DataFrame(kept_rows)

    aggregate_item_table = build_aggregate_item_table(candidate_items)
    aggregate_item_table_dsp_suite_gated = build_aggregate_item_table(candidate_items_dsp_suite_gated)
    if z_columns:
        aggregate_item_z = pd.DataFrame(z_columns, index=completed_signal.index)
        aggregate_item_clipped_z = pd.DataFrame(clipped_z_columns, index=completed_signal.index)
        aggregate_item_rank_z = pd.DataFrame(rank_z_columns, index=completed_signal.index)
    else:
        aggregate_item_z = pd.DataFrame(index=completed_signal.index)
        aggregate_item_clipped_z = pd.DataFrame(index=completed_signal.index)
        aggregate_item_rank_z = pd.DataFrame(index=completed_signal.index)

    return (
        aggregate_item_clipped_z,
        aggregate_item_rank_z,
        aggregate_item_table_dsp_suite_gated,
        aggregate_item_table,
        aggregate_item_z,
    )


@app.cell
def _(
    FactorAnalysis,
    aggregate_item_table,
    aggregate_item_z,
    completed_signal,
    horn_factor_count,
    nonnegative_factor_projection,
    np,
    pd,
):
    def zscore_vector(values: np.ndarray) -> np.ndarray:
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values, ddof=1))
        if not np.isfinite(std) or std <= 1e-12:
            return np.full_like(values, np.nan, dtype=float)
        return (values - mean) / std

    def varimax_rotation(loadings: np.ndarray, *, gamma: float = 1.0, max_iter: int = 100, tol: float = 1e-6):
        rows, cols = loadings.shape
        rotation = np.eye(cols)
        previous_score = 0.0
        for _ in range(max_iter):
            rotated = loadings @ rotation
            u, singular_values, vt = np.linalg.svd(
                loadings.T @ (rotated**3 - (gamma / rows) * rotated @ np.diag(np.diag(rotated.T @ rotated)))
            )
            rotation = u @ vt
            score = float(np.sum(singular_values))
            if previous_score and score < previous_score * (1.0 + tol):
                break
            previous_score = score
        return loadings @ rotation, rotation

    factor_item_table = pd.DataFrame()
    factor_scores = pd.DataFrame()
    factor_loadings = pd.DataFrame()
    factor_horn_parallel = pd.DataFrame()
    factor_summary = pd.DataFrame()
    if not aggregate_item_table.empty:
        factor_metrics = aggregate_item_table["metric"].astype(str).tolist()
        factor_z = aggregate_item_z[factor_metrics].to_numpy(dtype=float)
        if factor_z.shape[1] >= 2 and np.isfinite(factor_z).all():
            raw_noise_share = (
                pd.to_numeric(aggregate_item_table["noise_scale"], errors="coerce")
                / pd.to_numeric(aggregate_item_table["signal_scale"], errors="coerce")
            ) ** 2
            noise_share = raw_noise_share.to_numpy(dtype=float)
            noise_share[~np.isfinite(noise_share)] = np.nan
            factor_count, real_eigenvalues, random_p95 = horn_factor_count(factor_z, seed=42, n_mc=500)
            factor_count = min(int(factor_count), min(factor_z.shape) - 1)
            factor_horn_parallel = pd.DataFrame(
                {
                    "rank": np.arange(1, len(real_eigenvalues) + 1),
                    "real_eigenvalue": real_eigenvalues,
                    "random_p95": random_p95,
                    "selected": np.arange(1, len(real_eigenvalues) + 1) <= factor_count,
                }
            )
            horn_loadings, horn_uniquenesses, horn_projection = nonnegative_factor_projection(
                factor_z,
                noise_share,
                factor_count=factor_count,
                seed=0,
                max_iter=5000,
                tolerance=1e-7,
            )
            weighted_loadings = horn_loadings / horn_uniquenesses[:, None]
            posterior_cov = np.linalg.inv(np.eye(factor_count) + horn_loadings.T @ weighted_loadings)
            horn_theta = factor_z @ weighted_loadings @ posterior_cov
            horn_theta_z = np.column_stack([zscore_vector(horn_theta[:, index]) for index in range(factor_count)])
            horn_factor_mean = zscore_vector(np.nanmean(horn_theta_z, axis=1))
            horn_balance_penalized = zscore_vector(
                np.nanmean(horn_theta_z, axis=1) - 0.25 * np.nanstd(horn_theta_z, axis=1, ddof=1)
            )

            fa = FactorAnalysis(n_components=factor_count, random_state=0, max_iter=2000)
            fa_scores = fa.fit_transform(factor_z)
            varimax_loadings, varimax_rotation_matrix = varimax_rotation(fa.components_.T)
            varimax_scores = fa_scores @ varimax_rotation_matrix
            reference_score = np.nanmean(factor_z, axis=1)
            for factor_index in range(factor_count):
                _factor_corr = np.corrcoef(varimax_scores[:, factor_index], reference_score)[0, 1]
                if np.isfinite(_factor_corr) and _factor_corr < 0:
                    varimax_scores[:, factor_index] *= -1.0
                    varimax_loadings[:, factor_index] *= -1.0
            varimax_scores_z = np.column_stack(
                [zscore_vector(varimax_scores[:, index]) for index in range(factor_count)]
            )
            varimax_mean = zscore_vector(np.nanmean(varimax_scores_z, axis=1))

            factor_scores = completed_signal[["run_name", "registry_run_key"]].copy()
            factor_scores["item_factor_horn_balanced_projection"] = horn_factor_mean
            factor_scores["item_factor_varimax_diagnostic"] = varimax_mean
            factor_scores["item_factor_balance_penalized"] = horn_balance_penalized
            for factor_index in range(factor_count):
                factor_scores[f"item_factor_horn_theta_{factor_index + 1}"] = horn_theta_z[:, factor_index]
                factor_scores[f"item_factor_varimax_theta_{factor_index + 1}"] = varimax_scores_z[:, factor_index]

            _factor_loading_rows = []
            for item_index, _factor_item_row in enumerate(aggregate_item_table.itertuples(index=False)):
                base = {
                    "metric": str(_factor_item_row.metric),
                    "task": str(_factor_item_row.task),
                    "suite": str(_factor_item_row.suite),
                    "aggregate_suite": str(_factor_item_row.aggregate_suite),
                    "signal_to_noise": float(_factor_item_row.signal_to_noise),
                    "noise_share": None if np.isnan(noise_share[item_index]) else float(noise_share[item_index]),
                    "uniqueness": float(horn_uniquenesses[item_index]),
                    "projection_weight": float(horn_projection[item_index]),
                }
                for factor_index in range(factor_count):
                    _factor_loading_rows.append(
                        {
                            **base,
                            "model": "horn_nonnegative_noise_anchored",
                            "factor": f"F{factor_index + 1}",
                            "loading": float(horn_loadings[item_index, factor_index]),
                        }
                    )
                    _factor_loading_rows.append(
                        {
                            **base,
                            "model": "varimax_factor_analysis",
                            "factor": f"F{factor_index + 1}",
                            "loading": float(varimax_loadings[item_index, factor_index]),
                        }
                    )
            factor_loadings = pd.DataFrame(_factor_loading_rows)
            factor_item_table = aggregate_item_table.copy()
            factor_item_table["noise_share"] = noise_share
            factor_item_table["horn_projection_weight"] = horn_projection
            abs_projection = np.abs(horn_projection)
            projection_total = float(abs_projection.sum())
            factor_item_table["horn_abs_projection_share"] = (
                abs_projection / projection_total if projection_total > 0 else np.nan
            )
            factor_summary = pd.DataFrame(
                [
                    {
                        "candidate": "item_factor_horn_balanced_projection",
                        "factor_model": "horn_nonnegative_noise_anchored",
                        "factor_count": factor_count,
                        "items": len(factor_metrics),
                        "score_std": float(np.nanstd(horn_factor_mean, ddof=1)),
                    },
                    {
                        "candidate": "item_factor_varimax_diagnostic",
                        "factor_model": "varimax_factor_analysis",
                        "factor_count": factor_count,
                        "items": len(factor_metrics),
                        "score_std": float(np.nanstd(varimax_mean, ddof=1)),
                    },
                    {
                        "candidate": "item_factor_balance_penalized",
                        "factor_model": "horn_nonnegative_noise_anchored",
                        "factor_count": factor_count,
                        "items": len(factor_metrics),
                        "score_std": float(np.nanstd(horn_balance_penalized, ddof=1)),
                        "imbalance_penalty_beta": 0.25,
                    },
                ]
            )
    return factor_horn_parallel, factor_item_table, factor_loadings, factor_scores, factor_summary


@app.cell
def _(factor_horn_parallel, factor_loadings, factor_scores, factor_summary, go, mo, pd, px):
    if factor_scores.empty or factor_loadings.empty:
        factor_view = mo.md(
            "## Item-Level Factor Aggregates\n\nNo factor aggregate could be built from the selected item table."
        )
    else:
        horn_plot = px.line(
            factor_horn_parallel,
            x="rank",
            y=["real_eigenvalue", "random_p95"],
            title="Horn parallel analysis for item-level aggregate factors",
            labels={"value": "correlation-matrix eigenvalue", "rank": "factor rank", "variable": "series"},
            markers=True,
        )
        horn_plot.update_layout(height=420)
        selected_horn = factor_horn_parallel[factor_horn_parallel["selected"]].copy()
        loadings_for_heatmap = factor_loadings[factor_loadings["model"].eq("horn_nonnegative_noise_anchored")].copy()
        loading_matrix = loadings_for_heatmap.pivot_table(
            index="metric",
            columns="factor",
            values="loading",
            aggfunc="mean",
            fill_value=0.0,
        )
        loading_matrix = loading_matrix.loc[loading_matrix.abs().max(axis=1).sort_values(ascending=False).index]
        loading_heatmap = px.imshow(
            loading_matrix,
            aspect="auto",
            color_continuous_scale="RdYlGn_r",
            title="Noise-anchored nonnegative item factor loadings",
            labels={"x": "Factor", "y": "Metric", "color": "loading"},
        )
        loading_heatmap.update_layout(height=max(720, 16 * len(loading_matrix) + 100))
        factor_leaderboard = factor_scores[
            [
                "run_name",
                "item_factor_horn_balanced_projection",
                "item_factor_varimax_diagnostic",
                "item_factor_balance_penalized",
            ]
        ].sort_values("item_factor_horn_balanced_projection", ascending=False)
        factor_view = mo.vstack(
            [
                mo.md(
                    """
                    ## Item-Level Factor Aggregates

                    These factors are built directly from the selected smooth item
                    metrics, not from suite-level means. The primary candidate is
                    `item_factor_horn_balanced_projection`: Horn-selected `K`,
                    noise-share anchored nonnegative factor analysis, and an
                    equal mean over posterior factor scores.
                    """
                ),
                mo.ui.plotly(horn_plot),
                mo.md("### Selected Horn factors"),
                mo.ui.table(selected_horn, page_size=10),
                mo.ui.plotly(loading_heatmap),
                mo.md("### Factor candidate summary"),
                mo.ui.table(factor_summary, page_size=10),
                mo.md("### Top rows by item-level factor aggregate"),
                mo.ui.table(factor_leaderboard.head(25), page_size=25),
            ]
        )
    mo.output.replace(factor_view)
    return


@app.cell
def _(
    aggregate_item_table,
    aggregate_item_z,
    completed_signal,
    horn_factor_count,
    nonnegative_factor_projection,
    np,
    pd,
):
    def _factor_iteration_zscore(_values: np.ndarray) -> np.ndarray:
        _mean = float(np.nanmean(_values))
        _std = float(np.nanstd(_values, ddof=1))
        if not np.isfinite(_std) or _std <= 1e-12:
            return np.full_like(_values, np.nan, dtype=float)
        return (_values - _mean) / _std

    def _fit_nonnegative_factor_candidates(_items: pd.DataFrame, _prefix: str, _betas: tuple[float, ...]):
        if _items.empty or len(_items) < 2:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        _metrics = _items["metric"].astype(str).tolist()
        _z = aggregate_item_z[_metrics].to_numpy(dtype=float)
        if not np.isfinite(_z).all():
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        _noise_share = (
            pd.to_numeric(_items["noise_scale"], errors="coerce")
            / pd.to_numeric(_items["signal_scale"], errors="coerce")
        ).to_numpy(dtype=float) ** 2
        _noise_share[~np.isfinite(_noise_share)] = np.nan
        _factor_count, _real_eigenvalues, _random_p95 = horn_factor_count(_z, seed=42, n_mc=500)
        _factor_count = min(int(_factor_count), min(_z.shape) - 1)
        if _factor_count < 1:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        _loadings, _uniquenesses, _projection = nonnegative_factor_projection(
            _z,
            _noise_share,
            factor_count=_factor_count,
            seed=0,
            max_iter=5000,
            tolerance=1e-7,
        )
        _weighted_loadings = _loadings / _uniquenesses[:, None]
        _posterior_cov = np.linalg.inv(np.eye(_factor_count) + _loadings.T @ _weighted_loadings)
        _theta = _z @ _weighted_loadings @ _posterior_cov
        _theta_z = np.column_stack([_factor_iteration_zscore(_theta[:, _index]) for _index in range(_factor_count)])
        _mean_score = np.nanmean(_theta_z, axis=1)
        _std_score = np.nanstd(_theta_z, axis=1, ddof=1)
        _scores = completed_signal[["run_name", "registry_run_key"]].copy()
        _scores[f"{_prefix}_balanced_projection"] = _factor_iteration_zscore(_mean_score)
        _summary_rows = [
            {
                "candidate": f"{_prefix}_balanced_projection",
                "item_filter": _prefix,
                "factor_count": _factor_count,
                "items": len(_items),
                "imbalance_penalty_beta": 0.0,
                "score_std": float(np.nanstd(_scores[f"{_prefix}_balanced_projection"], ddof=1)),
            }
        ]
        for _beta in _betas:
            _candidate = f"{_prefix}_balance_penalized_beta{str(_beta).replace('.', 'p')}"
            _scores[_candidate] = _factor_iteration_zscore(_mean_score - _beta * _std_score)
            _summary_rows.append(
                {
                    "candidate": _candidate,
                    "item_filter": _prefix,
                    "factor_count": _factor_count,
                    "items": len(_items),
                    "imbalance_penalty_beta": _beta,
                    "score_std": float(np.nanstd(_scores[_candidate], ddof=1)),
                }
            )
        _loading_rows = []
        for _item_index, _item_row in enumerate(_items.itertuples(index=False)):
            for _factor_index in range(_factor_count):
                _loading_rows.append(
                    {
                        "item_filter": _prefix,
                        "metric": str(_item_row.metric),
                        "task": str(_item_row.task),
                        "suite": str(_item_row.suite),
                        "aggregate_suite": str(_item_row.aggregate_suite),
                        "factor": f"F{_factor_index + 1}",
                        "loading": float(_loadings[_item_index, _factor_index]),
                        "noise_share": None if np.isnan(_noise_share[_item_index]) else float(_noise_share[_item_index]),
                        "uniqueness": float(_uniquenesses[_item_index]),
                        "projection_weight": float(_projection[_item_index]),
                    }
                )
        return _scores, pd.DataFrame(_loading_rows), pd.DataFrame(_summary_rows)

    factor_iteration_spec_limit = 10
    factor_iteration_specs = [
        (
            "item_factor_controllable065",
            aggregate_item_table[
                pd.to_numeric(aggregate_item_table["dsp_controllability_score"], errors="coerce").ge(0.65)
            ].copy(),
        ),
        (
            "item_factor_controllable080",
            aggregate_item_table[
                pd.to_numeric(aggregate_item_table["dsp_controllability_score"], errors="coerce").ge(0.80)
            ].copy(),
        ),
        (
            "item_factor_optimize_controllable065",
            aggregate_item_table[
                aggregate_item_table["recommended_role"].eq("optimize")
                & pd.to_numeric(aggregate_item_table["dsp_controllability_score"], errors="coerce").ge(0.65)
            ].copy(),
        ),
        (
            "item_factor_task_proxy_controllable065",
            aggregate_item_table[
                aggregate_item_table["recommended_role"].eq("optimize")
                & aggregate_item_table["source_class"].isin(["lm-eval task", "custom task proxy"])
                & pd.to_numeric(aggregate_item_table["dsp_controllability_score"], errors="coerce").ge(0.65)
            ].copy(),
        ),
    ][:factor_iteration_spec_limit]
    _iteration_score_parts = []
    _iteration_loading_parts = []
    _iteration_summary_parts = []
    for _prefix, _items in factor_iteration_specs:
        _scores, _loadings, _summary = _fit_nonnegative_factor_candidates(_items, _prefix, (0.5, 1.0))
        if not _scores.empty:
            _iteration_score_parts.append(_scores)
        if not _loadings.empty:
            _iteration_loading_parts.append(_loadings)
        if not _summary.empty:
            _iteration_summary_parts.append(_summary)
    if _iteration_score_parts:
        factor_iteration_scores = completed_signal[["run_name", "registry_run_key"]].copy()
        for _scores in _iteration_score_parts:
            _score_columns = [column for column in _scores.columns if column not in {"run_name", "registry_run_key"}]
            factor_iteration_scores = factor_iteration_scores.merge(
                _scores[["run_name", "registry_run_key"] + _score_columns],
                on=["run_name", "registry_run_key"],
                how="left",
                validate="one_to_one",
            )
    else:
        factor_iteration_scores = pd.DataFrame()
    factor_iteration_loadings = (
        pd.concat(_iteration_loading_parts, ignore_index=True) if _iteration_loading_parts else pd.DataFrame()
    )
    factor_iteration_summary = (
        pd.concat(_iteration_summary_parts, ignore_index=True) if _iteration_summary_parts else pd.DataFrame()
    )
    return factor_iteration_loadings, factor_iteration_scores, factor_iteration_spec_limit, factor_iteration_summary


@app.cell
def _(factor_iteration_scores, factor_iteration_spec_limit, factor_iteration_summary, mo):
    if factor_iteration_summary.empty:
        factor_iteration_view = mo.md(
            f"## Factor Iteration Candidates\n\nNo stricter factor candidates were produced. Iteration spec limit: `{factor_iteration_spec_limit}`."
        )
    else:
        factor_iteration_view = mo.vstack(
            [
                mo.md(
                    f"""
                    ## Factor Iteration Candidates

                    The sprint loop is capped at `{factor_iteration_spec_limit}`
                    item-filter iterations. Each iteration can emit multiple
                    candidate score variants. These stricter candidates test
                    whether controllability filters and stronger factor-balance
                    penalties improve raw DSP optimum geometry.
                    """
                ),
                mo.ui.table(factor_iteration_summary, page_size=20),
                mo.ui.table(factor_iteration_scores.head(25), page_size=25),
            ]
        )
    mo.output.replace(factor_iteration_view)
    return


@app.cell
def _(
    aggregate_item_table,
    aggregate_item_table_dsp_suite_gated,
    aggregate_item_clipped_z,
    aggregate_item_rank_z,
    aggregate_item_z,
    aggregate_task_suite_mass,
    completed_signal,
    factor_iteration_scores,
    factor_scores,
    np,
    pd,
):
    def normalized_item_weights(items: pd.DataFrame, weight_column: str, *, suite_balance: bool) -> pd.Series:
        weights = pd.to_numeric(items[weight_column], errors="coerce").fillna(0.0).clip(lower=0.0)
        if not suite_balance:
            total = float(weights.sum())
            if total <= 0:
                return pd.Series(dtype=float)
            return weights / total
        suite_parts = []
        active = items.loc[weights > 0, ["metric", "aggregate_suite"]].copy()
        active["_weight"] = weights.loc[weights > 0].to_numpy(dtype=float)
        suites = sorted(active["aggregate_suite"].dropna().unique())
        if not suites:
            return pd.Series(dtype=float)
        suite_mass = 1.0 / len(suites)
        for suite, suite_frame in active.groupby("aggregate_suite"):
            suite_weights = suite_frame["_weight"]
            suite_total = float(suite_weights.sum())
            if suite_total <= 0:
                continue
            suite_parts.append(
                pd.Series(suite_mass * suite_weights.to_numpy(dtype=float) / suite_total, index=suite_frame.index)
            )
        if not suite_parts:
            return pd.Series(dtype=float)
        return pd.concat(suite_parts).sort_index()

    def role_balanced_item_weights(items: pd.DataFrame, weight_column: str, task_suite_mass: float) -> pd.Series:
        weights = pd.to_numeric(items[weight_column], errors="coerce").fillna(0.0).clip(lower=0.0)
        active = items.loc[weights > 0, ["metric", "aggregate_suite", "recommended_role"]].copy()
        if active.empty:
            return pd.Series(dtype=float)
        active["_weight"] = weights.loc[weights > 0].to_numpy(dtype=float)
        suite_roles = (
            active.groupby("aggregate_suite")["recommended_role"]
            .apply(lambda values: "task" if (values == "optimize").any() else "stabilizer")
            .to_dict()
        )
        present_roles = sorted(set(suite_roles.values()))
        if len(present_roles) == 1:
            role_mass = {present_roles[0]: 1.0}
        else:
            clipped_task_mass = float(np.clip(task_suite_mass, 0.0, 1.0))
            role_mass = {"task": clipped_task_mass, "stabilizer": 1.0 - clipped_task_mass}
        parts = []
        for role in present_roles:
            role_suites = sorted(suite for suite, suite_role in suite_roles.items() if suite_role == role)
            if not role_suites:
                continue
            per_suite_mass = role_mass.get(role, 0.0) / len(role_suites)
            for suite in role_suites:
                suite_frame = active[active["aggregate_suite"].eq(suite)]
                suite_weights = suite_frame["_weight"]
                suite_total = float(suite_weights.sum())
                if suite_total <= 0:
                    continue
                parts.append(
                    pd.Series(
                        per_suite_mass * suite_weights.to_numpy(dtype=float) / suite_total, index=suite_frame.index
                    )
                )
        if not parts:
            return pd.Series(dtype=float)
        return pd.concat(parts).sort_index()

    def weighted_score(items: pd.DataFrame, weights: pd.Series, value_frame: pd.DataFrame) -> pd.Series:
        if items.empty or weights.empty:
            return pd.Series(np.nan, index=completed_signal.index)
        metric_order = items.loc[weights.index, "metric"].astype(str).tolist()
        matrix = value_frame[metric_order].to_numpy(dtype=float)
        vector = weights.to_numpy(dtype=float)
        return pd.Series(matrix @ vector, index=completed_signal.index)

    aggregate_primary_weights = normalized_item_weights(aggregate_item_table, "primary_weight_raw", suite_balance=True)
    role_balanced_weights = role_balanced_item_weights(
        aggregate_item_table, "primary_weight_raw", aggregate_task_suite_mass.value
    )
    optimize_items = aggregate_item_table[aggregate_item_table["recommended_role"].eq("optimize")].copy()
    optimize_only_weights = normalized_item_weights(optimize_items, "primary_weight_raw", suite_balance=True)
    dsp_suite_gated_weights = normalized_item_weights(
        aggregate_item_table_dsp_suite_gated, "primary_weight_raw", suite_balance=True
    )
    dsp_items = aggregate_item_table[aggregate_item_table["passes_dsp_filter"]].copy()
    dsp_weights = normalized_item_weights(dsp_items, "dsp_weight_raw", suite_balance=True)
    unbalanced_dsp_weights = normalized_item_weights(dsp_items, "dsp_weight_raw", suite_balance=False)

    primary_score = weighted_score(aggregate_item_table, aggregate_primary_weights, aggregate_item_z)
    role_balanced_score = weighted_score(aggregate_item_table, role_balanced_weights, aggregate_item_z)
    optimize_only_score = weighted_score(optimize_items, optimize_only_weights, aggregate_item_z)
    dsp_suite_gated_score = weighted_score(
        aggregate_item_table_dsp_suite_gated, dsp_suite_gated_weights, aggregate_item_z
    )
    dsp_suite_score = weighted_score(dsp_items, dsp_weights, aggregate_item_z)
    dsp_unbalanced_score = weighted_score(dsp_items, unbalanced_dsp_weights, aggregate_item_z)
    clipped_primary_score = weighted_score(aggregate_item_table, aggregate_primary_weights, aggregate_item_clipped_z)
    clipped_role_balanced_score = weighted_score(aggregate_item_table, role_balanced_weights, aggregate_item_clipped_z)
    rank_primary_score = weighted_score(aggregate_item_table, aggregate_primary_weights, aggregate_item_rank_z)
    rank_role_balanced_score = weighted_score(aggregate_item_table, role_balanced_weights, aggregate_item_rank_z)

    suite_score_columns = {}
    for suite, suite_frame in aggregate_item_table.loc[aggregate_primary_weights.index].groupby("aggregate_suite"):
        suite_weights = aggregate_primary_weights.loc[suite_frame.index]
        suite_total = float(suite_weights.sum())
        if suite_total <= 0:
            continue
        local_weights = suite_weights / suite_total
        metric_order = suite_frame["metric"].astype(str).tolist()
        suite_score_columns[suite] = aggregate_item_z[metric_order].to_numpy(dtype=float) @ local_weights.to_numpy(
            dtype=float
        )

    suite_score_frame = pd.DataFrame(suite_score_columns, index=completed_signal.index)
    suite_factor_loadings = pd.DataFrame()
    factor_1 = pd.Series(np.nan, index=completed_signal.index)
    factor_blend_3 = pd.Series(np.nan, index=completed_signal.index)
    if suite_score_frame.shape[1] >= 2:
        suite_matrix = suite_score_frame.to_numpy(dtype=float)
        suite_std = np.nanstd(suite_matrix, axis=0, ddof=1)
        suite_keep = np.isfinite(suite_std) & (suite_std > 1e-12)
        if int(suite_keep.sum()) >= 2:
            suite_names = suite_score_frame.columns[suite_keep].tolist()
            suite_matrix = suite_matrix[:, suite_keep]
            suite_matrix = (suite_matrix - np.nanmean(suite_matrix, axis=0)) / np.nanstd(suite_matrix, axis=0, ddof=1)
            u, singular_values, vt = np.linalg.svd(suite_matrix, full_matrices=False)
            component_count = min(3, vt.shape[0])
            component_scores = {}
            loading_rows = []
            aligned_scores = []
            correlations = []
            for component_index in range(component_count):
                component_score = pd.Series(
                    u[:, component_index] * singular_values[component_index], index=completed_signal.index
                )
                corr = float(
                    pd.DataFrame({"primary": primary_score, "score": component_score}).corr(method="spearman").iloc[0, 1]
                )
                sign = -1.0 if np.isfinite(corr) and corr < 0 else 1.0
                component_score = sign * component_score
                corr = abs(corr) if np.isfinite(corr) else 0.0
                aligned_scores.append(component_score)
                correlations.append(corr)
                component_name = f"suite_factor_{component_index + 1}"
                component_scores[component_name] = component_score
                for suite, loading in zip(suite_names, sign * vt[component_index], strict=True):
                    loading_rows.append(
                        {
                            "component": component_name,
                            "suite": suite,
                            "loading": float(loading),
                            "singular_value": float(singular_values[component_index]),
                            "spearman_to_primary": corr,
                        }
                    )
            factor_1 = component_scores["suite_factor_1"]
            blend_weights = np.asarray(correlations, dtype=float)
            blend_weights = np.clip(blend_weights, 0.0, None)
            if float(blend_weights.sum()) <= 0:
                blend_weights = np.zeros(component_count, dtype=float)
                blend_weights[0] = 1.0
            else:
                blend_weights = blend_weights / blend_weights.sum()
            factor_blend_3 = sum(
                weight * component_series for weight, component_series in zip(blend_weights, aligned_scores, strict=True)
            )
            suite_factor_loadings = pd.DataFrame(loading_rows)

    aggregate_candidate_scores = completed_signal[["run_name", "registry_run_key"]].copy()
    aggregate_candidate_scores["suite_balanced_mean_no_dsp"] = primary_score
    aggregate_candidate_scores["role_balanced_mean_no_dsp"] = role_balanced_score
    aggregate_candidate_scores["optimize_only_suite_balanced"] = optimize_only_score
    aggregate_candidate_scores["suite_balanced_mean_dsp_suite_gated"] = dsp_suite_gated_score
    aggregate_candidate_scores["suite_balanced_mean_dsp_weighted"] = dsp_suite_score
    aggregate_candidate_scores["reliability_weighted_mean_unbalanced"] = dsp_unbalanced_score
    aggregate_candidate_scores["suite_balanced_clipped_mean_no_dsp"] = clipped_primary_score
    aggregate_candidate_scores["role_balanced_clipped_mean_no_dsp"] = clipped_role_balanced_score
    aggregate_candidate_scores["suite_balanced_rank_mean_no_dsp"] = rank_primary_score
    aggregate_candidate_scores["role_balanced_rank_mean_no_dsp"] = rank_role_balanced_score
    aggregate_candidate_scores["suite_factor_1"] = factor_1
    aggregate_candidate_scores["suite_factor_blend_3_diagnostic"] = factor_blend_3
    if not factor_scores.empty:
        factor_score_columns = [
            "item_factor_horn_balanced_projection",
            "item_factor_varimax_diagnostic",
            "item_factor_balance_penalized",
        ]
        aggregate_candidate_scores = aggregate_candidate_scores.merge(
            factor_scores[["run_name", "registry_run_key"] + factor_score_columns],
            on=["run_name", "registry_run_key"],
            how="left",
            validate="one_to_one",
        )
    if not factor_iteration_scores.empty:
        factor_iteration_score_columns = [
            column for column in factor_iteration_scores.columns if column not in {"run_name", "registry_run_key"}
        ]
        aggregate_candidate_scores = aggregate_candidate_scores.merge(
            factor_iteration_scores[["run_name", "registry_run_key"] + factor_iteration_score_columns],
            on=["run_name", "registry_run_key"],
            how="left",
            validate="one_to_one",
        )

    item_weight_parts = []
    for candidate_name, items, weights in [
        ("suite_balanced_mean_no_dsp", aggregate_item_table, aggregate_primary_weights),
        ("role_balanced_mean_no_dsp", aggregate_item_table, role_balanced_weights),
        ("optimize_only_suite_balanced", optimize_items, optimize_only_weights),
        ("suite_balanced_clipped_mean_no_dsp", aggregate_item_table, aggregate_primary_weights),
        ("role_balanced_clipped_mean_no_dsp", aggregate_item_table, role_balanced_weights),
        ("suite_balanced_rank_mean_no_dsp", aggregate_item_table, aggregate_primary_weights),
        ("role_balanced_rank_mean_no_dsp", aggregate_item_table, role_balanced_weights),
        (
            "suite_balanced_mean_dsp_suite_gated",
            aggregate_item_table_dsp_suite_gated,
            dsp_suite_gated_weights,
        ),
        ("suite_balanced_mean_dsp_weighted", dsp_items, dsp_weights),
        ("reliability_weighted_mean_unbalanced", dsp_items, unbalanced_dsp_weights),
    ]:
        if weights.empty:
            continue
        part = items.loc[
            weights.index,
            [
                "metric",
                "task",
                "suite",
                "aggregate_suite",
                "recommended_role",
                "signal_to_noise",
                "dsp_controllability_score",
            ],
        ].copy()
        part["candidate"] = candidate_name
        part["weight"] = weights.to_numpy(dtype=float)
        item_weight_parts.append(part)
    aggregate_candidate_item_weights = (
        pd.concat(item_weight_parts, ignore_index=True) if item_weight_parts else pd.DataFrame()
    )
    aggregate_candidate_suite_weights = (
        aggregate_candidate_item_weights.groupby(["candidate", "aggregate_suite"], as_index=False)["weight"]
        .sum()
        .rename(columns={"aggregate_suite": "suite"})
        if not aggregate_candidate_item_weights.empty
        else pd.DataFrame(columns=["candidate", "suite", "weight"])
    )
    aggregate_suite_item_summary = (
        aggregate_item_table.groupby("aggregate_suite", as_index=False)
        .agg(
            items=("metric", "count"),
            optimize_items=("recommended_role", lambda values: int((values == "optimize").sum())),
            stabilizer_items=("recommended_role", lambda values: int((values == "stabilizer").sum())),
            median_snr=("signal_to_noise", "median"),
            median_dsp_spearman=("dsp_controllability_score", "median"),
        )
        .rename(columns={"aggregate_suite": "suite"})
        .sort_values(["items", "median_snr"], ascending=[True, False])
    )
    suite_factor_summary = (
        suite_factor_loadings[["component", "singular_value", "spearman_to_primary"]]
        .drop_duplicates()
        .sort_values("component")
        if not suite_factor_loadings.empty
        else pd.DataFrame(columns=["component", "singular_value", "spearman_to_primary"])
    )
    return (
        aggregate_candidate_item_weights,
        aggregate_candidate_scores,
        aggregate_candidate_suite_weights,
        aggregate_suite_item_summary,
        suite_factor_loadings,
        suite_factor_summary,
        suite_score_frame,
    )


@app.cell
def _(aggregate_candidate_scores, aggregate_candidate_suite_weights, completed_signal, np, pd):
    aggregate_score_columns = [
        column for column in aggregate_candidate_scores.columns if column not in {"run_name", "registry_run_key"}
    ]
    proportional_mask = aggregate_candidate_scores["run_name"].eq("baseline_proportional")
    summary_rows = []
    for candidate in aggregate_score_columns:
        scores = pd.to_numeric(aggregate_candidate_scores[candidate], errors="coerce")
        valid = scores.notna()
        if valid.sum() == 0:
            continue
        top_index = scores.idxmax()
        proportional_score = float(scores[proportional_mask].iloc[0]) if proportional_mask.any() else np.nan
        ranks = scores.rank(ascending=False, method="min")
        summary_rows.append(
            {
                "candidate": candidate,
                "row_count": int(valid.sum()),
                "top_run_name": aggregate_candidate_scores.loc[top_index, "run_name"],
                "top_score": float(scores.loc[top_index]),
                "proportional_score": proportional_score,
                "top_minus_proportional": (
                    float(scores.loc[top_index] - proportional_score) if np.isfinite(proportional_score) else np.nan
                ),
                "proportional_rank": int(ranks[proportional_mask].iloc[0]) if proportional_mask.any() else np.nan,
                "score_std": float(scores.std(ddof=1)),
            }
        )
    aggregate_candidate_summary = pd.DataFrame(summary_rows)

    score_values = aggregate_candidate_scores[aggregate_score_columns].apply(pd.to_numeric, errors="coerce")
    aggregate_candidate_correlations = (
        score_values.corr(method="spearman").reset_index().rename(columns={"index": "candidate"})
    )
    aggregate_candidate_leaderboard = (
        aggregate_candidate_scores[["run_name", "registry_run_key"] + aggregate_score_columns]
        .assign(primary_rank=lambda frame: frame["suite_balanced_mean_no_dsp"].rank(ascending=False, method="min"))
        .sort_values("suite_balanced_mean_no_dsp", ascending=False)
        .head(25)
    )
    suite_count_summary = (
        aggregate_candidate_suite_weights.groupby("candidate", as_index=False)
        .agg(suites=("suite", "nunique"), max_suite_weight=("weight", "max"))
        .sort_values("candidate")
    )
    return (
        aggregate_candidate_correlations,
        aggregate_candidate_leaderboard,
        aggregate_candidate_summary,
        aggregate_score_columns,
        suite_count_summary,
    )


@app.cell
def _(
    aggregate_candidate_correlations,
    aggregate_candidate_item_weights,
    aggregate_candidate_leaderboard,
    aggregate_candidate_scores,
    aggregate_candidate_suite_weights,
    aggregate_candidate_summary,
    aggregate_item_table,
    aggregate_score_columns,
    mo,
    px,
    aggregate_suite_item_summary,
    suite_count_summary,
    suite_factor_loadings,
    suite_factor_summary,
):
    aggregate_correlation_matrix = aggregate_candidate_correlations.set_index("candidate")
    aggregate_correlation_plot = px.imshow(
        aggregate_correlation_matrix,
        zmin=-1.0,
        zmax=1.0,
        color_continuous_scale="RdYlGn_r",
        text_auto=".2f",
        title="Spearman correlation among candidate aggregates",
    )
    aggregate_correlation_plot.update_layout(height=520)
    primary_leaderboard = aggregate_candidate_leaderboard[["run_name", "suite_balanced_mean_no_dsp"]].copy()
    primary_leaderboard = primary_leaderboard.sort_values("suite_balanced_mean_no_dsp", ascending=True)
    primary_leaderboard_plot = px.bar(
        primary_leaderboard,
        x="suite_balanced_mean_no_dsp",
        y="run_name",
        orientation="h",
        title="Top observed rows by primary suite-balanced aggregate",
        labels={"suite_balanced_mean_no_dsp": "primary aggregate score", "run_name": "run"},
    )
    primary_leaderboard_plot.update_layout(height=720)
    suite_weight_plot = px.bar(
        aggregate_candidate_suite_weights,
        x="candidate",
        y="weight",
        color="suite",
        title="Effective suite weights by aggregate candidate",
    )
    suite_weight_plot.update_layout(height=620, xaxis_tickangle=-25)

    mo.vstack(
        [
            mo.md(
                f"""
                ## Candidate Aggregate Metrics

                Selected smooth aggregate items: `{len(aggregate_item_table)}`.

                Primary candidate: `suite_balanced_mean_no_dsp`. It uses role,
                SNR, and optional MoE R² weights, but no DSP-Spearman weight.
                This keeps later DSP fit quality from becoming circular.

                DSP-weighted candidates are diagnostics. Suite-factor candidates
                are computed from suite-level scores, not raw item-level PCA, so
                large suites cannot dominate purely by item count.
                """
            ),
            mo.md("### Candidate summary"),
            mo.ui.table(aggregate_candidate_summary, page_size=10),
            mo.md("### Suite count summary"),
            mo.ui.table(suite_count_summary, page_size=10),
            mo.md("### Selected item count by suite"),
            mo.ui.table(aggregate_suite_item_summary, page_size=20),
            mo.ui.plotly(aggregate_correlation_plot),
            mo.ui.plotly(primary_leaderboard_plot),
            mo.ui.plotly(suite_weight_plot),
            mo.md("### Top rows by primary aggregate"),
            mo.ui.table(aggregate_candidate_leaderboard, page_size=25),
            mo.md("### Aggregate item weights"),
            mo.ui.table(aggregate_candidate_item_weights, page_size=25),
            mo.md("### Suite factor summary"),
            mo.ui.table(suite_factor_summary, page_size=10),
            mo.md("### Suite factor loadings"),
            mo.ui.table(suite_factor_loadings, page_size=25),
        ]
    )
    return


@app.cell
def _(OUTPUT_DIR, aggregate_candidate_summary, mo, pd, px):
    dsp_candidate_summary_path = OUTPUT_DIR / "aggregate_candidate_effective_exposure_dsp_summary.csv"
    dsp_raw_weights_path = OUTPUT_DIR / "aggregate_candidate_effective_exposure_dsp_raw_optimum_weights.csv"
    if not dsp_candidate_summary_path.exists() or not dsp_raw_weights_path.exists():
        mo.md(
            """
            ## Aggregate Candidate DSP Fits

            No cached effective-exposure DSP aggregate-candidate fit is present yet.
            Generate it with:

            `uv run --with numpy --with pandas --with scipy --with scikit-learn python experiments/domain_phase_mix/exploratory/two_phase_many/fit_effective_exposure_dsp_aggregate_candidates_300m.py --workers 16`
            """
        )
        aggregate_candidate_dsp_summary = pd.DataFrame()
        aggregate_candidate_dsp_raw_weights = pd.DataFrame()
    else:
        aggregate_candidate_dsp_summary = pd.read_csv(dsp_candidate_summary_path)
        aggregate_candidate_dsp_raw_weights = pd.read_csv(dsp_raw_weights_path)
        current_scores = aggregate_candidate_summary[
            ["candidate", "top_score", "proportional_score", "score_std"]
        ].rename(
            columns={
                "top_score": "current_best_observed_aggregate_score",
                "proportional_score": "current_proportional_aggregate_score",
                "score_std": "current_score_std",
            }
        )
        aggregate_candidate_dsp_summary = aggregate_candidate_dsp_summary.merge(
            current_scores, on="candidate", how="left"
        )
        current_candidate_set = set(aggregate_candidate_summary["candidate"].astype(str))
        cached_candidate_set = set(aggregate_candidate_dsp_summary["candidate"].astype(str))
        missing_dsp_candidates = sorted(current_candidate_set - cached_candidate_set)
        extra_dsp_candidates = sorted(cached_candidate_set - current_candidate_set)
        aggregate_candidate_dsp_summary = aggregate_candidate_dsp_summary.assign(
            phase_max_weight=lambda frame: frame[["phase0_max_weight", "phase1_max_weight"]].max(axis=1),
            raw_optimum_sparse=lambda frame: (
                (frame["phase0_max_weight"] >= 0.5)
                | (frame["phase1_max_weight"] >= 0.5)
                | (frame["raw_nearest_observed_tv"] >= 0.5)
            ),
            raw_predicted_gain_vs_observed_best=lambda frame: (
                frame["raw_predicted_aggregate_score"] - frame["best_observed_aggregate_score"]
            ),
            cache_best_score_abs_delta=lambda frame: (
                frame["current_best_observed_aggregate_score"] - frame["best_observed_aggregate_score"]
            ).abs(),
            cache_proportional_score_abs_delta=lambda frame: (
                frame["current_proportional_aggregate_score"] - frame["proportional_aggregate_score"]
            ).abs(),
            cv_rmse_over_score_std=lambda frame: frame["cv_rmse"] / frame["current_score_std"],
        )
        aggregate_candidate_dsp_summary["cache_matches_current_aggregate"] = (
            aggregate_candidate_dsp_summary["cache_best_score_abs_delta"] <= 1e-8
        ) & (aggregate_candidate_dsp_summary["cache_proportional_score_abs_delta"] <= 1e-8)
        stale_cache_count = int((~aggregate_candidate_dsp_summary["cache_matches_current_aggregate"]).sum())
        display_columns = [
            "candidate",
            "oof_spearman",
            "oof_pearson",
            "cv_rmse",
            "cv_rmse_over_score_std",
            "cv_regret_at_1",
            "lower_tail_optimism",
            "best_observed_run_name",
            "best_observed_aggregate_score",
            "proportional_aggregate_score",
            "raw_predicted_aggregate_score",
            "raw_predicted_gain_vs_observed_best",
            "raw_nearest_observed_tv",
            "raw_nearest_observed_run_name",
            "phase0_max_weight",
            "phase1_max_weight",
            "phase_max_weight",
            "raw_phase0_support_gt_1e3",
            "raw_phase1_support_gt_1e3",
            "raw_optimum_sparse",
            "cache_matches_current_aggregate",
            "cache_best_score_abs_delta",
            "cache_proportional_score_abs_delta",
        ]
        dsp_fit_plot = px.scatter(
            aggregate_candidate_dsp_summary,
            x="oof_spearman",
            y="raw_nearest_observed_tv",
            color="phase_max_weight",
            hover_name="candidate",
            hover_data=["raw_optimum_sparse", "raw_predicted_gain_vs_observed_best"],
            color_continuous_scale="RdYlGn_r",
            title="Aggregate-candidate DSP fit quality vs raw-optimum extrapolation",
            labels={
                "oof_spearman": "OOF Spearman vs aggregate target",
                "raw_nearest_observed_tv": "raw optimum nearest-observed TV",
                "phase_max_weight": "largest phase-domain weight",
                "raw_optimum_sparse": "sparse/extrapolative",
                "raw_predicted_gain_vs_observed_best": "predicted gain over observed best",
            },
        )
        dsp_fit_plot.update_layout(height=520)

        raw_weight_rows = []
        for row in aggregate_candidate_dsp_summary.itertuples(index=False):
            candidate_weights = aggregate_candidate_dsp_raw_weights[
                aggregate_candidate_dsp_raw_weights["candidate"].eq(row.candidate)
            ]
            for phase_column in ["phase_0_weight", "phase_1_weight"]:
                top_weights = candidate_weights.sort_values(phase_column, ascending=False).head(8)
                raw_weight_rows.append(
                    {
                        "candidate": row.candidate,
                        "phase": phase_column.replace("_weight", ""),
                        "top_domains": ", ".join(
                            f"{weight_row.domain_name}={getattr(weight_row, phase_column):.3f}"
                            for weight_row in top_weights.itertuples(index=False)
                        ),
                    }
                )
        aggregate_candidate_dsp_top_raw_domains = pd.DataFrame(raw_weight_rows)
        mo.vstack(
            [
                mo.md(
                    f"""
                    ## Aggregate Candidate DSP Fits

                    This is a second-stage check: after constructing each aggregate
                    target from observed rows, fit canonical effective-exposure DSP
                    to that target and optimize the fitted surface. Good OOF rank
                    fit alone is insufficient; raw optima are marked sparse when
                    they are far from the observed manifold or concentrate a phase
                    on one domain.

                    Cached DSP rows that do not match the currently reactive
                    aggregate scores are stale and should not be interpreted.

                    Stale cached candidates: `{stale_cache_count}`.

                    Missing current candidates in DSP cache: `{", ".join(missing_dsp_candidates) or "none"}`.
                    Extra stale candidates in DSP cache: `{", ".join(extra_dsp_candidates) or "none"}`.
                    """
                ),
                mo.ui.plotly(dsp_fit_plot),
                mo.md("### DSP fit and raw-optimum diagnostics"),
                mo.ui.table(
                    aggregate_candidate_dsp_summary[display_columns].sort_values("oof_spearman", ascending=False),
                    page_size=10,
                ),
                mo.md("### Top raw-optimum domains by phase"),
                mo.ui.table(aggregate_candidate_dsp_top_raw_domains, page_size=20),
            ]
        )
    return aggregate_candidate_dsp_raw_weights, aggregate_candidate_dsp_summary


@app.cell
def _(aggregate_candidate_dsp_raw_weights, aggregate_candidate_dsp_summary, mo, pd, px):
    if aggregate_candidate_dsp_raw_weights.empty or aggregate_candidate_dsp_summary.empty:
        aggregate_optimum_mixture_view = mo.md(
            "### Predicted Optimum Mixtures\n\nNo cached raw optimum weights are available yet."
        )
        aggregate_optimum_mixture_long = pd.DataFrame()
        aggregate_optimum_candidate_selector = None
        aggregate_optimum_domain_order = []
    else:
        candidate_order = (
            aggregate_candidate_dsp_summary.sort_values("oof_spearman", ascending=False)["candidate"]
            .astype(str)
            .tolist()
        )
        sparse_candidates = set(
            aggregate_candidate_dsp_summary.loc[
                aggregate_candidate_dsp_summary["raw_optimum_sparse"], "candidate"
            ].astype(str)
        )
        candidate_display = {
            candidate: f"{candidate} [SPARSE RAW]" if candidate in sparse_candidates else candidate
            for candidate in candidate_order
        }
        domain_order = sorted(aggregate_candidate_dsp_raw_weights["domain_name"].astype(str).unique())
        phase_frames = []
        for phase_name, weight_column, epoch_column in [
            ("phase 0", "phase_0_weight", "phase_0_effective_epochs"),
            ("phase 1", "phase_1_weight", "phase_1_effective_epochs"),
        ]:
            phase_frame = aggregate_candidate_dsp_raw_weights[
                ["candidate", "domain_name", weight_column, epoch_column]
            ].copy()
            phase_frame = phase_frame.rename(columns={weight_column: "weight", epoch_column: "effective_epochs"})
            phase_frame["phase"] = phase_name
            phase_frames.append(phase_frame)
        aggregate_optimum_mixture_long = pd.concat(phase_frames, ignore_index=True)
        aggregate_optimum_mixture_long["candidate"] = pd.Categorical(
            aggregate_optimum_mixture_long["candidate"], categories=candidate_order, ordered=True
        )
        aggregate_optimum_mixture_long["domain_name"] = pd.Categorical(
            aggregate_optimum_mixture_long["domain_name"], categories=domain_order, ordered=True
        )
        aggregate_optimum_mixture_long["candidate_phase"] = (
            aggregate_optimum_mixture_long["candidate"].astype(str).map(candidate_display)
            + " / "
            + aggregate_optimum_mixture_long["phase"].astype(str)
        )
        heatmap_frame = aggregate_optimum_mixture_long.copy()
        heatmap_frame["candidate_phase"] = pd.Categorical(
            heatmap_frame["candidate_phase"],
            categories=[
                f"{candidate_display[candidate]} / {phase}"
                for candidate in candidate_order
                for phase in ["phase 0", "phase 1"]
            ],
            ordered=True,
        )
        heatmap_matrix = heatmap_frame.pivot_table(
            index="candidate_phase",
            columns="domain_name",
            values="weight",
            aggfunc="mean",
            fill_value=0.0,
            observed=False,
        )
        optimum_heatmap = px.imshow(
            heatmap_matrix,
            aspect="auto",
            color_continuous_scale="RdYlGn_r",
            title="Raw DSP predicted optima: full domain weights by candidate and phase",
            labels={"x": "Domain", "y": "Candidate / phase", "color": "Weight"},
        )
        optimum_heatmap.update_layout(
            height=max(760, 80 + 28 * heatmap_matrix.shape[0]),
            xaxis={"tickangle": -45},
        )

        default_candidate = "suite_balanced_mean_no_dsp"
        if default_candidate not in candidate_order:
            default_candidate = candidate_order[0]
        aggregate_optimum_candidate_selector = mo.ui.dropdown(
            options=candidate_order,
            value=default_candidate,
            label="Raw optimum candidate for full labeled phase plot",
        )
        aggregate_optimum_mixture_view = mo.vstack(
            [
                mo.md(
                    """
                    ## Predicted Optimum Mixtures

                    These are raw unconstrained optima from the effective-exposure
                    DSP fit to each aggregate candidate. The heatmap shows every
                    domain for every candidate and phase. The selected-candidate
                    bar chart labels each domain with `weight / effective epochs`.

                    Interpret these as geometry diagnostics, not deployable
                    mixtures, when nearest-observed TV is high or one phase
                    collapses onto a small number of domains.
                    """
                ),
                mo.ui.plotly(optimum_heatmap),
                aggregate_optimum_candidate_selector,
            ]
        )
        aggregate_optimum_domain_order = domain_order
    mo.output.replace(aggregate_optimum_mixture_view)
    return aggregate_optimum_candidate_selector, aggregate_optimum_domain_order, aggregate_optimum_mixture_long


@app.cell
def _(
    aggregate_candidate_dsp_summary,
    aggregate_optimum_candidate_selector,
    aggregate_optimum_domain_order,
    aggregate_optimum_mixture_long,
    mo,
    px,
):
    if aggregate_optimum_candidate_selector is None or aggregate_optimum_mixture_long.empty:
        selected_optimum_mixture_view = mo.md(
            "### Selected Predicted Optimum Mixture\n\nNo raw optimum mixture is available."
        )
    else:
        selected_candidate = str(aggregate_optimum_candidate_selector.value)
        selected_frame = aggregate_optimum_mixture_long[
            aggregate_optimum_mixture_long["candidate"].astype(str).eq(selected_candidate)
        ].copy()
        selected_summary = aggregate_candidate_dsp_summary[
            aggregate_candidate_dsp_summary["candidate"].astype(str).eq(selected_candidate)
        ]
        selected_sparse = bool(selected_summary["raw_optimum_sparse"].iloc[0]) if not selected_summary.empty else False
        title_suffix = " [SPARSE RAW OPTIMUM]" if selected_sparse else ""
        selected_frame["end_label"] = selected_frame.apply(
            lambda row: f"{row['weight']:.3f} / {row['effective_epochs']:.2f} ep", axis=1
        )
        selected_frame = selected_frame.sort_values(["phase", "domain_name"], ascending=[True, True])
        selected_bar = px.bar(
            selected_frame,
            x="weight",
            y="domain_name",
            color="weight",
            text="end_label",
            facet_col="phase",
            orientation="h",
            color_continuous_scale="RdYlGn_r",
            title=f"Raw DSP predicted optimum mixture: {selected_candidate}{title_suffix}",
            labels={
                "weight": "Mixture weight",
                "domain_name": "Domain",
                "end_label": "weight / effective epochs",
            },
        )
        selected_bar.update_yaxes(
            categoryorder="array",
            categoryarray=list(reversed(aggregate_optimum_domain_order)),
            matches=None,
        )
        selected_bar.update_traces(textposition="outside", cliponaxis=False)
        selected_bar.update_layout(height=1250, margin={"r": 180}, showlegend=False)
        selected_optimum_mixture_view = mo.ui.plotly(selected_bar)
    mo.output.replace(selected_optimum_mixture_view)
    return


@app.cell
def _(
    aggregate_candidate_item_weights,
    canonical_task_alias,
    completed_signal,
    mo,
    moe_loss_metrics,
    np,
    pd,
    px,
):
    moe_primary_weights = aggregate_candidate_item_weights[
        aggregate_candidate_item_weights["candidate"].eq("suite_balanced_mean_no_dsp")
    ].copy()
    moe_primary_weights["task_alias"] = moe_primary_weights["task"].map(canonical_task_alias)
    moe_aliases = set(moe_loss_metrics["task_alias"].astype(str))
    overlap_weights = moe_primary_weights[moe_primary_weights["task_alias"].isin(moe_aliases)].copy()
    if overlap_weights.empty:
        moe_aggregate_scores = pd.DataFrame()
        mo.md("### MoE overlap diagnostic\n\nNo primary aggregate items overlap the MoE dashboard task aliases.")
    else:
        alias_weights = (
            overlap_weights.groupby(["task_alias", "aggregate_suite"], as_index=False)["weight"]
            .sum()
            .rename(columns={"aggregate_suite": "suite"})
        )
        alias_weights["weight"] = alias_weights["weight"] / alias_weights["weight"].sum()
        moe_overlap_suite_summary = (
            alias_weights.groupby("suite", as_index=False)
            .agg(overlapping_tasks=("task_alias", "nunique"), overlap_weight=("weight", "sum"))
            .sort_values("overlap_weight", ascending=False)
        )
        moe_frame = moe_loss_metrics.merge(alias_weights, on="task_alias", how="inner")
        moe_frame["oriented_loss_value"] = -pd.to_numeric(moe_frame["loss_value"], errors="coerce")
        moe_frame["task_z"] = moe_frame.groupby("task_alias")["oriented_loss_value"].transform(
            lambda values: (values - values.mean()) / values.std(ddof=1) if values.std(ddof=1) > 0 else np.nan
        )
        moe_frame["weighted_task_z"] = moe_frame["task_z"] * moe_frame["weight"]
        moe_aggregate_scores = (
            moe_frame.groupby(["track", "track_label", "hidden_dim", "budget", "scale_label"], as_index=False)
            .agg(
                moe_overlap_aggregate=("weighted_task_z", "sum"),
                overlapping_tasks=("task_alias", "nunique"),
            )
            .sort_values(["hidden_dim", "track_label"])
        )
        moe_overlap_plot = px.line(
            moe_aggregate_scores,
            x="budget",
            y="moe_overlap_aggregate",
            color="track_label",
            markers=True,
            log_x=True,
            title="MoE overlap diagnostic for primary aggregate task subset",
            labels={
                "budget": "training FLOPs",
                "moe_overlap_aggregate": "overlap aggregate score, z units",
                "track_label": "track",
            },
        )
        moe_overlap_plot.update_layout(height=560)
        mo.vstack(
            [
                mo.md(
                    f"""
                    ### MoE Overlap Diagnostic

                    This is not a full validation of the aggregate: it only uses
                    the `{len(alias_weights)}` primary-aggregate task aliases
                    that overlap the MoE dashboard. David/raw-PPL items are
                    absent from the MoE tracks.
                    """
                ),
                mo.ui.plotly(moe_overlap_plot),
                mo.ui.table(moe_aggregate_scores, page_size=20),
                mo.md("### MoE overlap task weights"),
                mo.ui.table(alias_weights.sort_values("weight", ascending=False), page_size=20),
                mo.md("### MoE overlap suite distribution"),
                mo.ui.table(moe_overlap_suite_summary, page_size=20),
            ]
        )
    if overlap_weights.empty:
        moe_overlap_suite_summary = pd.DataFrame(columns=["suite", "overlapping_tasks", "overlap_weight"])
    return moe_aggregate_scores, moe_overlap_suite_summary


@app.cell
def _(best_snr_by_task, canonical_task_alias, r2_task_summary, snr_points):
    snr_by_task = best_snr_by_task(snr_points)
    r2_snr_join = r2_task_summary.copy()
    r2_snr_join["canonical_task_alias"] = r2_snr_join["task_alias"].map(canonical_task_alias)
    r2_snr_join = r2_snr_join.merge(
        snr_by_task[
            [
                "canonical_task_alias",
                "metric",
                "signal_to_noise",
                "family",
                "source_class",
                "metric_leaf",
            ]
        ],
        on="canonical_task_alias",
        how="left",
        suffixes=("", "_snr"),
    )
    r2_snr_join["has_snr_match"] = r2_snr_join["signal_to_noise"].notna()
    r2_snr_join["candidate_quality_score"] = r2_snr_join["mean_loglog_r2"] * r2_snr_join["signal_to_noise"].clip(lower=0)
    r2_snr_join = r2_snr_join.sort_values("candidate_quality_score", ascending=False)
    return (r2_snr_join,)


@app.cell
def _(px, r2_snr_join):
    r2_snr_scatter = px.scatter(
        r2_snr_join,
        x="signal_to_noise",
        y="mean_loglog_r2",
        color="task_group",
        symbol="loss_metric",
        hover_data=[
            "task_alias",
            "metric",
            "family",
            "source_class",
            "metric_leaf",
            "min_loglog_r2",
            "candidate_quality_score",
        ],
        title="MoE task metrics: 300M SNR vs MoE scaling R²",
        labels={
            "signal_to_noise": "best matched 300M SNR",
            "mean_loglog_r2": "mean MoE log-log R²",
            "task_group": "task group",
            "loss_metric": "MoE loss-like metric",
        },
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    r2_snr_scatter.add_hline(y=0.8, line_dash="dot", line_color="#777", annotation_text="R²=0.8")
    r2_snr_scatter.add_vline(x=2.0, line_dash="dot", line_color="#777", annotation_text="SNR=2")
    r2_snr_scatter.update_layout(height=620)
    return (r2_snr_scatter,)


@app.cell
def _(mo, r2_snr_join, r2_snr_scatter):
    joined_table = r2_snr_join[
        [
            "task_alias",
            "task_group",
            "loss_metric",
            "mean_loglog_r2",
            "min_loglog_r2",
            "signal_to_noise",
            "metric",
            "candidate_quality_score",
        ]
    ].head(15)
    missing_match = r2_snr_join.loc[~r2_snr_join["has_snr_match"], ["task_alias", "task_group", "loss_metric"]]
    mo.vstack(
        [
            mo.md(
                """
            ## Joint SNR × Scaling Diagnostic

            The scatter joins MoE scaling R² to the best current 300M SNR metric
            for the same task when an obvious alias exists. This is intentionally
            conservative; unmatched rows are kept visible below rather than
            fuzzy-matched silently.
            """
            ),
            mo.ui.plotly(r2_snr_scatter),
            mo.md("### Highest SNR × R² candidates among matched MoE tasks"),
            mo.ui.table(joined_table, page_size=15),
            mo.md("### MoE task aliases without a direct SNR-summary match"),
            mo.ui.table(missing_match, page_size=10),
        ]
    )
    return


@app.cell
def _(px, snr_points):
    raw_ppl_points = snr_points[snr_points["metric"].str.startswith("raw_ppl/")].copy()
    raw_ppl_points = raw_ppl_points.sort_values("signal_to_noise", ascending=False)
    raw_ppl_rank_plot = px.scatter(
        raw_ppl_points.assign(rank=lambda frame: range(1, len(frame) + 1)),
        x="rank",
        y="signal_to_noise",
        color="family",
        hover_data=["metric", "task", "metric_leaf", "signal_n", "noise_n", "signal_range", "noise_range"],
        title="David/#5005 raw-PPL metrics: 300M variable-subset SNR",
        labels={"rank": "raw-PPL metric rank by SNR", "signal_to_noise": "SNR", "family": "raw-PPL family"},
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    raw_ppl_rank_plot.add_hline(y=2.0, line_dash="dash", line_color="#444", annotation_text="SNR=2")
    raw_ppl_rank_plot.add_hline(y=1.0, line_dash="dot", line_color="#777", annotation_text="SNR=1")
    raw_ppl_rank_plot.update_layout(height=560)
    return raw_ppl_points, raw_ppl_rank_plot


@app.cell
def _(mo, raw_ppl_points, raw_ppl_rank_plot):
    raw_ppl_top = raw_ppl_points[
        ["metric", "family", "metric_leaf", "signal_to_noise", "signal_range", "noise_range"]
    ].head(15)
    mo.vstack(
        [
            mo.md(
                """
            ## David/#5005 Raw-PPL Metrics

            These are now available in the 300M raw matrix and SNR plot. They
            are strong candidates for smooth aggregate construction because many
            have SNR above 2. However, they do not yet have MoE scaling-track R²
            evidence unless the same raw-PPL evals have been run on those MoE
            checkpoints.
            """
            ),
            mo.ui.plotly(raw_ppl_rank_plot),
            mo.md("### Highest-SNR raw-PPL items"),
            mo.ui.table(raw_ppl_top, page_size=15),
        ]
    )
    return


@app.cell
def _(
    OUTPUT_DIR,
    aggregate_candidate_correlations,
    aggregate_candidate_item_weights,
    aggregate_candidate_scores,
    aggregate_candidate_suite_weights,
    aggregate_candidate_summary,
    aggregate_item_table,
    aggregate_suite_item_summary,
    controllability_table_with_tags,
    dsp_comparison_summary,
    factor_horn_parallel,
    factor_iteration_loadings,
    factor_iteration_scores,
    factor_iteration_summary,
    factor_item_table,
    factor_loadings,
    factor_scores,
    factor_summary,
    metric_best_by_item,
    metric_role_summary,
    metric_table,
    moe_aggregate_scores,
    moe_overlap_suite_summary,
    r2_snr_join,
    r2_task_summary,
    raw_ppl_points,
    suite_factor_loadings,
    suite_factor_summary,
):
    r2_task_summary.to_csv(OUTPUT_DIR / "moe_task_scaling_r2_summary.csv", index=False)
    r2_snr_join.to_csv(OUTPUT_DIR / "moe_task_snr_r2_join.csv", index=False)
    raw_ppl_points.to_csv(OUTPUT_DIR / "raw_ppl_snr_points.csv", index=False)
    controllability_table_with_tags.to_csv(OUTPUT_DIR / "metric_controllability_oof_ridge.csv", index=False)
    if dsp_comparison_summary is not None:
        dsp_comparison_summary.to_csv(
            OUTPUT_DIR / "metric_controllability_effective_exposure_dsp_role_summary.csv", index=False
        )
    metric_table.to_csv(OUTPUT_DIR / "reactive_metric_table.csv", index=False)
    metric_best_by_item.to_csv(OUTPUT_DIR / "reactive_metric_table_best_by_item.csv", index=False)
    metric_role_summary.to_csv(OUTPUT_DIR / "reactive_metric_table_role_summary.csv", index=False)
    aggregate_item_table.to_csv(OUTPUT_DIR / "aggregate_item_table.csv", index=False)
    aggregate_candidate_scores.to_csv(OUTPUT_DIR / "aggregate_candidate_scores.csv", index=False)
    aggregate_candidate_summary.to_csv(OUTPUT_DIR / "aggregate_candidate_summary.csv", index=False)
    aggregate_candidate_correlations.to_csv(OUTPUT_DIR / "aggregate_candidate_correlations.csv", index=False)
    aggregate_candidate_item_weights.to_csv(OUTPUT_DIR / "aggregate_candidate_item_weights.csv", index=False)
    aggregate_candidate_suite_weights.to_csv(OUTPUT_DIR / "aggregate_candidate_suite_weights.csv", index=False)
    aggregate_suite_item_summary.to_csv(OUTPUT_DIR / "aggregate_suite_item_summary.csv", index=False)
    suite_factor_summary.to_csv(OUTPUT_DIR / "aggregate_suite_factor_summary.csv", index=False)
    suite_factor_loadings.to_csv(OUTPUT_DIR / "aggregate_suite_factor_loadings.csv", index=False)
    factor_item_table.to_csv(OUTPUT_DIR / "aggregate_factor_item_table.csv", index=False)
    factor_loadings.to_csv(OUTPUT_DIR / "aggregate_factor_loadings.csv", index=False)
    factor_scores.to_csv(OUTPUT_DIR / "aggregate_factor_scores.csv", index=False)
    factor_summary.to_csv(OUTPUT_DIR / "aggregate_factor_summary.csv", index=False)
    factor_horn_parallel.to_csv(OUTPUT_DIR / "aggregate_factor_horn_parallel.csv", index=False)
    factor_iteration_scores.to_csv(OUTPUT_DIR / "aggregate_factor_iteration_scores.csv", index=False)
    factor_iteration_summary.to_csv(OUTPUT_DIR / "aggregate_factor_iteration_summary.csv", index=False)
    factor_iteration_loadings.to_csv(OUTPUT_DIR / "aggregate_factor_iteration_loadings.csv", index=False)
    moe_aggregate_scores.to_csv(OUTPUT_DIR / "aggregate_moe_overlap_scores.csv", index=False)
    moe_overlap_suite_summary.to_csv(OUTPUT_DIR / "aggregate_moe_overlap_suite_summary.csv", index=False)
    saved_outputs = {
        "moe_task_scaling_r2_summary": str(OUTPUT_DIR / "moe_task_scaling_r2_summary.csv"),
        "moe_task_snr_r2_join": str(OUTPUT_DIR / "moe_task_snr_r2_join.csv"),
        "raw_ppl_snr_points": str(OUTPUT_DIR / "raw_ppl_snr_points.csv"),
        "metric_controllability_oof_ridge": str(OUTPUT_DIR / "metric_controllability_oof_ridge.csv"),
        "metric_controllability_effective_exposure_dsp": str(
            OUTPUT_DIR / "metric_controllability_effective_exposure_dsp.csv"
        ),
        "metric_controllability_effective_exposure_dsp_role_summary": str(
            OUTPUT_DIR / "metric_controllability_effective_exposure_dsp_role_summary.csv"
        ),
        "reactive_metric_table": str(OUTPUT_DIR / "reactive_metric_table.csv"),
        "reactive_metric_table_best_by_item": str(OUTPUT_DIR / "reactive_metric_table_best_by_item.csv"),
        "reactive_metric_table_role_summary": str(OUTPUT_DIR / "reactive_metric_table_role_summary.csv"),
        "aggregate_item_table": str(OUTPUT_DIR / "aggregate_item_table.csv"),
        "aggregate_candidate_scores": str(OUTPUT_DIR / "aggregate_candidate_scores.csv"),
        "aggregate_candidate_summary": str(OUTPUT_DIR / "aggregate_candidate_summary.csv"),
        "aggregate_candidate_correlations": str(OUTPUT_DIR / "aggregate_candidate_correlations.csv"),
        "aggregate_candidate_item_weights": str(OUTPUT_DIR / "aggregate_candidate_item_weights.csv"),
        "aggregate_candidate_suite_weights": str(OUTPUT_DIR / "aggregate_candidate_suite_weights.csv"),
        "aggregate_suite_item_summary": str(OUTPUT_DIR / "aggregate_suite_item_summary.csv"),
        "aggregate_suite_factor_summary": str(OUTPUT_DIR / "aggregate_suite_factor_summary.csv"),
        "aggregate_suite_factor_loadings": str(OUTPUT_DIR / "aggregate_suite_factor_loadings.csv"),
        "aggregate_factor_item_table": str(OUTPUT_DIR / "aggregate_factor_item_table.csv"),
        "aggregate_factor_loadings": str(OUTPUT_DIR / "aggregate_factor_loadings.csv"),
        "aggregate_factor_scores": str(OUTPUT_DIR / "aggregate_factor_scores.csv"),
        "aggregate_factor_summary": str(OUTPUT_DIR / "aggregate_factor_summary.csv"),
        "aggregate_factor_horn_parallel": str(OUTPUT_DIR / "aggregate_factor_horn_parallel.csv"),
        "aggregate_factor_iteration_scores": str(OUTPUT_DIR / "aggregate_factor_iteration_scores.csv"),
        "aggregate_factor_iteration_summary": str(OUTPUT_DIR / "aggregate_factor_iteration_summary.csv"),
        "aggregate_factor_iteration_loadings": str(OUTPUT_DIR / "aggregate_factor_iteration_loadings.csv"),
        "aggregate_moe_overlap_scores": str(OUTPUT_DIR / "aggregate_moe_overlap_scores.csv"),
        "aggregate_moe_overlap_suite_summary": str(OUTPUT_DIR / "aggregate_moe_overlap_suite_summary.csv"),
        "aggregate_candidate_effective_exposure_dsp_summary": str(
            OUTPUT_DIR / "aggregate_candidate_effective_exposure_dsp_summary.csv"
        ),
        "aggregate_candidate_effective_exposure_dsp_raw_optimum_weights": str(
            OUTPUT_DIR / "aggregate_candidate_effective_exposure_dsp_raw_optimum_weights.csv"
        ),
    }
    return (saved_outputs,)


@app.cell
def _(mo, saved_outputs):
    mo.md(
        "\n".join(
            [
                "## Current Interpretation",
                "",
                "1. The raw-PPL additions are present in the 300M matrix and SNR summary; several look usable by SNR alone.",
                "2. MoE R² is currently only available for the MoE dashboard task metrics, not for all David raw-PPL metrics.",
                "3. Effective-exposure DSP improves the best-by-item controllability diagnostic over linear ridge overall, especially for stabilizer and optimize-role metrics.",
                "4. For aggregate construction, the first defensible filter is not just high SNR; prefer metrics that are high-SNR, DSP-controllable, and scale-predictable where scaling evidence exists.",
                "5. The next notebook step should fit candidate latent aggregates under multiple weighting schemes: SNR-only, SNR×R², suite-balanced, and factor-relevance-weighted.",
                "",
                "Saved diagnostic tables:",
                *[f"- `{name}`: `{path}`" for name, path in saved_outputs.items()],
            ]
        )
    )
    return


if __name__ == "__main__":
    app.run()
