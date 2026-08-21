# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "plotly",
# ]
# ///
"""Marimo notebook for MCQ smooth-proxy perturbation controllability."""

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

    return Path, go, mo, np, pd, px


@app.cell
def _(Path):
    TWO_PHASE_ROOT = Path(__file__).resolve().parent
    MATRIX_DIR = TWO_PHASE_ROOT / "metric_registry" / "raw_metric_matrix_300m"
    PPERT_DIR = TWO_PHASE_ROOT / "metric_registry" / "proportional_perturbation_scale_transfer"
    OUTPUT_DIR = TWO_PHASE_ROOT / "reference_outputs" / "ppert_mcq_smooth_controllability_20260518"
    IMG_DIR = OUTPUT_DIR / "img"
    MCQ_RESULTS_CSV = PPERT_DIR / "ppert_mcq_smooth_proxy_eval_results.csv"
    ENGLISH_LITE_RESULTS_CSV = PPERT_DIR / "ppert_english_lite_eval_results.csv"
    FIXED_NOISE_CSV = MATRIX_DIR / "noise_baseline_run00097_fixed_subset_300m.csv"
    VARIABLE_NOISE_CSV = MATRIX_DIR / "noise_baseline_run00097_variable_subset_300m.csv"
    PROPORTIONAL_NOISE_CSV = MATRIX_DIR / "noise_baseline_proportional_variable_subset_300m.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    return (
        ENGLISH_LITE_RESULTS_CSV,
        FIXED_NOISE_CSV,
        IMG_DIR,
        MCQ_RESULTS_CSV,
        OUTPUT_DIR,
        PROPORTIONAL_NOISE_CSV,
        VARIABLE_NOISE_CSV,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # 300M Proportional Perturbations: MCQ Smooth-Proxy Controllability

        This notebook uses the 300M/6B proportional perturbation experiment to
        debug whether MCQ smooth/logprob proxies are locally controllable under
        the current 39-domain partition. It combines the targeted custom
        `mcq_smooth/*` results with the English-lite `lm_eval/*` smooth metrics
        for BoolQ, COPA, CSQA, and WSC273.

        The domain-bump rows are **directional finite differences**, not pure
        unconstrained partial derivatives. Each intervention adds \(+0.05\)
        absolute mass to one domain and renormalizes the rest of the simplex,
        so the measured quantity is:

        \[
        \Delta_m(d) = m(w_{\mathrm{prop} \rightarrow d+0.05}) - m(w_{\mathrm{prop}})
        \]

        after sign orientation so positive values mean the smooth proxy
        improved. The main plots show noise-scaled effects, not slopes.
        The exported CSVs also include slope columns using intervention
        TV distance as the denominator; for the current domain and family
        bumps, this is \(0.05\). Noise-scaled effects prefer the proportional
        variable-subset noise baseline, since these perturbations are anchored
        at proportional. If that baseline is unavailable or incomplete, the
        notebook explicitly falls back to the `run_00097` fixed-subset noise
        baseline:

        \[
        z_m(d) = \frac{\Delta_m(d)}{\widehat{\sigma}_{m,\mathrm{primary}}}.
        \]

        Family bumps and quality swaps are shown separately because they are
        composed directions rather than one-domain bumps.

        `logprob`, `nll`, and `bpb` are often nearly collinear for the same
        task. The notebook keeps all metric leaves in the exported CSVs, but
        the headline plots avoid triple-counting by displaying `bpb` and
        omitting the redundant `logprob`/`nll` columns.
        """
    )
    return


@app.cell
def _():
    DISPLAY_METRIC_ORDER = [
        "choice_prob_norm",
        "choice_logprob_norm",
        "choice_logprob",
        "bpb",
    ]
    ALL_METRIC_ORDER = [
        "choice_prob_norm",
        "choice_logprob_norm",
        "choice_logprob",
        "choice_prob",
        "logprob",
        "bpb",
        "nll",
    ]
    TASK_ORDER = [
        "boolq_10shot",
        "copa_0shot",
        "csqa_5shot",
        "medmcqa_5shot",
        "sciq_5shot",
        "swag_0shot",
        "truthfulqa_mc1_0shot",
        "truthfulqa_mc2_0shot",
        "wsc273_0shot",
    ]
    TASK_SOURCE_PREFIX = {
        "boolq_10shot": "lm_eval",
        "copa_0shot": "lm_eval",
        "csqa_5shot": "lm_eval",
        "medmcqa_5shot": "mcq_smooth",
        "sciq_5shot": "mcq_smooth",
        "swag_0shot": "mcq_smooth",
        "truthfulqa_mc1_0shot": "mcq_smooth",
        "truthfulqa_mc2_0shot": "mcq_smooth",
        "wsc273_0shot": "lm_eval",
    }

    def metric_leaf(metric_name: str) -> str:
        return metric_name.split("/")[-1]

    def task_alias(metric_name: str) -> str:
        return metric_name.split("/")[1]

    def task_label(alias: str) -> str:
        return alias.replace("_5shot", "").replace("_0shot", "").replace("truthfulqa_", "tqa_").replace("_", " ")

    def oriented_direction(leaf: str) -> float:
        if leaf in {"bpb", "nll"}:
            return -1.0
        if leaf in {"choice_prob", "choice_prob_norm", "choice_logprob", "choice_logprob_norm", "logprob"}:
            return 1.0
        raise ValueError(f"Unexpected MCQ metric leaf: {leaf}")

    def domain_family(domain: str) -> str:
        if domain.startswith("dolma3_cc/"):
            return "dolma3_cc"
        if domain.startswith("dolmino_synth_"):
            return "dolmino_synth"
        if domain.startswith("dolmino_"):
            return "dolmino_other"
        if domain.startswith("dolma3_"):
            return "dolma3_other"
        return domain.split("/", maxsplit=1)[0]

    def short_domain(domain: str) -> str:
        return domain.replace("dolma3_cc/", "cc/").replace("dolma3_", "d3_").replace("dolmino_", "dm_")

    def spearman_corr(left, right) -> float:
        left_rank = left.rank(method="average")
        right_rank = right.rank(method="average")
        return float(left_rank.corr(right_rank))

    return (
        ALL_METRIC_ORDER,
        DISPLAY_METRIC_ORDER,
        TASK_SOURCE_PREFIX,
        TASK_ORDER,
        domain_family,
        metric_leaf,
        oriented_direction,
        short_domain,
        spearman_corr,
        task_alias,
        task_label,
    )


@app.cell
def _(
    ALL_METRIC_ORDER,
    ENGLISH_LITE_RESULTS_CSV,
    FIXED_NOISE_CSV,
    MCQ_RESULTS_CSV,
    PROPORTIONAL_NOISE_CSV,
    TASK_SOURCE_PREFIX,
    TASK_ORDER,
    VARIABLE_NOISE_CSV,
    metric_leaf,
    pd,
    task_alias,
):
    mcq_results = pd.read_csv(MCQ_RESULTS_CSV, low_memory=False)
    english_lite_results = pd.read_csv(ENGLISH_LITE_RESULTS_CSV, low_memory=False)
    fixed_noise = pd.read_csv(FIXED_NOISE_CSV, low_memory=False)
    variable_noise = pd.read_csv(VARIABLE_NOISE_CSV, low_memory=False)
    proportional_noise = pd.read_csv(PROPORTIONAL_NOISE_CSV, low_memory=False)

    metric_columns = [
        column
        for column in mcq_results.columns
        if column.startswith("mcq_smooth/")
        and metric_leaf(column) in set(ALL_METRIC_ORDER)
        and task_alias(column) in set(TASK_ORDER)
        and TASK_SOURCE_PREFIX[task_alias(column)] == "mcq_smooth"
    ]
    english_metric_columns = [
        column
        for column in english_lite_results.columns
        if column.startswith("lm_eval/")
        and metric_leaf(column) in set(ALL_METRIC_ORDER)
        and task_alias(column) in set(TASK_ORDER)
        and TASK_SOURCE_PREFIX[task_alias(column)] == "lm_eval"
    ]
    english_metric_columns = [column for column in english_metric_columns if column in fixed_noise.columns]
    combined_results = mcq_results.merge(
        english_lite_results[["panel", "run_name", *english_metric_columns]],
        on=["panel", "run_name"],
        how="left",
        validate="one_to_one",
    )
    metric_columns = sorted(
        [*metric_columns, *english_metric_columns],
        key=lambda column: (
            TASK_ORDER.index(task_alias(column)),
            ALL_METRIC_ORDER.index(metric_leaf(column)),
        ),
    )
    missing_metric_columns = [column for column in metric_columns if combined_results[column].isna().any()]
    if missing_metric_columns:
        raise ValueError(f"Merged perturbation results have missing metric columns: {missing_metric_columns}")

    baseline_rows = combined_results[combined_results["panel"].eq("proportional_baseline_anchor_300m_6b")].copy()
    if len(baseline_rows) != 1:
        raise ValueError(f"Expected one 300M proportional baseline anchor, found {len(baseline_rows)}")
    baseline_row = baseline_rows.iloc[0]
    perturbation_rows = combined_results[combined_results["panel"].eq("proportional_perturbation_300m_6b")].copy()
    if len(perturbation_rows) != 55:
        raise ValueError(f"Expected 55 300M perturbation rows, found {len(perturbation_rows)}")
    return (
        baseline_row,
        combined_results,
        fixed_noise,
        metric_columns,
        perturbation_rows,
        proportional_noise,
        variable_noise,
    )


@app.cell
def _(
    baseline_row,
    domain_family,
    fixed_noise,
    metric_columns,
    metric_leaf,
    np,
    oriented_direction,
    pd,
    perturbation_rows,
    proportional_noise,
    short_domain,
    task_alias,
    task_label,
    variable_noise,
):
    fixed_noise_std = fixed_noise[metric_columns].std(axis=0, ddof=1)
    variable_noise_std = variable_noise[metric_columns].std(axis=0, ddof=1)
    proportional_noise_has_metrics = (
        len(proportional_noise) == 10
        and set(metric_columns).issubset(proportional_noise.columns)
        and proportional_noise[metric_columns].notna().all().all()
    )
    if proportional_noise_has_metrics:
        primary_noise = proportional_noise
        primary_noise_kind = "proportional_variable_subset"
        primary_noise_label = "proportional variable-subset noise"
    else:
        primary_noise = fixed_noise
        primary_noise_kind = "run00097_fixed_subset_fallback"
        primary_noise_label = "run_00097 fixed-subset fallback; proportional variable-subset noise unavailable"
    primary_noise_std = primary_noise[metric_columns].std(axis=0, ddof=1)

    effect_rows = []
    for _, row in perturbation_rows.iterrows():
        row_dict = row.to_dict()
        target_domain = row_dict.get("target_domain")
        target_unit = row_dict.get("target_unit")
        intervention_type = row_dict["intervention_type"]
        tv_distance = float(row_dict["tv_distance"])
        target_mass_before = row_dict.get("target_mass_before")
        target_mass_after = row_dict.get("target_mass_after")
        target_mass_delta = (
            float(target_mass_after) - float(target_mass_before)
            if pd.notna(target_mass_before) and pd.notna(target_mass_after)
            else np.nan
        )
        if intervention_type in {"domain_bump", "family_bump"} and not np.isfinite(target_mass_delta):
            target_mass_delta = tv_distance
        for _metric in metric_columns:
            _leaf = metric_leaf(_metric)
            direction = oriented_direction(_leaf)
            raw_value = float(row_dict[_metric])
            baseline_value = float(baseline_row[_metric])
            raw_delta = raw_value - baseline_value
            improvement = direction * raw_delta
            primary_std = float(primary_noise_std[_metric])
            fixed_std = float(fixed_noise_std[_metric])
            variable_std = float(variable_noise_std[_metric])
            domain = str(target_domain) if intervention_type == "domain_bump" else str(target_unit)
            effect_rows.append(
                {
                    "intervention_id": row_dict["intervention_id"],
                    "intervention_type": intervention_type,
                    "target_unit": target_unit,
                    "target_domain": target_domain,
                    "target_family": row_dict.get("target_family"),
                    "domain_or_unit": domain,
                    "domain_label": short_domain(domain),
                    "domain_family": domain_family(domain) if intervention_type == "domain_bump" else intervention_type,
                    "task_alias": task_alias(_metric),
                    "task_label": task_label(task_alias(_metric)),
                    "metric": _metric,
                    "metric_leaf": _leaf,
                    "orientation": direction,
                    "baseline_value": baseline_value,
                    "raw_value": raw_value,
                    "raw_delta": raw_delta,
                    "improvement": improvement,
                    "tv_distance": tv_distance,
                    "target_mass_delta": target_mass_delta,
                    "slope_per_target_mass": (
                        improvement / target_mass_delta
                        if np.isfinite(target_mass_delta) and target_mass_delta != 0
                        else np.nan
                    ),
                    "slope_per_tv": improvement / tv_distance if tv_distance > 0 else np.nan,
                    "primary_noise_kind": primary_noise_kind,
                    "primary_noise_label": primary_noise_label,
                    "primary_noise_std": primary_std,
                    "fixed_noise_std": fixed_std,
                    "variable_noise_std": variable_std,
                    "z_primary": improvement / primary_std if primary_std > 0 else np.nan,
                    "z_fixed": improvement / fixed_std if fixed_std > 0 else np.nan,
                    "z_variable": improvement / variable_std if variable_std > 0 else np.nan,
                }
            )

    effects = pd.DataFrame(effect_rows)
    domain_effects = effects[effects["intervention_type"].eq("domain_bump")].copy()
    non_domain_effects = effects[~effects["intervention_type"].eq("domain_bump")].copy()
    return (
        domain_effects,
        effects,
        fixed_noise_std,
        non_domain_effects,
        primary_noise_label,
        primary_noise_std,
        variable_noise_std,
    )


@app.cell
def _(
    DISPLAY_METRIC_ORDER,
    OUTPUT_DIR,
    TASK_ORDER,
    domain_effects,
    effects,
    fixed_noise_std,
    np,
    pd,
    primary_noise_label,
    primary_noise_std,
    spearman_corr,
    variable_noise_std,
):
    summary_rows = []
    for (_task, _leaf), _group in domain_effects.groupby(["task_alias", "metric_leaf"], sort=False):
        abs_z = _group["z_primary"].abs()
        strongest = _group.iloc[int(abs_z.to_numpy().argmax())]
        summary_rows.append(
            {
                "task_alias": _task,
                "metric_leaf": _leaf,
                "domain_count": len(_group),
                "primary_noise_label": primary_noise_label,
                "primary_noise_std": float(primary_noise_std[_group["metric"].iloc[0]]),
                "fixed_noise_std": float(fixed_noise_std[_group["metric"].iloc[0]]),
                "variable_noise_std": float(variable_noise_std[_group["metric"].iloc[0]]),
                "median_abs_z_primary": float(abs_z.median()),
                "mean_abs_z_primary": float(abs_z.mean()),
                "max_abs_z_primary": float(abs_z.max()),
                "frac_abs_z_ge_2": float((abs_z >= 2.0).mean()),
                "frac_abs_z_ge_3": float((abs_z >= 3.0).mean()),
                "strongest_domain": strongest["target_domain"],
                "strongest_domain_z_primary": float(strongest["z_primary"]),
                "strongest_domain_improvement": float(strongest["improvement"]),
            }
        )
    controllability_summary = pd.DataFrame(summary_rows).sort_values(
        ["frac_abs_z_ge_2", "max_abs_z_primary"], ascending=[False, False]
    )

    potency = (
        domain_effects.assign(abs_z=lambda frame: frame["z_primary"].abs())
        .groupby(["target_domain", "domain_label", "domain_family"], as_index=False)
        .agg(
            rms_z=("z_primary", lambda values: float(np.sqrt(np.nanmean(np.square(values))))),
            mean_abs_z=("abs_z", "mean"),
            max_abs_z=("abs_z", "max"),
            positive_ge2=("z_primary", lambda values: int((values >= 2.0).sum())),
            negative_ge2=("z_primary", lambda values: int((values <= -2.0).sum())),
        )
        .sort_values("rms_z", ascending=False)
    )

    agreement_rows = []
    display_domain = domain_effects[domain_effects["metric_leaf"].isin(DISPLAY_METRIC_ORDER)].copy()
    for _task in TASK_ORDER:
        task_frame = display_domain[display_domain["task_alias"].eq(_task)]
        for _left in DISPLAY_METRIC_ORDER:
            for _right in DISPLAY_METRIC_ORDER:
                left_values = task_frame[task_frame["metric_leaf"].eq(_left)].set_index("target_domain")["z_primary"]
                right_values = task_frame[task_frame["metric_leaf"].eq(_right)].set_index("target_domain")["z_primary"]
                common = left_values.index.intersection(right_values.index)
                if len(common) >= 3:
                    corr = spearman_corr(left_values.loc[common], right_values.loc[common])
                else:
                    corr = np.nan
                agreement_rows.append(
                    {
                        "task_alias": _task,
                        "left_metric": _left,
                        "right_metric": _right,
                        "spearman_z_primary": corr,
                    }
                )
    metric_agreement = pd.DataFrame(agreement_rows)

    effects.to_csv(OUTPUT_DIR / "mcq_smooth_directional_effects.csv", index=False)
    domain_effects.to_csv(OUTPUT_DIR / "mcq_smooth_domain_directional_effects.csv", index=False)
    controllability_summary.to_csv(OUTPUT_DIR / "mcq_smooth_metric_controllability_summary.csv", index=False)
    potency.to_csv(OUTPUT_DIR / "mcq_smooth_domain_potency.csv", index=False)
    metric_agreement.to_csv(OUTPUT_DIR / "mcq_smooth_metric_agreement.csv", index=False)
    return controllability_summary, display_domain, metric_agreement, potency


@app.cell
def _(
    DISPLAY_METRIC_ORDER,
    IMG_DIR,
    TASK_ORDER,
    controllability_summary,
    display_domain,
    mo,
    pd,
    potency,
    primary_noise_label,
    px,
    task_label,
):
    primary_leaf = "choice_prob_norm"
    primary_heatmap = display_domain[display_domain["metric_leaf"].eq(primary_leaf)].copy()
    domain_order = potency.sort_values(["domain_family", "rms_z"], ascending=[True, False])["target_domain"].tolist()
    primary_heatmap["target_domain"] = pd.Categorical(
        primary_heatmap["target_domain"], categories=domain_order, ordered=True
    )
    primary_heatmap["task_label"] = pd.Categorical(
        primary_heatmap["task_label"],
        categories=[task_label(task) for task in TASK_ORDER],
        ordered=True,
    )
    primary_fig = px.imshow(
        primary_heatmap.pivot(index="target_domain", columns="task_label", values="z_primary").sort_index(),
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0.0,
        aspect="auto",
        title=f"Domain-bump directional effects on choice_prob_norm ({primary_noise_label}; positive helps)",
        labels={"x": "task", "y": "bumped domain", "color": "z_primary"},
    )
    primary_fig.update_layout(height=980, width=1250)
    primary_fig.write_html(IMG_DIR / "mcq_choice_prob_norm_domain_heatmap.html", include_plotlyjs="cdn")

    all_heatmap = display_domain.copy()
    all_heatmap["task_metric"] = all_heatmap["task_label"] + " / " + all_heatmap["metric_leaf"]
    column_order = [
        f"{task.replace('_5shot', '').replace('_0shot', '').replace('truthfulqa_', 'tqa_').replace('_', ' ')} / {leaf}"
        for task in TASK_ORDER
        for leaf in DISPLAY_METRIC_ORDER
        if not (task == "truthfulqa_mc2_0shot" and leaf in {"bpb", "nll"})
    ]
    all_heatmap["task_metric"] = pd.Categorical(all_heatmap["task_metric"], categories=column_order, ordered=True)
    all_heatmap["target_domain"] = pd.Categorical(all_heatmap["target_domain"], categories=domain_order, ordered=True)
    all_fig = px.imshow(
        all_heatmap.pivot(index="target_domain", columns="task_metric", values="z_primary").sort_index(),
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0.0,
        aspect="auto",
        title=f"Domain-bump directional effects for MCQ/logprob proxies ({primary_noise_label}; positive helps)",
        labels={"x": "task / metric", "y": "bumped domain", "color": "z_primary"},
    )
    all_fig.update_layout(height=980, width=3100)
    all_fig.write_html(IMG_DIR / "mcq_all_smooth_proxy_domain_heatmap.html", include_plotlyjs="cdn")

    summary_fig = px.scatter(
        controllability_summary[controllability_summary["metric_leaf"].isin(DISPLAY_METRIC_ORDER)].copy(),
        x="median_abs_z_primary",
        y="max_abs_z_primary",
        size="frac_abs_z_ge_2",
        color="task_alias",
        hover_data=["metric_leaf", "strongest_domain", "strongest_domain_z_primary"],
        title="MCQ/logprob proxy controllability summary across 39 domain bumps",
        labels={
            "median_abs_z_primary": "median |z| across domains",
            "max_abs_z_primary": "max |z| across domains",
            "frac_abs_z_ge_2": "fraction |z| >= 2",
        },
    )
    summary_fig.add_hline(y=2.0, line_dash="dash", line_color="gray")
    summary_fig.add_vline(x=2.0, line_dash="dash", line_color="gray")
    summary_fig.write_html(IMG_DIR / "mcq_metric_controllability_scatter.html", include_plotlyjs="cdn")

    mo.output.replace(
        mo.vstack(
            [
                mo.md("## Headline Domain-Bump Heatmaps"),
                mo.ui.plotly(primary_fig),
                mo.ui.plotly(all_fig),
                mo.md("## Metric-Level Controllability Summary"),
                mo.ui.plotly(summary_fig),
                mo.ui.table(controllability_summary, page_size=20),
                mo.md("## Domain Potency Across MCQ Smooth Proxies"),
                mo.ui.table(potency, page_size=20),
            ]
        )
    )
    return all_fig, primary_fig, summary_fig


@app.cell
def _(DISPLAY_METRIC_ORDER, display_domain, mo):
    task_options = sorted(display_domain["task_alias"].unique())
    metric_options = [metric for metric in DISPLAY_METRIC_ORDER if metric in set(display_domain["metric_leaf"])]
    task_dropdown = mo.ui.dropdown(task_options, value="sciq_5shot", label="Task")
    metric_dropdown = mo.ui.dropdown(metric_options, value="choice_prob_norm", label="Metric")
    mo.output.replace(mo.hstack([task_dropdown, metric_dropdown]))
    return metric_dropdown, task_dropdown


@app.cell
def _(IMG_DIR, display_domain, metric_dropdown, mo, px, task_dropdown):
    selected_task = task_dropdown.value
    selected_metric = metric_dropdown.value
    selected = display_domain[
        display_domain["task_alias"].eq(selected_task) & display_domain["metric_leaf"].eq(selected_metric)
    ].sort_values("z_primary")
    selected_fig = px.bar(
        selected,
        x="z_primary",
        y="domain_label",
        color="z_primary",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0.0,
        orientation="h",
        hover_data=["target_domain", "improvement", "slope_per_target_mass", "primary_noise_std"],
        title=f"Domain-bump directional effects: {selected_task} / {selected_metric}",
        labels={"z_primary": "noise-scaled improvement z", "domain_label": "bumped domain"},
    )
    selected_fig.add_vline(x=-2.0, line_dash="dash", line_color="gray")
    selected_fig.add_vline(x=2.0, line_dash="dash", line_color="gray")
    selected_fig.update_layout(height=980)
    selected_fig.write_html(IMG_DIR / "mcq_selected_task_metric_domain_effects.html", include_plotlyjs="cdn")
    mo.output.replace(mo.ui.plotly(selected_fig))
    return selected_fig


@app.cell
def _(DISPLAY_METRIC_ORDER, IMG_DIR, metric_agreement, mo, pd, px):
    agreement = metric_agreement.copy()
    agreement["pair"] = agreement["left_metric"] + " vs " + agreement["right_metric"]
    agreement["left_metric"] = pd.Categorical(agreement["left_metric"], categories=DISPLAY_METRIC_ORDER, ordered=True)
    agreement["right_metric"] = pd.Categorical(agreement["right_metric"], categories=DISPLAY_METRIC_ORDER, ordered=True)
    agreement_fig = px.imshow(
        agreement.pivot_table(
            index=["task_alias", "left_metric"],
            columns="right_metric",
            values="spearman_z_primary",
            observed=False,
        ),
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0.0,
        aspect="auto",
        title="Within-task cross-metric agreement of domain-bump z vectors",
        labels={"x": "right metric", "y": "task / left metric", "color": "Spearman"},
    )
    agreement_fig.update_layout(height=850)
    agreement_fig.write_html(IMG_DIR / "mcq_metric_agreement_heatmap.html", include_plotlyjs="cdn")
    mo.output.replace(mo.vstack([mo.md("## Cross-Metric Agreement"), mo.ui.plotly(agreement_fig)]))
    return agreement_fig


@app.cell
def _(DISPLAY_METRIC_ORDER, IMG_DIR, mo, non_domain_effects, pd, px):
    non_domain_display = non_domain_effects[
        non_domain_effects["metric_leaf"].eq("choice_prob_norm")
        & non_domain_effects["task_alias"].isin(["medmcqa_5shot", "sciq_5shot", "swag_0shot"])
    ].copy()
    non_domain_display["intervention_label"] = (
        non_domain_display["intervention_id"]
        .str.replace("qswap_", "quality: ", regex=False)
        .str.replace("family_", "family: ", regex=False)
    )
    non_domain_display["task_metric"] = non_domain_display["task_label"] + " / " + non_domain_display["metric_leaf"]
    non_domain_fig = px.bar(
        non_domain_display,
        x="z_primary",
        y="intervention_label",
        color="z_primary",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0.0,
        facet_col="task_alias",
        facet_col_wrap=3,
        hover_data=["metric_leaf", "improvement", "slope_per_tv"],
        title=(
            "Family and quality-swap intervention effects on choice_prob_norm "
            "(z units; shown separately from domain bumps)"
        ),
        labels={"z_primary": "noise-scaled improvement z", "intervention_label": "intervention"},
    )
    non_domain_fig.update_yaxes(matches=None)
    non_domain_fig.update_layout(height=900)
    non_domain_fig.write_html(IMG_DIR / "mcq_family_quality_effects.html", include_plotlyjs="cdn")
    mo.output.replace(mo.vstack([mo.md("## Family Bumps and Quality Swaps"), mo.ui.plotly(non_domain_fig)]))
    return non_domain_display, non_domain_fig


@app.cell
def _(
    DISPLAY_METRIC_ORDER,
    controllability_summary,
    display_domain,
    mo,
    non_domain_effects,
    pd,
    potency,
    primary_noise_label,
):
    domain_count = display_domain["target_domain"].nunique()
    task_count = display_domain["task_alias"].nunique()
    metric_count = display_domain["metric_leaf"].nunique()
    strong_cells = int((display_domain["z_primary"].abs() >= 2.0).sum())
    very_strong_cells = int((display_domain["z_primary"].abs() >= 3.0).sum())
    top_metrics = controllability_summary[controllability_summary["metric_leaf"].isin(DISPLAY_METRIC_ORDER)].head(8)[
        [
            "task_alias",
            "metric_leaf",
            "median_abs_z_primary",
            "max_abs_z_primary",
            "frac_abs_z_ge_2",
            "strongest_domain",
            "strongest_domain_z_primary",
        ]
    ]
    top_domains = potency.head(10)[
        ["target_domain", "domain_family", "rms_z", "max_abs_z", "positive_ge2", "negative_ge2"]
    ]
    mo.md(
        f"""
        ## Assessment

        Domain-bump coverage is complete for the targeted custom MCQ smooth
        proxies, and the English-lite metrics are merged by `(panel, run_name)`
        using the custom MCQ result table as the metadata authority.

        Domain-bump coverage is complete for the displayed proxy metrics:
        `{domain_count}` domain directions x `{task_count}` tasks x `{metric_count}`
        displayed smooth metrics. With the primary denominator
        `{primary_noise_label}`, `{strong_cells}` cells
        exceed `|z| >= 2` and `{very_strong_cells}` exceed `|z| >= 3`.

        The key question is not whether any single cell is large; there are many
        comparisons. A proxy looks more controllable when several domains move
        it coherently, its strongest directions exceed the primary-noise floor, and
        related smooth metrics for the same task agree on the domain ranking.

        Top metric/task controllability rows:
        {top_metrics.to_markdown(index=False)}

        Highest-potency domains across the displayed MCQ smooth proxies:
        {top_domains.to_markdown(index=False)}

        Non-domain interventions are present (`{non_domain_effects["intervention_id"].nunique()}` directions)
        but are intentionally excluded from the partial-derivative interpretation.
        """
    )
    return


if __name__ == "__main__":
    app.run()
