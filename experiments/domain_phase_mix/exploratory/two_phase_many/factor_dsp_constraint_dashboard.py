# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "plotly",
#     "pyarrow",
#     "scipy",
#     "scikit-learn",
# ]
# ///
"""Marimo dashboard for current factor-DSP constrained mixture selection."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    from experiments.domain_phase_mix.exploratory.two_phase_many.factor_dsp_constraint_dashboard_helpers import (
        PROPORTIONAL_RUN_NAME,
        active_task_thresholds_from_controls,
        add_materialized_epochs,
        candidate_names_satisfying_task_thresholds,
        candidate_names_satisfying_task_thresholds_wide,
        candidate_selector_options,
        candidate_weight_diagnostics,
        centered_task_slider_steps,
        combine_phase_weights,
        epoch_scale_table_from_mixture_weights,
        filter_candidate_summary,
        label_phase_weights_long,
        load_csv,
        load_dashboard_candidate_cache,
        load_parquet,
        load_selected_candidate_weights,
        nearest_observed_tv,
        oriented_task_delta_table,
        phase_weight_long_from_wide,
        sample_frontier_for_plot,
        selected_candidate_for_dashboard,
        selected_task_prediction_long,
    )

    return (
        PROPORTIONAL_RUN_NAME,
        Path,
        active_task_thresholds_from_controls,
        add_materialized_epochs,
        candidate_names_satisfying_task_thresholds,
        candidate_names_satisfying_task_thresholds_wide,
        candidate_selector_options,
        candidate_weight_diagnostics,
        centered_task_slider_steps,
        combine_phase_weights,
        epoch_scale_table_from_mixture_weights,
        filter_candidate_summary,
        go,
        json,
        label_phase_weights_long,
        load_csv,
        load_dashboard_candidate_cache,
        load_parquet,
        load_selected_candidate_weights,
        mo,
        nearest_observed_tv,
        oriented_task_delta_table,
        pd,
        phase_weight_long_from_wide,
        px,
        sample_frontier_for_plot,
        selected_candidate_for_dashboard,
        selected_task_prediction_long,
    )


@app.cell
def _(Path):
    TWO_PHASE_ROOT = Path(__file__).resolve().parent
    REPRO_ROOT = TWO_PHASE_ROOT / "reference_outputs" / "collaborator_grug_v4_aggregate_repro_20260525"
    CURRENT_AGG_DIR = REPRO_ROOT / "sent_raw_metric_matrix_300m_zip"
    CURRENT_DSP_DIR = REPRO_ROOT / "canonical_dsp_sent_zip"
    CANDIDATE_LIBRARY_DIR = TWO_PHASE_ROOT / "reference_outputs" / "factor_dsp_candidate_library_y_factor_20260526"
    ENDPOINT_DISCOVERY_DIR = CANDIDATE_LIBRARY_DIR / "endpoint_discovery"
    SENT_MATRIX_CSV = REPRO_ROOT / "sent_zip_input" / "raw_metric_matrix_300m" / "raw_metric_matrix_300m.csv"
    NOISE_CSV = REPRO_ROOT / "sent_zip_input" / "raw_metric_matrix_300m" / "noise_baseline_run00097_300m.csv"
    OUTPUT_DIR = TWO_PHASE_ROOT / "reference_outputs" / "factor_dsp_constraint_dashboard_20260526"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_COLUMN = "y_factor"
    return (
        CANDIDATE_LIBRARY_DIR,
        CURRENT_AGG_DIR,
        CURRENT_DSP_DIR,
        ENDPOINT_DISCOVERY_DIR,
        NOISE_CSV,
        OUTPUT_DIR,
        SENT_MATRIX_CSV,
        TARGET_COLUMN,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Factor-DSP Constraint Dashboard

    This is a clean-slate dashboard for the **current** aggregate target:
    `y_factor`, the reproduced 5-factor varimax aggregate used in the Grug-MoE
    v4 mixture work. The dashboard intentionally ignores older aggregate
    variants and older task-specific bundles.

    It combines:

    - observed 300M signal rows with actual `y_factor` values,
    - the canonical effective-exposure DSP fit to `y_factor`,
    - a precomputed 524k-candidate proportional-centered Sobol-logit library,
    - DSP endpoint-discovery interpolation paths,
    - materialized weights for proportional, dashboard_v4, collaborator_lcb,
      and the raw DSP optimum,
    - current selected-task oriented deltas against proportional for empirical
      guardrail checks of observed rows,
    - a local ridge task-response cache that predicts standardized task deltas
      for every cached candidate.

    Positive target gain means higher `y_factor` than proportional under the
    relevant score source. Positive task delta means better than proportional
    after each selected task's orientation. Task-slider reactivity is powered by
    a local ridge surrogate over phase weights; it is a dashboard approximation,
    not the canonical DSP aggregate model.
    """
    )
    return


@app.cell
def _(
    CANDIDATE_LIBRARY_DIR,
    CURRENT_AGG_DIR,
    CURRENT_DSP_DIR,
    ENDPOINT_DISCOVERY_DIR,
    NOISE_CSV,
    SENT_MATRIX_CSV,
    json,
    load_csv,
    load_dashboard_candidate_cache,
    load_parquet,
    pd,
):
    raw_matrix = load_csv(SENT_MATRIX_CSV)
    signal_frame = (
        raw_matrix.loc[raw_matrix["row_kind"].eq("signal")].copy()
        if "row_kind" in raw_matrix.columns
        else raw_matrix.copy()
    )
    aggregate_scores = load_csv(CURRENT_AGG_DIR / "aggregate_scores.csv")
    selected_tasks = load_csv(CURRENT_AGG_DIR / "selected_tasks.csv")
    factor_loadings_raw = load_csv(CURRENT_AGG_DIR / "factor_loadings.csv")
    observed_predictions = load_csv(CURRENT_DSP_DIR / "observed_predictions.csv")
    dsp_summary = load_csv(CURRENT_DSP_DIR / "summary.csv")
    mixture_weights = load_csv(CURRENT_DSP_DIR / "mixture_weights.csv")
    noise_frame = load_csv(NOISE_CSV) if NOISE_CSV.exists() else None
    with open(CURRENT_DSP_DIR / "summary.json") as f:
        dsp_summary_json = json.load(f)
    cached_candidate_summary = load_dashboard_candidate_cache(
        candidate_summary_path=CANDIDATE_LIBRARY_DIR / "candidate_summary.parquet",
        endpoint_path_summary_path=ENDPOINT_DISCOVERY_DIR / "endpoint_path_summary.parquet",
    )
    task_prediction_path = CANDIDATE_LIBRARY_DIR / "task_prediction_wide.parquet"
    task_prediction_metrics_path = CANDIDATE_LIBRARY_DIR / "task_prediction_metrics.csv"
    task_prediction_wide = load_parquet(task_prediction_path) if task_prediction_path.exists() else pd.DataFrame()
    task_prediction_metrics = (
        load_csv(task_prediction_metrics_path) if task_prediction_metrics_path.exists() else pd.DataFrame()
    )
    return (
        aggregate_scores,
        cached_candidate_summary,
        dsp_summary,
        dsp_summary_json,
        factor_loadings_raw,
        mixture_weights,
        noise_frame,
        observed_predictions,
        selected_tasks,
        signal_frame,
        task_prediction_metrics,
        task_prediction_wide,
    )


@app.cell
def _(factor_loadings_raw):
    factor_loadings = factor_loadings_raw.rename(columns={"Unnamed: 0": "task"}).melt(
        id_vars=["task"],
        var_name="factor",
        value_name="loading",
    )
    factor_loadings["factor"] = factor_loadings["factor"].map(lambda value: f"factor_{int(value) + 1}")
    factor_loadings["abs_loading"] = factor_loadings["loading"].abs()
    return (factor_loadings,)


@app.cell
def _(
    PROPORTIONAL_RUN_NAME,
    TARGET_COLUMN,
    aggregate_scores,
    cached_candidate_summary,
    candidate_weight_diagnostics,
    combine_phase_weights,
    dsp_summary_json,
    label_phase_weights_long,
    mixture_weights,
    nearest_observed_tv,
    noise_frame,
    oriented_task_delta_table,
    pd,
    phase_weight_long_from_wide,
    selected_tasks,
    signal_frame,
    task_prediction_wide,
):
    observed_weights = phase_weight_long_from_wide(signal_frame, candidate_column="run_name")
    named_weights = label_phase_weights_long(mixture_weights)
    eager_candidate_weights = combine_phase_weights(observed_weights, named_weights)
    weight_diagnostics = candidate_weight_diagnostics(eager_candidate_weights)
    named_nearest = nearest_observed_tv(named_weights, observed_weights)

    task_delta_table = oriented_task_delta_table(
        signal_frame,
        selected_tasks,
        PROPORTIONAL_RUN_NAME,
        noise_frame=noise_frame,
    )
    task_scale_summary = (
        task_delta_table.loc[:, ["task_column", "task_scale", "scale_source"]]
        .drop_duplicates("task_column")
        .merge(selected_tasks.rename(columns={"task": "task_column"}), on="task_column", how="left")
        .sort_values(["scale_source", "task_column"])
    )

    observed_scores = aggregate_scores.loc[:, ["run_name", "run_id", TARGET_COLUMN]].copy()
    proportional_actual = float(
        observed_scores.loc[observed_scores["run_name"].eq(PROPORTIONAL_RUN_NAME), TARGET_COLUMN].iloc[0]
    )
    observed_summary = observed_scores.rename(columns={"run_name": "candidate", TARGET_COLUMN: "target_score"}).copy()
    observed_summary["source"] = "observed_300m_signal"
    observed_summary["score_kind"] = "actual"
    observed_summary["baseline_score"] = proportional_actual
    observed_summary["target_gain"] = observed_summary["target_score"] - proportional_actual
    observed_summary["nearest_observed_run"] = observed_summary["candidate"]
    observed_summary["nearest_observed_tv"] = 0.0
    observed_summary["has_task_deltas"] = True
    observed_summary["target_gain_lcb"] = observed_summary["target_gain"]

    pred_y_by_label = dsp_summary_json["pred_y_by_label"]
    proportional_pred = float(pred_y_by_label["proportional"])
    named_summary = pd.DataFrame(
        [
            {
                "candidate": label,
                "target_score": float(score),
                "source": "canonical_dsp_mixture",
                "score_kind": "dsp_prediction",
                "baseline_score": proportional_pred,
                "target_gain": float(score) - proportional_pred,
                "has_task_deltas": False,
            }
            for label, score in pred_y_by_label.items()
        ]
    )
    named_summary = named_summary.merge(named_nearest, on="candidate", how="left")
    named_summary.loc[named_summary["candidate"].eq("proportional"), "nearest_observed_run"] = PROPORTIONAL_RUN_NAME
    named_summary.loc[named_summary["candidate"].eq("proportional"), "nearest_observed_tv"] = 0.0

    eager_summary = pd.concat([observed_summary, named_summary], ignore_index=True, sort=False)
    eager_summary = eager_summary.merge(weight_diagnostics, on="candidate", how="left")
    eager_summary["deployability_flag"] = (
        eager_summary["target_gain"].ge(0.0)
        & eager_summary["nearest_observed_tv"].fillna(1.0).le(0.45)
        & eager_summary["max_phase_weight"].fillna(1.0).le(0.50)
    )

    candidate_summary_all = pd.concat([eager_summary, cached_candidate_summary], ignore_index=True, sort=False)
    candidate_summary_all["deployability_flag"] = candidate_summary_all["deployability_flag"].fillna(False).astype(bool)
    if not task_prediction_wide.empty and "candidate" in task_prediction_wide.columns:
        predicted_candidates = set(task_prediction_wide["candidate"].astype(str))
        candidate_summary_all["has_task_predictions"] = (
            candidate_summary_all["candidate"].astype(str).isin(predicted_candidates)
        )
    else:
        candidate_summary_all["has_task_predictions"] = False
    return (
        candidate_summary_all,
        eager_candidate_weights,
        task_delta_table,
        task_scale_summary,
    )


@app.cell
def _(
    PROPORTIONAL_RUN_NAME,
    TARGET_COLUMN,
    candidate_summary_all,
    dsp_summary,
    mo,
    selected_tasks,
    signal_frame,
    task_prediction_metrics,
    task_prediction_wide,
    task_scale_summary,
):
    signal_count = len(signal_frame)
    observed_count = int(candidate_summary_all["source"].eq("observed_300m_signal").sum())
    named_count = int(candidate_summary_all["source"].eq("canonical_dsp_mixture").sum())
    safe_library_count = int(candidate_summary_all["source"].eq("sobol_logit_trust").sum())
    endpoint_path_count = int(candidate_summary_all["source"].eq("dsp_endpoint_path").sum())
    canonical_path_count = int(candidate_summary_all["source"].eq("canonical_dsp_path").sum())
    noise_scale_count = int(task_scale_summary["scale_source"].eq("noise_std").sum())
    fallback_scale_count = int(task_scale_summary["scale_source"].eq("signal_std_fallback").sum())
    task_prediction_count = len(task_prediction_wide)
    task_prediction_metric_count = len(task_prediction_metrics)
    dsp_row = dsp_summary.iloc[0]
    mo.md(
        f"""
        ## Loaded artifacts

        | Artifact | Count / value |
        | :--- | ---: |
        | Fixed target | `{TARGET_COLUMN}` |
        | Signal rows | {signal_count:,} |
        | Observed candidates | {observed_count:,} |
        | Canonical DSP named mixtures | {named_count:,} |
        | Safe Sobol-logit cache candidates | {safe_library_count:,} |
        | Canonical DSP interpolation-path candidates | {canonical_path_count:,} |
        | Endpoint-discovery path candidates | {endpoint_path_count:,} |
        | Current selected tasks | {len(selected_tasks):,} |
        | Task-predicted candidates | {task_prediction_count:,} |
        | Task-surrogate fit rows | {task_prediction_metric_count:,} |
        | Task scales from run00097 noise std | {noise_scale_count:,} |
        | Task scales from signal-std fallback | {fallback_scale_count:,} |
        | Baseline observed row | `{PROPORTIONAL_RUN_NAME}` |
        | Canonical DSP OOF Spearman | {float(dsp_row["oof_spearman"]):.3f} |
        | Canonical DSP OOF R2 | {float(dsp_row["oof_r2"]):.3f} |
        | Raw DSP optimum nearest observed TV | {float(dsp_row["raw_nearest_observed_tv"]):.3f} |
        """
    )
    return


@app.cell
def _(mo):
    min_target_gain_slider = mo.ui.slider(
        start=-1.0,
        stop=5.0,
        step=0.05,
        value=0.0,
        label="Minimum y_factor gain vs proportional",
    )
    max_nearest_tv_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.45,
        label="Maximum nearest-observed TV",
    )
    max_phase_weight_slider = mo.ui.slider(
        start=0.05,
        stop=1.0,
        step=0.01,
        value=0.50,
        label="Maximum single-domain phase weight",
    )
    keep_missing_task_predictions = mo.ui.checkbox(
        value=True,
        label="Keep candidates with missing task predictions when task constraints are active",
    )
    source_multiselect = mo.ui.multiselect(
        options=[
            "observed_300m_signal",
            "canonical_dsp_mixture",
            "sobol_logit_trust",
            "canonical_dsp_path",
            "dsp_endpoint_path",
        ],
        value=[
            "observed_300m_signal",
            "canonical_dsp_mixture",
            "sobol_logit_trust",
            "canonical_dsp_path",
            "dsp_endpoint_path",
        ],
        label="Candidate sources",
    )
    mo.vstack(
        [
            mo.md("## Candidate filters"),
            mo.hstack([min_target_gain_slider, max_nearest_tv_slider, max_phase_weight_slider]),
            source_multiselect,
            keep_missing_task_predictions,
            mo.md(
                """
                Source semantics:
                `sobol_logit_trust` is the local proportional-centered library;
                `dsp_endpoint_path` and `canonical_dsp_path` are extrapolative
                surrogate directions materialized as paths from proportional.
                Task constraints are evaluated against the local ridge task
                response cache when available.
                """
            ),
        ]
    )
    return (
        keep_missing_task_predictions,
        max_nearest_tv_slider,
        max_phase_weight_slider,
        min_target_gain_slider,
        source_multiselect,
    )


@app.cell
def _(centered_task_slider_steps, mo, selected_tasks):
    def short_task_label(task: str) -> str:
        return (
            task.replace("eval/uncheatable_eval/", "uncheatable/")
            .replace("teacher_forced/", "tf/")
            .replace("lm_eval/", "")
            .replace("choice_logprob", "clp")
        )

    task_options = selected_tasks["task"].astype(str).sort_values().tolist()
    task_slider_steps = centered_task_slider_steps()
    task_threshold_sliders = {
        task: mo.ui.slider(
            steps=task_slider_steps,
            value=0.0,
            label="",
            show_value=True,
            include_input=True,
            full_width=True,
        )
        for task in task_options
    }
    task_lock_checkboxes = {task: mo.ui.checkbox(value=False, label="lock") for task in task_options}
    task_rows = [
        mo.hstack(
            [
                task_lock_checkboxes[task],
                mo.md(f"`{short_task_label(task)}`"),
                task_threshold_sliders[task],
            ],
            widths=[1, 4, 7],
        )
        for task in task_options
    ]
    mo.vstack(
        [
            mo.md(
                """
                ## Per-task target / guardrail sliders

                All task sliders are centered at `0 = proportional`, with ticks
                in standardized oriented-delta units. Positive values request
                improvement over proportional; negative values allow regression.
                Moving a slider away from `0` activates it as a candidate
                filter. Use `lock` when you want to enforce the current value
                even if it is exactly `0`, i.e. no predicted regression versus
                proportional. After constraints select a candidate, the
                candidate-implied task deltas below show where the remaining
                task positions land.

                Observed signal rows use empirical task deltas inside the
                task-response cache. Cached model-only candidates use the local
                ridge surrogate over phase weights, so low-SNR or weakly
                controllable tasks should still be treated cautiously.
                """
            ),
            mo.hstack(
                [mo.md("**Lock?**"), mo.md("**Task**"), mo.md("**Target delta vs proportional**")], widths=[1, 4, 7]
            ),
            *task_rows,
        ]
    )
    return task_lock_checkboxes, task_threshold_sliders


@app.cell
def _(
    active_task_thresholds_from_controls,
    pd,
    task_lock_checkboxes,
    task_threshold_sliders,
):
    active_task_thresholds = active_task_thresholds_from_controls(
        slider_values={task: float(slider.value) for task, slider in task_threshold_sliders.items()},
        lock_values={task: bool(checkbox.value) for task, checkbox in task_lock_checkboxes.items()},
    )
    lock_values = {task: bool(checkbox.value) for task, checkbox in task_lock_checkboxes.items()}
    active_task_threshold_table = pd.DataFrame(
        [
            {
                "task_column": task,
                "min_standardized_delta": threshold,
                "locked": bool(lock_values.get(task, False)),
                "activation": "locked" if lock_values.get(task, False) else "moved_slider",
            }
            for task, threshold in active_task_thresholds.items()
        ]
    )
    return active_task_threshold_table, active_task_thresholds


@app.cell
def _(active_task_threshold_table, mo):
    if active_task_threshold_table.empty:
        _view = mo.md(
            "No active per-task constraints. Move a task slider away from `0`, or check `lock` to enforce exactly `0`."
        )
    else:
        _view = mo.vstack(
            [
                mo.md("### Active per-task constraints"),
                mo.ui.table(active_task_threshold_table.sort_values("task_column"), page_size=12),
            ]
        )
    _view  # noqa: B018
    return


@app.cell
def _(
    PROPORTIONAL_RUN_NAME,
    active_task_thresholds,
    candidate_names_satisfying_task_thresholds,
    candidate_names_satisfying_task_thresholds_wide,
    candidate_selector_options,
    candidate_summary_all,
    filter_candidate_summary,
    keep_missing_task_predictions,
    max_nearest_tv_slider,
    max_phase_weight_slider,
    min_target_gain_slider,
    source_multiselect,
    task_delta_table,
    task_prediction_wide,
):
    filtered_base = candidate_summary_all.loc[candidate_summary_all["source"].isin(source_multiselect.value)].copy()

    if active_task_thresholds:
        if not task_prediction_wide.empty:
            satisfying_task_candidates = candidate_names_satisfying_task_thresholds_wide(
                task_prediction_wide,
                active_task_thresholds,
                keep_missing=bool(keep_missing_task_predictions.value),
                all_candidates=set(filtered_base["candidate"].astype(str)),
            )
        else:
            satisfying_task_candidates = candidate_names_satisfying_task_thresholds(
                task_delta_table,
                active_task_thresholds,
                keep_missing=bool(keep_missing_task_predictions.value),
                all_candidates=set(filtered_base["candidate"].astype(str)),
            )
        filtered_base = filtered_base.loc[filtered_base["candidate"].isin(satisfying_task_candidates)].copy()

    filtered_candidates = filter_candidate_summary(
        filtered_base,
        min_target_gain=float(min_target_gain_slider.value),
        max_nearest_tv=float(max_nearest_tv_slider.value),
        max_phase_weight=float(max_phase_weight_slider.value),
    )
    preferred = "path_endpoint_kl2p0_mw2p0_t1p0"
    if preferred not in set(filtered_candidates["candidate"].astype(str)):
        preferred = (
            "raw_dsp_optimum"
            if "raw_dsp_optimum" in set(filtered_candidates["candidate"].astype(str))
            else PROPORTIONAL_RUN_NAME
        )
    candidate_options = candidate_selector_options(filtered_candidates, preferred=preferred, limit=2_000)
    default_candidate = candidate_options[0]
    return candidate_options, default_candidate, filtered_candidates


@app.cell
def _(candidate_options, default_candidate, mo):
    follow_recommended_candidate = mo.ui.checkbox(
        value=True,
        label="Automatically inspect best feasible recommendation",
    )
    candidate_selector = mo.ui.dropdown(
        options=candidate_options,
        value=default_candidate,
        label="Manual candidate mixture to inspect",
    )
    mo.vstack(
        [
            mo.md(
                """
                ## Recommendation mode

                With automatic mode enabled, moving any task slider away from
                `0` updates the feasible set and immediately switches the
                inspected mixture to the best remaining candidate. Disable it
                to inspect a manually chosen candidate from the dropdown.
                """
            ),
            follow_recommended_candidate,
            candidate_selector,
        ]
    )
    return candidate_selector, follow_recommended_candidate


@app.cell
def _(
    candidate_selector,
    default_candidate,
    follow_recommended_candidate,
    mo,
    selected_candidate_for_dashboard,
):
    selected_candidate_name = selected_candidate_for_dashboard(
        recommended_candidate=default_candidate,
        manual_candidate=str(candidate_selector.value),
        follow_recommendation=bool(follow_recommended_candidate.value),
    )
    mode = (
        "automatic best feasible recommendation" if follow_recommended_candidate.value else "manual dropdown selection"
    )
    mo.md(
        f"""
        **Currently inspecting:** `{selected_candidate_name}`

        Selection mode: {mode}. Current best feasible recommendation:
        `{default_candidate}`.
        """
    )
    return (selected_candidate_name,)


@app.cell
def _(filtered_candidates, mo, px, sample_frontier_for_plot):
    if filtered_candidates.empty:
        _view = mo.md("No candidates satisfy the current constraints.")
    else:
        frontier_frame = sample_frontier_for_plot(filtered_candidates, max_rows=20_000, top_rows=2_000)
        display_cols = [
            "candidate",
            "source",
            "score_kind",
            "target_score",
            "target_gain",
            "target_gain_lcb",
            "nearest_observed_tv",
            "nearest_observed_run",
            "max_phase_weight",
            "min_phase_support_gt_1e3",
            "mean_phase_effective_support",
            "has_task_deltas",
            "has_task_predictions",
            "deployability_flag",
            "optimization_success",
            "endpoint_kl_penalty",
            "path_t",
        ]
        visible = filtered_candidates.loc[:, [col for col in display_cols if col in filtered_candidates.columns]].head(
            100
        )
        frontier = px.scatter(
            frontier_frame,
            x="nearest_observed_tv",
            y="target_gain",
            color="source",
            symbol="score_kind",
            hover_data=[
                "candidate",
                "target_gain_lcb",
                "max_phase_weight",
                "nearest_observed_run",
                "has_task_deltas",
                "has_task_predictions",
                "deployability_flag",
            ],
            title=(
                "Candidate frontier for current y_factor aggregate "
                f"({len(frontier_frame):,} plotted / {len(filtered_candidates):,} filtered)"
            ),
            labels={
                "nearest_observed_tv": "Nearest observed TV (lower is safer)",
                "target_gain": "y_factor gain vs proportional",
            },
        )
        frontier.add_vline(x=0.45, line_dash="dot", line_color="gray")
        _view = mo.vstack(
            [
                mo.md("## Filtered candidate frontier"),
                mo.ui.plotly(frontier),
                mo.ui.table(visible, page_size=20),
            ]
        )
    _view  # noqa: B018
    return


@app.cell
def _(
    CANDIDATE_LIBRARY_DIR,
    ENDPOINT_DISCOVERY_DIR,
    add_materialized_epochs,
    eager_candidate_weights,
    epoch_scale_table_from_mixture_weights,
    go,
    load_selected_candidate_weights,
    mixture_weights,
    mo,
    selected_candidate_name,
):
    selected_weight_candidate = selected_candidate_name
    cache_weight_paths = [
        CANDIDATE_LIBRARY_DIR / "candidate_weights_wide.parquet",
        ENDPOINT_DISCOVERY_DIR / "endpoint_path_weights_wide.parquet",
    ]
    selected_weights = load_selected_candidate_weights(
        selected_weight_candidate,
        eager_weights=eager_candidate_weights,
        parquet_weight_paths=cache_weight_paths,
    )
    proportional = eager_candidate_weights.loc[eager_candidate_weights["candidate"].eq("baseline_proportional")].copy()
    if proportional.empty:
        proportional = eager_candidate_weights.loc[eager_candidate_weights["candidate"].eq("proportional")].copy()
    selected_with_baseline = selected_weights.merge(
        proportional.rename(columns={"weight": "proportional_weight"}).loc[
            :, ["domain", "phase", "proportional_weight"]
        ],
        on=["domain", "phase"],
        how="left",
    )
    selected_with_baseline["weight_delta"] = (
        selected_with_baseline["weight"] - selected_with_baseline["proportional_weight"]
    )
    epoch_scales = epoch_scale_table_from_mixture_weights(mixture_weights)
    selected_with_baseline = add_materialized_epochs(selected_with_baseline, epoch_scales)
    top_domains = (
        selected_with_baseline.groupby("domain", observed=True)["weight"]
        .max()
        .sort_values(ascending=False)
        .head(24)
        .index
    )
    plot_frame = selected_with_baseline.loc[selected_with_baseline["domain"].isin(top_domains)].copy()
    bars = go.Figure()
    for phase, group in plot_frame.groupby("phase", sort=True, observed=True):
        bars.add_bar(
            x=group["domain"],
            y=group["weight"],
            name=phase,
            customdata=group[["proportional_weight", "weight_delta"]],
            hovertemplate=(
                "domain=%{x}<br>weight=%{y:.4f}<br>"
                "proportional=%{customdata[0]:.4f}<br>delta=%{customdata[1]:+.4f}<extra></extra>"
            ),
        )
    bars.update_layout(
        title=f"Top materialized weights for {selected_weight_candidate}",
        barmode="group",
        xaxis={"tickangle": -45},
        yaxis_title="phase weight",
        height=540,
    )

    delta_heatmap = selected_with_baseline.pivot(index="phase", columns="domain", values="weight_delta").fillna(0.0)
    heatmap = go.Figure(
        data=go.Heatmap(
            z=delta_heatmap.to_numpy(),
            x=list(delta_heatmap.columns),
            y=list(delta_heatmap.index),
            colorscale="RdYlGn_r",
            zmid=0.0,
            colorbar={"title": "delta"},
        )
    )
    heatmap.update_layout(
        title=f"Weight delta vs proportional: {selected_weight_candidate}",
        xaxis={"tickangle": -45},
        height=440,
    )
    epoch_summary = (
        selected_with_baseline.sort_values("materialized_epochs", ascending=False)
        .groupby("phase", observed=True)
        .agg(
            max_materialized_epochs=("materialized_epochs", "max"),
            max_epoch_domain=("domain", "first"),
            mean_materialized_epochs=("materialized_epochs", "mean"),
            total_materialized_epochs=("materialized_epochs", "sum"),
        )
        .reset_index()
    )
    epoch_plot_frame = (
        selected_with_baseline.reindex(selected_with_baseline["materialized_epochs"].sort_values(ascending=False).index)
        .head(40)
        .copy()
    )
    epoch_bars = go.Figure()
    for phase, group in epoch_plot_frame.groupby("phase", sort=True, observed=True):
        epoch_bars.add_bar(
            x=group["domain"],
            y=group["materialized_epochs"],
            name=phase,
            customdata=group[["proportional_epochs", "epoch_delta_vs_proportional", "weight"]],
            hovertemplate=(
                "domain=%{x}<br>epochs=%{y:.3f}<br>"
                "proportional epochs=%{customdata[0]:.3f}<br>"
                "epoch delta=%{customdata[1]:+.3f}<br>"
                "weight=%{customdata[2]:.4f}<extra></extra>"
            ),
        )
    epoch_bars.update_layout(
        title=f"Materialized epochs for {selected_weight_candidate}",
        barmode="group",
        xaxis={"tickangle": -45},
        yaxis_title="materialized epochs",
        height=520,
    )
    mo.vstack(
        [
            mo.md("## Selected mixture"),
            mo.ui.plotly(bars),
            mo.ui.plotly(heatmap),
            mo.md("### Materialized epochs"),
            mo.ui.table(epoch_summary, page_size=5),
            mo.ui.plotly(epoch_bars),
            mo.ui.table(selected_with_baseline.sort_values("weight", ascending=False).head(60), page_size=20),
        ]
    )
    return (selected_with_baseline,)


@app.cell
def _(
    active_task_thresholds,
    mo,
    px,
    selected_candidate_name,
    selected_task_prediction_long,
    task_delta_table,
    task_prediction_wide,
):
    selected_task_candidate = selected_candidate_name
    selected_task_deltas = selected_task_prediction_long(
        task_prediction_wide,
        selected_task_candidate,
        thresholds=active_task_thresholds,
    )
    if selected_task_deltas.empty:
        selected_task_deltas = task_delta_table.loc[task_delta_table["candidate"].eq(selected_task_candidate)].rename(
            columns={"task_delta_standardized": "predicted_task_delta_standardized"}
        )
        if not selected_task_deltas.empty:
            selected_task_deltas["locked"] = selected_task_deltas["task_column"].isin(active_task_thresholds)
            selected_task_deltas["target_threshold"] = selected_task_deltas["task_column"].map(active_task_thresholds)
            selected_task_deltas["meets_target"] = ~selected_task_deltas["locked"] | (
                selected_task_deltas["predicted_task_delta_standardized"] >= selected_task_deltas["target_threshold"]
            )
    if selected_task_deltas.empty:
        _view = mo.md(
            "## Candidate-implied task slider positions\n\n" "No per-task deltas are available for the selected mixture."
        )
    else:
        selected_task_deltas["short_task"] = selected_task_deltas["task_column"].str.replace(
            "eval/uncheatable_eval/", "uncheatable/", regex=False
        )
        top = selected_task_deltas.reindex(
            selected_task_deltas["predicted_task_delta_standardized"].abs().sort_values(ascending=False).index
        ).head(41)
        selected_task_deltas["constraint_status"] = selected_task_deltas.apply(
            lambda row: (
                "locked pass"
                if row["locked"] and row["meets_target"]
                else ("locked fail" if row["locked"] else "unlocked")
            ),
            axis=1,
        )
        top["constraint_status"] = top.apply(
            lambda row: (
                "locked pass"
                if row["locked"] and row["meets_target"]
                else ("locked fail" if row["locked"] else "unlocked")
            ),
            axis=1,
        )
        task_plot = px.bar(
            top.sort_values("predicted_task_delta_standardized"),
            x="predicted_task_delta_standardized",
            y="short_task",
            color="constraint_status",
            orientation="h",
            title=f"Candidate-implied task slider positions vs proportional: {selected_task_candidate}",
            labels={"predicted_task_delta_standardized": "standardized oriented delta"},
            color_discrete_map={
                "locked pass": "#2ca25f",
                "locked fail": "#de2d26",
                "unlocked": "#6baed6",
            },
        )
        task_plot.add_vline(x=0.0, line_dash="dot", line_color="gray")
        task_plot.update_layout(height=900)
        _view = mo.vstack(
            [
                mo.md(
                    """
                    ## Candidate-implied task slider positions

                    These are the standardized task deltas implied by the
                    currently selected/recommended candidate. They are the
                    values the task sliders would move to if the dashboard were
                    showing candidate outcomes instead of target constraints.
                    """
                ),
                mo.ui.plotly(task_plot),
                mo.ui.table(selected_task_deltas.sort_values("predicted_task_delta_standardized"), page_size=20),
            ]
        )
    _view  # noqa: B018
    return


@app.cell
def _(factor_loadings, mo):
    factor_options = sorted(factor_loadings["factor"].dropna().unique().tolist())
    factor_selector = mo.ui.dropdown(
        options=factor_options,
        value=factor_options[0],
        label="Factor loading view",
    )
    factor_selector  # noqa: B018
    return (factor_selector,)


@app.cell
def _(factor_loadings, factor_selector, mo, px):
    selected_factor = factor_selector.value
    loadings = factor_loadings.loc[factor_loadings["factor"].eq(selected_factor)].copy()
    top_loadings = loadings.sort_values("abs_loading", ascending=False).head(41)
    loading_plot = px.bar(
        top_loadings.sort_values("loading"),
        x="loading",
        y="task",
        color="loading",
        color_continuous_scale="RdYlGn_r",
        orientation="h",
        title=f"Current aggregate factor loadings: {selected_factor}",
    )
    loading_plot.update_layout(height=860)
    mo.vstack(
        [
            mo.md("## Factor interpretation"),
            mo.ui.plotly(loading_plot),
            mo.ui.table(top_loadings, page_size=20),
        ]
    )
    return


@app.cell
def _(mo, observed_predictions, px):
    fit_scatter = px.scatter(
        observed_predictions,
        x="actual_y_factor",
        y="pred_y_factor",
        hover_data=["run_name", "actual_rank_desc", "pred_rank_desc"],
        title="Canonical DSP fit: observed vs predicted y_factor",
        labels={"actual_y_factor": "actual y_factor", "pred_y_factor": "DSP predicted y_factor"},
    )
    lo = float(min(observed_predictions["actual_y_factor"].min(), observed_predictions["pred_y_factor"].min()))
    hi = float(max(observed_predictions["actual_y_factor"].max(), observed_predictions["pred_y_factor"].max()))
    fit_scatter.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line={"dash": "dot", "color": "gray"})
    mo.vstack(
        [
            mo.md("## Canonical DSP fit check"),
            mo.ui.plotly(fit_scatter),
            mo.ui.table(observed_predictions.sort_values("actual_rank_desc").head(25), page_size=12),
        ]
    )
    return


@app.cell
def _(mo, task_scale_summary):
    mo.vstack(
        [
            mo.md("## Current selected-task scale sources"),
            mo.ui.table(task_scale_summary, page_size=20),
        ]
    )
    return


@app.cell
def _(mo, task_prediction_metrics):
    if task_prediction_metrics.empty:
        _view = mo.md("## Task-response surrogate quality\n\nNo task prediction cache is loaded.")
    else:
        display = task_prediction_metrics.sort_values("train_pearson", ascending=True).copy()
        _view = mo.vstack(
            [
                mo.md(
                    """
                    ## Task-response surrogate quality

                    Per-task slider reactivity uses a local ridge surrogate over
                    centered two-phase weights. Low train correlation here means
                    the task is weakly captured by this dashboard approximation.
                    """
                ),
                mo.ui.table(display, page_size=20),
            ]
        )
    _view  # noqa: B018
    return


@app.cell
def _(OUTPUT_DIR, filtered_candidates, selected_with_baseline):
    filtered_candidates.head(50_000).to_csv(OUTPUT_DIR / "filtered_candidates_latest.csv", index=False)
    selected_with_baseline.to_csv(OUTPUT_DIR / "selected_candidate_weights_latest.csv", index=False)
    return


@app.cell
def _(OUTPUT_DIR, mo):
    mo.md(
        f"""
    ## Local exports

    The notebook writes the top 50,000 current filtered candidates and selected
    candidate weights to:

    `{OUTPUT_DIR}`

    This is intentionally a dashboard snapshot, not a source-of-truth
    registry update.
    """
    )
    return


if __name__ == "__main__":
    app.run()
