# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "gradio",
#     "numpy",
#     "pandas",
#     "plotly",
#     "pyarrow",
#     "scikit-learn",
# ]
# ///
"""Gradio dashboard for factor-DSP constrained mixture recommendation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many.factor_dsp_constraint_dashboard_helpers import (  # noqa: E402
    PROPORTIONAL_RUN_NAME,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.factor_dsp_gradio_dashboard.dashboard_data import (  # noqa: E402
    DEFAULT_SOURCES,
    DashboardData,
    DashboardState,
    load_dashboard_data,
    manual_candidate_choices,
    resolved_starting_candidate,
    select_best_candidate,
    slider_values_for_state,
    starting_candidate_dropdown_value,
    update_constraint_from_lock,
    update_constraint_from_slider,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.factor_dsp_gradio_dashboard.dashboard_views import (  # noqa: E402
    candidate_frontier_figure,
    epoch_figure,
    locked_constraints_table,
    matched_candidate_table,
    mixture_weight_figure,
    quality_table,
    selected_task_prediction_table,
    selected_weight_tables,
    task_delta_figure,
    task_slider_metadata_from_metrics,
    weight_delta_heatmap,
)

TASK_SLIDER_MIN = -5.0
TASK_SLIDER_MAX = 5.0
TASK_SLIDER_STEP = 0.1
DEFAULT_MIN_TARGET_GAIN = 0.0
DEFAULT_MAX_NEAREST_TV = 0.45
DEFAULT_MAX_PHASE_WEIGHT = 0.50
TOP_CANDIDATE_ROWS = 80
APP_CSS = """
.task-slider-column {
    max-height: 78vh;
    overflow-y: auto;
    padding-right: 0.5rem;
}
.task-slider-column .form,
.task-slider-column .block,
.task-slider-column .column {
    margin-left: 0 !important;
    margin-right: 0 !important;
    max-width: 100% !important;
    width: 100% !important;
}
.candidate-panel {
    border-left: 1px solid #d6dbe6;
    padding-left: 1rem;
}
"""


def _include_sources(sources: list[str] | None) -> set[str] | None:
    if sources is None:
        return set(DEFAULT_SOURCES)
    selected = {str(source) for source in sources}
    return selected if selected else set()


def _round_slider_value(value: float) -> float:
    bounded = min(max(float(value), TASK_SLIDER_MIN), TASK_SLIDER_MAX)
    return round(bounded / TASK_SLIDER_STEP) * TASK_SLIDER_STEP


def _slider_updates(data: DashboardData, state: DashboardState) -> list[dict[str, Any]]:
    values = slider_values_for_state(data.task_predictions, state, data.selected_tasks)
    return [gr.update(value=_round_slider_value(values.get(task, 0.0))) for task in data.selected_tasks]


def _lock_updates(data: DashboardData, state: DashboardState) -> list[dict[str, Any]]:
    return [gr.update(value=task in state.constraints) for task in data.selected_tasks]


def _candidate_dropdown_update(data: DashboardData, state: DashboardState) -> dict[str, Any]:
    choices = manual_candidate_choices(data.candidate_summary)
    value = starting_candidate_dropdown_value(
        state,
        choices=choices,
        fallback=PROPORTIONAL_RUN_NAME,
    )
    return gr.update(choices=choices, value=value)


def _top_candidate_table(filtered: pd.DataFrame) -> pd.DataFrame:
    display_cols = [
        "candidate",
        "source",
        "score_kind",
        "target_gain",
        "target_gain_lcb",
        "nearest_observed_tv",
        "max_phase_weight",
        "min_phase_support_gt_1e3",
        "has_task_predictions",
        "deployability_flag",
    ]
    visible_cols = [col for col in display_cols if col in filtered.columns]
    return filtered.loc[:, visible_cols].head(TOP_CANDIDATE_ROWS).copy()


def _state_markdown(state: DashboardState, filtered: pd.DataFrame) -> str:
    if state.no_feasible:
        return (
            "### Current recommendation\n"
            "- No feasible precomputed candidate satisfies the active task constraints under the current filters.\n"
            f"- Last displayed candidate: `{state.candidate}`\n"
            f"- Active task constraints: `{len(state.constraints)}`\n"
            f"- Feasible candidates under current filters: `{len(filtered):,}`\n\n"
            "Locked sliders remain at their requested thresholds. Relax a task constraint or global filter to resume "
            "automatic candidate matching."
        )
    return (
        f"### Current recommendation\n"
        f"- Matched precomputed candidate: `{state.candidate}`\n"
        f"- Active task constraints: `{len(state.constraints)}`\n"
        f"- Feasible candidates under current filters: `{len(filtered):,}`\n\n"
        "Task sliders display the selected candidate's predicted deltas. "
        "The locked constraints table below preserves the actual requested thresholds."
    )


def _render_outputs(
    data: DashboardData,
    state: DashboardState,
    filtered: pd.DataFrame,
) -> list[Any]:
    task_table = selected_task_prediction_table(data, state)
    try:
        weights, epoch_summary = selected_weight_tables(data, state.candidate)
    except ValueError:
        weights = pd.DataFrame()
        epoch_summary = pd.DataFrame()

    return [
        *_lock_updates(data, state),
        *_slider_updates(data, state),
        _candidate_dropdown_update(data, state),
        _state_markdown(state, filtered),
        matched_candidate_table(data.candidate_summary, state),
        locked_constraints_table(data, state),
        _top_candidate_table(filtered),
        candidate_frontier_figure(filtered),
        task_delta_figure(task_table, state.candidate),
        mixture_weight_figure(weights, state.candidate),
        weight_delta_heatmap(weights, state.candidate),
        epoch_figure(weights, state.candidate),
        epoch_summary,
        quality_table(data),
        state,
    ]


def _filtered_without_state(
    data: DashboardData,
    *,
    min_target_gain: float,
    max_nearest_tv: float,
    max_phase_weight: float,
    sources: list[str] | None,
    keep_missing: bool,
) -> tuple[str, pd.DataFrame]:
    return select_best_candidate(
        data.candidate_summary,
        data.task_predictions,
        constraints={},
        min_target_gain=float(min_target_gain),
        max_nearest_tv=float(max_nearest_tv),
        max_phase_weight=float(max_phase_weight),
        include_sources=_include_sources(sources),
        keep_missing=bool(keep_missing),
    )


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks app."""
    data = load_dashboard_data()
    slider_metadata = task_slider_metadata_from_metrics(data.task_prediction_metrics, data.selected_tasks)
    initial_state = DashboardState(candidate=PROPORTIONAL_RUN_NAME, constraints={})
    _, initial_filtered = _filtered_without_state(
        data,
        min_target_gain=DEFAULT_MIN_TARGET_GAIN,
        max_nearest_tv=DEFAULT_MAX_NEAREST_TV,
        max_phase_weight=DEFAULT_MAX_PHASE_WEIGHT,
        sources=list(DEFAULT_SOURCES),
        keep_missing=False,
    )

    with gr.Blocks(title="Factor-DSP Constraint Dashboard") as demo:
        state = gr.State(initial_state)
        gr.Markdown(
            """
            # Factor-DSP Constraint Dashboard

            Start at proportional (`0` on every task slider). Move any task
            slider to set a minimum target delta; the app selects the best
            feasible cached candidate and then moves all sliders to that
            candidate's predicted task deltas. Each slider label starts with
            its surrogate readiness and train Pearson, e.g. `[usable r=0.62]`.
            The starting mixture dropdown resets the slider starting point;
            automatic recommendation remains enabled once constraints are set.
            Use a task's `Lock` checkbox to enforce
            the current slider value, including exactly `0`.
            """
        )
        gr.Markdown("## Task Targets")
        with gr.Row():
            clear_button = gr.Button("Clear task constraints")
        candidate_dropdown = gr.Dropdown(
            choices=manual_candidate_choices(data.candidate_summary),
            value=PROPORTIONAL_RUN_NAME,
            label="Starting mixture",
            info=(
                "Changing this clears task constraints and moves sliders to this precomputed candidate's "
                "predicted deltas."
            ),
            allow_custom_value=False,
        )
        with gr.Accordion("Global candidate filters", open=False):
            keep_missing = gr.Checkbox(
                value=False,
                label="Keep candidates missing task predictions",
            )
            source_filter = gr.CheckboxGroup(
                choices=list(DEFAULT_SOURCES),
                value=list(DEFAULT_SOURCES),
                label="Candidate sources",
            )
            min_target_gain = gr.Slider(
                minimum=-1.0,
                maximum=5.0,
                step=0.05,
                value=DEFAULT_MIN_TARGET_GAIN,
                label="Minimum y_factor gain",
            )
            max_nearest_tv = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=DEFAULT_MAX_NEAREST_TV,
                label="Maximum nearest-observed TV",
            )
            max_phase_weight = gr.Slider(
                minimum=0.05,
                maximum=1.0,
                step=0.01,
                value=DEFAULT_MAX_PHASE_WEIGHT,
                label="Maximum single-domain phase weight",
            )
        gr.Markdown(
            "### Task sliders\n"
            "Move a slider to create/update that task's target constraint. "
            "Check `Lock` to constrain a task at its current value, including exactly `0`. "
            "After recommendation, sliders move to predicted candidate outcomes. "
            "Treat `weak`, `caution`, and `unknown` sliders as guardrails rather than reliable steering targets."
        )
        task_sliders = []
        task_locks = []
        with gr.Accordion("Task target sliders", open=True):
            for task in data.selected_tasks:
                metadata = slider_metadata[task]
                with gr.Row():
                    task_locks.append(gr.Checkbox(value=False, label="Lock", scale=1))
                    task_sliders.append(
                        gr.Slider(
                            minimum=TASK_SLIDER_MIN,
                            maximum=TASK_SLIDER_MAX,
                            step=TASK_SLIDER_STEP,
                            value=0.0,
                            label=metadata["label"],
                            info=metadata["info"],
                            scale=11,
                        )
                    )

        gr.Markdown("## Current Recommendation")
        state_markdown = gr.Markdown()
        with gr.Row():
            candidate_summary = gr.Dataframe(
                label="Matched precomputed candidate",
                interactive=False,
                wrap=True,
            )
            locked_table = gr.Dataframe(
                label="Locked task constraints",
                interactive=False,
                wrap=True,
            )
        top_candidates = gr.Dataframe(
            label="Top feasible candidates",
            interactive=False,
            wrap=True,
        )
        with gr.Tab("Frontier"):
            frontier_plot = gr.Plot()
        with gr.Tab("Task Deltas"):
            task_plot = gr.Plot()
        with gr.Tab("Mixture"):
            weight_plot = gr.Plot()
            delta_heatmap = gr.Plot()
        with gr.Tab("Epochs"):
            epoch_plot = gr.Plot()
            epoch_summary_table = gr.Dataframe(
                label="Materialized epoch summary",
                interactive=False,
                wrap=True,
            )
        with gr.Tab("Task Surrogate Quality"):
            quality = gr.Dataframe(
                label="Task-response fit quality",
                interactive=False,
                wrap=True,
            )

        outputs = [
            *task_locks,
            *task_sliders,
            candidate_dropdown,
            state_markdown,
            candidate_summary,
            locked_table,
            top_candidates,
            frontier_plot,
            task_plot,
            weight_plot,
            delta_heatmap,
            epoch_plot,
            epoch_summary_table,
            quality,
            state,
        ]
        filter_inputs = [
            min_target_gain,
            max_nearest_tv,
            max_phase_weight,
            source_filter,
            keep_missing,
        ]

        def render_current(
            current_state: DashboardState,
            min_gain: float,
            max_tv: float,
            max_weight: float,
            sources: list[str] | None,
            missing_ok: bool,
        ) -> list[Any]:
            selected, filtered = select_best_candidate(
                data.candidate_summary,
                data.task_predictions,
                constraints=current_state.constraints,
                min_target_gain=float(min_gain),
                max_nearest_tv=float(max_tv),
                max_phase_weight=float(max_weight),
                include_sources=_include_sources(sources),
                keep_missing=bool(missing_ok),
            )
            next_state = DashboardState(
                candidate=(
                    current_state.candidate
                    if filtered.empty and current_state.constraints
                    else selected if current_state.constraints else resolved_starting_candidate(current_state)
                ),
                constraints=dict(current_state.constraints),
                starting_candidate=resolved_starting_candidate(current_state),
                no_feasible=bool(filtered.empty and current_state.constraints),
            )
            return _render_outputs(data, next_state, filtered)

        def on_slider_release(task_name: str, *args: Any) -> list[Any]:
            slider_values = args[: len(data.selected_tasks)]
            current_state = args[len(data.selected_tasks)]
            min_gain, max_tv, max_weight, sources, missing_ok = args[len(data.selected_tasks) + 1 :]
            task_index = data.selected_tasks.index(task_name)
            next_state, filtered = update_constraint_from_slider(
                current_state,
                task_name=task_name,
                threshold=float(slider_values[task_index]),
                candidate_summary=data.candidate_summary,
                task_predictions=data.task_predictions,
                min_target_gain=float(min_gain),
                max_nearest_tv=float(max_tv),
                max_phase_weight=float(max_weight),
                include_sources=_include_sources(sources),
                keep_missing=bool(missing_ok),
            )
            return _render_outputs(data, next_state, filtered)

        def on_lock_change(task_name: str, *args: Any) -> list[Any]:
            locked = bool(args[0])
            slider_values = args[1 : 1 + len(data.selected_tasks)]
            current_state = args[1 + len(data.selected_tasks)]
            min_gain, max_tv, max_weight, sources, missing_ok = args[1 + len(data.selected_tasks) + 1 :]
            task_index = data.selected_tasks.index(task_name)
            next_state, filtered = update_constraint_from_lock(
                current_state,
                task_name=task_name,
                threshold=float(slider_values[task_index]),
                locked=locked,
                candidate_summary=data.candidate_summary,
                task_predictions=data.task_predictions,
                min_target_gain=float(min_gain),
                max_nearest_tv=float(max_tv),
                max_phase_weight=float(max_weight),
                include_sources=_include_sources(sources),
                keep_missing=bool(missing_ok),
            )
            return _render_outputs(data, next_state, filtered)

        for task, slider in zip(data.selected_tasks, task_sliders, strict=True):
            # Release only: candidate-implied programmatic slider updates must not
            # create new user constraints.
            slider.release(
                fn=lambda *args, task_name=task: on_slider_release(task_name, *args),
                inputs=[*task_sliders, state, *filter_inputs],
                outputs=outputs,
            )
        for task, lock in zip(data.selected_tasks, task_locks, strict=True):
            lock.change(
                fn=lambda *args, task_name=task: on_lock_change(task_name, *args),
                inputs=[lock, *task_sliders, state, *filter_inputs],
                outputs=outputs,
            )

        def on_clear(
            current_state: DashboardState,
            min_gain: float,
            max_tv: float,
            max_weight: float,
            sources: list[str] | None,
            missing_ok: bool,
        ) -> list[Any]:
            starting_candidate = resolved_starting_candidate(current_state)
            next_state = DashboardState(
                candidate=starting_candidate,
                constraints={},
                starting_candidate=starting_candidate,
                no_feasible=False,
            )
            _, filtered = select_best_candidate(
                data.candidate_summary,
                data.task_predictions,
                constraints={},
                min_target_gain=float(min_gain),
                max_nearest_tv=float(max_tv),
                max_phase_weight=float(max_weight),
                include_sources=_include_sources(sources),
                keep_missing=bool(missing_ok),
            )
            return _render_outputs(data, next_state, filtered)

        def on_starting_mixture_change(
            starting_candidate: str,
            current_state: DashboardState,
            min_gain: float,
            max_tv: float,
            max_weight: float,
            sources: list[str] | None,
            missing_ok: bool,
        ) -> list[Any]:
            starting_state = DashboardState(
                candidate=str(starting_candidate),
                constraints={},
                starting_candidate=str(starting_candidate),
                no_feasible=False,
            )
            _, filtered = select_best_candidate(
                data.candidate_summary,
                data.task_predictions,
                constraints={},
                min_target_gain=float(min_gain),
                max_nearest_tv=float(max_tv),
                max_phase_weight=float(max_weight),
                include_sources=_include_sources(sources),
                keep_missing=bool(missing_ok),
            )
            return _render_outputs(data, starting_state, filtered)

        clear_button.click(
            fn=on_clear,
            inputs=[state, *filter_inputs],
            outputs=outputs,
        )
        candidate_dropdown.change(
            fn=on_starting_mixture_change,
            inputs=[candidate_dropdown, state, *filter_inputs],
            outputs=outputs,
        )
        for control in filter_inputs:
            event = control.release if hasattr(control, "release") else control.change
            event(
                fn=render_current,
                inputs=[state, *filter_inputs],
                outputs=outputs,
            )
        demo.load(
            fn=lambda: _render_outputs(data, initial_state, initial_filtered),
            inputs=None,
            outputs=outputs,
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    app = build_app()
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
