# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kaleido==0.2.1",
#   "pandas",
#   "plotly",
#   "wandb",
# ]
# ///
"""Collect and analyze the fixed-N tokens-per-parameter phase sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from experiments.domain_phase_mix.exploratory.two_phase_many.interactive_mixture_inspector import (
    mixture_inspector_payload,
    mixture_inspector_script,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_fixed_n_tpp_phase_sweep_20260712"
DEFAULT_PRIOR_RESULTS = (
    SCRIPT_DIR / "reference_outputs" / "decoupled_phase_information_validation_results_20260712" / "observed_results.csv"
)
DEFAULT_SURROGATE_PATHS = (
    SCRIPT_DIR
    / "reference_outputs"
    / "decoupled_phase_information_low_epsilon_validation_results_20260712"
    / "combined_uncheatable_paths.csv"
)
DEFAULT_MIXTURE_DIR = (
    SCRIPT_DIR / "reference_outputs" / "decoupled_phase_information_validation_panel_20260712" / "mixtures"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_fixed_n_tpp_phase_sweep_results_20260713"

TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
TRAIN_TAG = "delphi-fixed-n-tpp-phase-sweep"
EVAL_GROUP = "olmo_base_eval_table9_fixed_n_tpp_phase_sweep"
UNCHEATABLE_METRIC = "eval/uncheatable_eval/bpb"
TABLE9_METRIC = "olmo_base_easy/table9_51_component_macro_bpb"
EXPECTED_NEW_ROWS = 12
ORIGINAL_TPP = 1_576_534_016 / 358_306_688
ORIGINAL_DATA_SEED = 690_300
FIXED_N_DATA_SEED = 714_000
EXPORT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

# These are independent-run difference scales from the 3e18 proportional panel.
# They are reference magnitudes, not standard errors for the one-seed TPP sweep.
REFERENCE_DIFFERENCE_SD = {
    "uncheatable_bpb": 0.001291176543909418,
    "table9_macro_bpb": 0.0053340855895512955,
}

POLICY_ORDER = [
    "proportional",
    "tied",
    "epsilon_0.005",
    "epsilon_0.05",
    "epsilon_0.1",
    "epsilon_0.2",
]
POLICY_LABEL = {
    "proportional": "Proportional",
    "tied": "Aggregate-matched tied",
    "epsilon_0.005": r"Two-phase, $\\epsilon_{phase}=0.005$",
    "epsilon_0.05": r"Two-phase, $\\epsilon_{phase}=0.05$",
    "epsilon_0.1": r"Two-phase, $\\epsilon_{phase}=0.1$",
    "epsilon_0.2": r"Two-phase, $\\epsilon_{phase}=0.2$",
}
POLICY_BY_KEY = {
    "proportional": "proportional",
    "unch_effexp_tied": "tied",
    "unch_effexp_e0p005": "epsilon_0.005",
    "unch_effexp_e0p05": "epsilon_0.05",
    "unch_effexp_e0p1": "epsilon_0.1",
    "unch_effexp_e0p2": "epsilon_0.2",
}
PRIOR_CANDIDATE_BY_POLICY = {
    "tied": "dphase_unch05_tied",
    "epsilon_0.005": "dphase_unch05_eff_e0p005",
    "epsilon_0.05": "dphase_unch05_eff_e0p05",
    "epsilon_0.1": "dphase_unch05_eff_e0p1",
    "epsilon_0.2": "dphase_unch05_eff_e0p2",
}
MIXTURE_FILE_BY_POLICY = {policy: f"{candidate}.csv" for policy, candidate in PRIOR_CANDIDATE_BY_POLICY.items()}
INSPECTOR_POLICY_LABEL = {
    "tied": "Aggregate-matched tied",
    "epsilon_0.005": "Two-phase, <i>ε</i><sub>phase</sub> = 0.005",
    "epsilon_0.05": "Two-phase, <i>ε</i><sub>phase</sub> = 0.05",
    "epsilon_0.1": "Two-phase, <i>ε</i><sub>phase</sub> = 0.1",
    "epsilon_0.2": "Two-phase, <i>ε</i><sub>phase</sub> = 0.2",
}
PRIOR_PROPORTIONAL = {
    "uncheatable_bpb": 1.0383423566818237,
    "table9_macro_bpb": 1.1987310593670033,
    "training_wandb_name": "proportional_3e18-ebc4aa",
    "training_wandb_url": "https://wandb.ai/marin-community/marin/runs/proportional_3e18-ebc4aa",
    "eval_wandb_name": "t9_3e18_proportional",
    "eval_wandb_url": "https://wandb.ai/marin-community/marin-eval/runs/t4ax5ofn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--prior-results", type=Path, default=DEFAULT_PRIOR_RESULTS)
    parser.add_argument("--surrogate-paths", type=Path, default=DEFAULT_SURROGATE_PATHS)
    parser.add_argument("--mixture-dir", type=Path, default=DEFAULT_MIXTURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def run_metric(run: wandb.apis.public.Run, metric: str) -> float:
    value = run.summary.get(metric)
    if value is None:
        raise ValueError(f"Run {run.name} lacks {metric}")
    return float(value)


def select_finished_run(
    runs: list[wandb.apis.public.Run],
    *,
    expected_name: str,
    metric: str,
) -> tuple[wandb.apis.public.Run, int]:
    matches = [
        run
        for run in runs
        if run.state == "finished"
        and (run.name == expected_name or run.name.startswith(f"{expected_name}-"))
        and run.summary.get(metric) is not None
    ]
    if not matches:
        states = [(run.name, run.state) for run in runs if run.name.startswith(expected_name)]
        raise ValueError(f"No finished run for {expected_name}; matching attempts={states}")
    matches.sort(key=lambda run: run.created_at)
    return matches[-1], len(matches)


def collect_new_rows(manifest: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    api = wandb.Api(timeout=180)
    training_attempts = list(api.runs(TRAIN_PROJECT, filters={"tags": {"$in": [TRAIN_TAG]}}, per_page=100))
    eval_attempts = list(api.runs(EVAL_PROJECT, filters={"group": EVAL_GROUP}, per_page=100))

    rows: list[dict[str, Any]] = []
    finished_training_matches = 0
    finished_eval_matches = 0
    for spec in manifest:
        run_name = str(spec["run_name"])
        train_run, train_matches = select_finished_run(
            training_attempts,
            expected_name=run_name,
            metric=UNCHEATABLE_METRIC,
        )
        eval_run, eval_matches = select_finished_run(
            eval_attempts,
            expected_name=f"t9_{run_name}",
            metric=TABLE9_METRIC,
        )
        finished_training_matches += train_matches
        finished_eval_matches += eval_matches
        policy = POLICY_BY_KEY[str(spec["policy_key"])]
        rows.append(
            {
                "tpp": float(spec["tokens_per_parameter_realized"]),
                "tpp_target": float(spec["tokens_per_parameter_target"]),
                "policy": policy,
                "policy_label": POLICY_LABEL[policy],
                "phase_information_budget": float(spec["phase_information_budget"] or 0.0),
                "uncheatable_bpb": run_metric(train_run, UNCHEATABLE_METRIC),
                "table9_macro_bpb": run_metric(eval_run, TABLE9_METRIC),
                "data_seed": int(spec["data_seed"]),
                "trainer_seed": int(spec["trainer_seed"]),
                "actual_approximate_flops": float(spec["actual_approximate_flops"]),
                "source_panel": "fixed_n_tpp_sweep",
                "training_wandb_name": train_run.name,
                "training_wandb_url": train_run.url,
                "eval_wandb_name": eval_run.name,
                "eval_wandb_url": eval_run.url,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_NEW_ROWS or frame[["tpp_target", "policy"]].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_NEW_ROWS} unique new rows, got {len(frame)}")
    audit = {
        "training_api_attempts": len(training_attempts),
        "eval_api_attempts": len(eval_attempts),
        "finished_training_matches": finished_training_matches,
        "finished_eval_matches": finished_eval_matches,
        "non_finished_training_attempts": sum(run.state != "finished" for run in training_attempts),
        "non_finished_eval_attempts": sum(run.state != "finished" for run in eval_attempts),
    }
    return frame, audit


def prior_rows(prior_results: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for policy, candidate in PRIOR_CANDIDATE_BY_POLICY.items():
        matches = prior_results[prior_results["candidate"].eq(candidate)]
        if len(matches) != 1:
            raise ValueError(f"Expected one prior row for {candidate}, got {len(matches)}")
        row = matches.iloc[0]
        records.append(
            {
                "tpp": ORIGINAL_TPP,
                "tpp_target": ORIGINAL_TPP,
                "policy": policy,
                "policy_label": POLICY_LABEL[policy],
                "phase_information_budget": float(row["phase_information_budget"]),
                "uncheatable_bpb": float(row["observed_uncheatable_bpb"]),
                "table9_macro_bpb": float(row["observed_table9_macro_bpb"]),
                "data_seed": ORIGINAL_DATA_SEED,
                "trainer_seed": 0,
                "actual_approximate_flops": 3e18,
                "source_panel": "decoupled_phase_information_3e18",
                "training_wandb_name": row["training_wandb_name"],
                "training_wandb_url": row["training_wandb_url"],
                "eval_wandb_name": row["eval_wandb_name"],
                "eval_wandb_url": row["eval_wandb_url"],
            }
        )
    records.append(
        {
            "tpp": ORIGINAL_TPP,
            "tpp_target": ORIGINAL_TPP,
            "policy": "proportional",
            "policy_label": POLICY_LABEL["proportional"],
            "phase_information_budget": 0.0,
            "data_seed": None,
            "trainer_seed": 0,
            "actual_approximate_flops": 3e18,
            "source_panel": "delphi_3e18_baseline",
            **PRIOR_PROPORTIONAL,
        }
    )
    return pd.DataFrame(records)


def add_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    for metric in ("uncheatable_bpb", "table9_macro_bpb"):
        output[f"gain_vs_tied_{metric}"] = 0.0
        output[f"gain_vs_proportional_{metric}"] = 0.0
        for tpp, group in output.groupby("tpp"):
            tied = group.loc[group["policy"].eq("tied"), metric]
            proportional = group.loc[group["policy"].eq("proportional"), metric]
            if len(tied) != 1 or len(proportional) != 1:
                raise ValueError(f"Missing tied/proportional control at TPP {tpp}")
            mask = output["tpp"].eq(tpp)
            output.loc[mask, f"gain_vs_tied_{metric}"] = float(tied.iloc[0]) - output.loc[mask, metric]
            output.loc[mask, f"gain_vs_proportional_{metric}"] = float(proportional.iloc[0]) - output.loc[mask, metric]
    output["policy_order"] = output["policy"].map({policy: i for i, policy in enumerate(POLICY_ORDER)})
    return output.sort_values(["tpp", "policy_order"]).drop(columns="policy_order").reset_index(drop=True)


def summarize_frontiers(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    phase_policies = results[results["policy"].str.startswith("epsilon_")]
    for tpp, group in phase_policies.groupby("tpp"):
        all_rows = results[results["tpp"].eq(tpp)]
        for metric in ("uncheatable_bpb", "table9_macro_bpb"):
            best = group.loc[group[metric].idxmin()]
            tied = all_rows.loc[all_rows["policy"].eq("tied")].iloc[0]
            proportional = all_rows.loc[all_rows["policy"].eq("proportional")].iloc[0]
            phase_gain = float(tied[metric]) - float(best[metric])
            aggregate_gain = float(proportional[metric]) - float(tied[metric])
            rows.append(
                {
                    "metric": metric,
                    "tpp": tpp,
                    "best_policy": best["policy"],
                    "best_phase_information_budget": float(best["phase_information_budget"]),
                    "best_bpb": float(best[metric]),
                    "tied_bpb": float(tied[metric]),
                    "proportional_bpb": float(proportional[metric]),
                    "phase_gain_vs_tied": phase_gain,
                    "aggregate_gain_vs_proportional": aggregate_gain,
                    "phase_gain_fraction_of_aggregate_gain": phase_gain / aggregate_gain,
                    "gain_in_3e18_reference_difference_sd": phase_gain / REFERENCE_DIFFERENCE_SD[metric],
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "tpp"]).reset_index(drop=True)


def policy_colors() -> dict[str, str]:
    epsilon_colors = sample_colorscale("RdYlGn_r", [0.05, 0.35, 0.65, 0.95])
    return {
        "proportional": "#20262E",
        "tied": "#7A8594",
        "epsilon_0.005": epsilon_colors[0],
        "epsilon_0.05": epsilon_colors[1],
        "epsilon_0.1": epsilon_colors[2],
        "epsilon_0.2": epsilon_colors[3],
    }


def render_metric_paths(results: pd.DataFrame, output_dir: Path) -> None:
    colors = policy_colors()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Uncheatable eval", "OLMoBaseEval Table-9 macro"),
        horizontal_spacing=0.10,
    )
    for col, metric in enumerate(("uncheatable_bpb", "table9_macro_bpb"), start=1):
        for policy in POLICY_ORDER:
            path = results[results["policy"].eq(policy)].sort_values("tpp")
            fig.add_trace(
                go.Scatter(
                    x=path["tpp"],
                    y=path[metric],
                    mode="lines+markers",
                    name=POLICY_LABEL[policy],
                    legendgroup=policy,
                    showlegend=col == 1,
                    line={"color": colors[policy], "width": 2.5, "dash": "dash" if policy == "tied" else "solid"},
                    marker={"size": 9},
                    customdata=path[["training_wandb_name", "data_seed", "actual_approximate_flops"]],
                    hovertemplate=(
                        "%{customdata[0]}<br>TPP=%{x:.3f}<br>BPB=%{y:.6f}"
                        "<br>data seed=%{customdata[1]}<br>approx FLOPs=%{customdata[2]:.3e}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="Tokens per parameter", row=1, col=col)
        fig.update_yaxes(title_text="BPB (lower is better)" if col == 1 else None, row=1, col=col)
    fig.update_layout(
        title={"text": "Fixed-N performance as the token horizon grows", "x": 0.5},
        template="plotly_white",
        width=1500,
        height=680,
        margin={"l": 85, "r": 45, "t": 165, "b": 85},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.12, "xanchor": "center", "x": 0.5},
    )
    fig.update_annotations(font={"size": 19, "color": "#243B64"})
    fig.write_html(output_dir / "fixed_n_tpp_metric_paths.html", include_plotlyjs=True, config=EXPORT_CONFIG)
    fig.write_image(output_dir / "fixed_n_tpp_metric_paths.png", scale=2)


def render_phase_gains(results: pd.DataFrame, output_dir: Path) -> None:
    colors = policy_colors()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Uncheatable eval", "OLMoBaseEval Table-9 macro"),
        horizontal_spacing=0.10,
    )
    for col, metric in enumerate(("uncheatable_bpb", "table9_macro_bpb"), start=1):
        gain_column = f"gain_vs_tied_{metric}"
        for policy in POLICY_ORDER[2:]:
            path = results[results["policy"].eq(policy)].sort_values("tpp")
            fig.add_trace(
                go.Scatter(
                    x=path["tpp"],
                    y=path[gain_column],
                    mode="lines+markers",
                    name=POLICY_LABEL[policy],
                    legendgroup=policy,
                    showlegend=col == 1,
                    line={"color": colors[policy], "width": 2.5},
                    marker={"size": 9},
                    customdata=path[[metric, "training_wandb_name"]],
                    hovertemplate=(
                        "%{customdata[1]}<br>TPP=%{x:.3f}<br>gain vs tied=%{y:+.6f}"
                        "<br>BPB=%{customdata[0]:.6f}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        fig.add_hline(y=0.0, line={"color": "#4D5562", "dash": "dash", "width": 1.5}, row=1, col=col)
        fig.update_xaxes(title_text="Tokens per parameter", row=1, col=col)
        fig.update_yaxes(title_text="BPB gain vs aggregate-matched tied" if col == 1 else None, row=1, col=col)
    fig.update_layout(
        title={"text": "Phase-ordering value depends on token horizon and asymmetry", "x": 0.5},
        template="plotly_white",
        width=1500,
        height=680,
        margin={"l": 85, "r": 45, "t": 165, "b": 85},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.12, "xanchor": "center", "x": 0.5},
    )
    fig.update_annotations(font={"size": 19, "color": "#243B64"})
    fig.write_html(output_dir / "phase_gain_vs_tied.html", include_plotlyjs=True, config=EXPORT_CONFIG)
    fig.write_image(output_dir / "phase_gain_vs_tied.png", scale=2)


def predicted_uncheatable_path(surrogate_paths: pd.DataFrame) -> pd.DataFrame:
    budgets = [0.005, 0.05, 0.1, 0.2]
    path = surrogate_paths[
        surrogate_paths["objective"].eq("uncheatable")
        & surrogate_paths["anchor_tag"].eq("unch05")
        & surrogate_paths["family"].eq("effective_exposure")
        & surrogate_paths["phase_information_budget"].isin(budgets)
    ][["phase_information_budget", "predicted_gain_vs_tied"]].copy()
    path = path.drop_duplicates("phase_information_budget").sort_values("phase_information_budget")
    if path["phase_information_budget"].tolist() != budgets:
        raise ValueError(f"Expected surrogate predictions at {budgets}, got {path['phase_information_budget'].tolist()}")
    output = pd.concat(
        [
            pd.DataFrame({"phase_information_budget": [0.0], "predicted_gain_vs_tied": [0.0]}),
            path,
        ],
        ignore_index=True,
    )
    policy_by_budget = {
        0.0: "tied",
        0.005: "epsilon_0.005",
        0.05: "epsilon_0.05",
        0.1: "epsilon_0.1",
        0.2: "epsilon_0.2",
    }
    output["policy"] = output["phase_information_budget"].map(policy_by_budget)
    return output


def mixture_inspector_post_script(mixture_dir: Path) -> str:
    mixture_paths = {policy: mixture_dir / filename for policy, filename in MIXTURE_FILE_BY_POLICY.items()}
    payload = mixture_inspector_payload(mixture_paths, INSPECTOR_POLICY_LABEL)
    return mixture_inspector_script(payload, parameter_count=358_306_688)


def render_epsilon_paths_by_tpp(
    results: pd.DataFrame,
    surrogate_paths: pd.DataFrame,
    mixture_dir: Path,
    output_dir: Path,
) -> None:
    phase_results = results[~results["policy"].eq("proportional")].copy()
    predictions = predicted_uncheatable_path(surrogate_paths)
    tpp_values = sorted(phase_results["tpp"].unique())
    tpp_colors = dict(zip(tpp_values, sample_colorscale("RdYlGn_r", [0.08, 0.72, 0.92]), strict=True))
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable: raw BPB",
            "Uncheatable: difference from tied (lower is better)",
            "Table-9 secondary endpoint: raw BPB",
            "Table-9 secondary endpoint: difference from tied (lower is better)",
        ),
        horizontal_spacing=0.09,
        vertical_spacing=0.16,
    )

    metric_rows = (
        (1, "uncheatable_bpb", "gain_vs_tied_uncheatable_bpb"),
        (2, "table9_macro_bpb", "gain_vs_tied_table9_macro_bpb"),
    )
    for row, metric, gain_metric in metric_rows:
        for tpp in tpp_values:
            path = phase_results[phase_results["tpp"].eq(tpp)].sort_values("phase_information_budget")
            label = f"TPP {tpp:.1f} observed"
            customdata = pd.DataFrame(
                {
                    "policy": path["policy"],
                    "tpp": tpp,
                    "policy_label": path["policy_label"],
                    "data_seed": path["data_seed"],
                    "actual_approximate_flops": path["actual_approximate_flops"],
                    "metric": path[metric],
                    "difference_from_tied": -path[gain_metric],
                }
            )
            figure.add_trace(
                go.Scatter(
                    x=path["phase_information_budget"],
                    y=path[metric],
                    mode="lines+markers",
                    name=label,
                    legendgroup=f"tpp-{tpp}",
                    showlegend=row == 1,
                    line={"color": tpp_colors[tpp], "width": 3},
                    marker={"size": 9},
                    customdata=customdata,
                    hovertemplate=(
                        "%{customdata[2]}<br>epsilon_phase=%{x:.3f}<br>BPB=%{y:.6f}"
                        "<br>policy BPB - tied BPB=%{customdata[6]:+.6f}<br>data seed=%{customdata[3]}"
                        "<br>approx FLOPs=%{customdata[4]:.3e}<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=path["phase_information_budget"],
                    y=-path[gain_metric],
                    mode="lines+markers",
                    name=label,
                    legendgroup=f"tpp-{tpp}",
                    showlegend=False,
                    line={"color": tpp_colors[tpp], "width": 3},
                    marker={"size": 9},
                    customdata=customdata,
                    hovertemplate=(
                        "%{customdata[2]}<br>epsilon_phase=%{x:.3f}"
                        "<br>policy BPB - tied BPB=%{y:+.6f}"
                        "<br>BPB=%{customdata[5]:.6f}<br>data seed=%{customdata[3]}"
                        "<br>approx FLOPs=%{customdata[4]:.3e}<extra></extra>"
                    ),
                ),
                row=row,
                col=2,
            )

            if row == 1:
                tied_bpb = float(path.loc[path["policy"].eq("tied"), metric].iloc[0])
                figure.add_trace(
                    go.Scatter(
                        x=predictions["phase_information_budget"],
                        y=tied_bpb - predictions["predicted_gain_vs_tied"],
                        mode="lines+markers",
                        name=f"TPP {tpp:.1f} anchored prediction",
                        legendgroup=f"predicted-tpp-{tpp}",
                        showlegend=True,
                        line={"color": tpp_colors[tpp], "width": 2, "dash": "dash"},
                        marker={"size": 8, "symbol": "circle-open"},
                        customdata=pd.DataFrame(
                            {
                                "policy": predictions["policy"],
                                "tpp": tpp,
                                "difference_from_tied": -predictions["predicted_gain_vs_tied"],
                            }
                        ),
                        hovertemplate=(
                            "300M surrogate, anchored at this TPP's tied control"
                            "<br>epsilon_phase=%{x:.3f}<br>anchored predicted BPB=%{y:.6f}"
                            "<br>predicted policy BPB - tied BPB=%{customdata[2]:+.6f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )

        if row == 1:
            figure.add_trace(
                go.Scatter(
                    x=predictions["phase_information_budget"],
                    y=-predictions["predicted_gain_vs_tied"],
                    mode="lines+markers",
                    name="300M surrogate predicted difference (TPP-invariant)",
                    legendgroup="predicted-gain",
                    showlegend=True,
                    line={"color": "#25313C", "width": 2.5, "dash": "dash"},
                    marker={"size": 8, "symbol": "circle-open"},
                    customdata=pd.DataFrame(
                        {
                            "policy": predictions["policy"],
                            "tpp": None,
                            "difference_from_tied": -predictions["predicted_gain_vs_tied"],
                        }
                    ),
                    hovertemplate=(
                        "300M surrogate prediction"
                        "<br>epsilon_phase=%{x:.3f}"
                        "<br>predicted policy BPB - tied BPB=%{y:+.6f}<extra></extra>"
                    ),
                ),
                row=1,
                col=2,
            )

        figure.add_hline(y=0.0, line={"color": "#626C78", "dash": "dot", "width": 1.5}, row=row, col=2)
        figure.update_yaxes(title_text="BPB (lower is better)", row=row, col=1)
        figure.update_yaxes(title_text="Policy BPB - tied BPB (lower is better)", row=row, col=2)
        for col in (1, 2):
            figure.update_xaxes(
                title_text="Phase-information budget",
                tickmode="array",
                tickvals=[0.0, 0.005, 0.05, 0.1, 0.2],
                ticktext=["0", "0.005", "0.05", "0.1", "0.2"],
                range=[-0.008, 0.208],
                row=row,
                col=col,
            )

    figure.add_annotation(
        x=0.5,
        y=-0.19,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="center",
        text=(
            "<b>Experiment facts</b><br>Fixed architecture: 358.3M trainable parameters; 80% / 20% WSD. "
            "TPP 4.4 / 10 / 20 use 1.576B / 3.583B / 7.166B tokens.<br>"
            "The same Uncheatable-selected mixture CSVs are reused at every horizon. TPP 4.4 uses data seed 690300; "
            "TPP 10 and 20 use seed 714000; one seed per point.<br>"
            "Dashed predictions are the TPP-invariant 300M / 6B-token surrogate differences, offset to each "
            "observed tied control. Table-9 is a secondary evaluation of the Uncheatable-selected policies."
        ),
        font={"size": 12, "color": "#263B5B"},
    )
    figure.update_layout(
        title={
            "text": "Phase-information paths across token horizons: predicted versus observed",
            "x": 0.5,
        },
        template="plotly_white",
        width=1650,
        height=1120,
        margin={"l": 95, "r": 45, "t": 185, "b": 220},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.08, "xanchor": "center", "x": 0.5},
    )
    figure.update_annotations(font={"size": 18, "color": "#243B64"})
    figure.write_html(
        output_dir / "epsilon_paths_by_tpp_predicted_vs_observed.html",
        include_plotlyjs=True,
        config=EXPORT_CONFIG,
        post_script=mixture_inspector_post_script(mixture_dir),
    )
    figure.write_image(output_dir / "epsilon_paths_by_tpp_predicted_vs_observed.png", scale=2)


def markdown_table(frame: pd.DataFrame) -> str:
    rows = [
        (
            "| metric | TPP | best epsilon | best BPB | tied BPB | proportional BPB | phase gain | "
            "aggregate gain | phase / aggregate | ref. SD units |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in frame.to_dict(orient="records"):
        metric = "Uncheatable" if record["metric"] == "uncheatable_bpb" else "Table-9"
        rows.append(
            f"| {metric} | {record['tpp']:.3f} | {record['best_phase_information_budget']:.3f} | "
            f"{record['best_bpb']:.6f} | {record['tied_bpb']:.6f} | {record['proportional_bpb']:.6f} | "
            f"{record['phase_gain_vs_tied']:+.6f} | {record['aggregate_gain_vs_proportional']:+.6f} | "
            f"{record['phase_gain_fraction_of_aggregate_gain']:.1%} | "
            f"{record['gain_in_3e18_reference_difference_sd']:.2f} |"
        )
    return "\n".join(rows)


def write_report(results: pd.DataFrame, frontiers: pd.DataFrame, audit: dict[str, Any], output_dir: Path) -> None:
    uncheatable = frontiers[frontiers["metric"].eq("uncheatable_bpb")].set_index("tpp")
    table9 = frontiers[frontiers["metric"].eq("table9_macro_bpb")].set_index("tpp")
    uncheatable_paths = results.set_index(["tpp", "policy"])["gain_vs_tied_uncheatable_bpb"]
    tpp_values = sorted(uncheatable.index)
    original_tpp, tpp10, tpp20 = tpp_values
    uncheatable_phase_gains = [float(uncheatable.loc[tpp, "phase_gain_vs_tied"]) for tpp in (original_tpp, tpp10, tpp20)]
    uncheatable_aggregate_gains = [
        float(uncheatable.loc[tpp, "aggregate_gain_vs_proportional"]) for tpp in (original_tpp, tpp10, tpp20)
    ]
    uncheatable_gain_fractions = [
        float(uncheatable.loc[tpp, "phase_gain_fraction_of_aggregate_gain"]) for tpp in (original_tpp, tpp10, tpp20)
    ]
    table9_phase_gains = [float(table9.loc[tpp, "phase_gain_vs_tied"]) for tpp in (original_tpp, tpp10, tpp20)]
    epsilon_01_gains = [float(uncheatable_paths.loc[(tpp, "epsilon_0.1")]) for tpp in (original_tpp, tpp10, tpp20)]
    epsilon_02_gains = [float(uncheatable_paths.loc[(tpp, "epsilon_0.2")]) for tpp in (original_tpp, tpp10, tpp20)]
    report_lines = [
        "# Fixed-N phase benefit versus tokens per parameter",
        "",
        "## Completion",
        "",
        (
            f"- All {EXPECTED_NEW_ROWS}/{EXPECTED_NEW_ROWS} fixed-N training rows and all "
            f"{EXPECTED_NEW_ROWS}/{EXPECTED_NEW_ROWS} native Table-9 endpoints finished."
        ),
        (
            f"- W&B exposed {audit['training_api_attempts']} training attempts and "
            f"{audit['eval_api_attempts']} eval attempts. The collector selected the finished endpoint for each "
            f"manifest row; {audit['non_finished_eval_attempts']} eval attempts were non-finished retries."
        ),
        (
            f"- The TPP 10 and 20 rows share data seed {FIXED_N_DATA_SEED}. The historical TPP "
            f"{ORIGINAL_TPP:.3f} phase panel used seed {ORIGINAL_DATA_SEED}; cross-TPP trends are not paired."
        ),
        "",
        "## Results",
        "",
        markdown_table(frontiers),
        "",
        "## Findings",
        "",
        (
            "1. **Moderate phase asymmetry remains useful as training gets longer.** The best Uncheatable phase "
            f"gain over the exact aggregate-matched tied control is {uncheatable_phase_gains[0]:.6f} BPB at TPP "
            f"{original_tpp:.1f}, {uncheatable_phase_gains[1]:.6f} at TPP 10, and "
            f"{uncheatable_phase_gains[2]:.6f} at TPP 20. The winner shifts from epsilon 0.005 to epsilon 0.05."
        ),
        "",
        (
            "2. **The clearest scale-dependent crossing is epsilon 0.1.** Relative to tied, its Uncheatable gain "
            f"changes from {epsilon_01_gains[0]:+.6f} BPB at TPP {original_tpp:.1f} to "
            f"{epsilon_01_gains[1]:+.6f} at TPP 10 and {epsilon_01_gains[2]:+.6f} at TPP 20. A schedule that was "
            "mildly harmful at the short horizon becomes useful at both longer horizons."
        ),
        "",
        (
            "3. **More phase asymmetry is not monotonically better.** Epsilon 0.2 remains worse than tied at "
            f"every horizon: {epsilon_02_gains[0]:+.6f}, {epsilon_02_gains[1]:+.6f}, and "
            f"{epsilon_02_gains[2]:+.6f} BPB at TPP {original_tpp:.1f}, 10, and 20. Its deficit shrinks with more "
            "tokens, but it has not crossed by TPP 20."
        ),
        "",
        (
            "4. **Aggregate mixture choice still accounts for most of the value.** On Uncheatable, tied beats "
            f"proportional by {uncheatable_aggregate_gains[0]:.6f}, {uncheatable_aggregate_gains[1]:.6f}, and "
            f"{uncheatable_aggregate_gains[2]:.6f} BPB. The best additional phase-ordering gain is only "
            f"{uncheatable_gain_fractions[0]:.1%}, {uncheatable_gain_fractions[1]:.1%}, and "
            f"{uncheatable_gain_fractions[2]:.1%} of that aggregate gain."
        ),
        "",
        (
            "5. **The secondary Table-9 endpoint strengthens at longer horizons.** The best phase gain is "
            f"{table9_phase_gains[0]:.6f}, {table9_phase_gains[1]:.6f}, and {table9_phase_gains[2]:.6f} BPB. "
            "These policies were selected for Uncheatable, so the phase schedule is not merely overfitting its "
            "primary metric."
        ),
        "",
        "## Statistical boundary",
        "",
        (
            "Each TPP has one training seed. The final column divides the same-seed phase gain by the independent-"
            "run difference SD measured at the original 3e18 proportional anchor. It is a reference effect-size "
            "scale, not a p-value or a TPP-specific standard error. The data support token-horizon-dependent phase "
            "value, especially the epsilon 0.1 crossing, but do not establish a monotone law or identify a "
            "universal optimal epsilon."
        ),
        "",
        "## Modeling implication",
        "",
        (
            "A single deployment regularization setting should not be assumed to transfer across token horizons. "
            "At fixed model size, the useful phase-information budget moves outward from epsilon 0.005 toward "
            "epsilon 0.05 as D/N rises, while epsilon 0.2 remains too aggressive. The next clean test is a small "
            "repeat panel at the TPP-20 winner and tied control, followed by target-TPP tuning rather than "
            "extrapolating epsilon from the 300M surrogate alone."
        ),
    ]
    report = "\n".join(report_lines) + "\n"
    (output_dir / "report.md").write_text(report)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.panel_dir / "run_manifest.json").read_text())
    new_rows, audit = collect_new_rows(manifest)
    old_rows = prior_rows(pd.read_csv(args.prior_results))
    results = add_comparisons(pd.concat([old_rows, new_rows], ignore_index=True))
    frontiers = summarize_frontiers(results)

    results.to_csv(args.output_dir / "observed_results.csv", index=False)
    frontiers.to_csv(args.output_dir / "frontier_summary.csv", index=False)
    (args.output_dir / "collection_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    render_metric_paths(results, args.output_dir)
    render_phase_gains(results, args.output_dir)
    render_epsilon_paths_by_tpp(
        results,
        pd.read_csv(args.surrogate_paths),
        args.mixture_dir,
        args.output_dir,
    )
    write_report(results, frontiers, audit, args.output_dir)
    print(frontiers.to_string(index=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
