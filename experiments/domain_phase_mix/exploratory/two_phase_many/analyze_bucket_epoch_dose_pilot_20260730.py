# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Analyze the completed matched-scale bucket epoch-dose pilot."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
PANEL_DIR = REFERENCE_OUTPUTS / "bucket_epoch_dose_response_20260729" / "pilot"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "bucket_epoch_dose_response_20260729" / "pilot_results_20260730"

TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
UNCHEATABLE_METRIC = "eval/uncheatable_eval/bpb"
TABLE9_METRIC = "olmo_base_easy/table9_macro_bpb"
EXPECTED_RUNS_PER_SCALE = 37
PILOT_EXPERIMENT_ID = "exp_01kypvqav0saymyc7mbrqfh5gc"
FULL_GRID_MULTIPLIERS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
FULL_GRID_POLICIES_PER_SCALE = 277
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
COLORS = {
    "dolma3_cc/art_and_design_high": "#d95f3d",
    "dolma3_arxiv": "#e1ad01",
    "dolmino_synth_math": "#1f8a70",
}
DOMAIN_LABELS = {
    "dolma3_cc/art_and_design_high": "CC art/design high",
    "dolma3_arxiv": "Dolma 3 arXiv",
    "dolmino_synth_math": "Dolmino synth math",
}
TARGETS = {
    "uncheatable": ("uncheatable_bpb", "Uncheatable BPB"),
    "table9": ("table9_macro_bpb", "Table-9 macro BPB"),
}


@dataclass(frozen=True)
class ScaleConfig:
    label: str
    train_filter: dict[str, object]
    eval_groups: tuple[str, ...]


SCALES = {
    "60m": ScaleConfig(
        label="60M / 1.2B tokens",
        train_filter={"tags": "pinlin_calvin_xu/data_mixture/be60p_20260729"},
        eval_groups=("olmo_base_eval_table9_bucket_epoch_dose_60m_pilot_20260729",),
    ),
    "delphi_3e18": ScaleConfig(
        label="Delphi 3e18",
        train_filter={"tags": "bucket-epoch-dose-response"},
        eval_groups=(
            "olmo_base_eval_table9_bucket_epoch_dose_delphi_pilot_20260729",
            "olmo_base_eval_table9_bucket_epoch_dose_delphi_3e18_pilot_20260729",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=180)
    return parser.parse_args()


def finite_summary(run: Any, key: str) -> float | None:
    try:
        value = float(run.summary.get(key, np.nan))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def run_matches(scale: str, run: Any, run_name: str) -> bool:
    if scale == "60m":
        return str(run.name).endswith(f"/{run_name}")
    return str(run.name).startswith(f"{run_name}-")


def selected_training_run(scale: str, runs: list[Any], run_name: str) -> Any:
    candidates = [
        run for run in runs if run_matches(scale, run, run_name) and finite_summary(run, UNCHEATABLE_METRIC) is not None
    ]
    if len(candidates) != 1:
        raise ValueError(f"{scale}/{run_name}: expected one finite training run, found {len(candidates)}")
    return candidates[0]


def eval_runs_by_name(runs: list[Any]) -> dict[str, Any]:
    candidates: dict[str, list[Any]] = {}
    for run in runs:
        provenance = dict(run.config.get("provenance") or {})
        run_name = str(provenance.get("run_name") or "")
        value = finite_summary(run, TABLE9_METRIC)
        if not run_name or value is None:
            continue
        candidates.setdefault(run_name, []).append(run)

    selected: dict[str, Any] = {}
    for run_name, attempts in candidates.items():
        values = [float(finite_summary(run, TABLE9_METRIC)) for run in attempts]
        if max(values) - min(values) > 1e-10:
            raise ValueError(f"{run_name}: successful Table-9 retries disagree: {values}")
        selected[run_name] = max(attempts, key=lambda run: str(run.created_at))
    return selected


def collect_observations(panel_dir: Path, timeout: int) -> pd.DataFrame:
    api = wandb.Api(timeout=timeout)
    frames: list[pd.DataFrame] = []
    for scale, config in SCALES.items():
        manifest = pd.read_csv(panel_dir / scale / "run_manifest.csv")
        if len(manifest) != EXPECTED_RUNS_PER_SCALE or manifest["run_name"].duplicated().any():
            raise ValueError(f"{scale}: expected {EXPECTED_RUNS_PER_SCALE} unique manifest rows")

        training_runs = list(api.runs(TRAIN_PROJECT, filters=config.train_filter, per_page=100))
        eval_runs: list[Any] = []
        for group in config.eval_groups:
            eval_runs.extend(api.runs(EVAL_PROJECT, filters={"group": group}, per_page=150))
        selected_evals = eval_runs_by_name(eval_runs)

        rows = []
        for _, spec in manifest.iterrows():
            run_name = str(spec["run_name"])
            training = selected_training_run(scale, training_runs, run_name)
            evaluation = selected_evals.get(run_name)
            if evaluation is None:
                raise ValueError(f"{scale}/{run_name}: no successful Table-9 evaluation")
            rows.append(
                {
                    **spec.to_dict(),
                    "uncheatable_bpb": finite_summary(training, UNCHEATABLE_METRIC),
                    "table9_macro_bpb": finite_summary(evaluation, TABLE9_METRIC),
                    "training_wandb_id": training.id,
                    "training_wandb_url": training.url,
                    "table9_wandb_id": evaluation.id,
                    "table9_wandb_url": evaluation.url,
                }
            )
        frame = pd.DataFrame(rows)
        if frame[["uncheatable_bpb", "table9_macro_bpb"]].isna().any().any():
            raise ValueError(f"{scale}: non-finite target after join")
        frames.append(frame)

    observations = pd.concat(frames, ignore_index=True)
    if len(observations) != 2 * EXPECTED_RUNS_PER_SCALE:
        raise ValueError("Unexpected joined observation count")
    return observations


def group_values(
    observations: pd.DataFrame,
    scale: str,
    target_column: str,
    run_names: list[str],
) -> np.ndarray:
    indexed = observations.loc[observations["scale"] == scale].set_index("run_name")
    missing = sorted(set(run_names) - set(indexed.index))
    if missing:
        raise ValueError(f"{scale}: missing noise rows {missing}")
    return indexed.loc[run_names, target_column].to_numpy(dtype=float)


def noise_components(observations: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "anchor · trainer seed": [f"a_t{seed}" for seed in range(6)],
        "anchor · subset seed": ["a_t0", *[f"a_u{seed}" for seed in range(30, 35)]],
        "art/design x16 · trainer seed": ["q00_m16_u29", *[f"q00_m16_t{seed}" for seed in range(1, 6)]],
        "art/design x16 · subset seed": [f"q00_m16_u{seed}" for seed in range(29, 32)],
    }
    rows = []
    for scale in SCALES:
        for target, (column, _) in TARGETS.items():
            for source, run_names in groups.items():
                values = group_values(observations, scale, column, run_names)
                rows.append(
                    {
                        "scale": scale,
                        "target": target,
                        "source": source,
                        "n": len(values),
                        "mean_bpb": float(values.mean()),
                        "sd_bpb": float(values.std(ddof=1)),
                        "range_bpb": float(values.max() - values.min()),
                    }
                )
    return pd.DataFrame(rows)


def paired_effects(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_name = {20260729: "a_t0", 20260730: "a_u30", 20260731: "a_u31"}
    effects = []
    pairs = []
    for scale in SCALES:
        scale_rows = observations.loc[observations["scale"] == scale]
        indexed = scale_rows.set_index("run_name")
        interventions = scale_rows.loc[
            scale_rows["seed_block"].isin(["high_replay_subset_seed", "low_dose_subset_seed"])
        ]
        for target, (column, _) in TARGETS.items():
            for (domain, multiplier), group in interventions.groupby(["focal_domain", "epoch_multiplier"], sort=True):
                deltas = []
                for _, row in group.sort_values("simulated_epoch_subset_seed").iterrows():
                    subset_seed = int(row["simulated_epoch_subset_seed"])
                    baseline = indexed.loc[baseline_name[subset_seed]]
                    delta = float(row[column] - baseline[column])
                    deltas.append(delta)
                    pairs.append(
                        {
                            "scale": scale,
                            "target": target,
                            "focal_domain": domain,
                            "epoch_multiplier": float(multiplier),
                            "subset_seed": subset_seed,
                            "intervention_run": row["run_name"],
                            "baseline_run": baseline.name,
                            "intervention_bpb": float(row[column]),
                            "baseline_bpb": float(baseline[column]),
                            "delta_bpb": delta,
                        }
                    )
                values = np.asarray(deltas)
                sem = float(stats.sem(values))
                interval = stats.t.interval(0.95, len(values) - 1, loc=float(values.mean()), scale=sem)
                effects.append(
                    {
                        "scale": scale,
                        "target": target,
                        "focal_domain": domain,
                        "epoch_multiplier": float(multiplier),
                        "n_pairs": len(values),
                        "mean_delta_bpb": float(values.mean()),
                        "sd_delta_bpb": float(values.std(ddof=1)),
                        "ci95_low": float(interval[0]),
                        "ci95_high": float(interval[1]),
                        "improved_pairs": int((values < 0).sum()),
                        "effect_to_paired_sd": (
                            float(abs(values.mean()) / values.std(ddof=1)) if values.std(ddof=1) > 0 else float("inf")
                        ),
                    }
                )
    return pd.DataFrame(effects), pd.DataFrame(pairs)


def cross_scale_summary(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["target", "focal_domain", "epoch_multiplier"]
    wide = effects.pivot(index=keys, columns="scale", values="mean_delta_bpb").reset_index()
    for target, group in wide.groupby("target"):
        rows.append(
            {
                "target": target,
                "n_interventions": len(group),
                "pearson_r": float(stats.pearsonr(group["60m"], group["delphi_3e18"]).statistic),
                "spearman_r": float(stats.spearmanr(group["60m"], group["delphi_3e18"]).statistic),
                "same_sign": int((np.sign(group["60m"]) == np.sign(group["delphi_3e18"])).sum()),
            }
        )
    return pd.DataFrame(rows)


def style_figure(figure: go.Figure, title: str, height: int = 760) -> None:
    figure.update_layout(
        title={"text": title, "x": 0.04, "xanchor": "left"},
        template="plotly_white",
        height=height,
        margin={"l": 80, "r": 40, "t": 110, "b": 80},
        font={"family": "Avenir Next, Helvetica Neue, sans-serif", "color": "#173042"},
        paper_bgcolor="#fbf8f0",
        plot_bgcolor="#fbf8f0",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hoverlabel={"font_size": 13},
    )
    figure.update_xaxes(gridcolor="#ded8ca", zerolinecolor="#173042")
    figure.update_yaxes(gridcolor="#ded8ca", zerolinecolor="#173042")


def write_effect_plot(effects: pd.DataFrame, output_path: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        row_titles=[SCALES["60m"].label, SCALES["delphi_3e18"].label],
        column_titles=[TARGETS["uncheatable"][1], TARGETS["table9"][1]],
        vertical_spacing=0.14,
    )
    for row_index, scale in enumerate(SCALES, start=1):
        for column_index, target in enumerate(TARGETS, start=1):
            panel = effects.loc[(effects["scale"] == scale) & (effects["target"] == target)]
            for domain, group in panel.groupby("focal_domain", sort=False):
                group = group.sort_values("epoch_multiplier")
                figure.add_trace(
                    go.Scatter(
                        x=group["epoch_multiplier"],
                        y=group["mean_delta_bpb"],
                        error_y={
                            "type": "data",
                            "symmetric": False,
                            "array": group["ci95_high"] - group["mean_delta_bpb"],
                            "arrayminus": group["mean_delta_bpb"] - group["ci95_low"],
                        },
                        mode="lines+markers",
                        name=DOMAIN_LABELS[domain],
                        legendgroup=domain,
                        showlegend=row_index == 1 and column_index == 1,
                        line={"color": COLORS[domain], "width": 3},
                        marker={"size": 10},
                        customdata=np.column_stack(
                            [
                                group["improved_pairs"],
                                group["n_pairs"],
                                group["sd_delta_bpb"],
                                group["ci95_low"],
                                group["ci95_high"],
                            ]
                        ),
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>multiplier=%{x:g}<br>"
                            "paired Δ=%{y:+.5f} BPB<br>95% t interval="
                            "[%{customdata[3]:+.5f}, %{customdata[4]:+.5f}]"
                            "<br>improved=%{customdata[0]:.0f}/%{customdata[1]:.0f}"
                            "<br>paired SD=%{customdata[2]:.5f}<extra></extra>"
                        ),
                    ),
                    row=row_index,
                    col=column_index,
                )
            figure.add_hline(y=0, line={"color": "#173042", "width": 2}, row=row_index, col=column_index)
            figure.update_xaxes(type="log", tickvals=[0.25, 1, 4, 16], row=row_index, col=column_index)
    figure.update_xaxes(title_text="Focal-bucket epoch multiplier")
    figure.update_yaxes(title_text="Paired Δ BPB vs proportional (lower is better)", col=1)
    style_figure(
        figure,
        "Conditional bucket-dose effects across scale"
        "<br><sup>Means and 95% t intervals over three matched simulated-subset seeds; n=3 is descriptive.</sup>",
        height=860,
    )
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def write_noise_plot(noise: pd.DataFrame, output_path: Path) -> None:
    figure = make_subplots(
        rows=1,
        cols=2,
        column_titles=[TARGETS["uncheatable"][1], TARGETS["table9"][1]],
    )
    palette = {"60m": "#e76f51", "delphi_3e18": "#2a9d8f"}
    for column_index, target in enumerate(TARGETS, start=1):
        panel = noise.loc[noise["target"] == target]
        for scale in SCALES:
            group = panel.loc[panel["scale"] == scale]
            figure.add_trace(
                go.Bar(
                    x=group["source"],
                    y=group["sd_bpb"],
                    name=SCALES[scale].label,
                    legendgroup=scale,
                    showlegend=column_index == 1,
                    marker_color=palette[scale],
                    customdata=np.column_stack([group["n"], group["range_bpb"]]),
                    hovertemplate=(
                        "<b>%{x}</b><br>SD=%{y:.6f} BPB<br>n=%{customdata[0]:.0f}"
                        "<br>range=%{customdata[1]:.6f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column_index,
            )
    figure.update_yaxes(title_text="Sample SD (BPB)", col=1)
    figure.update_xaxes(tickangle=-25)
    style_figure(
        figure,
        "Training and simulated-subset variance"
        "<br><sup>Shared rows make trainer and subset contrasts directly comparable at the anchor and x16 replay.</sup>",
        height=650,
    )
    figure.update_layout(barmode="group")
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def write_cross_scale_plot(effects: pd.DataFrame, output_path: Path) -> None:
    keys = ["target", "focal_domain", "epoch_multiplier"]
    wide = effects.pivot(index=keys, columns="scale", values="mean_delta_bpb").reset_index()
    figure = make_subplots(
        rows=1,
        cols=2,
        column_titles=[TARGETS["uncheatable"][1], TARGETS["table9"][1]],
    )
    for column_index, target in enumerate(TARGETS, start=1):
        panel = wide.loc[wide["target"] == target]
        limit = float(max(abs(panel["60m"]).max(), abs(panel["delphi_3e18"]).max()) * 1.12)
        figure.add_trace(
            go.Scatter(
                x=[-limit, limit],
                y=[-limit, limit],
                mode="lines",
                line={"color": "#8c8c8c", "dash": "dash"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=column_index,
        )
        for domain, group in panel.groupby("focal_domain", sort=False):
            figure.add_trace(
                go.Scatter(
                    x=group["60m"],
                    y=group["delphi_3e18"],
                    mode="markers+text",
                    text=[f"x{value:g}" for value in group["epoch_multiplier"]],
                    textposition="top center",
                    name=DOMAIN_LABELS[domain],
                    legendgroup=domain,
                    showlegend=column_index == 1,
                    marker={"size": 13, "color": COLORS[domain], "line": {"color": "#173042", "width": 1}},
                    customdata=group[["epoch_multiplier"]].to_numpy(),
                    hovertemplate=(
                        "<b>%{fullData.name}</b> x%{customdata[0]:g}<br>"
                        "60M Δ=%{x:+.5f}<br>Delphi Δ=%{y:+.5f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column_index,
            )
        figure.update_xaxes(range=[-limit, limit], row=1, col=column_index)
        figure.update_yaxes(range=[-limit, limit], row=1, col=column_index)
    figure.update_xaxes(title_text="60M paired Δ BPB")
    figure.update_yaxes(title_text="Delphi 3e18 paired Δ BPB", col=1)
    style_figure(
        figure,
        "Do conditional dose effects transfer across scale?"
        "<br><sup>Each point is a three-seed paired mean; diagonal means equal effect size.</sup>",
        height=650,
    )
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def report_text(
    observations: pd.DataFrame,
    noise: pd.DataFrame,
    effects: pd.DataFrame,
    transfer: pd.DataFrame,
) -> str:
    lines = [
        "# Bucket epoch-dose pilot results",
        "",
        "## Completion",
        "",
        "- Both pilot parents are terminal and successful.",
        "- Metric join: 37/37 rows per scale with finite Uncheatable and native Table-9 BPB.",
        "- This completes the variance-gated pilot, not the gated 277-policy full sweep.",
        "",
        "## Paired conditional effects",
        "",
        "Negative deltas improve over the proportional policy with the same simulated-subset seed.",
        "",
        effects.assign(
            domain=effects["focal_domain"].map(DOMAIN_LABELS),
            mean_delta_bpb=effects["mean_delta_bpb"].map(lambda value: f"{value:+.6f}"),
            ci95=lambda frame: frame.apply(lambda row: f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}]", axis=1),
        )[["scale", "target", "domain", "epoch_multiplier", "mean_delta_bpb", "ci95", "improved_pairs"]].to_markdown(
            index=False
        ),
        "",
        "## Variance audit",
        "",
        noise.assign(
            sd_bpb=noise["sd_bpb"].map(lambda value: f"{value:.6f}"),
            range_bpb=noise["range_bpb"].map(lambda value: f"{value:.6f}"),
        )[["scale", "target", "source", "n", "sd_bpb", "range_bpb"]].to_markdown(index=False),
        "",
        "## Cross-scale transfer",
        "",
        transfer.assign(
            pearson_r=transfer["pearson_r"].map(lambda value: f"{value:+.3f}"),
            spearman_r=transfer["spearman_r"].map(lambda value: f"{value:+.3f}"),
        ).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
    ]

    for target in TARGETS:
        target_effects = effects.loc[effects["target"] == target]
        best = target_effects.loc[target_effects["mean_delta_bpb"].idxmin()]
        worst = target_effects.loc[target_effects["mean_delta_bpb"].idxmax()]
        lines.append(
            f"- {TARGETS[target][1]}: the largest pilot improvement is "
            f"{DOMAIN_LABELS[best['focal_domain']]} x{best['epoch_multiplier']:g} at "
            f"{best['scale']} ({best['mean_delta_bpb']:+.6f} BPB); the largest degradation is "
            f"{DOMAIN_LABELS[worst['focal_domain']]} x{worst['epoch_multiplier']:g} at "
            f"{worst['scale']} ({worst['mean_delta_bpb']:+.6f} BPB)."
        )
    lines.extend(
        [
            "- CC art/design high x16 is consistently harmful on both targets and scales; its effect is much "
            "larger than either measured trainer- or subset-seed variation.",
            "- Dolma 3 arXiv replay improves Uncheatable at both scales, while Dolmino synth math replay is the "
            "strongest and most consistent Table-9 intervention.",
            "- The high cross-scale correlations are encouraging but partly leverage-driven by the large "
            "art/design x16 degradation; sign and rank transfer are more informative than the raw correlations "
            "alone in this seven-intervention pilot.",
            "- The three-seed intervention intervals are diagnostics, not confirmatory confidence statements.",
            "- A full 39 x dose grid should be justified by effect transfer and subset-noise behavior, not by selecting "
            "the best of these seven pilot interventions.",
            "",
            "## Artifacts",
            "",
            "- `observations.csv`: deduplicated checkpoint-level values and W&B provenance.",
            "- `paired_effects.csv` and `paired_rows.csv`: matched subset-seed effects.",
            "- `variance_components.csv`: trainer- and subset-seed noise estimates.",
            "- `dose_effects.html`, `noise_components.html`, and `cross_scale_transfer.html`: interactive figures.",
            "",
            f"Joined observation rows: {len(observations)}.",
        ]
    )
    return "\n".join(lines) + "\n"


def gate_evidence(
    observations: pd.DataFrame,
    noise: pd.DataFrame,
    effects: pd.DataFrame,
    transfer: pd.DataFrame,
) -> dict[str, object]:
    """Build the scale-specific advancement decision consumed by the full launcher."""
    approved_scales = []
    scale_diagnostics: dict[str, object] = {}
    for scale in SCALES:
        scale_effects = effects.loc[effects["scale"] == scale]
        scale_noise = noise.loc[noise["scale"] == scale]
        trainer_noise = scale_noise.loc[scale_noise["source"].str.contains("trainer seed"), "sd_bpb"]
        max_effect = float(scale_effects["mean_delta_bpb"].abs().max())
        max_trainer_sd = float(trainer_noise.max())
        max_paired_signal_to_noise = float(scale_effects["effect_to_paired_sd"].max())
        resolvable = max_effect > max_trainer_sd and max_paired_signal_to_noise > 1.0
        if resolvable:
            approved_scales.append(scale)
        scale_diagnostics[scale] = {
            "joined_rows": int((observations["scale"] == scale).sum()),
            "max_abs_paired_effect_bpb": max_effect,
            "max_trainer_sd_bpb": max_trainer_sd,
            "max_effect_to_paired_sd": max_paired_signal_to_noise,
            "resolvable": resolvable,
        }

    gate_status = "pass" if approved_scales == list(SCALES) else "fail"
    return {
        "schema_version": 1,
        "gate_status": gate_status,
        "pilot_experiment_id": PILOT_EXPERIMENT_ID,
        "approved_scales": approved_scales,
        "approved_stage": "full",
        "full_grid": {
            "multipliers": list(FULL_GRID_MULTIPLIERS),
            "focal_weight_cap": 0.5,
            "policies_per_scale": FULL_GRID_POLICIES_PER_SCALE,
            "x32_interpretation": (
                "Exploratory and right-censored: the pilot directly identified replay variance only through x16."
            ),
        },
        "decision_basis": {
            "scale_diagnostics": scale_diagnostics,
            "cross_scale_transfer": transfer.to_dict(orient="records"),
            "subset_variance_conclusion": (
                "High-replay subset variation does not dominate the observed effect for all probe buckets."
            ),
            "deployment_exclusion": (
                "Conditional one-bucket minima are not independently combined into a deployment mixture."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations = collect_observations(args.panel_dir, args.wandb_timeout)
    noise = noise_components(observations)
    effects, pairs = paired_effects(observations)
    transfer = cross_scale_summary(effects)

    observations.to_csv(args.output_dir / "observations.csv", index=False)
    noise.to_csv(args.output_dir / "variance_components.csv", index=False)
    effects.to_csv(args.output_dir / "paired_effects.csv", index=False)
    pairs.to_csv(args.output_dir / "paired_rows.csv", index=False)
    transfer.to_csv(args.output_dir / "cross_scale_summary.csv", index=False)
    write_effect_plot(effects, args.output_dir / "dose_effects.html")
    write_noise_plot(noise, args.output_dir / "noise_components.html")
    write_cross_scale_plot(effects, args.output_dir / "cross_scale_transfer.html")
    (args.output_dir / "pilot_gate_evidence.json").write_text(
        json.dumps(gate_evidence(observations, noise, effects, transfer), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        report_text(observations, noise, effects, transfer),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
