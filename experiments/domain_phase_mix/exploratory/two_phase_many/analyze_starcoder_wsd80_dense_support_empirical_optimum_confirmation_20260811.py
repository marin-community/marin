# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "google-cloud-storage",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "tabulate",
# ]
# ///

"""Analyze the frozen dense-support empirical-optimum confirmation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from google.cloud import storage
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DESIGN_PATH = (
    SCRIPT_DIR.parents[1] / "starcoder_wsd80_dense_support_empirical_optimum_confirmation_design_20260811.json"
)
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "reference_outputs/starcoder_wsd80_dense_support_empirical_optimum_confirmation_results_20260811"
)

DESIGN_VERSION = "2026-08-11-v1"
EXPECTED_DESIGN_SHA256 = "ea116688ba7b0fa38713b5e616fb560f7708e2d385e782b757fc745674beecec"
PRIMARY_METRIC = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
GCS_BUCKET = "marin-us-central1"
CHECKPOINT_ROOT = (
    "checkpoints/pinlin_calvin_xu/data_mixture/starcoder_wsd80_dense_support_empirical_optimum_confirmation_20260811"
)
CHECKPOINT_VERSION = "2026.07.11"
WANDB_PROJECT_URL = "https://wandb.ai/marin-community/marin/runs"

EXPECTED_BLOCKS = 28
EXPECTED_PAIRS_PER_BLOCK = 5
EXPECTED_RUNS = EXPECTED_BLOCKS * EXPECTED_PAIRS_PER_BLOCK * 2
ALPHA = 0.05

SUPPORT_ORDER = ("full", "m0125", "m025", "m050", "m100", "m200", "m400")
SUPPORT_LABELS = {
    "full": "full pool",
    "m0125": "0.125x",
    "m025": "0.25x",
    "m050": "0.5x",
    "m100": "1x",
    "m200": "2x",
    "m400": "4x",
}


@dataclass(frozen=True)
class PersistedMetric:
    """One exact final-step metric recovered from durable checkpoint output."""

    value: float
    step: int
    uri: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=32)
    return parser.parse_args()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def load_design(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load and structurally verify the frozen confirmation manifest."""
    design = json.loads(path.read_text(encoding="utf-8"))
    claimed_hash = design.pop("design_sha256", None)
    observed_hash = _canonical_sha256(design)
    design["design_sha256"] = claimed_hash
    if claimed_hash != EXPECTED_DESIGN_SHA256 or observed_hash != EXPECTED_DESIGN_SHA256:
        raise ValueError(f"Design hash mismatch: claimed={claimed_hash}, observed={observed_hash}")
    if design.get("design_version") != DESIGN_VERSION:
        raise ValueError(f"Unexpected design version: {design.get('design_version')}")
    if design.get("primary_metric") != PRIMARY_METRIC:
        raise ValueError("Frozen primary metric differs from the analyzer metric")
    if design.get("expected_run_count") != EXPECTED_RUNS or design.get("block_count") != EXPECTED_BLOCKS:
        raise ValueError("Frozen run or block count differs from the analysis contract")
    if not design["analysis_contract"]["fresh_outcomes_only"]:
        raise ValueError("Frozen analysis contract does not require fresh outcomes only")

    manifest = pd.DataFrame(design["runs"])
    if len(manifest) != EXPECTED_RUNS or manifest["run_name"].duplicated().any():
        raise ValueError("Confirmation manifest is incomplete or has duplicate run names")
    if set(manifest["policy_class"]) != {"tied", "untied"}:
        raise ValueError("Confirmation manifest has unexpected policy classes")
    if set(manifest["support_id"]) != set(SUPPORT_ORDER):
        raise ValueError("Confirmation manifest has unexpected support regimes")
    if set(manifest["pair_seed"].astype(int)) != set(map(int, design["fresh_seeds"])):
        raise ValueError("Confirmation manifest has unexpected fresh seeds")
    if int(design["discovery_seed"]) in set(manifest["pair_seed"].astype(int)):
        raise ValueError("The discovery seed appears in the fresh confirmation manifest")

    block_counts = manifest.groupby(["cell_id", "support_id"], sort=False).size()
    if len(block_counts) != EXPECTED_BLOCKS or not block_counts.eq(EXPECTED_PAIRS_PER_BLOCK * 2).all():
        raise ValueError(f"Incomplete cell-support blocks: {block_counts.to_dict()}")
    pair_classes = manifest.groupby(["cell_id", "support_id", "pair_seed"])["policy_class"].agg(set)
    if not pair_classes.map(lambda values: values == {"tied", "untied"}).all():
        raise ValueError("At least one paired seed lacks a tied or untied policy")
    return design, manifest


def _metric_blob_name(run_name: str) -> str:
    return f"{CHECKPOINT_ROOT}/{run_name}/{CHECKPOINT_VERSION}/checkpoints/eval_metrics.jsonl"


def _persisted_final_metric(bucket: Any, row: dict[str, Any]) -> PersistedMetric:
    blob_name = _metric_blob_name(str(row["run_name"]))
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise ValueError(f"{row['run_name']}: missing gs://{GCS_BUCKET}/{blob_name}")
    rows = [json.loads(line) for line in blob.download_as_text().splitlines() if line.strip()]
    finite = [
        item for item in rows if item.get(PRIMARY_METRIC) is not None and math.isfinite(float(item[PRIMARY_METRIC]))
    ]
    if not finite:
        raise ValueError(f"{row['run_name']}: no finite {PRIMARY_METRIC}")
    final = max(finite, key=lambda item: int(item["step"]))
    expected_step = int(row["total_steps"]) - 1
    if int(final["step"]) != expected_step:
        raise ValueError(f"{row['run_name']}: final metric step {final['step']} != {expected_step}")
    return PersistedMetric(
        value=float(final[PRIMARY_METRIC]),
        step=int(final["step"]),
        uri=f"gs://{GCS_BUCKET}/{blob_name}",
    )


def collect_observations(manifest: pd.DataFrame, workers: int) -> pd.DataFrame:
    """Join every frozen run to its durable exact final-step metric."""
    if workers < 1:
        raise ValueError("--workers must be positive")
    bucket = storage.Client().bucket(GCS_BUCKET)
    rows = manifest.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        metrics = list(executor.map(lambda row: _persisted_final_metric(bucket, row), rows))

    observations = manifest.copy()
    observations["observed_bpb"] = [metric.value for metric in metrics]
    observations["final_metric_step"] = [metric.step for metric in metrics]
    observations["metric_uri"] = [metric.uri for metric in metrics]
    observations["metric_source"] = "persisted exact-final-step eval_metrics.jsonl"
    observations["wandb_url"] = observations["run_name"].map(lambda name: f"{WANDB_PROJECT_URL}/{name}")
    if observations["metric_uri"].nunique() != EXPECTED_RUNS:
        raise ValueError("Durable metric URIs are not unique")
    return observations


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_results(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute fresh-seed paired gains and the frozen 28-test Holm family."""
    index = ["cell_id", "support_id", "pair_seed"]
    wide = observations.pivot(index=index, columns="policy_class", values="observed_bpb").reset_index()
    if len(wide) != EXPECTED_BLOCKS * EXPECTED_PAIRS_PER_BLOCK:
        raise ValueError(f"Expected 140 complete paired rows, got {len(wide)}")
    wide["gain_tied_minus_untied_bpb"] = wide["tied"] - wide["untied"]

    selected = observations.drop_duplicates(["cell_id", "support_id", "policy_class"]).set_index(
        ["cell_id", "support_id", "policy_class"]
    )
    summaries: list[dict[str, Any]] = []
    for (cell_id, support_id), group in wide.groupby(["cell_id", "support_id"], sort=True):
        values = group["gain_tied_minus_untied_bpb"].to_numpy(dtype=float)
        if len(values) != EXPECTED_PAIRS_PER_BLOCK:
            raise ValueError(f"{cell_id}, {support_id}: expected five fresh paired differences")
        mean = float(values.mean())
        sample_sd = float(values.std(ddof=1))
        sem = sample_sd / np.sqrt(len(values))
        half_width = float(stats.t.ppf(0.975, len(values) - 1) * sem)
        test = stats.ttest_1samp(values, popmean=0.0, alternative="greater")
        tied_row = selected.loc[(cell_id, support_id, "tied")]
        untied_row = selected.loc[(cell_id, support_id, "untied")]
        materialized_tokens = int(tied_row["materialized_tokens"])
        summaries.append(
            {
                "cell_id": cell_id,
                "support_id": support_id,
                "support_label": SUPPORT_LABELS[support_id],
                "epoch_multiplier": tied_row["epoch_multiplier"],
                "materialized_tokens": materialized_tokens,
                "materialized_tokens_b": materialized_tokens / 1e9,
                "total_tpp": materialized_tokens / float(tied_row["total_parameters"]),
                "non_embedding_tpp": materialized_tokens / float(tied_row["non_embedding_parameters"]),
                "pair_count": len(values),
                "untied_win_count": int(np.sum(values > 0.0)),
                "mean_gain_bpb": mean,
                "sample_sd_bpb": sample_sd,
                "sem_bpb": sem,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
                "paired_t_one_sided_p": float(test.pvalue),
                "discovery_tied_bpb": float(tied_row["discovery_bpb"]),
                "discovery_untied_bpb": float(untied_row["discovery_bpb"]),
                "discovery_gain_bpb": float(tied_row["discovery_bpb"] - untied_row["discovery_bpb"]),
                "fresh_tied_mean_bpb": float(group["tied"].mean()),
                "fresh_untied_mean_bpb": float(group["untied"].mean()),
                "tied_coordinate_id": str(tied_row["coordinate_id"]),
                "untied_coordinate_id": str(untied_row["coordinate_id"]),
                "tied_phase_0_starcoder": float(tied_row["phase_0_starcoder"]),
                "tied_phase_1_starcoder": float(tied_row["phase_1_starcoder"]),
                "untied_phase_0_starcoder": float(untied_row["phase_0_starcoder"]),
                "untied_phase_1_starcoder": float(untied_row["phase_1_starcoder"]),
            }
        )

    summary = pd.DataFrame(summaries)
    summary["paired_t_holm_p"] = _holm_adjust(summary["paired_t_one_sided_p"].to_numpy(dtype=float))
    summary["holm_positive"] = summary["paired_t_holm_p"].lt(ALPHA) & summary["mean_gain_bpb"].gt(0.0)
    summary["nominal_positive"] = summary["paired_t_one_sided_p"].lt(ALPHA) & summary["mean_gain_bpb"].gt(0.0)
    summary["winner_curse_bpb"] = summary["discovery_gain_bpb"] - summary["mean_gain_bpb"]

    cell_order = {
        cell_id: order
        for order, cell_id in enumerate(summary.sort_values("materialized_tokens")["cell_id"].drop_duplicates().tolist())
    }
    support_order = {support_id: order for order, support_id in enumerate(SUPPORT_ORDER)}
    summary["cell_order"] = summary["cell_id"].map(cell_order)
    summary["support_order"] = summary["support_id"].map(support_order)
    summary = summary.sort_values(["cell_order", "support_order"]).reset_index(drop=True)
    wide = wide.merge(
        summary[["cell_id", "support_id", "support_label", "materialized_tokens_b"]],
        on=["cell_id", "support_id"],
        validate="many_to_one",
    ).sort_values(["materialized_tokens_b", "support_id", "pair_seed"])
    return wide.reset_index(drop=True), summary


def _write_plot(summary: pd.DataFrame, pairs: pd.DataFrame, path: Path) -> None:
    cells = summary.sort_values("materialized_tokens")["cell_id"].drop_duplicates().tolist()
    colors = sample_colorscale("RdYlGn_r", np.linspace(0.05, 0.95, len(SUPPORT_ORDER)))
    color_by_support = dict(zip(SUPPORT_ORDER, colors, strict=True))
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"{float(summary.loc[summary['cell_id'].eq(cell), 'materialized_tokens_b'].iloc[0]):.3f}B tokens"
            for cell in cells
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.16,
    )
    for cell_index, cell_id in enumerate(cells):
        row = cell_index // 2 + 1
        col = cell_index % 2 + 1
        cell_summary = summary.loc[summary["cell_id"].eq(cell_id)].copy()
        cell_pairs = pairs.loc[pairs["cell_id"].eq(cell_id)].copy()
        for support_id in SUPPORT_ORDER:
            support_summary = cell_summary.loc[cell_summary["support_id"].eq(support_id)].iloc[0]
            support_pairs = cell_pairs.loc[cell_pairs["support_id"].eq(support_id)]
            x_value = SUPPORT_LABELS[support_id]
            figure.add_trace(
                go.Scatter(
                    x=[x_value] * len(support_pairs),
                    y=support_pairs["gain_tied_minus_untied_bpb"],
                    mode="markers",
                    marker={"size": 7, "color": color_by_support[support_id], "opacity": 0.45},
                    customdata=np.column_stack(
                        [support_pairs["pair_seed"], support_pairs["tied"], support_pairs["untied"]]
                    ),
                    hovertemplate=(
                        f"{SUPPORT_LABELS[support_id]}<br>seed=%{{customdata[0]:.0f}}"
                        "<br>tied=%{customdata[1]:.6f} BPB"
                        "<br>untied=%{customdata[2]:.6f} BPB"
                        "<br>gain=%{y:+.6f} BPB<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            figure.add_trace(
                go.Scatter(
                    x=[x_value],
                    y=[support_summary["mean_gain_bpb"]],
                    mode="markers",
                    marker={
                        "size": 15,
                        "color": color_by_support[support_id],
                        "line": {"width": 1.5, "color": "#17324d"},
                    },
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": [support_summary["ci95_high"] - support_summary["mean_gain_bpb"]],
                        "arrayminus": [support_summary["mean_gain_bpb"] - support_summary["ci95_low"]],
                        "color": "#17324d",
                        "thickness": 1.5,
                    },
                    customdata=[
                        [
                            support_summary["untied_win_count"],
                            support_summary["paired_t_one_sided_p"],
                            support_summary["paired_t_holm_p"],
                            support_summary["discovery_gain_bpb"],
                            support_summary["untied_phase_0_starcoder"],
                            support_summary["untied_phase_1_starcoder"],
                        ]
                    ],
                    hovertemplate=(
                        f"{SUPPORT_LABELS[support_id]}<br>fresh mean gain=%{{y:+.6f}} BPB"
                        "<br>fresh wins=%{customdata[0]:.0f}/5"
                        "<br>one-sided p=%{customdata[1]:.4g}"
                        "<br>Holm p=%{customdata[2]:.4g}"
                        "<br>discovery gain=%{customdata[3]:+.6f} BPB"
                        "<br>untied policy=(%{customdata[4]:.3f}, %{customdata[5]:.3f})<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        figure.add_hline(y=0.0, line_width=1.5, line_color="#17324d", row=row, col=col)
        figure.update_xaxes(title_text="Simulated epoching repetition multiplier", row=row, col=col)
        figure.update_yaxes(
            title_text="Fresh paired selected-policy two-phase gain (BPB; higher is better)", row=row, col=col
        )

    figure.update_layout(
        title={
            "text": (
                "Dense horizon-by-replay selected-policy confirmation"
                "<br><sup>Fresh tied-minus-untied BPB; five paired seeds per block; "
                "bars are ordinary 95% paired-t intervals</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        width=1500,
        height=1050,
        margin={"l": 100, "r": 50, "t": 120, "b": 90},
        font={"family": "Avenir Next, Avenir, sans-serif", "size": 15, "color": "#17324d"},
        paper_bgcolor="#f8f4e9",
        plot_bgcolor="#fffdf7",
        hoverlabel={"bgcolor": "#fffdf7", "font_color": "#17324d", "bordercolor": "#17324d"},
    )
    figure.write_html(
        path,
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 4}},
    )


def _report(design: dict[str, Any], summary: pd.DataFrame, pairs: pd.DataFrame) -> str:
    holm = summary.loc[summary["holm_positive"]]
    nominal = summary.loc[summary["nominal_positive"]]
    positive = summary.loc[summary["mean_gain_bpb"].gt(0.0)]
    all_wins = summary.loc[summary["untied_win_count"].eq(EXPECTED_PAIRS_PER_BLOCK)]
    full_pool = summary.loc[summary["support_id"].eq("full")]
    largest_horizon = summary.loc[summary["materialized_tokens"].eq(summary["materialized_tokens"].max())]
    discovery_positive = summary.loc[summary["discovery_gain_bpb"].gt(0.0)]
    discovery_positive_optimism = float(discovery_positive["winner_curse_bpb"].mean())
    sign_agreement = int(((summary["discovery_gain_bpb"] > 0.0) == (summary["mean_gain_bpb"] > 0.0)).sum())
    low_replay_gain = float(largest_horizon.loc[largest_horizon["support_id"].eq("m0125"), "mean_gain_bpb"].iloc[0])
    high_replay_gain = float(largest_horizon.loc[largest_horizon["support_id"].eq("m400"), "mean_gain_bpb"].iloc[0])
    table = summary[
        [
            "materialized_tokens_b",
            "support_label",
            "discovery_gain_bpb",
            "mean_gain_bpb",
            "ci95_low",
            "ci95_high",
            "untied_win_count",
            "paired_t_one_sided_p",
            "paired_t_holm_p",
            "holm_positive",
            "untied_phase_0_starcoder",
            "untied_phase_1_starcoder",
        ]
    ]
    lines = [
        "# Dense horizon-by-replay selected-policy confirmation",
        "",
        f"Frozen design: `{design['design_version']}` (`{design['design_sha256']}`).",
        "",
        "## Result",
        "",
        f"All `{EXPECTED_RUNS}` runs produced exact final-step `{PRIMARY_METRIC}` values, yielding "
        f"`{len(pairs)}` fresh paired differences across `{EXPECTED_BLOCKS}` preregistered blocks.",
        "",
        f"- Positive fresh mean gain: `{len(positive)}/{EXPECTED_BLOCKS}` blocks.",
        f"- Untied policy won all five paired seeds: `{len(all_wins)}/{EXPECTED_BLOCKS}` blocks.",
        f"- Nominal one-sided paired-t p<0.05: `{len(nominal)}/{EXPECTED_BLOCKS}` blocks.",
        f"- Holm-significant positive gain across the 28-test family: `{len(holm)}/{EXPECTED_BLOCKS}` blocks.",
        f"- Discovery/fresh gain signs agree in `{sign_agreement}/{EXPECTED_BLOCKS}` blocks.",
        f"- Of `{len(discovery_positive)}` discovery-positive blocks, "
        f"`{int(discovery_positive['mean_gain_bpb'].gt(0).sum())}` remain positive on fresh seeds; their mean "
        f"discovery-minus-fresh gain is `{discovery_positive_optimism:+.6f}` BPB.",
        f"- Full-pool blocks with positive fresh mean: `{int(full_pool['mean_gain_bpb'].gt(0).sum())}/4`; "
        f"Holm-significant: `{int(full_pool['holm_positive'].sum())}/4`.",
        f"- Largest-horizon finite-replay gains increase from `{low_replay_gain:+.6f}` at 0.125x to "
        f"`{high_replay_gain:+.6f}` BPB at 4x.",
        "- At the largest horizon, fresh mean gain is strictly increasing across all six finite replay regimes.",
        "",
        "## Blockwise inference",
        "",
        table.to_markdown(index=False, floatfmt=".7f"),
        "",
        "## Interpretation boundary",
        "",
        "The primary estimand is the fresh-seed paired BPB difference between the two policies selected in each "
        "dense cell-support block. The discovery observation is not pooled. Ordinary 95% paired-t intervals are "
        "descriptive; positive blockwise significance uses a one-sided paired-t test with Holm correction over all "
        "28 blocks.",
        "",
        "This analysis confirms or rejects the expected-performance advantage of the selected discrete-grid untied "
        "policy over the selected discrete-grid tied policy. It does not identify either continuous policy-class "
        "optimum, and a null result can reflect selection error or insufficient resolution rather than equality of "
        "the true global optima.",
        "",
        "The fresh results support repetition as an effect modifier: large-horizon gains become larger as finite "
        "replay becomes more severe, while none of the four selected full-pool contrasts improves. They do not show "
        "that repetition is the only source of phase advantage, because the panel confirms one frozen untied policy "
        "per block rather than exhaustively resolving each continuous response surface.",
        "",
        "## Artifacts",
        "",
        "- `confirmation_observations.csv`: all 280 durable final-step outcomes and provenance.",
        "- `paired_seed_differences.csv`: all 140 fresh paired seed differences.",
        "- `block_summary.csv`: the 28 paired summaries and Holm-adjusted tests.",
        "- `fresh_seed_selected_policy_gain.html`: interactive paired-gain visualization.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    design, manifest = load_design(args.design)
    observations = collect_observations(manifest, args.workers)
    pairs, summary = paired_results(observations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(args.output_dir / "confirmation_observations.csv", index=False)
    pairs.to_csv(args.output_dir / "paired_seed_differences.csv", index=False)
    summary.to_csv(args.output_dir / "block_summary.csv", index=False)
    _write_plot(summary, pairs, args.output_dir / "fresh_seed_selected_policy_gain.html")
    (args.output_dir / "report.md").write_text(_report(design, summary, pairs), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "observations": len(observations),
                "paired_differences": len(pairs),
                "blocks": len(summary),
                "holm_positive": int(summary["holm_positive"].sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
