# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kaleido==0.2.1",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Collect and analyze the completed 2B StarCoder WSD80 aggregate fibers."""

from __future__ import annotations

import argparse
import json
import subprocess
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
PANEL_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_2b_complete_fibers_20260731"
DEFAULT_OUTPUT_DIR = PANEL_DIR / "results_20260731"
SCALE_FIBER_OBSERVATIONS = (
    REFERENCE_OUTPUTS / "starcoder_wsd80_scale_specific_tied_fibers_20260731" / "results_20260731" / "observations.csv"
)
SCALE_FRESH_CONFIRMATIONS = (
    REFERENCE_OUTPUTS
    / "starcoder_wsd80_scale_specific_tied_fibers_20260731"
    / "results_20260731"
    / "fresh_seed_confirmation.csv"
)

TRAIN_PROJECT = "marin-community/marin"
TRAIN_TAG = "starcoder_wsd80_2b_complete_fibers"
EXPECTED_NEW_RUNS = 29
EXPECTED_COMPLETE_COORDINATES = 51
REFERENCE_SEED = 20260711
TOKEN_BUDGET = 2_000_000_000
ANCHOR_METADATA = {
    0.35: (2, "measured_grid_minimum"),
    0.40: (3, "broad_basin_sensitivity"),
}
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=240)
    return parser.parse_args()


def persisted_final_metric(run: Any, key: str) -> tuple[float, int, str]:
    """Read the final metric from the checkpoint rather than W&B run state."""
    checkpoint_root = str(run.config["trainer"]["checkpointer"]["base_path"])
    uri = f"{checkpoint_root}/eval_metrics.jsonl"
    result = subprocess.run(
        ["gcloud", "storage", "cat", uri],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    finite = [row for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    if not finite:
        raise ValueError(f"{run.name}: no finite {key} in {uri}")
    final = max(finite, key=lambda row: int(row["step"]))
    return float(final[key]), int(final["step"]), uri


def collect_new_observations(
    panel_dir: Path,
    old_observations: pd.DataFrame,
    timeout: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    design = json.loads((panel_dir / "design_manifest.json").read_text(encoding="utf-8"))
    manifest = pd.DataFrame(design["runs"])
    if len(manifest) != EXPECTED_NEW_RUNS or manifest["run_name"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_NEW_RUNS} unique manifest rows")

    metric = str(design["objective_metric"])
    api = wandb.Api(timeout=timeout)
    runs = list(api.runs(TRAIN_PROJECT, filters={"tags": TRAIN_TAG}, per_page=100))
    by_name: dict[str, list[Any]] = {}
    for run in runs:
        by_name.setdefault(str(run.name), []).append(run)

    templates = {}
    for aggregate, (anchor_index, anchor_role) in ANCHOR_METADATA.items():
        candidates = old_observations.loc[
            old_observations["token_budget_requested"].eq(TOKEN_BUDGET)
            & np.isclose(old_observations["anchor_aggregate_starcoder"], aggregate)
            & old_observations["trainer_data_seed"].eq(REFERENCE_SEED)
        ]
        if candidates.empty:
            raise ValueError(f"Missing metadata template for aggregate {aggregate}")
        template = candidates.iloc[0].to_dict()
        template["anchor_index"] = anchor_index
        template["anchor_role"] = anchor_role
        templates[aggregate] = template

    rows = []
    for spec in manifest.to_dict("records"):
        candidates = by_name.get(str(spec["run_name"]), [])
        if len(candidates) != 1:
            raise ValueError(f"{spec['run_name']}: expected one W&B run, found {len(candidates)}")
        run = candidates[0]
        value, final_step, metric_uri = persisted_final_metric(run, metric)
        aggregate = float(spec["anchor_aggregate_starcoder"])
        anchor_index, anchor_role = ANCHOR_METADATA[aggregate]
        row = dict(templates[aggregate])
        row.update(spec)
        row.update(
            {
                "aggregate_starcoder_realized": (
                    float(spec["phase_0_fraction_realized"]) * float(spec["phase_0_starcoder"])
                    + float(spec["phase_1_fraction_realized"]) * float(spec["phase_1_starcoder"])
                ),
                "anchor_index": anchor_index,
                "anchor_role": anchor_role,
                "replicate_kind": "reference",
                "starcoder_bpb": value,
                "metric_source": "persisted eval_metrics.jsonl",
                "final_metric_step": final_step,
                "metric_uri": metric_uri,
                "wandb_id": str(run.id),
                "wandb_name": str(run.name),
                "wandb_state": str(run.state),
                "wandb_url": str(run.url),
                "observation_source": "complete 2B fiber panel",
            }
        )
        rows.append(row)

    observations = pd.DataFrame(rows).sort_values(["anchor_aggregate_starcoder", "signed_contrast_phase1_minus_phase0"])
    if len(observations) != EXPECTED_NEW_RUNS or observations["starcoder_bpb"].isna().any():
        raise ValueError("New observation collection is incomplete")
    expected_final_step = observations["total_steps"].astype(int) - 1
    if not observations["final_metric_step"].astype(int).eq(expected_final_step).all():
        raise ValueError("At least one checkpoint lacks its final-step metric")
    return observations.reset_index(drop=True), design


def complete_reference_fibers(
    old_observations: pd.DataFrame,
    new_observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = pd.concat([old_observations, new_observations], ignore_index=True, sort=False)
    keys = [
        "token_budget_requested",
        "phase_0_starcoder",
        "phase_1_starcoder",
        "trainer_data_seed",
        "simulated_epoch_subset_seed",
    ]
    if merged.duplicated(keys).any():
        duplicates = merged.loc[merged.duplicated(keys, keep=False), [*keys, "run_name"]]
        raise ValueError(f"Duplicate merged fiber observations:\n{duplicates}")

    reference = merged.loc[
        merged["token_budget_requested"].eq(TOKEN_BUDGET)
        & merged["trainer_data_seed"].eq(REFERENCE_SEED)
        & merged["simulated_epoch_subset_seed"].eq(REFERENCE_SEED)
        & merged["anchor_aggregate_starcoder"].isin(ANCHOR_METADATA)
    ].copy()
    reference = reference.sort_values(["anchor_aggregate_starcoder", "signed_contrast_phase1_minus_phase0"]).reset_index(
        drop=True
    )
    if len(reference) != EXPECTED_COMPLETE_COORDINATES:
        raise ValueError(f"Expected {EXPECTED_COMPLETE_COORDINATES} complete-fiber coordinates, got {len(reference)}")
    if reference.duplicated(["anchor_aggregate_starcoder", "signed_contrast_phase1_minus_phase0"]).any():
        raise ValueError("Complete reference fibers contain duplicate contrasts")
    return merged, reference


def fiber_deltas(reference: pd.DataFrame) -> pd.DataFrame:
    blocks = []
    for aggregate, block in reference.groupby("anchor_aggregate_starcoder", sort=True):
        tied_rows = block.loc[np.isclose(block["signed_contrast_phase1_minus_phase0"], 0.0)]
        if len(tied_rows) != 1:
            raise ValueError(f"Aggregate {aggregate}: expected one tied control")
        tied_bpb = float(tied_rows.iloc[0]["starcoder_bpb"])
        block = block.copy()
        block["delta_vs_tied_bpb"] = block["starcoder_bpb"] - tied_bpb
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def antithetic_decomposition(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for aggregate, block in deltas.groupby("anchor_aggregate_starcoder", sort=True):
        bpb_by_contrast = {
            float(row.signed_contrast_phase1_minus_phase0): float(row.starcoder_bpb) for row in block.itertuples()
        }
        tied_bpb = bpb_by_contrast[0.0]
        positive = sorted(value for value in bpb_by_contrast if value > 0)
        for magnitude in positive:
            mirrors = [value for value in bpb_by_contrast if np.isclose(value, -magnitude, atol=1e-10)]
            if not mirrors:
                continue
            plus_bpb = bpb_by_contrast[magnitude]
            minus_bpb = bpb_by_contrast[mirrors[0]]
            ordering = 0.5 * (plus_bpb - minus_bpb)
            cost = 0.5 * (plus_bpb + minus_bpb) - tied_bpb
            rows.append(
                {
                    "anchor_aggregate_starcoder": aggregate,
                    "abs_contrast": magnitude,
                    "tied_bpb": tied_bpb,
                    "minus_bpb": minus_bpb,
                    "plus_bpb": plus_bpb,
                    "ordering_effect": ordering,
                    "asymmetry_cost": cost,
                    "better_orientation_delta": cost - abs(ordering),
                }
            )
    return pd.DataFrame(rows)


def interval(values: np.ndarray) -> tuple[float, float]:
    half_width = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean() - half_width), float(values.mean() + half_width)


def repeated_arm_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = merged.loc[
        merged["token_budget_requested"].eq(TOKEN_BUDGET)
        & merged["anchor_aggregate_starcoder"].isin(ANCHOR_METADATA)
        & (
            np.isclose(merged["signed_contrast_phase1_minus_phase0"], -0.20)
            | np.isclose(merged["signed_contrast_phase1_minus_phase0"], 0.0)
            | np.isclose(merged["signed_contrast_phase1_minus_phase0"], 0.20)
        )
    ]
    for aggregate, block in selected.groupby("anchor_aggregate_starcoder", sort=True):
        pivot = block.pivot(
            index="trainer_data_seed",
            columns="signed_contrast_phase1_minus_phase0",
            values="starcoder_bpb",
        ).dropna()
        if len(pivot) != 5:
            raise ValueError(f"Aggregate {aggregate}: expected five complete repeated seeds")
        for contrast in (-0.20, 0.20):
            values = (pivot[contrast] - pivot[0.0]).to_numpy(dtype=float)
            low, high = interval(values)
            rows.append(
                {
                    "anchor_aggregate_starcoder": aggregate,
                    "contrast": contrast,
                    "seeds": len(values),
                    "mean_delta_vs_tied": float(values.mean()),
                    "sd_delta_vs_tied": float(values.std(ddof=1)),
                    "ci_low": low,
                    "ci_high": high,
                    "one_sided_p_improvement": float(stats.ttest_1samp(values, 0.0, alternative="less").pvalue),
                    "seeds_better": int((values < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def fiber_summary(deltas: pd.DataFrame, repeated: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for aggregate, block in deltas.groupby("anchor_aggregate_starcoder", sort=True):
        best = block.loc[block["starcoder_bpb"].idxmin()]
        tied = block.loc[np.isclose(block["signed_contrast_phase1_minus_phase0"], 0.0)].iloc[0]
        positive_confirmation = repeated.loc[
            np.isclose(repeated["anchor_aggregate_starcoder"], aggregate) & np.isclose(repeated["contrast"], 0.20)
        ].iloc[0]
        fresh_confirmation = fresh.loc[np.isclose(fresh["anchor_aggregate_starcoder"], aggregate)].iloc[0]
        rows.append(
            {
                "anchor_aggregate_starcoder": aggregate,
                "coordinate_count": len(block),
                "minimum_contrast": float(block["signed_contrast_phase1_minus_phase0"].min()),
                "maximum_contrast": float(block["signed_contrast_phase1_minus_phase0"].max()),
                "tied_bpb": float(tied["starcoder_bpb"]),
                "best_reference_contrast": float(best["signed_contrast_phase1_minus_phase0"]),
                "best_reference_bpb": float(best["starcoder_bpb"]),
                "best_reference_delta_vs_tied": float(best["delta_vs_tied_bpb"]),
                "reference_coordinates_better_than_tied": int((block["delta_vs_tied_bpb"] < 0).sum()),
                "plus_0p20_five_seed_mean_delta": float(positive_confirmation["mean_delta_vs_tied"]),
                "plus_0p20_five_seed_ci_low": float(positive_confirmation["ci_low"]),
                "plus_0p20_five_seed_ci_high": float(positive_confirmation["ci_high"]),
                "plus_0p20_five_seed_p": float(positive_confirmation["one_sided_p_improvement"]),
                "plus_0p20_seeds_better": int(positive_confirmation["seeds_better"]),
                "fresh_seed_mean_delta": float(fresh_confirmation["fresh_mean_delta_vs_tied"]),
                "fresh_seed_ci_low": float(fresh_confirmation["fresh_ci_low"]),
                "fresh_seed_ci_high": float(fresh_confirmation["fresh_ci_high"]),
                "fresh_seed_raw_p": float(fresh_confirmation["fresh_one_sided_p_improvement"]),
                "fresh_seed_holm_p_all_6": float(fresh_confirmation["holm_p_all_6_anchors"]),
                "fresh_seed_holm_p_primary_4": float(fresh_confirmation["holm_p_primary_4_anchors"]),
                "fresh_seeds_better": int(fresh_confirmation["fresh_seeds_better"]),
            }
        )
    return pd.DataFrame(rows)


def write_plot(deltas: pd.DataFrame, repeated: pd.DataFrame, output_path: Path) -> None:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Aggregate a=0.35", "Aggregate a=0.40"),
        horizontal_spacing=0.10,
    )
    colors = {0.35: "#2A9D8F", 0.40: "#E76F51"}
    for column, aggregate in enumerate(sorted(ANCHOR_METADATA), start=1):
        block = deltas.loc[np.isclose(deltas["anchor_aggregate_starcoder"], aggregate)]
        color = colors[aggregate]
        figure.add_trace(
            go.Scatter(
                x=block["signed_contrast_phase1_minus_phase0"],
                y=block["delta_vs_tied_bpb"],
                mode="lines+markers",
                line={"color": color, "width": 3},
                marker={"color": color, "size": 7, "line": {"color": "white", "width": 1}},
                name="reference-seed complete fiber",
                showlegend=column == 1,
                customdata=np.column_stack(
                    [block["phase_0_starcoder"], block["phase_1_starcoder"], block["starcoder_bpb"]]
                ),
                hovertemplate=(
                    "d=p1-p0=%{x:+.2f}<br>delta vs tied=%{y:+.6f} BPB"
                    "<br>p0=%{customdata[0]:.4f}<br>p1=%{customdata[1]:.4f}"
                    "<br>absolute BPB=%{customdata[2]:.6f}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        repeated_block = repeated.loc[np.isclose(repeated["anchor_aggregate_starcoder"], aggregate)]
        figure.add_trace(
            go.Scatter(
                x=repeated_block["contrast"],
                y=repeated_block["mean_delta_vs_tied"],
                mode="markers",
                marker={"symbol": "diamond", "size": 12, "color": "#173042"},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": repeated_block["ci_high"] - repeated_block["mean_delta_vs_tied"],
                    "arrayminus": repeated_block["mean_delta_vs_tied"] - repeated_block["ci_low"],
                    "thickness": 2,
                    "width": 5,
                },
                name="five-seed mean and 95% CI",
                showlegend=column == 1,
                hovertemplate="d=%{x:+.2f}<br>five-seed mean delta=%{y:+.6f} BPB<extra></extra>",
            ),
            row=1,
            col=column,
        )
        figure.add_hline(y=0.0, line={"color": "#173042", "dash": "dash", "width": 1.5}, row=1, col=column)
        figure.update_xaxes(title_text="contrast d = p1 - p0", row=1, col=column)
        figure.update_yaxes(title_text="BPB minus tied control", row=1, col=column)

    figure.update_layout(
        title={
            "text": (
                "Complete 2B StarCoder WSD80 fibers"
                "<br><sub>Moderate StarCoder-late contrast helps; both feasibility boundaries are worse. "
                "Lower is better.</sub>"
            ),
            "x": 0.03,
        },
        template="plotly_white",
        height=620,
        width=1400,
        margin={"l": 80, "r": 40, "t": 115, "b": 80},
        font={"family": "Avenir Next, Helvetica Neue, sans-serif", "color": "#173042"},
        paper_bgcolor="#fbf8f0",
        plot_bgcolor="#fbf8f0",
        legend={"orientation": "h", "y": -0.17, "x": 0.5, "xanchor": "center"},
    )
    figure.update_xaxes(gridcolor="#ded8ca")
    figure.update_yaxes(gridcolor="#ded8ca")
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)
    figure.write_image(output_path.with_suffix(".png"), scale=3)


def write_report(
    design: dict[str, Any],
    new_observations: pd.DataFrame,
    summary: pd.DataFrame,
    repeated: pd.DataFrame,
    fresh: pd.DataFrame,
    decomposition: pd.DataFrame,
    output_path: Path,
) -> None:
    crashed = int(new_observations["wandb_state"].ne("finished").sum())
    a35 = summary.loc[np.isclose(summary["anchor_aggregate_starcoder"], 0.35)].iloc[0]
    a40 = summary.loc[np.isclose(summary["anchor_aggregate_starcoder"], 0.40)].iloc[0]
    lines = [
        "# StarCoder WSD80 complete 2B fiber results",
        "",
        "## Completion",
        "",
        "- Iris parent: succeeded with exit 0, zero failures, and zero preemptions.",
        f"- New checkpoints: {len(new_observations)}/{design['design']['new_runs']} with final step-7628 metrics.",
        f"- W&B labels {crashed} runs as crashed; their final metrics were recovered from persisted checkpoint JSONL.",
        (
            f"- Complete reference-seed grids: {int(summary['coordinate_count'].sum())} coordinates "
            "across two aggregate fibers."
        ),
        "",
        "## Result",
        "",
        (
            "The complete grids strengthen the earlier counterexample but do not move the optimum toward a boundary. "
            f"At `a=0.35`, the best reference-seed point remains `d={a35['best_reference_contrast']:+.2f}` "
            f"with a {abs(a35['best_reference_delta_vs_tied']):.6f}-BPB improvement over tied. "
            f"At `a=0.40`, the best point remains `d={a40['best_reference_contrast']:+.2f}` "
            f"with a {abs(a40['best_reference_delta_vs_tied']):.6f}-BPB improvement."
        ),
        "",
        (
            "The inferential result fixes `d=+0.20` using the reference seed and evaluates four fresh seeds: "
            f"mean delta {a35['fresh_seed_mean_delta']:+.6f} BPB, "
            f"95% CI [{a35['fresh_seed_ci_low']:+.6f}, {a35['fresh_seed_ci_high']:+.6f}], "
            f"Holm p={a35['fresh_seed_holm_p_primary_4']:.6f}, with "
            f"{int(a35['fresh_seeds_better'])}/4 seeds better. Across all five seeds, the descriptive mean is "
            f"{a35['plus_0p20_five_seed_mean_delta']:+.6f} BPB. The new outer points are exploratory "
            "single-seed shape evidence."
        ),
        "",
        "## Hypothesis verdict",
        "",
        (
            "**Reject the hypothesis as a general empirical modeling constraint.** A policy on the complete "
            "sampled fiber "
            "through the best observed tied-grid anchor beats that anchor, and the `d=+0.20` improvement replicates. "
            "The exact population statement remains formally unresolved because the population tied optimum is unknown; "
            "the two sampled anchors only bracket its estimated basin."
        ),
        "",
        "The completion adds two useful restrictions:",
        "",
        "- The gain is localized to moderate positive contrast, not an endpoint artifact.",
        (
            "- Large negative contrast and large positive contrast are both harmful; a useful model needs "
            "aggregate-conditioned ordering benefit plus a growing asymmetry cost."
        ),
        "",
        "## Fiber summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Five-seed repeated arms",
        "",
        repeated.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Fresh-seed confirmation",
        "",
        fresh.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Antithetic decomposition",
        "",
        decomposition.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation boundary",
        "",
        (
            "- The new outer contrasts have one joint-randomness seed each; they resolve gross shape, "
            "not sub-noise differences."
        ),
        "- The grids stop just inside feasibility boundaries and do not test the exact endpoints.",
        "- Completing fibers at `a=0.35` and `a=0.40` does not identify the exact population tied optimum.",
        (
            "- The mathematical phase-weighted-dose theorem remains valid under its assumptions; these data "
            "reject imposing those assumptions as a hard empirical constraint."
        ),
        "",
        "## Provenance",
        "",
        f"- Objective: `{design['objective_metric']}`.",
        "- Aggregate matching uses the realized 2B 80/20 phase fractions.",
        "- Final metrics come from each checkpoint's `eval_metrics.jsonl`; W&B is used for identity and links only.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    old_observations = pd.read_csv(SCALE_FIBER_OBSERVATIONS)
    new_observations, design = collect_new_observations(args.panel_dir, old_observations, args.wandb_timeout)
    merged, reference = complete_reference_fibers(old_observations, new_observations)
    deltas = fiber_deltas(reference)
    decomposition = antithetic_decomposition(deltas)
    repeated = repeated_arm_summary(merged)
    fresh = pd.read_csv(SCALE_FRESH_CONFIRMATIONS)
    fresh = fresh.loc[fresh["token_budget_requested"].eq(TOKEN_BUDGET)].reset_index(drop=True)
    if len(fresh) != 2:
        raise ValueError("Expected two 2B fresh-seed confirmation rows")
    summary = fiber_summary(deltas, repeated, fresh)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    new_observations.to_csv(args.output_dir / "new_observations.csv", index=False)
    merged.to_csv(args.output_dir / "merged_scale_fiber_observations.csv", index=False)
    deltas.to_csv(args.output_dir / "complete_reference_fibers.csv", index=False)
    decomposition.to_csv(args.output_dir / "antithetic_decomposition.csv", index=False)
    repeated.to_csv(args.output_dir / "repeated_arm_summary.csv", index=False)
    fresh.to_csv(args.output_dir / "fresh_seed_confirmation.csv", index=False)
    summary.to_csv(args.output_dir / "fiber_summary.csv", index=False)
    write_plot(deltas, repeated, args.output_dir / "complete_2b_fibers.html")
    write_report(
        design,
        new_observations,
        summary,
        repeated,
        fresh,
        decomposition,
        args.output_dir / "report.md",
    )
    verdict = {
        "hypothesis": "No two-phase policy on the globally optimal tied policy's fiber can outperform it.",
        "verdict": "reject as a general empirical modeling constraint; exact population statement remains unresolved",
        "finite_grid_status": "contradicted at the best observed 2B tied-grid anchor",
        "tied_optimal_basin_status": "contradicted by the replicated a=0.35, d=+0.20 improvement",
        "population_status": "unresolved because the exact population tied optimum is unknown",
        "shape_result": (
            "both completed fibers are minimized at moderate positive contrast and worsen toward both boundaries"
        ),
        "modeling_recommendation": "use fiber optimality only as a falsifiable null or soft prior",
    }
    (args.output_dir / "hypothesis_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
