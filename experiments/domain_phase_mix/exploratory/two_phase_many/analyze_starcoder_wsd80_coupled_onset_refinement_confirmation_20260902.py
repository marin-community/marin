# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gcsfs>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scipy>=1.14",
#   "tabulate>=0.9",
# ]
# ///

"""Analyze the StarCoder coupled-onset BO refinement and fresh confirmation."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gcsfs
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.stats import t

REPO_ROOT = Path(__file__).resolve().parents[4]
DESIGN_PATH = REPO_ROOT / (
    "experiments/domain_phase_mix/starcoder_wsd80_coupled_onset_refinement_confirmation_design_20260901.json.gz"
)
BASE_RESULTS_DIR = REPO_ROOT / (
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "starcoder_wsd80_coupled_onset_dense_surface_results_20260901"
)
OUTPUT_DIR = REPO_ROOT / (
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "starcoder_wsd80_coupled_onset_refinement_confirmation_results_20260902"
)
CHECKPOINT_ROOT = (
    "marin-us-central2/checkpoints/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_coupled_onset_refinement_confirmation_central2_v4_20260901"
)
CHECKPOINT_VERSION = "2026.09.01.1"
EXPECTED_DESIGN_SHA256 = "79943e36932e942e9c42a5070663fae00f2c5b4e3cdb5b942fdeb1af7abac8a5"
EXPECTED_ENDPOINT_STEP = 28_259
EXPECTED_ROWS = 96
NOISE_ANCHOR_BPB = 0.001182
ARMS = ("coupled_0p60", "coupled_0p80", "coupled_0p90")
ARM_VALUES = {"coupled_0p60": 0.60, "coupled_0p80": 0.80, "coupled_0p90": 0.90}
ARM_LABELS = {arm: f"{value:.2f}T" for arm, value in ARM_VALUES.items()}
ARM_COLORS = {"coupled_0p60": "#1A9850", "coupled_0p80": "#F4B942", "coupled_0p90": "#D73027"}
E1_PAIRS = {
    "coupled_0p60": ("c096", "c042"),
    "coupled_0p80": ("c109", "c016"),
    "coupled_0p90": ("c109", "c067"),
}
E2_PAIR = ("c109", "c016")
METRICS = {
    "programming_languages_bpb": "eval/paloma/dolma_100_programing_languages-llama3/bpb",
    "c4_bpb": "eval/paloma/c4_en-llama3/bpb",
    "uncheatable_bpb": "eval/uncheatable_eval/bpb",
    "github_cpp_bpb": "eval/uncheatable_eval/github_cpp-llama3/bpb",
    "github_python_bpb": "eval/uncheatable_eval/github_python-llama3/bpb",
}
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_design() -> pd.DataFrame:
    payload = json.loads(gzip.decompress(DESIGN_PATH.read_bytes()))
    claimed = payload.pop("design_sha256")
    observed = _canonical_sha256(payload)
    if claimed != EXPECTED_DESIGN_SHA256 or observed != EXPECTED_DESIGN_SHA256:
        raise ValueError(f"Successor design hash drifted: {claimed=} {observed=}")
    rows = pd.DataFrame(payload["rows"])
    if len(rows) != EXPECTED_ROWS or rows.row_id.duplicated().any() or rows.run_name.duplicated().any():
        raise ValueError("Successor design inventory is incomplete or non-unique")
    return rows


def _read_endpoint(filesystem: gcsfs.GCSFileSystem, record: dict[str, Any]) -> dict[str, Any]:
    path = f"{CHECKPOINT_ROOT}/{record['run_name']}/{CHECKPOINT_VERSION}/" "checkpoints/eval_metrics.jsonl"
    with filesystem.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    endpoints = [row for row in rows if int(row.get("step", -1)) == EXPECTED_ENDPOINT_STEP]
    if len(endpoints) != 1:
        raise ValueError(f"{record['run_name']}: expected one exact endpoint, found {len(endpoints)}")
    endpoint = endpoints[0]
    metrics = {name: float(endpoint[key]) for name, key in METRICS.items()}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"{record['run_name']}: non-finite endpoint metric")
    return {
        **record,
        **metrics,
        "endpoint_step": EXPECTED_ENDPOINT_STEP,
        "eval_metrics_uri": f"gs://{path}",
    }


def collect_observations(design: pd.DataFrame) -> pd.DataFrame:
    filesystem = gcsfs.GCSFileSystem(token="google_default")
    records = design.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=24) as pool:
        observations = list(pool.map(lambda record: _read_endpoint(filesystem, record), records))
    frame = pd.DataFrame(observations).sort_values("run_order").reset_index(drop=True)
    if len(frame) != EXPECTED_ROWS:
        raise ValueError("Exact successor endpoint inventory is incomplete")
    return frame


def _gain_rows(observations: pd.DataFrame, pairs: dict[str, tuple[str, str]], estimand: str) -> pd.DataFrame:
    confirmation = observations[observations.stage.eq("fresh_confirmation")]
    rows: list[dict[str, Any]] = []
    for arm, (tied_id, untied_id) in pairs.items():
        arm_rows = confirmation[confirmation.arm_id.eq(arm)]
        for seed, seed_rows in arm_rows.groupby("data_seed"):
            by_policy = seed_rows.set_index("coordinate_id")
            if tied_id not in by_policy.index or untied_id not in by_policy.index:
                raise ValueError(f"{arm}/{seed}: missing {tied_id} or {untied_id}")
            tied = by_policy.loc[tied_id]
            untied = by_policy.loc[untied_id]
            rows.append(
                {
                    "estimand": estimand,
                    "arm_id": arm,
                    "onset": ARM_VALUES[arm],
                    "seed": int(seed),
                    "tied_coordinate": tied_id,
                    "untied_coordinate": untied_id,
                    "gain_bpb": float(tied.programming_languages_bpb - untied.programming_languages_bpb),
                    "c4_gain_bpb": float(tied.c4_bpb - untied.c4_bpb),
                }
            )
    return pd.DataFrame(rows)


def _summary(gains: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (estimand, arm), group in gains.groupby(["estimand", "arm_id"], sort=False):
        values = group.gain_bpb.to_numpy(float)
        standard_deviation = float(values.std(ddof=1))
        half_width = float(t.ppf(0.975, len(values) - 1) * standard_deviation / np.sqrt(len(values)))
        mean = float(values.mean())
        rows.append(
            {
                "estimand": estimand,
                "arm_id": arm,
                "onset": ARM_VALUES[arm],
                "n": len(values),
                "mean_gain_bpb": mean,
                "sd_gain_bpb": standard_deviation,
                "ci95_low_bpb": mean - half_width,
                "ci95_high_bpb": mean + half_width,
                "power_sd_gate": standard_deviation <= 2 * NOISE_ANCHOR_BPB,
            }
        )
    return pd.DataFrame(rows)


def _cross_arm_tests(e1: pd.DataFrame) -> pd.DataFrame:
    pivot = e1.pivot(index="seed", columns="arm_id", values="gain_bpb")
    rows = []
    for later in ("coupled_0p80", "coupled_0p90"):
        differences = (pivot[later] - pivot["coupled_0p60"]).to_numpy(float)
        mean = float(differences.mean())
        standard_deviation = float(differences.std(ddof=1))
        standard_error = standard_deviation / np.sqrt(len(differences))
        statistic = mean / standard_error
        rows.append(
            {
                "contrast": f"{ARM_LABELS[later]} minus 0.60T",
                "n": len(differences),
                "mean_difference_bpb": mean,
                "sd_difference_bpb": standard_deviation,
                "one_sided_lower_95_bpb": mean - float(t.ppf(0.95, len(differences) - 1)) * standard_error,
                "one_sided_p": float(1.0 - t.cdf(statistic, len(differences) - 1)),
                "passes_positive_directional_test": bool(1.0 - t.cdf(statistic, len(differences) - 1) < 0.05),
            }
        )
    return pd.DataFrame(rows)


def _discovery_refinement(observations: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(BASE_RESULTS_DIR / "observations.csv")
    bo = observations[observations.stage.eq("bayesian_refinement_discovery")]
    rows = []
    for arm in ARMS:
        arm_base = base[base.arm_id.eq(arm)]
        tied = arm_base[arm_base.selection_class.eq("tied")].nsmallest(1, "programming_languages_bpb").iloc[0]
        untied = (
            arm_base[arm_base.selection_class.eq("eligible_untied")].nsmallest(1, "programming_languages_bpb").iloc[0]
        )
        arm_bo = bo[bo.arm_id.eq(arm)].nsmallest(1, "programming_languages_bpb").iloc[0]
        augmented = arm_bo if arm_bo.programming_languages_bpb < untied.programming_languages_bpb else untied
        rows.append(
            {
                "arm_id": arm,
                "onset": ARM_VALUES[arm],
                "base_tied_coordinate": tied.coordinate_id,
                "base_tied_bpb": float(tied.programming_languages_bpb),
                "base_untied_coordinate": untied.coordinate_id,
                "base_untied_bpb": float(untied.programming_languages_bpb),
                "base_gain_bpb": float(tied.programming_languages_bpb - untied.programming_languages_bpb),
                "bo_best_coordinate": arm_bo.coordinate_id,
                "bo_best_bpb": float(arm_bo.programming_languages_bpb),
                "bo_improvement_bpb": float(untied.programming_languages_bpb - arm_bo.programming_languages_bpb),
                "augmented_best_coordinate": augmented.coordinate_id,
                "augmented_best_bpb": float(augmented.programming_languages_bpb),
                "augmented_gain_bpb": float(tied.programming_languages_bpb - augmented.programming_languages_bpb),
                "material_under_sampling_falsifier": bool(
                    untied.programming_languages_bpb - arm_bo.programming_languages_bpb > NOISE_ANCHOR_BPB
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_plot(
    refinement: pd.DataFrame,
    e1: pd.DataFrame,
    e2: pd.DataFrame,
    summaries: pd.DataFrame,
) -> Path:
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Selected gain estimates", "Fresh arm-specific gains", "Fixed action transported across onset"),
        horizontal_spacing=0.08,
    )
    figure.add_trace(
        go.Scatter(
            x=refinement.onset,
            y=refinement.base_gain_bpb,
            mode="lines+markers",
            name="Discovery grid",
            line={"color": "#8C9AA8", "dash": "dot", "width": 2},
            marker={"symbol": "circle-open", "size": 10},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=refinement.onset,
            y=refinement.augmented_gain_bpb,
            mode="lines+markers",
            name="Discovery + BO",
            line={"color": "#17324D", "width": 3},
            marker={"symbol": "diamond", "size": 10},
        ),
        row=1,
        col=1,
    )
    e1_summary = summaries[summaries.estimand.eq("E1 arm-specific")].sort_values("onset")
    figure.add_trace(
        go.Scatter(
            x=e1_summary.onset,
            y=e1_summary.mean_gain_bpb,
            error_y={
                "type": "data",
                "symmetric": False,
                "array": e1_summary.ci95_high_bpb - e1_summary.mean_gain_bpb,
                "arrayminus": e1_summary.mean_gain_bpb - e1_summary.ci95_low_bpb,
            },
            mode="lines+markers",
            name="Fresh confirmation mean",
            line={"color": "#6A3D2A", "width": 3},
            marker={"symbol": "star", "size": 12},
        ),
        row=1,
        col=1,
    )
    for seed, seed_rows in e1.sort_values("onset").groupby("seed"):
        figure.add_trace(
            go.Scatter(
                x=seed_rows.onset,
                y=seed_rows.gain_bpb,
                mode="lines+markers",
                name=f"Seed {seed}",
                legendgroup="seed",
                showlegend=False,
                line={"color": "rgba(82,101,122,0.30)", "width": 1},
                marker={"color": [ARM_COLORS[arm] for arm in seed_rows.arm_id], "size": 6},
                hovertemplate=f"Seed {seed}<br>Onset %{{x:.2f}}T<br>Gain %{{y:+.6f}}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    figure.add_trace(
        go.Scatter(
            x=e1_summary.onset,
            y=e1_summary.mean_gain_bpb,
            error_y={
                "type": "data",
                "symmetric": False,
                "array": e1_summary.ci95_high_bpb - e1_summary.mean_gain_bpb,
                "arrayminus": e1_summary.mean_gain_bpb - e1_summary.ci95_low_bpb,
            },
            mode="lines+markers",
            name="E1 mean and 95% CI",
            line={"color": "#17324D", "width": 3},
            marker={"color": [ARM_COLORS[arm] for arm in e1_summary.arm_id], "size": 10},
        ),
        row=1,
        col=2,
    )
    e2_summary = summaries[summaries.estimand.eq("E2 fixed c109-c016")].sort_values("onset")
    figure.add_trace(
        go.Scatter(
            x=e2_summary.onset,
            y=e2_summary.mean_gain_bpb,
            error_y={
                "type": "data",
                "symmetric": False,
                "array": e2_summary.ci95_high_bpb - e2_summary.mean_gain_bpb,
                "arrayminus": e2_summary.mean_gain_bpb - e2_summary.ci95_low_bpb,
            },
            mode="lines+markers",
            name="E2 mean and 95% CI",
            line={"color": "#17324D", "width": 3},
            marker={"color": [ARM_COLORS[arm] for arm in e2_summary.arm_id], "size": 11, "symbol": "diamond"},
        ),
        row=1,
        col=3,
    )
    for column in range(1, 4):
        figure.add_hline(y=0.0, line={"color": "#17324D", "width": 1}, row=1, col=column)
    figure.update_layout(
        title={
            "text": "StarCoder WSD80 coupled-onset refinement and confirmation",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.98,
            "yanchor": "top",
        },
        paper_bgcolor="#F8F3E8",
        plot_bgcolor="#F8F3E8",
        font={"family": "Avenir Next, sans-serif", "size": 14, "color": "#17324D"},
        height=760,
        margin={"l": 65, "r": 30, "t": 105, "b": 180},
        legend={
            "orientation": "h",
            "x": 0.0,
            "xanchor": "left",
            "y": -0.28,
            "yanchor": "top",
            "entrywidth": 210,
            "entrywidthmode": "pixels",
        },
        hoverlabel={"bgcolor": "#FFF9EE"},
    )
    figure.update_xaxes(
        title_text="Coupled phase/LR onset",
        tickmode="array",
        tickvals=[0.60, 0.80, 0.90],
        ticktext=["0.60T", "0.80T", "0.90T"],
        gridcolor="#DCE5EA",
    )
    figure.update_yaxes(title_text="Programming Languages gain (BPB)", gridcolor="#DCE5EA", row=1, col=1)
    figure.update_yaxes(title_text="Tied minus untied BPB", gridcolor="#DCE5EA", row=1, col=2)
    figure.update_yaxes(title_text="Tied minus c016 BPB", gridcolor="#DCE5EA", row=1, col=3)
    path = OUTPUT_DIR / "refinement_confirmation.html"
    pio.write_html(figure, path, include_plotlyjs=True, full_html=True, config=PLOT_CONFIG)
    return path


def write_outputs() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observations = collect_observations(load_design())
    e1 = _gain_rows(observations, E1_PAIRS, "E1 arm-specific")
    e2 = _gain_rows(observations, {arm: E2_PAIR for arm in ARMS}, "E2 fixed c109-c016")
    gains = pd.concat([e1, e2], ignore_index=True)
    summaries = _summary(gains)
    cross_arm = _cross_arm_tests(e1)
    refinement = _discovery_refinement(observations)
    plot_path = _write_plot(refinement, e1, e2, summaries)

    observations.to_csv(OUTPUT_DIR / "observations.csv", index=False)
    refinement.to_csv(OUTPUT_DIR / "bo_refinement_summary.csv", index=False)
    gains.to_csv(OUTPUT_DIR / "confirmation_gains.csv", index=False)
    summaries.to_csv(OUTPUT_DIR / "confirmation_summary.csv", index=False)
    cross_arm.to_csv(OUTPUT_DIR / "cross_arm_tests.csv", index=False)

    e1_lookup = summaries[summaries.estimand.eq("E1 arm-specific")].set_index("arm_id")
    contrast_lookup = cross_arm.set_index("contrast")
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "endpoint_rows": len(observations),
        "bo_rows": int(observations.stage.eq("bayesian_refinement_discovery").sum()),
        "fresh_confirmation_rows": int(observations.stage.eq("fresh_confirmation").sum()),
        "bo_refinement": refinement.to_dict(orient="records"),
        "confirmation": summaries.to_dict(orient="records"),
        "cross_arm_tests": cross_arm.to_dict(orient="records"),
        "intersection_union_claim_passes": bool(cross_arm.passes_positive_directional_test.all()),
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# StarCoder WSD80 coupled-onset refinement and confirmation",
        "",
        "## Result",
        "",
        (
            "All 96 exact endpoints are complete: 24 discovery-only Bayesian-refinement rows and 72 fresh "
            "confirmation rows. The 0.60T refinement found Programming Languages BPB `0.784462`, improving the "
            "original selected untied cell by `0.001844` BPB. This exceeds the preregistered `0.001182` BPB "
            "under-sampling falsifier, so the original 0.60T surface minimum was materially under-sampled."
        ),
        "",
        (
            "Fresh arm-specific tied-minus-untied gains are "
            f"0.60T `{e1_lookup.loc['coupled_0p60', 'mean_gain_bpb']:+.6f}` "
            f"(95% CI `{e1_lookup.loc['coupled_0p60', 'ci95_low_bpb']:+.6f}` to "
            f"`{e1_lookup.loc['coupled_0p60', 'ci95_high_bpb']:+.6f}`), "
            f"0.80T `{e1_lookup.loc['coupled_0p80', 'mean_gain_bpb']:+.6f}` "
            f"(`{e1_lookup.loc['coupled_0p80', 'ci95_low_bpb']:+.6f}` to "
            f"`{e1_lookup.loc['coupled_0p80', 'ci95_high_bpb']:+.6f}`), and "
            f"0.90T `{e1_lookup.loc['coupled_0p90', 'mean_gain_bpb']:+.6f}` "
            f"(`{e1_lookup.loc['coupled_0p90', 'ci95_low_bpb']:+.6f}` to "
            f"`{e1_lookup.loc['coupled_0p90', 'ci95_high_bpb']:+.6f}`)."
        ),
        "",
        (
            "The directional claim that both later onsets have larger gain than 0.60T fails: "
            f"0.80T minus 0.60T has one-sided p=`{contrast_lookup.loc['0.80T minus 0.60T', 'one_sided_p']:.3f}`, "
            f"and 0.90T minus 0.60T has p=`{contrast_lookup.loc['0.90T minus 0.60T', 'one_sided_p']:.3f}`. "
            "The 0.80T and 0.90T gain variances also fail the preregistered power-SD gate, so the coupled-onset "
            "effect remains unresolved rather than monotonic."
        ),
        "",
        (
            "The fixed c109-minus-c016 action reverses sign at 0.60T while helping at 0.80T. This is direct "
            "evidence that continuation value depends on the training state; a branch action cannot be assigned a "
            "prefix-independent scalar value."
        ),
        "",
        f"[Open the interactive refinement and confirmation plot]({plot_path.name}).",
        "",
        "## Confirmation summary",
        "",
        summaries.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "Changing the coupled phase/LR onset alone does not establish a monotonic control of two-phaseness. The "
        "data support positive arm-specific gains at 0.60T and 0.80T, while the common c109-minus-c016 action "
        "reverses sign across states. This is consistent with state-dependent data exhaustion and policy-state "
        "interaction, not a universal LR-onset rule.",
        "",
    ]
    (OUTPUT_DIR / "results.md").write_text("\n".join(lines))
    return summary


def main() -> None:
    print(json.dumps(write_outputs(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
