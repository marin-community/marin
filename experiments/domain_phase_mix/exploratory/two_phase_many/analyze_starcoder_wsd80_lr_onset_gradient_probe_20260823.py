# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gcsfs>=2025.5.1",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scipy>=1.14",
# ]
# ///
"""Analyze the frozen StarCoder WSD80 LR-onset gradient intervention."""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import gcsfs
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
INPUT_DIR = REPO_ROOT / "experiments/domain_phase_mix/manifests/starcoder_wsd80_lr_onset_gradient_probe_v1_20260823"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/starcoder_wsd80_lr_onset_gradient_probe_results_20260823"
RESULT_ROOT = (
    "gs://marin-us-central1/analysis/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_lr_onset_gradient_probe_v1_20260823"
)
VERSION = "2026.08.23.1"
RELEASE_SHA256 = "f9cf79331aab726a967d257e40ffbac09e3baf4bd7650baa1008f931addbb4fb"
ARMS = ("decay_0p60", "decay_0p80", "decay_0p90", "no_decay")
ARM_LABELS = {
    "decay_0p60": "Decay starts 0.60T",
    "decay_0p80": "Decay starts 0.80T",
    "decay_0p90": "Decay starts 0.90T",
    "no_decay": "No decay",
}
ARM_COLORS = {
    "decay_0p60": "#d73027",
    "decay_0p80": "#fc8d59",
    "decay_0p90": "#91cf60",
    "no_decay": "#1a9850",
}
TIMES = (0.55, 0.70, 0.80, 0.90, 0.95)
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 2_026_082_303
MAX_WORKERS = 64
TOTAL_STEPS = 3_820
PEAK_LEARNING_RATE = 0.02
PEAK_ADAM_LEARNING_RATE = 0.008
DECAY_ONSET_STEPS = {
    "decay_0p60": round(0.60 * TOTAL_STEPS),
    "decay_0p80": round(0.80 * TOTAL_STEPS),
    "decay_0p90": round(0.90 * TOTAL_STEPS),
    "no_decay": None,
}
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}


def _result_uri(row: Mapping[str, Any]) -> str:
    return f"{RESULT_ROOT}/{row['row_id']}/{VERSION}/result.json"


def _read_document(fs: gcsfs.GCSFileSystem, row: Mapping[str, Any]) -> dict[str, Any]:
    uri = _result_uri(row)
    with fs.open(uri.removeprefix("gs://"), "rb") as handle:
        document = json.load(handle)
    if document.get("release_sha256") != RELEASE_SHA256 or document.get("row") != row:
        raise RuntimeError(f"Output identity does not match the frozen row: {uri}")
    if document.get("endpoint_metrics_read") is not False:
        raise RuntimeError(f"Endpoint-result leakage marker is not false: {uri}")
    return document


def optimizer_step_learning_rate(arm: str, state_step: int, peak: float) -> float:
    onset = DECAY_ONSET_STEPS[arm]
    if onset is None or state_step <= onset:
        return peak
    progress = min(1.0, (state_step - onset) / (TOTAL_STEPS - onset))
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_documents() -> list[dict[str, Any]]:
    rows = json.loads((INPUT_DIR / "probe_manifest.json").read_text())
    if len(rows) != 192 or len({row["row_id"] for row in rows}) != 192:
        raise RuntimeError("Frozen manifest must contain exactly 192 unique rows")
    fs = gcsfs.GCSFileSystem()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        documents = list(executor.map(lambda row: _read_document(fs, row), rows))
    return documents


def flatten_documents(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for document in documents:
        metadata = document["row"]
        corrected = document["noise_corrected_source_gradient_statistics"]["projected"]["trunk"]
        unprojected = document["noise_corrected_source_gradient_statistics"]["raw"]["trunk"]
        raw = document["combined_source_gradient_statistics"]["projected"]["trunk"]
        update = document["combined_source_optimizer_update_statistics"]["projected"]["trunk"]
        lr = document["restored_learning_rates"]["observed"]
        state_step = int(document["restored_state_step"])
        rows.append(
            {
                "row_id": metadata["row_id"],
                "arm": metadata["arm"],
                "training_seed": int(metadata["training_seed"]),
                "checkpoint_step": int(metadata["checkpoint_step"]),
                "restored_state_step": state_step,
                "normalized_time": float(metadata["normalized_time"]),
                "steps_from_decay_onset": metadata["steps_from_decay_onset"],
                "learning_rate": float(lr["learning_rate"]),
                "adam_learning_rate": float(lr["adam_lr"]),
                "optimizer_update_learning_rate": optimizer_step_learning_rate(
                    metadata["arm"], state_step, PEAK_LEARNING_RATE
                ),
                "optimizer_update_adam_learning_rate": optimizer_step_learning_rate(
                    metadata["arm"], state_step, PEAK_ADAM_LEARNING_RATE
                ),
                "disattenuated_gradient_cosine": float(corrected["disattenuated_cosine"]),
                "gradient_cosine": float(raw["cosine"]),
                "unprojected_disattenuated_gradient_cosine": float(unprojected["disattenuated_cosine"]),
                "starcoder_gradient_norm": float(raw["left_norm"]),
                "nemotron_gradient_norm": float(raw["right_norm"]),
                "optimizer_update_cosine": float(update["cosine"]) if update["cosine_defined"] else np.nan,
                "starcoder_update_norm": float(update["left_norm"]),
                "nemotron_update_norm": float(update["right_norm"]),
                "starcoder_reliability": float(corrected["left_split_half_reliability"]),
                "nemotron_reliability": float(corrected["right_split_half_reliability"]),
                "starcoder_spearman_brown_reliability": float(corrected["left_spearman_brown_reliability"]),
                "nemotron_spearman_brown_reliability": float(corrected["right_spearman_brown_reliability"]),
            }
        )
    frame = pd.DataFrame(rows)
    expected = pd.MultiIndex.from_product(
        [ARMS, range(2_026_081_000, 2_026_081_008), range(6)],
        names=["arm", "training_seed", "time_index"],
    )
    if len(frame) != len(expected) or frame[["arm", "training_seed", "checkpoint_step"]].duplicated().any():
        raise RuntimeError("Flattened result inventory is incomplete or duplicated")
    if set(frame["arm"]) != set(ARMS) or frame.groupby(["arm", "training_seed"]).size().ne(6).any():
        raise RuntimeError("Each arm/seed must expose all six frozen checkpoints")
    if set(frame["training_seed"]) != set(range(2_026_081_000, 2_026_081_008)):
        raise RuntimeError("Training-seed inventory drifted")
    zero_update_lr = frame["optimizer_update_learning_rate"].eq(0.0)
    undefined_update = frame["optimizer_update_cosine"].isna()
    if not zero_update_lr.equals(undefined_update):
        raise RuntimeError("Optimizer-update cosine definedness does not match the next-step LR convention")
    return frame.sort_values(["arm", "training_seed", "normalized_time"]).reset_index(drop=True)


def bootstrap_interval(values: Sequence[float], *, key: str) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}:{key}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))
    means = array[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def paired_wilcoxon(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(stats.wilcoxon(array, alternative="two-sided", method="exact").pvalue)


def holm_adjust(values: Sequence[float]) -> list[float]:
    p_values = np.asarray(values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def at_time(frame: pd.DataFrame, arm: str, time: float, metric: str) -> pd.Series:
    selected = frame[frame["arm"].eq(arm) & np.isclose(frame["normalized_time"], time)]
    return selected.set_index("training_seed")[metric].sort_index()


def final(frame: pd.DataFrame, arm: str, metric: str) -> pd.Series:
    selected = frame[frame["arm"].eq(arm)].sort_values("normalized_time").groupby("training_seed").tail(1)
    return selected.set_index("training_seed")[metric].sort_index()


def summarize_trajectory(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, time), group in frame.groupby(["arm", "normalized_time"], sort=False):
        values = group["disattenuated_gradient_cosine"].to_numpy()
        low, high = bootstrap_interval(values, key=f"trajectory:{arm}:{time}")
        rows.append(
            {
                "arm": arm,
                "normalized_time": float(time),
                "n_seeds": len(values),
                "disattenuated_gradient_cosine_mean": float(values.mean()),
                "disattenuated_gradient_cosine_seed_sd": float(values.std(ddof=1)),
                "disattenuated_gradient_cosine_ci95_low": low,
                "disattenuated_gradient_cosine_ci95_high": high,
                "raw_gradient_cosine_mean": float(group["gradient_cosine"].mean()),
                "optimizer_update_cosine_mean": float(group["optimizer_update_cosine"].mean()),
                "learning_rate_mean": float(group["learning_rate"].mean()),
                "starcoder_reliability_min": float(group["starcoder_reliability"].min()),
                "nemotron_reliability_min": float(group["nemotron_reliability"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values(["arm", "normalized_time"])


def paired_contrast_summary(values: pd.Series, *, contrast: str, key: str) -> dict[str, Any]:
    low, high = bootstrap_interval(values, key=key)
    return {
        "contrast": contrast,
        "n_paired_seeds": len(values),
        "mean_difference": float(values.mean()),
        "seed_sd": float(values.std(ddof=1)),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "exact_wilcoxon_p_two_sided": paired_wilcoxon(values),
        "positive_pairs": int((values > 0).sum()),
    }


def repeated_measure_test(values: pd.DataFrame, *, name: str, estimate_definition: str) -> dict[str, Any]:
    if values.isna().any().any():
        return {
            "test": name,
            "statistic": np.nan,
            "p_raw": np.nan,
            "estimate": np.nan,
            "estimate_definition": estimate_definition,
            "estimable": False,
        }
    result = stats.friedmanchisquare(*(values[column] for column in values))
    return {
        "test": name,
        "statistic": float(result.statistic),
        "p_raw": float(result.pvalue),
        "estimate": float(values.mean(axis=0).max() - values.mean(axis=0).min()),
        "estimate_definition": estimate_definition,
        "estimable": True,
    }


def time_axis_tests(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    matched_fraction = pd.concat(
        {
            "decay_0p60_at_0p80": at_time(frame, "decay_0p60", 0.80, metric),
            "decay_0p80_at_0p90": at_time(frame, "decay_0p80", 0.90, metric),
            "decay_0p90_at_0p95": at_time(frame, "decay_0p90", 0.95, metric),
        },
        axis=1,
    )
    matched_time = pd.concat({arm: at_time(frame, arm, 0.90, metric) for arm in ARMS}, axis=1)
    onset_declines = pd.concat(
        {arm: at_time(frame, arm, 0.90, metric) - final(frame, arm, metric) for arm in ARMS}, axis=1
    )
    no_decay_stability = final(frame, "no_decay", metric) - at_time(frame, "no_decay", 0.55, metric)

    tests = [
        repeated_measure_test(
            matched_fraction,
            name="matched_decay_fraction_0p5",
            estimate_definition="range of arm means",
        ),
        repeated_measure_test(
            matched_time,
            name="matched_time_0p90",
            estimate_definition="range of arm means",
        ),
        repeated_measure_test(
            onset_declines,
            name="onset_anchored_0p90",
            estimate_definition="range of 0.90T-to-final decline means",
        ),
        {
            "test": "no_decay_stability",
            "statistic": float(no_decay_stability.mean()),
            "p_raw": paired_wilcoxon(no_decay_stability),
            "estimate": float(no_decay_stability.mean()),
            "estimate_definition": "final minus 0.55T mean cosine",
            "estimable": True,
        },
    ]
    test_frame = pd.DataFrame(tests)
    adjusted_input = test_frame["p_raw"].fillna(1.0).tolist()
    test_frame["p_holm"] = holm_adjust(adjusted_input)
    test_frame.loc[~test_frame["estimable"], "p_holm"] = np.nan
    return test_frame


def run_tests(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    primary = final(frame, "no_decay", "disattenuated_gradient_cosine") - final(
        frame, "decay_0p60", "disattenuated_gradient_cosine"
    )
    primary_low, primary_high = bootstrap_interval(primary, key="primary")
    primary_summary = {
        "contrast": "final no-decay minus final 0.60T-decay cosine",
        "n_paired_seeds": len(primary),
        "mean_difference": float(primary.mean()),
        "seed_sd": float(primary.std(ddof=1)),
        "bootstrap_ci95_low": primary_low,
        "bootstrap_ci95_high": primary_high,
        "exact_wilcoxon_p_two_sided": paired_wilcoxon(primary),
        "positive_pairs": int((primary > 0).sum()),
    }
    raw_primary = final(frame, "no_decay", "gradient_cosine") - final(frame, "decay_0p60", "gradient_cosine")
    unprojected_primary = final(frame, "no_decay", "unprojected_disattenuated_gradient_cosine") - final(
        frame, "decay_0p60", "unprojected_disattenuated_gradient_cosine"
    )
    sensitivity_summary = {
        "uncorrected_projected_gradient_cosine": paired_contrast_summary(
            raw_primary,
            contrast="final no-decay minus final 0.60T-decay uncorrected cosine",
            key="primary:uncorrected",
        ),
        "unprojected_disattenuated_gradient_cosine": paired_contrast_summary(
            unprojected_primary,
            contrast=(
                "final no-decay minus final 0.60T-decay unprojected split-half noise-corrected " "raw-gradient cosine"
            ),
            key="primary:unprojected",
        ),
    }
    paired = pd.concat(
        {
            "disattenuated_projected": primary,
            "uncorrected_projected": raw_primary,
            "disattenuated_unprojected": unprojected_primary,
        },
        axis=1,
    ).reset_index()
    frozen_secondary = time_axis_tests(frame, "optimizer_update_cosine")
    raw_gradient_sensitivity = time_axis_tests(frame, "gradient_cosine")
    return paired, frozen_secondary, raw_gradient_sensitivity, primary_summary, sensitivity_summary


def reliability_summary(frame: pd.DataFrame) -> dict[str, Any]:
    no_decay = frame[frame["arm"].eq("no_decay")]
    threshold = 0.5
    failures = no_decay[(no_decay["starcoder_reliability"] < threshold) | (no_decay["nemotron_reliability"] < threshold)]
    advisories = frame[
        ~frame["arm"].eq("no_decay")
        & ((frame["starcoder_reliability"] < threshold) | (frame["nemotron_reliability"] < threshold))
    ]
    decay_primary = frame[frame["arm"].eq("decay_0p60")].sort_values("normalized_time").groupby("training_seed").tail(1)
    primary_low_reliability = decay_primary[
        (decay_primary["starcoder_reliability"] < threshold) | (decay_primary["nemotron_reliability"] < threshold)
    ]
    primary_sb_pass = decay_primary[
        (decay_primary["starcoder_spearman_brown_reliability"] >= threshold)
        & (decay_primary["nemotron_spearman_brown_reliability"] >= threshold)
    ]
    return {
        "threshold": threshold,
        "no_decay_gate_passed": failures.empty,
        "no_decay_failure_rows": failures["row_id"].tolist(),
        "decay_arm_advisory_count": len(advisories),
        "decay_arm_advisory_rows": advisories["row_id"].tolist(),
        "primary_decay_rows_below_threshold": len(primary_low_reliability),
        "primary_decay_row_ids_below_threshold": primary_low_reliability["row_id"].tolist(),
        "primary_decay_rows_passing_spearman_brown_threshold": len(primary_sb_pass),
        "primary_decay_nemotron_spearman_brown_range": [
            float(decay_primary["nemotron_spearman_brown_reliability"].min()),
            float(decay_primary["nemotron_spearman_brown_reliability"].max()),
        ],
        "minimum_starcoder_reliability": float(frame["starcoder_reliability"].min()),
        "minimum_nemotron_reliability": float(frame["nemotron_reliability"].min()),
    }


def stationarity_diagnostic(frame: pd.DataFrame) -> dict[str, Any]:
    def summarize(arm: str, *, final_checkpoint: bool) -> dict[str, float]:
        selected = frame[frame["arm"].eq(arm)]
        if final_checkpoint:
            selected = selected.sort_values("normalized_time").groupby("training_seed").tail(1)
        else:
            selected = selected[np.isclose(selected["normalized_time"], 0.55)]
        starcoder = float(selected["starcoder_gradient_norm"].mean())
        nemotron = float(selected["nemotron_gradient_norm"].mean())
        return {
            "starcoder_gradient_norm_mean": starcoder,
            "nemotron_gradient_norm_mean": nemotron,
            "starcoder_to_nemotron_norm_ratio": starcoder / nemotron,
            "raw_gradient_cosine_mean": float(selected["gradient_cosine"].mean()),
        }

    return {
        "tied_starcoder_weight": 0.35,
        "stationary_weighted_mixture_ratio_prediction": 0.65 / 0.35,
        "shared_0p55T": summarize("no_decay", final_checkpoint=False),
        "no_decay_final": summarize("no_decay", final_checkpoint=True),
        "decay_0p60_final": summarize("decay_0p60", final_checkpoint=True),
        "interpretation": (
            "The decay arm moves toward the anti-aligned norm ratio implied by stationarity of the weighted mixture; "
            "this is a convergence diagnostic, not evidence that harmful optimizer-update conflict causes endpoint gain."
        ),
    }


def build_figure(trajectory: pd.DataFrame, frame: pd.DataFrame, primary: Mapping[str, Any]) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=(
            "Noise-corrected raw-gradient alignment",
            "Optimizer-update alignment",
            "Final raw-gradient alignment by seed",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )
    for arm in ARMS:
        group = trajectory[trajectory["arm"].eq(arm)]
        error = group["disattenuated_gradient_cosine_ci95_high"] - group["disattenuated_gradient_cosine_mean"]
        figure.add_trace(
            go.Scatter(
                x=group["normalized_time"],
                y=group["disattenuated_gradient_cosine_mean"],
                error_y={
                    "type": "data",
                    "array": error,
                    "arrayminus": (
                        group["disattenuated_gradient_cosine_mean"] - group["disattenuated_gradient_cosine_ci95_low"]
                    ),
                },
                mode="lines+markers",
                name=ARM_LABELS[arm],
                line={"color": ARM_COLORS[arm], "width": 3},
                marker={"size": 9},
                legendgroup=arm,
                hovertemplate="%{x:.3f}T<br>cosine %{y:.3f}<extra>%{fullData.name}</extra>",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=group["normalized_time"],
                y=group["optimizer_update_cosine_mean"],
                mode="lines+markers",
                name=ARM_LABELS[arm],
                line={"color": ARM_COLORS[arm], "width": 2},
                marker={"size": 7},
                legendgroup=arm,
                showlegend=False,
                hovertemplate="%{x:.3f}T<br>cosine %{y:.3f}<extra>%{fullData.name}</extra>",
            ),
            row=2,
            col=1,
        )

    endpoint = frame.sort_values("normalized_time").groupby(["arm", "training_seed"]).tail(1)
    for arm in ARMS:
        group = endpoint[endpoint["arm"].eq(arm)]
        figure.add_trace(
            go.Scatter(
                x=[ARM_LABELS[arm]] * len(group),
                y=group["gradient_cosine"],
                mode="markers",
                name=ARM_LABELS[arm],
                marker={"color": ARM_COLORS[arm], "size": 9, "opacity": 0.8},
                legendgroup=arm,
                showlegend=False,
                customdata=group["training_seed"],
                hovertemplate="seed %{customdata}<br>uncorrected cosine %{y:.3f}<extra>%{fullData.name}</extra>",
            ),
            row=2,
            col=2,
        )
    figure.update_xaxes(title_text="Training progress", range=[0.52, 1.02], tickformat=".2f", row=1, col=1)
    figure.update_yaxes(title_text="Noise-corrected raw-gradient cosine", range=[-1, 1], row=1, col=1)
    figure.update_xaxes(title_text="Training progress", range=[0.52, 1.02], tickformat=".2f", row=2, col=1)
    figure.update_yaxes(title_text="Optimizer-update cosine", range=[-1, 1], row=2, col=1)
    figure.update_xaxes(title_text="LR schedule", row=2, col=2)
    figure.update_yaxes(title_text="Final uncorrected cosine", range=[-1, 1], row=2, col=2)
    figure.update_layout(
        title=(
            "StarCoder WSD80 LR-onset intervention"
            f"<br><sup>Primary no-decay minus 0.60T-decay = {primary['mean_difference']:+.3f}; "
            f"95% CI [{primary['bootstrap_ci95_low']:+.3f}, {primary['bootstrap_ci95_high']:+.3f}]</sup>"
        ),
        template="plotly_white",
        paper_bgcolor="#f8f4ea",
        plot_bgcolor="#fffdf7",
        font={"family": "Avenir Next, sans-serif", "color": "#153047", "size": 15},
        height=1050,
        margin={"l": 80, "r": 40, "t": 115, "b": 70},
        legend={"orientation": "h", "y": 1.02, "x": 0.5, "xanchor": "center"},
    )
    return figure


def verdict(
    primary: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
    reliability: Mapping[str, Any],
) -> str:
    if not reliability["no_decay_gate_passed"]:
        return "Inconclusive: the frozen no-decay reliability gate failed."
    if reliability["primary_decay_rows_below_threshold"]:
        raw = sensitivity["uncorrected_projected_gradient_cosine"]
        if raw["exact_wilcoxon_p_two_sided"] < 0.05 and raw["bootstrap_ci95_low"] > 0:
            return (
                "The frozen split-half noise-corrected primary is formally inconclusive under its stated "
                "low-reliability rule, "
                "but the preregistered uncorrected sensitivity shows a large, seed-consistent LR-schedule effect."
            )
        return "Inconclusive: the frozen primary includes low-reliability decay-arm rows."
    if primary["exact_wilcoxon_p_two_sided"] < 0.05 and primary["bootstrap_ci95_low"] > 0:
        return (
            "Supported: moving LR decay to 0.60T causes substantially lower final StarCoder-Nemotron "
            "raw-gradient alignment than keeping LR constant."
        )
    return "Not supported: the preregistered final contrast does not clear its inferential gate."


def write_outputs(frame: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trajectory = summarize_trajectory(frame)
    paired, frozen_secondary, raw_gradient_sensitivity, primary, sensitivity = run_tests(frame)
    reliability = reliability_summary(frame)
    stationarity = stationarity_diagnostic(frame)
    conclusion = verdict(primary, sensitivity, reliability)
    summary = {
        "schema_version": "2026-08-23-starcoder-wsd80-lr-onset-analysis-v1",
        "release_sha256": RELEASE_SHA256,
        "row_count": len(frame),
        "primary": primary,
        "primary_sensitivities": sensitivity,
        "frozen_optimizer_update_secondary": frozen_secondary.to_dict(orient="records"),
        "postfreeze_raw_gradient_time_axis_sensitivity": raw_gradient_sensitivity.to_dict(orient="records"),
        "reliability": reliability,
        "stationarity_diagnostic": stationarity,
        "optimizer_update_definition": {
            "json_path": "combined_source_optimizer_update_statistics.projected.trunk.cosine",
            "meaning": (
                "Mean corrected data-induced optimizer update, where corrected means data_update minus the "
                "zero-gradient optimizer-memory update; combined means averaged across the two reference halves."
            ),
            "learning_rate_convention": (
                "The update counterfactual calls state.take_step at restored_state_step, so it uses the schedule at "
                "that next optimizer step. The adjacent learning_rate column records the preceding checkpoint "
                "schedule step; optimizer_update_learning_rate records the rate that scales the counterfactual."
            ),
        },
        "verdict": conclusion,
        "endpoint_metrics_read": False,
    }
    frozen_secondary_by_test = frozen_secondary.set_index("test").to_dict(orient="index")
    raw_sensitivity_by_test = raw_gradient_sensitivity.set_index("test").to_dict(orient="index")
    frame.to_csv(OUTPUT_DIR / "measurements.csv", index=False)
    trajectory.to_csv(OUTPUT_DIR / "trajectory_summary.csv", index=False)
    paired.to_csv(OUTPUT_DIR / "primary_paired_effects.csv", index=False)
    frozen_secondary.to_csv(OUTPUT_DIR / "frozen_optimizer_update_secondary.csv", index=False)
    raw_gradient_sensitivity.to_csv(OUTPUT_DIR / "postfreeze_raw_gradient_time_axis_sensitivity.csv", index=False)
    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    figure = build_figure(trajectory, frame, primary)
    plot = pio.to_html(figure, include_plotlyjs=True, full_html=False, config=PLOT_CONFIG)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StarCoder WSD80 LR-onset intervention</title>
<style>
body {{ margin:0; background:#f8f4ea; color:#153047; font-family:"Avenir Next",sans-serif; }}
main {{ max-width:1500px; margin:auto; padding:48px 30px 70px; }}
a {{ color:#b74624; font-weight:700; }}
h1 {{ font-family:Georgia,serif; font-size:clamp(2.2rem,5vw,4.8rem); margin:0 0 12px; }}
.lede {{ max-width:1000px; font-size:1.22rem; line-height:1.55; }}
.verdict {{ margin:30px 0; padding:24px 28px; background:#fffdf7; border-left:7px solid #d65a31; font-size:1.25rem; line-height:1.5; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; margin:24px 0; }}
.card {{ background:#fffdf7; border:1px solid #d7cdb9; padding:20px; }}
.value {{ font:700 1.8rem Georgia,serif; }}
.plot {{ background:#fffdf7; border:1px solid #d7cdb9; padding:10px; }}
small {{ color:#5b6e7c; }}
</style></head><body><main>
<!-- STUDY_BACK_LINK -->
<h1>Does gradient decline follow LR decay?</h1>
<p class="lede">Thirty-two tied 35% StarCoder trajectories differ only in when cosine LR decay begins: 0.60T, 0.80T, 0.90T, or never. At six matched checkpoints, fixed held-out StarCoder and frozen Nemotron reference panels measure source-gradient geometry. Eight training seeds pair every arm. No endpoint benchmark was read.</p>
<div class="verdict"><strong>Preregistered answer.</strong> {conclusion}</div>
<div class="grid">
<div class="card"><small>Frozen split-half noise-corrected primary</small><div class="value">{primary['mean_difference']:+.3f}</div><div>95% CI [{primary['bootstrap_ci95_low']:+.3f}, {primary['bootstrap_ci95_high']:+.3f}]</div></div>
<div class="card"><small>Uncorrected paired sensitivity</small><div class="value">{sensitivity['uncorrected_projected_gradient_cosine']['mean_difference']:+.3f}</div><div>95% CI [{sensitivity['uncorrected_projected_gradient_cosine']['bootstrap_ci95_low']:+.3f}, {sensitivity['uncorrected_projected_gradient_cosine']['bootstrap_ci95_high']:+.3f}]</div></div>
<div class="card"><small>Exact paired Wilcoxon</small><div class="value">p = {primary['exact_wilcoxon_p_two_sided']:.4g}</div><div>{primary['positive_pairs']}/8 paired effects positive</div></div>
<div class="card"><small>Reliability</small><div class="value">Contract-sensitive</div><div>No-decay gate passes; {reliability['primary_decay_rows_below_threshold']}/8 primary decay rows are below 0.5.</div></div>
</div>
<div class="plot">{plot}</div>
<p class="lede"><strong>Interpretation.</strong> Holding peak LR constant preserves StarCoder-Nemotron raw-gradient agreement, while earlier decay drives their finite-batch gradients toward anti-alignment as both norms collapse. The uncorrected contrast is large and repeats in every seed, so the direction is not created by reliability correction. The frozen split-half noise-corrected primary nevertheless cannot be called a clean pass because every 0.60T-decay endpoint has sub-threshold Nemotron split-half reliability. Corrected optimizer-update cosine remains positive while the update is nonzero and becomes undefined when the next optimizer step has zero scheduled LR.</p>
<p class="lede"><strong>What may generate the pattern.</strong> At stationarity of a tied 35%/65% objective, 0.35g<sub>SC</sub> + 0.65g<sub>N</sub> = 0 requires anti-alignment and a norm ratio of 1.857. The 0.60T-decay arm moves from ratio {stationarity['shared_0p55T']['starcoder_to_nemotron_norm_ratio']:.2f}, cosine {stationarity['shared_0p55T']['raw_gradient_cosine_mean']:+.2f}, to ratio {stationarity['decay_0p60_final']['starcoder_to_nemotron_norm_ratio']:.2f}, cosine {stationarity['decay_0p60_final']['raw_gradient_cosine_mean']:+.2f}. Thus the result is consistent with decay enabling convergence toward the weighted-mixture stationary geometry, not with harmful optimizer-update conflict.</p>
<p class="lede"><strong>Primary-statistic terminology.</strong> “Split-half noise-corrected raw-gradient cosine” is the descriptive name used here. For each source, the dot product between gradients from two non-overlapping reference-batch halves estimates squared signal norm; those signal norms replace the noisy observed norms in the cosine denominator. The frozen machine-readable field calls this <code>disattenuated_cosine</code>. The estimate can become unstable when split-half reliability is low, which is why reliability is reported separately.</p>
<p><small>Intervals resample the eight paired training seeds. The frozen optimizer-update family and post-freeze raw-gradient time-axis sensitivity are separated in adjacent CSV/JSON files. No endpoint benchmark was read.</small></p>
</main></body></html>"""
    (OUTPUT_DIR / "lr_onset_gradient_causality.html").write_text(document)

    report = f"""# StarCoder WSD80 LR-onset gradient intervention

## Result

{conclusion}

The preregistered split-half noise-corrected final contrast was **{primary['mean_difference']:+.6f}** cosine (no decay minus decay from 0.60T), with paired bootstrap 95% CI **[{primary['bootstrap_ci95_low']:+.6f}, {primary['bootstrap_ci95_high']:+.6f}]**, exact paired Wilcoxon **p={primary['exact_wilcoxon_p_two_sided']:.6g}**, and **{primary['positive_pairs']}/8** positive seed pairs.

The preregistered uncorrected sensitivity was **{sensitivity['uncorrected_projected_gradient_cosine']['mean_difference']:+.6f}**, 95% CI **[{sensitivity['uncorrected_projected_gradient_cosine']['bootstrap_ci95_low']:+.6f}, {sensitivity['uncorrected_projected_gradient_cosine']['bootstrap_ci95_high']:+.6f}]**, exact paired Wilcoxon **p={sensitivity['uncorrected_projected_gradient_cosine']['exact_wilcoxon_p_two_sided']:.6g}**, with **{sensitivity['uncorrected_projected_gradient_cosine']['positive_pairs']}/8** effects positive.

## Scope

- All 32 trajectories use the same tied 35% StarCoder mixture, model, token horizon, optimizer, data references, and eight seeds. Only LR-decay onset changes.
- The primary statistic is the frozen split-half noise-corrected projected-trunk raw-gradient cosine. For source gradients `g_s,A` and `g_s,B` from two non-overlapping reference-batch halves, the within-source dot product estimates squared signal norm. The cross-source dot product is divided by the geometric mean of those two signal-norm estimates. The frozen machine-readable field name is `disattenuated_cosine`; reports use the descriptive name.
- No endpoint metric was read. The experiment establishes gradient-geometry causality, not endpoint benefit.
- The no-decay reliability gate {'passed' if reliability['no_decay_gate_passed'] else 'failed'}, but all {reliability['primary_decay_rows_below_threshold']} 0.60T-decay endpoints have sub-threshold Nemotron reliability. The contract simultaneously calls decay-arm reliability advisory and says low reliability makes the primary inconclusive; the report therefore does not claim a clean frozen-primary pass.
- The Spearman-Brown reliability associated with this noise correction is above 0.5 for all {reliability['primary_decay_rows_passing_spearman_brown_threshold']} primary decay rows (Nemotron range {reliability['primary_decay_nemotron_spearman_brown_range'][0]:.3f}-{reliability['primary_decay_nemotron_spearman_brown_range'][1]:.3f}). The threshold itself was frozen on raw split-half reliability, so this supports numerical stability without rewriting the gate.
- The frozen secondary statistic is optimizer-update cosine. Its onset-to-final comparison is not estimable because terminal updates under cosine decay have zero norm. Raw-gradient time-axis tests are reported separately as post-freeze mechanistic sensitivities, not relabeled as frozen tests.
- At matched decay fraction, the optimizer-update range is **{frozen_secondary_by_test['matched_decay_fraction_0p5']['estimate']:.3f}** versus **{raw_sensitivity_by_test['matched_decay_fraction_0p5']['estimate']:.3f}** for raw gradients; at 0.90T the ranges are **{frozen_secondary_by_test['matched_time_0p90']['estimate']:.3f}** versus **{raw_sensitivity_by_test['matched_time_0p90']['estimate']:.3f}**. Statistical significance of the small optimizer differences is not evidence of material conflict.
- Here a corrected optimizer update means `data_update - zero_gradient_optimizer_memory_update`; `combined` means the two reference halves are averaged. The counterfactual uses the LR at `restored_state_step`, emitted as `optimizer_update_learning_rate`; the logged checkpoint LR is the preceding schedule step.
- The stable no-decay arm and onset-following raw-gradient divergence identify LR schedule as a cause of the measured geometry. Moving onset also changes cumulative LR, which was frozen as a proposed time axis but not analyzed, so the experiment does not distinguish onset timing from cumulative optimization distance.
- The 0.60T-decay endpoint has mean StarCoder/Nemotron norm ratio **{stationarity['decay_0p60_final']['starcoder_to_nemotron_norm_ratio']:.3f}** versus the weighted-stationarity prediction **{stationarity['stationary_weighted_mixture_ratio_prediction']:.3f}**, while both norms collapse. This favors a convergence interpretation over harmful source conflict.
- The experiment does not establish an endpoint-performance benefit or a two-phase mechanism; endpoint metrics were mechanically excluded.

## Artifacts

- Interactive report: `lr_onset_gradient_causality.html`
- Frozen measurements: `measurements.csv`
- Trajectory summaries: `trajectory_summary.csv`
- Primary paired effects: `primary_paired_effects.csv`
- Frozen optimizer-update secondary family: `frozen_optimizer_update_secondary.csv`
- Post-freeze raw-gradient time-axis sensitivity: `postfreeze_raw_gradient_time_axis_sensitivity.csv`
- Machine-readable result: `analysis_summary.json`
"""
    (OUTPUT_DIR / "report.md").write_text(report)


def main() -> None:
    write_outputs(flatten_documents(load_documents()))


if __name__ == "__main__":
    main()
