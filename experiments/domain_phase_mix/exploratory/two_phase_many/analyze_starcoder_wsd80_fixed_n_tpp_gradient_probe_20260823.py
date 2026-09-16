# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gcsfs>=2025.5.1",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
# ]
# ///
"""Analyze the frozen fixed-N TPP gradient-onset probe panel."""

import csv
import hashlib
import itertools
import json
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

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_fixed_n_tpp_gradient_probe_v1_20260822"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_fixed_n_tpp_gradient_probe_results_20260823"
RESULT_ROOT = (
    "gs://marin-us-central1/analysis/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_fixed_n_tpp_gradient_probe_v1_20260822"
)
ARTIFACT_VERSION = "2026.08.22.1"
RELEASE_SHA256 = "32358ed236eb73117ebe50eed3aea055becdc59d4ff67acd5ef7ac21cc62ad28"
AUDIT_SHA256 = "009b69df156d1b66973078877aad46b38851efa6fa4554ae907f73bd7c0be2a9"
PARENT_INPUT_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_gradient_mechanism_repair_v10_20260818"
PARENT_RESULT_ROOT = (
    "gs://marin-us-central1/analysis/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_gradient_mechanism_repair_v10_20260818"
)
PARENT_ARTIFACT_VERSION = "2026.08.18.10"
PARENT_RELEASE_SHA256 = "051dc75c4ee6baa67b3df7f4ff305e4da8f83cadb5a1b3f18edf889176b00d3b"

CELL_ORDER = (
    "r0_shared_h0640_s03820",
    "r1_increase_d_h0640_s07320",
    "r2_increase_d_h0640_s14960",
    "r3_increase_d_h0640_s28260",
)
STATE_ORDER = (
    "fraction_0p55",
    "fraction_0p70",
    "decay_minus_256",
    "decay_minus_64",
    "decay_onset",
    "decay_plus_64",
    "decay_plus_256",
    "fraction_0p90",
)
COMMON_STATES = ("fraction_0p55", "fraction_0p70", "decay_onset", "fraction_0p90")
PLATEAU_STATES = ("fraction_0p55", "fraction_0p70")
TARGET_LABELS = {
    "paloma_c4_en": "C4",
    "paloma_programming_languages": "Programming Languages",
    "uncheatable_github_python": "GitHub Python",
    "uncheatable_wikipedia_english": "Wikipedia",
}
SOURCE_LABELS = {
    "nemotron_aggregate": "Nemotron",
    "starcoder_excluded_global": "StarCoder",
}
COLORS = ("#1a9850", "#91cf60", "#fdae61", "#d73027")
SOURCE_COLORS = {"nemotron_aggregate": "#1b7f79", "starcoder_excluded_global": "#d65a31"}
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 2_026_082_301
MAX_WORKERS = 64


def read_rows() -> list[dict[str, str]]:
    with (INPUT_DIR / "full_probe_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 256 or len({row["row_id"] for row in rows}) != 256:
        raise RuntimeError("Frozen manifest must contain exactly 256 unique rows")
    return rows


def read_document(fs: gcsfs.GCSFileSystem, row: Mapping[str, str]) -> dict[str, Any]:
    uri = f"{RESULT_ROOT}/full/{row['group_id']}/{ARTIFACT_VERSION}/rows/{row['row_id']}.json"
    with fs.open(uri.removeprefix("gs://"), "rb") as handle:
        document = json.load(handle)
    if document.get("release_sha256") != RELEASE_SHA256 or document.get("row") != row:
        raise RuntimeError(f"Output identity does not match the frozen manifest: {uri}")
    if document.get("endpoint_metrics_read") is not False:
        raise RuntimeError(f"Endpoint leakage marker is not false: {uri}")
    return document


def load_documents() -> list[dict[str, Any]]:
    fs = gcsfs.GCSFileSystem()
    rows = read_rows()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        documents = list(executor.map(lambda row: read_document(fs, row), rows))
    if len(documents) != 256:
        raise RuntimeError("Incomplete document inventory")
    return documents


def flatten_documents(
    documents: Sequence[Mapping[str, Any]], release: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    time_grid = release["design_validation"]["time_grid"]
    cell_tpp = release["design_validation"]["cell_tpp"]
    source_rows: list[dict[str, Any]] = []
    utility_rows: list[dict[str, Any]] = []
    target_gradient_rows: list[dict[str, Any]] = []
    for document in documents:
        metadata = document["row"]
        cell = metadata["cell_id"]
        state = metadata["checkpoint_label"]
        common = {
            "row_id": metadata["row_id"],
            "cell_id": cell,
            "checkpoint_label": state,
            "training_seed": int(metadata["training_seed"]),
            "normalized_time": float(time_grid[cell][state]["normalized_time"]),
            "steps_from_lr_decay_onset": int(time_grid[cell][state]["steps_from_lr_decay_onset"]),
            "total_parameter_tpp": float(cell_tpp[cell]["total_parameter_tpp"]),
            "non_embedding_parameter_tpp": float(cell_tpp[cell]["non_embedding_parameter_tpp"]),
            "materialized_tokens": int(cell_tpp[cell]["materialized_tokens"]),
        }
        pair = document["source_pair_statistics"]["starcoder__vs__nemotron"]
        source_row = dict(common)
        for statistic in ("gradient", "optimizer_update"):
            for geometry in ("projected", "raw"):
                values = pair[statistic][geometry]["trunk"]
                if values.get("cosine_defined") is not True or values.get("cosine") is None:
                    raise RuntimeError(f"Undefined {statistic}/{geometry} cosine for {metadata['row_id']}")
                suffix = "" if geometry == "projected" else "_raw"
                source_row[f"{statistic}{suffix}_cosine"] = float(values["cosine"])
                source_row[f"{statistic}{suffix}_dot"] = float(values["dot"])
                source_row[f"{statistic}{suffix}_left_norm"] = float(values["left_norm"])
                source_row[f"{statistic}{suffix}_right_norm"] = float(values["right_norm"])
        source_rows.append(source_row)

        for target, sources in document["target_source_gradient_statistics"].items():
            for source, bundle in sources.items():
                row = {**common, "target": target, "source": source}
                for geometry in ("projected", "raw"):
                    values = bundle[geometry]["trunk"]
                    if values.get("cosine_defined") is not True or values.get("cosine") is None:
                        raise RuntimeError(
                            f"Undefined target-gradient {geometry} cosine for {metadata['row_id']}/{target}/{source}"
                        )
                    suffix = "" if geometry == "projected" else "_raw"
                    row[f"target_source_gradient{suffix}_cosine"] = float(values["cosine"])
                    row[f"target_source_gradient{suffix}_dot"] = float(values["dot"])
                    row[f"target_gradient{suffix}_norm"] = float(values["left_norm"])
                    row[f"source_gradient{suffix}_norm"] = float(values["right_norm"])
                target_gradient_rows.append(row)

        for target, sources in document["target_source_utility_statistics"].items():
            for source, bundle in sources.items():
                row = {**common, "target": target, "source": source}
                for geometry in ("projected", "raw"):
                    values = bundle[geometry]["trunk"]
                    if values.get("cosine_defined") is not True or values.get("cosine") is None:
                        raise RuntimeError(
                            f"Undefined utility {geometry} cosine for {metadata['row_id']}/{target}/{source}"
                        )
                    suffix = "" if geometry == "projected" else "_raw"
                    row[f"utility{suffix}_cosine"] = float(values["cosine"])
                    row[f"utility{suffix}_dot"] = float(values["dot"])
                    row[f"utility_target_gradient{suffix}_norm"] = float(values["left_norm"])
                    row[f"source_update{suffix}_norm"] = float(values["right_norm"])
                utility_rows.append(row)
    source = pd.DataFrame(source_rows)
    utility = pd.DataFrame(utility_rows)
    target_gradient = pd.DataFrame(target_gradient_rows)
    if len(source) != 256 or len(utility) != 2_048 or len(target_gradient) != 2_048:
        raise RuntimeError(
            "Flattened inventory mismatch: "
            f"source={len(source)}, utility={len(utility)}, target_gradient={len(target_gradient)}"
        )
    return source, utility, target_gradient


def bootstrap_interval(values: Sequence[float], *, key: str) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}:{key}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def exact_sign_flip_p(values: Sequence[float], *, alternative: str) -> float:
    array = np.asarray(values, dtype=float)
    observed = float(array.mean())
    null = np.asarray(
        [np.mean(array * np.asarray(signs)) for signs in itertools.product((-1.0, 1.0), repeat=len(array))]
    )
    if alternative == "greater":
        return float(np.mean(null >= observed - 1e-15))
    if alternative == "two_sided":
        return float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
    raise ValueError(f"Unknown alternative: {alternative}")


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def primary_effects(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    plateau = (
        source[source["checkpoint_label"].isin(PLATEAU_STATES)]
        .groupby(["cell_id", "training_seed"], as_index=True)["gradient_cosine"]
        .mean()
        .rename("plateau_cosine")
    )
    endpoint = (
        source[source["checkpoint_label"].eq("fraction_0p90")]
        .set_index(["cell_id", "training_seed"])["gradient_cosine"]
        .rename("cosine_0p90")
    )
    seed_effects = pd.concat([plateau, endpoint], axis=1).reset_index()
    seed_effects["decline"] = seed_effects["plateau_cosine"] - seed_effects["cosine_0p90"]
    tpp = source.groupby("cell_id")["total_parameter_tpp"].first()
    seed_effects["total_parameter_tpp"] = seed_effects["cell_id"].map(tpp)

    summaries: list[dict[str, Any]] = []
    for cell in CELL_ORDER:
        group = seed_effects[seed_effects["cell_id"].eq(cell)]
        values = group["decline"].to_numpy()
        low, high = bootstrap_interval(values, key=f"primary:{cell}")
        summaries.append(
            {
                "cell_id": cell,
                "total_parameter_tpp": float(group["total_parameter_tpp"].iloc[0]),
                "n_paired_seeds": len(values),
                "plateau_cosine_mean": float(group["plateau_cosine"].mean()),
                "cosine_0p90_mean": float(group["cosine_0p90"].mean()),
                "decline_mean": float(values.mean()),
                "decline_seed_sd": float(values.std(ddof=1)),
                "decline_bootstrap_ci95_low": low,
                "decline_bootstrap_ci95_high": high,
                "decline_exact_sign_flip_p_greater": exact_sign_flip_p(values, alternative="greater"),
                "positive_seed_pairs": int(np.sum(values > 0)),
            }
        )
    summary = pd.DataFrame(summaries)
    summary["decline_holm_p_greater"] = holm_adjust(summary["decline_exact_sign_flip_p_greater"])
    return seed_effects, summary


def summarize_trajectory(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (cell, state), group in source.groupby(["cell_id", "checkpoint_label"], sort=False):
        values = group["gradient_cosine"].to_numpy()
        low, high = bootstrap_interval(values, key=f"trajectory:{cell}:{state}")
        rows.append(
            {
                "cell_id": cell,
                "checkpoint_label": state,
                "normalized_time": float(group["normalized_time"].iloc[0]),
                "steps_from_lr_decay_onset": int(group["steps_from_lr_decay_onset"].iloc[0]),
                "total_parameter_tpp": float(group["total_parameter_tpp"].iloc[0]),
                "gradient_cosine_mean": float(values.mean()),
                "gradient_cosine_seed_sd": float(values.std(ddof=1)),
                "gradient_cosine_ci95_low": low,
                "gradient_cosine_ci95_high": high,
            }
        )
    frame = pd.DataFrame(rows)
    frame["cell_order"] = frame["cell_id"].map({cell: index for index, cell in enumerate(CELL_ORDER)})
    frame["state_order"] = frame["checkpoint_label"].map({state: index for index, state in enumerate(STATE_ORDER)})
    return frame.sort_values(["cell_order", "state_order"]).drop(columns=["cell_order", "state_order"])


def fit_hinge_profile(times: np.ndarray, values: np.ndarray) -> tuple[float, float, float, float]:
    tau_grid = np.linspace(0.55, 0.90, 1_401)
    hinge = np.maximum(times[None, :] - tau_grid[:, None], 0.0)
    centered_hinge = hinge - hinge.mean(axis=1, keepdims=True)
    centered_values = values - values.mean()
    denominator = np.sum(centered_hinge**2, axis=1)
    gamma = np.divide(
        -np.sum(centered_hinge * centered_values[None, :], axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    gamma = np.maximum(gamma, 0.0)
    alpha = values.mean() + gamma * hinge.mean(axis=1)
    residual = values[None, :] - (alpha[:, None] - gamma[:, None] * hinge)
    squared_error = np.sum(residual**2, axis=1)
    best = int(np.argmin(squared_error))
    tolerance = max(1e-12, float(squared_error[best]) * 1e-10)
    minimizing = np.flatnonzero(squared_error <= squared_error[best] + tolerance)
    return (
        float(tau_grid[minimizing[0]]),
        float(tau_grid[minimizing[-1]]),
        float(gamma[best]),
        float(alpha[best]),
    )


def hinge_summaries(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell in CELL_ORDER:
        group = source[source["cell_id"].eq(cell) & source["checkpoint_label"].isin(COMMON_STATES)]
        wide = group.pivot(index="training_seed", columns="normalized_time", values="gradient_cosine")
        times = wide.columns.to_numpy(dtype=float)
        seed_values = wide.to_numpy(dtype=float)
        tau_low, tau_high, gamma, alpha = fit_hinge_profile(times, seed_values.mean(axis=0))
        rows.append(
            {
                "cell_id": cell,
                "total_parameter_tpp": float(group["total_parameter_tpp"].iloc[0]),
                "hinge_tau_profile_low": tau_low,
                "hinge_tau_profile_high": tau_high,
                "hinge_gamma_point": gamma,
                "hinge_alpha_point": alpha,
            }
        )
    return pd.DataFrame(rows)


def secondary_summaries(source: pd.DataFrame, utility: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    optimizer_plateau = (
        source[source["checkpoint_label"].isin(PLATEAU_STATES)]
        .groupby(["cell_id", "training_seed"])["optimizer_update_cosine"]
        .mean()
    )
    optimizer_end = source[source["checkpoint_label"].eq("fraction_0p90")].set_index(["cell_id", "training_seed"])[
        "optimizer_update_cosine"
    ]
    optimizer_effect = (optimizer_plateau - optimizer_end).rename("decline").reset_index()
    optimizer_rows: list[dict[str, Any]] = []
    for cell in CELL_ORDER:
        values = optimizer_effect[optimizer_effect["cell_id"].eq(cell)]["decline"].to_numpy()
        low, high = bootstrap_interval(values, key=f"optimizer:{cell}")
        optimizer_rows.append(
            {
                "cell_id": cell,
                "decline_mean": float(values.mean()),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "positive_seed_pairs": int(np.sum(values > 0)),
            }
        )

    utility_plateau = (
        utility[utility["checkpoint_label"].isin(PLATEAU_STATES)]
        .groupby(["cell_id", "training_seed", "target", "source"])["utility_cosine"]
        .mean()
    )
    utility_end = utility[utility["checkpoint_label"].eq("fraction_0p90")].set_index(
        ["cell_id", "training_seed", "target", "source"]
    )["utility_cosine"]
    utility_effect = (utility_plateau - utility_end).rename("decline").reset_index()
    utility_rows: list[dict[str, Any]] = []
    for keys, group in utility_effect.groupby(["cell_id", "target", "source"], sort=False):
        cell, target, utility_source = keys
        values = group["decline"].to_numpy()
        low, high = bootstrap_interval(values, key=f"utility:{cell}:{target}:{utility_source}")
        utility_rows.append(
            {
                "cell_id": cell,
                "target": target,
                "source": utility_source,
                "decline_mean": float(values.mean()),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "positive_seed_pairs": int(np.sum(values > 0)),
                "exact_sign_flip_p_greater": exact_sign_flip_p(values, alternative="greater"),
            }
        )
    return pd.DataFrame(optimizer_rows), pd.DataFrame(utility_rows)


def decay_local_summaries(source: pd.DataFrame) -> pd.DataFrame:
    onset = source[source["checkpoint_label"].eq("decay_onset")].set_index(["cell_id", "training_seed"])[
        "gradient_cosine"
    ]
    rows: list[dict[str, Any]] = []
    for state in ("decay_plus_64", "decay_plus_256"):
        later = source[source["checkpoint_label"].eq(state)].set_index(["cell_id", "training_seed"])["gradient_cosine"]
        paired = (onset - later).rename("decline").reset_index()
        for cell in CELL_ORDER:
            values = paired[paired["cell_id"].eq(cell)]["decline"].to_numpy()
            low, high = bootstrap_interval(values, key=f"local:{cell}:{state}")
            time = source[source["cell_id"].eq(cell) & source["checkpoint_label"].eq(state)]["normalized_time"].iloc[0]
            rows.append(
                {
                    "cell_id": cell,
                    "checkpoint_label": state,
                    "normalized_time": float(time),
                    "onset_minus_later_mean": float(values.mean()),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "positive_seed_pairs": int(np.sum(values > 0)),
                    "exact_sign_flip_p_greater": exact_sign_flip_p(values, alternative="greater"),
                }
            )
    return pd.DataFrame(rows)


def onset_reference_summaries(source: pd.DataFrame) -> pd.DataFrame:
    index = ["cell_id", "training_seed"]
    plateau = source[source["checkpoint_label"].isin(PLATEAU_STATES)].groupby(index)["gradient_cosine"].mean()
    onset = source[source["checkpoint_label"].eq("decay_onset")].set_index(index)["gradient_cosine"]
    endpoint = source[source["checkpoint_label"].eq("fraction_0p90")].set_index(index)["gradient_cosine"]
    paired = pd.concat(
        {
            "pre_onset_rise": onset - plateau,
            "onset_to_0p90_decline": onset - endpoint,
        },
        axis=1,
    ).reset_index()
    rows: list[dict[str, Any]] = []
    for cell in CELL_ORDER:
        group = paired[paired["cell_id"].eq(cell)]
        for effect in ("pre_onset_rise", "onset_to_0p90_decline"):
            values = group[effect].to_numpy()
            low, high = bootstrap_interval(values, key=f"onset-reference:{cell}:{effect}")
            rows.append(
                {
                    "cell_id": cell,
                    "effect": effect,
                    "mean": float(values.mean()),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "positive_seed_pairs": int(np.sum(values > 0)),
                    "exact_sign_flip_p_greater": exact_sign_flip_p(values, alternative="greater"),
                }
            )
    return pd.DataFrame(rows)


def geometry_sensitivity(source: pd.DataFrame, target_gradient: pd.DataFrame) -> dict[str, float]:
    source_delta = np.abs(source["gradient_cosine"] - source["gradient_raw_cosine"])
    target_delta = np.abs(
        target_gradient["target_source_gradient_cosine"] - target_gradient["target_source_gradient_raw_cosine"]
    )
    return {
        "source_source_max_abs_projected_minus_raw_cosine": float(source_delta.max()),
        "source_source_mean_abs_projected_minus_raw_cosine": float(source_delta.mean()),
        "target_source_max_abs_projected_minus_raw_cosine": float(target_delta.max()),
        "target_source_mean_abs_projected_minus_raw_cosine": float(target_delta.mean()),
    }


def gradient_norm_summaries(source: pd.DataFrame) -> pd.DataFrame:
    selected = source[source["checkpoint_label"].isin(("decay_onset", "fraction_0p90"))]
    rows: list[dict[str, Any]] = []
    for (cell, state), group in selected.groupby(["cell_id", "checkpoint_label"], sort=False):
        rows.append(
            {
                "cell_id": cell,
                "checkpoint_label": state,
                "starcoder_raw_gradient_norm_mean": float(group["gradient_raw_left_norm"].mean()),
                "nemotron_raw_gradient_norm_mean": float(group["gradient_raw_right_norm"].mean()),
                "raw_gradient_cosine_mean": float(group["gradient_raw_cosine"].mean()),
            }
        )
    return pd.DataFrame(rows)


def target_gradient_summaries(target_gradient: pd.DataFrame) -> pd.DataFrame:
    index = ["cell_id", "training_seed", "target", "source"]
    plateau = (
        target_gradient[target_gradient["checkpoint_label"].isin(PLATEAU_STATES)]
        .groupby(index)["target_source_gradient_raw_cosine"]
        .mean()
    )
    endpoint = target_gradient[target_gradient["checkpoint_label"].eq("fraction_0p90")].set_index(index)[
        "target_source_gradient_raw_cosine"
    ]
    effects = (plateau - endpoint).rename("decline").reset_index()
    rows: list[dict[str, Any]] = []
    for keys, group in effects.groupby(["cell_id", "target", "source"], sort=False):
        cell, target, source_name = keys
        values = group["decline"].to_numpy()
        low, high = bootstrap_interval(values, key=f"target-gradient:{cell}:{target}:{source_name}")
        rows.append(
            {
                "cell_id": cell,
                "target": target,
                "source": source_name,
                "decline_mean": float(values.mean()),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "positive_seed_pairs": int(np.sum(values > 0)),
                "exact_sign_flip_p_greater": exact_sign_flip_p(values, alternative="greater"),
            }
        )
    return pd.DataFrame(rows)


def parent_precision_sensitivity(fs: gcsfs.GCSFileSystem, fixed_rows: Sequence[Mapping[str, str]]) -> pd.DataFrame:
    with (PARENT_INPUT_DIR / "full_mechanism_manifest.csv").open(newline="") as handle:
        parent_rows = list(csv.DictReader(handle))
    parent_index: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in parent_rows:
        key = (row["trajectory_id"], row["checkpoint_label"], row["training_seed"])
        if key in parent_index:
            raise RuntimeError(f"Duplicate parent precision row: {key}")
        parent_index[key] = row

    selected = [
        row
        for row in fixed_rows
        if row["cell_id"] == CELL_ORDER[-1] and row["checkpoint_label"] in {"fraction_0p55", "fraction_0p90"}
    ]
    measurements: list[dict[str, Any]] = []
    for fixed_row in selected:
        key = (
            fixed_row["trajectory_id"],
            fixed_row["checkpoint_label"],
            fixed_row["training_seed"],
        )
        parent_row = parent_index[key]
        parent_uri = (
            f"{PARENT_RESULT_ROOT}/full/{parent_row['group_id']}/{PARENT_ARTIFACT_VERSION}/"
            f"rows/{parent_row['row_id']}.json"
        )
        with fs.open(parent_uri.removeprefix("gs://"), "rb") as handle:
            document = json.load(handle)
        if document.get("release_sha256") != PARENT_RELEASE_SHA256 or document.get("row") != parent_row:
            raise RuntimeError(f"Parent precision identity mismatch: {parent_uri}")
        values = document["source_pair_statistics"]["starcoder__vs__nemotron"]["gradient"]["raw"]["trunk"]
        blocks = json.loads(parent_row["distribution_block_counts_json"])["starcoder_excluded_global"]
        if blocks != 32:
            raise RuntimeError(f"Expected 32-block parent precision, got {blocks}: {parent_uri}")
        measurements.append(
            {
                "checkpoint_label": fixed_row["checkpoint_label"],
                "training_seed": int(fixed_row["training_seed"]),
                "parent_blocks": blocks,
                "parent_raw_gradient_cosine": float(values["cosine"]),
            }
        )

    parent = pd.DataFrame(measurements)
    return parent


def summarize_precision_sensitivity(source: pd.DataFrame, parent: pd.DataFrame) -> pd.DataFrame:
    reduced = source[
        source["cell_id"].eq(CELL_ORDER[-1]) & source["checkpoint_label"].isin(("fraction_0p55", "fraction_0p90"))
    ][["checkpoint_label", "training_seed", "gradient_raw_cosine"]]
    paired = reduced.merge(parent, on=["checkpoint_label", "training_seed"], validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for state, group in paired.groupby("checkpoint_label", sort=True):
        delta = group["parent_raw_gradient_cosine"] - group["gradient_raw_cosine"]
        rows.append(
            {
                "quantity": state,
                "blocks_16_mean": float(group["gradient_raw_cosine"].mean()),
                "blocks_32_mean": float(group["parent_raw_gradient_cosine"].mean()),
                "blocks_32_minus_16_mean": float(delta.mean()),
            }
        )
    wide_16 = paired.pivot(index="training_seed", columns="checkpoint_label", values="gradient_raw_cosine")
    wide_32 = paired.pivot(index="training_seed", columns="checkpoint_label", values="parent_raw_gradient_cosine")
    decline_16 = wide_16["fraction_0p55"] - wide_16["fraction_0p90"]
    decline_32 = wide_32["fraction_0p55"] - wide_32["fraction_0p90"]
    rows.append(
        {
            "quantity": "fraction_0p55_minus_fraction_0p90",
            "blocks_16_mean": float(decline_16.mean()),
            "blocks_32_mean": float(decline_32.mean()),
            "blocks_32_minus_16_mean": float((decline_32 - decline_16).mean()),
        }
    )
    return pd.DataFrame(rows)


def trend_diagnostics(seed_effects: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Any]:
    means = summary.set_index("cell_id").loc[list(CELL_ORDER), "decline_mean"].to_numpy()
    ranks = pd.Series(means).rank().to_numpy()
    tpp_ranks = np.arange(1, len(CELL_ORDER) + 1, dtype=float)
    spearman = float(np.corrcoef(tpp_ranks, ranks)[0, 1])
    first = seed_effects[seed_effects["cell_id"].eq(CELL_ORDER[0])].set_index("training_seed")["decline"]
    last = seed_effects[seed_effects["cell_id"].eq(CELL_ORDER[-1])].set_index("training_seed")["decline"]
    paired = (last - first).to_numpy()
    low, high = bootstrap_interval(paired, key="trend:r3-minus-r0")
    approximate_resolution = 1.96 * float(paired.std(ddof=1)) / np.sqrt(len(paired))
    return {
        "mean_decline_spearman_vs_tpp": spearman,
        "mean_declines_monotone_increasing": bool(np.all(np.diff(means) >= 0)),
        "r3_minus_r0_decline_mean": float(paired.mean()),
        "r3_minus_r0_decline_bootstrap_ci95_low": low,
        "r3_minus_r0_decline_bootstrap_ci95_high": high,
        "r3_minus_r0_exact_sign_flip_p_two_sided": exact_sign_flip_p(paired, alternative="two_sided"),
        "r3_minus_r0_positive_seed_pairs": int(np.sum(paired > 0)),
        "r3_minus_r0_approximate_two_sided_95pct_resolution": approximate_resolution,
    }


def _style_figure(figure: go.Figure, *, height: int, margin_top: int = 64) -> None:
    figure.update_layout(
        template="plotly_white",
        autosize=True,
        height=height,
        margin={"l": 72, "r": 30, "t": margin_top, "b": 64},
        font={"family": "Avenir Next, sans-serif", "color": "#17324d", "size": 14},
        paper_bgcolor="#fffdf8",
        plot_bgcolor="#fffdf8",
        hoverlabel={"font": {"family": "Avenir Next, sans-serif"}},
    )


def source_alignment_figure(trajectory: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for cell, color in zip(CELL_ORDER, COLORS, strict=True):
        group = trajectory[trajectory["cell_id"].eq(cell)].sort_values("normalized_time")
        tpp = float(group["total_parameter_tpp"].iloc[0])
        figure.add_trace(
            go.Scatter(
                x=group["normalized_time"],
                y=group["gradient_cosine_mean"],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": group["gradient_cosine_ci95_high"] - group["gradient_cosine_mean"],
                    "arrayminus": group["gradient_cosine_mean"] - group["gradient_cosine_ci95_low"],
                    "thickness": 1.5,
                },
                mode="lines+markers",
                name=f"TPP {tpp:.2f}",
                legendgroup=cell,
                line={"color": color, "width": 3},
                marker={"size": 8},
                customdata=np.column_stack((group["checkpoint_label"], group["steps_from_lr_decay_onset"])),
                hovertemplate=(
                    "%{fullData.name}<br>state %{customdata[0]}<br>time %{x:.4f}T<br>"
                    "offset %{customdata[1]} updates<br>cosine %{y:.4f}<extra></extra>"
                ),
            )
        )
    figure.add_vline(x=0.8, line_width=2, line_color="#17324d")
    figure.add_annotation(
        x=0.8,
        y=1,
        yref="paper",
        text="LR decay begins",
        showarrow=False,
        textangle=-90,
        xshift=-13,
        yshift=-54,
        font={"size": 12},
    )
    figure.update_xaxes(title_text="Training progress", range=[0.53, 0.92], tickformat=".2f")
    figure.update_yaxes(title_text="Projected-trunk gradient cosine", range=[0.24, 0.61])
    figure.update_layout(legend={"orientation": "h", "x": 0, "y": 1.02, "xanchor": "left", "yanchor": "bottom"})
    _style_figure(figure, height=500, margin_top=78)
    return figure


def primary_decline_figure(primary: pd.DataFrame) -> go.Figure:
    ordered = primary.set_index("cell_id").loc[list(CELL_ORDER)].reset_index()
    figure = go.Figure(
        go.Scatter(
            x=ordered["total_parameter_tpp"],
            y=ordered["decline_mean"],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": ordered["decline_bootstrap_ci95_high"] - ordered["decline_mean"],
                "arrayminus": ordered["decline_mean"] - ordered["decline_bootstrap_ci95_low"],
                "thickness": 2,
            },
            mode="lines+markers",
            line={"color": "#17324d", "width": 2},
            marker={
                "color": list(COLORS),
                "size": 13,
                "line": {"color": "#17324d", "width": 1.5},
            },
            customdata=np.column_stack((ordered["positive_seed_pairs"], ordered["decline_holm_p_greater"])),
            hovertemplate=(
                "TPP %{x:.2f}<br>decline %{y:.4f}<br>declining seeds %{customdata[0]:.0f}/8"
                "<br>Holm p %{customdata[1]:.4f}<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0, line_width=1.5, line_color="#17324d")
    figure.update_xaxes(
        title_text="Total-parameter tokens per parameter",
        type="log",
        tickmode="array",
        tickvals=ordered["total_parameter_tpp"],
        ticktext=[f"{value:.2f}" for value in ordered["total_parameter_tpp"]],
    )
    figure.update_yaxes(title_text="Plateau minus 0.90T cosine", rangemode="tozero")
    _style_figure(figure, height=440, margin_top=24)
    return figure


def decay_local_figure(trajectory: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for cell, color in zip(CELL_ORDER, COLORS, strict=True):
        group = trajectory[trajectory["cell_id"].eq(cell)]
        tpp = float(group["total_parameter_tpp"].iloc[0])
        local = group[
            group["checkpoint_label"].isin(
                ("decay_minus_256", "decay_minus_64", "decay_onset", "decay_plus_64", "decay_plus_256")
            )
        ].sort_values("steps_from_lr_decay_onset")
        figure.add_trace(
            go.Scatter(
                x=local["steps_from_lr_decay_onset"],
                y=local["gradient_cosine_mean"],
                mode="lines+markers",
                line={"color": color, "width": 3},
                marker={"size": 8},
                name=f"TPP {tpp:.2f}",
                hovertemplate=f"TPP {tpp:.2f}<br>offset %{{x}} updates<br>cosine %{{y:.4f}}<extra></extra>",
            )
        )
    figure.add_vline(x=0, line_width=2, line_color="#17324d")
    figure.update_xaxes(title_text="Updates from LR-decay onset")
    figure.update_yaxes(title_text="Projected-trunk gradient cosine", range=[0.24, 0.61])
    figure.update_layout(showlegend=False)
    _style_figure(figure, height=440, margin_top=24)
    return figure


def target_utility_figure(primary: pd.DataFrame, utility: pd.DataFrame) -> go.Figure:
    targets = list(TARGET_LABELS)
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=tuple(TARGET_LABELS[target] for target in targets),
        horizontal_spacing=0.12,
        vertical_spacing=0.20,
    )
    tpp_by_cell = primary.set_index("cell_id")["total_parameter_tpp"]
    for index, target in enumerate(targets):
        row = index // 2 + 1
        col = index % 2 + 1
        for utility_source in SOURCE_LABELS:
            group = utility[utility["target"].eq(target) & utility["source"].eq(utility_source)]
            ordered = group.set_index("cell_id").loc[list(CELL_ORDER)].reset_index()
            x = [float(tpp_by_cell.loc[cell]) for cell in CELL_ORDER]
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=ordered["decline_mean"],
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": ordered["bootstrap_ci95_high"] - ordered["decline_mean"],
                        "arrayminus": ordered["decline_mean"] - ordered["bootstrap_ci95_low"],
                        "thickness": 1.5,
                    },
                    mode="lines+markers",
                    name=SOURCE_LABELS[utility_source],
                    legendgroup=utility_source,
                    showlegend=index == 0,
                    line={"width": 2.5, "color": SOURCE_COLORS[utility_source]},
                    marker={"size": 8},
                    hovertemplate=(
                        f"{TARGET_LABELS[target]} · {SOURCE_LABELS[utility_source]}"
                        "<br>TPP %{x:.2f}<br>utility decline %{y:.4f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        figure.add_hline(y=0, line_width=1, line_color="#17324d", row=row, col=col)
        figure.update_xaxes(
            type="log",
            tickmode="array",
            tickvals=x,
            ticktext=[f"{value:.1f}" for value in x],
            row=row,
            col=col,
        )
        figure.update_yaxes(rangemode="tozero", row=row, col=col)
    figure.update_xaxes(title_text="Total TPP", row=2, col=1)
    figure.update_xaxes(title_text="Total TPP", row=2, col=2)
    figure.update_yaxes(title_text="Utility-cosine decline", row=1, col=1)
    figure.update_yaxes(title_text="Utility-cosine decline", row=2, col=1)
    figure.update_layout(legend={"orientation": "h", "x": 0, "y": 1.05, "xanchor": "left", "yanchor": "bottom"})
    _style_figure(figure, height=680, margin_top=88)
    return figure


def build_figures(
    trajectory: pd.DataFrame,
    primary: pd.DataFrame,
    utility: pd.DataFrame,
) -> dict[str, go.Figure]:
    return {
        "source-alignment": source_alignment_figure(trajectory),
        "primary-decline": primary_decline_figure(primary),
        "decay-local": decay_local_figure(trajectory),
        "target-utility": target_utility_figure(primary, utility),
    }


def build_dashboard_html(
    trajectory: pd.DataFrame,
    primary: pd.DataFrame,
    utility: pd.DataFrame,
    precision: pd.DataFrame,
    diagnostics: Mapping[str, Any],
) -> str:
    figures = build_figures(trajectory, primary, utility)
    fragments: dict[str, str] = {}
    for index, (name, figure) in enumerate(figures.items()):
        fragments[name] = pio.to_html(
            figure,
            include_plotlyjs=index == 0,
            full_html=False,
            config=PLOT_CONFIG,
            div_id=f"fixed-n-tpp-{name}",
        )
    decline = precision[precision["quantity"].eq("fraction_0p55_minus_fraction_0p90")].iloc[0]
    r3_minus_r0 = diagnostics["r3_minus_r0_decline_mean"]
    r3_low = diagnostics["r3_minus_r0_decline_bootstrap_ci95_low"]
    r3_high = diagnostics["r3_minus_r0_decline_bootstrap_ci95_high"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Does TPP control gradient divergence?</title>
<style>
:root{{
  --ink:#183149;--muted:#536674;--paper:#fbf8ef;--card:#fffdf7;
  --teal:#1b7f79;--orange:#d65a31;--line:#d8cfbd;
}}
*{{box-sizing:border-box}}
body{{
  margin:0;background:var(--paper);color:var(--ink);
  font-family:"Avenir Next",Avenir,sans-serif;
}}
main{{max-width:1500px;margin:0 auto;padding:52px 32px 88px}}
a{{color:var(--teal);font-weight:700;text-decoration:none}}
.eyebrow{{
  margin:18px 0 0;color:var(--orange);font-size:13px;font-weight:800;
  letter-spacing:.15em;text-transform:uppercase;
}}
h1{{
  font-family:Georgia,serif;font-size:clamp(42px,6vw,72px);
  line-height:1;margin:12px 0 20px;max-width:1100px;
}}
.deck{{font-size:19px;line-height:1.55;max-width:1050px;color:var(--muted)}}
.facts{{
  display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;
  margin:32px 0;background:var(--line);border:1px solid var(--line);
}}
.fact{{background:var(--card);padding:18px 20px}}
.fact b{{display:block;font-family:Georgia,serif;font-size:25px;margin-bottom:4px}}
.fact span{{font-size:13px;line-height:1.4;color:var(--muted)}}
.verdict{{
  margin:30px 0 34px;padding:23px 26px;background:#15394a;color:#fffdf7;
  border-left:7px solid var(--orange);font-size:17px;line-height:1.55;
}}
.plots{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}}
.panel{{min-width:0;background:var(--card);border:1px solid var(--line)}}
.panel.wide{{grid-column:1/-1}}
.panel-head{{padding:24px 26px 0}}
.panel h2{{font-family:Georgia,serif;font-size:28px;margin:0 0 7px}}
.panel p{{margin:0;color:var(--muted);line-height:1.5}}
.plot{{width:100%;min-width:0;overflow:hidden}}
.plot .js-plotly-plot,.plot .plot-container,.plot .svg-container{{width:100%!important}}
.notes{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-top:28px}}
.note{{padding:20px 22px;background:var(--card);border-top:4px solid var(--teal)}}
.note h3{{font-family:Georgia,serif;font-size:21px;margin:0 0 7px}}
.note p{{font-size:14px;line-height:1.5;color:var(--ink)}}
footer{{margin-top:42px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);line-height:1.5}}
@media(max-width:1050px){{
  .plots{{grid-template-columns:1fr}}.panel.wide{{grid-column:auto}}
  .facts{{grid-template-columns:repeat(2,minmax(0,1fr))}}.notes{{grid-template-columns:1fr}}
}}
@media(max-width:640px){{
  main{{padding:34px 14px 60px}}.facts{{grid-template-columns:1fr}}
  .panel-head{{padding:20px 18px 0}}
}}
</style></head><body><main><!-- STUDY_BACK_LINK -->
<p class="eyebrow">StarCoder WSD80 mechanistic study · fixed-N scaling extension</p>
<h1>Does TPP control when source gradients diverge?</h1>
<p class="deck">We hold model size fixed and increase the training horizon across four rungs. At eight restored
states per rung, the same frozen StarCoder and Nemotron panels estimate source-gradient alignment. The question
is whether higher tokens per parameter moves or amplifies the late decline.</p>
<section class="facts">
  <div class="fact"><b>256 / 256</b><span>frozen probe rows completed</span></div>
  <div class="fact"><b>4.77&ndash;35.27</b><span>total-parameter TPP ladder</span></div>
  <div class="fact"><b>32 / 32</b><span>seed-rung pairs with a lower 0.90T cosine</span></div>
  <div class="fact"><b>{decline['blocks_16_mean']:.4f} &rarr; {decline['blocks_32_mean']:.4f}</b>
    <span>high-TPP decline at 16 versus 32 blocks</span></div>
</section>
<section class="verdict"><strong>Answer:</strong> measured finite-batch cosine is lower by 0.90T at every rung,
but the decline is not monotone in TPP. The highest-minus-lowest-rung contrast is {r3_minus_r0:+.4f}, 95% CI
[{r3_low:+.4f}, {r3_high:+.4f}]. The common grid has only one point after 0.80T, so it identifies neither an
exact onset nor a TPP-controlled change point.</section>
<section class="plots">
<article class="panel wide"><div class="panel-head"><h2>Source alignment over normalized training time</h2>
<p>Each line is one token-horizon rung. The vertical bar is the shared 0.80T learning-rate-decay onset;
intervals are paired seed-bootstrap 95% intervals.</p></div>
<div class="plot">{fragments['source-alignment']}</div></article>
<article class="panel"><div class="panel-head"><h2>Preregistered late decline</h2>
<p>Mean 0.55T/0.70T cosine minus 0.90T cosine. Positive means less measured alignment late.</p></div>
<div class="plot">{fragments['primary-decline']}</div></article>
<article class="panel"><div class="panel-head"><h2>Absolute updates around decay onset</h2>
<p>Colors match the first panel. Absolute update offsets do not reveal a common fixed-update onset.</p></div>
<div class="plot">{fragments['decay-local']}</div></article>
<article class="panel wide"><div class="panel-head"><h2>Target-source utility decline</h2>
<p>For each evaluation target, compare the decline in alignment of Nemotron and heldout-StarCoder optimizer
updates. These are descriptive secondary measurements.</p></div>
<div class="plot">{fragments['target-utility']}</div></article>
</section>
<section class="notes">
  <article class="note"><h3>Measurement caveat</h3><p>Mean-gradient norms fall 39&ndash;53% from
  0.80T to 0.90T. Finite-batch cosine attenuation therefore remains viable at lower rungs.</p></article>
  <article class="note"><h3>Precision control</h3><p>At TPP 35.27, doubling each source estimate from
  16 to 32 blocks changes the decline by only {decline['blocks_32_minus_16_mean']:+.4f}. This supports
  robustness at that rung, not every rung.</p></article>
  <article class="note"><h3>Surrogate implication</h3><p>Do not use TPP alone as a temporal change-point
  feature. A shifted or removed LR decay and repeated independent panels are needed to separate schedule timing
  from attenuation.</p></article>
</section>
<footer>All panels use the frozen 256-row release and read no endpoint metrics. The ladder changes D, TPP,
update count, and cumulative exposure together; total and non-embedding TPP differ only by a fixed factor.</footer>
</main></body></html>"""


def write_dashboard(
    trajectory: pd.DataFrame,
    primary: pd.DataFrame,
    utility: pd.DataFrame,
    precision: pd.DataFrame,
    diagnostics: Mapping[str, Any],
) -> None:
    document = build_dashboard_html(trajectory, primary, utility, precision, diagnostics)
    (OUTPUT_DIR / "fixed_n_tpp_gradient_onset.html").write_text(document)


def write_report(
    primary: pd.DataFrame,
    hinges: pd.DataFrame,
    optimizer: pd.DataFrame,
    utility: pd.DataFrame,
    decay_local: pd.DataFrame,
    onset_reference: pd.DataFrame,
    target_gradient: pd.DataFrame,
    gradient_norms: pd.DataFrame,
    geometry: Mapping[str, float],
    precision: pd.DataFrame,
    diagnostics: Mapping[str, Any],
) -> None:
    table_rows = []
    for _, row in primary.iterrows():
        onset_drop = onset_reference[
            onset_reference["cell_id"].eq(row["cell_id"]) & onset_reference["effect"].eq("onset_to_0p90_decline")
        ].iloc[0]
        table_rows.append(
            "| {tpp:.2f} | {non_embedding_tpp:.2f} | {decline:.4f} | [{low:.4f}, {high:.4f}] | "
            "{onset_drop:.4f} | {positive}/8 | {p:.4f} |".format(
                tpp=row["total_parameter_tpp"],
                non_embedding_tpp=row["total_parameter_tpp"] * 210_052_480 / 45_884_800,
                decline=row["decline_mean"],
                low=row["decline_bootstrap_ci95_low"],
                high=row["decline_bootstrap_ci95_high"],
                onset_drop=onset_drop["mean"],
                positive=int(row["positive_seed_pairs"]),
                p=row["decline_holm_p_greater"],
            )
        )
    utility_range = (float(utility["decline_mean"].min()), float(utility["decline_mean"].max()))
    target_gradient_range = (
        float(target_gradient["decline_mean"].min()),
        float(target_gradient["decline_mean"].max()),
    )
    optimizer_range = (float(optimizer["decline_mean"].min()), float(optimizer["decline_mean"].max()))
    plus_256 = decay_local[decay_local["checkpoint_label"].eq("decay_plus_256")]
    plus_256_values = plus_256.set_index("cell_id").loc[list(CELL_ORDER), "onset_minus_later_mean"].tolist()
    spearman = diagnostics["mean_decline_spearman_vs_tpp"]
    r3_minus_r0 = diagnostics["r3_minus_r0_decline_mean"]
    r3_minus_r0_low = diagnostics["r3_minus_r0_decline_bootstrap_ci95_low"]
    r3_minus_r0_high = diagnostics["r3_minus_r0_decline_bootstrap_ci95_high"]
    r3_minus_r0_p = diagnostics["r3_minus_r0_exact_sign_flip_p_two_sided"]
    resolution = diagnostics["r3_minus_r0_approximate_two_sided_95pct_resolution"]
    onset_declines = (
        onset_reference[onset_reference["effect"].eq("onset_to_0p90_decline")]
        .set_index("cell_id")
        .loc[list(CELL_ORDER), "mean"]
        .to_numpy()
    )
    onset_spearman = float(np.corrcoef(np.arange(1, 5), pd.Series(onset_declines).rank())[0, 1])
    r0_pre_onset = onset_reference[
        onset_reference["cell_id"].eq(CELL_ORDER[0]) & onset_reference["effect"].eq("pre_onset_rise")
    ].iloc[0]
    norm_wide = gradient_norms.pivot(
        index="cell_id",
        columns="checkpoint_label",
        values=["starcoder_raw_gradient_norm_mean", "nemotron_raw_gradient_norm_mean"],
    )
    starcoder_norm_drop = 1.0 - (
        norm_wide["starcoder_raw_gradient_norm_mean"]["fraction_0p90"]
        / norm_wide["starcoder_raw_gradient_norm_mean"]["decay_onset"]
    )
    nemotron_norm_drop = 1.0 - (
        norm_wide["nemotron_raw_gradient_norm_mean"]["fraction_0p90"]
        / norm_wide["nemotron_raw_gradient_norm_mean"]["decay_onset"]
    )
    precision_decline = precision[precision["quantity"].eq("fraction_0p55_minus_fraction_0p90")].iloc[0]
    all_positive = int(primary["positive_seed_pairs"].sum())
    primary_p = sorted(set(primary["decline_exact_sign_flip_p_greater"]))
    primary_holm = sorted(set(primary["decline_holm_p_greater"]))
    if primary_p != [0.00390625] or primary_holm != [0.015625]:
        raise RuntimeError("Unexpected primary p-value pattern")
    table_header = (
        "| Total TPP | Non-embedding TPP | Preregistered decline | Bootstrap 95% CI | "
        "0.80T-to-0.90T drop | Seeds declining | Holm p |"
    )
    report = f"""# Fixed-N TPP gradient-onset result

## Verdict

The preregistered 16-block estimator shows lower StarCoder-Nemotron gradient cosine at 0.90T in all
four fixed-N rungs and all {all_positive}/32 paired seeds. The strongest available precision control
shows that the high-TPP result is robust to doubling each source estimate from 16 to 32 blocks: the
0.55T-to-0.90T drop changes only from {precision_decline['blocks_16_mean']:.4f} to
{precision_decline['blocks_32_mean']:.4f}. Raw and projected trunk cosines also agree to within
{geometry['source_source_max_abs_projected_minus_raw_cosine']:.2g} everywhere, ruling out the Muon
tangent projection as the explanation.

The panel nevertheless does **not** establish that the underlying population-gradient angle declines
at all four rungs. Mean gradient norms fall substantially over the same interval, so finite-batch
cosine attenuation remains a viable explanation outside the one rung with the doubled-precision
control. Nor does the panel identify a TPP-controlled onset: the coarse common grid has only one point
after 0.80T, and the cross-rung contrast is too imprecise to support an equivalence claim.

## Primary result

The preregistered effect is the within-seed mean cosine at 0.55T and 0.70T minus the cosine at 0.90T.
Positive values mean the finite-batch projected-trunk cosine estimator became smaller.

{table_header}
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

The one-sided exact sign-flip p-value is {primary_p[0]:.4f} in every rung and
{primary_holm[0]:.4f} after Holm correction. These tests establish a change in the measured cosine,
not that estimator attenuation is absent.

Across rung means, Spearman correlation between TPP and decline is {spearman:.2f}. The
highest-minus-lowest-TPP paired difference is {r3_minus_r0:+.4f}, 95% bootstrap CI
[{r3_minus_r0_low:+.4f}, {r3_minus_r0_high:+.4f}], two-sided exact p={r3_minus_r0_p:.3f}.
The approximate two-sided 95% resolution is +/-{resolution:.4f}, nearly the size of the observed
declines. Higher TPP therefore does not produce a resolvably larger decline, but this is not evidence
of no TPP effect.

The preregistered reference matters. Relative to the actual 0.80T decay-onset checkpoint, the four
drops are {onset_declines[0]:.4f}, {onset_declines[1]:.4f}, {onset_declines[2]:.4f}, and
{onset_declines[3]:.4f}: strictly decreasing with TPP (Spearman {onset_spearman:.1f}). The shortest
rung rises by {r0_pre_onset['mean']:.4f} before onset, 95% CI
[{r0_pre_onset['bootstrap_ci95_low']:.4f}, {r0_pre_onset['bootstrap_ci95_high']:.4f}], which is why
the plateau-referenced ordering differs. We therefore retain the preregistered statistic but do not
interpret its cross-rung ordering mechanistically.

The hinge onset is structurally unidentified: with only one common point after 0.80T, every hinge
location in [0.80T, 0.90T) has exactly the same fitted error. Seed uncertainty also prevents treating
0.80T as a firm lower bound outside the shortest rung. The result is only "a change is visible by
0.90T." It is compatible with normalized progress or fraction of LR decay elapsed; because LR decay
always starts at 0.80T, those explanations are collinear here.

## Measurement checks

- StarCoder mean-gradient norms fall by {starcoder_norm_drop.min():.0%} to
  {starcoder_norm_drop.max():.0%} from 0.80T to 0.90T; Nemotron norms fall by
  {nemotron_norm_drop.min():.0%} to {nemotron_norm_drop.max():.0%}. A cosine of noisy mean gradients
  is attenuated as signal norms approach the batch-noise floor. The frozen summaries do not retain
  enough per-block gradient geometry for a split-half correction at the lower rungs.
- At TPP 35.27, the existing 32-block parent measurements reproduce the late decline almost exactly:
  {precision_decline['blocks_16_mean']:.4f} at 16 blocks versus
  {precision_decline['blocks_32_mean']:.4f} at 32. This shows robustness to a twofold precision
  increase, but does not eliminate residual attenuation or calibrate the lower rungs.
- The unprojected and projected source-source trajectories are numerically indistinguishable; the
  maximum absolute cosine difference is
  {geometry['source_source_max_abs_projected_minus_raw_cosine']:.2g}. Target-source raw-gradient
  comparisons are now also materialized rather than inferred only from optimizer-transformed utility.

## Secondary results

- Target-source utility cosines decline by {utility_range[0]:.3f} to {utility_range[1]:.3f} across the
  target/source/rung cells. Raw target-source gradient cosines change by
  {target_gradient_range[0]:+.3f} to {target_gradient_range[1]:+.3f}. Both are descriptive because
  target block counts vary and the same attenuation issue applies.
- Source-source optimizer-update cosine does not mirror the raw-gradient decline: its
  plateau-minus-0.90T values range from {optimizer_range[0]:+.3f} to {optimizer_range[1]:+.3f}.
  The measured finite difference `Delta(g)-Delta(0)` becomes more mutually aligned for both sources,
  while both source updates become less aligned with evaluation targets. This is a property of the
  nonlinear optimizer map, not a causal attribution to momentum.
- The late raw-gradient decline occurs in the full source pool without finite-support replay and is
  measured already at total TPP 4.77. It therefore cannot be attributed solely to simulated
  repetition, although attenuation is not ruled out there.
- At a fixed 256 updates after LR-decay onset, the rung-wise declines are
  {plus_256_values[0]:+.3f}, {plus_256_values[1]:+.3f}, {plus_256_values[2]:+.3f}, and
  {plus_256_values[3]:+.3f}. Only the shortest rung has a 95% interval excluding zero. This is
  compatible with normalized or decay-fraction progress rather than a fixed absolute-update clock;
  it cannot distinguish normalized training time from the LR schedule.

## Implication

For surrogate modeling, TPP should not be used as a standalone change-point controller based on this
panel. Normalized progress and optimizer state remain plausible temporal features, but the experiment
does not identify which causes the late change and does not show that source-gradient conflict creates
a two-phase endpoint advantage. A shifted- or no-decay intervention plus repeated gradient panels at
fixed checkpoints would separate schedule causality from finite-batch attenuation.

## Scope and provenance

- 256/256 rows completed: four fixed-N cells, eight paired seeds, and eight temporal states.
- Whole-panel audit `{AUDIT_SHA256}` passed with zero missing or unexpected rows, one runtime
  environment, and `endpoint_metrics_read=false`.
- Lower-rung outcomes were uninspected when frozen, but the r3 trajectory motivated the extension.
  This is not untouched global confirmation.
- D, TPP, absolute update count, and exposure co-vary. Total- and non-embedding-parameter TPP differ
  only by a fixed factor here.
- Independent Opus 5 review identified the attenuation, reference-choice, power, and optimizer-map
  caveats incorporated above. The precision sensitivity uses only previously completed frozen rows.
- Interactive plot: `fixed_n_tpp_gradient_onset.html`.
"""
    (OUTPUT_DIR / "report.md").write_text(report)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    release = json.loads((INPUT_DIR / "release.json").read_text())
    audit = json.loads((INPUT_DIR / "final_whole_panel_audit.json").read_text())
    if release.get("release_sha256") != RELEASE_SHA256:
        raise RuntimeError("Frozen release identity drifted")
    if audit.get("audit_sha256") != AUDIT_SHA256 or audit.get("endpoint_metrics_read") is not False:
        raise RuntimeError("Frozen whole-panel audit identity drifted")
    documents = load_documents()
    source, utility_raw, target_gradient_raw = flatten_documents(documents, release)
    seed_effects, primary = primary_effects(source)
    raw_source = source.copy()
    raw_source["gradient_cosine"] = raw_source["gradient_raw_cosine"]
    _, raw_primary = primary_effects(raw_source)
    trajectory = summarize_trajectory(source)
    hinges = hinge_summaries(source)
    optimizer, utility = secondary_summaries(source, utility_raw)
    decay_local = decay_local_summaries(source)
    onset_reference = onset_reference_summaries(source)
    target_gradient = target_gradient_summaries(target_gradient_raw)
    gradient_norms = gradient_norm_summaries(source)
    geometry = geometry_sensitivity(source, target_gradient_raw)
    parent_precision = parent_precision_sensitivity(gcsfs.GCSFileSystem(), read_rows())
    precision = summarize_precision_sensitivity(source, parent_precision)
    diagnostics = trend_diagnostics(seed_effects, primary)

    seed_effects.to_csv(OUTPUT_DIR / "primary_seed_effects.csv", index=False)
    primary.to_csv(OUTPUT_DIR / "primary_summary.csv", index=False)
    raw_primary.to_csv(OUTPUT_DIR / "raw_geometry_primary_summary.csv", index=False)
    trajectory.to_csv(OUTPUT_DIR / "source_alignment_trajectory.csv", index=False)
    hinges.to_csv(OUTPUT_DIR / "hinge_summary.csv", index=False)
    optimizer.to_csv(OUTPUT_DIR / "optimizer_update_summary.csv", index=False)
    utility.to_csv(OUTPUT_DIR / "target_source_utility_summary.csv", index=False)
    target_gradient.to_csv(OUTPUT_DIR / "target_source_gradient_summary.csv", index=False)
    decay_local.to_csv(OUTPUT_DIR / "decay_local_summary.csv", index=False)
    onset_reference.to_csv(OUTPUT_DIR / "onset_reference_summary.csv", index=False)
    gradient_norms.to_csv(OUTPUT_DIR / "gradient_norm_summary.csv", index=False)
    precision.to_csv(OUTPUT_DIR / "precision_sensitivity_summary.csv", index=False)
    source.to_csv(OUTPUT_DIR / "source_measurements.csv", index=False)
    utility_raw.to_csv(OUTPUT_DIR / "target_source_utility_measurements.csv", index=False)
    target_gradient_raw.to_csv(OUTPUT_DIR / "target_source_gradient_measurements.csv", index=False)

    summary = {
        "release_sha256": RELEASE_SHA256,
        "audit_sha256": AUDIT_SHA256,
        "row_count": len(source),
        "utility_row_count": len(utility_raw),
        "primary": primary.to_dict(orient="records"),
        "raw_geometry_primary": raw_primary.to_dict(orient="records"),
        "hinges": hinges.to_dict(orient="records"),
        "decay_local": decay_local.to_dict(orient="records"),
        "onset_reference": onset_reference.to_dict(orient="records"),
        "gradient_norms": gradient_norms.to_dict(orient="records"),
        "geometry_sensitivity": geometry,
        "precision_sensitivity": precision.to_dict(orient="records"),
        "trend_diagnostics": diagnostics,
        "interpretation": (
            "measured_late_cosine_decline_all_rungs_high_tpp_precision_robust_"
            "lower_rung_attenuation_and_tpp_onset_unresolved"
        ),
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(
        primary,
        hinges,
        optimizer,
        utility,
        decay_local,
        onset_reference,
        target_gradient,
        gradient_norms,
        geometry,
        precision,
        diagnostics,
    )
    write_dashboard(trajectory, primary, utility, precision, diagnostics)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
