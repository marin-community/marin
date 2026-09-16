# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Collect and analyze the 60M fixed-aggregate phase-order panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import plotly.express as px
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_MANIFEST = REFERENCE_OUTPUTS / "60m_fixed_aggregate_phase_order_panel_20260725/candidate_manifest.csv"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "60m_fixed_aggregate_phase_order_results_20260726"

TRAIN_PROJECT = "marin-community/marin"
TRAIN_TAG = "pinlin_calvin_xu/data_mixture/dm60po_20260725"
TABLE9_PROJECT = "marin-community/marin-eval"
TABLE9_GROUP = "olmo_base_eval_table9_60m_fixed_aggregate_phase_order_20260725"
UNCHEATABLE_KEY = "eval/uncheatable_eval/bpb"
TABLE9_KEY = "olmo_base_easy/table9_macro_bpb"
EXPECTED_MANIFEST_SHA256 = "8dee2c13f66527c6cc1d579630256540b42111cb7a73be5c01083d0b698f93f7"
EXPECTED_CHECKPOINT_STEP = 4576
EXPECTED_RUNS = 140
EXPECTED_NONFINISHED_TRAINING_INDICES = frozenset({56, 58, 61, 77})
RUN_INDEX_PATTERN = re.compile(r"(?<![A-Za-z0-9])p(\d{3})(?=[_-]|$)")
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

TARGETS = {
    "uncheatable": "uncheatable_bpb",
    "table9": "table9_macro_bpb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=180)
    return parser.parse_args()


def run_index(name: str) -> int | None:
    match = RUN_INDEX_PATTERN.search(name)
    if match is None:
        return None
    return int(match.group(1))


def validate_manifest(manifest: pd.DataFrame, manifest_path: Path, expected_sha256: str) -> None:
    actual_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Manifest SHA-256 changed: {actual_sha256} != {expected_sha256}")
    if len(manifest) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} manifest rows, found {len(manifest)}")
    if manifest["candidate_id"].duplicated().any():
        duplicates = manifest.loc[manifest["candidate_id"].duplicated(keep=False), "candidate_id"].tolist()
        raise ValueError(f"Manifest candidate IDs are not unique: {duplicates}")
    expected_run_ids = np.arange(7_250_000, 7_250_000 + EXPECTED_RUNS)
    if not np.array_equal(manifest["run_id"].to_numpy(dtype=int), expected_run_ids):
        raise ValueError("Manifest row order no longer matches the submitted run IDs")


def training_metadata_by_index(
    runs: list[Any],
) -> dict[int, dict[str, object]]:
    selected: dict[int, dict[str, object]] = {}
    for run in runs:
        index = run_index(run.name)
        if index is None:
            raise ValueError(f"Could not parse a panel index from training run {run.name!r}")
        if index in selected:
            raise ValueError(f"Multiple training runs resolved to panel index {index}")
        config = dict(run.config)
        hf_save_path = str(config.get("hf_save_path", ""))
        if not hf_save_path.endswith("/hf"):
            raise ValueError(f"Training run {run.name!r} has an invalid hf_save_path: {hf_save_path!r}")
        checkpoint_root = hf_save_path.removesuffix("/hf")
        checkpoint_index = run_index(checkpoint_root)
        if checkpoint_index != index:
            raise ValueError(
                f"Training run {run.name!r} resolved to index {index}, but its checkpoint resolved to {checkpoint_index}"
            )
        selected[index] = {
            "wandb_run_id": run.id,
            "wandb_run_name": run.name,
            "wandb_state": run.state,
            "wandb_created_at": run.created_at,
            "checkpoint_root": checkpoint_root,
            "data_seed": int(config["data_seed"]),
            "trainer_seed": int(config["trainer"]["seed"]),
        }
    expected_indices = set(range(EXPECTED_RUNS))
    if set(selected) != expected_indices:
        raise ValueError(f"Training run indices differ from the manifest: {sorted(expected_indices - set(selected))}")
    return selected


def table9_metric_by_index(
    runs: list[Any],
    manifest: pd.DataFrame,
) -> dict[int, dict[str, object]]:
    candidate_to_index = {
        str(candidate_id): index for index, candidate_id in enumerate(manifest["candidate_id"].astype(str))
    }
    candidates: dict[int, list[dict[str, object]]] = {}
    for run in runs:
        summary = dict(run.summary)
        metric = summary.get(TABLE9_KEY)
        if run.state != "finished" or metric is None:
            continue
        config = dict(run.config)
        provenance = dict(config.get("provenance") or {})
        candidate_id = provenance.get("candidate_id")
        if candidate_id is not None:
            candidate_id = str(candidate_id)
            if candidate_id not in candidate_to_index:
                raise ValueError(f"Unknown Table-9 candidate_id {candidate_id!r} on run {run.name!r}")
            index = candidate_to_index[candidate_id]
            provenance_mode = "candidate_id"
        else:
            source_run_name = str(provenance.get("source_run_name") or run.name)
            parsed_index = run_index(source_run_name)
            if parsed_index is None:
                raise ValueError(f"Table-9 run {run.name!r} has neither candidate_id nor a source panel index")
            index = parsed_index
            candidate_id = str(manifest.iloc[index]["candidate_id"])
            provenance_mode = "source_run_name_and_checkpoint_index"

        checkpoint_path = str(config.get("checkpoint_path", ""))
        checkpoint_index = run_index(checkpoint_path)
        if checkpoint_index != index:
            raise ValueError(
                f"Table-9 run {run.name!r} resolved to index {index}, but its checkpoint resolved to {checkpoint_index}"
            )
        expected_suffix = f"/hf/step-{EXPECTED_CHECKPOINT_STEP}"
        if not checkpoint_path.endswith(expected_suffix):
            raise ValueError(f"Table-9 run {run.name!r} did not evaluate the final checkpoint: {checkpoint_path}")
        candidates.setdefault(index, []).append(
            {
                "value": float(metric),
                "candidate_id": candidate_id,
                "provenance_mode": provenance_mode,
                "checkpoint_path": checkpoint_path,
                "checkpoint_root": checkpoint_path.removesuffix(expected_suffix),
                "wandb_run_id": run.id,
                "wandb_run_name": run.name,
                "wandb_state": run.state,
                "wandb_created_at": run.created_at,
            }
        )

    selected: dict[int, dict[str, object]] = {}
    for index, records in candidates.items():
        values = {float(record["value"]) for record in records}
        checkpoint_roots = {str(record["checkpoint_root"]) for record in records}
        if len(values) != 1 or len(checkpoint_roots) != 1:
            raise ValueError(f"Completed Table-9 retries disagree for panel index {index}: {records}")
        selected[index] = max(records, key=lambda record: str(record["wandb_created_at"]))

    expected_indices = set(range(len(manifest)))
    if set(selected) != expected_indices:
        raise ValueError(
            f"Completed Table-9 indices differ from the manifest: {sorted(expected_indices - set(selected))}"
        )
    return selected


def read_final_uncheatable(checkpoint_root: str) -> dict[str, object]:
    metrics_path = f"{checkpoint_root.rstrip('/')}/checkpoints/eval_metrics.jsonl"
    final_step = -1
    final_value: float | None = None
    values_by_step: dict[int, float] = {}
    with fsspec.open(metrics_path, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            if UNCHEATABLE_KEY not in record:
                continue
            step = int(record.get("step", -1))
            value = float(record[UNCHEATABLE_KEY])
            previous = values_by_step.get(step)
            if previous is not None and previous != value:
                raise ValueError(f"{metrics_path} has conflicting {UNCHEATABLE_KEY!r} values at step {step}")
            values_by_step[step] = value
            if step >= final_step:
                final_step = step
                final_value = value
    if final_value is None:
        raise ValueError(f"No {UNCHEATABLE_KEY!r} value in {metrics_path}")
    if final_step != EXPECTED_CHECKPOINT_STEP:
        raise ValueError(f"{metrics_path} ends at step {final_step}, expected {EXPECTED_CHECKPOINT_STEP}")
    return {"value": final_value, "step": final_step, "metrics_path": metrics_path}


def collect_observed_results(manifest: pd.DataFrame, *, timeout: int) -> pd.DataFrame:
    api = wandb.Api(timeout=timeout)
    training_runs = list(api.runs(TRAIN_PROJECT, filters={"tags": {"$in": [TRAIN_TAG]}}, per_page=EXPECTED_RUNS + 20))
    table9_runs = list(api.runs(TABLE9_PROJECT, filters={"group": TABLE9_GROUP}, per_page=EXPECTED_RUNS + 100))
    training = training_metadata_by_index(training_runs)
    table9 = table9_metric_by_index(table9_runs, manifest)
    nonfinished_indices = {index for index, row in training.items() if row["wandb_state"] != "finished"}
    if nonfinished_indices != EXPECTED_NONFINISHED_TRAINING_INDICES:
        raise ValueError(
            "Unexpected set of non-finished training runs: "
            f"{sorted(nonfinished_indices)} != {sorted(EXPECTED_NONFINISHED_TRAINING_INDICES)}"
        )

    for index in range(len(manifest)):
        manifest_row = manifest.iloc[index]
        training_row = training[index]
        table9_row = table9[index]
        if int(training_row["data_seed"]) != int(manifest_row["data_seed"]):
            raise ValueError(f"Training data seed mismatch at panel index {index}")
        if int(training_row["trainer_seed"]) != int(manifest_row["trainer_seed"]):
            raise ValueError(f"Training trainer seed mismatch at panel index {index}")
        if training_row["checkpoint_root"] != table9_row["checkpoint_root"]:
            raise ValueError(f"Training and Table-9 checkpoint roots disagree at panel index {index}")
        if table9_row["candidate_id"] != manifest_row["candidate_id"]:
            raise ValueError(f"Table-9 candidate provenance disagrees at panel index {index}")

    indices = list(range(len(manifest)))
    checkpoint_roots = [str(training[index]["checkpoint_root"]) for index in indices]
    with ThreadPoolExecutor(max_workers=16) as executor:
        final_metrics = dict(zip(indices, executor.map(read_final_uncheatable, checkpoint_roots), strict=True))

    observed = manifest.copy()
    observed.insert(0, "panel_index", np.arange(len(observed), dtype=int))
    observed["uncheatable_bpb"] = observed["panel_index"].map(
        {index: row["value"] for index, row in final_metrics.items()}
    )
    observed["uncheatable_metric_step"] = observed["panel_index"].map(
        {index: row["step"] for index, row in final_metrics.items()}
    )
    observed["uncheatable_metric_source"] = "gcs_final_eval_metrics_jsonl"
    observed["uncheatable_metrics_path"] = observed["panel_index"].map(
        {index: row["metrics_path"] for index, row in final_metrics.items()}
    )
    observed["table9_macro_bpb"] = observed["panel_index"].map({index: row["value"] for index, row in table9.items()})
    observed["table9_metric_source"] = "finished_native_table9_wandb"
    observed["table9_provenance_mode"] = observed["panel_index"].map(
        {index: row["provenance_mode"] for index, row in table9.items()}
    )
    observed["checkpoint_root"] = observed["panel_index"].map(
        {index: row["checkpoint_root"] for index, row in training.items()}
    )
    observed["training_wandb_run_id"] = observed["panel_index"].map(
        {index: row["wandb_run_id"] for index, row in training.items()}
    )
    observed["training_wandb_run_name"] = observed["panel_index"].map(
        {index: row["wandb_run_name"] for index, row in training.items()}
    )
    observed["training_wandb_state"] = observed["panel_index"].map(
        {index: row["wandb_state"] for index, row in training.items()}
    )
    observed["training_wandb_created_at"] = observed["panel_index"].map(
        {index: row["wandb_created_at"] for index, row in training.items()}
    )
    observed["table9_wandb_run_id"] = observed["panel_index"].map(
        {index: row["wandb_run_id"] for index, row in table9.items()}
    )
    observed["table9_wandb_run_name"] = observed["panel_index"].map(
        {index: row["wandb_run_name"] for index, row in table9.items()}
    )
    observed["table9_wandb_state"] = observed["panel_index"].map(
        {index: row["wandb_state"] for index, row in table9.items()}
    )
    observed["table9_wandb_created_at"] = observed["panel_index"].map(
        {index: row["wandb_created_at"] for index, row in table9.items()}
    )
    observed["table9_checkpoint_path"] = observed["panel_index"].map(
        {index: row["checkpoint_path"] for index, row in table9.items()}
    )
    return observed


def add_same_seed_control_deltas(observed: pd.DataFrame) -> pd.DataFrame:
    result = observed.copy()
    control_keys = ["anchor_id", "seed_block"]
    controls = result[result["is_control"]].copy()
    duplicate_controls = controls.duplicated(control_keys, keep=False)
    if duplicate_controls.any():
        duplicate_rows = controls.loc[duplicate_controls, ["candidate_id", *control_keys]].to_dict("records")
        raise ValueError(f"Expected one tied control per anchor and seed block: {duplicate_rows}")
    treatment_keys = set(map(tuple, result.loc[~result["is_control"], control_keys].itertuples(index=False, name=None)))
    available_control_keys = set(map(tuple, controls[control_keys].itertuples(index=False, name=None)))
    if not treatment_keys.issubset(available_control_keys):
        raise ValueError(f"Treatments lack same-seed controls: {sorted(treatment_keys - available_control_keys)}")
    controls = controls.set_index(control_keys)
    for metric in TARGETS.values():
        control_by_key = controls[metric].to_dict()
        control_column = f"{metric}_same_seed_control"
        delta_column = f"{metric}_delta_vs_control"
        result[control_column] = [
            control_by_key.get((anchor, seed_block), np.nan)
            for anchor, seed_block in zip(result["anchor_id"], result["seed_block"], strict=True)
        ]
        result[delta_column] = result[metric] - result[control_column]
    return result


def cluster_bootstrap_mean_interval(
    frame: pd.DataFrame,
    *,
    value_column: str,
    cluster_column: str,
    seed: int,
    draws: int = 20_000,
) -> tuple[float, float]:
    clusters = [group[value_column].dropna().to_numpy(dtype=float) for _, group in frame.groupby(cluster_column)]
    if len(clusters) < 2 or any(len(cluster) == 0 for cluster in clusters):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        means[draw] = np.concatenate([clusters[index] for index in sampled]).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def treatment_summary(observed: pd.DataFrame) -> pd.DataFrame:
    treatments = observed[~observed["is_control"]]
    rows: list[dict[str, object]] = []
    for target_index, (target, metric) in enumerate(TARGETS.items()):
        delta_column = f"{metric}_delta_vs_control"
        for anchor_index, (anchor, group) in enumerate(treatments.groupby("anchor_id")):
            deltas = group[delta_column].dropna()
            ci_low, ci_high = cluster_bootstrap_mean_interval(
                group,
                value_column=delta_column,
                cluster_column="seed_block",
                seed=20_260_726 + 100 * target_index + anchor_index,
            )
            rows.append(
                {
                    "target": target,
                    "anchor_id": anchor,
                    "observed_treatments": len(deltas),
                    "planned_treatments": len(group),
                    "mean_delta": deltas.mean(),
                    "median_delta": deltas.median(),
                    "sd_delta": deltas.std(ddof=1),
                    "seed_block_sensitivity_low": ci_low,
                    "seed_block_sensitivity_high": ci_high,
                    "fraction_better": (deltas < 0).mean(),
                    "best_delta": deltas.min(),
                    "worst_delta": deltas.max(),
                }
            )
    return pd.DataFrame(rows)


def pair_decomposition(observed: pd.DataFrame) -> pd.DataFrame:
    treatments = observed[~observed["is_control"]]
    rows: list[dict[str, object]] = []
    keys = ["anchor_id", "pair_id", "replicate_index", "seed_block"]
    for target, metric in TARGETS.items():
        control_column = f"{metric}_same_seed_control"
        for key, group in treatments.dropna(subset=[metric]).groupby(keys):
            sign_counts = group["sign"].value_counts().to_dict()
            if len(group) != 2 or sign_counts != {"plus": 1, "minus": 1}:
                raise ValueError(f"Expected one plus and one minus row for {target} pair {key}: {sign_counts}")
            plus = group[group["sign"].eq("plus")].iloc[0]
            minus = group[group["sign"].eq("minus")].iloc[0]
            control = float(plus[control_column])
            if control != float(minus[control_column]):
                raise ValueError(f"Pair {key} does not share a same-seed {target} control")
            plus_value = float(plus[metric])
            minus_value = float(minus[metric])
            rows.append(
                {
                    "target": target,
                    "anchor_id": key[0],
                    "pair_id": key[1],
                    "replicate_index": key[2],
                    "seed_block": key[3],
                    "direction_family": plus["direction_family"],
                    "direction_id": plus["direction_id"],
                    "hypothesis": plus["hypothesis"],
                    "phase_tv": plus["phase_tv"],
                    "plus_candidate_id": plus["candidate_id"],
                    "minus_candidate_id": minus["candidate_id"],
                    "plus_bpb": plus_value,
                    "minus_bpb": minus_value,
                    "same_seed_control_bpb": control,
                    "order_half_effect_plus_minus": (plus_value - minus_value) / 2,
                    "symmetric_asymmetry_cost": (plus_value + minus_value) / 2 - control,
                    "best_orientation_delta": min(plus_value, minus_value) - control,
                    "plus_delta": plus_value - control,
                    "minus_delta": minus_value - control,
                    "better_orientation": (
                        "plus_named_left_later" if plus_value < minus_value else "minus_named_left_earlier"
                    ),
                }
            )
    return pd.DataFrame(rows)


def pair_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target, anchor), group in pairs.groupby(["target", "anchor_id"]):
        rows.append(
            {
                "target": target,
                "anchor_id": anchor,
                "complete_pairs": len(group),
                "mean_order_half_effect": group["order_half_effect_plus_minus"].mean(),
                "mean_absolute_order_half_effect": group["order_half_effect_plus_minus"].abs().mean(),
                "sd_order_half_effect": group["order_half_effect_plus_minus"].std(ddof=1),
                "mean_symmetric_asymmetry_cost": group["symmetric_asymmetry_cost"].mean(),
                "median_symmetric_asymmetry_cost": group["symmetric_asymmetry_cost"].median(),
                "best_orientation_delta": group["best_orientation_delta"].min(),
            }
        )
    return pd.DataFrame(rows)


def hypothesis_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target, anchor, hypothesis), group in pairs.groupby(["target", "anchor_id", "hypothesis"]):
        rows.append(
            {
                "target": target,
                "anchor_id": anchor,
                "hypothesis": hypothesis,
                "complete_pairs": len(group),
                "plus_named_group_later_better_count": (group["order_half_effect_plus_minus"] < 0).sum(),
                "mean_order_half_effect": group["order_half_effect_plus_minus"].mean(),
                "mean_absolute_order_half_effect": group["order_half_effect_plus_minus"].abs().mean(),
                "mean_symmetric_asymmetry_cost": group["symmetric_asymmetry_cost"].mean(),
                "median_best_orientation_delta": group["best_orientation_delta"].median(),
                "best_orientation_delta": group["best_orientation_delta"].min(),
            }
        )
    return pd.DataFrame(rows)


def direction_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["target", "anchor_id", "direction_family", "direction_id", "hypothesis", "phase_tv"]
    for key, group in pairs.groupby(keys):
        rows.append(
            {
                "target": key[0],
                "anchor_id": key[1],
                "direction_family": key[2],
                "direction_id": key[3],
                "hypothesis": key[4],
                "phase_tv": key[5],
                "complete_pairs": len(group),
                "plus_named_group_later_better_count": (group["order_half_effect_plus_minus"] < 0).sum(),
                "mean_order_half_effect": group["order_half_effect_plus_minus"].mean(),
                "mean_symmetric_asymmetry_cost": group["symmetric_asymmetry_cost"].mean(),
                "best_orientation_delta": group["best_orientation_delta"].min(),
            }
        )
    return pd.DataFrame(rows)


def sentinel_repeat_comparisons(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["target", "anchor_id", "pair_id"]
    for key, group in pairs.groupby(keys):
        if len(group) == 1:
            continue
        if len(group) != 2 or set(group["replicate_index"]) != {0, 1}:
            raise ValueError(f"Unexpected repeated antithetic pair structure for {key}")
        base = group[group["replicate_index"].eq(0)].iloc[0]
        repeat = group[group["replicate_index"].eq(1)].iloc[0]
        rows.append(
            {
                "target": key[0],
                "anchor_id": key[1],
                "pair_id": key[2],
                "direction_id": base["direction_id"],
                "base_seed_block": base["seed_block"],
                "repeat_seed_block": repeat["seed_block"],
                "base_order_half_effect": base["order_half_effect_plus_minus"],
                "repeat_order_half_effect": repeat["order_half_effect_plus_minus"],
                "order_half_effect_difference": (
                    repeat["order_half_effect_plus_minus"] - base["order_half_effect_plus_minus"]
                ),
                "base_symmetric_asymmetry_cost": base["symmetric_asymmetry_cost"],
                "repeat_symmetric_asymmetry_cost": repeat["symmetric_asymmetry_cost"],
                "symmetric_asymmetry_cost_difference": (
                    repeat["symmetric_asymmetry_cost"] - base["symmetric_asymmetry_cost"]
                ),
                "base_best_orientation_delta": base["best_orientation_delta"],
                "repeat_best_orientation_delta": repeat["best_orientation_delta"],
                "best_orientation_delta_difference": repeat["best_orientation_delta"] - base["best_orientation_delta"],
                "orientation_agrees": base["better_orientation"] == repeat["better_orientation"],
            }
        )
    return pd.DataFrame(rows)


def sentinel_repeat_summary(comparisons: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target, group in comparisons.groupby("target"):
        order_difference = group["order_half_effect_difference"]
        cost_difference = group["symmetric_asymmetry_cost_difference"]
        best_difference = group["best_orientation_delta_difference"]
        rows.append(
            {
                "target": target,
                "repeated_antithetic_pairs": len(group),
                "orientation_agreement_fraction": group["orientation_agrees"].mean(),
                "order_half_effect_repeat_difference_sd": order_difference.std(ddof=1),
                "implied_per_run_noise_sd": order_difference.std(ddof=1),
                "implied_order_half_effect_noise_sd": order_difference.std(ddof=1) / np.sqrt(2),
                "asymmetry_cost_repeat_difference_sd": cost_difference.std(ddof=1),
                "best_delta_repeat_difference_sd": best_difference.std(ddof=1),
                "best_delta_repeat_difference_max_abs": best_difference.abs().max(),
            }
        )
    return pd.DataFrame(rows)


def best_rows(observed: pd.DataFrame) -> pd.DataFrame:
    treatments = observed[~observed["is_control"]]
    rows: list[pd.Series] = []
    for target, metric in TARGETS.items():
        delta_column = f"{metric}_delta_vs_control"
        for _, group in treatments.dropna(subset=[delta_column]).groupby("anchor_id"):
            best = group.loc[group[delta_column].idxmin()].copy()
            best["target"] = target
            best["metric"] = metric
            best["delta_vs_control"] = best[delta_column]
            rows.append(best)
    return pd.DataFrame(rows)


def write_plots(observed: pd.DataFrame, pairs: pd.DataFrame, output_dir: Path) -> None:
    treatments = observed[~observed["is_control"]]
    effect_rows: list[pd.DataFrame] = []
    for target, metric in TARGETS.items():
        frame = treatments.dropna(subset=[metric]).copy()
        frame["target"] = target
        frame["delta_vs_control"] = frame[f"{metric}_delta_vs_control"]
        effect_rows.append(frame)
    effects = pd.concat(effect_rows, ignore_index=True)
    effect_figure = px.scatter(
        effects,
        x="phase_tv",
        y="delta_vs_control",
        color="delta_vs_control",
        color_continuous_scale="RdYlGn_r",
        symbol="direction_family",
        facet_row="target",
        facet_col="anchor_id",
        hover_name="candidate_id",
        hover_data=["hypothesis", "sign", "seed_block", "replicate_index"],
        labels={"phase_tv": "Phase total variation", "delta_vs_control": "BPB minus same-seed tied control"},
        title="60M fixed-aggregate phase effects",
    )
    effect_figure.add_hline(y=0, line_color="#173247", line_width=1.5)
    effect_figure.update_layout(template="plotly_white", height=900)
    effect_figure.write_html(output_dir / "phase_effects.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    pair_figure = px.scatter(
        pairs,
        x="order_half_effect_plus_minus",
        y="symmetric_asymmetry_cost",
        color="best_orientation_delta",
        color_continuous_scale="RdYlGn_r",
        symbol="direction_family",
        facet_row="target",
        facet_col="anchor_id",
        hover_name="pair_id",
        hover_data=["hypothesis", "phase_tv", "better_orientation", "best_orientation_delta"],
        labels={
            "order_half_effect_plus_minus": "Order half-effect: (+ later - - earlier) / 2",
            "symmetric_asymmetry_cost": "Mean asymmetry cost relative to tied control",
        },
        title="Antithetic order and curvature decomposition",
    )
    pair_figure.add_hline(y=0, line_color="#173247", line_width=1.5)
    pair_figure.add_vline(x=0, line_color="#173247", line_width=1.5)
    pair_figure.update_layout(template="plotly_white", height=900)
    pair_figure.write_html(output_dir / "antithetic_decomposition.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[columns].copy()
    for column in display.columns:
        if pd.api.types.is_bool_dtype(display[column]):
            display[column] = display[column].map(lambda value: str(bool(value)) if pd.notna(value) else "")
        elif pd.api.types.is_integer_dtype(display[column]):
            display[column] = display[column].map(lambda value: str(int(value)) if pd.notna(value) else "")
        elif pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.6f}" if pd.notna(value) else "")
    return display.to_markdown(index=False, disable_numparse=True)


def write_report(
    observed: pd.DataFrame,
    treatments: pd.DataFrame,
    pairs: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    directions: pd.DataFrame,
    repeat_comparisons: pd.DataFrame,
    repeat_metrics: pd.DataFrame,
    best: pd.DataFrame,
    output_dir: Path,
) -> None:
    coverage = {target: int(observed[metric].notna().sum()) for target, metric in TARGETS.items()}
    control_rows: list[dict[str, object]] = []
    controls = observed[observed["is_control"]]
    for target, metric in TARGETS.items():
        for anchor, group in controls.groupby("anchor_id"):
            control_rows.append(
                {
                    "target": target,
                    "anchor_id": anchor,
                    "observed_controls": group[metric].notna().sum(),
                    "mean": group[metric].mean(),
                    "sd": group[metric].std(ddof=1),
                }
            )
    control_summary = pd.DataFrame(control_rows)

    paired_targets = observed.dropna(subset=list(TARGETS.values()))
    paired_treatments = paired_targets[~paired_targets["is_control"]]
    effect_correlation = (
        paired_treatments[["uncheatable_bpb_delta_vs_control", "table9_macro_bpb_delta_vs_control"]]
        .corr(method="spearman")
        .iloc[0, 1]
    )
    anchor_correlations = {
        anchor: (
            group[["uncheatable_bpb_delta_vs_control", "table9_macro_bpb_delta_vs_control"]]
            .corr(method="spearman")
            .iloc[0, 1]
        )
        for anchor, group in paired_treatments.groupby("anchor_id")
    }
    treatment_table = markdown_table(
        treatments,
        [
            "target",
            "anchor_id",
            "observed_treatments",
            "planned_treatments",
            "mean_delta",
            "median_delta",
            "sd_delta",
            "seed_block_sensitivity_low",
            "seed_block_sensitivity_high",
            "fraction_better",
            "best_delta",
            "worst_delta",
        ],
    )
    pair_table = markdown_table(
        pair_metrics,
        [
            "target",
            "anchor_id",
            "complete_pairs",
            "mean_order_half_effect",
            "mean_absolute_order_half_effect",
            "sd_order_half_effect",
            "mean_symmetric_asymmetry_cost",
            "median_symmetric_asymmetry_cost",
            "best_orientation_delta",
        ],
    )
    mechanism_table = markdown_table(
        directions[directions["direction_id"].isin(["curated_noncc_vs_cc", "dolmino_vs_broad", "cc_high_vs_remainder"])],
        [
            "target",
            "anchor_id",
            "direction_family",
            "direction_id",
            "phase_tv",
            "complete_pairs",
            "plus_named_group_later_better_count",
            "mean_order_half_effect",
            "mean_symmetric_asymmetry_cost",
            "best_orientation_delta",
        ],
    )
    repeat_table = markdown_table(
        repeat_metrics,
        [
            "target",
            "repeated_antithetic_pairs",
            "orientation_agreement_fraction",
            "order_half_effect_repeat_difference_sd",
            "implied_per_run_noise_sd",
            "implied_order_half_effect_noise_sd",
            "asymmetry_cost_repeat_difference_sd",
            "best_delta_repeat_difference_sd",
            "best_delta_repeat_difference_max_abs",
        ],
    )
    repeat_detail_table = markdown_table(
        repeat_comparisons,
        [
            "target",
            "anchor_id",
            "pair_id",
            "base_order_half_effect",
            "repeat_order_half_effect",
            "base_best_orientation_delta",
            "repeat_best_orientation_delta",
            "orientation_agrees",
        ],
    )
    best_table = markdown_table(
        best,
        [
            "target",
            "anchor_id",
            "candidate_id",
            "direction_family",
            "hypothesis",
            "sign",
            "phase_tv",
            "delta_vs_control",
        ],
    )
    control_table = markdown_table(
        control_summary,
        ["target", "anchor_id", "observed_controls", "mean", "sd"],
    )
    nonfinished_training = observed[observed["training_wandb_state"].ne("finished")]
    nonfinished_table = markdown_table(
        nonfinished_training,
        [
            "panel_index",
            "candidate_id",
            "training_wandb_state",
            "uncheatable_metric_step",
            "table9_wandb_state",
        ],
    )
    provenance_counts = observed["table9_provenance_mode"].value_counts().to_dict()
    if coverage["table9"] == len(observed):
        table9_completion_note = (
            f"- Native Table-9 coverage is complete for all {len(observed)} rows; no target values were imputed."
        )
    else:
        table9_completion_note = (
            f"- Table-9 conclusions remain provisional until all {len(observed)} native evaluations are recovered. "
            "Missing rows are retained as missing and never imputed."
        )

    report = f"""# 60M fixed-aggregate phase-order results

## Coverage

- Manifest rows: {len(observed)}.
- Uncheatable: {coverage["uncheatable"]}/{len(observed)}.
- Native Table-9: {coverage["table9"]}/{len(observed)}.
- Rows carrying both targets: {len(paired_targets)}.
- Non-control rows carrying both targets: {len(paired_treatments)}.
- Spearman correlation between non-control same-seed phase effects across targets: {effect_correlation:.3f}.
- Per-anchor effect correlations: {", ".join(f"{anchor}={value:.3f}" for anchor, value in anchor_correlations.items())}.

Uncheatable values come from each checkpoint's persisted step-{EXPECTED_CHECKPOINT_STEP}
`checkpoints/eval_metrics.jsonl`, not W&B run summaries. Native Table-9 values
come only from finished evaluator runs whose checkpoint and candidate provenance
match the SHA-bound source manifest. Table-9 provenance is candidate-ID-bound
for {provenance_counts.get("candidate_id", 0)} rows and source-run/checkpoint-index-bound
for {provenance_counts.get("source_run_name_and_checkpoint_index", 0)} direct-recovery rows.

The panel holds aggregate mixture, training compute, model, seeds, and phase
fractions fixed within each comparison. For each antithetic pair, `+` places
the named left-hand group later and `-` places it earlier. The decomposition is

$$
o = \\frac{{L_+ - L_-}}{{2}}, \\qquad
c = \\frac{{L_+ + L_-}}{{2}} - L_0,
$$

where \\(o\\) is the odd phase-order effect and \\(c\\) is the even cost or benefit
of introducing that magnitude of asymmetry around the tied control.

## Same-seed treatment effects

{treatment_table}

## Antithetic decomposition

{pair_table}

The absolute order effect is descriptive, not noise-corrected. "Best of the two
orientations" fractions are intentionally omitted because taking a minimum of
two noisy observations has a null probability above 0.5. Every
`best_orientation_delta` is likewise a selected descriptive extreme, not an
unadjusted significance estimate.

## Mechanism checks

For these named directions, `plus` means the named group is later. Counts are
reported directly rather than compared through target-independent BPB thresholds.

{mechanism_table}

Only `dolmino_vs_broad` and `cc_high_vs_remainder` have fresh-seed sentinel
repeats. `curated_noncc_vs_cc` recurs across two anchors and two phase-TV
magnitudes, but has no dedicated fresh-seed repeat.

## Sentinel-repeat reproducibility

The panel repeats four antithetic pairs under fresh seed blocks. Under
independent equal-variance run noise, the SD of repeat differences in the order
half-effect estimates the per-run noise SD; dividing it by \\(\\sqrt{{2}}\\)
estimates the noise SD of one order half-effect. With only four repeated pairs
per target, both estimates are themselves noisy.

{repeat_table}

{repeat_detail_table}

For Table-9, the repeat-implied per-run noise estimate is larger than the
four-control SD reported below. The report therefore treats neither sparse
estimate as calibrated enough for post-selection significance claims and uses
the larger repeat-implied scale when making qualitative comparisons.

## Best observed treatment per anchor

{best_table}

Each minimum was selected after inspecting 66 treatments. These are exploratory
extrema, not unadjusted confirmatory estimates.

## Tied-control noise scale

{control_table}

Each tied-control SD has only four observations (three degrees of freedom).
Seed-block sensitivity intervals above quantify how the designed-panel mean
changes when the four observed seed blocks are resampled. They are not
confidence intervals over a random population of policies.

## Tracker-state exceptions

Four training trackers remain `crashed`, but each row has a persisted
step-{EXPECTED_CHECKPOINT_STEP} Uncheatable evaluation and a finished native
Table-9 evaluation of the exact same final checkpoint. The analysis allow-lists
these four indices and fails on any change to that set.

{nonfinished_table}

## Interpretation

- Phase order is identifiable in specific designed directions at 60M, but not
  as a generic benefit from arbitrary asymmetry. The best observed treatments
  are minima selected from 66 candidates per anchor, so this report makes no
  unadjusted significance claim from those extrema. Untouched confirmation is
  required.
- Curated non-Common-Crawl later and broad Common Crawl earlier is the strongest
  recurring direction across anchors and targets. Dolmino-late also repeats in
  the expected direction.
- The Common-Crawl high-quality-vs-remainder intervention favors the named group
  earlier across both targets, a cross-target counterexample to a universal
  "higher quality later" rule.
- Agnostic spanning directions have no universal signed order. Optimization
  must learn mechanism-specific order effects and an even asymmetry cost; total
  phase TV alone is insufficient.
{table9_completion_note}

## Artifacts

- `observed_results.csv`: all manifest fields, W&B IDs, targets, and same-seed deltas.
- `pair_decomposition.csv`: exact antithetic odd/even decomposition.
- `treatment_summary.csv`, `pair_summary.csv`, `hypothesis_summary.csv`, `direction_summary.csv`.
- `sentinel_repeat_comparisons.csv`, `sentinel_repeat_summary.csv`.
- `phase_effects.html`, `antithetic_decomposition.html`.
"""
    (output_dir / "report.md").write_text(report)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    validate_manifest(manifest, args.manifest, EXPECTED_MANIFEST_SHA256)
    observed = add_same_seed_control_deltas(collect_observed_results(manifest, timeout=args.wandb_timeout))
    treatments = treatment_summary(observed)
    pairs = pair_decomposition(observed)
    pair_metrics = pair_summary(pairs)
    hypotheses = hypothesis_summary(pairs)
    directions = direction_summary(pairs)
    repeat_comparisons = sentinel_repeat_comparisons(pairs)
    repeat_metrics = sentinel_repeat_summary(repeat_comparisons)
    best = best_rows(observed)

    observed.to_csv(args.output_dir / "observed_results.csv", index=False)
    treatments.to_csv(args.output_dir / "treatment_summary.csv", index=False)
    pairs.to_csv(args.output_dir / "pair_decomposition.csv", index=False)
    pair_metrics.to_csv(args.output_dir / "pair_summary.csv", index=False)
    hypotheses.to_csv(args.output_dir / "hypothesis_summary.csv", index=False)
    directions.to_csv(args.output_dir / "direction_summary.csv", index=False)
    repeat_comparisons.to_csv(args.output_dir / "sentinel_repeat_comparisons.csv", index=False)
    repeat_metrics.to_csv(args.output_dir / "sentinel_repeat_summary.csv", index=False)
    best.to_csv(args.output_dir / "best_treatments.csv", index=False)
    write_plots(observed, pairs, args.output_dir)
    write_report(
        observed,
        treatments,
        pairs,
        pair_metrics,
        directions,
        repeat_comparisons,
        repeat_metrics,
        best,
        args.output_dir,
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
