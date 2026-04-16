# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "numpy", "pandas", "wandb"]
# ///
"""Build current 300M signal-to-noise tables and compare them against the 60M baseline."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
import re

import fsspec
import pandas as pd
import wandb

from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import STRATIFIED_RUN_NAME
from experiments.domain_phase_mix.two_phase_many_observed_runs import load_original_qsplit240_with_core_baselines

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_60M_RANKED_CSV = SCRIPT_DIR / "eval_signal_to_noise_ranked.csv"
INPUT_300M_COMPLETED_CSV = SCRIPT_DIR / "qsplit240_300m_6b_completed_vs_60m.csv"
OUTPUT_300M_RANKED_CSV = SCRIPT_DIR / "eval_signal_to_noise_ranked_300m_current.csv"
OUTPUT_COMPARISON_CSV = SCRIPT_DIR / "eval_signal_to_noise_60m_vs_300m_current.csv"
OUTPUT_SUMMARY_JSON = SCRIPT_DIR / "eval_signal_to_noise_60m_vs_300m_current_summary.json"

WANDB_ENTITY = "marin-community"
WANDB_PROJECT = "marin"
QSPLIT240_300M_RESULTS_GLOB = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b/"
    "shard_??of08/collect_results*/results.csv"
)
QSPLIT240_300M_PARITY_RESULTS_GLOB = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_qsplit240_300m_6b_parity_rerun/**/collect_results*/results.csv"
)
RUN00097_300M_FIXED_SUBSET_RESULTS_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_run00097_300m_6b_fixed_subset/collect_results-605e6a/results.csv"
)
RUN00097_300M_PARITY_RESULTS_GLOB = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_run00097_300m_6b_parity_rerun/collect_results*/results.csv"
)
RUN_ROOT_PATTERN = re.compile(r"([^/]+-[0-9a-f]+)/checkpoints/eval_metrics\.jsonl$")


def _read_gcs_csv(uri: str) -> pd.DataFrame:
    data = subprocess.check_output(["gsutil", "cat", uri], text=True)
    return pd.read_csv(io.StringIO(data))


def _list_gcs_paths(pattern: str) -> list[str]:
    try:
        output = subprocess.check_output(["gsutil", "ls", pattern], text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def _allowed_run_names() -> set[str]:
    run_names = {run.run_name for run in load_original_qsplit240_with_core_baselines()}
    run_names.add(STRATIFIED_RUN_NAME)
    return run_names


def _wandb_run_id_from_checkpoint_root(checkpoint_root: str) -> str:
    return checkpoint_root.rstrip("/").rsplit("/", 1)[-1]


def _load_signal_row_from_checkpoint_root(fs: fsspec.AbstractFileSystem, checkpoint_root: str) -> dict[str, object]:
    metrics_path = checkpoint_root.removeprefix("gs://") + "/checkpoints/eval_metrics.jsonl"
    root_name_match = RUN_ROOT_PATTERN.search(metrics_path)
    if root_name_match is None:
        raise ValueError(f"Could not parse run root from {metrics_path}")
    run_name = root_name_match.group(1).rsplit("-", 1)[0]

    with fs.open(metrics_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"No eval metrics found at {metrics_path}")

    payload = json.loads(lines[-1])
    payload["run_name"] = run_name
    payload["checkpoint_root"] = checkpoint_root
    return payload


def _load_qsplit_300m_harvest_rows() -> pd.DataFrame:
    shard_uris = sorted(_list_gcs_paths(QSPLIT240_300M_RESULTS_GLOB))
    if not shard_uris:
        return pd.DataFrame()
    frame = pd.concat([_read_gcs_csv(uri) for uri in shard_uris], ignore_index=True)
    frame = frame.loc[frame["status"] == "completed"].copy()
    frame["wandb_run_id"] = frame["wandb_run_id"].astype(str)
    return frame


def _load_optional_results_glob(pattern: str) -> pd.DataFrame:
    uris = sorted(_list_gcs_paths(pattern))
    if not uris:
        return pd.DataFrame()
    return pd.concat([_read_gcs_csv(uri) for uri in uris], ignore_index=True)


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith(("eval/", "lm_eval/")))


def _overlay_metric_columns(base: pd.DataFrame, overlay: pd.DataFrame, *, key_column: str) -> pd.DataFrame:
    if overlay.empty:
        return base

    overlay = overlay.drop_duplicates(subset=[key_column], keep="last").copy()
    metric_columns = _metric_columns(overlay)
    if not metric_columns:
        return base

    merged = base.merge(
        overlay[[key_column, *metric_columns]],
        on=key_column,
        how="left",
        suffixes=("", "__overlay"),
    )
    for column in metric_columns:
        overlay_column = f"{column}__overlay"
        if overlay_column not in merged.columns:
            continue
        if column in merged.columns:
            merged[column] = merged[overlay_column].combine_first(merged[column])
        else:
            merged[column] = merged[overlay_column]
        merged = merged.drop(columns=[overlay_column])

    return merged


def _wandb_summary_row(run: wandb.apis.public.Run) -> dict[str, object]:
    row: dict[str, object] = {
        "wandb_run_id": run.id,
        "wandb_run_name": run.name,
        "status": run.state,
    }
    for key, value in run.summary.items():
        if isinstance(value, int | float) and key.startswith(("eval/", "lm_eval/")):
            row[key] = value
    return row


def _load_wandb_rows_by_run_id(run_ids: list[str]) -> dict[str, dict[str, object]]:
    api = wandb.Api()
    rows: dict[str, dict[str, object]] = {}
    for run_id in run_ids:
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
        rows[run_id] = _wandb_summary_row(run)
    return rows


def _load_300m_signal_frame() -> pd.DataFrame:
    fs = fsspec.filesystem("gs")
    completed = pd.read_csv(INPUT_300M_COMPLETED_CSV)
    completed = completed.loc[completed["run_name"].isin(_allowed_run_names())].copy()
    completed["wandb_run_id"] = completed["checkpoint_root"].map(_wandb_run_id_from_checkpoint_root)

    harvest = _load_qsplit_300m_harvest_rows()
    harvested_by_id = (
        harvest.drop_duplicates(subset=["wandb_run_id"], keep="last").set_index("wandb_run_id").to_dict(orient="index")
        if not harvest.empty
        else {}
    )
    missing_run_ids = [
        run_id for run_id in completed["wandb_run_id"].astype(str).tolist() if run_id not in harvested_by_id
    ]
    wandb_by_id = _load_wandb_rows_by_run_id(missing_run_ids)

    rows: list[dict[str, object]] = []
    harvest_count = 0
    wandb_count = 0
    checkpoint_fallback_count = 0

    for completed_row in completed.to_dict(orient="records"):
        run_id = str(completed_row["wandb_run_id"])
        run_name = str(completed_row["run_name"])
        checkpoint_root = str(completed_row["checkpoint_root"])

        if run_id in harvested_by_id:
            row = dict(harvested_by_id[run_id])
            row["run_name"] = run_name
            row["checkpoint_root"] = checkpoint_root
            rows.append(row)
            harvest_count += 1
            continue

        if run_id in wandb_by_id:
            row = dict(wandb_by_id[run_id])
            row["run_name"] = run_name
            row["checkpoint_root"] = checkpoint_root
            rows.append(row)
            wandb_count += 1
            continue

        row = _load_signal_row_from_checkpoint_root(fs, checkpoint_root)
        row["wandb_run_id"] = run_id
        rows.append(row)
        checkpoint_fallback_count += 1

    combined = pd.DataFrame(rows).sort_values("run_name").reset_index(drop=True)
    parity = _load_optional_results_glob(QSPLIT240_300M_PARITY_RESULTS_GLOB)
    parity = parity.loc[parity["run_name"].isin(_allowed_run_names())].copy() if not parity.empty else parity
    combined = _overlay_metric_columns(combined, parity, key_column="run_name")
    combined.attrs["harvest_count"] = harvest_count
    combined.attrs["wandb_count"] = wandb_count
    combined.attrs["checkpoint_fallback_count"] = checkpoint_fallback_count
    combined.attrs["parity_overlay_rows"] = len(parity)
    return combined


def _snr_row(
    *,
    eval_name: str,
    metric: str,
    source: str,
    primary_metric_kind: str,
    signal_values: pd.Series,
    noise_values: pd.Series,
) -> dict[str, float | int | str]:
    signal = signal_values.dropna()
    noise = noise_values.dropna()
    if len(signal) < 2:
        raise ValueError(f"{metric}: need at least 2 signal values, got {len(signal)}")
    if len(noise) < 2:
        raise ValueError(f"{metric}: need at least 2 noise values, got {len(noise)}")

    signal_scale = float(signal.std(ddof=1))
    noise_scale = float(noise.std(ddof=1))
    return {
        "eval_name": eval_name,
        "metric": metric,
        "source": source,
        "primary_metric_kind": primary_metric_kind,
        "signal_n": len(signal),
        "noise_n": len(noise),
        "signal_scale": signal_scale,
        "noise_scale": noise_scale,
        "signal_range": float(signal.max() - signal.min()),
        "signal_to_noise": signal_scale / noise_scale,
    }


def _build_300m_ranked_table(signal_300m: pd.DataFrame) -> pd.DataFrame:
    ranked_60m = pd.read_csv(INPUT_60M_RANKED_CSV)
    noise_300m = _read_gcs_csv(RUN00097_300M_FIXED_SUBSET_RESULTS_URI)
    noise_300m = noise_300m.loc[noise_300m["cohort"] == "seed_sweep"].reset_index(drop=True)
    parity_noise = _load_optional_results_glob(RUN00097_300M_PARITY_RESULTS_GLOB)
    parity_noise = (
        parity_noise.loc[parity_noise["cohort"] == "seed_sweep"].reset_index(drop=True)
        if not parity_noise.empty
        else parity_noise
    )
    noise_300m = _overlay_metric_columns(noise_300m, parity_noise, key_column="run_name")

    rows: list[dict[str, float | int | str]] = []
    for _, meta_row in ranked_60m.iterrows():
        metric = str(meta_row["metric"])
        if metric not in signal_300m.columns or metric not in noise_300m.columns:
            continue
        rows.append(
            _snr_row(
                eval_name=str(meta_row["eval_name"]),
                metric=metric,
                source=str(meta_row["source"]),
                primary_metric_kind=str(meta_row["primary_metric_kind"]),
                signal_values=signal_300m[metric],
                noise_values=noise_300m[metric],
            )
        )

    return pd.DataFrame(rows).sort_values("signal_to_noise", ascending=False).reset_index(drop=True)


def _build_comparison_table(frame_300m: pd.DataFrame) -> pd.DataFrame:
    ranked_60m = pd.read_csv(INPUT_60M_RANKED_CSV)
    merged = ranked_60m.merge(
        frame_300m,
        on=["eval_name", "metric", "source", "primary_metric_kind"],
        how="inner",
        suffixes=("_60m", "_300m_current"),
    )
    merged["signal_to_noise_delta"] = merged["signal_to_noise_300m_current"] - merged["signal_to_noise_60m"]
    merged["signal_to_noise_ratio"] = merged["signal_to_noise_300m_current"] / merged["signal_to_noise_60m"]
    return merged.sort_values("signal_to_noise_300m_current", ascending=False).reset_index(drop=True)


def main() -> None:
    signal_300m = _load_300m_signal_frame()
    ranked_300m = _build_300m_ranked_table(signal_300m)
    comparison = _build_comparison_table(ranked_300m)

    ranked_300m.to_csv(OUTPUT_300M_RANKED_CSV, index=False)
    comparison.to_csv(OUTPUT_COMPARISON_CSV, index=False)

    summary = {
        "n_metrics_compared": len(comparison),
        "signal_n_300m_current_min": int(comparison["signal_n_300m_current"].min()),
        "signal_n_300m_current_max": int(comparison["signal_n_300m_current"].max()),
        "noise_n_300m": int(comparison["noise_n_300m_current"].iloc[0]) if len(comparison) else 0,
        "signal_rows_from_harvest": int(signal_300m.attrs.get("harvest_count", 0)),
        "signal_rows_from_wandb": int(signal_300m.attrs.get("wandb_count", 0)),
        "signal_rows_from_checkpoint_fallback": int(signal_300m.attrs.get("checkpoint_fallback_count", 0)),
        "signal_rows_from_parity_overlay": int(signal_300m.attrs.get("parity_overlay_rows", 0)),
        "metrics_missing_at_300m": sorted(set(pd.read_csv(INPUT_60M_RANKED_CSV)["metric"]) - set(ranked_300m["metric"])),
        "top_300m_metrics": (
            comparison.head(10)[["metric", "signal_to_noise_300m_current", "signal_to_noise_60m"]].to_dict(
                orient="records"
            )
        ),
    }
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"Wrote {OUTPUT_300M_RANKED_CSV}")
    print(f"Wrote {OUTPUT_COMPARISON_CSV}")
    print(f"Wrote {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
