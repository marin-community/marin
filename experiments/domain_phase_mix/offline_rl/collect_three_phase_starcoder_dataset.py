# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect 3-phase StarCoder trajectories for offline RL."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsspec
import pandas as pd

from experiments.domain_phase_mix.offline_rl.contracts import DEFAULT_OBJECTIVE_METRIC

logger = logging.getLogger(__name__)

DEFAULT_DISPLAY_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/three_phase_starcoder"
DEFAULT_SOURCE_EXPERIMENTS = (
    "pinlin_calvin_xu/data_mixture/three_phase_starcoder_1",
    "pinlin_calvin_xu/data_mixture/three_phase_starcoder_2",
)
DEFAULT_HISTORY_KEYS = (
    "train/loss",
    "eval/loss",
    DEFAULT_OBJECTIVE_METRIC,
    "throughput/total_tokens",
)
EXPECTED_RUN_COUNT = 160
DEFAULT_HISTORY_SAMPLES = 20000

_RUN_RE = re.compile(r"/run_(\d+)$")
_BASE_RE = re.compile(r"/base_(\d+)$")


@dataclass(frozen=True)
class CollectConfig:
    """Config for collecting StarCoder offline-RL trajectories."""

    output_dir: str
    wandb_entity: str = "marin-community"
    wandb_project: str = "marin"
    display_name_prefix: str = DEFAULT_DISPLAY_NAME_PREFIX
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC
    expected_run_count: int = EXPECTED_RUN_COUNT
    csv_fallback_path: str = "experiments/domain_phase_mix/exploratory/three_phase_starcoder.csv"
    source_experiments: tuple[str, ...] = DEFAULT_SOURCE_EXPERIMENTS
    history_samples: int = DEFAULT_HISTORY_SAMPLES


def infer_source_experiment(display_name: str) -> str:
    """Infer source experiment name from a run display name."""
    if "/run_" in display_name:
        return display_name.rsplit("/run_", maxsplit=1)[0]
    if "/base_" in display_name:
        return display_name.rsplit("/base_", maxsplit=1)[0]
    return display_name.rsplit("/", maxsplit=1)[0]


def infer_local_run_id(display_name: str) -> int | None:
    """Infer local run id from a display name."""
    match = _RUN_RE.search(display_name)
    if match:
        return int(match.group(1))

    match = _BASE_RE.search(display_name)
    if match:
        value = int(match.group(1))
        return value if value >= 90000 else 90000 + value

    return None


def phase_weights_to_columns(phase_weights: dict[str, dict[str, float]]) -> dict[str, float]:
    """Flatten nested phase/domain weights into `phase_k_domain` columns."""
    columns: dict[str, float] = {}
    for phase_name, domain_weights in phase_weights.items():
        for domain_name, value in domain_weights.items():
            columns[f"{phase_name}_{domain_name}"] = float(value)
    return columns


def dedupe_history_rows(history_long: pd.DataFrame) -> pd.DataFrame:
    """Dedupe repeated history rows by run/step/metric, keeping the latest."""
    if history_long.empty:
        return history_long

    deduped = (
        history_long.sort_values("_scan_index")
        .drop_duplicates(subset=["wandb_run_id", "step", "metric_key"], keep="last")
        .drop(columns=["_scan_index"])
        .reset_index(drop=True)
    )
    return deduped


def build_wide_history(history_long: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form history rows into a step-indexed wide table."""
    if history_long.empty:
        return pd.DataFrame(
            columns=[
                "wandb_run_id",
                "source_experiment",
                "local_run_id",
                "run_name",
                "step",
                "total_tokens",
            ]
        )

    base_cols = ["wandb_run_id", "source_experiment", "local_run_id", "run_name", "step"]
    total_tokens = history_long.groupby(base_cols, as_index=False)["total_tokens"].last()
    values = history_long.pivot_table(
        index=base_cols,
        columns="metric_key",
        values="metric_value",
        aggfunc="last",
    ).reset_index()
    values.columns.name = None
    return values.merge(total_tokens, on=base_cols, how="left")


def _resolve_weight_config_paths(prefix: str, source_experiment: str) -> list[str]:
    pattern = f"{prefix}/{source_experiment}/weight_configs-*/weight_configs.json"
    fs, base = fsspec.core.url_to_fs(pattern)
    matches = fs.glob(base)
    if not matches:
        return []

    resolved = []
    protocol = fs.protocol[0] if isinstance(fs.protocol, (tuple, list)) else fs.protocol
    for match in matches:
        if protocol:
            resolved.append(f"{protocol}://{match}")
        else:
            resolved.append(match)
    return resolved


def _load_phase_weights_from_weight_configs(
    source_experiments: tuple[str, ...],
    gcs_prefix: str,
) -> dict[tuple[str, int], dict[str, dict[str, float]]]:
    weights: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    for source_experiment in source_experiments:
        config_paths = _resolve_weight_config_paths(gcs_prefix, source_experiment)
        if not config_paths:
            continue
        if len(config_paths) > 1:
            logger.warning("Found multiple weight config files for %s; using %s", source_experiment, config_paths[0])
        with fsspec.open(config_paths[0]) as f:
            payload = json.load(f)
        for item in payload.get("configs", []):
            run_id = int(item["run_id"])
            phase_weights = item.get("phase_weights", {})
            weights[(source_experiment, run_id)] = phase_weights
    return weights


def _load_phase_weights_from_csv(
    csv_path: str | Path,
) -> dict[str, dict[str, dict[str, float]]]:
    path = Path(csv_path)
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if "wandb_run_id" not in df.columns:
        return {}

    fallback: dict[str, dict[str, dict[str, float]]] = {}
    for _, row in df.iterrows():
        wandb_run_id = row.get("wandb_run_id")
        if not isinstance(wandb_run_id, str) or not wandb_run_id:
            continue
        phase_weights: dict[str, dict[str, float]] = {}
        for column, value in row.items():
            if not isinstance(column, str) or not column.startswith("phase_"):
                continue
            if pd.isna(value):
                continue
            parts = column.split("_")
            if len(parts) < 3:
                continue
            phase_name = "_".join(parts[:2])
            domain_name = "_".join(parts[2:])
            phase_weights.setdefault(phase_name, {})
            phase_weights[phase_name][domain_name] = float(value)
        if phase_weights:
            fallback[wandb_run_id] = phase_weights

    return fallback


def _extract_numeric_summary(summary: Any, key: str) -> float | None:
    value = summary.get(key) if hasattr(summary, "get") else None
    if isinstance(value, int | float):
        return float(value)
    return None


def _fetch_history_with_retries(run, keys: tuple[str, ...], samples: int) -> pd.DataFrame:
    for attempt in range(1, 4):
        try:
            return run.history(
                keys=["_step", *keys],
                samples=samples,
                pandas=True,
            )
        except Exception:
            logger.warning(
                "history fetch failed for run=%s attempt=%d/3; retrying",
                run.id,
                attempt,
                exc_info=True,
            )
            if attempt == 3:
                raise
            time.sleep(3 * attempt)
    raise RuntimeError("unreachable")


def collect_history_long_rows(
    run,
    metric_keys: tuple[str, ...],
    history_samples: int,
) -> list[dict[str, Any]]:
    """Collect metric history rows for one W&B run."""
    rows: list[dict[str, Any]] = []
    scan_index = 0
    history = _fetch_history_with_retries(run, metric_keys, samples=history_samples)
    for _, entry in history.iterrows():
        step = entry.get("_step")
        if pd.isna(step):
            continue
        for metric_key in metric_keys:
            if metric_key == "throughput/total_tokens":
                continue
            metric_value = entry.get(metric_key)
            if metric_value is None or pd.isna(metric_value):
                continue
            total_tokens = entry.get("throughput/total_tokens")
            total_tokens_value = None
            if total_tokens is not None and not pd.isna(total_tokens):
                total_tokens_value = float(total_tokens)

            rows.append(
                {
                    "wandb_run_id": run.id,
                    "source_experiment": infer_source_experiment(run.display_name),
                    "local_run_id": infer_local_run_id(run.display_name),
                    "run_name": run.display_name,
                    "step": int(step),
                    "total_tokens": total_tokens_value,
                    "metric_key": metric_key,
                    "metric_value": float(metric_value),
                    "_scan_index": scan_index,
                }
            )
            scan_index += 1
    return rows


def collect_dataset(config: CollectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect run metadata and trajectory history from W&B."""
    import wandb

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=60)
    regex = rf"^{re.escape(config.display_name_prefix)}"
    filters = {"display_name": {"$regex": regex}, "state": "finished"}
    wb_runs = list(api.runs(f"{config.wandb_entity}/{config.wandb_project}", filters=filters))
    wb_runs.sort(key=lambda run: run.display_name)

    if config.expected_run_count > 0 and len(wb_runs) != config.expected_run_count:
        raise ValueError(f"Expected {config.expected_run_count} runs but found {len(wb_runs)}")

    gcs_prefix = os.environ.get("MARIN_PREFIX", "gs://marin-us-central1")
    map_from_configs = _load_phase_weights_from_weight_configs(
        source_experiments=config.source_experiments,
        gcs_prefix=gcs_prefix,
    )
    map_from_csv = _load_phase_weights_from_csv(config.csv_fallback_path)

    run_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []

    for index, wb_run in enumerate(wb_runs):
        if index % 10 == 0:
            logger.info("Collecting run history %d/%d (%s)", index + 1, len(wb_runs), wb_run.id)
        source_experiment = infer_source_experiment(wb_run.display_name)
        local_run_id = infer_local_run_id(wb_run.display_name)
        phase_weights = {}
        if local_run_id is not None:
            phase_weights = map_from_configs.get((source_experiment, local_run_id), {})
        if not phase_weights:
            phase_weights = map_from_csv.get(wb_run.id, {})

        history_rows.extend(
            collect_history_long_rows(
                wb_run,
                DEFAULT_HISTORY_KEYS,
                history_samples=config.history_samples,
            )
        )

        objective_summary = _extract_numeric_summary(wb_run.summary, config.objective_metric)
        row = {
            "wandb_run_id": wb_run.id,
            "run_name": wb_run.display_name,
            "source_experiment": source_experiment,
            "local_run_id": local_run_id,
            "status": wb_run.state,
            config.objective_metric: objective_summary,
        }
        row.update(phase_weights_to_columns(phase_weights))
        run_rows.append(row)

    runs_df = pd.DataFrame(run_rows)
    history_long_df = dedupe_history_rows(pd.DataFrame(history_rows))
    history_wide_df = build_wide_history(history_long_df)

    runs_df.to_parquet(output_dir / "runs.parquet", index=False)
    history_long_df.to_parquet(output_dir / "history_long.parquet", index=False)
    history_wide_df.to_parquet(output_dir / "history_wide.parquet", index=False)

    metadata = {
        "n_runs": len(runs_df),
        "n_history_rows_long": len(history_long_df),
        "n_history_rows_wide": len(history_wide_df),
        "objective_metric": config.objective_metric,
        "display_name_prefix": config.display_name_prefix,
    }
    with (output_dir / "collection_manifest.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return runs_df, history_long_df, history_wide_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect 3-phase StarCoder trajectories for offline RL.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/domain_phase_mix/offline_rl/artifacts/three_phase_starcoder",
        help="Directory for collected parquet artifacts.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="marin-community",
        help="W&B entity.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="marin",
        help="W&B project.",
    )
    parser.add_argument(
        "--display-name-prefix",
        type=str,
        default=DEFAULT_DISPLAY_NAME_PREFIX,
        help="Display-name prefix regex anchor.",
    )
    parser.add_argument(
        "--objective-metric",
        type=str,
        default=DEFAULT_OBJECTIVE_METRIC,
        help="Objective metric key.",
    )
    parser.add_argument(
        "--expected-run-count",
        type=int,
        default=EXPECTED_RUN_COUNT,
        help="Expected number of finished runs (0 to disable assertion).",
    )
    parser.add_argument(
        "--csv-fallback-path",
        type=str,
        default="experiments/domain_phase_mix/exploratory/three_phase_starcoder.csv",
        help="Fallback CSV for phase weights if weight_configs are unavailable.",
    )
    parser.add_argument(
        "--history-samples",
        type=int,
        default=DEFAULT_HISTORY_SAMPLES,
        help="Maximum history points per run fetched via W&B history API.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    collect_dataset(
        CollectConfig(
            output_dir=args.output_dir,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            display_name_prefix=args.display_name_prefix,
            objective_metric=args.objective_metric,
            expected_run_count=args.expected_run_count,
            csv_fallback_path=args.csv_fallback_path,
            history_samples=args.history_samples,
        )
    )


if __name__ == "__main__":
    main()
