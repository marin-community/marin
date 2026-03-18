# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for data quality scaling experiments."""

import json
import logging
import os
from dataclasses import dataclass

import fsspec

logger = logging.getLogger(__name__)


def read_eval_metrics(output_path: str) -> list[dict]:
    """Read eval_metrics.jsonl from a training run, deduplicating by step."""
    metrics_file = os.path.join(output_path, "checkpoints", "eval_metrics.jsonl")
    records = []
    with fsspec.open(metrics_file, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    seen = {}
    for r in records:
        seen[r["step"]] = r
    return sorted(seen.values(), key=lambda r: r["step"])


def get_final_metrics(output_path: str) -> dict:
    """Get the last eval record from a run."""
    records = read_eval_metrics(output_path)
    if not records:
        return {}
    return records[-1]


@dataclass(frozen=True)
class SelectBestLRConfig:
    """Config for selecting the best learning rate from a set of tuning runs.

    Attributes:
        tuning_run_paths: Output paths of the tuning runs (for reading eval metrics).
        tuning_run_configs: Configs of the tuning runs (for reading the LR).
            Must be parallel to tuning_run_paths.
        output_path: Where to write the selection result.
        metric_key: The eval metric to minimize. Defaults to "eval/high/loss".
    """
    tuning_run_paths: list[str]
    tuning_run_configs: list
    output_path: str
    metric_key: str = "eval/high/loss"


def select_best_lr(config: SelectBestLRConfig):
    """Select the best LR by finding the tuning run with the lowest final metric."""
    best_lr = None
    best_loss = float("inf")

    for run_path, run_config in zip(config.tuning_run_paths, config.tuning_run_configs):
        lr = run_config.train_config.optimizer.learning_rate
        metrics = get_final_metrics(run_path)
        loss = metrics.get(config.metric_key)
        if loss is None:
            logger.warning(f"lr={lr}: no {config.metric_key} found, skipping")
            continue
        logger.info(f"lr={lr}: {config.metric_key} = {loss}")
        if loss < best_loss:
            best_loss = loss
            best_lr = lr

    if best_lr is None:
        raise RuntimeError("No valid tuning runs found")

    logger.info(f"Best LR: {best_lr} ({config.metric_key} = {best_loss})")

    result = {"best_lr": best_lr, "best_loss": best_loss}
    output_file = os.path.join(config.output_path, "best_lr.json")
    with fsspec.open(output_file, "wt") as f:
        json.dump(result, f, indent=2)
