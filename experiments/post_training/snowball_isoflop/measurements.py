# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read bounded training histories and separate warm throughput from startup."""

import json
import math
import statistics
from dataclasses import dataclass

import wandb
from rigging.filesystem.buckets import filesystem_for
from rigging.secrets import get_secret

from experiments.post_training.snowball_isoflop.config import SEQ_LEN, VERSION, training_flops_per_position

H100_BF16_FLOPS = 989e12
WANDB_PROJECT = "nyu-dice-lab/snowball-isoflop"


@dataclass(frozen=True)
class WarmMeasurement:
    warm_steps: tuple[int, ...]
    learner_step_seconds: float
    inclusive_step_seconds: float
    learner_positions_per_second_per_gpu: float
    inclusive_positions_per_second_per_gpu: float
    learner_analytic_mfu_percent: float
    inclusive_analytic_mfu_percent: float
    reported_mfu_percent: float | None
    minimum_memory_headroom_fraction: float | None
    maximum_peak_gib: float | None
    sender_dropped: float | None
    receiver_dropped: float | None


def read_json(path: str) -> dict:
    fs, key = filesystem_for(path)
    with fs.open(key, "r") as stream:
        return json.load(stream)


def training_history(cell_name: str, steps: int) -> tuple[str, list[dict]]:
    run = wandb.Api(timeout=30, api_key=get_secret("WANDB_API_KEY")).run(f"{WANDB_PROJECT}/{VERSION}-{cell_name}")
    rows = list(run.scan_history(page_size=100, min_step=0, max_step=steps))
    return run.state, rows


def warm_measurement(rows: list[dict], size: int, batch_size: int, warmup_updates: int = 3) -> WarmMeasurement:
    """Require five complete warm updates and retain missing diagnostics explicitly."""
    training = [row for row in rows if "train/loss" in row]
    if not training or any(not math.isfinite(float(row["train/loss"])) for row in training):
        raise ValueError("Training history lacks finite losses")
    warm = [row for row in training if int(row["global_step"]) >= warmup_updates]
    required = ("throughput/duration", "throughput/loading_time", "throughput/hook_time", "throughput/gc_time")
    if len(warm) < 5 or any(any(key not in row for key in required) for row in warm):
        raise ValueError("At least five complete warm update records are required")
    durations = [float(row["throughput/duration"]) for row in warm]
    inclusive = [sum(float(row[key]) for key in required) for row in warm]
    if any(not math.isfinite(value) or value <= 0 for value in (*durations, *inclusive)):
        raise ValueError("Invalid warm timing measurements")
    learner = statistics.median(durations)
    elapsed = statistics.median(inclusive)
    positions_per_gpu = batch_size * SEQ_LEN / 8
    flops_per_gpu = positions_per_gpu * training_flops_per_position(size)
    complete_memory = all(
        row.get("memory/stats_devices_measured") == 8 and "memory/min_headroom_fraction" in row for row in warm
    )
    peaks = [
        float(value)
        for row in warm
        for key, value in row.items()
        if key.startswith("memory/gpu") and key.endswith("/peak_gib")
    ]
    reported = [float(row["throughput/mfu"]) for row in warm if "throughput/mfu" in row]
    sender = [float(row["train/moe/sender_dropped"]) for row in training if "train/moe/sender_dropped" in row]
    receiver = [float(row["train/moe/receiver_dropped"]) for row in training if "train/moe/receiver_dropped" in row]
    return WarmMeasurement(
        warm_steps=tuple(int(row["global_step"]) for row in warm),
        learner_step_seconds=learner,
        inclusive_step_seconds=elapsed,
        learner_positions_per_second_per_gpu=positions_per_gpu / learner,
        inclusive_positions_per_second_per_gpu=positions_per_gpu / elapsed,
        learner_analytic_mfu_percent=100 * flops_per_gpu / (learner * H100_BF16_FLOPS),
        inclusive_analytic_mfu_percent=100 * flops_per_gpu / (elapsed * H100_BF16_FLOPS),
        reported_mfu_percent=statistics.median(reported) if reported else None,
        minimum_memory_headroom_fraction=(
            min(float(row["memory/min_headroom_fraction"]) for row in warm) if complete_memory else None
        ),
        maximum_peak_gib=max(peaks) if peaks else None,
        sender_dropped=sum(sender) if len(sender) == len(training) else None,
        receiver_dropped=sum(receiver) if len(receiver) == len(training) else None,
    )
