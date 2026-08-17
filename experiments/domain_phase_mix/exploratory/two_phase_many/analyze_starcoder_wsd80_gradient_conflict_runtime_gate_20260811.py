# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "wandb>=0.21",
# ]
# ///

"""Evaluate WSD80 gradient-panel runtime gates without reading endpoint losses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import re
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import wandb

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_stress as stress

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
LONG_GATE_DESIGN_DIR = stress.full.DESIGN_DIR
LONG_GATE_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/starcoder_wsd80_gradient_conflict_long_gate_results_20260811"
LONG_GATE_IRIS_JOB = "/calvinxu/dm-starcoder-wsd80-gradient-conflict-long-gate-20260811"
LONG_GATE_TRAJECTORIES = (
    ("gcf_p3_r3d28260_m100a_confirmed-two-phase-winner_s2026082000", "m100a"),
    ("gcf_p3_r3d28260_full_confirmed-two-phase-winner_s2026082000", "full"),
    ("gcf_p1_r3d28260_m100b_common-tied-035_s2026081000", "m100b"),
)
LONG_GATE_WANDB_PREFIX = "gcfv8r3_"
LONG_GATE_TOTAL_STEPS = 28_260
LONG_GATE_TERMINAL_STEP = 28_259
LONG_GATE_BOUNDARY_STEP = 22_608
LONG_GATE_CHECKPOINT_ROOT = (
    "gs://marin-us-central1/checkpoints/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_gradient_conflict_20260811_retry3/trajectories"
)
LONG_GATE_VERSION = "2026.08.11.3"
STEADY_STATE_START_STEP = 64
LOADING_P99_MAX = 0.010
LOADING_OVER_ONE_SECOND_FRACTION_MAX = 8 / (LONG_GATE_TERMINAL_STEP - STEADY_STATE_START_STEP + 1)
LOADING_MAX = 60.0
DUTY_FRACTION_MIN = 0.99
TOKENS_PER_SECOND_P50_MIN = 690_000.0
MFU_P50_MIN = 33.0
ROLLING_WINDOW_STEPS = 1_000
ROLLING_TOKENS_PER_SECOND_P50_MIN = 660_000.0
EVENT_TOKENS_PER_SECOND_P50_MIN = 660_000.0
EVENT_LOADING_P99_MAX = 0.025
EVENT_RECOVERY_FRACTION_MIN = 0.97
STRESS_EVENT_SLOWDOWN_ROLLING_WINDOW_STEPS = 32
STRESS_EVENT_SLOWDOWN_FRACTION = 0.95
STRESS_SHORT_WINDOW_TOKENS_PER_SECOND_MIN = 660_000.0
STRESS_SHORT_WINDOW_MAX_POSITIONS = 96
STRESS_SHORT_WINDOW_EXCEEDANCE_MAX_ROWS = 1
STRESS_SYNCHRONIZED_QUORUM_FRACTION = 5 / 6
STRESS_SYNCHRONIZED_DEPRESSION_FRACTION = 0.95
STRESS_SYNCHRONIZED_BIN_SECONDS = 12.0
STRESS_SYNCHRONIZED_MAX_SECONDS = 36.0
PAUSE_ACCOUNTING_SLACK_SECONDS = 1.0
UNEXPLAINED_PAUSE_LONGEST_SECONDS_MAX = 5.0
UNEXPLAINED_PAUSE_TOTAL_SECONDS_MAX = 15.0
CHECKPOINT_PAUSE_SECONDS_MAX = 60.0
CHECKPOINT_PAUSE_CROSS_ROW_SPREAD_SECONDS_MAX = 30.0
HISTORY_COVERAGE_FRACTION_MIN = 0.95
EVENT_HISTORY_COVERAGE_FRACTION_MIN = 0.95
FINITE_TO_FULL_LOADING_P99_RATIO_MAX = 1.5
CONCURRENT_OVERLAP_SECONDS_MIN = 120.0
CONCURRENT_OVERLAP_FRACTION_MIN = 0.95
RUNTIME_START_SKEW_SECONDS_MAX = 180.0
RENDEZVOUS_READY_SPREAD_SECONDS_MAX = 450.0
COMPLETION_RENDEZVOUS_READY_SPREAD_SECONDS_MAX = 450.0
RENDEZVOUS_RUNTIME_START_EARLY_TOLERANCE_SECONDS = 60.0
RUNTIME_ACCOUNTING_RATIO_MIN = 0.95
RUNTIME_ACCOUNTING_RATIO_MAX = 1.25
PERMANENT_CHECKPOINT_BYTES_MAX = 2_350_000_000
C12_ASSIGNMENT_ALPHA = 0.01
C12_ASSIGNMENT_WINDOW_STEPS = 64
C12_LEAD_PLACEBO_OFFSET_STEPS = 256
HISTORY_KEYS = (
    "_step",
    "_runtime",
    "_timestamp",
    "global_step",
    "throughput/loading_time",
    "throughput/duration",
    "throughput/tokens_per_second",
    "throughput/mfu",
)


@dataclass(frozen=True)
class RunSpec:
    """Expected identity and state transitions for one runtime-gate row."""

    trajectory_id: str
    wandb_run_id: str
    support_id: str
    terminal_step: int
    checkpoint_root: str
    expected_checkpoint_steps: tuple[int, ...]
    event_steps: tuple[tuple[str, int], ...]
    event_window_steps: int
    total_steps: int
    data_switch_step: int
    optimizer_decay_step: int
    phase_0_starcoder: float
    phase_1_starcoder: float
    training_seed: int
    support_batches: int | None
    support_start_batches: int | None
    support_pool_seed: int | None
    train_holdout_sequences_per_component: int
    train_holdout_seed: int
    train_holdout_partition: str


@dataclass(frozen=True)
class SlowdownDiagnostic:
    """Coverage and duration of one rolling-median slowdown scan."""

    longest_positions: int
    analyzable_positions: int
    possible_positions: int

    @property
    def analyzable_fraction(self) -> float:
        return self.analyzable_positions / self.possible_positions


@dataclass(frozen=True)
class RollingThroughputDiagnostic:
    """Gap-safe rolling throughput summary over one step interval."""

    minimum_median: float
    minimum_center_step: float
    longest_below_threshold_positions: int
    analyzable_positions: int
    possible_positions: int

    @property
    def analyzable_fraction(self) -> float:
        return self.analyzable_positions / self.possible_positions


@dataclass(frozen=True)
class PauseInterval:
    """A host-time interval not accounted for by one recorded training step."""

    right_step: int
    start: float
    stop: float
    seconds: float
    classification: str


def _percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        raise ValueError("Cannot compute a percentile from an empty metric series")
    return float(np.percentile(values, percentile))


def _utc_epoch(value: str, *, field: str) -> float:
    """Parse an explicitly timezone-qualified ISO timestamp."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include an explicit UTC offset")
    return parsed.timestamp()


def _wandb_history(run: Any) -> dict[str, np.ndarray]:
    rows: list[dict[str, Any]] = []
    rows_by_step: dict[int, dict[str, Any]] = {}
    previous_step: int | None = None
    metric_keys = HISTORY_KEYS[4:]
    for row in run.scan_history(keys=list(HISTORY_KEYS), page_size=10_000):
        if (
            row.get("global_step") is None
            or row.get("_runtime") is None
            or any(row.get(key) is None for key in metric_keys)
        ):
            continue
        step = int(row["global_step"])
        if step in rows_by_step:
            if all(row.get(key) == rows_by_step[step].get(key) for key in HISTORY_KEYS):
                continue
            raise ValueError(f"Conflicting duplicate global_step {step} in throughput history for {run.id}")
        if previous_step is not None and step <= previous_step:
            raise ValueError(f"Non-monotonic throughput history for {run.id}: {previous_step} -> {step}")
        rows.append(row)
        rows_by_step[step] = row
        previous_step = step
    if not rows:
        raise ValueError(f"No training-throughput history for {run.id}")
    created_epoch = _utc_epoch(str(run.created_at), field=f"W&B created_at for {run.id}")
    runtime = np.array([float(row["_runtime"]) for row in rows])
    if not np.all(np.isfinite(runtime)):
        raise ValueError(f"Non-finite W&B runtime history for {run.id}")
    if runtime.size > 1 and not np.all(np.diff(runtime) > 0):
        raise ValueError(f"Non-monotonic W&B runtime history for {run.id}")
    host_timestamp = np.array([float(row.get("_timestamp", math.nan)) for row in rows])
    if not np.all(np.isfinite(host_timestamp)):
        raise ValueError(f"Non-finite W&B host timestamps for {run.id}")
    if host_timestamp.size > 1 and not np.all(np.diff(host_timestamp) >= 0):
        raise ValueError(f"Non-monotonic W&B host timestamps for {run.id}")
    series: dict[str, np.ndarray] = {
        "global_step": np.array([int(row["global_step"]) for row in rows]),
        "runtime": runtime,
        "timestamp": created_epoch + runtime,
        "host_timestamp": host_timestamp,
    }
    for key in metric_keys:
        series[key] = np.array([float(row[key]) for row in rows])
    return series


def _window(series: dict[str, np.ndarray], start: int, stop: int) -> dict[str, np.ndarray]:
    selected = (series["global_step"] >= start) & (series["global_step"] < stop)
    return {key: values[selected] for key, values in series.items()}


def _pause_intervals(
    series: dict[str, np.ndarray],
    *,
    expected_checkpoint_steps: tuple[int, ...],
) -> tuple[PauseInterval, ...]:
    """Classify host-time gaps using only declared checkpoints and recorded step accounting."""
    pauses: list[PauseInterval] = []
    checkpoint_steps = set(expected_checkpoint_steps)
    for index in range(1, len(series["global_step"])):
        observed_seconds = float(series["host_timestamp"][index] - series["host_timestamp"][index - 1])
        accounted_seconds = float(series["throughput/duration"][index] + series["throughput/loading_time"][index])
        pause_seconds = observed_seconds - accounted_seconds
        if pause_seconds <= PAUSE_ACCOUNTING_SLACK_SECONDS:
            continue
        right_step = int(series["global_step"][index])
        stop = float(series["host_timestamp"][index] - accounted_seconds)
        start = float(series["host_timestamp"][index - 1])
        pauses.append(
            PauseInterval(
                right_step=right_step,
                start=start,
                stop=stop,
                seconds=pause_seconds,
                classification="checkpoint" if right_step in checkpoint_steps else "unexplained",
            )
        )
    return tuple(pauses)


def _pause_summary(
    pauses: tuple[PauseInterval, ...],
    *,
    expected_checkpoint_steps: tuple[int, ...],
) -> tuple[dict[str, Any], list[str]]:
    """Apply fixed per-row pause bounds and retain every classified interval."""
    checkpoint_seconds = {
        str(step): sum(
            pause.seconds for pause in pauses if pause.classification == "checkpoint" and pause.right_step == step
        )
        for step in expected_checkpoint_steps
    }
    unexplained = tuple(pause for pause in pauses if pause.classification == "unexplained")
    unexplained_total = sum(pause.seconds for pause in unexplained)
    unexplained_longest = max((pause.seconds for pause in unexplained), default=0.0)
    checkpoint_longest = max(checkpoint_seconds.values(), default=0.0)
    failures: list[str] = []
    if unexplained_longest > UNEXPLAINED_PAUSE_LONGEST_SECONDS_MAX:
        failures.append(
            f"longest unexplained pause {unexplained_longest:.3f}s > " f"{UNEXPLAINED_PAUSE_LONGEST_SECONDS_MAX:.3f}s"
        )
    if unexplained_total > UNEXPLAINED_PAUSE_TOTAL_SECONDS_MAX:
        failures.append(f"total unexplained pause {unexplained_total:.3f}s > {UNEXPLAINED_PAUSE_TOTAL_SECONDS_MAX:.3f}s")
    if checkpoint_longest > CHECKPOINT_PAUSE_SECONDS_MAX:
        failures.append(f"checkpoint pause {checkpoint_longest:.3f}s > {CHECKPOINT_PAUSE_SECONDS_MAX:.3f}s")
    return (
        {
            "accounting_slack_seconds": PAUSE_ACCOUNTING_SLACK_SECONDS,
            "checkpoint_seconds_by_step": checkpoint_seconds,
            "checkpoint_longest_seconds": checkpoint_longest,
            "unexplained_count": len(unexplained),
            "unexplained_total_seconds": unexplained_total,
            "unexplained_longest_seconds": unexplained_longest,
            "intervals": [
                {
                    "right_step": pause.right_step,
                    "start": pause.start,
                    "stop": pause.stop,
                    "seconds": pause.seconds,
                    "classification": pause.classification,
                }
                for pause in pauses
            ],
            "status": "pass" if not failures else "fail",
        },
        failures,
    )


def _minimum_rolling_median(values: np.ndarray, window: int) -> float:
    if values.size < window:
        raise ValueError(f"Need at least {window} throughput samples, got {values.size}")
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    return float(np.min(np.median(windows, axis=1)))


def _longest_true_run(values: np.ndarray) -> int:
    """Return the longest consecutive run of true values."""
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _rolling_throughput_diagnostic(
    series: dict[str, np.ndarray],
    *,
    start_step: int,
    stop_step: int,
    window_steps: int,
    threshold: float,
) -> RollingThroughputDiagnostic:
    """Scan rolling throughput without allowing windows to bridge history gaps."""
    selected = _window(series, start_step, stop_step)
    steps = selected["global_step"]
    tokens_per_second = selected["throughput/tokens_per_second"]
    minimum_median = math.inf
    minimum_center_step = math.nan
    longest = 0
    analyzable_positions = 0
    segment_start = 0
    discontinuities = np.flatnonzero(np.diff(steps) != 1)
    for segment_stop in (*tuple(int(index + 1) for index in discontinuities), len(steps)):
        segment_steps = steps[segment_start:segment_stop]
        segment_tokens = tokens_per_second[segment_start:segment_stop]
        if len(segment_tokens) >= window_steps:
            windows = np.lib.stride_tricks.sliding_window_view(segment_tokens, window_steps)
            medians = np.median(windows, axis=1)
            centers = (segment_steps[: len(medians)] + segment_steps[window_steps - 1 :]) / 2
            minimum_index = int(np.argmin(medians))
            if float(medians[minimum_index]) < minimum_median:
                minimum_median = float(medians[minimum_index])
                minimum_center_step = float(centers[minimum_index])
            below_threshold = medians < threshold
            analyzable_positions += len(below_threshold)
            longest = max(longest, _longest_true_run(below_threshold))
        segment_start = segment_stop
    possible_positions = stop_step - start_step - window_steps + 1
    if possible_positions <= 0:
        raise ValueError("Rolling throughput interval is shorter than its window")
    if not math.isfinite(minimum_median):
        raise ValueError("No contiguous rolling throughput window is available")
    return RollingThroughputDiagnostic(
        minimum_median=minimum_median,
        minimum_center_step=minimum_center_step,
        longest_below_threshold_positions=longest,
        analyzable_positions=analyzable_positions,
        possible_positions=possible_positions,
    )


def _post_event_slowdown_positions(
    series: dict[str, np.ndarray],
    *,
    event_step: int,
    stop_step: int,
    pre_tokens_per_second_p50: float,
) -> SlowdownDiagnostic:
    """Measure rolling-median slowdown positions without crossing history gaps."""
    diagnostic = _rolling_throughput_diagnostic(
        series,
        start_step=event_step,
        stop_step=stop_step,
        window_steps=STRESS_EVENT_SLOWDOWN_ROLLING_WINDOW_STEPS,
        threshold=STRESS_EVENT_SLOWDOWN_FRACTION * pre_tokens_per_second_p50,
    )
    return SlowdownDiagnostic(
        longest_positions=diagnostic.longest_below_threshold_positions,
        analyzable_positions=diagnostic.analyzable_positions,
        possible_positions=diagnostic.possible_positions,
    )


def _checkpoint_steps(root: str) -> tuple[int, ...]:
    completed = subprocess.run(
        ["gcloud", "storage", "ls", f"{root}/"],
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = f"{root}/step-"
    return tuple(
        sorted(
            int(line.removeprefix(prefix).rstrip("/"))
            for line in completed.stdout.splitlines()
            if line.startswith(prefix)
        )
    )


def _gcloud_size(path: str) -> int:
    completed = subprocess.run(
        ["gcloud", "storage", "du", "--summarize", path],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.split()[0])


def _parse_iris_summary(job: str, output: str) -> dict[str, Any]:
    state = re.search(r"^State: (\w+)\s+exit=(\S+)\s+failures=(\d+)\s+preemptions=(\d+)$", output, re.M)
    tasks = re.search(r"^Tasks: (\d+)/(\d+) completed(?:\s+.*)?$", output, re.M)
    if state is None or tasks is None:
        raise ValueError(f"Could not parse Iris summary for {job}")
    running = re.search(r"\brunning=(\d+)\b", tasks.group(0))
    return {
        "job": job,
        "state": state.group(1),
        "exit": state.group(2),
        "failures": int(state.group(3)),
        "preemptions": int(state.group(4)),
        "completed_tasks": int(tasks.group(1)),
        "total_tasks": int(tasks.group(2)),
        "running_tasks": 0 if running is None else int(running.group(1)),
    }


def _iris_summary(job: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["uv", "run", "iris", "--config", "lib/iris/config/marin.yaml", "job", "summary", job],
        check=True,
        capture_output=True,
        text=True,
    )
    return _parse_iris_summary(job, completed.stdout)


def _iris_child_summaries(job: str) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "uv",
            "run",
            "iris",
            "--config",
            "lib/iris/config/marin.yaml",
            "job",
            "list",
            "--prefix",
            f"{job}/",
            "--limit",
            "1000",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    child_jobs = tuple(line.split(maxsplit=1)[0] for line in completed.stdout.splitlines() if line.startswith(f"{job}/"))
    return [_iris_summary(child_job) for child_job in child_jobs]


def _long_gate_specs() -> tuple[RunSpec, ...]:
    if _sha256(stress.full.DESIGN_MANIFEST) != stress.full.EXPECTED_DESIGN_MANIFEST_SHA256:
        raise ValueError("Review-v9 design manifest drifted before long-gate analysis")
    with (LONG_GATE_DESIGN_DIR / "checkpoint_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (LONG_GATE_DESIGN_DIR / "trajectory_manifest.csv").open(newline="") as handle:
        trajectories_by_id = {row["trajectory_id"]: row for row in csv.DictReader(handle)}
    steps_by_id: dict[str, list[int]] = {}
    for row in rows:
        steps_by_id.setdefault(row["trajectory_id"], []).append(int(row["checkpoint_step"]))
    specs: list[RunSpec] = []
    for trajectory_id, support_id in LONG_GATE_TRAJECTORIES:
        trajectory = trajectories_by_id[trajectory_id]
        specs.append(
            RunSpec(
                trajectory_id=trajectory_id,
                wandb_run_id=f"{LONG_GATE_WANDB_PREFIX}{trajectory_id}",
                support_id=support_id,
                terminal_step=LONG_GATE_TERMINAL_STEP,
                checkpoint_root=f"{LONG_GATE_CHECKPOINT_ROOT}/{trajectory_id}/{LONG_GATE_VERSION}/checkpoints",
                expected_checkpoint_steps=tuple(steps_by_id[trajectory_id]),
                event_steps=(("phase_and_decay", LONG_GATE_BOUNDARY_STEP),),
                event_window_steps=1_000,
                total_steps=int(trajectory["total_steps"]),
                data_switch_step=int(trajectory["boundary_step"]),
                optimizer_decay_step=int(trajectory["optimizer_decay_step"]),
                phase_0_starcoder=float(trajectory["phase_0_starcoder"]),
                phase_1_starcoder=float(trajectory["phase_1_starcoder"]),
                training_seed=int(trajectory["training_seed"]),
                support_batches=int(trajectory["support_batches"]) if trajectory["support_batches"] else None,
                support_start_batches=(
                    int(trajectory["support_start_batches"]) if trajectory["support_start_batches"] else None
                ),
                support_pool_seed=int(trajectory["support_pool_seed"]) if trajectory["support_pool_seed"] else None,
                train_holdout_sequences_per_component=int(trajectory["train_holdout_sequences_per_component"]),
                train_holdout_seed=int(trajectory["train_holdout_seed"]),
                train_holdout_partition=trajectory["train_holdout_partition"],
            )
        )
    return tuple(specs)


def _stress_specs(
    stage: int,
    generation: int = stress.DEFAULT_GENERATION,
    cohort_attempt: int | None = None,
) -> tuple[RunSpec, ...]:
    return tuple(
        RunSpec(
            trajectory_id=row.trajectory_id,
            wandb_run_id=stress.wandb_run_id(row, generation, cohort_attempt),
            support_id=row.support_id,
            terminal_step=stress.TERMINAL_STEP,
            checkpoint_root=(
                "gs://marin-us-central1/tmp/ttl=14d/checkpoints/"
                f"{stress.namespace_name(generation)}/stage-c{stage:02d}/{row.trajectory_id}/"
                f"{stress.namespace_version(generation)}"
                f"{'/attempt-' + format(cohort_attempt, '03d') if cohort_attempt is not None else ''}/checkpoints"
            ),
            expected_checkpoint_steps=(stress.DATA_SWITCH_STEP, stress.TERMINAL_STEP),
            event_steps=(("data_switch", row.boundary_step), ("optimizer_decay", row.optimizer_decay_step)),
            event_window_steps=128,
            total_steps=row.total_steps,
            data_switch_step=row.boundary_step,
            optimizer_decay_step=row.optimizer_decay_step,
            phase_0_starcoder=row.phase_0_starcoder,
            phase_1_starcoder=row.phase_1_starcoder,
            training_seed=row.training_seed,
            support_batches=row.support_batches,
            support_start_batches=row.support_start_batches,
            support_pool_seed=row.support_pool_seed,
            train_holdout_sequences_per_component=row.train_holdout_sequences_per_component,
            train_holdout_seed=row.train_holdout_seed,
            train_holdout_partition=row.train_holdout_partition,
        )
        for row in stress.rows_for_stage(stage)
    )


def _validate_run_config(run: Any, spec: RunSpec) -> list[str]:
    """Check that W&B recorded the runtime schedule attributed to this row."""
    failures: list[str] = []
    config = run.config
    trainer = config.get("trainer", {})
    data = config.get("data", {})
    optimizer = config.get("optimizer", {})
    expected_cap = None if spec.support_batches is None else {"dolma/starcoder": spec.support_batches}
    expected_start = None if spec.support_start_batches is None else {"dolma/starcoder": spec.support_start_batches}
    expected_holdout = {
        name: spec.train_holdout_sequences_per_component for name in stress.full.EXPECTED_TRAINING_COMPONENT_NAMES
    }
    checks = (
        (trainer.get("num_train_steps") == spec.total_steps, "W&B num_train_steps drifted"),
        (trainer.get("seed") == spec.training_seed, "W&B trainer seed drifted"),
        (
            trainer.get("distributed", {}).get("initialize_jax_distributed") is False,
            "W&B cross-replica JAX initialization was not disabled",
        ),
        (config.get("data_seed") == spec.training_seed, "W&B data seed drifted"),
        (
            config.get("optimizer_schedule_num_train_steps") is None,
            "W&B optimizer horizon override unexpectedly set",
        ),
        (
            optimizer.get("decay") == spec.total_steps - spec.optimizer_decay_step,
            "W&B optimizer decay onset drifted",
        ),
        (data.get("max_train_batches") == expected_cap, "W&B finite-support cap drifted"),
        (data.get("max_train_batches_start") == expected_start, "W&B finite-support offset drifted"),
        (
            data.get("max_train_batches_subset_seed") == spec.support_pool_seed,
            "W&B support-pool seed drifted",
        ),
        (data.get("train_holdout_sequences") == expected_holdout, "W&B holdout counts drifted"),
        (data.get("train_holdout_seed") == spec.train_holdout_seed, "W&B holdout seed drifted"),
        (
            data.get("train_holdout_partition") == spec.train_holdout_partition,
            "W&B holdout partition drifted",
        ),
        (data.get("permutation_type") == "feistel", "W&B permutation type drifted"),
        (data.get("mixture_block_size") == stress.base.MIXTURE_BLOCK_SIZE, "W&B mixture block size drifted"),
        (data.get("experiment_budget") is None, "W&B experiment budget unexpectedly set"),
        (data.get("target_budget") is None, "W&B target budget unexpectedly set"),
        (data.get("simulated_epoch_subset_seed") is None, "W&B simulated subset unexpectedly set"),
    )
    failures.extend(message for passed, message in checks if not passed)
    phase_weights = data.get("train_weights")
    if not isinstance(phase_weights, list) or len(phase_weights) != 2:
        failures.append("W&B phase schedule is not a two-entry list")
        return failures
    if [entry[0] for entry in phase_weights] != [0, spec.data_switch_step]:
        failures.append("W&B data-switch boundary drifted")
    total_nemotron_tokens = sum(stress.base.NEMOTRON_TOKEN_COUNTS.values())
    for phase_index, expected_weight in enumerate((spec.phase_0_starcoder, spec.phase_1_starcoder)):
        weights = phase_weights[phase_index][1]
        if set(weights) != set(stress.full.EXPECTED_TRAINING_COMPONENT_NAMES):
            failures.append(f"W&B phase-{phase_index} component identities drifted")
            continue
        expected_weights = {
            f"nemotron_cc/{split}-llama3": (1.0 - expected_weight) * count / total_nemotron_tokens
            for split, count in stress.base.NEMOTRON_TOKEN_COUNTS.items()
        }
        expected_weights["dolma/starcoder"] = expected_weight
        for component, component_weight in expected_weights.items():
            if not math.isclose(float(weights[component]), component_weight, rel_tol=0.0, abs_tol=1e-12):
                failures.append(f"W&B phase-{phase_index} weight drifted for {component}")
    return failures


def _evaluate_run(
    spec: RunSpec,
    api: wandb.Api,
    *,
    enforce_per_row_event_recovery: bool = True,
    enforce_pause_gates: bool = False,
) -> dict[str, Any]:
    run = api.run(f"marin-community/marin/{spec.wandb_run_id}")
    failures: list[str] = []
    if run.state != "finished":
        failures.append(f"W&B state is {run.state!r}, not 'finished'")
    if int(run.summary.get("global_step", -1)) != spec.terminal_step:
        failures.append(f"terminal global_step is {run.summary.get('global_step')}, expected {spec.terminal_step}")
    failures.extend(_validate_run_config(run, spec))
    series = _wandb_history(run)
    pauses = _pause_intervals(series, expected_checkpoint_steps=spec.expected_checkpoint_steps)
    pause_summary, pause_failures = _pause_summary(
        pauses,
        expected_checkpoint_steps=spec.expected_checkpoint_steps,
    )
    if enforce_pause_gates:
        failures.extend(pause_failures)
    runtime_span = float(series["runtime"][-1] - series["runtime"][0])
    accounted_runtime = float(np.sum(series["throughput/duration"][1:] + series["throughput/loading_time"][1:]))
    if accounted_runtime <= 0:
        raise ValueError(f"Accounted step runtime is not positive for {run.id}")
    runtime_accounting_ratio = runtime_span / accounted_runtime
    if not RUNTIME_ACCOUNTING_RATIO_MIN <= runtime_accounting_ratio <= RUNTIME_ACCOUNTING_RATIO_MAX:
        failures.append(
            f"W&B runtime/accounted-step ratio {runtime_accounting_ratio:.6f} is outside "
            f"[{RUNTIME_ACCOUNTING_RATIO_MIN:.6f}, {RUNTIME_ACCOUNTING_RATIO_MAX:.6f}]"
        )
    steady = _window(series, STEADY_STATE_START_STEP, spec.terminal_step + 1)
    loading = steady["throughput/loading_time"]
    duration = steady["throughput/duration"]
    tokens_per_second = steady["throughput/tokens_per_second"]
    mfu = steady["throughput/mfu"]
    loading_p99 = _percentile(loading, 99)
    loading_over_one = int(np.sum(loading > 1.0))
    loading_max = float(np.max(loading))
    duty_fraction = float(np.sum(duration) / (np.sum(duration) + np.sum(loading)))
    tokens_p50 = _percentile(tokens_per_second, 50)
    mfu_p50 = _percentile(mfu, 50)
    rolling_tokens_p50_min = _minimum_rolling_median(tokens_per_second, ROLLING_WINDOW_STEPS)
    short_window = _rolling_throughput_diagnostic(
        series,
        start_step=STEADY_STATE_START_STEP,
        stop_step=spec.terminal_step + 1,
        window_steps=STRESS_EVENT_SLOWDOWN_ROLLING_WINDOW_STEPS,
        threshold=STRESS_SHORT_WINDOW_TOKENS_PER_SECOND_MIN,
    )
    causal_window = _rolling_throughput_diagnostic(
        series,
        start_step=spec.data_switch_step + spec.event_window_steps,
        stop_step=spec.terminal_step + 1,
        window_steps=STRESS_EVENT_SLOWDOWN_ROLLING_WINDOW_STEPS,
        threshold=STRESS_SHORT_WINDOW_TOKENS_PER_SECOND_MIN,
    )
    expected_history_samples = spec.terminal_step - STEADY_STATE_START_STEP + 1
    history_coverage_fraction = len(steady["global_step"]) / expected_history_samples
    loading_over_one_max = math.ceil(LOADING_OVER_ONE_SECOND_FRACTION_MAX * expected_history_samples - 1e-12)
    checks = (
        (loading_p99 <= LOADING_P99_MAX, f"loading p99 {loading_p99:.6f}s > {LOADING_P99_MAX:.6f}s"),
        (
            loading_over_one <= loading_over_one_max,
            f"{loading_over_one} loading events >1s exceeds per-horizon limit {loading_over_one_max}",
        ),
        (loading_max <= LOADING_MAX, f"loading max {loading_max:.3f}s > {LOADING_MAX:.3f}s"),
        (duty_fraction >= DUTY_FRACTION_MIN, f"duty {duty_fraction:.6f} < {DUTY_FRACTION_MIN:.6f}"),
        (tokens_p50 >= TOKENS_PER_SECOND_P50_MIN, f"throughput p50 {tokens_p50:.0f} < {TOKENS_PER_SECOND_P50_MIN:.0f}"),
        (mfu_p50 >= MFU_P50_MIN, f"MFU p50 {mfu_p50:.3f} < {MFU_P50_MIN:.3f}"),
        (
            rolling_tokens_p50_min >= ROLLING_TOKENS_PER_SECOND_P50_MIN,
            f"rolling throughput p50 {rolling_tokens_p50_min:.0f} < {ROLLING_TOKENS_PER_SECOND_P50_MIN:.0f}",
        ),
        (
            history_coverage_fraction >= HISTORY_COVERAGE_FRACTION_MIN,
            f"runtime history coverage {history_coverage_fraction:.6f} < {HISTORY_COVERAGE_FRACTION_MIN:.6f}",
        ),
    )
    failures.extend(message for passed, message in checks if not passed)

    events: list[dict[str, Any]] = []
    for event_index, (label, event_step) in enumerate(spec.event_steps):
        pre = _window(series, event_step - spec.event_window_steps, event_step)
        post = _window(series, event_step, event_step + spec.event_window_steps)
        pre_tokens_p50 = _percentile(pre["throughput/tokens_per_second"], 50)
        post_tokens_p50 = _percentile(post["throughput/tokens_per_second"], 50)
        pre_duration_p50 = _percentile(pre["throughput/duration"], 50)
        post_duration_p50 = _percentile(post["throughput/duration"], 50)
        post_loading_p99 = _percentile(post["throughput/loading_time"], 99)
        recovery_fraction = post_tokens_p50 / pre_tokens_p50
        next_event_step = (
            spec.event_steps[event_index + 1][1] if event_index + 1 < len(spec.event_steps) else spec.terminal_step + 1
        )
        post_event_slowdown = _post_event_slowdown_positions(
            series,
            event_step=event_step,
            stop_step=next_event_step,
            pre_tokens_per_second_p50=pre_tokens_p50,
        )
        pre_coverage_fraction = len(pre["global_step"]) / spec.event_window_steps
        post_coverage_fraction = len(post["global_step"]) / spec.event_window_steps
        event_failures: list[str] = []
        if post_tokens_p50 < EVENT_TOKENS_PER_SECOND_P50_MIN:
            event_failures.append(f"post-event throughput p50 {post_tokens_p50:.0f} is below gate")
        if post_loading_p99 > EVENT_LOADING_P99_MAX:
            event_failures.append(f"post-event loading p99 {post_loading_p99:.6f}s is above gate")
        if enforce_per_row_event_recovery and recovery_fraction < EVENT_RECOVERY_FRACTION_MIN:
            event_failures.append(f"throughput recovery {recovery_fraction:.6f} is below gate")
        if pre_coverage_fraction < EVENT_HISTORY_COVERAGE_FRACTION_MIN:
            event_failures.append(f"pre-event history coverage {pre_coverage_fraction:.6f} is below gate")
        if post_coverage_fraction < EVENT_HISTORY_COVERAGE_FRACTION_MIN:
            event_failures.append(f"post-event history coverage {post_coverage_fraction:.6f} is below gate")
        failures.extend(f"{label}: {message}" for message in event_failures)
        events.append(
            {
                "event": label,
                "step": event_step,
                "window_steps": spec.event_window_steps,
                "pre_tokens_per_second_p50": pre_tokens_p50,
                "post_tokens_per_second_p50": post_tokens_p50,
                "pre_duration_p50_seconds": pre_duration_p50,
                "post_duration_p50_seconds": post_duration_p50,
                "post_loading_p99_seconds": post_loading_p99,
                "throughput_recovery_fraction": recovery_fraction,
                "post_event_slowdown_rolling_window_steps": STRESS_EVENT_SLOWDOWN_ROLLING_WINDOW_STEPS,
                "post_event_slowdown_threshold_fraction": STRESS_EVENT_SLOWDOWN_FRACTION,
                "post_event_longest_slowdown_positions": post_event_slowdown.longest_positions,
                "post_event_slowdown_analyzable_positions": post_event_slowdown.analyzable_positions,
                "post_event_slowdown_possible_positions": post_event_slowdown.possible_positions,
                "post_event_slowdown_analyzable_fraction": post_event_slowdown.analyzable_fraction,
                "pre_history_coverage_fraction": pre_coverage_fraction,
                "post_history_coverage_fraction": post_coverage_fraction,
                "pre_wallclock_start": float(pre["host_timestamp"][0]),
                "pre_wallclock_end": float(pre["host_timestamp"][-1]),
                "post_wallclock_start": float(post["host_timestamp"][0]),
                "post_wallclock_end": float(post["host_timestamp"][-1]),
                "failures": event_failures,
            }
        )

    observed_checkpoint_steps = _checkpoint_steps(spec.checkpoint_root)
    if observed_checkpoint_steps != spec.expected_checkpoint_steps:
        failures.append(f"checkpoint steps {observed_checkpoint_steps} do not match {spec.expected_checkpoint_steps}")
    checkpoint_sizes = {
        str(step): _gcloud_size(f"{spec.checkpoint_root}/step-{step}") for step in observed_checkpoint_steps
    }
    if checkpoint_sizes and max(checkpoint_sizes.values()) > PERMANENT_CHECKPOINT_BYTES_MAX:
        failures.append(f"checkpoint size {max(checkpoint_sizes.values())} exceeds gate")

    return {
        "trajectory_id": spec.trajectory_id,
        "wandb_run_id": spec.wandb_run_id,
        "wandb_url": run.url,
        "support_id": spec.support_id,
        "wandb_created_at": str(run.created_at),
        "first_runtime_timestamp": float(series["timestamp"][0]),
        "last_runtime_timestamp": float(series["timestamp"][-1]),
        "first_host_timestamp": float(series["host_timestamp"][0]),
        "last_host_timestamp": float(series["host_timestamp"][-1]),
        "runtime_span_seconds": runtime_span,
        "accounted_step_seconds": accounted_runtime,
        "runtime_accounting_ratio": runtime_accounting_ratio,
        "history_samples": len(steady["global_step"]),
        "history_coverage_fraction": history_coverage_fraction,
        "loading_p99_seconds": loading_p99,
        "loading_over_one_second_count": loading_over_one,
        "loading_over_one_second_count_max": loading_over_one_max,
        "loading_max_seconds": loading_max,
        "duty_fraction": duty_fraction,
        "tokens_per_second_p50": tokens_p50,
        "mfu_percent_p50": mfu_p50,
        "rolling_tokens_per_second_p50_min": rolling_tokens_p50_min,
        "short_window_tokens_per_second_min": short_window.minimum_median,
        "short_window_minimum_center_step": short_window.minimum_center_step,
        "short_window_longest_below_floor_positions": short_window.longest_below_threshold_positions,
        "short_window_analyzable_positions": short_window.analyzable_positions,
        "short_window_possible_positions": short_window.possible_positions,
        "short_window_analyzable_fraction": short_window.analyzable_fraction,
        "causal_window_tokens_per_second_min": causal_window.minimum_median,
        "causal_window_minimum_center_step": causal_window.minimum_center_step,
        "optimizer_decay_step": spec.optimizer_decay_step,
        "events": events,
        "_runtime_series": series,
        "checkpoint_steps": list(observed_checkpoint_steps),
        "checkpoint_physical_bytes": checkpoint_sizes,
        "pause_accounting": pause_summary,
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }


def _iris_failures(iris: dict[str, Any], *, allow_preemptions: bool = False) -> list[str]:
    """Return release-blocking Iris terminal-state failures."""
    if (
        iris["state"] == "succeeded"
        and iris["exit"] == "0"
        and iris["failures"] == 0
        and (allow_preemptions or iris["preemptions"] == 0)
        and iris["completed_tasks"] == iris["total_tasks"]
        and iris["running_tasks"] == 0
    ):
        return []
    return [f"Iris job did not finish cleanly: {iris}"]


def _stress_cohort_child_evidence(
    children: list[dict[str, Any]],
    *,
    stage: int,
    generation: int,
    final_attempt: int,
    replicas: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate abandoned preempted children and one clean final cohort."""
    pattern = re.compile(rf"/{re.escape(stress.cohort_child_name(stage, generation, 0)[:-3])}(\d{{3}})$")
    by_attempt: dict[int, dict[str, Any]] = {}
    failures: list[str] = []
    for child in children:
        match = pattern.search(str(child["job"]))
        if match is None:
            failures.append(f"unexpected stress child identity: {child['job']}")
            continue
        attempt = int(match.group(1))
        if attempt in by_attempt:
            failures.append(f"duplicate stress cohort attempt {attempt}")
            continue
        by_attempt[attempt] = child

    expected_attempts = set(range(final_attempt + 1))
    if set(by_attempt) != expected_attempts:
        failures.append(f"stress cohort attempts {sorted(by_attempt)} != expected {sorted(expected_attempts)}")
    for attempt, child in sorted(by_attempt.items()):
        if child["total_tasks"] != replicas:
            failures.append(f"stress cohort attempt {attempt} task count {child['total_tasks']} != expected {replicas}")
        if attempt == final_attempt:
            failures.extend(_iris_failures(child))
            continue
        if not (
            child["state"] != "succeeded"
            and child["failures"] == 0
            and child["preemptions"] > 0
            and child["running_tasks"] == 0
        ):
            failures.append(f"abandoned cohort attempt {attempt} was not preemption-only: {child}")
    return by_attempt.get(final_attempt), failures


def _forced_preemption_recovery_failures(*, required: bool, final_attempt: int) -> list[str]:
    """Require one replaced cohort only for the deliberate recovery gate."""
    if not required or final_attempt >= 1:
        return []
    return ["forced-preemption gate did not replace an abandoned cohort attempt"]


def _frozen_cohort_contract_failures(
    preregistration: dict[str, Any],
    *,
    cohort_attempt: int,
    iris_attempt: int,
) -> list[str]:
    """Validate static placement and attempt rules frozen for this gate."""
    cohort = preregistration.get("design", {}).get("cohort", {})
    integrity = preregistration.get("release_gate", {}).get("concurrency_and_integrity", {})
    failures: list[str] = []
    if integrity.get("iris_topology_coscheduling_forbidden") and (
        cohort.get("placement_mode") != stress.CohortPlacementMode.INDEPENDENT
        or cohort.get("coscheduling_group_by") is not None
        or cohort.get("coscheduling_enabled") is not False
    ):
        failures.append(f"frozen independent-placement contract drifted: {cohort}")
    if (
        integrity.get("parent_managed_whole_cohort_replacement_forbidden")
        and int(cohort.get("parent_managed_whole_cohort_retries", -1)) != 0
    ):
        failures.append("frozen independent cohort unexpectedly permits parent-managed replacement")
    if integrity.get("zero_parent_and_iris_attempt_required") and (cohort_attempt != 0 or iris_attempt != 0):
        failures.append(
            "independent fail-closed gate requires parent attempt 0 and Iris attempt 0: "
            f"parent={cohort_attempt}, iris={iris_attempt}"
        )
    return failures


def _rendezvous_evidence(
    rendezvous_id: str,
    specs: tuple[RunSpec, ...],
    *,
    distinct_physical_workers_required: bool = False,
    required_worker_region: str | None = None,
    required_worker_id_regex: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Read and validate the exact stage-level admission barrier."""
    row_ids = tuple(spec.trajectory_id for spec in specs)
    fs, _, ready_paths, release_path = stress._rendezvous_paths(stress.RENDEZVOUS_ROOT, rendezvous_id, row_ids)
    failures: list[str] = []
    markers: list[dict[str, Any]] = []
    marker_identities: dict[str, dict[str, Any]] = {}
    require_gcs_metadata = stress.RENDEZVOUS_ROOT.startswith("gs://")
    for row_id, marker_path in zip(row_ids, ready_paths, strict=True):
        if not fs.exists(marker_path):
            failures.append(f"missing rendezvous marker for {row_id}")
            continue
        with fs.open(marker_path) as handle:
            marker = json.load(handle)
        worker_claim_id = marker.get("worker_claim_id")
        physical_worker_id = marker.get("physical_worker_id")
        worker_region = marker.get("worker_region")
        expected_marker = (
            stress._ready_marker_payload(
                worker_claim_id=worker_claim_id,
                row_id=row_id,
                rendezvous_id=rendezvous_id,
                row_ids=row_ids,
                physical_worker_id=physical_worker_id,
                worker_region=worker_region,
            )
            if isinstance(worker_claim_id, str) and worker_claim_id
            else None
        )
        if (
            marker.get("protocol_version") != stress.RENDEZVOUS_PROTOCOL_VERSION
            or marker.get("rendezvous_id") != rendezvous_id
            or marker.get("row_id") != row_id
            or marker.get("row_ids") != list(row_ids)
            or expected_marker is None
            or marker.get("marker_nonce") != expected_marker["marker_nonce"]
        ):
            failures.append(f"rendezvous marker metadata drifted for {row_id}")
            continue
        info = fs.info(marker_path)
        generation = None if info.get("generation") is None else str(info["generation"])
        created_at = info.get("timeCreated")
        created_epoch = None if created_at is None else _utc_epoch(str(created_at), field=f"timeCreated for {row_id}")
        if require_gcs_metadata and (generation is None or created_epoch is None):
            failures.append(f"GCS generation/timeCreated metadata is missing for {row_id}")
            continue
        identity = {
            "marker_nonce": str(marker["marker_nonce"]),
            "generation": generation,
            "worker_claim_id": worker_claim_id,
            "physical_worker_id": physical_worker_id,
            "worker_region": worker_region,
        }
        marker_identities[row_id] = identity
        markers.append(
            {
                **marker,
                "generation": generation,
                "time_created": created_at,
                "time_created_epoch": created_epoch,
            }
        )

    release: dict[str, Any] | None = None
    if not fs.exists(release_path):
        failures.append("missing rendezvous release marker")
    else:
        with fs.open(release_path) as handle:
            candidate = json.load(handle)
        if (
            candidate.get("protocol_version") != stress.RENDEZVOUS_PROTOCOL_VERSION
            or candidate.get("rendezvous_id") != rendezvous_id
            or candidate.get("row_ids") != list(row_ids)
            or candidate.get("ready_markers") != marker_identities
        ):
            failures.append("rendezvous release metadata drifted")
        else:
            info = fs.info(release_path)
            generation = None if info.get("generation") is None else str(info["generation"])
            created_at = info.get("timeCreated")
            created_epoch = None if created_at is None else _utc_epoch(str(created_at), field="release timeCreated")
            if require_gcs_metadata and (generation is None or created_epoch is None):
                failures.append("GCS generation/timeCreated metadata is missing for the release")
            release = {
                **candidate,
                "generation": generation,
                "time_created": created_at,
                "time_created_epoch": created_epoch,
            }

    ready_epochs = [marker["time_created_epoch"] for marker in markers if marker["time_created_epoch"] is not None]
    worker_claim_ids = [str(marker["worker_claim_id"]) for marker in markers]
    if len(set(worker_claim_ids)) != len(worker_claim_ids):
        failures.append("rendezvous worker claim IDs are not unique")
    physical_worker_ids = [marker.get("physical_worker_id") for marker in markers]
    if distinct_physical_workers_required:
        valid_worker_ids = [worker_id for worker_id in physical_worker_ids if isinstance(worker_id, str) and worker_id]
        if len(valid_worker_ids) != len(specs):
            failures.append(f"physical worker identity coverage {len(valid_worker_ids)}/{len(specs)}")
        elif len(set(valid_worker_ids)) != len(valid_worker_ids):
            failures.append("rendezvous physical worker IDs are not unique")
    if required_worker_region is not None:
        wrong_regions = {
            str(marker.get("worker_region"))
            for marker in markers
            if marker.get("worker_region") != required_worker_region
        }
        if wrong_regions:
            failures.append(f"rendezvous workers landed outside {required_worker_region}: {sorted(wrong_regions)}")
    if required_worker_id_regex is not None:
        worker_pattern = re.compile(required_worker_id_regex)
        invalid_worker_ids = [
            str(worker_id)
            for worker_id in physical_worker_ids
            if not isinstance(worker_id, str) or worker_pattern.fullmatch(worker_id) is None
        ]
        if invalid_worker_ids:
            failures.append(f"rendezvous worker identities do not match the frozen placement: {invalid_worker_ids}")
    release_after_last_ready_seconds: float | None = None
    barrier_wait_seconds_by_row: dict[str, float] | None = None
    barrier_wait_tpu_seconds_total: float | None = None
    if len(ready_epochs) == len(specs) and release is not None and release["time_created_epoch"] is not None:
        release_after_last_ready_seconds = release["time_created_epoch"] - max(ready_epochs)
        if release_after_last_ready_seconds < 0:
            failures.append(
                f"GCS release was created {abs(release_after_last_ready_seconds):.3f}s before the last ready marker"
            )
        barrier_wait_seconds_by_row = {
            str(marker["row_id"]): release["time_created_epoch"] - float(marker["time_created_epoch"])
            for marker in markers
        }
        barrier_wait_tpu_seconds_total = sum(barrier_wait_seconds_by_row.values())

    return (
        {
            "rendezvous_id": rendezvous_id,
            "expected_row_ids": list(row_ids),
            "markers": markers,
            "release": release,
            "ready_spread_seconds": max(ready_epochs) - min(ready_epochs) if ready_epochs else None,
            "release_after_last_ready_seconds": release_after_last_ready_seconds,
            "barrier_wait_seconds_by_row": barrier_wait_seconds_by_row,
            "barrier_wait_tpu_seconds_total": barrier_wait_tpu_seconds_total,
        },
        failures,
    )


def _event_overlap_failures(rows: list[dict[str, Any]], overlap_start: float, overlap_end: float) -> list[str]:
    failures: list[str] = []
    for row in rows:
        for event in row["events"]:
            for window in ("pre", "post"):
                window_start = float(event[f"{window}_wallclock_start"])
                window_end = float(event[f"{window}_wallclock_end"])
                if window_start < overlap_start or window_end > overlap_end:
                    failures.append(
                        f"{row['trajectory_id']} {event['event']} {window}-window "
                        f"[{window_start:.3f}, {window_end:.3f}] "
                        f"is outside all-row overlap [{overlap_start:.3f}, {overlap_end:.3f}]"
                    )
    return failures


def _stress_event_recovery_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize event-relative recovery without treating event alignment as causal."""
    event_labels = tuple(event["event"] for event in rows[0]["events"])
    if any(tuple(event["event"] for event in row["events"]) != event_labels for row in rows):
        raise ValueError("Stress rows do not share one event layout")

    diagnostics: list[dict[str, Any]] = []
    for event_index, event_label in enumerate(event_labels):
        event_rows = [(row, row["events"][event_index]) for row in rows]
        recoveries = np.array([float(event["throughput_recovery_fraction"]) for _, event in event_rows])
        slowdown_positions = np.array([int(event["post_event_longest_slowdown_positions"]) for _, event in event_rows])
        slowdown_analyzable_fractions = np.array(
            [float(event["post_event_slowdown_analyzable_fraction"]) for _, event in event_rows]
        )
        support_diagnostics: list[dict[str, Any]] = []
        for support_id in sorted({row["support_id"] for row, _ in event_rows}):
            support_recoveries = np.array(
                [
                    float(event["throughput_recovery_fraction"])
                    for row, event in event_rows
                    if row["support_id"] == support_id
                ]
            )
            support_diagnostics.append(
                {
                    "support_id": support_id,
                    "row_count": len(support_recoveries),
                    "recovery_median": _percentile(support_recoveries, 50),
                    "recovery_p25": _percentile(support_recoveries, 25),
                    "recovery_max": float(np.max(support_recoveries)),
                    "recovery_min": float(np.min(support_recoveries)),
                }
            )

        diagnostics.append(
            {
                "event": event_label,
                "row_count": len(event_rows),
                "recovery_median": _percentile(recoveries, 50),
                "recovery_p25": _percentile(recoveries, 25),
                "recovery_min": float(np.min(recoveries)),
                "slowdown_positions_max": int(np.max(slowdown_positions)),
                "slowdown_analyzable_fraction_min": float(np.min(slowdown_analyzable_fractions)),
                "slowdown_by_row": [
                    {
                        "trajectory_id": row["trajectory_id"],
                        "positions": int(event["post_event_longest_slowdown_positions"]),
                        "analyzable_fraction": float(event["post_event_slowdown_analyzable_fraction"]),
                    }
                    for row, event in event_rows
                ],
                "support_diagnostics": support_diagnostics,
                "gate_role": "diagnostic_only",
            }
        )
    return diagnostics


def _stress_short_window_gate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Reject sustained absolute throughput collapse without anchoring to an event."""
    exceeding = [
        row for row in rows if int(row["short_window_longest_below_floor_positions"]) > STRESS_SHORT_WINDOW_MAX_POSITIONS
    ]
    failures: list[str] = []
    minimum_analyzable_fraction = min(float(row["short_window_analyzable_fraction"]) for row in rows)
    if minimum_analyzable_fraction < HISTORY_COVERAGE_FRACTION_MIN:
        failures.append(
            f"short-window analyzable fraction {minimum_analyzable_fraction:.6f} < "
            f"{HISTORY_COVERAGE_FRACTION_MIN:.6f}"
        )
    if len(exceeding) > STRESS_SHORT_WINDOW_EXCEEDANCE_MAX_ROWS:
        failures.append(
            f"{len(exceeding)} rows exceed {STRESS_SHORT_WINDOW_MAX_POSITIONS} consecutive short-window positions "
            f"below {STRESS_SHORT_WINDOW_TOKENS_PER_SECOND_MIN:.0f} tok/s"
        )
    return (
        {
            "window_steps": STRESS_EVENT_SLOWDOWN_ROLLING_WINDOW_STEPS,
            "throughput_floor": STRESS_SHORT_WINDOW_TOKENS_PER_SECOND_MIN,
            "max_positions": STRESS_SHORT_WINDOW_MAX_POSITIONS,
            "exceedance_max_rows": STRESS_SHORT_WINDOW_EXCEEDANCE_MAX_ROWS,
            "exceedance_count": len(exceeding),
            "analyzable_fraction_min": minimum_analyzable_fraction,
            "rows": [
                {
                    "trajectory_id": row["trajectory_id"],
                    "minimum_median": row["short_window_tokens_per_second_min"],
                    "minimum_center_step": row["short_window_minimum_center_step"],
                    "longest_below_floor_positions": row["short_window_longest_below_floor_positions"],
                    "analyzable_fraction": row["short_window_analyzable_fraction"],
                }
                for row in rows
            ],
            "status": "pass" if not failures else "fail",
        },
        failures,
    )


def _stress_synchronized_depression_gate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Detect synchronized relative slowdown on a shared host-time grid outside declared checkpoint pauses."""
    overlap = _all_row_overlap(rows)
    complete_bins = int((overlap["end"] - overlap["start"]) // STRESS_SYNCHRONIZED_BIN_SECONDS)
    if complete_bins < 1:
        raise ValueError("All-row overlap is shorter than one synchronized-depression bin")
    edges = overlap["start"] + np.arange(complete_bins + 1) * STRESS_SYNCHRONIZED_BIN_SECONDS
    quorum_required = math.ceil(STRESS_SYNCHRONIZED_QUORUM_FRACTION * len(rows))
    checkpoint_intervals = [
        (float(interval["start"]), float(interval["stop"]))
        for row in rows
        for interval in row["pause_accounting"]["intervals"]
        if interval["classification"] == "checkpoint"
    ]
    depressed_counts: list[int] = []
    covered_counts: list[int] = []
    excluded_checkpoint_bins = 0
    for start, stop in itertools.pairwise(edges):
        if any(
            start < checkpoint_stop and stop > checkpoint_start
            for checkpoint_start, checkpoint_stop in checkpoint_intervals
        ):
            excluded_checkpoint_bins += 1
            continue
        depressed = 0
        covered = 0
        for row in rows:
            series = row["_runtime_series"]
            selected = (series["host_timestamp"] >= start) & (series["host_timestamp"] < stop)
            if not np.any(selected):
                continue
            covered += 1
            baseline = float(row["tokens_per_second_p50"])
            observed = _percentile(series["throughput/tokens_per_second"][selected], 50)
            depressed += observed < STRESS_SYNCHRONIZED_DEPRESSION_FRACTION * baseline
        covered_counts.append(covered)
        depressed_counts.append(depressed)
    depressed_quorum = np.array(
        [
            covered == len(rows) and depressed >= quorum_required
            for covered, depressed in zip(covered_counts, depressed_counts, strict=True)
        ]
    )
    if not covered_counts:
        raise ValueError("No synchronized-depression bins remain after declared checkpoint exclusions")
    longest_bins = _longest_true_run(depressed_quorum)
    longest_seconds = longest_bins * STRESS_SYNCHRONIZED_BIN_SECONDS
    failures: list[str] = []
    if min(covered_counts) < len(rows):
        failures.append(f"synchronized-depression bins cover only {min(covered_counts)}/{len(rows)} rows")
    if longest_seconds > STRESS_SYNCHRONIZED_MAX_SECONDS:
        failures.append(f"synchronized depression lasts {longest_seconds:.3f}s > {STRESS_SYNCHRONIZED_MAX_SECONDS:.3f}s")
    return (
        {
            "bin_seconds": STRESS_SYNCHRONIZED_BIN_SECONDS,
            "grid_anchor": "all-row host-time overlap start",
            "excluded_checkpoint_bins": excluded_checkpoint_bins,
            "analyzed_bins": len(covered_counts),
            "depression_fraction": STRESS_SYNCHRONIZED_DEPRESSION_FRACTION,
            "quorum_required": quorum_required,
            "max_seconds": STRESS_SYNCHRONIZED_MAX_SECONDS,
            "longest_depressed_bins": longest_bins,
            "longest_depressed_seconds": longest_seconds,
            "minimum_covered_rows": min(covered_counts),
            "depressed_counts": depressed_counts,
            "status": "pass" if not failures else "fail",
        },
        failures,
    )


def _log_throughput_ratio(
    series: dict[str, np.ndarray],
    *,
    pre_start: int,
    pre_stop: int,
    post_start: int,
    post_stop: int,
) -> float:
    """Return a complete-window log ratio of post versus pre median throughput."""
    pre = _window(series, pre_start, pre_stop)["throughput/tokens_per_second"]
    post = _window(series, post_start, post_stop)["throughput/tokens_per_second"]
    if len(pre) != pre_stop - pre_start or len(post) != post_stop - post_start:
        raise ValueError(
            "C12 onset inference requires complete contiguous histories: "
            f"pre={len(pre)}/{pre_stop - pre_start}, post={len(post)}/{post_stop - post_start}"
        )
    pre_median = float(np.median(pre))
    post_median = float(np.median(post))
    if pre_median <= 0 or post_median <= 0:
        raise ValueError("C12 onset inference requires positive throughput")
    return math.log(post_median / pre_median)


def _exact_balanced_assignment_test(
    response_by_row: list[dict[int, float]],
    observed_onsets: tuple[int, ...],
    *,
    alternative: str,
) -> dict[str, Any]:
    """Enumerate every unique balanced reassignment of the frozen onset labels."""
    assignments = tuple(sorted(set(itertools.permutations(observed_onsets))))
    observed = float(np.mean([response_by_row[index][onset] for index, onset in enumerate(observed_onsets)]))
    statistics_by_assignment = np.array(
        [
            np.mean([response_by_row[index][onset] for index, onset in enumerate(assignment)])
            for assignment in assignments
        ]
    )
    tolerance = 1e-15
    if alternative == "lower":
        p_value = float(np.mean(statistics_by_assignment <= observed + tolerance))
    elif alternative == "two-sided":
        null_center = float(np.mean(statistics_by_assignment))
        p_value = float(
            np.mean(np.abs(statistics_by_assignment - null_center) >= abs(observed - null_center) - tolerance)
        )
    else:
        raise ValueError(f"Unknown assignment-test alternative: {alternative}")
    return {
        "observed_mean_log_ratio": observed,
        "observed_fractional_change": math.exp(observed) - 1.0,
        "alternative": alternative,
        "exact_p_value": p_value,
        "assignment_count": len(assignments),
        "null_mean_log_ratio": float(np.mean(statistics_by_assignment)),
        "null_min_log_ratio": float(np.min(statistics_by_assignment)),
        "null_max_log_ratio": float(np.max(statistics_by_assignment)),
    }


def _c12_decay_assignment_diagnostic(primary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Test whether throughput changes track randomized optimizer-decay onset labels."""
    ordered_rows = sorted(primary_rows, key=lambda row: str(row["trajectory_id"]))
    observed_onsets = tuple(int(row["optimizer_decay_step"]) for row in ordered_rows)
    if len(ordered_rows) != 8:
        raise ValueError(f"C12 requires eight m100a rows, got {len(ordered_rows)}")
    if observed_onsets != stress.C12_PRIMARY_OPTIMIZER_DECAY_STEPS:
        raise ValueError(f"C12 onset assignment drifted: {observed_onsets}")
    candidate_onsets = tuple(sorted(set(observed_onsets)))
    if any(observed_onsets.count(onset) != 2 for onset in candidate_onsets) or len(candidate_onsets) != 4:
        raise ValueError("C12 requires four optimizer-decay onset labels replicated exactly twice")

    primary_response: list[dict[int, float]] = []
    pretrend_response: list[dict[int, float]] = []
    lead_placebo_response: list[dict[int, float]] = []
    for row in ordered_rows:
        series = row["_runtime_series"]
        primary_by_onset: dict[int, float] = {}
        pretrend_by_onset: dict[int, float] = {}
        placebo_by_onset: dict[int, float] = {}
        for onset in candidate_onsets:
            window = C12_ASSIGNMENT_WINDOW_STEPS
            primary_by_onset[onset] = _log_throughput_ratio(
                series,
                pre_start=onset - window,
                pre_stop=onset,
                post_start=onset,
                post_stop=onset + window,
            )
            pretrend_by_onset[onset] = _log_throughput_ratio(
                series,
                pre_start=onset - 2 * window,
                pre_stop=onset - window,
                post_start=onset - window,
                post_stop=onset,
            )
            placebo_onset = onset - C12_LEAD_PLACEBO_OFFSET_STEPS
            placebo_by_onset[onset] = _log_throughput_ratio(
                series,
                pre_start=placebo_onset - window,
                pre_stop=placebo_onset,
                post_start=placebo_onset,
                post_stop=placebo_onset + window,
            )
        primary_response.append(primary_by_onset)
        pretrend_response.append(pretrend_by_onset)
        lead_placebo_response.append(placebo_by_onset)

    primary_test = _exact_balanced_assignment_test(primary_response, observed_onsets, alternative="lower")
    pretrend_test = _exact_balanced_assignment_test(pretrend_response, observed_onsets, alternative="two-sided")
    lead_placebo_test = _exact_balanced_assignment_test(
        lead_placebo_response,
        observed_onsets,
        alternative="two-sided",
    )
    primary_significant = primary_test["exact_p_value"] <= C12_ASSIGNMENT_ALPHA
    falsification_failed = (
        pretrend_test["exact_p_value"] <= C12_ASSIGNMENT_ALPHA
        or lead_placebo_test["exact_p_value"] <= C12_ASSIGNMENT_ALPHA
    )
    if falsification_failed:
        classification = "falsification_failed"
    elif primary_significant:
        classification = "decay_aligned"
    else:
        classification = "no_detectable_decay_alignment"

    return {
        "classification": classification,
        "primary_support_id": "m100a",
        "primary_row_count": len(ordered_rows),
        "trajectory_ids": [row["trajectory_id"] for row in ordered_rows],
        "optimizer_decay_steps": list(observed_onsets),
        "onset_assignment_seed": stress.C12_ONSET_ASSIGNMENT_SEED,
        "candidate_onsets": list(candidate_onsets),
        "alpha": C12_ASSIGNMENT_ALPHA,
        "window_steps": C12_ASSIGNMENT_WINDOW_STEPS,
        "lead_placebo_offset_steps": C12_LEAD_PLACEBO_OFFSET_STEPS,
        "primary": primary_test,
        "pretrend_falsification": pretrend_test,
        "lead_placebo_falsification": lead_placebo_test,
        "gate_role": "diagnostic_only; classification never overrides the operational release gate",
    }


def _stress_decay_alignment_diagnostic(rows: list[dict[str, Any]], *, stage: int) -> dict[str, Any]:
    """Run the frozen stage-specific optimizer-onset diagnostic."""
    primary_rows = [row for row in rows if row["support_id"] == "m100a"]
    if stage == 12:
        return _c12_decay_assignment_diagnostic(primary_rows)
    return {
        "classification": "underpowered",
        "decision_rule": (
            "C6 reports event-study inputs only. Exact onset-permutation inference at alpha=0.01 begins at C12, "
            "where eight same-support rows provide four replicated onset cohorts."
        ),
        "primary_support_id": "m100a",
        "primary_row_count": len(primary_rows),
        "optimizer_decay_steps": [row["optimizer_decay_step"] for row in primary_rows],
        "minimum_center_steps": [row["causal_window_minimum_center_step"] for row in primary_rows],
        "gate_role": "diagnostic_only",
    }


def _safe_stress_decay_alignment_diagnostic(rows: list[dict[str, Any]], *, stage: int) -> dict[str, Any]:
    """Run the diagnostic without allowing it to suppress the operational report."""
    try:
        return _stress_decay_alignment_diagnostic(rows, stage=stage)
    except Exception as error:
        logger.exception("Diagnostic-only decay-alignment analysis is unavailable")
        return {
            "classification": "unavailable",
            "reason": f"{type(error).__name__}: {error}",
            "gate_role": "diagnostic_only; unavailability never overrides the operational release gate",
        }


def _checkpoint_pause_gate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Bound checkpoint finalization and cross-row spread independently of training synchrony."""
    steps = sorted(
        {step for row in rows for step in row["pause_accounting"]["checkpoint_seconds_by_step"]},
        key=int,
    )
    by_step: list[dict[str, Any]] = []
    failures: list[str] = []
    for step in steps:
        seconds = [float(row["pause_accounting"]["checkpoint_seconds_by_step"][step]) for row in rows]
        spread = max(seconds) - min(seconds)
        if spread > CHECKPOINT_PAUSE_CROSS_ROW_SPREAD_SECONDS_MAX:
            failures.append(
                f"checkpoint step {step} cross-row pause spread {spread:.3f}s > "
                f"{CHECKPOINT_PAUSE_CROSS_ROW_SPREAD_SECONDS_MAX:.3f}s"
            )
        by_step.append(
            {
                "step": int(step),
                "minimum_seconds": min(seconds),
                "median_seconds": float(statistics.median(seconds)),
                "maximum_seconds": max(seconds),
                "spread_seconds": spread,
            }
        )
    return (
        {
            "per_row_seconds_max": CHECKPOINT_PAUSE_SECONDS_MAX,
            "cross_row_spread_seconds_max": CHECKPOINT_PAUSE_CROSS_ROW_SPREAD_SECONDS_MAX,
            "by_step": by_step,
            "status": "pass" if not failures else "fail",
        },
        failures,
    )


def _all_row_overlap(rows: list[dict[str, Any]], *, timeline: str = "host") -> dict[str, Any]:
    """Measure the common host timeline."""
    if not rows:
        raise ValueError("Cannot measure concurrency without runtime rows")
    if timeline != "host":
        raise ValueError(f"Unknown overlap timeline: {timeline}")
    first_key = f"first_{timeline}_timestamp"
    last_key = f"last_{timeline}_timestamp"
    span_key = "runtime_span_seconds"
    overlap_start = max(float(row[first_key]) for row in rows)
    overlap_end = min(float(row[last_key]) for row in rows)
    longest_runtime_span = max(float(row[span_key]) for row in rows)
    if longest_runtime_span <= 0:
        raise ValueError("Longest runtime span must be positive")
    overlap_seconds = overlap_end - overlap_start
    return {
        "start": overlap_start,
        "end": overlap_end,
        "seconds": overlap_seconds,
        "fraction_of_longest_span": overlap_seconds / longest_runtime_span,
        "longest_span_seconds": longest_runtime_span,
        "timeline": timeline,
    }


def _write_report(
    *,
    mode: str,
    stage: int | None,
    iris_job: str,
    specs: tuple[RunSpec, ...],
    output_dir: Path,
    previous_stage_report_sha256: str | None,
    previous_stage_generation: int | None,
    rendezvous_id: str | None,
    generation: int | None,
    cohort_attempt: int | None,
    iris_attempt: int | None,
    preregistration_sha256: str | None,
    preregistration_evidence: dict[str, Any] | None,
    preregistration: dict[str, Any] | None,
    forced_preemption_recovery_required: bool,
) -> tuple[dict[str, Any], Path]:
    api = wandb.Api(timeout=60)
    iris = _iris_summary(iris_job)
    iris_children = _iris_child_summaries(iris_job)
    rows = [
        _evaluate_run(
            spec,
            api,
            enforce_per_row_event_recovery=mode != "stress",
            enforce_pause_gates=mode == "stress",
        )
        for spec in specs
    ]
    failures = _iris_failures(iris)
    final_cohort_child: dict[str, Any] | None = None
    if mode == "stress":
        if stage is None or generation is None or cohort_attempt is None or iris_attempt is None:
            raise ValueError("Stress child validation requires stage, generation, cohort attempt, and Iris attempt")
        final_cohort_child, child_failures = _stress_cohort_child_evidence(
            iris_children,
            stage=stage,
            generation=generation,
            final_attempt=cohort_attempt,
            replicas=len(specs),
        )
        failures.extend(child_failures)
        failures.extend(
            _forced_preemption_recovery_failures(
                required=forced_preemption_recovery_required,
                final_attempt=cohort_attempt,
            )
        )
    else:
        if len(iris_children) != len(specs):
            failures.append(f"Iris child count {len(iris_children)} != expected {len(specs)}")
        for child in iris_children:
            failures.extend(_iris_failures(child))
    overlap = _all_row_overlap(rows)
    overlap_start = overlap["start"]
    overlap_end = overlap["end"]
    concurrent_overlap_seconds = overlap["seconds"]
    longest_runtime_span_seconds = overlap["longest_span_seconds"]
    concurrent_overlap_fraction = overlap["fraction_of_longest_span"]
    runtime_start_skew_seconds = max(row["first_host_timestamp"] for row in rows) - min(
        row["first_host_timestamp"] for row in rows
    )
    if concurrent_overlap_seconds < CONCURRENT_OVERLAP_SECONDS_MIN:
        failures.append(
            f"all-row runtime overlap {concurrent_overlap_seconds:.3f}s < {CONCURRENT_OVERLAP_SECONDS_MIN:.3f}s"
        )
    if mode == "stress" and concurrent_overlap_fraction < CONCURRENT_OVERLAP_FRACTION_MIN:
        failures.append(
            f"all-row runtime overlap fraction {concurrent_overlap_fraction:.6f} < "
            f"{CONCURRENT_OVERLAP_FRACTION_MIN:.6f} of the longest row span"
        )
    if mode == "stress" and runtime_start_skew_seconds > RUNTIME_START_SKEW_SECONDS_MAX:
        failures.append(f"runtime start skew {runtime_start_skew_seconds:.3f}s > {RUNTIME_START_SKEW_SECONDS_MAX:.3f}s")
    failures.extend(_event_overlap_failures(rows, overlap_start, overlap_end))
    rendezvous: dict[str, Any] | None = None
    completion_rendezvous: dict[str, Any] | None = None
    integrity = (
        {} if preregistration is None else preregistration.get("release_gate", {}).get("concurrency_and_integrity", {})
    )
    if mode == "stress" and preregistration is not None:
        if cohort_attempt is None or iris_attempt is None:
            raise ValueError("Stress contract validation requires parent and Iris attempts")
        failures.extend(
            _frozen_cohort_contract_failures(
                preregistration,
                cohort_attempt=cohort_attempt,
                iris_attempt=iris_attempt,
            )
        )
    ready_spread_seconds_max = float(
        integrity.get("rendezvous_ready_spread_seconds_max", RENDEZVOUS_READY_SPREAD_SECONDS_MAX)
    )
    completion_ready_spread_seconds_max = float(
        integrity.get(
            "completion_rendezvous_ready_spread_seconds_max",
            COMPLETION_RENDEZVOUS_READY_SPREAD_SECONDS_MAX,
        )
    )
    distinct_physical_workers_required = bool(integrity.get("distinct_physical_worker_per_row_required", False))
    required_worker_region = integrity.get("required_realized_worker_region")
    required_worker_id_regex = integrity.get("required_realized_worker_id_regex")
    if mode == "stress":
        if rendezvous_id is None:
            failures.append("stress analysis lacks the exact rendezvous ID")
        else:
            rendezvous, rendezvous_failures = _rendezvous_evidence(
                rendezvous_id,
                specs,
                distinct_physical_workers_required=distinct_physical_workers_required,
                required_worker_region=required_worker_region,
                required_worker_id_regex=required_worker_id_regex,
            )
            failures.extend(rendezvous_failures)
            ready_spread_seconds = rendezvous["ready_spread_seconds"]
            if ready_spread_seconds is None:
                failures.append("rendezvous ready spread is unavailable")
            elif ready_spread_seconds > ready_spread_seconds_max:
                failures.append(
                    f"rendezvous ready spread {ready_spread_seconds:.3f}s > " f"{ready_spread_seconds_max:.3f}s"
                )
            marker_claim_ids = {str(marker["worker_claim_id"]) for marker in rendezvous["markers"]}
            final_child_job = None if final_cohort_child is None else str(final_cohort_child["job"])
            expected_claim_ids = (
                {f"{final_child_job}/{task_index}:{iris_attempt}" for task_index in range(len(specs))}
                if final_child_job is not None
                else set()
            )
            if marker_claim_ids != expected_claim_ids:
                failures.append(
                    "rendezvous worker claims do not match the final Iris cohort attempt: "
                    f"claims={sorted(marker_claim_ids)}, expected={sorted(expected_claim_ids)}"
                )
            release = rendezvous["release"]
            if release is not None and release["time_created_epoch"] is not None:
                earliest_runtime_delay = min(row["first_host_timestamp"] for row in rows) - release["time_created_epoch"]
                rendezvous["earliest_runtime_delay_after_release_seconds"] = earliest_runtime_delay
                if earliest_runtime_delay < -RENDEZVOUS_RUNTIME_START_EARLY_TOLERANCE_SECONDS:
                    failures.append(
                        f"earliest runtime timestamp precedes rendezvous release by "
                        f"{abs(earliest_runtime_delay):.3f}s"
                    )
            if stage is not None and generation is not None and cohort_attempt is not None and iris_attempt is not None:
                completion_id = stress.completion_rendezvous_id(stage, generation, cohort_attempt, iris_attempt)
                completion_rendezvous, completion_failures = _rendezvous_evidence(
                    completion_id,
                    specs,
                    distinct_physical_workers_required=distinct_physical_workers_required,
                    required_worker_region=required_worker_region,
                    required_worker_id_regex=required_worker_id_regex,
                )
                failures.extend(completion_failures)
                if rendezvous is not None:
                    start_workers = {
                        str(marker["row_id"]): marker.get("physical_worker_id") for marker in rendezvous["markers"]
                    }
                    completion_workers = {
                        str(marker["row_id"]): marker.get("physical_worker_id")
                        for marker in completion_rendezvous["markers"]
                    }
                    if start_workers != completion_workers:
                        failures.append("physical worker identities changed between start and completion barriers")
                completion_claim_ids = {str(marker["worker_claim_id"]) for marker in completion_rendezvous["markers"]}
                if completion_claim_ids != expected_claim_ids:
                    failures.append(
                        "completion worker claims do not match the final Iris cohort: "
                        f"claims={sorted(completion_claim_ids)}, expected={sorted(expected_claim_ids)}"
                    )
                completion_ready_spread_seconds = completion_rendezvous["ready_spread_seconds"]
                if completion_ready_spread_seconds is None:
                    failures.append("completion rendezvous ready spread is unavailable")
                elif completion_ready_spread_seconds > completion_ready_spread_seconds_max:
                    failures.append(
                        f"completion rendezvous ready spread {completion_ready_spread_seconds:.3f}s > "
                        f"{completion_ready_spread_seconds_max:.3f}s"
                    )
    loading_p99_by_support: dict[str, list[float]] = {}
    for row in rows:
        loading_p99_by_support.setdefault(row["support_id"], []).append(row["loading_p99_seconds"])
    full_loading = statistics.median(loading_p99_by_support.get("full", []))
    finite_loading = [value for support_id in ("m100a", "m100b") for value in loading_p99_by_support.get(support_id, [])]
    finite_to_full_ratio = max(finite_loading) / full_loading
    if finite_to_full_ratio > FINITE_TO_FULL_LOADING_P99_RATIO_MAX:
        failures.append(
            f"finite/full loading-p99 ratio {finite_to_full_ratio:.6f} exceeds "
            f"{FINITE_TO_FULL_LOADING_P99_RATIO_MAX:.6f}"
        )
    stress_event_recovery: list[dict[str, Any]] | None = None
    stress_short_window: dict[str, Any] | None = None
    stress_synchronized_depression: dict[str, Any] | None = None
    stress_decay_alignment: dict[str, Any] | None = None
    checkpoint_pause: dict[str, Any] | None = None
    if mode == "stress":
        if stage is None:
            raise ValueError("Stress analysis requires a concurrency stage")
        stress_event_recovery = _stress_event_recovery_diagnostics(rows)
        stress_short_window, short_window_failures = _stress_short_window_gate(rows)
        failures.extend(short_window_failures)
        stress_synchronized_depression, synchronized_failures = _stress_synchronized_depression_gate(rows)
        failures.extend(synchronized_failures)
        stress_decay_alignment = _safe_stress_decay_alignment_diagnostic(rows, stage=stage)
        checkpoint_pause, checkpoint_pause_failures = _checkpoint_pause_gate(rows)
        failures.extend(checkpoint_pause_failures)
    if any(row["status"] != "pass" for row in rows):
        failures.append("one or more runtime rows failed their thresholds")
    for row in rows:
        row.pop("_runtime_series", None)
    report = {
        "report_version": "2026-08-13-runtime-gate-v12",
        "analysis_scope": "operational_only_no_endpoint_metrics",
        "endpoint_metrics_read": False,
        "scientific_inference_allowed": False,
        "mode": mode,
        "stage": stage,
        "generation": generation,
        "cohort_attempt": cohort_attempt,
        "iris_attempt": iris_attempt,
        "preregistration_sha256": preregistration_sha256,
        "preregistration_evidence": preregistration_evidence,
        "previous_stage_report_sha256": previous_stage_report_sha256,
        "previous_stage_generation": previous_stage_generation,
        "status": "pass" if not failures else "fail",
        "iris": iris,
        "iris_children": iris_children,
        "preemption_recovery": {
            "parent_historical_preemptions_allowed": False,
            "abandoned_child_preemptions_allowed": mode == "stress",
            "final_attempt_requires_fresh_namespaces": mode == "stress",
            "parent_managed_whole_cohort_replacement": mode == "stress",
            "abandoned_cohort_attempts": 0 if cohort_attempt is None else cohort_attempt,
            "cohort_attempt": cohort_attempt,
            "iris_attempt": iris_attempt,
            "assignment_time_iris_redispatch_allowed": mode == "stress",
            "forced_preemption_recovery_required": forced_preemption_recovery_required,
        },
        "thresholds": {
            "steady_state_start_step": STEADY_STATE_START_STEP,
            "loading_p99_seconds_max": LOADING_P99_MAX,
            "loading_over_one_second_fraction_max": LOADING_OVER_ONE_SECOND_FRACTION_MAX,
            "loading_seconds_max": LOADING_MAX,
            "duty_fraction_min": DUTY_FRACTION_MIN,
            "tokens_per_second_p50_min": TOKENS_PER_SECOND_P50_MIN,
            "mfu_percent_p50_min": MFU_P50_MIN,
            "rolling_window_steps": ROLLING_WINDOW_STEPS,
            "rolling_tokens_per_second_p50_min": ROLLING_TOKENS_PER_SECOND_P50_MIN,
            "event_tokens_per_second_p50_min": EVENT_TOKENS_PER_SECOND_P50_MIN,
            "event_loading_p99_seconds_max": EVENT_LOADING_P99_MAX,
            "event_recovery_fraction_min": EVENT_RECOVERY_FRACTION_MIN,
            "event_recovery_fraction_min_scope": "long-gate-only",
            "stress_event_slowdown_rolling_window_steps": STRESS_EVENT_SLOWDOWN_ROLLING_WINDOW_STEPS,
            "stress_event_slowdown_fraction": STRESS_EVENT_SLOWDOWN_FRACTION,
            "stress_event_recovery_gate_role": "diagnostic-only; event causality is under test",
            "stress_short_window_tokens_per_second_min": STRESS_SHORT_WINDOW_TOKENS_PER_SECOND_MIN,
            "stress_short_window_max_positions": STRESS_SHORT_WINDOW_MAX_POSITIONS,
            "stress_short_window_exceedance_max_rows": STRESS_SHORT_WINDOW_EXCEEDANCE_MAX_ROWS,
            "stress_synchronized_quorum_fraction": STRESS_SYNCHRONIZED_QUORUM_FRACTION,
            "stress_synchronized_depression_fraction": STRESS_SYNCHRONIZED_DEPRESSION_FRACTION,
            "stress_synchronized_bin_seconds": STRESS_SYNCHRONIZED_BIN_SECONDS,
            "stress_synchronized_max_seconds": STRESS_SYNCHRONIZED_MAX_SECONDS,
            "pause_accounting_slack_seconds": PAUSE_ACCOUNTING_SLACK_SECONDS,
            "unexplained_pause_longest_seconds_max": UNEXPLAINED_PAUSE_LONGEST_SECONDS_MAX,
            "unexplained_pause_total_seconds_max": UNEXPLAINED_PAUSE_TOTAL_SECONDS_MAX,
            "checkpoint_pause_seconds_max": CHECKPOINT_PAUSE_SECONDS_MAX,
            "checkpoint_pause_cross_row_spread_seconds_max": CHECKPOINT_PAUSE_CROSS_ROW_SPREAD_SECONDS_MAX,
            "history_coverage_fraction_min": HISTORY_COVERAGE_FRACTION_MIN,
            "event_history_coverage_fraction_min": EVENT_HISTORY_COVERAGE_FRACTION_MIN,
            "finite_to_full_loading_p99_ratio_max": FINITE_TO_FULL_LOADING_P99_RATIO_MAX,
            "all_row_concurrent_overlap_seconds_min": CONCURRENT_OVERLAP_SECONDS_MIN,
            "all_row_concurrent_overlap_fraction_min": CONCURRENT_OVERLAP_FRACTION_MIN,
            "runtime_start_skew_seconds_max": RUNTIME_START_SKEW_SECONDS_MAX,
            "rendezvous_ready_spread_seconds_max": ready_spread_seconds_max,
            "completion_rendezvous_ready_spread_seconds_max": completion_ready_spread_seconds_max,
            "distinct_physical_worker_per_row_required": distinct_physical_workers_required,
            "required_realized_worker_region": required_worker_region,
            "required_realized_worker_id_regex": required_worker_id_regex,
            "rendezvous_runtime_start_early_tolerance_seconds": RENDEZVOUS_RUNTIME_START_EARLY_TOLERANCE_SECONDS,
            "runtime_accounting_ratio_min": RUNTIME_ACCOUNTING_RATIO_MIN,
            "runtime_accounting_ratio_max": RUNTIME_ACCOUNTING_RATIO_MAX,
            "permanent_checkpoint_physical_bytes_max": PERMANENT_CHECKPOINT_BYTES_MAX,
            "c12_assignment_alpha": C12_ASSIGNMENT_ALPHA,
            "c12_assignment_window_steps": C12_ASSIGNMENT_WINDOW_STEPS,
            "c12_lead_placebo_offset_steps": C12_LEAD_PLACEBO_OFFSET_STEPS,
        },
        "finite_to_full_loading_p99_ratio": finite_to_full_ratio,
        "all_row_overlap_start": overlap_start,
        "all_row_overlap_end": overlap_end,
        "all_row_concurrent_overlap_seconds": concurrent_overlap_seconds,
        "all_row_concurrent_overlap_fraction": concurrent_overlap_fraction,
        "longest_runtime_span_seconds": longest_runtime_span_seconds,
        "runtime_start_skew_seconds": runtime_start_skew_seconds,
        "rendezvous": rendezvous,
        "completion_rendezvous": completion_rendezvous,
        "stress_event_recovery": stress_event_recovery,
        "stress_short_window": stress_short_window,
        "stress_synchronized_depression": stress_synchronized_depression,
        "stress_decay_alignment": stress_decay_alignment,
        "checkpoint_pause": checkpoint_pause,
        "runs": rows,
        "failures": failures,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "runtime_gate.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        f"# WSD80 gradient-conflict {mode} runtime gate",
        "",
        f"**Verdict:** `{report['status']}`",
        "",
        "Endpoint losses were neither requested from W&B nor used. This gate is operational only.",
        "",
        "| Run | Support | Load p99 | Tok/s p50 | MFU p50 | Duty | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['trajectory_id']} | {row['support_id']} | {row['loading_p99_seconds']:.6f}s | "
            f"{row['tokens_per_second_p50']:,.0f} | {row['mfu_percent_p50']:.3f}% | "
            f"{row['duty_fraction']:.6f} | {row['status']} |"
        )
    if failures:
        lines.extend(("", "## Failures", "", *(f"- {failure}" for failure in failures)))
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    return report, output_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_preregistration(expected_sha256: str, generation: int, stage: int) -> dict[str, Any]:
    preregistration = stress.validate_preregistration(expected_sha256, generation, stage)
    expected_analyzer_sha256 = preregistration.get("implementation_sha256", {}).get("runtime_gate")
    observed_analyzer_sha256 = _sha256(Path(__file__))
    if observed_analyzer_sha256 != expected_analyzer_sha256:
        raise ValueError(
            f"Runtime analyzer drifted from preregistration: {observed_analyzer_sha256} != "
            f"{expected_analyzer_sha256}"
        )
    return preregistration


def _remote_preregistration_evidence(expected_sha256: str, generation: int) -> dict[str, Any]:
    remote_url = stress.remote_preregistration_path(generation)
    fs, path = fsspec.core.url_to_fs(remote_url)
    if not fs.exists(path):
        raise ValueError(f"Missing remote preregistration: {remote_url}")
    with fs.open(path, "rb") as handle:
        payload = handle.read()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(f"Remote preregistration drifted: {observed_sha256} != {expected_sha256}")
    info = fs.info(path)
    return {
        "path": remote_url,
        "sha256": observed_sha256,
        "generation": None if info.get("generation") is None else str(info["generation"]),
    }


def _upload(local_path: Path, remote_path: str) -> dict[str, Any]:
    """Create an immutable remote report and verify its bytes."""
    subprocess.run(
        ["gcloud", "storage", "cp", "--if-generation-match=0", str(local_path), remote_path],
        check=True,
    )
    fs, path = fsspec.core.url_to_fs(remote_path)
    with fs.open(path, "rb") as handle:
        remote_sha256 = hashlib.sha256(handle.read()).hexdigest()
    local_sha256 = _sha256(local_path)
    if remote_sha256 != local_sha256:
        raise RuntimeError(f"Remote report drifted after upload: {remote_sha256} != {local_sha256}")
    info = fs.info(path)
    return {
        "path": remote_path,
        "sha256": remote_sha256,
        "generation": None if info.get("generation") is None else str(info["generation"]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("long-gate", "stress"))
    parser.add_argument("--stage", type=int, choices=stress.STAGE_CONCURRENCIES)
    parser.add_argument("--generation", type=int)
    parser.add_argument("--cohort-attempt", type=int)
    parser.add_argument("--preregistration-sha256")
    parser.add_argument("--iris-job")
    parser.add_argument("--previous-stage-report-sha256")
    parser.add_argument("--previous-stage-generation", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def _resolve_cohort_execution(stage: int, generation: int, requested: int | None) -> tuple[int, int]:
    """Select the latest parent and Iris attempt that crossed both barriers."""
    released = stress.released_cohort_executions(stress.RENDEZVOUS_ROOT, stage, generation)
    if not released:
        raise ValueError("No stress cohort attempt reached the rendezvous release barrier")
    completed = stress.released_cohort_executions(
        stress.RENDEZVOUS_ROOT,
        stage,
        generation,
        completion=True,
    )
    if not completed:
        raise ValueError("No stress cohort attempt reached the all-row completion barrier")
    if requested is None:
        selected = max(completed)
    else:
        requested_executions = tuple(execution for execution in completed if execution[0] == requested)
        if len(requested_executions) != 1:
            raise ValueError(
                f"Requested cohort attempt {requested} has {len(requested_executions)} completed Iris executions: "
                f"{requested_executions}"
            )
        selected = requested_executions[0]
    if selected not in released or selected not in completed:
        raise ValueError(
            f"Cohort execution {selected} lacks a start or completion release; "
            f"started={released}, completed={completed}"
        )
    if selected != max(released) or selected != max(completed):
        raise ValueError(
            f"Cohort execution {selected} is not the latest complete execution; "
            f"started={released}, completed={completed}"
        )
    return selected


def main() -> None:
    args = _parse_args()
    if args.mode == "long-gate":
        if args.stage is not None:
            raise ValueError("Long-gate analysis does not accept --stage")
        specs = _long_gate_specs()
        output_dir = LONG_GATE_OUTPUT_DIR
        iris_job = args.iris_job or LONG_GATE_IRIS_JOB
        if args.previous_stage_report_sha256 is not None:
            raise ValueError("Long-gate analysis does not accept a predecessor report")
        if args.previous_stage_generation is not None or args.generation is not None:
            raise ValueError("Long-gate analysis does not accept stress generations")
        if args.cohort_attempt is not None:
            raise ValueError("Long-gate analysis does not accept --cohort-attempt")
        if args.preregistration_sha256 is not None:
            raise ValueError("Long-gate analysis does not accept a stress preregistration")
        if args.output_root is not None:
            raise ValueError("Long-gate analysis does not accept --output-root")
        remote_path = (
            "gs://marin-us-central1/analysis/pinlin_calvin_xu/data_mixture/"
            "starcoder_wsd80_gradient_conflict_long_gate_20260811/runtime_gate.json"
        )
        rendezvous_id = None
        cohort_attempt = None
        iris_attempt = None
        preregistration_evidence = None
        preregistration = None
        forced_preemption_recovery_required = False
    else:
        if (
            args.stage is None
            or args.generation is None
            or args.iris_job is None
            or args.output_root is None
            or args.preregistration_sha256 is None
        ):
            raise ValueError(
                "Stress analysis requires --stage, --generation, --iris-job, --output-root, and "
                "--preregistration-sha256"
            )
        stress._validate_generation(args.generation)
        preregistration = _validate_preregistration(args.preregistration_sha256, args.generation, args.stage)
        forced_preemption_recovery_required = bool(
            preregistration.get("fault_injection", {}).get("forced_preemption_recovery_required", False)
        )
        preregistration_evidence = _remote_preregistration_evidence(
            args.preregistration_sha256,
            args.generation,
        )
        cohort_attempt, iris_attempt = _resolve_cohort_execution(args.stage, args.generation, args.cohort_attempt)
        rendezvous_id = stress.cohort_rendezvous_id(args.stage, args.generation, cohort_attempt, iris_attempt)
        specs = _stress_specs(args.stage, args.generation, cohort_attempt)
        output_dir = args.output_root / f"stage-c{args.stage:02d}"
        iris_job = args.iris_job
        stress._validate_previous_stage(
            args.stage,
            args.previous_stage_report_sha256,
            args.previous_stage_generation,
            preregistration=preregistration,
        )
        remote_path = stress.report_path(args.stage, args.generation)
    report, output_path = _write_report(
        mode=args.mode,
        stage=args.stage,
        iris_job=iris_job,
        specs=specs,
        output_dir=output_dir,
        previous_stage_report_sha256=args.previous_stage_report_sha256,
        previous_stage_generation=args.previous_stage_generation,
        rendezvous_id=rendezvous_id,
        generation=args.generation,
        cohort_attempt=cohort_attempt,
        iris_attempt=iris_attempt,
        preregistration_sha256=args.preregistration_sha256,
        preregistration_evidence=preregistration_evidence,
        preregistration=preregistration,
        forced_preemption_recovery_required=forced_preemption_recovery_required,
    )
    report_sha256 = _sha256(output_path)
    result: dict[str, Any] = {
        "status": report["status"],
        "report": str(output_path),
        "sha256": report_sha256,
    }
    if args.upload:
        if report["status"] != "pass":
            raise RuntimeError("Refusing to upload a non-passing runtime gate")
        result["remote"] = _upload(output_path, remote_path)
    print(json.dumps(result))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
