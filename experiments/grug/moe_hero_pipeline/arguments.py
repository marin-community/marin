# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded pipeline trial with optional checkpoint save and resume."""

import argparse
from collections.abc import Sequence

from experiments.grug.moe_hero_ep.optimizer import ExpertNormalization
from experiments.grug.moe_hero_pipeline.pipeline import AutomaticPipelineSchedule, QbBiasMode


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse pipeline arguments and reject incompatible recipe settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-hero", action="store_true")
    parser.add_argument("--coordinator-address", help="External JAX coordinator host:port")
    parser.add_argument("--num-processes", type=int)
    parser.add_argument("--process-id", type=int)
    parser.add_argument("--local-device-id", type=int)
    parser.add_argument("--processes-per-task", type=int, help="Local process count for the main recipe runtime")
    parser.add_argument("--real-data", action="store_true", help="Read the current immutable Hero Harrier mixture")
    parser.add_argument("--data-output-root", help="Caller-owned root for resolving the real-data context")
    parser.add_argument(
        "--data-schedule-steps", type=int, help="Hero mixture schedule horizon, independent of trial steps"
    )
    parser.add_argument(
        "--main-hero-recipe",
        action="store_true",
        help="Use current main Hero model, FP32 params, and compute-scaled MuonH",
    )
    parser.add_argument("--attention-implementation", choices=("gpu_fa4_cute", "gpu_fa4_cute_sm100"))
    parser.add_argument(
        "--moe-implementation",
        choices=("ragged_all_to_all", "fixed_pooled_wave_all_to_all"),
        help="Explicit transport adapter for the main recipe on different hardware",
    )
    parser.add_argument("--diagnostic-layers", type=int, help="Use fewer full-width hero layers for fault isolation")
    parser.add_argument(
        "--schedule",
        type=AutomaticPipelineSchedule,
        choices=list(AutomaticPipelineSchedule),
        default=AutomaticPipelineSchedule.STANDARD_1F1B,
    )
    parser.add_argument("--physical-stages", type=int)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--expert-axis-size", type=int, default=1)
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--expert-waves", type=int, default=3)
    parser.add_argument("--offload-opt-state", action="store_true")
    parser.add_argument("--offload-activations", action="store_true")
    parser.add_argument(
        "--park-state-during-warmup",
        action="store_true",
        help="Park real device state on pinned host during disposable warmup, then restore it",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--stop-after-step", type=int, help="End a bounded run before the configured total steps")
    parser.add_argument("--checkpoint-root", help="Restore from and save under this checkpoint directory")
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument(
        "--synchronize-devices-after-step",
        action="store_true",
        help="Diagnostic CUDA synchronization on every local device after each optimizer step",
    )
    parser.add_argument("--optimizer", choices=("adamw", "muonh"), default="adamw")
    parser.add_argument(
        "--expert-normalization",
        type=ExpertNormalization,
        choices=list(ExpertNormalization),
        default=ExpertNormalization.ALL_EXPERTS,
        help="MuonH norm across a layer's expert bank or separately for each expert",
    )
    parser.add_argument(
        "--qb-bias-mode",
        type=QbBiasMode,
        choices=list(QbBiasMode),
        default=QbBiasMode.ADAPTIVE,
        help="Freeze the current QB bias while continuing expert and router-weight training",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", help="Optional rank-zero W&B console capture in marin-community/marin_moe")
    parser.add_argument("--compilation-cache", default="/tmp/hero-pipeline-jax-cache")
    args = parser.parse_args(argv)
    if args.main_hero_recipe and args.optimizer != "muonh":
        parser.error("main-hero-recipe requires --optimizer muonh")
    if args.main_hero_recipe and (args.processes_per_task is None or args.processes_per_task < 1):
        parser.error("main-hero-recipe requires a positive processes-per-task count")
    if args.diagnostic_layers is not None and (
        not (args.full_hero or args.main_hero_recipe) or args.diagnostic_layers < 1
    ):
        parser.error("diagnostic-layers requires full-hero and a positive layer count")
    if args.sequence_length is not None and args.sequence_length < 1:
        parser.error("sequence-length must be positive")
    if args.real_data and (not args.main_hero_recipe or not args.data_output_root):
        parser.error("real-data requires main-hero-recipe and data-output-root")
    if args.real_data and (args.data_schedule_steps is None or args.data_schedule_steps < args.steps):
        parser.error("real-data requires data-schedule-steps at least as large as steps")
    if args.moe_implementation is not None and not args.main_hero_recipe:
        parser.error("moe-implementation requires main-hero-recipe")
    if args.expert_waves < 1:
        parser.error("expert-waves must be positive")
    if args.steps < 1 or args.expert_axis_size < 1:
        parser.error("steps and expert-axis-size must be positive")
    if args.stop_after_step is not None and not 1 <= args.stop_after_step <= args.steps:
        parser.error("stop-after-step must be between 1 and steps")
    if args.checkpoint_every_steps < 0 or (args.checkpoint_every_steps and not args.checkpoint_root):
        parser.error("checkpoint-every-steps must be nonnegative and requires checkpoint-root when set")

    if (args.schedule == AutomaticPipelineSchedule.DUALPIPE_V) != (args.physical_stages is not None):
        raise ValueError("DualPipeV requires physical-stages; other schedules use one logical stage per physical stage")
    return args
