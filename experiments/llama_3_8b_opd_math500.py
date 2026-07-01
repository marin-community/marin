# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# nodryrun because vLLM is not installed by default

"""Executor-backed OPD Math500 smoke launcher for Llama 3.1 8B."""

import argparse
import logging
import os

from levanter.checkpoint import CheckpointDebugConfig
from marin.execution.executor import executor_main
from marin.rl.opd_losses import OPDSampledTokenReverseKLLoss
from marin.rl.rl_experiment_utils import RLExperimentConfig, executor_main_config_for_rl_experiment, make_rl_step
from marin.rl.teacher import INITIAL_POLICY_TEACHER_CHECKPOINT, TeacherConfig

from experiments.llama_3_8b_rl_math500 import (
    DEFAULT_CHECKPOINTER_SAVE_INTERVAL,
    DEFAULT_EVAL_FREQUENCY,
    DEFAULT_EVAL_N_EXAMPLES,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TPU_TYPE,
    DEFAULT_TPU_WORKER_RAM,
    LLAMA_3_1_8B_INSTRUCT,
    PROJECT_NAME,
    build_math500_curriculum,
    build_run_name,
)

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME_SUFFIX = "opd-math500-smoke"
DEFAULT_NUM_TRAIN_STEPS = 20
DEFAULT_N_PROMPTS = 16
DEFAULT_TRAIN_BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Stable logical run name for checkpoint/W&B resume across relaunches. "
        "If omitted, a fresh timestamped name is generated.",
    )
    parser.add_argument(
        "--experiment-name-suffix",
        default=DEFAULT_EXPERIMENT_NAME_SUFFIX,
        help="Suffix used in the executor step name, checkpoint path, and W&B runs.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default=None,
        help="Teacher checkpoint or Hugging Face repo. If omitted, uses the student initial checkpoint.",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=DEFAULT_NUM_TRAIN_STEPS,
        help="Number of trainer steps to run.",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=DEFAULT_N_PROMPTS,
        help="Number of prompts sampled per rollout batch.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
        help="Trainer batch size. Keep this aligned with available smoke rollout count.",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=DEFAULT_EVAL_FREQUENCY,
        help="Full-eval cadence in completed trainer steps.",
    )
    parser.add_argument(
        "--checkpointer-save-interval",
        type=int,
        default=DEFAULT_CHECKPOINTER_SAVE_INTERVAL,
        help="Seconds between trainer checkpoint saves.",
    )
    parser.add_argument(
        "--keep-last-temporary-checkpoints",
        type=int,
        default=5,
        help="Number of complete temporary checkpoints to retain after a successful temporary checkpoint save. "
        "Use 0 to delete temporary checkpoints after they commit.",
    )
    parser.add_argument(
        "--debug-checkpointer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose trainer-side checkpoint diagnostics.",
    )
    parser.add_argument(
        "--debug-checkpointer-log-interval",
        type=float,
        default=60.0,
        help="Seconds between checkpoint progress logs when debug checkpointer is enabled.",
    )
    parser.add_argument(
        "--debug-checkpointer-dump-stacks-after",
        type=float,
        default=60.0,
        help="Dump Python thread stacks after this many seconds in one checkpoint phase.",
    )
    parser.add_argument(
        "--train-tpu-type",
        default=DEFAULT_TPU_TYPE,
        help="TPU type for the trainer job.",
    )
    parser.add_argument(
        "--inference-tpu-type",
        default=DEFAULT_TPU_TYPE,
        help="TPU type for rollout worker jobs.",
    )
    parser.add_argument(
        "--num-train-slices",
        type=int,
        default=1,
        help="Number of TPU slices for the trainer job.",
    )
    parser.add_argument(
        "--train-ram",
        default=DEFAULT_TPU_WORKER_RAM,
        help="Host RAM request for the trainer TPU job.",
    )
    parser.add_argument(
        "--inference-ram",
        default=DEFAULT_TPU_WORKER_RAM,
        help="Host RAM request for the rollout TPU job.",
    )
    parser.add_argument(
        "--zone",
        default=None,
        help="Optional concrete zone for trainer and rollout TPU jobs.",
    )
    parser.add_argument(
        "--project-name",
        default=PROJECT_NAME,
        help="W&B project name for trainer and rollout runs.",
    )
    return parser.parse_args()


def build_experiment_config(args: argparse.Namespace) -> RLExperimentConfig:
    if args.train_batch_size > args.n_prompts:
        raise ValueError("train_batch_size must be <= n_prompts for the OPD smoke run")

    teacher_checkpoint = args.teacher_checkpoint or INITIAL_POLICY_TEACHER_CHECKPOINT

    tags = [
        "rl",
        "opd",
        "sampled-token-reverse-kl",
        "math500",
        args.experiment_name_suffix,
        LLAMA_3_1_8B_INSTRUCT.safe_name,
    ]

    return RLExperimentConfig(
        model_config=LLAMA_3_1_8B_INSTRUCT,
        rl_loss=OPDSampledTokenReverseKLLoss(
            synchronous=False,
            clip_epsilon_low=None,
            clip_epsilon_high=None,
            vocab_tile_size=32064,
        ),
        teacher=TeacherConfig(checkpoint=teacher_checkpoint),
        experiment_name_suffix=args.experiment_name_suffix,
        project_name=args.project_name,
        tags=tags,
        num_train_steps=args.num_train_steps,
        checkpointer_save_interval=args.checkpointer_save_interval,
        keep_last_temporary_checkpoints=args.keep_last_temporary_checkpoints,
        checkpoint_debug=CheckpointDebugConfig(
            enabled=args.debug_checkpointer,
            log_interval=args.debug_checkpointer_log_interval,
            dump_stacks_after=args.debug_checkpointer_dump_stacks_after,
        ),
        train_batch_size=args.train_batch_size,
        per_device_parallelism=16,
        learning_rate=1e-7,
        max_input_tokens=DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        n_prompts=args.n_prompts,
        n_generations_per_prompt=1,
        replay_buffer_max_samples=1,
        num_rollout_workers=1,
        train_tpu_type=args.train_tpu_type,
        inference_tpu_type=args.inference_tpu_type,
        num_train_slices=args.num_train_slices,
        train_ram=args.train_ram,
        inference_ram=args.inference_ram,
        zone=args.zone,
        inflight_weight_updates=False,
        max_rollout_step_delay=0,
        weight_transfer_sync_interval_steps=1,
        max_weight_transfer_wait_time=600,
        inference_n=1,
    )


def main() -> None:
    if os.getenv("CI") is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    experiment_config = build_experiment_config(args)
    run_name = build_run_name(experiment_config, args.run_name)
    curriculum = build_math500_curriculum(run_name, experiment_config, args.eval_frequency)
    step = make_rl_step(
        name=run_name,
        config=experiment_config,
        curriculum=curriculum,
    )
    executor_config = executor_main_config_for_rl_experiment(experiment_config)

    logger.info(
        "Launching executor OPD Math500 run %s (teacher=%s, train_tpu=%s, inference_tpu=%s, "
        "train_ram=%s, inference_ram=%s, zone=%s, executor_prefix=%s, eval_examples=%d)",
        run_name,
        experiment_config.teacher.checkpoint if experiment_config.teacher is not None else None,
        experiment_config.train_tpu_type,
        experiment_config.inference_tpu_type,
        experiment_config.train_ram,
        experiment_config.inference_ram,
        experiment_config.zone,
        executor_config.prefix,
        DEFAULT_EVAL_N_EXAMPLES,
    )

    executor_main(
        executor_config,
        steps=[step],
        description="Executor-backed sampled-token OPD Math500 smoke run for Llama 3.1 8B",
    )


if __name__ == "__main__":
    main()
