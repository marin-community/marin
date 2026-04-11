# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# nodryrun because vLLM is not installed by default

"""Smoke launcher for manifest-backed single-turn OpenReward RL on Qwen3 8B."""

import argparse
import datetime
import logging
import os
from collections.abc import Sequence

from experiments.models import qwen3_8b as qwen3_8b_checkpoint
from levanter.models.qwen import Qwen3Config
from marin.execution.executor import executor_main
from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
from marin.rl.environments import EnvConfig
from marin.rl.rl_experiment_utils import (
    ModelConfig,
    RLExperimentConfig,
    config_class_path,
    executor_main_config_for_rl_experiment,
    make_rl_step,
)
from marin.rl.rl_losses import RLOOLoss
from marin.training.run_environment import resolve_required_env_vars

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-8B"
PROJECT_NAME = "marin_openreward"
DEFAULT_EXPERIMENT_NAME_SUFFIX = "openreward-smoke"
DEFAULT_NUM_TRAIN_STEPS = 50
DEFAULT_N_PROMPTS = 8
DEFAULT_NUM_ROLLOUT_WORKERS = 1
DEFAULT_EVAL_FREQUENCY = 1
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_WORKER_RAM = "400g"
DEFAULT_MAX_INPUT_TOKENS = 4096
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_EVAL_N_EXAMPLES = 8

QWEN3_8B_OPENREWARD = ModelConfig(
    name=MODEL_NAME,
    type="qwen",
    artifact=qwen3_8b_checkpoint,
    config_class_path=config_class_path(Qwen3Config),
    pip_dependency_groups=["rl"],
)


def _default_rl_loss() -> RLOOLoss:
    return RLOOLoss(
        kl_coef=0.0,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.28,
        synchronous=True,
        do_trainer_inference_mismatch_importance_sampling=True,
        tis_importance_sampling_ratio_max=2.0,
        do_overlong_filtering=True,
        vocab_tile_size=151936,
    )


def required_env_var_names(env_var_names: Sequence[str]) -> list[str]:
    """Validate that env vars are present and return de-duplicated names."""
    resolved = resolve_required_env_vars(env_var_names)
    return list(resolved)


def build_openreward_curriculum(
    run_id: str,
    config: RLExperimentConfig,
    *,
    train_manifest_path: str,
    eval_manifest_path: str | None,
    base_url: str | None,
    variant: str | None,
    api_key_env_var: str | None,
    secret_env_vars: Sequence[str],
    eval_frequency: int,
) -> CurriculumConfig:
    sampling_params = SamplingParams(
        temperature=1.0,
        n_prompts=config.n_prompts,
        n_generations_per_prompt=config.n_generations_per_prompt,
        max_output_tokens=config.max_output_tokens,
        top_k=config.inference_top_k,
        stop_tokens=None,
    )
    resolved_eval_manifest_path = eval_manifest_path or train_manifest_path

    return CurriculumConfig(
        lessons={
            "openreward": LessonConfig(
                lesson_id="openreward",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.openreward_env.OpenRewardEnv",
                    env_args={
                        "train_manifest_path": train_manifest_path,
                        "eval_manifest_path": resolved_eval_manifest_path,
                        "base_url": base_url,
                        "api_key_env_var": api_key_env_var,
                        "variant": variant,
                        "secret_env_vars": list(secret_env_vars) or None,
                    },
                ),
                dependencies=[],
                sampling_params=sampling_params,
            ),
        },
        eval_frequency=eval_frequency,
        micro_eval_frequency=None,
        actor_name=f"curriculum-{run_id}",
        eval_n_examples=DEFAULT_EVAL_N_EXAMPLES,
        max_seq_len=config.max_input_tokens + config.max_output_tokens,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True, help="Path to the prepared OpenReward train manifest JSON.")
    parser.add_argument(
        "--eval-manifest",
        default=None,
        help="Optional path to a prepared OpenReward eval manifest JSON. Defaults to --train-manifest for smoke runs.",
    )
    parser.add_argument(
        "--tool-secret-env",
        action="append",
        default=[],
        help="Environment variable name to forward into OpenReward tool session secrets. Repeat as needed.",
    )
    parser.add_argument("--openreward-base-url", default=None, help="Override the OpenReward API base URL.")
    parser.add_argument("--openreward-variant", default=None, help="Optional OpenReward environment variant.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Stable logical run name for checkpoint and W&B resume. Defaults to a timestamped smoke name.",
    )
    parser.add_argument(
        "--experiment-name-suffix",
        default=DEFAULT_EXPERIMENT_NAME_SUFFIX,
        help="Suffix used in the executor step name, checkpoint path, and W&B runs.",
    )
    parser.add_argument("--project-name", default=PROJECT_NAME, help="W&B project name for trainer and rollout runs.")
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
        "--eval-frequency",
        type=int,
        default=DEFAULT_EVAL_FREQUENCY,
        help="Full-eval cadence in completed trainer steps.",
    )
    parser.add_argument(
        "--num-rollout-workers",
        type=int,
        default=DEFAULT_NUM_ROLLOUT_WORKERS,
        help="Number of rollout worker jobs to launch.",
    )
    parser.add_argument("--train-tpu-type", default=DEFAULT_TPU_TYPE, help="TPU type for the trainer job.")
    parser.add_argument("--inference-tpu-type", default=DEFAULT_TPU_TYPE, help="TPU type for rollout worker jobs.")
    parser.add_argument(
        "--train-ram",
        default=DEFAULT_TPU_WORKER_RAM,
        help="Host RAM request for the trainer TPU job.",
    )
    parser.add_argument(
        "--inference-ram",
        default=DEFAULT_TPU_WORKER_RAM,
        help="Host RAM request for each rollout TPU job.",
    )
    parser.add_argument("--zone", default=None, help="Optional concrete zone for trainer and rollout TPU jobs.")
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help="Maximum prompt tokens to admit into the rollout worker.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum generated tokens per rollout response.",
    )
    return parser.parse_args()


def build_experiment_config(args: argparse.Namespace, *, runtime_env_vars: Sequence[str] = ()) -> RLExperimentConfig:
    tags = [
        "rl",
        "openreward",
        "smoke",
        args.experiment_name_suffix,
        QWEN3_8B_OPENREWARD.safe_name,
    ]

    return RLExperimentConfig(
        model_config=QWEN3_8B_OPENREWARD,
        rl_loss=_default_rl_loss(),
        experiment_name_suffix=args.experiment_name_suffix,
        project_name=args.project_name,
        tags=tags,
        num_train_steps=args.num_train_steps,
        train_batch_size=256,
        per_device_parallelism=4,
        learning_rate=2e-6,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        n_prompts=args.n_prompts,
        n_generations_per_prompt=4,
        num_rollout_workers=args.num_rollout_workers,
        train_tpu_type=args.train_tpu_type,
        inference_tpu_type=args.inference_tpu_type,
        train_ram=args.train_ram,
        inference_ram=args.inference_ram,
        zone=args.zone,
        inflight_weight_updates=False,
        max_rollout_step_delay=0,
        weight_transfer_sync_interval_steps=1,
        max_weight_transfer_wait_time=0,
        runtime_env_vars=list(runtime_env_vars),
    )


def build_run_name(config: RLExperimentConfig, explicit_run_name: str | None) -> str:
    if explicit_run_name is not None:
        return explicit_run_name

    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_base_name = config.model_config.name.split("/")[-1].lower()
    return f"{model_base_name}-{config.experiment_name_suffix}-{datestamp}"


def main() -> None:
    if os.getenv("CI") is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    runtime_env_vars = required_env_var_names(["OPENREWARD_API_KEY", *args.tool_secret_env])
    experiment_config = build_experiment_config(args, runtime_env_vars=runtime_env_vars)
    run_name = build_run_name(experiment_config, args.run_name)
    curriculum = build_openreward_curriculum(
        run_name,
        experiment_config,
        train_manifest_path=args.train_manifest,
        eval_manifest_path=args.eval_manifest,
        base_url=args.openreward_base_url,
        variant=args.openreward_variant,
        api_key_env_var="OPENREWARD_API_KEY",
        secret_env_vars=args.tool_secret_env,
        eval_frequency=args.eval_frequency,
    )
    step = make_rl_step(
        name=run_name,
        config=experiment_config,
        curriculum=curriculum,
    )
    executor_config = executor_main_config_for_rl_experiment(experiment_config)

    logger.info(
        "Launching OpenReward smoke run %s (train_manifest=%s, eval_manifest=%s, variant=%s, "
        "rollout_workers=%d, executor_prefix=%s)",
        run_name,
        args.train_manifest,
        args.eval_manifest or args.train_manifest,
        args.openreward_variant,
        experiment_config.num_rollout_workers,
        executor_config.prefix,
    )

    executor_main(
        executor_config,
        steps=[step],
        description="Executor-backed Iris OpenReward smoke run for Qwen3 8B",
    )


if __name__ == "__main__":
    main()
