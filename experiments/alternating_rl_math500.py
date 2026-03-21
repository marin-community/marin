# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# nodryrun because vLLM is not installed by default

from __future__ import annotations

import argparse
import dataclasses
import logging
import os

from levanter.models.llama import LlamaConfig

from marin.rl.alternating import (
    AlternatingClusterConfig,
    AlternatingPhaseQuotaConfig,
    AlternatingRLConfig,
    AlternatingRunPaths,
    ExistingPodPhaseHooks,
    resolve_container_image,
    run_controller,
    run_export_policy_from_config_path,
    run_materialization_from_config_path,
    run_prepare_sampling_from_config_path,
    run_sampling_host_from_config_path,
    run_training_phase_from_config_path,
    save_controller_config,
)
from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
from marin.rl.environments import EnvConfig
from marin.rl.environments.inference_ctx import vLLMInferenceContextConfig
from marin.rl.rl_losses import RLOOLoss

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_EVAL_EXAMPLES_PER_LESSON = 500
ALTERNATING_WANDB_PROJECT = "alternate_rl"

try:
    from marin.rl.rl_experiment_utils import ModelConfig, RLExperimentConfig, make_rl_step

    _RL_UTILS_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    ModelConfig = None
    RLExperimentConfig = None
    make_rl_step = None
    _RL_UTILS_IMPORT_ERROR = err


if ModelConfig is not None:
    LLAMA_3_1_8B = ModelConfig(
        name="meta-llama/Llama-3.1-8B-Instruct",
        type="llama",
        tokenizer="meta-llama/Llama-3.1-8B-Instruct",
        checkpoint="meta-llama/Llama-3.1-8B-Instruct",
        config_class=LlamaConfig,
    )
else:
    LLAMA_3_1_8B = None


def _require_rl_utils() -> None:
    if _RL_UTILS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Running alternating RL requires the vLLM dependency group. "
            "Install the project extras that provide `vllm`."
        ) from _RL_UTILS_IMPORT_ERROR


def create_math_curriculum(
    run_id: str,
    experiment_config: RLExperimentConfig,
    *,
    seed: int,
    eval_n_examples: int,
) -> CurriculumConfig:
    """Create the MATH-500 curriculum used by alternating RL."""
    default_sampling = SamplingParams(
        temperature=1.0,
        n_prompts=experiment_config.n_prompts,
        n_generations_per_prompt=experiment_config.n_generations_per_prompt,
        max_output_tokens=experiment_config.max_output_tokens,
        top_k=4096,
        stop_tokens=None,
    )

    lessons = {
        "math_full": LessonConfig(
            lesson_id="math_full",
            env_config=EnvConfig(
                env_class="marin.rl.environments.math_env.MathEnv",
                env_args={"seed": seed},
            ),
            dependencies=[],
            sampling_params=default_sampling,
        )
    }

    return CurriculumConfig(
        lessons=lessons,
        eval_frequency=10,
        micro_eval_frequency=9_999_999,
        actor_name=f"alternating-curriculum-{run_id}",
        eval_n_examples=eval_n_examples,
        max_seq_len=experiment_config.max_input_tokens + experiment_config.max_output_tokens,
    )


def _default_experiment_config() -> RLExperimentConfig:
    _require_rl_utils()
    assert LLAMA_3_1_8B is not None
    assert RLExperimentConfig is not None
    return RLExperimentConfig(
        model_config=LLAMA_3_1_8B,
        rl_loss=RLOOLoss(
            kl_coef=0.0,
            clip_epsilon_low=0.2,
            clip_epsilon_high=0.28,
            synchronous=True,
            do_trainer_inference_mismatch_importance_sampling=True,
            tis_importance_sampling_ratio_max=2.0,
            do_overlong_filtering=True,
            vocab_tile_size=32064,
        ),
        experiment_name_suffix="alternating-math500",
        project_name=ALTERNATING_WANDB_PROJECT,
        train_batch_size=1024,
        per_device_parallelism=16,
        learning_rate=2e-6,
        max_input_tokens=1024,
        max_output_tokens=1024,
        n_prompts=64,
        n_generations_per_prompt=16,
        replay_buffer_max_samples=1,
        inflight_weight_updates=False,
        max_rollout_step_delay=0,
    )


def _experiment_config_from_args(args: argparse.Namespace) -> RLExperimentConfig:
    experiment_config = _default_experiment_config()
    overrides: dict[str, int | float] = {}
    if args.train_batch_size is not None:
        overrides["train_batch_size"] = args.train_batch_size
    if args.max_input_tokens is not None:
        overrides["max_input_tokens"] = args.max_input_tokens
    if args.max_output_tokens is not None:
        overrides["max_output_tokens"] = args.max_output_tokens
    if args.n_prompts is not None:
        overrides["n_prompts"] = args.n_prompts
    if args.n_generations_per_prompt is not None:
        overrides["n_generations_per_prompt"] = args.n_generations_per_prompt
    if args.inference_gpu_memory_utilization is not None:
        overrides["inference_gpu_memory_utilization"] = args.inference_gpu_memory_utilization
    if not overrides:
        return experiment_config
    return dataclasses.replace(experiment_config, **overrides)


def _groups_per_training_step(config: RLExperimentConfig) -> int:
    if config.train_batch_size % config.n_generations_per_prompt != 0:
        raise ValueError(
            "train_batch_size must be divisible by n_generations_per_prompt "
            "so alternating RL can derive rollout-group quotas: "
            f"{config.train_batch_size} % {config.n_generations_per_prompt} != 0"
        )
    return config.train_batch_size // config.n_generations_per_prompt


def _pass_through_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in (
        "HF_TOKEN",
        "WANDB_API_KEY",
        "WANDB_MODE",
        "WANDB_BASE_URL",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def _build_controller_config(args: argparse.Namespace) -> AlternatingRLConfig:
    experiment_config = _experiment_config_from_args(args)
    eval_n_examples = (
        args.eval_examples_per_lesson if args.eval_examples_per_lesson is not None else DEFAULT_EVAL_EXAMPLES_PER_LESSON
    )
    curriculum = create_math_curriculum(
        args.run_id,
        experiment_config,
        seed=args.seed,
        eval_n_examples=eval_n_examples,
    )
    assert make_rl_step is not None
    step = make_rl_step(args.run_id, experiment_config, curriculum)
    job_config = step.config
    if not isinstance(job_config.inference_config, vLLMInferenceContextConfig):
        raise TypeError("alternating math500 expects a vLLM inference config")

    run_root = f"{args.shared_root.rstrip('/')}/{args.run_id}"
    trainer = dataclasses.replace(
        job_config.trainer,
        checkpointer=dataclasses.replace(
            job_config.trainer.checkpointer,
            base_path=f"{run_root}/checkpoints",
        ),
    )
    image_digest = resolve_container_image(args.image)
    return AlternatingRLConfig(
        run_id=args.run_id,
        shared_root=args.shared_root,
        image_digest=image_digest,
        seed=args.seed,
        cluster=AlternatingClusterConfig(
            tpu_name=args.tpu_name,
            tpu_type=args.tpu_type,
            zone=args.zone,
            num_hosts=args.num_hosts,
            local_tensor_parallel_size=args.local_tensor_parallel_size,
            node_count=args.node_count,
            capacity_type=args.capacity_type,
            runtime_version=args.runtime_version,
        ),
        quotas=AlternatingPhaseQuotaConfig(
            steps_per_phase=args.steps_per_phase,
            num_train_steps=args.num_train_steps or experiment_config.num_train_steps,
            groups_per_training_step=_groups_per_training_step(experiment_config),
            eval_examples_per_lesson=curriculum.eval_n_examples,
        ),
        trainer=trainer,
        model=job_config.model,
        optimizer=job_config.train_params.optimizer,
        loss=job_config.train_params.rl_loss,
        curriculum=curriculum,
        inference=job_config.inference_config,
        replay_buffer=job_config.train_params.replay_buffer,
        tokenizer_name=job_config.tokenizer,
        initial_checkpoint=job_config.initial_checkpoint,
        vocab_size=job_config.vocab_size,
        env=_pass_through_env(),
    )


def _controller_mode(args: argparse.Namespace) -> None:
    config = _build_controller_config(args)
    paths = AlternatingRunPaths.from_config(config)
    save_controller_config(config, paths)
    final_state = run_controller(config, ExistingPodPhaseHooks())
    logger.info("alternating RL completed: %s", final_state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alternating single-pod RL on MATH-500")
    subparsers = parser.add_subparsers(dest="command", required=True)

    controller = subparsers.add_parser("controller", help="run the local alternating RL controller")
    controller.add_argument("--run-id", required=True)
    controller.add_argument("--shared-root", required=True)
    controller.add_argument("--image", required=True)
    controller.add_argument("--tpu-name", required=True)
    controller.add_argument("--tpu-type", required=True)
    controller.add_argument("--zone", required=True)
    controller.add_argument("--num-hosts", type=int, required=True)
    controller.add_argument("--local-tensor-parallel-size", type=int, default=4)
    controller.add_argument("--node-count", type=int, default=1)
    controller.add_argument("--capacity-type", default="on-demand")
    controller.add_argument("--runtime-version", default=None)
    controller.add_argument("--steps-per-phase", type=int, required=True)
    controller.add_argument("--num-train-steps", type=int, default=None)
    controller.add_argument("--seed", type=int, default=DEFAULT_SEED)
    controller.add_argument("--train-batch-size", type=int, default=None)
    controller.add_argument("--n-prompts", type=int, default=None)
    controller.add_argument("--n-generations-per-prompt", type=int, default=None)
    controller.add_argument("--eval-examples-per-lesson", type=int, default=None)
    controller.add_argument("--max-input-tokens", type=int, default=None)
    controller.add_argument("--max-output-tokens", type=int, default=None)
    controller.add_argument("--inference-gpu-memory-utilization", type=float, default=None)

    for subcommand in ("prepare-sampling", "materialize", "train-phase", "export-policy"):
        subparser = subparsers.add_parser(subcommand)
        subparser.add_argument("--config-path", required=True)
        subparser.add_argument("--phase-id", type=int, required=True)

    sampling_host = subparsers.add_parser("sampling-host")
    sampling_host.add_argument("--config-path", required=True)
    sampling_host.add_argument("--phase-id", type=int, required=True)
    sampling_host.add_argument("--host-ordinal", type=int, required=True)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "controller":
        _controller_mode(args)
        return
    if args.command == "prepare-sampling":
        run_prepare_sampling_from_config_path(args.config_path, args.phase_id)
        return
    if args.command == "sampling-host":
        run_sampling_host_from_config_path(args.config_path, args.phase_id, args.host_ordinal)
        return
    if args.command == "materialize":
        run_materialization_from_config_path(args.config_path, args.phase_id)
        return
    if args.command == "train-phase":
        run_training_phase_from_config_path(args.config_path, args.phase_id)
        return
    if args.command == "export-policy":
        run_export_policy_from_config_path(args.config_path, args.phase_id)
        return

    raise AssertionError(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
