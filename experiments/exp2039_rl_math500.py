# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# nodryrun because vLLM is not installed by default

from __future__ import annotations

import argparse
import dataclasses
import datetime
import logging
import math
import os
import shlex
import sys
from dataclasses import dataclass

from iris.marin_fs import check_path_in_region
from levanter.models.llama import LlamaConfig
from marin.execution.executor import executor_main
from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
from marin.rl.environments import EnvConfig
from marin.rl.environments.inference_ctx import vLLMInferenceContextConfig
from marin.rl.rl_job import RLJob
from marin.rl.rollout_worker import RolloutWorker
from marin.rl.train_worker import TrainWorker
from marin.rl.weight_transfer import WeightTransferMode
from marin.rl.rl_losses import RLOOLoss

logger = logging.getLogger(__name__)

TARGET_CONCURRENCY_PER_SAMPLER = 256

# Default seed for reproducibility. Controls:
# - MathEnv dataset sampling (which problems are selected)
# - RolloutWorker RNG (curriculum lesson sampling, vLLM token sampling)
# - TrainWorker RNG (model construction key, replay buffer sampling)
# - Levanter Trainer RNG (training key for dropout)
# Override with --seed on the CLI.
DEFAULT_SEED = 42

try:
    from marin.rl.rl_experiment_utils import (
        ModelConfig,
        RLExperimentConfig,
        make_rl_step,
    )

    _RL_UTILS_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    ModelConfig = None
    RLExperimentConfig = None
    make_rl_step = None
    _RL_UTILS_IMPORT_ERROR = err


@dataclass(frozen=True)
class DeploymentPreset:
    trainer_tpu_type: str
    sampler_tpu_type: str
    trainer_zone: str
    sampler_zone: str


DEPLOYMENT_PRESETS: dict[str, DeploymentPreset] = {
    "v5p_east5a": DeploymentPreset(
        trainer_tpu_type="v5p-8",
        sampler_tpu_type="v5p-8",
        trainer_zone="us-east5-a",
        sampler_zone="us-east5-a",
    ),
    "v5p_central1a": DeploymentPreset(
        trainer_tpu_type="v5p-8",
        sampler_tpu_type="v5p-8",
        trainer_zone="us-central1-a",
        sampler_zone="us-central1-a",
    ),
    "v6e_euw4a": DeploymentPreset(
        trainer_tpu_type="v6e-16",
        sampler_tpu_type="v6e-8",
        trainer_zone="europe-west4-a",
        sampler_zone="europe-west4-a",
    ),
    "v6e_east1d": DeploymentPreset(
        trainer_tpu_type="v6e-16",
        sampler_tpu_type="v6e-8",
        trainer_zone="us-east1-d",
        sampler_zone="us-east1-d",
    ),
}


if ModelConfig is not None:
    llama_3_1_8b = ModelConfig(
        name="meta-llama/Llama-3.1-8B-Instruct",
        type="llama",
        tokenizer="meta-llama/Llama-3.1-8B-Instruct",
        checkpoint="meta-llama/Llama-3.1-8B-Instruct",
        config_class=LlamaConfig,
    )
else:
    llama_3_1_8b = None


def _require_rl_utils() -> None:
    if _RL_UTILS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Running this mode requires the vLLM dependency group. " "Install the project extras that provide `vllm`."
        ) from _RL_UTILS_IMPORT_ERROR


def create_math_curriculum(
    run_id: str, experiment_config: RLExperimentConfig, seed: int = DEFAULT_SEED
) -> CurriculumConfig:
    """Create progressive math curriculum: comparison -> easy -> medium -> hard."""

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
        ),
    }

    return CurriculumConfig(
        lessons=lessons,
        eval_frequency=10,
        micro_eval_frequency=9999999,  # Effectively disable micro-eval
        actor_name=f"curriculum-{run_id}",
        eval_n_examples=500,  # for math500
        max_seq_len=experiment_config.max_input_tokens + experiment_config.max_output_tokens,
    )


def _default_experiment_config() -> RLExperimentConfig:
    _require_rl_utils()
    assert llama_3_1_8b is not None
    assert RLExperimentConfig is not None
    return RLExperimentConfig(
        model_config=llama_3_1_8b,
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
        experiment_name_suffix="math-lr=2e-6-bs=1024",
        train_batch_size=1024,
        per_device_parallelism=16,
        learning_rate=2e-6,
        max_input_tokens=1024,
        max_output_tokens=1024,
        n_prompts=64,
        n_generations_per_prompt=16,
        inflight_weight_updates=True,
        max_rollout_step_delay=1,
    )


def _model_base_name(experiment_config: RLExperimentConfig) -> str:
    model_base_name = experiment_config.model_config.name.split("/")[-1].lower()
    return model_base_name.replace("-instruct", "i")


def _default_run_name(experiment_config: RLExperimentConfig) -> str:
    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{_model_base_name(experiment_config)}-{experiment_config.experiment_name_suffix}-{datestamp}"


def _zone_to_region(zone: str) -> str:
    return zone.rsplit("-", 1)[0]


def _resolve_preset(args: argparse.Namespace) -> DeploymentPreset:
    preset = DEPLOYMENT_PRESETS[args.deployment_preset]
    return DeploymentPreset(
        trainer_tpu_type=args.trainer_tpu_type or preset.trainer_tpu_type,
        sampler_tpu_type=args.sampler_tpu_type or preset.sampler_tpu_type,
        trainer_zone=args.trainer_zone or preset.trainer_zone,
        sampler_zone=args.sampler_zone or preset.sampler_zone,
    )


def _resolve_rollout_shape_config(base_config: RLExperimentConfig, rollout_shape: str) -> RLExperimentConfig:
    if rollout_shape == "exp2039":
        return base_config

    n_generations = base_config.n_generations_per_prompt
    n_prompts = max(1, math.ceil(TARGET_CONCURRENCY_PER_SAMPLER / n_generations))
    return dataclasses.replace(base_config, n_prompts=n_prompts)


def _ensure_shared_root_region(shared_root: str, deployment: DeploymentPreset) -> None:
    if not shared_root.startswith("gs://"):
        raise ValueError(
            "Manual no-Ray mode requires --shared-root to be a gs:// path so trainer and sampler can share state."
        )

    trainer_region = _zone_to_region(deployment.trainer_zone)
    sampler_region = _zone_to_region(deployment.sampler_zone)
    if trainer_region != sampler_region:
        raise ValueError(
            f"Trainer zone {deployment.trainer_zone} and sampler zone "
            f"{deployment.sampler_zone} are in different regions."
        )

    check_path_in_region("shared_root", shared_root, trainer_region, local_ok=False)


def _resolve_bootstrap_checkpoint_path(initial_checkpoint: str | None, explicit_path: str | None) -> str | None:
    if explicit_path:
        return explicit_path
    if initial_checkpoint and (initial_checkpoint.startswith("gs://") or initial_checkpoint.startswith("s3://")):
        return initial_checkpoint
    return None


def _build_manual_job_config(
    run_id: str,
    experiment_config: RLExperimentConfig,
    curriculum: CurriculumConfig,
    shared_root: str,
    deployment: DeploymentPreset,
    bootstrap_checkpoint_path: str | None,
):
    _require_rl_utils()
    assert make_rl_step is not None
    step = make_rl_step(
        name=run_id,
        config=experiment_config,
        curriculum=curriculum,
    )
    job_config = step.config

    run_root = f"{shared_root.rstrip('/')}/{run_id}"
    checkpointer = dataclasses.replace(
        job_config.trainer.checkpointer,
        base_path=f"{run_root}/checkpoints",
    )
    trainer = dataclasses.replace(job_config.trainer, checkpointer=checkpointer)

    rollout_storage = dataclasses.replace(
        job_config.rollout_storage,
        path=f"{run_root}/rollouts",
    )
    weight_transfer = dataclasses.replace(
        job_config.weight_transfer,
        mode=WeightTransferMode.ARROW_FLIGHT,
        coordinator_backend="filesystem",
        coordinator_metadata_path=f"{run_root}/weight_transfer/arrow_flight_coordinator.json",
    )

    assert isinstance(job_config.inference_config, vLLMInferenceContextConfig)
    inference_config = dataclasses.replace(
        job_config.inference_config,
        enable_fast_bootstrap=bootstrap_checkpoint_path is not None,
        bootstrap_checkpoint_path=bootstrap_checkpoint_path,
    )

    run_config = dataclasses.replace(
        job_config.run_config,
        train_tpu_type=deployment.trainer_tpu_type,
        inference_tpu_type=deployment.sampler_tpu_type,
        num_rollout_workers=1,
    )

    return dataclasses.replace(
        job_config,
        run_id=run_id,
        trainer=trainer,
        rollout_storage=rollout_storage,
        weight_transfer=weight_transfer,
        inference_config=inference_config,
        run_config=run_config,
    )


def _launch_command_for_role(
    role: str,
    run_id: str,
    shared_root: str,
    deployment_preset: str,
    rollout_shape: str,
    deployment: DeploymentPreset,
    bootstrap_checkpoint_path: str | None,
    seed: int = DEFAULT_SEED,
) -> str:
    if role == "trainer":
        tpu_name = f"{run_id}-trainer"
        tpu_type = deployment.trainer_tpu_type
        zone = deployment.trainer_zone
    elif role == "sampler":
        tpu_name = f"{run_id}-sampler"
        tpu_type = deployment.sampler_tpu_type
        zone = deployment.sampler_zone
    else:
        raise ValueError(f"Unknown role {role}")

    cmd = [
        "uv",
        "run",
        "python",
        "lib/levanter/infra/launch.py",
        "--foreground",
        f"--tpu_name={tpu_name}",
        f"--tpu_type={tpu_type}",
        f"--zone={zone}",
        "-e",
        "TPU_BACKEND_TYPE",
        "jax",
        "-e",
        "PJRT_DEVICE",
        "TPU",
        "-e",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "0",
        "--",
        "uv",
        "run",
        "python",
        "experiments/exp2039_rl_math500.py",
        "--mode",
        role,
        "--deployment-preset",
        deployment_preset,
        "--trainer-zone",
        deployment.trainer_zone,
        "--sampler-zone",
        deployment.sampler_zone,
        "--trainer-tpu-type",
        deployment.trainer_tpu_type,
        "--sampler-tpu-type",
        deployment.sampler_tpu_type,
        "--run-id",
        run_id,
        "--shared-root",
        shared_root,
        "--rollout-shape",
        rollout_shape,
        "--seed",
        str(seed),
    ]
    if bootstrap_checkpoint_path:
        cmd.extend(["--bootstrap-checkpoint-path", bootstrap_checkpoint_path])
    return shlex.join(cmd)


def _run_manual_mode(args: argparse.Namespace):
    if not args.run_id:
        raise ValueError("--run-id is required for manual modes: trainer, sampler, launch-plan")
    if not args.shared_root:
        raise ValueError("--shared-root is required for manual modes: trainer, sampler, launch-plan")

    deployment = _resolve_preset(args)
    _ensure_shared_root_region(args.shared_root, deployment)

    if args.mode == "launch-plan":
        bootstrap_checkpoint_path = args.bootstrap_checkpoint_path
        trainer_cmd = _launch_command_for_role(
            role="trainer",
            run_id=args.run_id,
            shared_root=args.shared_root,
            deployment_preset=args.deployment_preset,
            rollout_shape=args.rollout_shape,
            deployment=deployment,
            bootstrap_checkpoint_path=bootstrap_checkpoint_path,
            seed=args.seed,
        )
        sampler_cmd = _launch_command_for_role(
            role="sampler",
            run_id=args.run_id,
            shared_root=args.shared_root,
            deployment_preset=args.deployment_preset,
            rollout_shape=args.rollout_shape,
            deployment=deployment,
            bootstrap_checkpoint_path=bootstrap_checkpoint_path,
            seed=args.seed,
        )
        print("# Trainer launch command")
        print(trainer_cmd)
        print("")
        print("# Sampler launch command")
        print(sampler_cmd)
        return

    seed = args.seed
    experiment_config = _resolve_rollout_shape_config(_default_experiment_config(), args.rollout_shape)
    bootstrap_checkpoint_path = _resolve_bootstrap_checkpoint_path(
        experiment_config.model_config.checkpoint,
        args.bootstrap_checkpoint_path,
    )
    curriculum = create_math_curriculum(args.run_id, experiment_config, seed=seed)
    job_config = _build_manual_job_config(
        run_id=args.run_id,
        experiment_config=experiment_config,
        curriculum=curriculum,
        shared_root=args.shared_root,
        deployment=deployment,
        bootstrap_checkpoint_path=bootstrap_checkpoint_path,
    )

    trainer_overrides = {"seed": seed}
    if args.num_train_steps is not None:
        trainer_overrides["num_train_steps"] = args.num_train_steps
    job_config = dataclasses.replace(
        job_config,
        seed=seed,
        trainer=dataclasses.replace(job_config.trainer, **trainer_overrides),
    )

    train_worker_config, rollout_worker_config = RLJob(job_config).to_worker_configs()

    try:
        if args.mode == "trainer":
            TrainWorker(config=train_worker_config).train()
        elif args.mode == "sampler":
            # levanter.initialize() (called by TrainWorker) sets up logging,
            # but the sampler skips it to avoid JAX init deadlocks with vLLM.
            # Configure the marin logger so diagnostic PHASE logs are visible.
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", force=True)
            RolloutWorker(config=rollout_worker_config).run()
        else:
            raise ValueError(f"Unsupported manual mode: {args.mode}")
    except Exception:
        logger.exception("FATAL: %s worker crashed", args.mode.upper())
        sys.exit(1)


def _run_executor_mode():
    _require_rl_utils()
    assert make_rl_step is not None
    experiment_config = _default_experiment_config()
    name = _default_run_name(experiment_config)
    curriculum = create_math_curriculum(name, experiment_config)
    experiments = [
        make_rl_step(
            name=name,
            config=experiment_config,
            curriculum=curriculum,
        )
    ]
    executor_main(
        steps=experiments,
        description="Async RL math training experiments",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="exp2039 RL MATH-500")
    parser.add_argument(
        "--mode",
        choices=["executor", "trainer", "sampler", "launch-plan"],
        default="executor",
        help="executor keeps existing cluster-based flow; trainer/sampler/launch-plan use no-Ray manual mode.",
    )
    parser.add_argument(
        "--deployment-preset",
        choices=sorted(DEPLOYMENT_PRESETS.keys()),
        default="v5p_east5a",
        help="Trainer/sampler TPU shape and zone preset.",
    )
    parser.add_argument("--run-id", default=None, help="Manual mode run id shared by trainer and sampler.")
    parser.add_argument("--shared-root", default=None, help="Shared gs:// root for checkpoints, rollouts and metadata.")
    parser.add_argument("--trainer-zone", default=None)
    parser.add_argument("--sampler-zone", default=None)
    parser.add_argument("--trainer-tpu-type", default=None)
    parser.add_argument("--sampler-tpu-type", default=None)
    parser.add_argument(
        "--rollout-shape",
        choices=["auto", "exp2039"],
        default="auto",
        help="auto targets ~256 concurrent completions per sampler replica.",
    )
    parser.add_argument(
        "--bootstrap-checkpoint-path",
        default=None,
        help="Optional gs:// checkpoint path for fast vLLM bootstrap.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=None,
        help="Override number of training steps (default: use experiment config).",
    )
    return parser.parse_args()


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    args = _parse_args()
    if args.mode == "executor":
        _run_executor_mode()
    else:
        _run_manual_mode(args)


if __name__ == "__main__":
    main()
