# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# nodryrun because vLLM is not installed by default

"""Regression probe: executor + small GPU-only RL launch."""

import argparse
import datetime
import logging
import os

from levanter.models.llama import LlamaConfig
from marin.execution.executor import executor_main

from experiments.iris_rl_gpu_smoke import (
    CANONICAL_MODEL_NAME,
    DEFAULT_EXPERIMENT_REGION,
    DEFAULT_GPU_COUNT,
    DEFAULT_GPU_TYPE,
    DEFAULT_NUM_TRAIN_STEPS,
    gpu_smoke_curriculum,
    gpu_smoke_resources,
    gpu_smoke_rollout_count,
    resolve_gpu_smoke_model_path,
    gpu_smoke_train_batch_size,
)
from marin.rl.rl_experiment_utils import (
    ModelConfig,
    RLExperimentConfig,
    config_class_path,
    executor_main_config_for_rl_experiment,
    make_rl_step,
)
from marin.rl.rl_losses import RLOOLoss

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_SUFFIX = "exec-gcs-small-gpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-name-suffix",
        default=DEFAULT_EXPERIMENT_SUFFIX,
        help="Run-name suffix used for job and W&B labeling.",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=DEFAULT_NUM_TRAIN_STEPS,
        help="Number of RL training steps to execute.",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_EXPERIMENT_REGION,
        help="Region for trainer and rollout GPU jobs.",
    )
    parser.add_argument(
        "--gpu-type",
        default=DEFAULT_GPU_TYPE,
        help="GPU type shared by trainer and rollout workers.",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=DEFAULT_GPU_COUNT,
        help="Number of GPUs per trainer and rollout worker host.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Model artifact path or Hugging Face model id. Defaults to the canonical cached artifact in --region.",
    )
    return parser.parse_args()


def build_model_config(model_path: str) -> ModelConfig:
    return ModelConfig(
        name=CANONICAL_MODEL_NAME,
        type="llama",
        artifact=model_path,
        config_class_path=config_class_path(LlamaConfig),
    )


def build_debug_config(
    *,
    experiment_name_suffix: str,
    num_train_steps: int,
    region: str,
    gpu_type: str,
    gpu_count: int,
    model_path: str | None,
) -> RLExperimentConfig:
    resolved_model_path = resolve_gpu_smoke_model_path(region=region, model_path=model_path)
    tags = ["rl", "iris-debug", "regression", "gpu", experiment_name_suffix]
    return RLExperimentConfig(
        model_config=build_model_config(resolved_model_path),
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
        experiment_name_suffix=experiment_name_suffix,
        project_name="marin_iris_rl_debug",
        tags=tags,
        num_train_steps=num_train_steps,
        train_batch_size=gpu_smoke_train_batch_size(gpu_count),
        per_device_parallelism=1,
        learning_rate=2e-6,
        max_input_tokens=256,
        max_output_tokens=256,
        n_prompts=1,
        n_generations_per_prompt=gpu_smoke_rollout_count(gpu_count),
        num_rollout_workers=1,
        train_resources=gpu_smoke_resources(region=region, gpu_type=gpu_type, gpu_count=gpu_count),
        rollout_resources=gpu_smoke_resources(region=region, gpu_type=gpu_type, gpu_count=gpu_count),
        inference_tensor_parallel_size=gpu_count,
        weight_transfer_sync_interval_steps=1,
        max_weight_transfer_wait_time=300,
        inflight_weight_updates=False,
    )


def main() -> None:
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    args = parse_args()
    debug_config = build_debug_config(
        experiment_name_suffix=args.experiment_name_suffix,
        num_train_steps=args.num_train_steps,
        region=args.region,
        gpu_type=args.gpu_type,
        gpu_count=args.gpu_count,
        model_path=args.model_path,
    )

    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{args.experiment_name_suffix}-{datestamp}"
    curriculum = gpu_smoke_curriculum(
        run_id=name,
        max_input_tokens=debug_config.max_input_tokens,
        max_output_tokens=debug_config.max_output_tokens,
        num_generations=debug_config.n_generations_per_prompt,
    )
    step = make_rl_step(
        name=name,
        config=debug_config,
        curriculum=curriculum,
    )

    executor_main(
        executor_main_config_for_rl_experiment(debug_config),
        steps=[step],
        description=(f"Iris RL regression probe: executor + small GPU-only ({args.num_train_steps} training steps)"),
    )


if __name__ == "__main__":
    main()
