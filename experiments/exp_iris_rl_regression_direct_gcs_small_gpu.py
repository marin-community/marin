# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression probe E5: direct + GCS + small GPU-only RL launch.

This is the first planned GPU-only smoke path:
- no executor_main wrapper
- coordinator runs directly in the outer Iris job
- trainer and rollout both use one GPU host shape
- model bootstrap defaults to the cached regional GCS artifact path
- sync vLLM only; inflight updates stay disabled
"""

import argparse
import datetime
import logging
from dataclasses import replace

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.iris_rl_gpu_smoke import (
    CANONICAL_MODEL_NAME,
    DEFAULT_EXPERIMENT_REGION,
    DEFAULT_GPU_COUNT,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_GPU_TYPE,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_NUM_ROLLOUT_WORKERS,
    DEFAULT_NUM_TRAIN_STEPS,
    gpu_smoke_curriculum,
    gpu_smoke_prefix,
    gpu_smoke_resources,
    gpu_smoke_rollout_count,
    resolve_gpu_smoke_model_path,
    gpu_smoke_train_batch_size,
)
from marin.rl.environments.inference_ctx import VLLMSamplingConfig, vLLMInferenceContextConfig
from marin.rl.orchestration import _run_rl_coordinator
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rl_experiment_utils import resolve_train_attention_backend, vllm_load_format_for_model_path
from marin.rl.rl_job import RLJobConfig, RunConfig, TrainParams
from marin.rl.rl_losses import RLOOLoss
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.rollout_worker import RolloutTrackerConfig
from marin.rl.weight_transfer import WeightTransferConfig, WeightTransferMode

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_SUFFIX = "direct-gcs-small-gpu"


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
        help="Region-local Marin GCS model artifact path. Defaults to the canonical cached artifact in --region.",
    )
    return parser.parse_args()


def build_run_config(*, region: str, gpu_type: str, gpu_count: int) -> RunConfig:
    train_resources = gpu_smoke_resources(region=region, gpu_type=gpu_type, gpu_count=gpu_count)
    rollout_resources = gpu_smoke_resources(region=region, gpu_type=gpu_type, gpu_count=gpu_count)
    return RunConfig(
        train_resources=train_resources,
        rollout_resources=rollout_resources,
        num_rollout_workers=DEFAULT_NUM_ROLLOUT_WORKERS,
    )


def build_job_config(
    *,
    name: str,
    num_train_steps: int,
    region: str,
    gpu_type: str,
    gpu_count: int,
    model_path: str | None,
) -> RLJobConfig:
    resolved_model_path = resolve_gpu_smoke_model_path(region=region, model_path=model_path)
    run_config = build_run_config(region=region, gpu_type=gpu_type, gpu_count=gpu_count)
    train_resources = run_config.train_resources
    rollout_generations = gpu_smoke_rollout_count(gpu_count)
    marin_prefix = gpu_smoke_prefix(region)

    converter = HFCheckpointConverter(
        LlamaConfig,
        reference_checkpoint=resolved_model_path,
        tokenizer=resolved_model_path,
    )
    hf_config = converter.default_hf_config
    model_config = replace(
        LlamaConfig.from_hf_config(hf_config),
        max_seq_len=DEFAULT_MAX_INPUT_TOKENS + DEFAULT_MAX_OUTPUT_TOKENS,
        tokenizer=resolved_model_path,
        attn_backend=resolve_train_attention_backend(train_resources),
    )

    curriculum = gpu_smoke_curriculum(
        run_id=name,
        max_input_tokens=DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        num_generations=rollout_generations,
    )

    return RLJobConfig(
        model=model_config,
        vocab_size=hf_config.vocab_size,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin_iris_rl_debug",
                name=name,
                tags=["rl", "iris-debug", "regression", "e5", "gpu", "direct-gcs-small"],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=gpu_smoke_train_batch_size(gpu_count),
            per_device_parallelism=1,
            num_train_steps=num_train_steps,
            steps_per_eval=100,
            checkpointer=CheckpointerConfig(
                base_path=f"{marin_prefix}/checkpoints/{name}",
                save_interval=datetime.timedelta(seconds=600),
            ),
            mesh=MeshConfig(
                axes={"context": 1, "model": 1},
                shared_mapping={"mlp": "model", "heads": "model", "position": "context"},
            ),
        ),
        train_params=TrainParams(
            optimizer=AdamConfig(learning_rate=2e-6, lr_schedule="constant"),
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
            replay_buffer=ReplayBufferConfig(capacity=4096, alpha=3.0, max_samples=1, max_rollout_step_delay=0),
        ),
        curriculum=curriculum,
        tokenizer=resolved_model_path,
        inference_type="vllm",
        inference_config=vLLMInferenceContextConfig(
            model_name=resolved_model_path,
            canonical_model_name=CANONICAL_MODEL_NAME,
            max_model_len=DEFAULT_MAX_INPUT_TOKENS + DEFAULT_MAX_OUTPUT_TOKENS,
            tensor_parallel_size=gpu_count,
            gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
            device_kind="gpu",
            sampling_params=VLLMSamplingConfig(
                temperature=1.0,
                n=rollout_generations,
                max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                stop=["<|eot_id|>"],
                include_stop_str_in_output=True,
                logprobs=1,
                top_k=4096,
            ),
            load_format=vllm_load_format_for_model_path(resolved_model_path),
        ),
        initial_checkpoint=resolved_model_path,
        rollout_storage=RolloutStorageConfig(
            storage_type=StorageType.FILE,
            path=f"{marin_prefix}/rollouts/{name}",
        ),
        weight_transfer=WeightTransferConfig(
            mode=WeightTransferMode.ARROW_FLIGHT,
            sync_interval_steps=1,
            max_weight_transfer_wait_time=300,
            coordinator_name=f"wt-coord-{name}",
        ),
        inflight_weight_updates=False,
        run_id=name,
        log_freq=1,
        run_config=run_config,
        rollout_tracker=RolloutTrackerConfig(
            project="marin_iris_rl_debug",
            name=f"{name}-rollout",
            tags=["rl", "iris-debug", "regression", "e5", "gpu", "rollout"],
        ),
        pip_dependency_groups=["vllm", "math"],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{args.experiment_name_suffix}-{datestamp}"

    job_config = build_job_config(
        name=name,
        num_train_steps=args.num_train_steps,
        region=args.region,
        gpu_type=args.gpu_type,
        gpu_count=args.gpu_count,
        model_path=args.model_path,
    )

    logger.info(
        "Running E5 direct + GCS + GPU smoke probe: %s (gpu=%s x%d, region=%s)",
        name,
        args.gpu_type,
        args.gpu_count,
        args.region,
    )
    _run_rl_coordinator(job_config)


if __name__ == "__main__":
    main()
