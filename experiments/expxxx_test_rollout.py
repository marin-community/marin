"""Rollout-worker smoke test that exercises Async vLLM inference.

This script builds a minimal `RolloutWorkerConfig` wired to `AsyncvLLMInferenceContext`,
runs a single rollout batch against `MockEnv`, and exits. It is intended for quickly
verifying that inflight weight updates + async vLLM initialization succeed on a TPU
Ray cluster.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import logging
import uuid

import jmp
import ray
from transformers import AutoConfig, AutoTokenizer
from vllm import SamplingParams as VLLMSamplingParams

from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import RayConfig as LevanterRayConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config
from levanter.trainer import TrainerConfig
from levanter.tracker.wandb import WandbConfig

from marin.execution.executor import OutputName
from marin.rl import rollout_worker as rollout_worker_module
from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams as LessonSamplingParams
from marin.rl.environments.base import EnvConfig
from marin.rl.environments.inference_ctx.vllm import InferenceMode, vLLMInferenceContextConfig
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.rollout_worker import RolloutWorker, RolloutWorkerConfig
from marin.rl.weight_transfer import WeightTransferClient, WeightTransferConfig

logger = logging.getLogger(__name__)


MODEL_FAMILIES = {
    "qwen": Qwen3Config,
    "llama": LlamaConfig,
}


class NoOpWeightTransferClient(WeightTransferClient):
    """Stub weight transfer client for smoke tests."""

    def receive_weights(self, old_model):
        return None

    def cleanup(self) -> None:  # pragma: no cover - no-op
        return None

    def get_metrics(self) -> dict:
        return {}


def patch_weight_transfer_client() -> None:
    """Force rollout worker to use a local no-op weight transfer client."""

    def _factory(*_, **__):
        return NoOpWeightTransferClient()

    rollout_worker_module.create_weight_transfer_client = _factory  # type: ignore[attr-defined]


def _build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_lm_config(model_name: str, model_family: str, seq_len: int):
    hf_config = AutoConfig.from_pretrained(model_name)
    config_cls = MODEL_FAMILIES[model_family]
    lm_config = config_cls.from_hf_config(hf_config)
    return dataclasses.replace(lm_config, seq_len=seq_len, tokenizer=model_name)


def _build_trainer_config(batch_size: int, per_device_parallelism: int) -> TrainerConfig:
    return TrainerConfig(
        tracker=WandbConfig(project="async-rollout-smoke", mode="disabled"),
        log_xla_hlo=False,
        log_jaxprs=False,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=batch_size,
        per_device_parallelism=per_device_parallelism,
        num_train_steps=1,
        steps_per_eval=1,
        checkpointer=CheckpointerConfig(
            base_path=OutputName("checkpoints"),
            save_interval=datetime.timedelta(seconds=600),
        ),
        tensor_parallel_axes=["mlp", "heads"],
        fsdp_axis="embed",
        batch_axis="batch",
        ray=LevanterRayConfig(auto_start_cluster=False),
    )


def _build_curriculum(
    run_id: str,
    task_type: str,
    difficulty: str,
    seed: int,
    max_tokens: int,
    n_prompts: int,
    n_generations: int,
) -> CurriculumConfig:
    lesson_sampling = LessonSamplingParams(
        temperature=0.0,
        n_prompts=n_prompts,
        n_generations_per_prompt=n_generations,
        max_tokens=max_tokens,
        stop_tokens=None,
    )

    lesson = LessonConfig(
        lesson_id="mock_addition",
        env_config=EnvConfig(
            env_class="marin.rl.environments.mock_env.MockEnv",
            env_args={"task_type": task_type, "difficulty": difficulty, "seed": seed},
        ),
        sampling_params=lesson_sampling,
    )

    return CurriculumConfig(
        lessons={"mock_addition": lesson},
        eval_frequency=1000,
        eval_n_examples=1,
        micro_eval_frequency=1000,
        micro_eval_n_examples=1,
        actor_name=f"async-rollout-curriculum-{run_id}",
    )


def build_rollout_worker_config(args: argparse.Namespace) -> RolloutWorkerConfig:
    tokenizer = _build_tokenizer(args.model_name)
    lm_config = _build_lm_config(args.model_name, args.model_family, args.max_model_len)
    trainer_config = _build_trainer_config(args.train_batch_size, args.per_device_parallelism)
    curriculum = _build_curriculum(
        args.run_id,
        args.task_type,
        args.task_difficulty,
        args.seed,
        args.max_model_len,
        args.n_prompts,
        args.n_generations,
    )

    inference_sampling = VLLMSamplingParams(
        temperature=1.0,
        n=args.vllm_beam_width,
        max_tokens=args.max_model_len,
        stop=None,
        include_stop_str_in_output=True,
        logprobs=1,
    )

    inference_config = vLLMInferenceContextConfig(
        model_name=args.model_name,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        sampling_params=inference_sampling,
        mode=InferenceMode.ASYNC,
    )

    rollout_storage = RolloutStorageConfig(
        storage_type=StorageType.IN_MEMORY,
        queue_name=f"async-rollout-{args.run_id}",
        queue_maxlen=2,
    )

    weight_transfer = WeightTransferConfig(
        max_weight_transfer_wait_time=0.0,
    )

    return RolloutWorkerConfig(
        curriculum_config=curriculum,
        rollout_storage=rollout_storage,
        weight_transfer=weight_transfer,
        tokenizer=tokenizer,
        run_id=args.run_id,
        trainer=trainer_config,
        model=lm_config,
        inference_type="vllm",
        inference_config=inference_config,
        seed=args.seed,
        max_rollouts=args.max_rollouts,
        log_freq=1,
        initial_checkpoint=args.model_name,
        system_prompt="You are an accurate math assistant.",
        inflight_weight_updates=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async vLLM rollout worker smoke test.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B", help="HF model to load.")
    parser.add_argument("--model-family", choices=MODEL_FAMILIES.keys(), default="qwen")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--n-prompts", type=int, default=2)
    parser.add_argument("--n-generations", type=int, default=2)
    parser.add_argument("--vllm-beam-width", type=int, default=1, help="Number of samples per prompt for vLLM.")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--per-device-parallelism", type=int, default=1)
    parser.add_argument("--task-type", default="addition", help="MockEnv task type.")
    parser.add_argument("--task-difficulty", default="easy")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-rollouts", type=int, default=1)
    parser.add_argument("--ray-address", default=None, help="Ray address, defaults to auto.")
    parser.add_argument("--run-id", default=f"async-rollout-smoke-{uuid.uuid4().hex[:6]}")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    if not ray.is_initialized():
        ray.init(address=args.ray_address or "auto", ignore_reinit_error=True)

    patch_weight_transfer_client()

    worker_config = build_rollout_worker_config(args)

    logger.info("Starting rollout worker smoke test with model %s", args.model_name)
    worker = RolloutWorker(worker_config)
    try:
        worker.run()
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down worker.")
        worker.stop()
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
