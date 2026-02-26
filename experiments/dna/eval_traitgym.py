# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate a DNA checkpoint on TraitGym Mendelian (VEP) benchmark.

Follows the same two-level pattern as run_levanter_train_lm: the ExecutorStep
runs on a CPU coordinator, which submits the actual eval to the TPU pod via
a Fray JobRequest.

Usage (via Ray, from us-central1):
    uv run lib/marin/src/marin/run/ray_run.py --no_wait \
        --config infra/marin-us-central1.yaml \
        --env_vars WANDB_API_KEY=${WANDB_API_KEY} \
        -- python experiments/dna/eval_traitgym.py
"""

import dataclasses
import logging
import os

import jmp
import levanter.eval_harness as eval_harness
import levanter.infra.cli_helpers
from fray.v2 import Entrypoint, JobRequest, ResourceConfig, TpuConfig, create_environment, current_client
from levanter.distributed import RayConfig
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig

from experiments.dna.defaults import DNA_RESOURCES_V1, DNA_TOKENIZER_V1, dna_qwen3_0_6b_256_v1
from experiments.evals.task_configs import TRAITGYM_MENDELIAN_V2, convert_to_levanter_task_config
from iris.temp_buckets import get_temp_bucket_path
from marin.execution.executor import ExecutorStep, executor_main

logger = logging.getLogger(__name__)

CHECKPOINT = "gs://marin-dna-us-central1/checkpoints/exp57-balanced-mixture-qwen3_0_6b-r02-4aeb57/hf/step-46000"


@dataclasses.dataclass
class EvalTraitGymConfig:
    checkpoint_path: str
    tokenizer: str
    resources: ResourceConfig = dataclasses.field(default_factory=lambda: DNA_RESOURCES_V1)
    # Credentials are stored in the config so they survive serialization across
    # Fray job boundaries (Ray job → CPU coordinator → TPU pod).
    wandb_api_key: str | None = None
    hf_token: str | None = None


def _build_eval_env(config: EvalTraitGymConfig) -> dict[str, str]:
    """Build environment variables for the TPU eval job.

    Mirrors the env setup in run_levanter_train_lm.  Credentials are read
    from the config (which survives serialization) with os.environ as fallback.
    """
    default_launch_config = levanter.infra.cli_helpers.load_config()
    default_env = default_launch_config.env_for_accel(config.resources.device.variant) or {}
    env = {str(k): str(v) for k, v in default_env.items()}

    wandb_key = config.wandb_api_key or os.environ.get("WANDB_API_KEY")
    if wandb_key:
        env["WANDB_API_KEY"] = wandb_key

    hf_token = config.hf_token or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        env["HF_TOKEN"] = hf_token

    env.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    env.setdefault("HF_ALLOW_CODE_EVAL", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("TPU_MIN_LOG_LEVEL", "2")
    env.setdefault("TPU_STDERR_LOG_LEVEL", "2")

    temp_cache_path = get_temp_bucket_path(ttl_days=30, prefix="compilation-cache")
    if temp_cache_path is not None:
        env["JAX_COMPILATION_CACHE_DIR"] = temp_cache_path
        logger.info("JAX compilation cache on temp bucket: %s", temp_cache_path)

    return env


def _run_eval_on_tpu(
    eval_config: eval_harness.EvalHarnessMainConfig,
    wandb_api_key: str | None,
    hf_token: str | None,
):
    """Thin wrapper that sets credentials in os.environ before running eval.

    This runs on the TPU pod.  Credentials are passed as serialized function
    arguments because Fray/Ray runtime_env env_vars don't reliably propagate
    to TPU SliceActor tasks.
    """
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    eval_harness.run_eval_harness_main(eval_config)


def run_eval_traitgym(config: EvalTraitGymConfig):
    """Submit TraitGym Mendelian evaluation to run on a TPU pod.

    This function runs on a CPU coordinator node. It builds the eval config
    and environment, then submits the actual evaluation as a Fray job that
    runs on the TPU pod.
    """
    env = _build_eval_env(config)

    tasks = convert_to_levanter_task_config([TRAITGYM_MENDELIAN_V2])

    eval_config = eval_harness.EvalHarnessMainConfig(
        eval_harness=eval_harness.LmEvalHarnessConfig(
            task_spec=tasks,
            include_path="experiments/evals/custom_tasks/dna_vep",
            log_samples=False,
        ),
        tokenizer=config.tokenizer,
        checkpoint_path=config.checkpoint_path,
        checkpoint_is_hf=True,
        trainer=TrainerConfig(
            tracker=NoopConfig(),  # TODO: restore WandbConfig once env propagation is fixed
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            per_device_eval_parallelism=64,
            ray=RayConfig(auto_start_cluster=False),
        ),
        model=dna_qwen3_0_6b_256_v1,
    )

    logger.info(f"Evaluating checkpoint: {config.checkpoint_path}")

    client = current_client()

    extras = []
    if isinstance(config.resources.device, TpuConfig):
        extras.append("tpu")

    # Pass credentials as serialized function arguments so they survive
    # across Fray/Ray job boundaries (runtime_env env_vars don't reliably
    # propagate to TPU SliceActor tasks).
    job_request = JobRequest(
        name="eval_traitgym",
        entrypoint=Entrypoint.from_callable(
            _run_eval_on_tpu,
            args=[eval_config, env.get("WANDB_API_KEY"), env.get("HF_TOKEN")],
        ),
        resources=config.resources,
        environment=create_environment(env_vars=env, extras=extras),
        max_retries_failure=10,
    )
    job = client.submit(job_request)
    job.wait(raise_on_failure=True)

    logger.info("Evaluation complete.")


_ENV_KEYS_TO_FORWARD = ["WANDB_API_KEY", "HUGGING_FACE_HUB_TOKEN"]

eval_step = ExecutorStep(
    name="eval/traitgym_mendelian_v2_train",
    description="Evaluate DNA checkpoint on TraitGym Mendelian (VEP) using LLR scoring.",
    fn=run_eval_traitgym,
    config=EvalTraitGymConfig(
        checkpoint_path=CHECKPOINT,
        tokenizer=DNA_TOKENIZER_V1,
        # Store credentials in the config so they survive serialization across
        # job boundaries (entry point → CPU coordinator → TPU pod).
        wandb_api_key=os.environ.get("WANDB_API_KEY"),
        hf_token=os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"),
    ),
    # Also forward via env_vars as a belt-and-suspenders approach.
    env_vars={k: os.environ[k] for k in _ENV_KEYS_TO_FORWARD if k in os.environ},
    # No TPU resources here — runs on CPU coordinator.
    # TPU resources are requested inside run_eval_traitgym via JobRequest.
)

if __name__ == "__main__":
    executor_main(steps=[eval_step])
