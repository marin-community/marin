# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration for small GPU-only Iris RL smoke experiments."""

from urllib.parse import urlparse

from fray.types import ResourceConfig
from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
from marin.rl.environments import EnvConfig
from marin.rl.placement import marin_prefix_for_region
from rigging.filesystem import REGION_TO_DATA_BUCKET

CANONICAL_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_ARTIFACT_SUBPATH = "models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f"

DEFAULT_EXPERIMENT_REGION = "us-central1"
DEFAULT_GPU_TYPE = "H100"
DEFAULT_GPU_COUNT = 4
DEFAULT_NUM_TRAIN_STEPS = 5

DEFAULT_MAX_INPUT_TOKENS = 256
DEFAULT_MAX_OUTPUT_TOKENS = 256
DEFAULT_NUM_ROLLOUT_WORKERS = 1
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90

GPU_WORKER_CPU = 32
GPU_WORKER_RAM = "240g"
GPU_WORKER_DISK = "128g"
_DATA_BUCKET_TO_REGION = {bucket: bucket_region for bucket_region, bucket in REGION_TO_DATA_BUCKET.items()}


def gpu_smoke_resources(*, region: str, gpu_type: str, gpu_count: int) -> ResourceConfig:
    """Return the single-host GPU resource shape for the RL smoke path."""
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    return ResourceConfig.with_gpu(
        gpu_type,
        count=gpu_count,
        cpu=GPU_WORKER_CPU,
        ram=GPU_WORKER_RAM,
        disk=GPU_WORKER_DISK,
        regions=[region],
    )


def gpu_smoke_train_batch_size(gpu_count: int) -> int:
    """Use one training example per GPU for the smallest stable trainer shape."""
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    return gpu_count


def gpu_smoke_rollout_count(gpu_count: int) -> int:
    """Keep rollout count aligned with the trainer batch for the first GPU milestone."""
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    return gpu_count


def gpu_smoke_curriculum(
    *,
    run_id: str,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    num_generations: int,
) -> CurriculumConfig:
    """Single-lesson math curriculum sized for a short GPU smoke run."""
    return CurriculumConfig(
        lessons={
            "math_full": LessonConfig(
                lesson_id="math_full",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.math_env.MathEnv",
                    env_args={"seed": 42},
                ),
                dependencies=[],
                sampling_params=SamplingParams(
                    temperature=1.0,
                    n_prompts=1,
                    n_generations_per_prompt=num_generations,
                    max_output_tokens=max_output_tokens,
                    top_k=4096,
                    stop_tokens=None,
                ),
            ),
        },
        eval_frequency=5,
        micro_eval_frequency=None,
        actor_name=f"curriculum-{run_id}",
        eval_n_examples=max(4, num_generations),
        max_seq_len=max_input_tokens + max_output_tokens,
    )


def gpu_smoke_prefix(region: str) -> str:
    """Return the regional Marin prefix used for smoke-run outputs."""
    return marin_prefix_for_region(region)


def gpu_smoke_model_path(region: str) -> str:
    """Return the canonical region-local GCS model artifact for smoke runs."""
    return f"{gpu_smoke_prefix(region)}/{MODEL_ARTIFACT_SUBPATH}"


def validate_gpu_smoke_model_path(*, region: str, model_path: str) -> None:
    """Reject cross-region Marin GCS reads while allowing external model sources."""
    parsed = urlparse(model_path)
    if parsed.scheme != "gs":
        return

    expected_bucket = REGION_TO_DATA_BUCKET.get(region.lower())
    model_bucket_region = _DATA_BUCKET_TO_REGION.get(parsed.netloc)
    if model_bucket_region is None or parsed.netloc == expected_bucket:
        return

    expected_prefix = f"gs://{expected_bucket}" if expected_bucket is not None else f"region {region!r}"
    actual_prefix = f"gs://{parsed.netloc}"
    raise ValueError(
        "GPU smoke model_path must stay in the launcher region to avoid cross-region model reads. "
        f"Expected a path under {expected_prefix!r}, got {actual_prefix!r} for region {model_bucket_region!r}."
    )


def resolve_gpu_smoke_model_path(*, region: str, model_path: str | None) -> str:
    """Return the validated model artifact path for a smoke run."""
    if model_path is None:
        return gpu_smoke_model_path(region)
    validate_gpu_smoke_model_path(region=region, model_path=model_path)
    return model_path


DEFAULT_MODEL_PATH = gpu_smoke_model_path(DEFAULT_EXPERIMENT_REGION)
