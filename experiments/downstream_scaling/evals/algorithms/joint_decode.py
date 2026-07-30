# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Joint-decode completion algorithm for downstream-scaling evals.

Decodes from two models in lockstep. At each step:
  1. Get top-k tokens from model A (decoder) and model B (advisor).
  2. Among A's top-k, pick the token with highest rank in B's top-k.
  3. If A's top-k and B's top-k don't overlap, fall back to A's top-1.

Execution is delegated to the joint-decode package via
``joint_decode_backend`` (modern protocol: sliding-window admission, holds,
force-stop on peer finish); the selection rule is the package's
``select_top_rank``, ported from this module's original implementation.

Hash stability: the config dataclasses, their defaults, and the step
construction below are byte-frozen — executor step hashes cover exactly the
versioned config values under the step name, so completed experiments keep
resolving. Do not change them; execution internals live in the backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from thalas.execution.executor import ExecutorStep, InputName, MirroredValue
from thalas.execution.remote import remote
from thalas.execution.types import this_output_path, versioned

from experiments.downstream_scaling.evals.framework.xregion.pool import EnginePlacement
from experiments.downstream_scaling.evals.utils import version_path

logger = logging.getLogger(__name__)

VLLM_TPU_ENV_VARS: dict[str, str] = {
    "MARIN_VLLM_MODE": "native",
    # Required at `uv sync` time so vllm's setup.py skips CUDA-version
    # detection (which asserts CUDA_HOME). Propagated to the container build
    # via remote(env_vars=...).
    "VLLM_TARGET_DEVICE": "tpu",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
}


@dataclass(frozen=True)
class JointDecodeSamplingConfig:
    n_samples: int
    max_tokens: int
    top_k_a: int
    top_k_b: int
    seed: int
    # Retained for executor cache-key stability; no longer consumed.
    temperature: float = 1.0
    top_p: float = 1.0
    stop: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.top_k_a < 1 or self.top_k_b < 1:
            raise ValueError("top_k_a and top_k_b must both be >= 1")
        if self.n_samples != 1:
            raise ValueError(f"joint_decode is deterministic per prompt; n_samples must be 1 (got {self.n_samples})")


@dataclass(frozen=True)
class JointDecodeModelConfig:
    max_model_len: int = 8192
    gpu_memory_utilization: float | None = None
    enable_prefix_caching: bool = False
    # Halve the RPA-kernel KV-page block size. Required for delphi-shaped
    # models (otherwise vmem error); harms perf on standard models like llama,
    # so default off.
    apply_rpa_block_size_patch: bool = False


@dataclass(frozen=True)
class JointDecodeExecutionConfig:
    num_workers: int
    worker_resources: ResourceConfig
    chunk_size: int = 512
    # Cap on in-flight requests per engine pair. Under the package backend
    # this bounds the sliding admission window (None → whole chunk).
    microbatch_size: int | None = None
    chip_a: int = 0
    chip_b: int = 1
    barrier_timeout_s: float = 60.0
    server_port: int = 0

    def __post_init__(self) -> None:
        if self.microbatch_size is not None and self.microbatch_size < 1:
            raise ValueError(f"microbatch_size must be >= 1 or None (got {self.microbatch_size})")


@dataclass(frozen=True)
class JointDecodeConfig:
    sampling: JointDecodeSamplingConfig
    advisor_model_path: str | InputName | MirroredValue
    decoder_model: JointDecodeModelConfig
    advisor_model: JointDecodeModelConfig
    execution: JointDecodeExecutionConfig


@dataclass(frozen=True)
class JointDecodeCompletionStepConfig:
    output_path: str
    decoder_model_path: str
    advisor_model_path: str
    prompts_path: str
    sampling: JointDecodeSamplingConfig
    decoder_model: JointDecodeModelConfig
    advisor_model: JointDecodeModelConfig
    num_workers: int
    chunk_size: int
    microbatch_size: int
    worker_resources: ResourceConfig
    chip_a: int
    chip_b: int
    barrier_timeout_s: float
    server_port: int


@dataclass(frozen=True)
class JointDecodeCompletionAlgorithm:
    config: JointDecodeConfig

    def make_completions_step(
        self,
        *,
        name: str,
        model_path: str | InputName | MirroredValue,
        prompts_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        return make_joint_decode_completion_step(
            name=name,
            model_path=model_path,
            prompts_path=prompts_path,
            config=self.config,
        )


def make_joint_decode_completion_step(
    *,
    name: str,
    model_path: str | InputName | MirroredValue,
    prompts_path: str | InputName | MirroredValue,
    config: JointDecodeConfig,
) -> ExecutorStep:
    microbatch_size = (
        config.execution.chunk_size if config.execution.microbatch_size is None else config.execution.microbatch_size
    )
    return ExecutorStep(
        name=name,
        fn=remote(
            run_joint_decode_completion_chunks,
            resources=config.execution.worker_resources,
            pip_dependency_groups=["vllm", "tpu"],
            env_vars=VLLM_TPU_ENV_VARS,
        ),
        config=JointDecodeCompletionStepConfig(
            output_path=this_output_path(),
            decoder_model_path=version_path(model_path),  # type: ignore[arg-type]
            advisor_model_path=version_path(config.advisor_model_path),  # type: ignore[arg-type]
            prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
            sampling=versioned(config.sampling),  # type: ignore[arg-type]
            decoder_model=versioned(config.decoder_model),  # type: ignore[arg-type]
            advisor_model=versioned(config.advisor_model),  # type: ignore[arg-type]
            num_workers=config.execution.num_workers,
            chunk_size=versioned(config.execution.chunk_size),  # type: ignore[arg-type]
            microbatch_size=microbatch_size,
            worker_resources=config.execution.worker_resources,
            chip_a=config.execution.chip_a,
            chip_b=config.execution.chip_b,
            barrier_timeout_s=config.execution.barrier_timeout_s,
            server_port=config.execution.server_port,
        ),
    )


def run_joint_decode_completion_chunks(config: JointDecodeCompletionStepConfig) -> None:
    # Lazy: the joint-decode package rides the linux-only vllm extra; this
    # module must stay importable for step construction everywhere.
    from joint_decode.selection import select_top_rank  # noqa: PLC0415

    from experiments.downstream_scaling.evals.algorithms import joint_decode_backend  # noqa: PLC0415

    joint_decode_backend.run_completion_chunks(
        output_path=config.output_path,
        prompts_path=config.prompts_path,
        decoder=joint_decode_backend.EngineModelParams(
            model_path=config.decoder_model_path,
            max_model_len=config.decoder_model.max_model_len,
            gpu_memory_utilization=config.decoder_model.gpu_memory_utilization,
            enable_prefix_caching=config.decoder_model.enable_prefix_caching,
            apply_rpa_block_size_patch=config.decoder_model.apply_rpa_block_size_patch,
        ),
        advisor=joint_decode_backend.EngineModelParams(
            model_path=config.advisor_model_path,
            max_model_len=config.advisor_model.max_model_len,
            gpu_memory_utilization=config.advisor_model.gpu_memory_utilization,
            enable_prefix_caching=config.advisor_model.enable_prefix_caching,
            apply_rpa_block_size_patch=config.advisor_model.apply_rpa_block_size_patch,
        ),
        n_samples=config.sampling.n_samples,
        max_tokens=config.sampling.max_tokens,
        top_k_a=config.sampling.top_k_a,
        top_k_b=config.sampling.top_k_b,
        seed=config.sampling.seed,
        stop=tuple(config.sampling.stop or ()),
        select_token=select_top_rank,
        decoder_placement=EnginePlacement((config.chip_a,), (1, 1, 1), 1),
        advisor_placement=EnginePlacement((config.chip_b,), (1, 1, 1), 1),
        max_microbatch_size=config.microbatch_size,
        max_num_batched_tokens=None,
        barrier_timeout_s=config.barrier_timeout_s,
        chunk_size=config.chunk_size,
        aggregate_workers=config.num_workers,
        algorithm="joint_decode",
    )
