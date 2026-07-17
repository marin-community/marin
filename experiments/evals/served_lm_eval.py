# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from fray.types import ResourceConfig
from marin.evaluation.lm_eval import LM_EVAL_UV_PACKAGES, LmEvalRun, run_iris_brokered_lm_eval
from marin.execution.lazy import Artifact, ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.inference.vllm import (
    DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER,
    BrokeredVllmSystemConfig,
    InferenceWorkerConfig,
)
from rigging.filesystem import StoragePath

VLLM_WORKER_ENV_VARS = (
    ("VLLM_ENABLE_V1_MULTIPROCESSING", "0"),
    ("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1"),
    ("VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION", "1"),
    ("VLLM_TPU_SKIP_PRECOMPILE", "1"),
)


@dataclass(frozen=True)
class ServedLmEvalBenchmark:
    tasks: Sequence[str]
    output_path: str
    confirm_run_unsafe_code: bool = False
    parent_env_vars: Mapping[str, str] = field(default_factory=dict)
    model: str = "Qwen/Qwen3-0.6B-Base"
    tokenizer: str = "Qwen/Qwen3-0.6B"
    tpu_type: str = "v5litepod-4"
    worker_ram: str = "96g"
    parent_cpu: float = 0.5
    parent_ram: str = "6g"
    parent_disk: str = "16g"
    region: str | None = "us-west4"
    workers: int = 1
    num_concurrent: int = DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER


@dataclass(frozen=True)
class BrokeredLmEvalArtifactConfig:
    inference: BrokeredVllmSystemConfig
    lm_eval_uv_packages: tuple[str, ...]
    eval_run: LmEvalRun


def _run_brokered_lm_eval_artifact(
    inference: BrokeredVllmSystemConfig,
    eval_run: LmEvalRun,
) -> None:
    output_path = eval_run.output_path
    with tempfile.TemporaryDirectory() as local_output:
        run_iris_brokered_lm_eval(inference, replace(eval_run, output_path=local_output), {})
        StoragePath(output_path).upload_from(local_output + "/", recursive=True)


def brokered_lm_eval_step(
    inference: BrokeredVllmSystemConfig,
    eval_run: LmEvalRun,
    *,
    name: str,
    version: str,
    parent_resources: ResourceConfig,
    parent_env_vars: Mapping[str, str],
) -> ArtifactStep[Artifact]:
    """Build a lazy artifact containing lm-eval metrics and samples."""
    if inference.worker_resources is None:
        raise ValueError("inference.worker_resources must be set for a brokered lm-eval artifact")

    def build_config(context: StepContext) -> BrokeredLmEvalArtifactConfig:
        return BrokeredLmEvalArtifactConfig(
            inference=inference,
            lm_eval_uv_packages=LM_EVAL_UV_PACKAGES,
            eval_run=replace(eval_run, output_path=context.output_path),
        )

    def run_step(config: BrokeredLmEvalArtifactConfig) -> None:
        if config.lm_eval_uv_packages != LM_EVAL_UV_PACKAGES:
            raise ValueError("artifact lm-eval packages must match the pinned runtime packages")
        remote(
            _run_brokered_lm_eval_artifact,
            name=name,
            resources=parent_resources,
            env_vars=dict(parent_env_vars),
        )(config.inference, config.eval_run)

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=Artifact,
        run=run_step,
        build_config=build_config,
        deps=(),
        runtime_args={
            "parent_resources": parent_resources,
        },
    )


def brokered_lm_eval_configs(
    benchmark: ServedLmEvalBenchmark,
    *,
    limit: int | None,
) -> tuple[BrokeredVllmSystemConfig, LmEvalRun]:
    inference = brokered_vllm_config(
        model=benchmark.model,
        tokenizer=benchmark.tokenizer,
        workers=benchmark.workers,
        num_concurrent=benchmark.num_concurrent,
    )
    inference = replace(
        inference,
        worker_resources=ResourceConfig.with_tpu(
            benchmark.tpu_type,
            ram=benchmark.worker_ram,
            regions=[benchmark.region] if benchmark.region is not None else None,
        ),
        worker_env_vars=dict(VLLM_WORKER_ENV_VARS),
    )
    eval_run = LmEvalRun(
        tasks=benchmark.tasks,
        output_path=benchmark.output_path,
        limit=limit,
        confirm_run_unsafe_code=benchmark.confirm_run_unsafe_code,
        extra_model_args={
            "num_concurrent": inference.workers.max_in_flight_per_worker,
            "timeout": int(inference.proxy.request_timeout_seconds),
        },
    )
    return inference, eval_run


def served_lm_eval_step(
    benchmark: ServedLmEvalBenchmark,
    *,
    name: str,
    version: str,
    limit: int | None = None,
) -> ArtifactStep[Artifact]:
    """Build a brokered lm-eval artifact runnable by a Marin StepRunner."""
    inference, eval_run = brokered_lm_eval_configs(
        benchmark,
        limit=limit,
    )
    return brokered_lm_eval_step(
        inference,
        eval_run,
        name=name,
        version=version,
        parent_resources=ResourceConfig.with_cpu(
            cpu=benchmark.parent_cpu,
            ram=benchmark.parent_ram,
            disk=benchmark.parent_disk,
            regions=[benchmark.region] if benchmark.region else None,
            preemptible=False,
        ),
        parent_env_vars=benchmark.parent_env_vars,
    )


def brokered_vllm_config(
    *,
    model: str,
    tokenizer: str,
    workers: int,
    num_concurrent: int,
) -> BrokeredVllmSystemConfig:
    config = BrokeredVllmSystemConfig(
        model=model,
        tokenizer=tokenizer,
        workers=InferenceWorkerConfig(count=workers, max_in_flight_per_worker=num_concurrent),
    )
    timeout = config.server.timeout_seconds
    return replace(
        config,
        server=replace(config.server, timeout_seconds=timeout),
        proxy=replace(
            config.proxy,
            request_timeout_seconds=timeout,
            readiness_timeout_seconds=timeout,
            ignored_request_fields=("seed",),
        ),
    )
