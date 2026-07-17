# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace

from fray.types import ResourceConfig
from marin.evaluation.lm_eval import LM_EVAL_UV_PACKAGES, LmEvalRun, run_lm_eval
from marin.execution.lazy import Artifact, ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.inference.vllm import (
    DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER,
    BrokeredVllmSystemConfig,
    InferenceWorkerConfig,
    start_iris_brokered_vllm,
)
from rigging.filesystem import StoragePath

VLLM_WORKER_ENV_VARS = (
    ("VLLM_ENABLE_V1_MULTIPROCESSING", "0"),
    ("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1"),
    ("VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION", "1"),
    ("VLLM_TPU_SKIP_PRECOMPILE", "1"),
)


@dataclass(frozen=True)
class BrokeredLmEvalArtifactConfig:
    inference: BrokeredVllmSystemConfig
    lm_eval_uv_packages: tuple[str, ...]
    eval_run: LmEvalRun
    output_path: str


def _run_brokered_lm_eval_artifact(
    inference: BrokeredVllmSystemConfig,
    eval_run: LmEvalRun,
    output_path: str,
) -> None:
    with tempfile.TemporaryDirectory() as local_output:
        with start_iris_brokered_vllm(inference) as model:
            run_lm_eval(model, eval_run, local_output)
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
    worker_resources = inference.worker_resources
    inference = replace(inference, worker_resources=None)
    eval_run = replace(
        eval_run,
        extra_model_args={
            "num_concurrent": inference.workers.max_in_flight_per_worker,
            "timeout": int(inference.proxy.request_timeout_seconds),
            **eval_run.extra_model_args,
        },
    )

    def build_config(context: StepContext) -> BrokeredLmEvalArtifactConfig:
        return BrokeredLmEvalArtifactConfig(
            inference=replace(inference, worker_resources=context.runtime_arg("worker_resources")),
            lm_eval_uv_packages=LM_EVAL_UV_PACKAGES,
            eval_run=eval_run,
            output_path=context.output_path,
        )

    def run_step(config: BrokeredLmEvalArtifactConfig) -> None:
        if config.lm_eval_uv_packages != LM_EVAL_UV_PACKAGES:
            raise ValueError("artifact lm-eval packages must match the pinned runtime packages")
        remote(
            _run_brokered_lm_eval_artifact,
            name=name,
            resources=parent_resources,
            env_vars=dict(parent_env_vars),
        )(config.inference, config.eval_run, config.output_path)

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=Artifact,
        run=run_step,
        build_config=build_config,
        deps=(),
        runtime_args={
            "parent_resources": parent_resources,
            "worker_resources": worker_resources,
        },
    )


def brokered_vllm_config(
    *,
    model: str,
    tokenizer: str,
    worker_resources: ResourceConfig,
    workers: int = 1,
    num_concurrent: int = DEFAULT_BROKERED_MAX_IN_FLIGHT_PER_WORKER,
) -> BrokeredVllmSystemConfig:
    config = BrokeredVllmSystemConfig(
        model=model,
        tokenizer=tokenizer,
        workers=InferenceWorkerConfig(count=workers, max_in_flight_per_worker=num_concurrent),
    )
    timeout = config.server.timeout_seconds
    return replace(
        config,
        worker_resources=worker_resources,
        worker_env_vars=dict(VLLM_WORKER_ENV_VARS),
        proxy=replace(
            config.proxy,
            request_timeout_seconds=timeout,
            readiness_timeout_seconds=timeout,
            ignored_request_fields=("seed",),
        ),
    )
