# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from fray.types import ResourceConfig
from marin.evaluation.lm_eval import LM_EVAL_UV_PACKAGES, LmEvalResults, LmEvalRun, run_lm_eval
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.inference.vllm import (
    BrokeredVllmSystemConfig,
    start_iris_brokered_vllm,
)
from rigging.filesystem import StoragePath

_EVAL_PARENT_RESOURCES = ResourceConfig.with_cpu(
    cpu=0.5,
    ram="6g",
    disk="16g",
    preemptible=False,
)


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
    parent_env_vars: Mapping[str, str],
) -> ArtifactStep[LmEvalResults]:
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
    results_path = str(StoragePath("mirror://") / name / version)

    def build_config(context: StepContext) -> dict[str, Any]:
        return {
            "inference": replace(inference, worker_resources=context.runtime_arg("worker_resources")),
            "lm_eval_uv_packages": LM_EVAL_UV_PACKAGES,
            "eval_run": eval_run,
            "results_path": results_path,
        }

    def run_step(config: dict[str, Any]) -> LmEvalResults:
        if config["lm_eval_uv_packages"] != LM_EVAL_UV_PACKAGES:
            raise ValueError("artifact lm-eval packages must match the pinned runtime packages")
        remote(
            _run_brokered_lm_eval_artifact,
            name=name,
            resources=_EVAL_PARENT_RESOURCES,
            env_vars=dict(parent_env_vars),
        )(config["inference"], config["eval_run"], config["results_path"])
        return LmEvalResults(results_path=config["results_path"])

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=LmEvalResults,
        run=run_step,
        build_config=build_config,
        deps=(),
        runtime_args={
            "worker_resources": worker_resources,
        },
    )
