# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from fray.types import ResourceConfig
from marin.evaluation.harbor_eval import HarborEvalResults, HarborEvalRun, run_harbor_eval
from marin.evaluation.lm_eval import LM_EVAL_UV_PACKAGES, LmEvalResults, LmEvalRun, run_lm_eval
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.inference.config import (
    BrokerConfig,
    IrisConfig,
    ServedModelConfig,
    VllmEngineConfig,
)
from marin.inference.iris import remote_inference
from marin.inference.types import RunningModel
from rigging.filesystem import StoragePath

_EVAL_PARENT_RESOURCES = ResourceConfig.with_cpu(
    cpu=0.5,
    ram="6g",
    disk="16g",
    preemptible=False,
)

# The harbor parent runs the agent LLM loops in-process, one per concurrent trial.
_HARBOR_EVAL_PARENT_RESOURCES = ResourceConfig.with_cpu(
    cpu=2,
    ram="8g",
    disk="32g",
    preemptible=False,
)
_HARBOR_EVAL_PARENT_DEPENDENCY_GROUPS = ["harbor"]

OT_TB_LITE_DATASET = "openthoughts-tblite"
OT_TB_LITE_VERSION = "2.0"
# The registry of the marin-community harbor fork, pinned to the workspace harbor rev.
MARIN_HARBOR_REGISTRY_URL = (
    "https://raw.githubusercontent.com/marin-community/harbor/0d0dbe8904ed984137a0dfa764583f1508cf3274/registry.json"
)


@dataclass(frozen=True)
class BrokeredEvalInference:
    model: ServedModelConfig
    engine: VllmEngineConfig
    iris: IrisConfig
    instances: int = 1
    broker: BrokerConfig = field(default_factory=BrokerConfig)


def _run_brokered_eval_artifact[RunT](
    inference: BrokeredEvalInference,
    eval_run: RunT,
    output_path: str,
    run_eval: Callable[[RunningModel, RunT, str], None],
) -> None:
    with tempfile.TemporaryDirectory() as local_output:
        with remote_inference(
            inference.model,
            inference.engine,
            inference.iris,
            instances=inference.instances,
            broker=inference.broker,
        ) as session:
            run_eval(session.model, eval_run, local_output)
        StoragePath(output_path).upload_from(local_output + "/", recursive=True)


def _require_pinned_lm_eval_packages(config: Mapping[str, Any]) -> None:
    if config["lm_eval_uv_packages"] != LM_EVAL_UV_PACKAGES:
        raise ValueError("artifact lm-eval packages must match the pinned runtime packages")


def _brokered_eval_step[ResultsT, RunT](
    inference: BrokeredEvalInference,
    eval_run: RunT,
    *,
    name: str,
    version: str,
    artifact_type: type[ResultsT],
    run_eval: Callable[[RunningModel, RunT, str], None],
    parent_resources: ResourceConfig,
    parent_env_vars: Mapping[str, str] | None = None,
    parent_dependency_groups: list[str] | None = None,
    config_extras: Mapping[str, Any] | None = None,
    check_config: Callable[[Mapping[str, Any]], None] | None = None,
) -> ArtifactStep[ResultsT]:
    """Shared scaffold: serve the model brokered, run one eval against it, keep the results.

    ``config_extras`` ride in the artifact config so pin changes invalidate it;
    ``check_config`` re-validates a stored config against the current runtime.
    """
    worker_resources = inference.iris.worker_resources
    results_path = str(StoragePath("mirror://") / name / version)

    def build_config(context: StepContext) -> dict[str, Any]:
        return {
            "inference": replace(
                inference,
                iris=replace(inference.iris, worker_resources=context.runtime_arg("worker_resources")),
            ),
            "eval_run": eval_run,
            "results_path": results_path,
            **(config_extras or {}),
        }

    def run_step(config: dict[str, Any]) -> ResultsT:
        if check_config is not None:
            check_config(config)
        remote(
            _run_brokered_eval_artifact,
            name=name,
            resources=parent_resources,
            env_vars=dict(parent_env_vars or {}),
            pip_dependency_groups=parent_dependency_groups,
        )(config["inference"], config["eval_run"], config["results_path"], run_eval)
        return artifact_type(results_path=config["results_path"])

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=artifact_type,
        run=run_step,
        build_config=build_config,
        deps=(),
        runtime_args={
            "worker_resources": worker_resources,
        },
    )


def brokered_lm_eval_step(
    inference: BrokeredEvalInference,
    eval_run: LmEvalRun,
    *,
    name: str,
    version: str,
    parent_env_vars: Mapping[str, str],
) -> ArtifactStep[LmEvalResults]:
    """Build a lazy artifact containing lm-eval metrics and samples."""
    eval_run = replace(
        eval_run,
        extra_model_args={
            "num_concurrent": inference.broker.worker.max_in_flight,
            "timeout": int(inference.broker.proxy.request_timeout_seconds),
            **eval_run.extra_model_args,
        },
    )
    return _brokered_eval_step(
        inference,
        eval_run,
        name=name,
        version=version,
        artifact_type=LmEvalResults,
        run_eval=run_lm_eval,
        parent_resources=_EVAL_PARENT_RESOURCES,
        parent_env_vars=parent_env_vars,
        config_extras={"lm_eval_uv_packages": LM_EVAL_UV_PACKAGES},
        check_config=_require_pinned_lm_eval_packages,
    )


def brokered_harbor_eval_step(
    inference: BrokeredEvalInference,
    eval_run: HarborEvalRun,
    *,
    name: str,
    version: str,
) -> ArtifactStep[HarborEvalResults]:
    """Build a lazy artifact containing a Harbor dataset's trial results."""
    return _brokered_eval_step(
        inference,
        eval_run,
        name=name,
        version=version,
        artifact_type=HarborEvalResults,
        run_eval=run_harbor_eval,
        parent_resources=_HARBOR_EVAL_PARENT_RESOURCES,
        parent_dependency_groups=_HARBOR_EVAL_PARENT_DEPENDENCY_GROUPS,
    )


def brokered_ot_tb_lite_step(
    inference: BrokeredEvalInference,
    *,
    model_name: str,
    version: str,
    container_profile: str,
    n_tasks: int | None = None,
    n_concurrent_trials: int = 4,
) -> ArtifactStep[HarborEvalResults]:
    """OpenThoughts-TBLite from the marin-community harbor fork, on Iris sandboxes."""
    eval_run = HarborEvalRun(
        dataset=OT_TB_LITE_DATASET,
        dataset_version=OT_TB_LITE_VERSION,
        registry_url=MARIN_HARBOR_REGISTRY_URL,
        n_tasks=n_tasks,
        n_concurrent_trials=n_concurrent_trials,
        container_profile=container_profile,
    )
    return brokered_harbor_eval_step(
        inference,
        eval_run,
        name=f"evals/{model_name}/ot-tb-lite",
        version=version,
    )
