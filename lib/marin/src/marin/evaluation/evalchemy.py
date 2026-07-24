# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Evalchemy against an already-running OpenAI-compatible model."""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from iris.client import Job, JobFailedError, iris_ctx
from iris.cluster.constraints import region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from rigging.filesystem import url_to_fs

from marin.evaluation.eval_result import EvalchemyResult
from marin.evaluation.evalchemy_client import CONFIG_ENV_KEY
from marin.evaluation.evalchemy_runtime import EVALCHEMY_IMAGE, EVALCHEMY_PYTHON
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.samples import export_lm_eval_samples
from marin.inference.types import InferenceObserver, RunningModel

logger = logging.getLogger(__name__)

DEFAULT_NUM_CONCURRENT = 16
LOG_TAIL_LINES = 100
_CHILD_WAIT_SLICE_SECONDS = 60.0
_PROPAGATED_ENV_KEYS = ("HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT")
_EVAL_CLIENT_SCRIPT = "lib/marin/src/marin/evaluation/evalchemy_client.py"


class PipelineStage(StrEnum):
    """Stage used to classify a mechanism failure in an eval record."""

    SERVE = "serve"
    EVAL = "eval"
    ARTIFACTS = "artifacts"


class EvalPipelineError(RuntimeError):
    """A mechanism failure carrying child jobs and bounded log tails."""

    def __init__(
        self,
        message: str,
        *,
        stage: PipelineStage,
        jobs: dict[str, str],
        log_tails: dict[str, tuple[str, ...]],
    ):
        super().__init__(message)
        self.stage = stage
        self.jobs = jobs
        self.log_tails = log_tails


class EndpointMovedError(EvalPipelineError):
    """The concrete endpoint moved while lm-eval was retrying its old address."""


def job_log_tail(job: Job, limit: int = LOG_TAIL_LINES) -> tuple[str, ...]:
    """Fetch the final log lines without masking the original job failure."""
    try:
        entries = [entry for task in job.tasks() for entry in task.logs()]
    except Exception:
        logger.warning("could not fetch log tail for %s", job, exc_info=True)
        return ()
    return tuple(entry.data.rstrip("\n") for entry in entries[-limit:])


def _propagated_env(**extra: str) -> dict[str, str]:
    env = {key: os.environ[key] for key in _PROPAGATED_ENV_KEYS if os.environ.get(key)}
    env.update(extra)
    return env


@dataclass(frozen=True)
class EvalchemyRuntimeConfig:
    """Execution policy for the Evalchemy HTTP-client child."""

    image: str = EVALCHEMY_IMAGE
    cpu: float = 8.0
    memory: str = "32g"
    disk: str = "50g"
    region: str | None = None


@dataclass(frozen=True)
class EvalchemyRunConfig:
    """One Evalchemy task group evaluated against a running model."""

    name: str
    tasks: tuple[EvalTaskConfig, ...]
    apply_chat_template: bool = False
    max_gen_toks: int = 2048
    max_eval_instances: int | None = None
    num_concurrent: int = DEFAULT_NUM_CONCURRENT
    extra_gen_kwargs: dict[str, str] = field(default_factory=dict)
    runtime: EvalchemyRuntimeConfig = field(default_factory=EvalchemyRuntimeConfig)


@dataclass(frozen=True)
class EvalchemyOutcome:
    """A completed result tree and its child job identity."""

    jobs: dict[str, str]
    result: EvalchemyResult


def _task_dir(task: EvalTaskConfig) -> str:
    """Return the durable subdirectory identity for one task configuration."""
    return task.task_alias or f"{task.name}_{task.num_fewshot}shot"


def _run_config_json(model: RunningModel, config: EvalchemyRunConfig, output_dir: str) -> str:
    """Build the plain JSON boundary consumed by the isolated Evalchemy image."""
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("Evalchemy requires RunningModel.tokenizer")
    return json.dumps(
        {
            "base_url": model.endpoint.base_url,
            "model_id": model.endpoint.model,
            "tokenizer": tokenizer,
            "tasks": [
                {
                    "name": task.name,
                    "num_fewshot": task.num_fewshot,
                    "dir": _task_dir(task),
                    "generation": task.generation,
                    "unsafe_code": task.unsafe_code,
                    "completion_only": task.completion_only,
                }
                for task in config.tasks
            ],
            "out_path": output_dir,
            "apply_chat_template": config.apply_chat_template,
            "max_gen_toks": config.max_gen_toks,
            "extra_gen_kwargs": dict(config.extra_gen_kwargs),
            "max_eval_instances": config.max_eval_instances,
            "num_concurrent": config.num_concurrent,
        }
    )


def _verify_durable_artifacts(output_dir: str) -> None:
    fs, path = url_to_fs(output_dir)
    objects = fs.find(path)
    results = [
        item
        for item in objects
        if item.rsplit("/", 1)[-1].startswith("results_") and item.endswith(".json")
    ]
    logger.info(
        "Durable Evalchemy artifacts under %s: %d object(s), %d result file(s)",
        output_dir,
        len(objects),
        len(results),
    )
    if not results:
        raise RuntimeError(f"no Evalchemy results_*.json landed under {output_dir!r}")


def _submit_evalchemy_child(
    model: RunningModel,
    config: EvalchemyRunConfig,
    output_dir: str,
    observer: InferenceObserver | None,
) -> str:
    client = iris_ctx().client
    child_id = uuid.uuid4().hex[:8]
    constraints = [region_constraint([config.runtime.region])] if config.runtime.region else None
    command = f'exec {EVALCHEMY_PYTHON} "$IRIS_WORKDIR/{_EVAL_CLIENT_SCRIPT}"'
    eval_job = client.submit(
        entrypoint=Entrypoint.from_command("bash", "-c", command),
        name=f"eval-{config.name.replace('.', '-')}-{child_id}",
        resources=ResourceSpec(
            cpu=config.runtime.cpu,
            memory=config.runtime.memory,
            disk=config.runtime.disk,
        ),
        environment=EnvironmentSpec(
            env_vars=_propagated_env(
                JAX_PLATFORMS="cpu",
                HF_ALLOW_CODE_EVAL="1",
                OPENAI_API_KEY="local-endpoint",
                TQDM_MININTERVAL="30",
                **{CONFIG_ENV_KEY: _run_config_json(model, config, output_dir)},
            )
        ),
        task_image=config.runtime.image,
        constraints=constraints,
        max_retries_failure=0,
    )
    eval_path = str(eval_job.job_id)
    logger.info(
        "Submitted Evalchemy job %s for %s against model %s",
        eval_job,
        config.name,
        model.endpoint.model,
    )
    try:
        while True:
            try:
                eval_job.wait(timeout=_CHILD_WAIT_SLICE_SECONDS)
                break
            except TimeoutError:
                pass
            if observer is not None and observer.endpoint_departed(model):
                eval_job.terminate()
                raise EndpointMovedError(
                    f"Evalchemy job {eval_path} terminated after its inference endpoint moved",
                    stage=PipelineStage.EVAL,
                    jobs={"eval": eval_path},
                    log_tails={"eval": job_log_tail(eval_job)},
                )
    except JobFailedError as exc:
        raise EvalPipelineError(
            f"Evalchemy job {eval_path} failed: {exc}",
            stage=PipelineStage.EVAL,
            jobs={"eval": eval_path},
            log_tails={"eval": job_log_tail(eval_job)},
        ) from exc
    return eval_path


def run_evalchemy(
    model: RunningModel,
    config: EvalchemyRunConfig,
    output_dir: str,
    *,
    observer: InferenceObserver | None = None,
) -> EvalchemyOutcome:
    """Run Evalchemy against ``model`` and validate its durable result tree."""
    if not config.tasks:
        raise ValueError("Evalchemy requires at least one task")
    if "://" not in output_dir:
        raise ValueError(f"Evalchemy output_dir {output_dir!r} is not an object-store path")
    eval_job = _submit_evalchemy_child(model, config, output_dir, observer)
    try:
        _verify_durable_artifacts(output_dir)
        parquets = export_lm_eval_samples(output_dir)
    except Exception as exc:
        raise EvalPipelineError(
            str(exc),
            stage=PipelineStage.ARTIFACTS,
            jobs={"eval": eval_job},
            log_tails={},
        ) from exc
    logger.info(
        "Evalchemy run %s wrote %d sample parquet file(s) under %s",
        config.name,
        len(parquets),
        output_dir,
    )
    return EvalchemyOutcome(
        jobs={"eval": eval_job},
        result=EvalchemyResult(path=output_dir),
    )
