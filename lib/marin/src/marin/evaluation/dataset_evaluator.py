# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run immutable evaluation corpora against a launcher-owned inference session."""

import json
import logging
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from evalrecords.records import CuratedAgentRef, CuratedSettings, RunStatus, TaskImageBuild
from iris.client.client import JobFailedError, iris_ctx
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from openai import OpenAI
from rigging.timing import Duration
from rolloutengine.agents.mini_swe.agent import MiniSweModelFormat
from rolloutengine.agents.pi.agent import PiThinkingFormat
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits
from shellbox.machine import Backend, NetworkPolicy

from marin.evaluation.model_config import ModelConfig
from marin.evaluation.runner import (
    BACKEND_POLL_SECONDS,
    EvaluationError,
    EvaluationOutcome,
    InferenceBackendFinished,
    InferenceUnavailable,
    check_inference_backend,
    job_diagnostics,
)
from marin.evaluation.task_worker import (
    DAYTONA_API_KEY,
    DEFAULT_MAX_ATTEMPTS,
    JUDGE_API_KEY,
    JUDGE_ENDPOINT,
    SOLVER_API_KEY,
    SOLVER_ENDPOINT,
    WORKER_TIMEOUT,
    execution_config,
    machine_backends,
    run_worker,
    worker_egress,
    worker_setup_script,
)
from marin.inference.iris import RemoteInferenceSession
from marin.inference.types import InferenceBackendState
from marin.rollouts.evaluation import EvaluationRunConfig, ModelIdentity

RUN_CLEANUP_TIMEOUT = 60
VERIFIER_TIMEOUT = 600


PI_THINKING_FORMAT = "thinking_format"
MINI_SWE_MODEL_FORMAT = "model_format"
AGENT_EXTRA_BODY = "extra_body"


def run_max_tokens(context: int, settings: CuratedSettings, model: ModelConfig) -> int:
    """The run's reply budget under a served context window."""
    return min(settings.max_tokens or context, model.generation.max_gen_toks or context, context - 1)


def pi_thinking_format(model: ModelConfig) -> PiThinkingFormat | None:
    """How Pi switches this model's thinking, from the ``thinking_format`` agent argument."""
    value = model.agent.agent_kwargs.get(PI_THINKING_FORMAT)
    return None if value is None else PiThinkingFormat(value)


def mini_swe_model_format(model: ModelConfig) -> MiniSweModelFormat:
    """How mini-swe-agent formats this model's history, from the ``model_format`` agent argument, as in Harbor."""
    return MiniSweModelFormat(model.agent.agent_kwargs.get(MINI_SWE_MODEL_FORMAT, MiniSweModelFormat.OPENAI))


def runtime_config(
    model: ModelConfig, settings: CuratedSettings, session: RemoteInferenceSession
) -> EvaluationRunConfig:
    """Read the running server's context limit before admitting any evaluation rows."""
    endpoint = session.model.endpoint
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    with OpenAI(base_url=endpoint.base_url, api_key=endpoint.api_key or "local-endpoint") as client:
        served = next(model for model in client.models.list().data if model.id == endpoint.model)
    context = (served.model_extra or {}).get("max_model_len")
    if not isinstance(context, int) or context < 2:
        raise ValueError(f"Server {endpoint.model} did not report a usable max_model_len")
    assert model.effective_tokenizer_revision is not None
    return EvaluationRunConfig(
        model=ModelIdentity(
            name=endpoint.model,
            revision=model.revision or model.identity or model.location,
            tokenizer=model.tokenizer or model.location,
            tokenizer_revision=model.effective_tokenizer_revision,
            context_window=context,
            serving={"max_model_len": str(context)},
            chat_template=model.serve.chat_template,
        ),
        max_tokens=run_max_tokens(context, settings, model),
        generation_extra_body={key: json.loads(value) for key, value in model.generation.extra_gen_kwargs.items()},
        chat_template_kwargs=dict(model.generation.chat_template_kwargs),
        thinking_off_template_kwargs=dict(model.generation.thinking_off_template_kwargs),
        agent_extra_body=json.loads(model.agent.agent_kwargs.get(AGENT_EXTRA_BODY, "{}")),
        pi_thinking_format=pi_thinking_format(model),
        mini_swe_model_format=mini_swe_model_format(model),
        concurrency=settings.concurrency,
        source_limit=settings.source_limit,
        seed=settings.seed,
        judge=settings.judge,
        task_evidence=settings.task_evidence,
        limits=RolloutLimits(
            max_turns=1,
            max_total_tokens=None,
            model_turn_timeout=model.evaluation_request_timeout,
            tool_turn_timeout=None,
            total_turn_timeout=model.evaluation_request_timeout,
            attempt_timeout=model.evaluation_request_timeout + VERIFIER_TIMEOUT,
            verifier_timeout=VERIFIER_TIMEOUT,
            cleanup_timeout=RUN_CLEANUP_TIMEOUT,
        ),
        image_build=(
            settings.task_machine.image_build if settings.task_machine is not None else TaskImageBuild.SNAPSHOT
        ),
        task_machine=(
            MachineRuntimeSpec(
                backend=Backend(settings.task_machine.backend),
                # Agent episodes replace the network policy and GPUs with each task's own.
                network=NetworkPolicy.DENY,
                cpus=settings.task_machine.cpus,
                memory_mb=settings.task_machine.memory_mb,
                storage_mb=settings.task_machine.storage_mb,
                gpus=0,
                user=None,
                startup_timeout=settings.task_machine.startup_timeout,
                cleanup_timeout=RUN_CLEANUP_TIMEOUT,
            )
            if settings.task_machine is not None
            else None
        ),
    )


@dataclass(frozen=True)
class DatasetEvaluator:
    """Execute raw attempts; summary computation belongs to Evaldash."""

    datasets: tuple[str, ...]
    model: ModelConfig
    settings: CuratedSettings
    judge_endpoint: str | None = None
    agent: CuratedAgentRef | None = None
    """The recorded agent plan, whose budgets assume the server runs ``serve.max_model_len``."""
    task_secrets: tuple[str, ...] = ()
    """Launch variables that task machines read through ``${NAME}``; the worker receives them as secrets."""

    def __call__(
        self,
        session: RemoteInferenceSession,
        output_dir: str,
        env_vars: Mapping[str, str],
        *,
        backend_state: Callable[[], InferenceBackendState],
        judge: RemoteInferenceSession | None = None,
    ) -> EvaluationOutcome:
        endpoint = session.model.endpoint
        secrets = {SOLVER_ENDPOINT: endpoint.base_url, SOLVER_API_KEY: endpoint.api_key or "local-endpoint"}
        if self.settings.judge is not None:
            if judge is None:
                assert self.judge_endpoint is not None
                secrets.update({JUDGE_ENDPOINT: self.judge_endpoint, JUDGE_API_KEY: env_vars[JUDGE_API_KEY]})
            else:
                secrets.update(
                    {
                        JUDGE_ENDPOINT: judge.model.endpoint.base_url,
                        JUDGE_API_KEY: judge.model.endpoint.api_key or "local-judge",
                    }
                )
        secrets.update({name: env_vars[name] for name in self.task_secrets})
        config = execution_config(
            runtime_config(self.model, self.settings, session).model_copy(update={"task_secrets": self.task_secrets})
        )
        if self.agent is not None and config.model.context_window != self.model.serve.max_model_len:
            raise ValueError(
                f"Server context {config.model.context_window} differs from the planned agent window "
                f"{self.model.serve.max_model_len}"
            )
        if Backend.DAYTONA in machine_backends(config):
            secrets[DAYTONA_API_KEY] = env_vars[DAYTONA_API_KEY]
        job = iris_ctx().client.submit(
            entrypoint=Entrypoint.from_callable(run_worker, self.datasets, output_dir, config, DEFAULT_MAX_ATTEMPTS),
            name=f"curated-eval-{uuid.uuid4().hex[:8]}",
            resources=ResourceSpec(cpu=8, memory="12g", disk="64g"),
            environment=EnvironmentSpec(
                setup_scripts=(worker_setup_script(config),),
                secrets=secrets,
                env_vars={key: value for key, value in env_vars.items() if key not in secrets},
            ),
            egress_allow=worker_egress(config),
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
            timeout=Duration.from_seconds(WORKER_TIMEOUT),
            max_retries_failure=0,
        )
        try:
            while True:
                check_inference_backend(backend_state())
                try:
                    job.wait(timeout=BACKEND_POLL_SECONDS)
                    break
                except TimeoutError:
                    continue
            check_inference_backend(backend_state())
        except (InferenceUnavailable, InferenceBackendFinished):
            job.cancel()
            job.wait(timeout=float("inf"), raise_on_failure=False)
            raise
        except JobFailedError as error:
            check_inference_backend(backend_state())
            jobs, tails = job_diagnostics((job,), "eval")
            raise EvaluationError(
                f"Curated evaluation failed: {error}", status=RunStatus.FAILED, jobs=jobs, log_tails=tails
            ) from error
        return EvaluationOutcome(metrics={}, jobs={"eval": str(job.job_id)})
