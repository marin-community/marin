# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run immutable evaluation corpora against a launcher-owned inference session."""

import json
import logging
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from evalrecords.records import CuratedSettings, RunStatus
from iris.client.client import JobFailedError, iris_ctx
from iris.cluster.setup_scripts import default_setup_script
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from openai import OpenAI
from rigging.timing import Duration
from rolloutengine.models import RolloutLimits

from marin.evaluation.eval_env import EVALUATION_REQUEST_TIMEOUT_SECONDS
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
    DEFAULT_MAX_ATTEMPTS,
    JUDGE_API_KEY,
    JUDGE_ENDPOINT,
    SOLVER_API_KEY,
    SOLVER_ENDPOINT,
    WORKER_PYTHON,
    WORKER_TIMEOUT,
    execution_config,
    run_worker,
)
from marin.inference.iris import RemoteInferenceSession
from marin.inference.types import InferenceBackendState
from marin.rollouts.evaluation import EvaluationRunConfig, ModelIdentity


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
        ),
        max_tokens=min(settings.max_tokens or context, model.generation.max_gen_toks or context, context - 1),
        generation_extra_body={key: json.loads(value) for key, value in model.generation.extra_gen_kwargs.items()},
        chat_template_kwargs=dict(model.generation.chat_template_kwargs),
        thinking_off_template_kwargs=dict(model.generation.thinking_off_template_kwargs),
        concurrency=settings.concurrency,
        source_limit=settings.source_limit,
        seed=settings.seed,
        judge=settings.judge,
        limits=RolloutLimits(
            max_turns=1,
            max_total_tokens=None,
            model_turn_timeout=EVALUATION_REQUEST_TIMEOUT_SECONDS,
            tool_turn_timeout=None,
            total_turn_timeout=EVALUATION_REQUEST_TIMEOUT_SECONDS,
            attempt_timeout=2400,
            verifier_timeout=600,
            cleanup_timeout=60,
        ),
    )


@dataclass(frozen=True)
class DatasetEvaluator:
    """Execute raw attempts; summary computation belongs to Evaldash."""

    datasets: tuple[str, ...]
    model: ModelConfig
    settings: CuratedSettings
    judge_endpoint: str | None = None

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
        config = runtime_config(self.model, self.settings, session)
        job = iris_ctx().client.submit(
            entrypoint=Entrypoint.from_callable(
                run_worker, self.datasets, output_dir, execution_config(config), DEFAULT_MAX_ATTEMPTS
            ),
            name=f"curated-eval-{uuid.uuid4().hex[:8]}",
            resources=ResourceSpec(cpu=8, memory="32g", disk="64g"),
            environment=EnvironmentSpec(
                setup_scripts=(default_setup_script(packages=("marin-core",), python_version=WORKER_PYTHON),),
                secrets=secrets,
                env_vars=dict(env_vars),
            ),
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
