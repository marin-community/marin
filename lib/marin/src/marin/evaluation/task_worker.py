# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Connect published evaluation corpora to model transports and isolated graders."""

import asyncio
import contextlib
import logging

from levanter.compat.hf_checkpoints import load_tokenizer
from openai import AsyncOpenAI
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import get_secret, require_secret
from rigging.timing import retry_with_backoff
from rolloutengine.chat import ChatModel
from rolloutengine.completion import CompletionModel
from rolloutengine.likelihood import VllmLikelihood
from rolloutengine.models import MachineRuntimeSpec, ModelRequest, ModelResponseError, ModelTurn
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.machine import Backend, NetworkPolicy
from verifyit.modes.grade_judge import JudgeConnection

from marin.rollouts.evaluation import EvaluationInfrastructureError, EvaluationRunConfig, run_evaluation

DEFAULT_MAX_ATTEMPTS = 3
# The Iris image's native build exposes the Linux namespace and pidfd APIs Shellbox requires.
WORKER_PYTHON = "/usr/local/bin/python3.12"
WORKER_TIMEOUT = 36 * 60 * 60
SOLVER_ENDPOINT = "EVALUATION_SOLVER_ENDPOINT"
SOLVER_API_KEY = "EVALUATION_SOLVER_API_KEY"
JUDGE_ENDPOINT = "EVALUATION_JUDGE_ENDPOINT"
JUDGE_API_KEY = "EVALUATION_JUDGE_API_KEY"


def execution_config(config: EvaluationRunConfig) -> EvaluationRunConfig:
    """Default to network-denied Shellbox graders, preserving explicit machine settings."""
    verifier = config.verifier_machine or MachineRuntimeSpec(
        backend=Backend.LOCAL,
        network=NetworkPolicy.DENY,
        cpus=None,
        memory_mb=None,
        storage_mb=None,
        gpus=0,
        user=None,
        startup_timeout=600,
        cleanup_timeout=config.limits.cleanup_timeout,
    )
    backends = {selection.backend for selection in (config.task_machine, verifier) if selection is not None}
    if unsupported := backends - {Backend.LOCAL}:
        raise ValueError(f"This entrypoint has no factories for {sorted(unsupported)}; supply them to run_evaluation")
    return config.model_copy(update={"verifier_machine": verifier})


async def _no_endpoint(_request: ModelRequest) -> ModelTurn:
    raise ModelResponseError("No endpoint was supplied, and no stored model response answers this request", response={})


async def execute(
    datasets: tuple[str, ...],
    output: str,
    config: EvaluationRunConfig,
    endpoint: str | None,
    api_key: str | None,
    judge_connection: JudgeConnection | None,
    *,
    responses_from: str | None = None,
) -> None:
    """Evaluate the supplied corpora and persist raw results, retaining completed attempts on resume.

    Model responses stored in the result root ``responses_from`` answer identical requests. Without an endpoint,
    the other generation attempts fail as retryable, and corpora scored by likelihood are refused.
    """
    configure_coreweave_s3()
    config = execution_config(config)
    tokenizer = load_tokenizer(config.model.tokenizer, revision=config.model.tokenizer_revision, trust_remote_code=False)
    prefix = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if endpoint is not None and prefix is None:
        raise ValueError("Likelihood scoring needs a tokenizer BOS or EOS prefix token")
    machines = {Backend.LOCAL: LocalMachineFactory()}
    connection = (
        AsyncOpenAI(base_url=endpoint, api_key=api_key, max_retries=0, timeout=config.limits.model_turn_timeout)
        if endpoint is not None
        else contextlib.nullcontext()
    )
    async with connection as client:
        await run_evaluation(
            datasets,
            output,
            config,
            ChatModel(client, config.model.name, config.max_tokens) if client is not None else _no_endpoint,
            completion_model=CompletionModel(client, config.model.name) if client is not None else _no_endpoint,
            likelihood=(
                VllmLikelihood(client, config.model.name, tokenizer, config.model.context_window, prefix)
                if client is not None
                else None
            ),
            machine_factories=machines,
            tokenizer=tokenizer,
            judge_connection=judge_connection,
            reuse_from=responses_from,
        )


def execute_with_retries(
    datasets: tuple[str, ...],
    output: str,
    config: EvaluationRunConfig,
    endpoint: str | None,
    api_key: str | None,
    judge_connection: JudgeConnection | None,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    responses_from: str | None = None,
) -> None:
    """Resume infrastructure failures with the same run identity and Marin's standard backoff."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    retry_with_backoff(
        lambda: asyncio.run(
            execute(datasets, output, config, endpoint, api_key, judge_connection, responses_from=responses_from)
        ),
        retryable=lambda error: isinstance(error, EvaluationInfrastructureError),
        max_attempts=max_attempts,
        operation="TaskSpec evaluation",
    )


def run_worker(
    datasets: tuple[str, ...],
    output: str,
    config: EvaluationRunConfig,
    max_attempts: int,
    responses_from: str | None = None,
) -> None:
    """Execute the corpora and persist resumable raw results using Iris-provided connections."""
    logging.basicConfig(level=logging.INFO)
    # Iris endpoint URLs can contain scoped credentials.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    judge_endpoint = get_secret(JUDGE_ENDPOINT)
    judge = JudgeConnection(judge_endpoint, require_secret(JUDGE_API_KEY)) if judge_endpoint else None
    endpoint = (get_secret(SOLVER_ENDPOINT) or None) if responses_from else require_secret(SOLVER_ENDPOINT)
    execute_with_retries(
        datasets,
        output,
        config,
        endpoint,
        require_secret(SOLVER_API_KEY) if endpoint else None,
        judge,
        max_attempts=max_attempts,
        responses_from=responses_from,
    )
