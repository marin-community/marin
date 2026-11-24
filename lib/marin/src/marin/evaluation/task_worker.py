# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Connect published evaluation corpora to model transports and isolated graders."""

import asyncio
import logging
from collections.abc import Mapping
from types import MappingProxyType

from iris.cluster.setup_scripts import default_setup_script
from levanter.compat.hf_checkpoints import load_tokenizer
from openai import AsyncOpenAI
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import get_secret, require_secret
from rigging.timing import retry_with_backoff
from rolloutengine.agents.mini_swe.agent import MiniSweOptions, mini_swe_harness
from rolloutengine.agents.model_relay import VllmTokenizer
from rolloutengine.agents.pi.agent import PiOptions, PiThinking, pi_harness
from rolloutengine.agents.terminus.agent import TerminusOptions, TerminusParser, terminus_harness
from rolloutengine.chat import ChatModel
from rolloutengine.completion import CompletionModel
from rolloutengine.harness.core import Harness
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.likelihood import VllmLikelihood
from rolloutengine.models import MachineRuntimeSpec
from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.machine import Backend, MachineFactory, NetworkPolicy
from taskcompendium.evaluation.spec import EpisodeAgent, PiAgent
from verifyit.modes.grade_judge import JudgeConnection

from marin.rollouts.evaluation import (
    EpisodeBudget,
    EvaluationInfrastructureError,
    EvaluationRunConfig,
    HarnessFactory,
    run_evaluation,
)

DEFAULT_MAX_ATTEMPTS = 3
# The Iris image's native build exposes the Linux namespace and pidfd APIs Shellbox requires.
WORKER_PYTHON = "/usr/local/bin/python3.12"
WORKER_TIMEOUT = 36 * 60 * 60
SOLVER_ENDPOINT = "EVALUATION_SOLVER_ENDPOINT"
SOLVER_API_KEY = "EVALUATION_SOLVER_API_KEY"
JUDGE_ENDPOINT = "EVALUATION_JUDGE_ENDPOINT"
JUDGE_API_KEY = "EVALUATION_JUDGE_API_KEY"
DAYTONA_API_KEY = "DAYTONA_API_KEY"
"""The worker secret holding the Daytona eval-org key; launchers resolve it from ``DAYTONA_EVAL_API_KEY``."""
DAYTONA_EGRESS = ("app.daytona.io", "*.app.daytona.io")
MACHINE_BACKENDS = frozenset({Backend.LOCAL, Backend.DOCKER, Backend.DAYTONA})
MODEL_REQUEST_RETRIES = 8
"""Retries of a model request that fails with a rate limit, server error or lost connection.

Harbor's agents retry such a request up to nine times, and the inference proxy answers overload with
429 and ``Retry-After``; without retries, one rejected request ends an agent attempt. The client
backs off between retries, and the session's model-turn timeout bounds them all.
"""
# Records carry these as agent provenance, keyed by agent kind. Bump a harness's revision when its prompts,
# parsing, tools or options change, so its runs get new identities instead of resuming older episodes.
# tests/evaluation/test_harness_revisions.py pins each revision to a digest of its harness's sources.
EPISODE_HARNESS_REVISIONS: Mapping[str, str] = MappingProxyType(
    {"shell": "2026.11.21.1", "terminus-2": "2026.11.21.1", "pi": "2026.11.21.1", "mini-swe-agent": "2026.11.21.1"}
)


def _shell(_agent: EpisodeAgent, _budget: EpisodeBudget) -> Harness:
    return ShellHarness(command_timeout=None)


def _terminus(_agent: EpisodeAgent, budget: EpisodeBudget) -> Harness:
    return terminus_harness(
        TerminusOptions(
            parser=TerminusParser.JSON,
            max_input_tokens=budget.max_input_tokens,
            max_output_tokens=budget.max_output_tokens,
            served_max_model_len=budget.context_window,
        )
    )


def episode_harnesses(config: EvaluationRunConfig, client: AsyncOpenAI) -> Mapping[str, HarnessFactory]:
    """The harness factory of each agent kind; each attempt builds its own instance for its model budgets.

    Pi and mini-swe-agent count their prompt tokens with the policy server's ``/tokenize`` through ``client``.
    """
    tokenizer = VllmTokenizer(client, config.model.name)

    def pi(agent: EpisodeAgent, budget: EpisodeBudget) -> Harness:
        assert isinstance(agent, PiAgent)
        if config.pi_thinking_format is None:
            raise ValueError("Pi episodes require the model's pi_thinking_format")
        options = PiOptions(
            model=config.model.name,
            thinking_format=config.pi_thinking_format,
            max_input_tokens=budget.max_input_tokens,
            max_output_tokens=budget.max_output_tokens,
            compaction_reserve_tokens=agent.compaction_reserve_tokens,
            version=agent.version,
            thinking=None if agent.thinking is None else PiThinking(agent.thinking),
            setup_timeout=agent.setup_timeout,
        )
        return pi_harness(options, tokenizer)

    def mini_swe(_agent: EpisodeAgent, budget: EpisodeBudget) -> Harness:
        options = MiniSweOptions(
            max_input_tokens=budget.max_input_tokens,
            max_output_tokens=budget.max_output_tokens,
            model_format=config.mini_swe_model_format,
        )
        return mini_swe_harness(options, tokenizer)

    factories = {"shell": _shell, "terminus-2": _terminus, "pi": pi, "mini-swe-agent": mini_swe}
    # Planning records a revision for each kind it accepts; the worker must build every one of them.
    assert factories.keys() == EPISODE_HARNESS_REVISIONS.keys(), sorted(
        factories.keys() ^ EPISODE_HARNESS_REVISIONS.keys()
    )
    return MappingProxyType(factories)


def machine_backends(config: EvaluationRunConfig) -> frozenset[str]:
    return frozenset(
        selection.backend for selection in (config.task_machine, config.verifier_machine) if selection is not None
    )


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
    config = config.model_copy(update={"verifier_machine": verifier})
    if unsupported := machine_backends(config) - MACHINE_BACKENDS:
        raise ValueError(f"This entrypoint has no factories for {sorted(unsupported)}; supply them to run_evaluation")
    return config


def worker_setup_script(config: EvaluationRunConfig) -> str:
    """Install the worker environment, with the Daytona SDK when the run selects Daytona machines."""
    extras = ("daytona",) if Backend.DAYTONA in machine_backends(config) else ()
    return default_setup_script(packages=("marin-core",), extras=extras, python_version=WORKER_PYTHON)


def worker_egress(config: EvaluationRunConfig) -> tuple[str, ...]:
    return DAYTONA_EGRESS if Backend.DAYTONA in machine_backends(config) else ()


def daytona_connection(api_key: str) -> DaytonaConnection:
    return DaytonaConnection(api_key=api_key)


def machine_factories(config: EvaluationRunConfig, daytona: DaytonaConnection | None) -> dict[str, MachineFactory]:
    """Construct the factory of each backend the run selects."""
    factories: dict[str, MachineFactory] = {}
    for backend in machine_backends(config):
        match backend:
            case Backend.LOCAL:
                factories[backend] = LocalMachineFactory()
            case Backend.DOCKER:
                factories[backend] = DockerMachineFactory()
            case Backend.DAYTONA:
                if daytona is None:
                    raise ValueError("Daytona machines require an explicit Daytona connection")
                # The Daytona SDK is optional; marin-core's daytona extra installs it.
                from shellbox.backends.daytona.machine import DaytonaImageBuild, DaytonaMachineFactory  # noqa: PLC0415

                factories[backend] = DaytonaMachineFactory(
                    client_factory=daytona.client_factory(), image_build=DaytonaImageBuild(config.image_build)
                )
            case _:
                raise ValueError(f"Evaluation workers have no factory for {backend!r}")
    return factories


def model_client(endpoint: str, api_key: str, timeout: float | None) -> AsyncOpenAI:
    """The served model's client, which retries transient request failures."""
    return AsyncOpenAI(base_url=endpoint, api_key=api_key, max_retries=MODEL_REQUEST_RETRIES, timeout=timeout)


async def execute(
    datasets: tuple[str, ...],
    output: str,
    config: EvaluationRunConfig,
    endpoint: str,
    api_key: str,
    judge_connection: JudgeConnection | None,
    daytona: DaytonaConnection | None = None,
    *,
    responses_from: str | None = None,
) -> None:
    """Evaluate the supplied corpora and persist raw results, retaining completed attempts on resume.

    Model responses stored in the result root ``responses_from`` answer identical requests.
    """
    configure_coreweave_s3()
    config = execution_config(config)
    tokenizer = load_tokenizer(config.model.tokenizer, revision=config.model.tokenizer_revision, trust_remote_code=False)
    prefix = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if prefix is None:
        raise ValueError("Likelihood scoring needs a tokenizer BOS or EOS prefix token")
    machines = machine_factories(config, daytona)
    async with model_client(endpoint, api_key, config.limits.model_turn_timeout) as client:
        await run_evaluation(
            datasets,
            output,
            config,
            # Each request sets its own output limit: the protocol's, or for Pi, Pi's.
            ChatModel(client, config.model.name, max_tokens=None),
            completion_model=CompletionModel(client, config.model.name),
            likelihood=VllmLikelihood(client, config.model.name, tokenizer, config.model.context_window, prefix),
            machine_factories=machines,
            harnesses=episode_harnesses(config, client),
            tokenizer=tokenizer,
            judge_connection=judge_connection,
            reuse_from=responses_from,
        )


def execute_with_retries(
    datasets: tuple[str, ...],
    output: str,
    config: EvaluationRunConfig,
    endpoint: str,
    api_key: str,
    judge_connection: JudgeConnection | None,
    *,
    daytona: DaytonaConnection | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    responses_from: str | None = None,
) -> None:
    """Resume infrastructure failures with the same run identity and Marin's standard backoff."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    retry_with_backoff(
        lambda: asyncio.run(
            execute(
                datasets, output, config, endpoint, api_key, judge_connection, daytona, responses_from=responses_from
            )
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
    daytona = (
        daytona_connection(require_secret(DAYTONA_API_KEY)) if Backend.DAYTONA in machine_backends(config) else None
    )
    execute_with_retries(
        datasets,
        output,
        config,
        require_secret(SOLVER_ENDPOINT),
        require_secret(SOLVER_API_KEY),
        judge,
        daytona=daytona,
        max_attempts=max_attempts,
        responses_from=responses_from,
    )
