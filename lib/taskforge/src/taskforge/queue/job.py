# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The boundary between a run's configuration and the host it runs on: a laptop or an Iris task.

``run_job`` is the one entry point. It reads the run's secrets once (``host_secrets``), resolves
the GLM endpoint, takes the machine factories for the host, places the run root, restores a
previous attempt's archive into it, copies the policy in, builds the run's services, runs
``queue.run.run_queue`` and writes ``summary.json``.

Under Iris, secrets travel in the job's environment, and Iris copies ``IRIS_JOB_ENV`` into every
child job, including each sandbox the model controls. ``scrub_child_environment`` therefore removes
the configured secret variables and the submitter keys Iris forwards (``HF_TOKEN``,
``WANDB_API_KEY``) from this process's environment and from ``IRIS_JOB_ENV`` before any sandbox
exists. On a laptop secrets are read from files and never enter the environment.
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from rigging.filesystem.storage_path import StoragePath

from taskforge.build.sdk import BuildServices
from taskforge.build.template import standard
from taskforge.canonical import pretty_json, write_atomic
from taskforge.ledger.finelog import run_ledger
from taskforge.ledger.records import Ledger
from taskforge.llm.agent import AgentTool
from taskforge.llm.client import GlmClient, GlmEndpoint, endpoint_in_task
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.rollout_model import GlmRolloutModel
from taskforge.llm.web import web_tools
from taskforge.loop.policy import POLICY
from taskforge.loop.program import LEDGER_DIR, LoopServices
from taskforge.proposal.source import ProposalSource
from taskforge.queue.config import LaptopGlm, ParallelKeyEnv, ParallelKeyFile, RelayGlm, RunConfig
from taskforge.queue.run import FailedItems, RunSummary, run_queue
from taskforge.sandbox.factories import MachineHost, factory_capabilities, machine_factories
from taskforge.triage.checks import Check, CheckContext
from taskforge.triage.program import RubricProgram
from taskforge.validate.controls import ServerTokenizer

GLM_TOKEN_KEY = "GLM_API_TOKEN"
PARALLEL_KEY = "PARALLEL_KEY"
# Iris copies these from the submitting process into every child job (iris.cluster.types.EnvironmentSpec).
SUBMITTER_KEYS = ("HF_TOKEN", "WANDB_API_KEY")
IRIS_JOB_ENV = "IRIS_JOB_ENV"
IRIS_CONTROLLER_URL_ENV = "IRIS_CONTROLLER_URL"
IRIS_OUTPUT_DIR_ENV = "IRIS_OUTPUT_DIR"
# A sandbox environment name is a credential when one of its ``_``-separated words is in SECRET_WORDS
# or it ends with one of SECRET_SUFFIXES (so GPG_KEY and TOKENIZERS_* are not).
SECRET_WORDS = frozenset({"SECRET", "TOKEN", "PASSWORD"})
SECRET_SUFFIXES = ("KEY_ID", "API_KEY", "ACCESS_KEY")
POLICY_FILE = "policy.json"
SUMMARY_FILE = "summary.json"
# Builders sample at the model maximum and continue on length; only validation rollouts may not continue.
BUILD_POLICY = LLMPolicy()


def read_key(path: Path, key: str) -> str:
    """The value of the ``<key>=`` line in ``path``; the value is never echoed."""
    for line in path.read_text().splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and name.strip() == key and value.strip():
            return value.strip()
    raise ValueError(f"{path} has no non-empty {key}= line")


def secret_names(names: list[str]) -> list[str]:
    """The credential-shaped names among ``names``, the environment variable names a sandbox sees."""
    return [name for name in names if SECRET_WORDS & set(name.split("_")) or name.endswith(SECRET_SUFFIXES)]


def scrub_child_environment(names: Sequence[str]) -> dict[str, str]:
    """Remove ``names`` from this process's environment and from ``IRIS_JOB_ENV``, which Iris copies into
    every child job. Returns the removed values that were set and non-empty."""
    removed = {name: os.environ[name] for name in names if os.environ.get(name)}
    for name in names:
        os.environ.pop(name, None)
    if IRIS_JOB_ENV in os.environ:
        job_env = json.loads(os.environ[IRIS_JOB_ENV] or "{}")
        for name in names:
            job_env.pop(name, None)
        os.environ[IRIS_JOB_ENV] = json.dumps(job_env)
    return removed


@dataclass(frozen=True)
class HostSecrets:
    """The run's secrets, read once; kept out of ``repr``."""

    glm_token: str = field(repr=False)
    parallel_key: str | None = field(repr=False)


def host_secrets(config: RunConfig) -> HostSecrets:
    """Read the GLM token and the Parallel key the config points at.

    Variables the config names are removed from the environment once read, and under Iris the
    submitter keys are removed too, so no sandbox child inherits a credential.
    """
    env_names = [
        *((config.glm.token_env,) if isinstance(config.glm, RelayGlm) else ()),
        *((config.web.env,) if isinstance(config.web, ParallelKeyEnv) else ()),
    ]
    submitter = SUBMITTER_KEYS if config.host is MachineHost.IRIS else ()
    environment = scrub_child_environment((*env_names, *submitter))
    missing = [name for name in env_names if name not in environment]
    if missing:
        raise ValueError(f"environment variables {missing} are unset or empty")
    match config.glm:
        case LaptopGlm(token_file=token_file):
            glm_token = read_key(token_file, GLM_TOKEN_KEY)
        case RelayGlm(token_env=token_env):
            glm_token = environment[token_env]
    match config.web:
        case ParallelKeyFile(path=path):
            parallel_key = read_key(path, PARALLEL_KEY)
        case ParallelKeyEnv(env=env):
            parallel_key = environment[env]
        case None:
            parallel_key = None
    return HostSecrets(glm_token=glm_token, parallel_key=parallel_key)


def glm_endpoint(config: RunConfig, token: str) -> GlmEndpoint:
    """The configured endpoint; a relay is resolved through Iris, so only inside an Iris task."""
    match config.glm:
        case LaptopGlm(base_url=base_url, pool=pool):
            return GlmEndpoint(base_url=base_url.rstrip("/"), token=token, pool=pool)
        case RelayGlm(relay_job=relay_job, pool=pool):
            return endpoint_in_task(relay_job, token, pool)


def run_root(config: RunConfig) -> Path:
    """The run root on this host: ``config.root`` on a laptop, under ``$IRIS_OUTPUT_DIR`` in an Iris task."""
    if config.host is MachineHost.LAPTOP:
        return config.root
    output_dir = os.environ.get(IRIS_OUTPUT_DIR_ENV)
    if not output_dir:
        raise ValueError(f"{IRIS_OUTPUT_DIR_ENV} is unset; an Iris run root lives in the attempt's output dir")
    return Path(output_dir) / config.root


def controller_url(host: MachineHost) -> str | None:
    if host is MachineHost.LAPTOP:
        return None
    url = os.environ.get(IRIS_CONTROLLER_URL_ENV)
    if not url:
        raise ValueError(f"{IRIS_CONTROLLER_URL_ENV} is unset; the IRIS host runs only inside an Iris task")
    return url


def restore(source: str, root: Path) -> None:
    """Copy a previous attempt's archived run root into ``root``, which must be empty or absent.

    The archive must hold a ``ledger/`` directory: a restore that would resume nothing raises.
    """
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"restore_from needs an empty run root; {root} has files")
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        root.rmdir()
    StoragePath(source.rstrip("/")).download_to(str(root), recursive=True)
    if not (root / LEDGER_DIR).is_dir():
        raise ValueError(f"restore_from {source} holds no {LEDGER_DIR}/ directory")


@dataclass(frozen=True)
class RunInputs[IdeaT]:
    """What a run takes beyond its config: its ideas, the source that proposes from them, and triage's checks.

    The capability layer supplies these for the capability catalog.
    """

    ideas: Mapping[str, IdeaT]
    source: ProposalSource[IdeaT]
    checks: tuple[Check, ...]
    rubric: RubricProgram
    check_context: CheckContext


type InputsFactory[IdeaT] = Callable[[GlmClient, Path], RunInputs[IdeaT]]
"""Builds a run's inputs from the run's GLM client and run root (a source and a rubric call GLM and
keep their call stores under the root)."""


def prepare_root(config: RunConfig) -> Path:
    """Place the run root on this host, restore ``config.restore_from`` into it, and copy the policy in."""
    root = run_root(config)
    if config.restore_from is not None:
        restore(config.restore_from, root)
    root.mkdir(parents=True, exist_ok=True)
    write_atomic(root / POLICY_FILE, POLICY.dump_json(config.policy, indent=2))
    return root


@asynccontextmanager
async def loop_services[IdeaT](
    config: RunConfig,
    endpoint: GlmEndpoint,
    secrets: HostSecrets,
    root: Path,
    ledger: Ledger,
    inputs: InputsFactory[IdeaT],
) -> AsyncIterator[tuple[LoopServices[IdeaT], Mapping[str, IdeaT]]]:
    """The run's ``LoopServices`` and ideas: one GLM client, the host's factories, ``width`` slots.

    Builders sample at ``BUILD_POLICY``; the solver, adversaries and the control tokenizer at the
    validation policy's sampling.
    """
    factories = machine_factories(config.host, controller_url(config.host))
    sampling = config.policy.validation.sampling
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(GlmClient(endpoint))
        tools: tuple[AgentTool, ...] = ()
        if secrets.parallel_key is not None:
            http = await stack.enter_async_context(httpx.AsyncClient())
            tools = web_tools(http, secrets.parallel_key)
        run_inputs = inputs(client, root)
        services = LoopServices(
            client=client,
            source=run_inputs.source,
            checks=run_inputs.checks,
            rubric=run_inputs.rubric,
            check_context=run_inputs.check_context,
            template=standard,
            build=BuildServices(client=client, policy=BUILD_POLICY, factories=factories, ledger=ledger, web_tools=tools),
            engine=config.engine.settings(factories, factory_capabilities(config.host)),
            rollout_model=GlmRolloutModel(client, sampling),
            tokenize=ServerTokenizer(client, sampling),
            ledger=ledger,
            root=root,
            slots=asyncio.Semaphore(config.width),
        )
        yield services, run_inputs.ideas


async def run_job[IdeaT](config: RunConfig, inputs: InputsFactory[IdeaT], failed: FailedItems) -> RunSummary:
    """Run ``config`` on this host to completion and write ``summary.json`` into the run root."""
    secrets = host_secrets(config)
    endpoint = glm_endpoint(config, secrets.glm_token)
    root = prepare_root(config)
    with run_ledger(root / LEDGER_DIR, config.run_id) as ledger:
        async with loop_services(config, endpoint, secrets, root, ledger, inputs) as (services, ideas):
            summary = await run_queue(ideas, config.policy, services, failed)
    write_atomic(root / SUMMARY_FILE, pretty_json({"run_id": config.run_id, **summary.summary_json()}).encode())
    return summary
