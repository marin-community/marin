# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one build in a child process, under a wall clock enforced by killing the process.

``run_build`` runs a program's steps on its caller's event loop. A step that computes in pure
Python never yields, so neither ``STEP_TIMEOUT`` nor any other deadline on that loop can fire, and
every other workpiece on the loop waits with it. ``run_build_in_process`` runs one workpiece's round in a
child process instead, ``python -m taskforge.stage40_build_with_program.build_process <request> <result>``, and
the driver's loop goes on serving other workpieces while it waits.

Everything a step uses is a network or file service already: the GLM router over HTTP, the Iris
controller for sandboxes, the step store on S3 or local disk, and the JSONL ledger. The child
therefore receives no objects, only a ``BuildRequest`` written as JSON to
``<round_dir>/build-request.json``, and constructs its own model client and ``RLEnvBuildServices``
from it. Secrets stay out of that file: ``BuildModelConfig.token_env``,
``BuildServiceConfig.parallel_key_env`` and ``BuildServiceConfig.registry_credential_env`` name the
variables of the inherited environment that hold the GLM token, the Parallel key and the registry
credential. The child calls ``run_build``, which writes the draft under
``round_dir`` as it does in process, writes a ``BuildResult`` to ``<round_dir>/build-result.json``
and exits 0. Its log goes to ``<round_dir>/build-process.log``.

Two deadlines bound a build, and neither presumes whose fault an expiry is: each raises
``BuildExpired``, which a driver holds as abandoned for someone to read. ``STEP_TIMEOUT`` is the
first, inside the child: a step that awaits past it expires with clock ``STEP``, naming the step.
The wall clock of ``run_build_in_process`` is the last, and it holds against a step that never
yields. When it expires, the child's process group receives SIGTERM and, ``TERMINATE_GRACE`` seconds
later, SIGKILL, and the call raises ``BuildExpired`` with clock ``WALL``, naming no step. A call
cancelled while it waits stops the child the same way.

``run_build_in_process`` raises what ``run_build`` would have raised in process, so a caller
handles the two alike: ``BuildFailure`` with its step, ``BuildExpired`` with its clock and step,
``BuildInfrastructureFailure`` with its cause, ``ModelUnavailable`` and ``OSError`` with their
messages, and for any other exception a ``RuntimeError`` whose first line is the exception's type and
message, followed by the child's traceback. A child that exits without a result raises
``BuildInfrastructureFailure`` with cause ``BUILD_PROCESS`` and the end of its log.

The child calls the model at the fixed ``BuildModelConfig.base_url``. A driver behind an Iris relay
resolves the relay's endpoint for each request, so every build starts wherever the relay is now.
With ``BuildServiceConfig.image_cluster`` set, the child publishes images with an
``IrisImageBuilder`` that submits its jobs to ``controller_url`` and pushes with the credential it
parses from ``registry_credential_env`` when it starts. The child reads no secret store, so the host
needs neither ``gcloud`` nor Google credentials; whoever launches the host's job reads the secret
with ``image_builder.registry_credential_json`` at submit time and puts the JSON in that variable.
Without an image cluster, as on a laptop, ``RLEnvBuildServices.images`` is ``None`` and a program
that publishes an image fails with ``NO_IMAGE_BUILDER``.
"""

import asyncio
import logging
import os
import sys
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import httpx
from pydantic import Field, TypeAdapter
from rigging.filesystem.atomic import atomic_rename
from rigging.log_setup import configure_logging

from taskforge.child_process import stop_process_group
from taskforge.ledger.jsonl import JsonlLedger
from taskforge.llm.agent import ReasoningReplay
from taskforge.llm.client import ModelClient, ModelUnavailable
from taskforge.llm.glm_client import FixedEndpoint, GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy, LLMProviderConfig
from taskforge.llm.web import web_tools
from taskforge.sandbox.factories import MachineHost, machine_factories
from taskforge.sandbox.image_builder import IrisImageBuilder, RegistryCredential
from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage30_author.loader import load_program_source
from taskforge.stage40_build_with_program.context import BuildExpired, BuildFailure, BuildInfrastructureFailure, ExpiredClock
from taskforge.stage40_build_with_program.infrastructure import InfrastructureCause
from taskforge.stage40_build_with_program.run import DRAFT_DIR, ResearchRequirement, TaskDraft, load_draft, run_build
from taskforge.stage40_build_with_program.services import AgentLimits, RLEnvBuildServices

MODULE = "taskforge.stage40_build_with_program.build_process"
REQUEST_FILE = "build-request.json"
RESULT_FILE = "build-result.json"
LOG_FILE = "build-process.log"
LOG_TAIL_LINES = 20
"""Lines of the child's log carried by the failure of a child that wrote no result."""


@dataclass(frozen=True)
class BuildModelConfig:
    """The GLM endpoint and provider settings the child builds its ``GlmClient`` from.

    Attributes:
        base_url: The router's OpenAI-compatible base URL, ending in ``/v1``.
        pool: The router pool the token selects.
        token_env: The environment variable holding the pool's bearer token.
        provider: The model id, retries, timeouts and connection pool.
    """

    base_url: str
    pool: Pool
    token_env: str
    provider: LLMProviderConfig


@dataclass(frozen=True)
class BuildServiceConfig:
    """What the child builds its ``RLEnvBuildServices`` from.

    Attributes:
        host: Where the build runs, as ``machine_factories`` takes it.
        controller_url: The Iris controller sandboxes are submitted to on ``MachineHost.IRIS``;
            ``None`` on a laptop.
        parallel_key_env: The environment variable holding the Parallel API key for the web
            tools, or ``None`` for a build without web tools.
        ledger_root: The ``JsonlLedger`` root, or ``None`` to record nothing.
        reasoning_replay: Whether every agent loop replays its assistant turns' reasoning.
        agent_limits: The bounds of every agent loop.
        image_cluster: The Iris cluster the image builder's build and push jobs run on, submitted
            through ``controller_url``, or ``None`` for a build without an image builder. The
            registry is the credential's; ``sandbox_effects.image_repository`` names the repository.
        registry_credential_env: The environment variable holding the registry credential's JSON
            (``registry``, ``user``, ``password``), required with ``image_cluster``. The host's
            launcher reads it from Secret Manager with ``image_builder.registry_credential_json``
            at submit time and sets it, conventionally as ``image_builder.REGISTRY_CREDENTIAL_ENV``;
            the host itself never reads Secret Manager.
    """

    host: MachineHost
    controller_url: str | None
    parallel_key_env: str | None
    ledger_root: Path | None
    reasoning_replay: ReasoningReplay
    agent_limits: AgentLimits
    image_cluster: str | None
    registry_credential_env: str | None


@dataclass(frozen=True)
class BuildRequest:
    """One workpiece's round, as ``run_build`` takes it, with the program as source and the host as config."""

    workpiece_id: str
    round: int
    program_source: str
    proposal: TaskProposal
    round_dir: Path
    store_root: str
    model: BuildModelConfig
    sampling: LLMPolicy
    research: ResearchRequirement
    invalidate: frozenset[str]
    model_digest: str | None
    services: BuildServiceConfig


class Raised(StrEnum):
    """The kind of exception other than a build failure, expiry or host failure that ended a build in the child."""

    MODEL_UNAVAILABLE = "model_unavailable"
    OS_ERROR = "os_error"
    OTHER = "other"


@dataclass(frozen=True)
class Built:
    """``run_build`` returned, and its draft is in ``<round_dir>/draft/``."""

    kind: Literal["built"] = "built"


@dataclass(frozen=True)
class Failed:
    """``run_build`` raised ``BuildFailure``."""

    message: str
    step: str | None
    kind: Literal["failed"] = "failed"


@dataclass(frozen=True)
class Expired:
    """``run_build`` raised ``BuildExpired``."""

    clock: ExpiredClock
    step: str | None
    message: str
    kind: Literal["expired"] = "expired"


@dataclass(frozen=True)
class HostFailed:
    """``run_build`` raised ``BuildInfrastructureFailure``."""

    cause: str
    message: str
    kind: Literal["host_failed"] = "host_failed"


@dataclass(frozen=True)
class Crashed:
    """``run_build`` or the program's loading raised another exception.

    ``text`` is the exception's type and message, a blank line, then the child's traceback.
    """

    raised: Raised
    text: str
    kind: Literal["crashed"] = "crashed"


BuildResult = Annotated[Built | Failed | Expired | HostFailed | Crashed, Field(discriminator="kind")]

REQUEST: TypeAdapter[BuildRequest] = TypeAdapter(BuildRequest)
RESULT: TypeAdapter[BuildResult] = TypeAdapter(BuildResult)


async def run_build_in_process(request: BuildRequest, *, wall_timeout: float) -> TaskDraft:
    """Run ``request`` in a child process for at most ``wall_timeout`` seconds and return its draft.

    Raises:
        BuildFailure: the build failed as the program's.
        BuildExpired: the build ran past ``wall_timeout`` (``WALL``), or raised one in the child.
        BuildInfrastructureFailure: the host failed, or the child wrote no result.
        ModelUnavailable: the model stayed unreachable.
        OSError: the child hit a storage fault.
        RuntimeError: the build raised any other exception; the message holds its text.
    """
    round_dir = request.round_dir
    round_dir.mkdir(parents=True, exist_ok=True)
    request_path, result_path, log_path = round_dir / REQUEST_FILE, round_dir / RESULT_FILE, round_dir / LOG_FILE
    request_path.write_bytes(REQUEST.dump_json(request, indent=2))
    result_path.unlink(missing_ok=True)
    with log_path.open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            MODULE,
            str(request_path),
            str(result_path),
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    try:
        await asyncio.wait_for(process.wait(), wall_timeout)
    except TimeoutError:
        raise BuildExpired(f"the build ran for more than {wall_timeout:.0f} seconds", ExpiredClock.WALL) from None
    finally:
        await stop_process_group(process)
    if not result_path.exists():
        lines = log_path.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:]
        raise BuildInfrastructureFailure(
            InfrastructureCause.BUILD_PROCESS,
            f"the build process exited with status {process.returncode} without a result; its log ends:\n"
            + "\n".join(lines),
        )
    result = RESULT.validate_json(result_path.read_bytes())
    if isinstance(result, Built):
        return load_draft(round_dir / DRAFT_DIR)
    if isinstance(result, Failed):
        raise BuildFailure(result.message, result.step)
    if isinstance(result, Expired):
        raise BuildExpired(result.message, result.clock, result.step)
    if isinstance(result, HostFailed):
        raise BuildInfrastructureFailure(result.cause, result.message)
    if result.raised is Raised.MODEL_UNAVAILABLE:
        raise ModelUnavailable(result.text, ())
    if result.raised is Raised.OS_ERROR:
        raise OSError(result.text)
    raise RuntimeError(result.text)


def _secret(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"the build reads a secret from {name}, which is not set")
    return value


def image_builder(config: BuildServiceConfig) -> IrisImageBuilder | None:
    """The child's image builder, or ``None`` when ``config`` names no image cluster.

    Raises:
        ValueError: ``config`` names an image cluster but no controller URL or no credential
            variable, or the variable is unset or does not hold a registry credential.
    """
    if config.image_cluster is None:
        return None
    if config.registry_credential_env is None:
        raise ValueError("an image cluster needs registry_credential_env, the variable holding the registry credential")
    credential = RegistryCredential.from_json(_secret(config.registry_credential_env))
    return IrisImageBuilder(
        controller_url=config.controller_url, target_cluster=config.image_cluster, credential=credential
    )


@asynccontextmanager
async def _resources(request: BuildRequest) -> AsyncIterator[tuple[ModelClient, RLEnvBuildServices]]:
    """The model client and services of ``request``, open for the build."""
    model, config = request.model, request.services
    endpoint = GlmEndpoint(model.base_url, _secret(model.token_env), model.pool)
    async with GlmClient(FixedEndpoint(endpoint), model.provider) as client, httpx.AsyncClient() as http:
        yield client, RLEnvBuildServices(
            host=config.host,
            factories=machine_factories(config.host, config.controller_url),
            images=image_builder(config),
            web_tools=() if config.parallel_key_env is None else web_tools(http, _secret(config.parallel_key_env)),
            ledger=None if config.ledger_root is None else JsonlLedger(config.ledger_root),
            reasoning_replay=config.reasoning_replay,
            agent_limits=config.agent_limits,
        )


async def build_result(request: BuildRequest) -> BuildResult:
    """Run ``request`` in this process and describe how ``run_build`` ended.

    An error constructing the model client or the services propagates, so the child exits without
    a result and the driver reports the host's failure.
    """
    async with _resources(request) as (model, services):
        try:
            await run_build(
                load_program_source(request.program_source),
                request.proposal,
                request.workpiece_id,
                request.round_dir,
                request.store_root,
                model,
                request.sampling,
                services,
                request.round,
                research=request.research,
                invalidate=request.invalidate,
                model_digest=request.model_digest,
            )
        except BuildFailure as error:
            return Failed(error.message, error.step)
        except BuildExpired as error:
            return Expired(error.clock, error.step, error.message)
        except BuildInfrastructureFailure as error:
            return HostFailed(error.cause, error.message)
        except ModelUnavailable as error:
            return _crashed(Raised.MODEL_UNAVAILABLE, error)
        except OSError as error:
            return _crashed(Raised.OS_ERROR, error)
        except Exception as error:
            return _crashed(Raised.OTHER, error)
    return Built()


def _crashed(raised: Raised, error: Exception) -> Crashed:
    return Crashed(raised, f"{type(error).__name__}: {error}\n\n{''.join(traceback.format_exception(error))}")


def main() -> None:
    """``python -m taskforge.stage40_build_with_program.build_process <request.json> <result.json>``."""
    _, request_path, result_path = sys.argv
    configure_logging(logging.INFO)
    request = REQUEST.validate_json(Path(request_path).read_bytes())
    result = asyncio.run(build_result(request))
    with atomic_rename(result_path) as temp:
        Path(temp).write_bytes(RESULT.dump_json(result, indent=2))


if __name__ == "__main__":
    main()
