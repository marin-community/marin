# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind real adapters and executable policy at the process boundary."""

import logging
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import override

import httpx
from iris.client.client import iris_ctx
from iris.cluster.client import get_job_info
from iris.cluster.types import JobName
from pydantic import TypeAdapter
from rolloutengine.models import ModelRequest, ModelTurn
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.machine import MachineFactory
from taskcompendium.models import EnvironmentRequirements, TaskResource
from taskforge.llm.glm_client import EndpointSource, FixedEndpoint, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy, LLMProviderConfig
from taskforge.llm.rollout_model import model_turn

from forge.arms import transitions
from forge.config import ForgeConfig, WorkspaceHydration
from forge.effects.context import Context
from forge.harness.loop import AgenticHarness
from forge.harness.tools import adversary_tools, standard_tools
from forge.harness.tools.validate import ValidateTool
from forge.harness.web import ParallelWeb, WebClient, WebFetch, WebSearch, fetch, search
from forge.lowering import (
    LoweringSettings,
    host_requirements,
    lowering_settings,
    machine_backend,
    machine_runtime,
    rollout_engine,
)
from forge.model.content_hash import digest
from forge.model.llm import Sampling
from forge.model.review import BandOutcome, BandRules, Budget
from forge.model.workpiece import Submission, Workpiece
from forge.policy import Policy
from forge.resources.bundle import Resources
from forge.resources.clock import Clock, Now, now
from forge.resources.env import Env, EnvFactory, ShellboxEnvFactory
from forge.resources.grading import GradeCandidate, GradingResource, grade
from forge.resources.ids import Ids, NewId, new_id
from forge.resources.inference import CompletionBackend, GlmInference, InferenceRequest, complete
from forge.resources.provider import EndpointLimitedClient, WaitingBackend
from forge.resources.rollouts import RolloutResource, SolverRollout, run_rollout
from forge.resources.sandbox.factories import (
    SANDBOX_SCHEDULING_TIMEOUT,
    MachineHost,
    local_factory,
    shellsim_factory,
)
from forge.stages.rules import default_review


# Source: acceptance-measure worktree, taskforge/llm/glm_client.py:168-196.
@dataclass(frozen=True, slots=True)
class RelayEndpoint(EndpointSource):
    """Resolve the live Iris relay; copied from Taskforge's acceptance-measure branch."""

    job: str
    token: str = field(repr=False)
    pool: Pool
    model: str

    @override
    def current(self) -> GlmEndpoint:
        client = iris_ctx().client
        if client is None:
            raise RuntimeError("Iris client is unavailable inside the task")
        endpoints = client.resolver_for_job(JobName.from_string(self.job)).resolve(self.model).endpoints
        if not endpoints:
            raise RuntimeError(f"GLM relay {self.job} has no registered endpoint for {self.model}")
        url = endpoints[0].url.rstrip("/")
        return GlmEndpoint(url if url.endswith("/v1") else f"{url}/v1", self.token, self.pool)


def endpoint_source(base_url: str | None, relay_job: str | None, token: str, model: str) -> EndpointSource:
    if bool(base_url) == bool(relay_job):
        raise ValueError("Set exactly one of GLM_RELAY_JOB and GLM_BASE_URL")
    if relay_job:
        return RelayEndpoint(relay_job, token, Pool.BULK, model)
    assert base_url is not None
    return FixedEndpoint(GlmEndpoint(base_url, token, Pool.BULK))


def install(ctx: Context, resources: Resources) -> None:
    for event, execute, resource in (
        (InferenceRequest, partial(complete, notify=ctx.emit), resources.inference),
        (SolverRollout, partial(run_rollout, notify=ctx.emit), resources.rollouts),
        (GradeCandidate, partial(grade, notify=ctx.emit), resources.grading),
        (Now, now, resources.clock),
        (NewId, new_id, resources.ids),
        (WebSearch, partial(search, notify=ctx.emit), resources.web),
        (WebFetch, partial(fetch, notify=ctx.emit), resources.web),
    ):
        ctx.resource(resource, resource.fingerprint)
        ctx.handle(event, execute, uses=(resource,))


@dataclass(frozen=True, slots=True)
class Environments(EnvFactory):
    factories: Mapping[str, MachineFactory]
    settings: LoweringSettings

    def __post_init__(self) -> None:
        object.__setattr__(self, "factories", MappingProxyType(dict(self.factories)))

    @override
    async def create(self, requirements: EnvironmentRequirements) -> Env:
        requirements = host_requirements(requirements, self.factories)
        runtime = machine_runtime(self.settings.task_machine, requirements, self.settings.host)
        factory = self.factories[machine_backend(requirements, self.settings.host).value]
        cleanup = self.settings.limits.cleanup_timeout
        if cleanup is None:
            raise ValueError("Environment cleanup timeout must be explicit")
        return await ShellboxEnvFactory(factory, runtime, cleanup, logging.info).create(requirements)


async def solver_turn(client: CompletionBackend, sampling: LLMPolicy, request: ModelRequest) -> ModelTurn:
    if request.continuation is not None or request.prompt_token_ids is not None:
        raise ValueError("GLM solver requires messages trace")
    return model_turn(await client.complete(request.messages, sampling, request.options))


def build_resources(
    policy: Policy, client: CompletionBackend, factories: Mapping[str, MachineFactory], web: WebClient
) -> Resources:
    settings = lowering_settings(policy)
    command_timeout = policy.setting("solver.command_timeout", float)
    engine = rollout_engine(factories, command_timeout=command_timeout)
    execution = digest((settings, command_timeout))
    provider = {key: value for key, value in policy.entries.items() if key.startswith("provider.")}
    sampling = TypeAdapter(LLMPolicy).validate_python(
        {key.removeprefix("sampling."): value for key, value in policy.entries.items() if key.startswith("sampling.")}
    )
    return Resources(
        Environments(factories, settings),
        GlmInference(client, digest(provider)),
        RolloutResource(engine, partial(solver_turn, client, sampling), digest((provider, sampling, execution))),
        GradingResource(engine, execution),
        Clock(),
        Ids(),
        factories,
        web,
    )


def submit_gate(submission: Submission) -> tuple[str, ...]:
    return () if submission.controls else ("Supply at least one control candidate",)


def hydration(_wp: Workpiece, _policy: Policy) -> tuple[TaskResource, ...]:
    return ()


def prompt_limit(policy: Policy) -> int:
    """Use the served model context without reserving an output allowance."""
    return policy.setting("provider.context_tokens", int)


def build_config(policy: Policy, arm: str, factories: Mapping[str, MachineFactory]) -> ForgeConfig:
    settings = lowering_settings(policy)
    sampling = TypeAdapter(Sampling).validate_python(
        {key.removeprefix("sampling."): value for key, value in policy.entries.items() if key.startswith("sampling.")}
    )
    author = AgenticHarness(
        prompt_limit(policy),
        sampling,
        policy.prompt("author.nudge").template,
        policy.setting("tools.output_chars", int),
    )
    adversaries = {
        name: AgenticHarness(
            prompt_limit(policy),
            sampling,
            "Finish your audit.",
            policy.setting("tools.output_chars", int),
        )
        for name in policy.setting("adversary.roster", tuple)
    }
    tools = {"author": standard_tools(policy=policy, factories=factories, settings=settings, gate=submit_gate)}
    root = policy.setting("author.workdir", str)
    if arm == "T":
        tools["author"] = {
            **tools["author"],
            "validate": ValidateTool(submit_gate, f"{root}/task.json", f"{root}/controls.json"),
        }
    tools.update(
        {
            name: adversary_tools(task_path=f"{root}/task.json", factories=factories, settings=settings)
            for name in adversaries
        }
    )
    budget = Budget(
        policy.setting("budget.max_modify_rounds", int), policy.setting("budget.max_validation_retries", int)
    )
    band = BandRules(
        policy.setting("band.low", float), policy.setting("band.high", float), frozenset({BandOutcome.IN_BAND})
    )
    return ForgeConfig(
        policy,
        author,
        adversaries,
        WorkspaceHydration((), hydration),
        tools,
        submit_gate,
        default_review,
        band,
        budget,
        transitions(arm, budget, policy),
    )


def provider_policy(policy: Policy) -> Policy:
    """Resolve trial deadlines so availability waits cannot exhaust them first."""
    window = policy.setting("provider.unavailable_wait", float)
    turn = max(policy.setting("solver.turn_timeout", float), window + policy.setting("provider.stall_timeout", float))
    attempt = max(policy.setting("solver.attempt_timeout", float), 2 * turn)
    resolved = policy.overlaid({"solver.turn_timeout": turn, "solver.attempt_timeout": attempt})
    for key in ("provider.unavailable_wait", "solver.turn_timeout", "solver.attempt_timeout"):
        print(f"POLICY {key} declared={policy.setting(key, float)} resolved={resolved.setting(key, float)}", flush=True)
    return resolved


@asynccontextmanager
async def runtime(
    policy: Policy, arm: str, source: EndpointSource, parallel_api_key: str
) -> AsyncIterator[tuple[ForgeConfig, Resources]]:
    policy = provider_policy(policy)
    provider_fields = {
        key.removeprefix("provider."): value
        for key, value in policy.entries.items()
        if key.startswith("provider.")
        and key not in ("provider.context_tokens", "provider.unavailable_wait", "provider.max_attempts")
    }
    # The wrapper owns both busy retries and empty-pool waits under one deadline.
    provider = TypeAdapter(LLMProviderConfig).validate_python({**provider_fields, "max_attempts": 1, "hold_timeout": 0})
    print(f"GLM {type(source).__name__} resolved {source.current().base_url}", flush=True)
    host = MachineHost(policy.setting("runtime.host", str))
    # Source: acceptance-measure taskforge/sandbox/factories.py; driver.py controller-url env.
    container: MachineFactory
    if host == MachineHost.IRIS:
        job = get_job_info()
        controller = os.environ.get("IRIS_CONTROLLER_URL") or (job.controller_address if job else None)
        if not controller:
            raise ValueError("Iris sandboxes require a controller URL or an Iris job context")
        container = IrisMachineFactory(controller_url=controller, scheduling_timeout=SANDBOX_SCHEDULING_TIMEOUT)
    else:
        container = DockerMachineFactory()
    factories: dict[str, MachineFactory] = {container.backend.value: container, "shellsim": shellsim_factory()}
    if host == MachineHost.LAPTOP:
        factories["local"] = local_factory()
    headers = {"x-api-key": parallel_api_key}
    async with (
        EndpointLimitedClient(source, provider) as client,
        httpx.AsyncClient(base_url="https://api.parallel.ai/v1/", headers=headers, timeout=60) as http,
    ):
        waiting = WaitingBackend(
            client,
            policy.setting("provider.unavailable_wait", float),
            provider.backoff_initial,
            provider.backoff_maximum,
            provider.backoff_factor,
        )
        yield build_config(policy, arm, factories), build_resources(policy, waiting, factories, ParallelWeb(http))
