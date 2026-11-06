# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind real adapters and executable policy at the process boundary."""

import logging
import os
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import override

import httpx
from pydantic import TypeAdapter
from rolloutengine.models import ModelRequest, ModelTurn
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.gvisor.machine import GvisorMachineFactory
from shellbox.machine import MachineFactory
from taskcompendium.models import EnvironmentRequirements, TaskResource
from taskforge.llm.glm_client import FixedEndpoint, GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy, LLMProviderConfig
from taskforge.llm.rollout_model import model_turn

from forge.arms import transitions
from forge.config import ForgeConfig, WorkspaceHydration
from forge.effects.context import Context
from forge.harness.loop import AgenticHarness, Limits
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
from forge.resources.inference import GlmInference, InferenceRequest, complete
from forge.resources.rollouts import RolloutResource, SolverRollout, run_rollout
from forge.resources.sandbox.factories import MachineHost, local_factory, shellsim_factory
from forge.stages.rules import default_review


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


@dataclass(frozen=True, slots=True)
class SolverModel:
    client: GlmClient
    sampling: LLMPolicy

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        if request.continuation is not None or request.prompt_token_ids is not None:
            raise ValueError("GLM solver requires messages trace")
        return model_turn(await self.client.complete(request.messages, self.sampling, request.options))


def build_resources(
    policy: Policy, client: GlmClient, factories: Mapping[str, MachineFactory], web: WebClient
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
        RolloutResource(engine, SolverModel(client, sampling), digest((provider, sampling, execution))),
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


def build_config(policy: Policy, arm: str, factories: Mapping[str, MachineFactory]) -> ForgeConfig:
    settings = lowering_settings(policy)
    sampling = TypeAdapter(Sampling).validate_python(
        {key.removeprefix("sampling."): value for key, value in policy.entries.items() if key.startswith("sampling.")}
    )
    author = AgenticHarness(
        Limits(policy.setting("author.max_turns", int), policy.setting("author.max_prompt_tokens", int)),
        sampling,
        policy.prompt("author.nudge").template,
    )
    adversaries = {
        name: AgenticHarness(Limits(policy.setting("adversary.max_turns", int), None), sampling, "Finish your audit.")
        for name in policy.setting("adversary.roster", tuple)
    }
    tools = {"author": standard_tools(policy=policy, factories=factories, settings=settings, gate=submit_gate)}
    if arm == "T":
        root = policy.setting("author.workdir", str)
        tools["author"] = {
            **tools["author"],
            "validate": ValidateTool(submit_gate, f"{root}/task.json", f"{root}/controls.json"),
        }
    tools.update(
        {
            name: adversary_tools(
                policy=policy, task_path="/workspace/task.json", factories=factories, settings=settings
            )
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


@asynccontextmanager
async def runtime(policy: Policy, arm: str) -> AsyncIterator[tuple[ForgeConfig, Resources]]:
    provider = TypeAdapter(LLMProviderConfig).validate_python(
        {key.removeprefix("provider."): value for key, value in policy.entries.items() if key.startswith("provider.")}
    )
    source = FixedEndpoint(GlmEndpoint(os.environ["GLM_BASE_URL"], os.environ["GLM_API_KEY"], Pool.BULK))
    host = MachineHost(policy.setting("runtime.host", str))
    container = GvisorMachineFactory() if host == MachineHost.IRIS else DockerMachineFactory()
    factories: dict[str, MachineFactory] = {container.backend.value: container, "shellsim": shellsim_factory()}
    if sys.platform == "linux":
        factories["local"] = local_factory()
    async with (
        GlmClient(source, provider) as client,
        httpx.AsyncClient(
            base_url="https://api.parallel.ai/v1/",
            headers={"x-api-key": os.environ.get("PARALLEL_API_KEY", "")},
            timeout=60,
        ) as http,
    ):
        yield build_config(policy, arm, factories), build_resources(policy, client, factories, ParallelWeb(http))
