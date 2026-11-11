# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind real adapters and executable policy at the process boundary."""

import logging
import os
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import override

import httpx
from iris.client.client import iris_ctx
from iris.cluster.client import get_job_info
from iris.cluster.types import JobName
from iris.rpc.errors import DEFAULT_RETRY_MAX_ELAPSED
from pydantic import TypeAdapter
from rolloutengine.models import GradedRollout, ModelRequest, ModelTurn
from rolloutengine.validation import InvalidTaskSpec
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.backends.local.machine import LocalMachineFactory, SandboxUnavailable, SystemDirectories
from shellbox.machine import MachineFactory
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import EnvironmentRequirements, TaskResource
from taskforge.llm.client import ModelUnavailable
from taskforge.llm.glm_client import EndpointSource, FixedEndpoint, GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy, LLMProviderConfig
from taskforge.llm.rollout_model import model_turn

from forge.arms import transitions
from forge.config import ForgeConfig, WorkspaceHydration
from forge.effects.context import Context
from forge.effects.events import Note, Resolved
from forge.effects.issues import Hold
from forge.harness.loop import AgenticHarness
from forge.harness.tools import adversary_tools, standard_tools
from forge.harness.tools.validate import ValidateTool
from forge.lowering import (
    LOCAL_SANDBOX,
    LoweringSettings,
    check_verification,
    host_requirements,
    lower_with_settings,
    lowering_settings,
    machine_backend,
    machine_runtime,
    rollout_engine,
)
from forge.model.content_hash import digest
from forge.model.proposal import Verification
from forge.model.review import BandOutcome, BandRules, Budget
from forge.model.workpiece import Submission, Workpiece
from forge.policy import Policy, load_policy, resolution_lines
from forge.policy import sampling as policy_sampling
from forge.resources.bundle import Resources
from forge.resources.clock import Clock, Now, now
from forge.resources.decision_model import CLEF_MODEL, ClefDecisionModel, DecisionModelClient, DecisionModelEngine
from forge.resources.env import Env, EnvFactory, ShellboxEnvFactory
from forge.resources.grading import GradeCandidate, GradingResource, grade
from forge.resources.ids import Ids, NewId, new_id
from forge.resources.inference import CompletionBackend, GlmInference, InferenceRequest, complete
from forge.resources.judge import JudgeEngine
from forge.resources.provider import WaitingBackend
from forge.resources.rollouts import RolloutResource, SolverRollout, run_rollout
from forge.resources.sandbox.factories import (
    SANDBOX_SCHEDULING_TIMEOUT,
    shellsim_factory,
)
from forge.resources.verifier import RoundVerifier
from forge.resources.web import ParallelWeb, WebClient, WebFetch, WebSearch, fetch, search
from forge.stages.rules import default_review


class EndpointUnregistered(ModelUnavailable, RuntimeError):
    """The relay has no endpoint right now, e.g. while a fresh controller waits for relays to re-register.

    As ``ModelUnavailable`` the provider wrapper retries it inside its availability window. As ``RuntimeError``
    the decision-model client keeps treating it as unreachable.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, ())


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
            raise EndpointUnregistered("Iris client is unavailable inside the task")
        endpoints = client.resolver_for_job(JobName.from_string(self.job)).resolve(self.model).endpoints
        if not endpoints:
            raise EndpointUnregistered(f"GLM relay {self.job} has no registered endpoint for {self.model}")
        url = endpoints[0].url.rstrip("/")
        return GlmEndpoint(url if url.endswith("/v1") else f"{url}/v1", self.token, self.pool)


SHELL_ASSIGNMENT = re.compile(r"[A-Z][A-Z0-9_]*=")
"""A token file line such as ``GLM_API_TOKEN=...`` passed whole instead of its value."""


def endpoint_source(base_url: str | None, relay_job: str | None, token: str, model: str, pool: str) -> EndpointSource:
    if SHELL_ASSIGNMENT.match(token):
        raise ValueError("GLM_API_KEY holds a shell assignment (NAME=value); pass only the value")
    if bool(base_url) == bool(relay_job):
        raise ValueError("Set exactly one of GLM_RELAY_JOB and GLM_BASE_URL")
    if relay_job:
        return RelayEndpoint(relay_job, token, Pool(pool), model)
    assert base_url is not None
    return FixedEndpoint(GlmEndpoint(base_url, token, Pool(pool)))


@dataclass(frozen=True, slots=True)
class PooledEndpoint(EndpointSource):
    source: EndpointSource
    pool: Pool

    def current(self) -> GlmEndpoint:
        return replace(self.source.current(), pool=self.pool)


def install(ctx: Context, resources: Resources) -> None:
    async def cleanup_notes(event: Resolved[object]) -> None:
        value = event.value.grade if isinstance(event.value, GradedRollout) else event.value
        if isinstance(value, GradeResult) and value.diagnostics.get("cleanup_errors"):
            await ctx.emit(Note(f"Cleanup failed: {value.diagnostics['cleanup_errors']}"))

    rollouts, grading = resources.rollouts, resources.grading
    if (decision_model := resources.decision_model) is not None:
        rollouts = replace(rollouts, engine=DecisionModelEngine(rollouts.engine, ctx, decision_model))
        grading = replace(
            grading,
            engine=DecisionModelEngine(grading.engine, ctx, decision_model),
            fresh_engine=(
                DecisionModelEngine(grading.fresh_engine, ctx, decision_model) if grading.fresh_engine else None
            ),
        )
    if (judge := resources.role_inference.get("judge")) is not None:
        rollouts = replace(rollouts, engine=JudgeEngine(rollouts.engine, ctx, judge))
        grading = replace(
            grading,
            engine=JudgeEngine(grading.engine, ctx, judge),
            fresh_engine=JudgeEngine(grading.fresh_engine, ctx, judge) if grading.fresh_engine else None,
        )
    ctx.observe(Resolved, cleanup_notes)
    if resources.verifiers is not None:
        resources.verifiers.notify = ctx.emit
    for event, execute, resource in (
        (InferenceRequest, partial(complete, notify=ctx.emit), resources.inference_for("author")),
        (SolverRollout, partial(run_rollout, notify=ctx.emit), rollouts),
        (GradeCandidate, partial(grade, notify=ctx.emit), grading),
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
        requirements = host_requirements(requirements)
        runtime = machine_runtime(self.settings.task_machine, requirements)
        factory = self.factories[machine_backend(requirements).value]
        cleanup = self.settings.limits.cleanup_timeout
        if cleanup is None:
            raise ValueError("Environment cleanup timeout must be explicit")
        return await ShellboxEnvFactory(factory, runtime, cleanup, logging.info).create(requirements)


async def solver_turn(client: CompletionBackend, sampling: LLMPolicy, request: ModelRequest) -> ModelTurn:
    if request.continuation is not None or request.prompt_token_ids is not None:
        raise ValueError("GLM solver requires messages trace")
    return model_turn(await client.complete(request.messages, sampling, request.options))


def build_resources(
    policy: Policy,
    clients: Mapping[str, CompletionBackend],
    factories: Mapping[str, MachineFactory],
    web: WebClient,
    decision_model: DecisionModelClient | None = None,
) -> Resources:
    settings = lowering_settings(policy)
    command_timeout = policy.setting("solver.command_timeout", float)
    base_engine = rollout_engine(factories, command_timeout=command_timeout)
    engine = verifiers = RoundVerifier(base_engine, settings.limits.cleanup_timeout)
    execution = digest((settings, command_timeout))
    provider = {key: value for key, value in policy.entries.items() if key.startswith("provider.")}
    sampling = policy_sampling(policy)
    inference = {
        role: GlmInference(
            client,
            digest(
                (
                    provider,
                    role,
                    policy.setting(f"{'solver' if role == 'judge' else role}.context_tokens", int),
                    policy.entries.get(f"{role}.pool", policy.setting("provider.pool", str)),
                )
            ),
        )
        for role, client in clients.items()
    }
    return Resources(
        Environments(factories, settings),
        inference["author"],
        RolloutResource(
            engine,
            partial(solver_turn, clients["solver"], sampling),
            digest((provider, clients["solver"].model, sampling, execution)),
        ),
        GradingResource(engine, execution, base_engine),
        Clock(),
        Ids(),
        factories,
        web,
        inference,
        verifiers,
        decision_model,
    )


def submit_gate(
    submission: Submission,
    verification: Verification,
    *,
    settings: LoweringSettings,
    factories: Mapping[str, MachineFactory],
) -> tuple[str, ...]:
    if not submission.controls:
        return ("Supply at least one control candidate",)
    try:
        check_verification(submission.task, verification, settings)
        lower_with_settings(submission.task, settings, factories)
    except InvalidTaskSpec as error:
        return (str(error),)
    except (ValueError, NotImplementedError) as error:
        raise Hold(f"Lowering failed: {error}") from error
    return ()


def hydration(_wp: Workpiece, _policy: Policy) -> tuple[TaskResource, ...]:
    return ()


def prompt_limit(policy: Policy, role: str = "author") -> int:
    """Use the served model context without reserving an output allowance."""
    return policy.setting(f"{role}.context_tokens", int)


def build_config(policy: Policy, arm: str, factories: Mapping[str, MachineFactory]) -> ForgeConfig:
    settings = lowering_settings(policy)
    sampling = policy_sampling(policy)
    gate = partial(submit_gate, settings=settings, factories=factories)
    author = AgenticHarness(
        prompt_limit(policy),
        sampling,
        policy.prompt("author.nudge").template,
        policy.setting("tools.output_chars", int),
    )
    adversaries = {
        name: AgenticHarness(
            prompt_limit(policy, "adversary"),
            sampling,
            policy.prompt("adversary.nudge").template,
            policy.setting("tools.output_chars", int),
        )
        for name in policy.setting("adversary.roster", tuple)
    }
    root = PurePosixPath(policy.setting("author.workdir", str))
    task_path, controls_path = str(root / "task.json"), str(root / "controls.json")
    tools = {
        "author": standard_tools(
            task_path=task_path,
            controls_path=controls_path,
            policy=policy,
            factories=factories,
            settings=settings,
            gate=gate,
        )
    }
    if arm == "T":
        tools["author"] = {
            **tools["author"],
            "validate": ValidateTool(gate, task_path, controls_path),
        }
    tools.update(
        {
            name: adversary_tools(task_path=task_path, policy=policy, factories=factories, settings=settings)
            for name in adversaries
        }
    )
    budget = Budget(
        policy.setting("budget.max_modify_rounds", int), policy.setting("budget.max_validation_retries", int)
    )
    band = BandRules(
        policy.setting("band.low", float),
        policy.setting("band.high", float),
        (
            frozenset({BandOutcome.IN_BAND, BandOutcome.TOO_EASY})
            if policy.setting("band.accept_too_easy", bool)
            else frozenset({BandOutcome.IN_BAND})
        ),
    )
    return ForgeConfig(
        policy,
        author,
        adversaries,
        WorkspaceHydration((), hydration),
        tools,
        gate,
        default_review,
        band,
        budget,
        transitions(arm, budget, policy),
    )


def provider_policy(policy: Policy) -> Policy:
    """Reject deadlines that expire before the provider availability window, and decision_model without Clef."""
    if Verification.DECISION_MODEL in policy.setting("proposals.verifications", tuple):
        if policy.setting("decision_model.model", str) != CLEF_MODEL:
            raise ValueError(
                f"proposals.verifications includes decision_model, which needs decision_model.model {CLEF_MODEL!r}"
            )
        if policy.setting("decision_model.unavailable_wait", float) + policy.setting(
            "decision_model.request_timeout", float
        ) >= policy.setting("solver.turn_timeout", float):
            raise ValueError("decision_model.unavailable_wait plus request_timeout must end before solver.turn_timeout")
    window = policy.setting("provider.unavailable_wait", float)
    turn = policy.setting("solver.turn_timeout", float)
    if turn < window + policy.setting("provider.stall_timeout", float):
        raise ValueError("solver.turn_timeout must cover provider.unavailable_wait plus provider.stall_timeout")
    if policy.setting("solver.attempt_timeout", float) < 2 * turn:
        raise ValueError("solver.attempt_timeout must cover two solver.turn_timeout windows")
    return policy


@asynccontextmanager
async def runtime(
    policy: Policy, arm: str, source: EndpointSource, parallel_api_key: str
) -> AsyncIterator[tuple[ForgeConfig, Resources]]:
    policy = provider_policy(policy)
    declared = load_policy(Path(__file__).parent / "policy/defaults")
    logging.info("\n".join(resolution_lines(declared, policy)))
    logging.info(
        "provider.unavailable_wait declared=%s resolved=%s iris_call_cap=%gs",
        declared.setting("provider.unavailable_wait", float),
        policy.setting("provider.unavailable_wait", float),
        DEFAULT_RETRY_MAX_ELAPSED,
    )
    provider_fields = {
        key.removeprefix("provider."): value
        for key, value in policy.entries.items()
        if key.startswith("provider.") and key not in ("provider.unavailable_wait", "provider.pool")
    }
    # The wrapper owns both busy retries and empty-pool waits under one deadline.
    provider_fields.update(max_attempts=1, hold_timeout=0)
    job = get_job_info()
    controller = os.environ.get("IRIS_CONTROLLER_URL") or (job.controller_address if job else None)
    if not controller:
        raise ValueError("Iris sandboxes require a controller URL or an Iris job context")
    container = IrisMachineFactory(controller_url=controller, scheduling_timeout=SANDBOX_SCHEDULING_TIMEOUT)
    factories: dict[str, MachineFactory] = {
        container.backend.value: container,
        "shellsim": shellsim_factory(output_budget_mb=policy.setting("sandbox.output_budget_mb", int)),
    }
    if LOCAL_SANDBOX:  # THROWAWAY: bwrap sandboxes inside the (privileged) driver pod
        try:
            local: MachineFactory = LocalMachineFactory(system_directories=SystemDirectories.COPY_ON_WRITE)
        except SandboxUnavailable as error:
            logging.warning("bwrap copy-on-write unavailable (%s); using read-only system directories", error)
            local = LocalMachineFactory(system_directories=SystemDirectories.READ_ONLY)
        # THROWAWAY: inside the privileged Iris pod the keeper's write to its per-machine cgroup.procs fails with
        # EOPNOTSUPP (canary p-119, 2026-11-09 15:34Z). Run without per-machine cgroups; the driver pod's memory
        # request is the only enforced bound on sandbox memory.
        local.cgroup_parent = None  # type: ignore[attr-defined]
        factories[local.backend.value] = local
        logging.info("FORGE_LOCAL_SANDBOX=1: grader-base machines run on %s", local.backend.value)
    headers = {"x-api-key": parallel_api_key}
    async with AsyncExitStack() as owned:
        clients: dict[str, CompletionBackend] = {}
        roles = (
            ("author", "solver", "adversary", "judge")
            if policy.setting("judge.model", str)
            else ("author", "solver", "adversary")
        )
        for role in roles:
            model = policy.setting(f"{role}.model", str)
            provider = TypeAdapter(LLMProviderConfig).validate_python({**provider_fields, "model": model})
            role_source = replace(source, model=model) if isinstance(source, RelayEndpoint) else source
            if role == "judge":
                role_source = PooledEndpoint(role_source, Pool(policy.setting("judge.pool", str)))
            logging.info("MODEL role=%s model=%s endpoint=%s", role, model, role_source.current().base_url)
            client = await owned.enter_async_context(GlmClient(role_source, provider, limit_output=False))
            clients[role] = WaitingBackend(
                client,
                policy.setting("provider.unavailable_wait", float),
                provider.backoff_initial,
                provider.backoff_maximum,
                provider.backoff_factor,
                role=role,
            )
        decision_model = None
        if (decision_name := policy.setting("decision_model.model", str)) == CLEF_MODEL:
            route = policy.setting("decision_model.route", str)
            model_source = replace(source, model=route) if isinstance(source, RelayEndpoint) else source
            limit = policy.setting("decision_model.max_inflight", int)
            logging.info(
                "MODEL role=decision_model model=%s route declared=%s resolved=%s/systemone max_inflight=%d",
                decision_name,
                route,
                model_source.current().base_url,
                limit,
            )
            decision_model = ClefDecisionModel(
                model_source,
                await owned.enter_async_context(httpx.AsyncClient()),
                request_timeout=policy.setting("decision_model.request_timeout", float),
                unavailable_wait=policy.setting("decision_model.unavailable_wait", float),
                max_inflight=limit,
            )
        elif decision_name:
            raise ValueError(f"decision_model.model must be {CLEF_MODEL!r} or empty, not {decision_name!r}")
        http = await owned.enter_async_context(
            httpx.AsyncClient(base_url="https://api.parallel.ai/v1/", headers=headers, timeout=60)
        )
        yield (
            build_config(policy, arm, factories),
            build_resources(policy, clients, factories, ParallelWeb(http), decision_model),
        )
