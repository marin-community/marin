# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind real adapters and executable policy at the process boundary."""

import logging
import math
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field, fields, replace
from functools import partial
from pathlib import PurePosixPath
from string import Formatter
from types import MappingProxyType
from typing import NamedTuple, override

import httpx
from connectrpc.errors import ConnectError
from iris.client.client import iris_ctx
from iris.cluster.client import get_job_info
from iris.cluster.types import JobName
from iris.rpc.errors import DEFAULT_RETRY_MAX_ELAPSED, is_retryable_error
from rolloutengine.models import GenerationLimitReached, GradedRollout, ModelRequest, ModelTurn
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.machine import MachineFactory
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import EnvironmentRequirements

from taskforge.config import TaskforgeConfig
from taskforge.effects.context import Context
from taskforge.effects.events import Note, Resolved
from taskforge.effects.issues import Hold
from taskforge.harness.loop import AgenticHarness
from taskforge.harness.role import Role
from taskforge.harness.tools import adversary_tools, standard_tools
from taskforge.harness.tools.evidence import MIN_OUTPUT_CHARS
from taskforge.llm.client import ContextExhausted, GenerationTimedOut, ModelClient, Provider
from taskforge.llm.glm_provider import (
    EndpointSource,
    EndpointUnavailable,
    FixedEndpoint,
    GlmEndpoint,
    GlmProvider,
    Pool,
)
from taskforge.llm.policy import GLM_5_3, LLMPolicy, ModelProfile, RetryPolicy
from taskforge.llm.wire import model_turn
from taskforge.lowering import (
    InvalidTaskSpec,
    LoweringSettings,
    host_requirements,
    lower_with_settings,
    lowering_settings,
    machine_backend,
    machine_runtime,
    rollout_engine,
)
from taskforge.model.content_hash import digest
from taskforge.model.review import BandOutcome, BandRules, Budget
from taskforge.model.workpiece import Submission
from taskforge.policy import Policy, load_policy, resolution_lines
from taskforge.policy import sampling as policy_sampling
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.resources.bundle import Resources
from taskforge.resources.decision_model import (
    ClefReader,
    DecisionModel,
    DecisionModelEngine,
    DecisionModelPolicy,
    DecisionModelRead,
    DecisionModelReader,
    decided_rollout,
    read_level,
)
from taskforge.resources.env import Env, EnvFactory, ShellboxEnvFactory
from taskforge.resources.grading import GradeCandidate, GradingResource, grade
from taskforge.resources.ids import Ids, NewId, new_id
from taskforge.resources.inference import CompletionBackend, GlmInference, InferenceRequest, complete
from taskforge.resources.judge import JudgeEngine
from taskforge.resources.rollouts import RolloutResource, SolverGeneration, SolverRollout, checkpoint, run_rollout
from taskforge.resources.sandbox.factories import (
    SANDBOX_SCHEDULING_TIMEOUT,
    shellsim_factory,
)
from taskforge.resources.verifier import RoundVerifier
from taskforge.resources.web import ParallelWeb, WebClient, WebFetch, WebSearch, fetch, search
from taskforge.stages.author import AuthorReference, author_reference
from taskforge.stages.rules import default_review

DEFAULT_PROFILES: Mapping[str, ModelProfile] = MappingProxyType({})
NO_PROVIDERS: Mapping[str, Provider] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class RelayEndpoint(EndpointSource):
    """The endpoint that the Iris relay job ``job`` registers for ``model``, resolved on every call.

    A relay with no endpoint, as while a fresh Iris controller waits for relays to re-register, or a controller that
    stays unavailable through the lookup's retries is ``EndpointUnavailable``: the model client waits it out.
    """

    job: str
    token: str = field(repr=False)
    pool: Pool
    model: str

    @override
    def current(self) -> GlmEndpoint:
        client = iris_ctx().client
        if client is None:
            raise EndpointUnavailable("Iris client is unavailable inside the task")
        try:
            endpoints = client.resolver_for_job(JobName.from_string(self.job)).resolve(self.model).endpoints
        except ConnectError as error:
            if not is_retryable_error(error):
                raise
            raise EndpointUnavailable(f"Iris endpoint lookup failed: {error}") from error
        if not endpoints:
            raise EndpointUnavailable(f"GLM relay {self.job} has no registered endpoint for {self.model}")
        url = endpoints[0].url.rstrip("/")
        return GlmEndpoint(url if url.endswith("/v1") else f"{url}/v1", self.token, self.pool)


def endpoint_source(base_url: str | None, relay_job: str | None, token: str, model: str, pool: str) -> EndpointSource:
    if bool(base_url) == bool(relay_job):
        raise ValueError("Pass exactly one of --glm-base-url and --glm-relay-job")
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
            ctx.notify(Note(f"Cleanup failed: {value.diagnostics['cleanup_errors']}"))

    rollouts, grading = resources.rollouts, resources.grading
    solve = partial(run_rollout, notify=ctx.notify)
    if (judge := resources.role_inference.get("judge")) is not None:
        rollouts = replace(rollouts, engine=JudgeEngine(rollouts.engine, ctx, judge))
        grading = replace(
            grading,
            engine=JudgeEngine(grading.engine, ctx, judge),
            fresh_engine=JudgeEngine(grading.fresh_engine, ctx, judge) if grading.fresh_engine else None,
        )
    if (decision_model := resources.decision_model) is not None:
        rollouts = replace(rollouts, engine=DecisionModelEngine(rollouts.engine, ctx))
        fresh = DecisionModelEngine(grading.fresh_engine, ctx) if grading.fresh_engine else None
        grading = replace(grading, engine=DecisionModelEngine(grading.engine, ctx), fresh_engine=fresh)
        ctx.handle(DecisionModelRead, partial(read_level, ctx=ctx), decision_model)
        solve = partial(decided_rollout, ctx=ctx)
    ctx.observe(Resolved, cleanup_notes)
    ctx.handle(AuthorReference, author_reference)
    if resources.verifiers is not None:
        resources.verifiers.notify = ctx.notify
    for event, execute, resource in (
        (InferenceRequest, partial(complete, notify=ctx.notify), resources.inference_for("author")),
        (SolverRollout, solve, rollouts),
        (SolverGeneration, checkpoint, rollouts),
        (GradeCandidate, partial(grade, notify=ctx.notify), grading),
        (NewId, new_id, resources.ids),
        (WebSearch, partial(search, notify=ctx.notify), resources.web),
        (WebFetch, partial(fetch, notify=ctx.notify), resources.web),
    ):
        ctx.handle(event, execute, resource)


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
        return await ShellboxEnvFactory(factory, runtime, cleanup).create(requirements)


async def solver_turn(client: CompletionBackend, sampling: LLMPolicy, request: ModelRequest) -> ModelTurn:
    """One solver turn. Only positive evidence that the model ran out ends the rollout, a failed trial that is graded
    and counted: a recognised context overflow at the generation limit, or generation sustained past
    ``solver.time_limit`` (``taskforge.llm.client``) at the time limit. Every outage propagates, so no trial is
    recorded."""
    if request.continuation is not None or request.prompt_token_ids is not None:
        raise ValueError("GLM solver requires messages trace")
    try:
        completion = await client.complete(request.messages, sampling, request.options)
    except (ContextExhausted, GenerationTimedOut) as error:
        raise GenerationLimitReached(()) from error
    return model_turn(completion)


def retry_policy(policy: Policy, role: str) -> RetryPolicy:
    """The provider settings of ``policy``; only a solver generation is bounded, by ``solver.time_limit``."""
    turn = policy.setting("solver.time_limit", float)
    return RetryPolicy(
        policy.setting("provider.unavailable_wait", float),
        policy.setting("provider.backoff_initial", float),
        policy.setting("provider.backoff_maximum", float),
        policy.setting("provider.backoff_factor", float),
        generation_timeout=turn if role == "solver" else None,
    )


def decision_model_policy(policy: Policy) -> DecisionModelPolicy | None:
    """The ``decision_model.*`` settings, or None while ``decision_model.model`` is empty."""
    if not (model := policy.setting("decision_model.model", str)):
        return None
    names = [item.name for item in fields(DecisionModelPolicy)[2:]]
    concurrency = policy.setting("decision_model.concurrency", int)
    timings = [policy.setting(f"decision_model.{name}", float) for name in names]
    named = zip(("concurrency", *names), (concurrency, *timings), strict=True)
    if bad := [f"decision_model.{name}" for name, value in named if not 0 < value < math.inf]:
        raise ValueError(f"{', '.join(bad)} must be positive and finite")
    return DecisionModelPolicy(model, concurrency, *timings)


def clef(policy: Policy, source: EndpointSource, http: httpx.AsyncClient) -> ClefReader:
    """The library's default decision model: Clef on ``source`` at ``decision_model.route``, for a run and its driver."""
    route = policy.setting("decision_model.route", str)
    route_source = replace(source, model=route) if isinstance(source, RelayEndpoint) else source
    logging.info("MODEL role=decision_model route=%s endpoint=%r", route, route_source)
    return ClefReader(route_source, http)


def build_resources(
    policy: Policy,
    providers: Mapping[str, Provider],
    factories: Mapping[str, MachineFactory],
    web: WebClient,
    profiles: Mapping[str, ModelProfile] = DEFAULT_PROFILES,
    decision_model: DecisionModelReader | None = None,
) -> Resources:
    """Executors for ``policy``: each role's model through the one model client, over its ``providers`` entry.

    ``profiles`` overrides the model profile per role; a role without one uses ``GLM_5_3``. Each provider must serve
    its role's declared ``<role>.model``, the model that replay keys record. ``decision_model`` reads for a configured
    ``decision_model.model`` and only then.
    """
    decided = decision_model_policy(policy)
    if (decided is None) != (decision_model is None):
        raise ValueError("A decision-model reader is supplied exactly when policy decision_model.model is set")
    for role, provider in providers.items():
        if provider.model != (declared := policy.setting(f"{role}.model", str)):
            raise ValueError(f"{role} provider serves {provider.model!r}, policy declares {role}.model={declared!r}")
    clients = {
        role: ModelClient(provider, profiles.get(role, GLM_5_3), retry_policy(policy, role), role=role)
        for role, provider in providers.items()
    }
    settings = lowering_settings(policy)
    command_timeout = policy.setting("solver.command_timeout", float)
    base_engine = rollout_engine(factories, command_timeout=command_timeout)
    verifiers = RoundVerifier(base_engine, settings.limits.cleanup_timeout)
    # Solver trials and candidate grades run on the same machines, whose ShellSim output cap can fail either.
    execution = digest((settings, command_timeout, policy.setting("sandbox.output_budget_mb", int)))
    solver = clients["solver"]
    sampling = policy_sampling(policy, solver.profile.sampling)
    # Replay keys carry the model, role, request and non-default profile identity. Rollout and grade keys also carry
    # non-default profiles of roles whose inference a replayed outer result skips.
    inference = {role: GlmInference(client, role) for role, client in clients.items()}
    profile = solver.profile.replay_identity()
    # The spec no longer carries the generation bound (the client owns it), so the solver's key does: a changed limit
    # buys its trials and generations again, and no other role's identity moves.
    limit = ("solver.time_limit", solver.retry.generation_timeout)
    judge = clients["judge"].profile.replay_identity() if "judge" in clients else None
    graders = () if judge is None else (("judge", judge),)
    graders += () if decided is None else (("decision_model", decided.model),)
    return Resources(
        Environments(factories, settings),
        inference["author"],
        RolloutResource(
            verifiers.rollouts,
            partial(solver_turn, solver, sampling),
            digest((solver.model, sampling, execution, limit) + (() if profile is None else (profile,)) + graders),
        ),
        GradingResource(verifiers.grades, digest((execution, *graders)) if graders else execution, base_engine),
        Ids(),
        factories,
        web,
        inference,
        verifiers,
        None if decision_model is None or decided is None else DecisionModel(decision_model, decided),
    )


def submit_gate(
    submission: Submission, *, settings: LoweringSettings, factories: Mapping[str, MachineFactory]
) -> tuple[str, ...]:
    if not submission.controls:
        return ("Supply at least one control candidate",)
    try:
        lower_with_settings(submission.task, settings, factories)
    except InvalidTaskSpec as error:
        return (str(error),)
    except (ValueError, NotImplementedError) as error:
        raise Hold(f"Lowering failed: {error}") from error
    return ()


def agentic_harness(policy: Policy, role: str, profiles: Mapping[str, ModelProfile]) -> AgenticHarness:
    """The ``role`` session under its profile (``GLM_5_3`` by default): the prompt limit is ``<role>.context_tokens``."""
    profile = profiles.get(role, GLM_5_3)
    limit = policy.setting(f"{role}.context_tokens", int)
    if limit > profile.context_tokens:
        raise ValueError(f"{role}.context_tokens {limit} exceeds the model window {profile.context_tokens}")
    return AgenticHarness(
        limit,
        policy_sampling(policy, profile.sampling),
        policy.prompt(f"{role}.nudge").template,
        policy.setting("tools.output_chars", int),
        profile.replay_arguments,
    )


def build_config(
    policy: Policy, factories: Mapping[str, MachineFactory], profiles: Mapping[str, ModelProfile] = DEFAULT_PROFILES
) -> TaskforgeConfig:
    """The run's sessions and rules; ``profiles`` overrides the model profile per role, as in ``build_resources``."""
    # A user prompt that renders only {name} uses the roster name as its strategy; one that renders {brief} needs it.
    template = policy.prompt("adversary.user").template  # {brief[0]} and {brief.x} need a brief as {brief} does
    fields = {field.partition(".")[0].partition("[")[0] for _, field, _, _ in Formatter().parse(template) if field}
    roster = policy.setting("adversary.roster", tuple) if "brief" in fields else ()
    unbriefed = [name for name in roster if f"adversary.brief.{name}" not in policy.entries]
    if unbriefed:
        raise ValueError(
            f"adversary.user renders {{brief}}, but adversary.roster names {', '.join(unbriefed)} without a prompt "
            "adversary.brief.<name>"
        )
    if (revisions := policy.setting("adversary.max_revisions", int)) < 0:
        raise ValueError(f"adversary.max_revisions must be at least 0, not {revisions}")
    if (chars := policy.setting("tools.output_chars", int)) < MIN_OUTPUT_CHARS:
        raise ValueError(
            f"tools.output_chars {chars} leaves no page of evidence for solve_review: at least {MIN_OUTPUT_CHARS}"
        )
    settings = lowering_settings(policy)
    root = PurePosixPath(policy.setting("author.workdir", str))
    task_path, controls_path = str(root / "task.json"), str(root / "controls.json")
    author = Role(
        "author",
        agentic_harness(policy, "author", profiles),
        standard_tools(
            task_path=task_path,
            controls_path=controls_path,
            policy=policy,
            factories=factories,
            settings=settings,
            gate=partial(submit_gate, settings=settings, factories=factories),
        ),
    )
    adversary = Role(
        "adversary",
        agentic_harness(policy, "adversary", profiles),
        adversary_tools(task_path=task_path, policy=policy, factories=factories, settings=settings),
    )
    final = ("solve_review.final.system", "solve_review.final.user", "solve_review.final.nudge", "author.notes")
    for prompt in ("solve_review.system", "solve_review.user", *final):
        policy.prompt(prompt)  # a missing prompt fails here, not at a round's first review
    solve_review = Role("solve_review", agentic_harness(policy, "solve_review", profiles), {})
    budget = Budget(
        policy.setting("budget.max_modify_rounds", int),
        policy.setting("budget.max_validation_retries", int),
        policy.setting("budget.max_missing_trials", int),
        policy.setting("budget.final_revisions", int),
    )
    slots, beat = policy.setting("solver.concurrency", int), policy.setting("solver.heartbeat_interval", float)
    confirmation = policy.setting("solver.confirmation_k", int)
    if slots < 1 or not 0 < beat < math.inf or min(budget.max_missing_trials, budget.final_revisions, confirmation) < 0:
        raise ValueError(
            "solver.concurrency, solver.heartbeat_interval, budget.max_missing_trials, budget.final_revisions or "
            "solver.confirmation_k is out of range"
        )
    band = BandRules(
        policy.setting("band.low", float),
        policy.setting("band.high", float),
        frozenset({BandOutcome.IN_BAND, BandOutcome.TOO_HARD, BandOutcome.TOO_EASY})
        - (set() if policy.setting("band.accept_too_easy", bool) else {BandOutcome.TOO_EASY}),
    )
    return TaskforgeConfig(policy, author, adversary, solve_review, default_review, band, budget)


def provider_policy(policy: Policy) -> Policy:
    """Reject a solver time limit that bounds no generation, and an attempt cap that ends no later than one could."""
    limit = policy.setting("solver.time_limit", float)
    if not 0 < limit < math.inf:
        raise ValueError("solver.time_limit must be positive and finite")
    if not limit < policy.setting("solver.attempt_timeout", float) < math.inf:
        # At equal values the attempt cap fires first, so a time-limited trial would be recorded as an unscored cap.
        raise ValueError("solver.attempt_timeout must be finite and greater than solver.time_limit")
    return policy


class CampaignWiring(NamedTuple):
    """The stock campaign the driver can resume under its proposal, policy and journal format."""

    config: TaskforgeConfig
    resources: Resources


@asynccontextmanager
async def runtime(
    policy: Policy,
    source: EndpointSource,
    parallel_api_key: str,
    *,
    providers: Mapping[str, Provider] = NO_PROVIDERS,
    profiles: Mapping[str, ModelProfile] = DEFAULT_PROFILES,
    decision_model: DecisionModelReader | None = None,
    controller_url: str | None = None,
) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
    """The process-boundary bindings for ``policy``.

    Each role reaches ``<role>.model`` through a ``GlmProvider`` on ``source`` unless ``providers`` names one for it,
    and uses ``GLM_5_3`` unless ``profiles`` does. A configured ``decision_model.model`` reads through
    ``decision_model``, by default Clef on ``source`` at ``decision_model.route``. Iris sandboxes reach
    ``controller_url``, by default the controller of the Iris job this process runs in. The caller owns what it passes.
    """
    stock = not providers and not profiles and decision_model is None
    policy = provider_policy(policy)
    declared = load_policy(DEFAULTS_DIR)
    logging.info("\n".join(resolution_lines(declared, policy)))
    logging.info(
        "provider.unavailable_wait declared=%s resolved=%s iris_call_cap=%gs",
        declared.setting("provider.unavailable_wait", float),
        policy.setting("provider.unavailable_wait", float),
        DEFAULT_RETRY_MAX_ELAPSED,
    )
    job = get_job_info()
    controller = controller_url or (job.controller_address if job else None)
    if not controller:
        raise ValueError("Iris sandboxes require a controller URL or an Iris job context")
    container = IrisMachineFactory(controller_url=controller, scheduling_timeout=SANDBOX_SCHEDULING_TIMEOUT)
    factories: dict[str, MachineFactory] = {
        container.backend.value: container,
        "shellsim": shellsim_factory(output_budget_mb=policy.setting("sandbox.output_budget_mb", int)),
    }
    headers = {"x-api-key": parallel_api_key}
    async with AsyncExitStack() as owned:
        resolved = dict(providers)
        roles = ("author", "solver", "adversary", "solve_review") + (
            ("judge",) if policy.setting("judge.model", str) else ()
        )
        stall = policy.setting("provider.stall_timeout", float)
        connections = policy.setting("provider.max_connections", int)
        for role in (role for role in roles if role not in providers):
            model = policy.setting(f"{role}.model", str)
            role_source = replace(source, model=model) if isinstance(source, RelayEndpoint) else source
            if role == "judge":
                role_source = PooledEndpoint(role_source, Pool(policy.setting("judge.pool", str)))
            # The endpoint resolves at the first request, inside the model client's availability window.
            logging.info("MODEL role=%s model=%s endpoint=%r", role, model, role_source)
            resolved[role] = await owned.enter_async_context(GlmProvider(role_source, model, stall, connections))
        http = await owned.enter_async_context(
            httpx.AsyncClient(base_url="https://api.parallel.ai/v1/", headers=headers, timeout=60)
        )
        web = ParallelWeb(http)
        if decision_model is None and decision_model_policy(policy) is not None:
            decision_model = clef(policy, source, await owned.enter_async_context(httpx.AsyncClient()))
        resources = build_resources(policy, resolved, factories, web, profiles, decision_model)
        config = build_config(policy, factories, profiles)
        yield CampaignWiring(config, resources) if stock else (config, resources)
