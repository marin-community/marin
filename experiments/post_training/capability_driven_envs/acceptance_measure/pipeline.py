# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The live stack for one allocation: model clients, build services and a ``StageCoordinator`` with the arm applied.

Every build (the author's ``run_build`` tool and the build stage) runs in a child process through
``taskforge_adapters.build_process.run_build_in_process``, under the arm's ``build.wall_timeout``, so a step that
never yields cannot stall the allocation. The child receives no secrets, only the names of the variables that hold
them.

Validation is ``validate_round`` followed by ``shadows.shadow_round``. A shadow that raises is written as an error
record and never changes the round: shadows are measurement only.
"""

import json
import logging
import traceback
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import NoReturn

import httpx
from shellbox.machine import Backend
from taskforge.ledger.jsonl import JsonlLedger
from taskforge.llm.agent import ReasoningReplay
from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMProviderConfig
from taskforge.llm.recording import CallLedger
from taskforge.llm.rollout_model import GlmRolloutModel
from taskforge.sandbox.factories import MachineHost
from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage20_triage.preflight import preflight
from taskforge.stage20_triage.verdict import Verdict
from taskforge.stage30_author.author import PROGRAM_FILE, AuthorContext
from taskforge.stage40_build_with_program.agentic_author import SessionBuild, load_in_child
from taskforge.stage40_build_with_program.program_sdk import load_program
from taskforge.stage40_build_with_program.run import ResearchRequirement, TaskDraft, build_model_digest
from taskforge.stage40_build_with_program.services import RLEnvBuildServices
from taskforge.stage50_validate.round import ValidationContext, ValidationResult, ValidationSettings, validate_round
from taskforge.stage50_validate.trials import EngineSettings
from taskforge.stage_coordinator.coordinator import AUTHORED_DIR, STORE_DIR, StageCoordinator
from taskforge.stage_coordinator.event_log import JsonEventStore
from taskforge.stage_coordinator.stage_functions import StageFunctions
from taskforge_adapters.backends import machine_factories
from taskforge_adapters.build_process import (
    BuildModelConfig,
    BuildRequest,
    BuildServiceConfig,
    image_builder,
    run_build_in_process,
)
from taskforge_adapters.glm_client import EndpointSource, FixedEndpoint, GlmClient, GlmEndpoint, Pool, RelayEndpoint
from taskforge_adapters.web import web_tools

from experiments.post_training.capability_driven_envs.acceptance_measure.author import TreatedAgenticAuthor
from experiments.post_training.capability_driven_envs.acceptance_measure.review import SHADOW_FILE, treated_review
from experiments.post_training.capability_driven_envs.acceptance_measure.runtime import (
    BUILD_AGENT_LIMITS,
    SAMPLING,
    TRIAL_LIMITS,
    ArmRuntime,
)
from experiments.post_training.capability_driven_envs.acceptance_measure.shadows import Judge, shadow_round
from experiments.post_training.capability_driven_envs.generate_tasks import GLM_PROVIDER, POOL_TOKEN_ENV
from experiments.post_training.capability_driven_envs.generate_tasks_agentic import DEEPSEEK_MODEL

logger = logging.getLogger(__name__)

LEDGER_DIR = "ledger"
DEEPSEEK_CONTEXT = 614400
POOL = Pool.BULK


@dataclass(frozen=True)
class Host:
    """Where the allocation runs and how it reaches models, machines and images; secrets by variable name only."""

    host: MachineHost
    backends: tuple[Backend, ...]
    controller_url: str | None
    relay_job: str | None
    base_url: str | None
    parallel_key_env: str | None
    image_cluster: str | None
    registry_credential_env: str | None

    def __post_init__(self) -> None:
        if (self.relay_job is None) == (self.base_url is None):
            raise ValueError("give exactly one of a relay job and a base URL")

    @property
    def token_env(self) -> str:
        return POOL_TOKEN_ENV[POOL]


def provider(model: str) -> LLMProviderConfig:
    window = DEEPSEEK_CONTEXT if model == DEEPSEEK_MODEL else GLM_PROVIDER.context_window
    return replace(GLM_PROVIDER, model=model, context_window=window)


def endpoint(host: Host, model: str, token: str) -> EndpointSource:
    if host.relay_job is not None:
        return RelayEndpoint(host.relay_job, token, POOL, model)
    assert host.base_url is not None
    return FixedEndpoint(GlmEndpoint(host.base_url.rstrip("/"), token, POOL))


def build_model_config(host: Host, model: str) -> BuildModelConfig:
    return BuildModelConfig(
        base_url=host.base_url or "",
        pool=POOL,
        token_env=host.token_env,
        provider=provider(model),
        relay_job=host.relay_job,
    )


def build_service_config(host: Host, ledger_root: Path) -> BuildServiceConfig:
    return BuildServiceConfig(
        host=host.host,
        backends=host.backends,
        controller_url=host.controller_url,
        parallel_key_env=host.parallel_key_env,
        ledger_root=ledger_root,
        reasoning_replay=ReasoningReplay.KEEP,
        agent_limits=BUILD_AGENT_LIMITS,
        image_cluster=host.image_cluster,
        registry_credential_env=host.registry_credential_env,
    )


def latest_authored(round_dir: Path) -> str:
    """The program the author stage just wrote: ``authored/<k>/program.py`` with the largest ``k``."""
    numbered = sorted(int(p.name) for p in (round_dir / AUTHORED_DIR).iterdir() if p.name.isdigit())
    if not numbered:
        raise FileNotFoundError(f"{round_dir} has no authored program")
    return (round_dir / AUTHORED_DIR / str(numbered[-1]) / PROGRAM_FILE).read_text()


def no_repair(proposal: TaskProposal, verdict: Verdict) -> NoReturn:
    raise AssertionError("budget.max_triage_repairs is 0 in every arm, so triage never asks for a repair")


@asynccontextmanager
async def clients(host: Host, models: Mapping[str, str], token: str) -> AsyncIterator[dict[str, ModelClient]]:
    """One client per distinct model, keyed by role."""
    async with AsyncExitStack() as stack:
        by_model = {
            m: await stack.enter_async_context(GlmClient(endpoint(host, m, token), provider(m)))
            for m in set(models.values())
        }
        yield {role: by_model[model] for role, model in models.items()}


def coordinator_for(
    runtime: ArmRuntime,
    root: Path,
    host: Host,
    models: Mapping[str, ModelClient],
    services: RLEnvBuildServices,
    research: ResearchRequirement,
) -> StageCoordinator:
    """The coordinator of one allocation rooted at ``root``, with every stage the arm changes substituted."""
    ledger_root = root / LEDGER_DIR
    store = str(root / STORE_DIR)
    build_client = models["build"]
    digest = build_model_digest(build_client, SAMPLING, services)
    model_config = build_model_config(host, runtime.models["build"])
    service_config = build_service_config(host, ledger_root)

    def request(
        workpiece_id: str, round_no: int, source: str, proposal: TaskProposal, round_dir: Path, invalidate
    ) -> BuildRequest:
        return BuildRequest(
            workpiece_id, round_no, source, proposal, round_dir, store, model_config, SAMPLING, research,
            frozenset(invalidate), digest, service_config,
        )  # fmt: skip

    def builds(context: AuthorContext, proposal: TaskProposal) -> SessionBuild:
        async def builder(source: str, run_dir: Path, *, invalidate: frozenset[str]) -> TaskDraft:
            built = request(context.workpiece_id, context.round, source, proposal, run_dir, invalidate)
            return await run_build_in_process(built, wall_timeout=runtime.build_wall_timeout)

        return SessionBuild(
            builder, store, services, JsonlLedger(ledger_root).path_for(context.workpiece_id), load_in_child
        )

    async def build(
        program,
        proposal,
        workpiece_id,
        round_dir,
        store_root,
        model,
        sampling,
        services_,
        round_no,
        *,
        research,
        model_digest,
    ):
        built = request(workpiece_id, round_no, latest_authored(round_dir), proposal, round_dir, ())
        return await run_build_in_process(built, wall_timeout=runtime.build_wall_timeout)

    settings = ValidationSettings(
        host=host.host,
        engine=EngineSettings(services.factories, runtime.command_timeout, TRIAL_LIMITS),
        k=runtime.solver_k,
        max_retries=runtime.max_retries,
        retry_backoff=runtime.retry_backoff,
        sampling=SAMPLING,
        adversary=runtime.adversaries,
        ledger=services.ledger,
    )
    judge = (
        None
        if runtime.judge_prompt is None
        else Judge(models["shadow_judge"], SAMPLING, runtime.judge_prompt, runtime.judge_max_claims)
    )

    async def validate(draft: TaskDraft, context: ValidationContext) -> ValidationResult:
        solver = GlmRolloutModel(
            models["solver"], SAMPLING, CallLedger(services.ledger, context.workpiece_id, context.round, "solver")
        )
        result = await validate_round(draft, context, settings, solver, models["controls"], models["adversary"])
        await _shadow(draft, context, result, runtime, judge)
        return result

    author = TreatedAgenticAuthor(
        runtime.author_config,
        models["author"],
        SAMPLING,
        services.ledger,
        builds,
        runtime.policy,
        treatment=runtime.author,
    )
    stages = StageFunctions(
        triage=preflight,
        repair=no_repair,
        author=author,
        load_program=load_program,
        build=build,
        validate=validate,
        review=partial(treated_review, treatment=runtime.review, root=root),
    )
    return StageCoordinator(
        stages, runtime.policy, root, JsonEventStore(root), build_client, SAMPLING, services, research, digest
    )


async def _shadow(
    draft: TaskDraft, context: ValidationContext, result: ValidationResult, runtime: ArmRuntime, judge: Judge | None
) -> None:
    if not runtime.shadow_l4:
        return
    try:
        await shadow_round(draft, context, result, runtime.review.taxonomy, judge)
    except Exception:
        # Measurement only: a failing shadow is recorded and must not hold or change the round.
        logger.warning("shadow checks failed for %s round %d", context.workpiece_id, context.round, exc_info=True)
        (context.round_dir / SHADOW_FILE).write_text(json.dumps({"error": traceback.format_exc(), "claims": []}) + "\n")


@asynccontextmanager
async def live_services(host: Host, ledger_root: Path, secrets: Mapping[str, str]) -> AsyncIterator[RLEnvBuildServices]:
    """The parent's ``RLEnvBuildServices``, built as ``build_process`` builds the child's."""
    config = build_service_config(host, ledger_root)
    async with httpx.AsyncClient() as http:
        yield RLEnvBuildServices(
            host=host.host,
            factories=machine_factories(host.backends, host.controller_url),
            images=image_builder(config),
            web_tools=() if host.parallel_key_env is None else web_tools(http, secrets[host.parallel_key_env]),
            ledger=JsonlLedger(ledger_root),
            reasoning_replay=ReasoningReplay.KEEP,
            agent_limits=BUILD_AGENT_LIMITS,
        )
