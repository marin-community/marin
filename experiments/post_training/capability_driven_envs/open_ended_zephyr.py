# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-map Zephyr port for complete open-ended Taskforge bundle lifecycles."""

import asyncio
import json
import os
import tempfile
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from fray.types import ResourceConfig
from iris.client.client import iris_ctx
from iris.cluster.types import JobName
from taskforge.llm.agent import ReasoningReplay
from taskforge.llm.client import ModelClient
from taskforge.llm.glm_client import API_ROOT, EndpointSource, FixedEndpoint, GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy, LLMProviderConfig, ReasoningEffort
from taskforge.sandbox.factories import MachineHost, machine_factories
from taskforge.stage10_proposal.model import Environment, SolverMachine
from taskforge.stage40_build_with_program.run import DRAFT_DIR, load_draft, step_output
from taskforge.stage40_build_with_program.sdk import solver_machine
from taskforge.stage40_build_with_program.services import AgentLimits, RLEnvBuildServices
from taskforge.stage_coordinator.coordinator import DECISION_FILE, STORE_DIR, VALIDATION_FILE
from taskforge.stage_coordinator.event_log import EVENTS_FILE, WORKPIECES_DIR
from verifyit.modes.grade_judge import JudgeConnection
from verifyit.spec import JudgeSpec
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.capability_driven_envs.open_ended import OpenEndedTaskBundle
from experiments.post_training.capability_driven_envs.open_ended_coordinator import (
    open_ended_coordinator,
    verifyit_control_scorer,
)
from experiments.post_training.capability_driven_envs.open_ended_proposal_source import (
    OpenEndedProposalConfig,
    OpenEndedProposalSource,
)
from experiments.post_training.capability_driven_envs.open_ended_runner import OpenEndedRuntime, run_open_ended_bundle
from experiments.post_training.capability_driven_envs.open_ended_task_design import OpenEndedTaskDesign

IRIS_CONTROLLER_URL_ENV = "IRIS_CONTROLLER_URL"


@dataclass(frozen=True)
class OpenAIModelPort:
    """Non-secret endpoint configuration serialized to Zephyr workers."""

    model: str
    api_key_env: str
    base_url: str | None = None
    relay_job: str | None = None
    relay_service: str | None = None
    pool: Pool = Pool.BULK

    def __post_init__(self) -> None:
        if (self.base_url is None) == (self.relay_job is None):
            raise ValueError("exactly one of base_url and relay_job is required")
        if self.base_url is not None and not self.base_url.endswith(API_ROOT):
            raise ValueError(f"base_url must end in {API_ROOT}")
        if self.relay_job is not None and not self.relay_service:
            raise ValueError("relay_service is required with relay_job")
        if not self.model or not self.api_key_env:
            raise ValueError("model and api_key_env must be nonempty")


@dataclass(frozen=True)
class IrisRelayEndpoint:
    """Resolve a configured Iris relay from the worker on each endpoint refresh."""

    relay_job: str
    service: str
    token: str = field(repr=False)
    pool: Pool = Pool.BULK

    def current(self) -> GlmEndpoint:
        client = iris_ctx().client
        if client is None:
            raise RuntimeError("Iris client is unavailable inside the Zephyr worker")
        endpoints = client.resolver_for_job(JobName.from_string(self.relay_job)).resolve(self.service).endpoints
        if not endpoints:
            raise RuntimeError(f"relay {self.relay_job!r} has no registered service {self.service!r}")
        base_url = endpoints[0].url.rstrip("/")
        return GlmEndpoint(base_url if base_url.endswith(API_ROOT) else f"{base_url}{API_ROOT}", self.token, self.pool)


@dataclass(frozen=True)
class OpenEndedZephyrConfig:
    """Execution policy for one open-ended lifecycle map."""

    port: OpenAIModelPort
    proposals_per_bundle: int = 1
    control_concurrency: int = 2


def _proposal_config(bundle: OpenEndedTaskBundle) -> OpenEndedProposalConfig:
    machine = solver_machine(bundle.environment_requirements)
    if machine is SolverMachine.PACKAGES:
        return OpenEndedProposalConfig(Environment.REASONING, machine)
    if machine is SolverMachine.SHELLSIM:
        return OpenEndedProposalConfig(Environment.SHELLSIM, machine)
    if machine is SolverMachine.CPYTHON:
        return OpenEndedProposalConfig(Environment.REASONING, machine)
    raise ValueError(f"open-ended Zephyr does not support solver machine {machine}")


def _provider(port: OpenAIModelPort) -> LLMProviderConfig:
    return LLMProviderConfig(
        model=port.model,
        max_attempts=4,
        hold_timeout=900,
        backoff_initial=1,
        backoff_maximum=30,
        backoff_factor=2,
        stall_timeout=300,
        max_connections=16,
        structured_repairs=1,
    )


def _policy() -> LLMPolicy:
    return LLMPolicy(
        max_tokens=4096,
        reasoning_effort=ReasoningEffort.LOW,
        temperature=0.7,
        top_p=0.95,
        max_continuations=0,
    )


def _services(controller_url: str) -> RLEnvBuildServices:
    factories = machine_factories(MachineHost.IRIS, controller_url)
    return RLEnvBuildServices(
        host=MachineHost.IRIS,
        factories=factories,
        images=None,
        web_tools=(),
        ledger=None,
        reasoning_replay=ReasoningReplay.KEEP,
        agent_limits=AgentLimits(max_turns=4, max_tokens=4096, max_calls=8),
        decision=None,
    )


def _workpiece_artifacts(root: Path, workpiece_id: str) -> dict[str, object]:
    workpiece = root / WORKPIECES_DIR / workpiece_id
    artifacts: dict[str, object] = {}
    events = workpiece / EVENTS_FILE
    stored_events = json.loads(events.read_bytes()) if events.is_file() else None
    if stored_events is not None:
        artifacts["events"] = stored_events
        failure_paths = [
            workpiece / event["attrs"]["failure_file"]
            for event in stored_events["events"]
            if "failure_file" in event["attrs"]
        ]
        artifacts["failures"] = [
            {"file": str(path.relative_to(workpiece)), "text": path.read_text()}
            for path in failure_paths
            if path.is_file()
        ]
    rounds = sorted((workpiece / "rounds").glob("*"))
    if rounds:
        latest = rounds[-1]
        for key, name in (("validation", VALIDATION_FILE), ("decision", DECISION_FILE)):
            path = latest / name
            if path.is_file():
                artifacts[key] = json.loads(path.read_bytes())
    return artifacts


def _draft_artifacts(root: Path, workpiece_id: str) -> dict[str, object]:
    rounds = sorted((root / WORKPIECES_DIR / workpiece_id / "rounds").glob("*"))
    if not rounds:
        return {}
    draft = rounds[-1] / DRAFT_DIR
    if not draft.is_dir():
        return {}
    loaded = load_draft(draft)
    design = step_output(loaded, str(root / STORE_DIR), "design", OpenEndedTaskDesign)
    return {"task_json": loaded.task.model_dump_json(), "design": design.model_dump(mode="json")}


async def run_bundle_lifecycle(
    bundle: OpenEndedTaskBundle,
    *,
    model: ModelClient,
    policy: LLMPolicy,
    services: RLEnvBuildServices,
    control_scorer: Callable[[JudgeSpec, str], Awaitable[float]],
    root: Path,
    proposals_per_bundle: int = 1,
    control_concurrency: int = 2,
) -> dict[str, object]:
    """Run proposal through calibrated review and return portable accepted-task rows."""
    source = OpenEndedProposalSource(model, policy, _proposal_config(bundle))
    runtime = OpenEndedRuntime(
        proposal_source=source,
        coordinator=lambda hydrated: open_ended_coordinator(
            root=root / "coordinator",
            bundle=hydrated,
            model=model,
            sampling=policy,
            services=services,
            judge_model=model.model,
            control_scorer=control_scorer,
            control_concurrency=control_concurrency,
        ),
    )
    result = await run_open_ended_bundle(
        bundle,
        runtime,
        slot_count=proposals_per_bundle,
        root=root,
    )
    slots: list[dict[str, object]] = []
    for slot in result.slots:
        item = slot.model_dump(mode="json")
        if slot.workpiece_id is not None:
            item.update(_workpiece_artifacts(root / "coordinator", slot.workpiece_id))
            item.update(_draft_artifacts(root / "coordinator", slot.workpiece_id))
        slots.append(item)
    return {
        "bundle_id": result.bundle_id,
        "bundle_digest": result.bundle_digest,
        "proposal_key": result.proposal_cache_key,
        "release_status": str(bundle.release_status),
        "source": bundle.source.model_dump(mode="json"),
        "model": model.model,
        "slots": slots,
    }


async def _run_bundle(row: dict[str, object], config: OpenEndedZephyrConfig) -> dict[str, object]:
    bundle = OpenEndedTaskBundle.model_validate_json(str(row["bundle_json"]))
    token = os.environ.get(config.port.api_key_env)
    if not token:
        raise RuntimeError(f"worker lacks required API credential environment variable {config.port.api_key_env}")
    controller_url = os.environ.get(IRIS_CONTROLLER_URL_ENV)
    if not controller_url:
        raise RuntimeError(f"worker lacks {IRIS_CONTROLLER_URL_ENV}; this port runs inside an Iris task")
    source: EndpointSource
    if config.port.base_url is not None:
        source = FixedEndpoint(GlmEndpoint(config.port.base_url, token, config.port.pool))
    else:
        assert config.port.relay_job is not None and config.port.relay_service is not None
        source = IrisRelayEndpoint(config.port.relay_job, config.port.relay_service, token, config.port.pool)
    connection = JudgeConnection(source.current().base_url, token)
    policy = _policy()
    with tempfile.TemporaryDirectory(prefix=f"open-ended-{bundle.bundle_id}-") as temporary:
        root = Path(temporary)
        async with GlmClient(source, _provider(config.port)) as model:
            return await run_bundle_lifecycle(
                bundle,
                model=model,
                policy=policy,
                services=_services(controller_url),
                control_scorer=verifyit_control_scorer(connection),
                root=root,
                proposals_per_bundle=config.proposals_per_bundle,
                control_concurrency=config.control_concurrency,
            )


def run_bundle_shard(
    rows: Iterable[dict[str, object]],
    shard_info: object,
    *,
    config: OpenEndedZephyrConfig,
) -> Iterator[dict[str, object]]:
    """Run each bundle's complete lifecycle sequentially inside one Zephyr shard."""
    del shard_info
    for row in rows:
        yield asyncio.run(_run_bundle(row, config))


def run_open_ended_zephyr(
    bundles: Iterable[OpenEndedTaskBundle],
    *,
    config: OpenEndedZephyrConfig,
    output: str,
    max_workers: int = 32,
    logical_shards: int = 128,
    region: str | None = "us-central1",
    preemptible: bool = True,
) -> object:
    """Execute one outer Zephyr map over fully hydrated bundle rows."""
    rows = [{"bundle_json": bundle.model_dump_json()} for bundle in bundles]
    if not rows:
        raise ValueError("at least one bundle is required")
    shards = min(logical_shards, len(rows))
    dataset = Dataset.from_list(rows)
    if shards > 1:
        dataset = dataset.reshard(shards)
    pipeline = dataset.map_shard(partial(run_bundle_shard, config=config)).write_jsonl(output)
    regions = [region] if region else None
    with ZephyrContext(
        max_workers=min(max_workers, shards),
        resources=ResourceConfig(cpu=1, ram="4g", disk="16g", regions=regions, preemptible=preemptible),
        coordinator_resources=ResourceConfig(
            cpu=0.1,
            ram="2g",
            disk="8g",
            regions=regions,
            preemptible=preemptible,
        ),
        name="open-ended-taskforge",
        heartbeat_timeout=900,
    ) as context:
        return context.execute(pipeline, name="open-ended-taskforge")
