# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-map Zephyr port for complete open-ended Taskforge bundle lifecycles."""

import asyncio
import json
import tempfile
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy
from taskforge.stage10_proposal.model import Environment, SolverMachine
from taskforge.stage40_build_with_program.run import DRAFT_DIR, load_draft, step_output
from taskforge.stage40_build_with_program.sdk import solver_machine
from taskforge.stage40_build_with_program.services import RLEnvBuildServices
from taskforge.stage50_validate.round import VALIDATION_FILE
from taskforge.stage_coordinator.coordinator import DECISION_FILE, STORE_DIR
from taskforge.stage_coordinator.event_log import EVENTS_FILE, WORKPIECES_DIR
from taskforge_adapters.campaign import OpenAIModelPort, campaign_worker
from verifyit.modes.grade_judge import JudgeConnection
from verifyit.spec import JudgeSpec

from experiments.post_training.capability_driven_envs.open_ended import OpenEndedTaskBundle
from experiments.post_training.capability_driven_envs.open_ended_coordinator import (
    CALIBRATION_FILE,
    open_ended_coordinator,
    verifyit_control_scorer,
)
from experiments.post_training.capability_driven_envs.open_ended_proposal_source import (
    OpenEndedProposalConfig,
    OpenEndedProposalSource,
)
from experiments.post_training.capability_driven_envs.open_ended_runner import OpenEndedRuntime, run_open_ended_bundle
from experiments.post_training.capability_driven_envs.open_ended_task_design import OpenEndedTaskDesign
from experiments.post_training.taskforge_zephyr import TaskforgeZephyrConfig, run_taskforge_zephyr


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
        for key, name in (
            ("validation", VALIDATION_FILE),
            ("calibration", CALIBRATION_FILE),
            ("decision", DECISION_FILE),
        ):
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
    with tempfile.TemporaryDirectory(prefix=f"open-ended-{bundle.bundle_id}-") as temporary:
        root = Path(temporary)
        async with campaign_worker(config.port) as worker:
            connection = JudgeConnection(worker.endpoint.current().base_url, worker.token)
            return await run_bundle_lifecycle(
                bundle,
                model=worker.model,
                policy=worker.policy,
                services=worker.services,
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
    return run_taskforge_zephyr(
        rows,
        run_shard=partial(run_bundle_shard, config=config),
        output=output,
        config=TaskforgeZephyrConfig(
            name="open-ended-taskforge",
            max_workers=max_workers,
            logical_shards=logical_shards,
            region=region,
            preemptible=preemptible,
        ),
    )
