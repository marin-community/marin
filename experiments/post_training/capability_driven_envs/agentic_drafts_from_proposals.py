# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Draft tasks from the accepted proposals of a curriculum proposal run, with GLM on Ortet and no other model.

Each accepted reasoning or shellsim proposal gets one short ``taskforge.agentic.drafts`` session. When ``submit``
accepts its task, the build program runs again for ``--instances`` task ids with no model call. A Zephyr pipeline
splits the proposals into shards of ``--shard-size``; each worker runs the sessions of its shard concurrently, at most
``--sessions`` at a time, and their GLM turns go through one ``GlmBatchTransport`` (``--transport batch``) or straight
to the relay (``--transport stream``). For each proposal the run writes, under
``<output-root>/<run>/<capability>/slot-NN/``:

- ``tasks/<task id>.json``: the instances;
- ``workspace.tgz``: the workspace that the build program reads, with the program;
- ``agent.json.gz``: the agent's conversation;
- ``result.json``: the proposal, the program, turns, tokens, wall time, the instance ids and the failure.

A proposal that has a ``result.json`` is skipped, so a relaunch with the same ``--run`` resumes. A session that ends
in a model or machine failure writes no ``result.json``, so the relaunch drafts it again. Bubblewrap needs the
privileged container profile, which the driver's job, the coordinator and the workers need. Run it inside an Iris job on
``cw-rno2a``, with ``GLM_BULK_TOKEN`` set::

    iris --cluster=cw-rno2a job run --priority batch --cpu 1 --memory 2GB --extra cpu --timeout 0 \\
        --container-profile container_profile_privileged -e GLM_BULK_TOKEN "$(cat ~/.config/glm/bulk_token)" -- \\
        python -m experiments.post_training.capability_driven_envs.agentic_drafts_from_proposals \\
        --run probe-20261106-a --limit 20 --workers 1
"""

import asyncio
import dataclasses
import gzip
import json
import logging
import math
import time
import uuid
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import click
import httpx
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from rolloutengine.models import RolloutOperation
from shellbox.machine import Backend
from taskforge.agentic.drafts import DraftConfig, DraftProposal, DraftResult, draft_task
from taskforge.agentic.instances import task_instances
from taskforge.agentic.tools import PROGRAM_TIMEOUT
from taskforge.llm.glm_client import EndpointSource, GlmEndpoint, Pool
from verifyit.modes.grade_judge import JudgeConnection
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.capability_driven_envs.generate_tasks import RelayEndpoint, pool_token
from experiments.post_training.capability_driven_envs.generate_tasks_agentic import (
    COMMAND_TIMEOUT,
    DEEPSEEK_MODEL,
    SOLVER_LIMITS,
    RelayModel,
    agent_limits,
    chat_model,
    solver_factory,
    workspace_factory,
)
from experiments.post_training.capability_driven_envs.glm_batch import GlmBatchTransport, bulk_batch_api
from experiments.post_training.glm import GLM_MODEL

logger = logging.getLogger(__name__)

PROPOSALS = "s3://marin-us-east-02a/marin/taskforge/proposals/curriculum-v3-n25-20261102/run"
OUTPUT_ROOT = "s3://marin-us-east-08a/marin/taskforge/agentic_drafts"
RELAY_JOB = "/muchanem/glm53-relay-rno2a"
ACCEPT = "accept"
RESULT_FILE = "result.json"
PRIVILEGED = "CONTAINER_PROFILE_PRIVILEGED"
LISTING_THREADS = 32
"""Threads that read the capabilities' ``result.json`` files when the driver lists the accepted slots."""
BATCH_FLUSH = 1.0
BATCH_POLL = 5.0
SESSION_TIMEOUT = 3600
"""The default seconds that one draft session can take. A session that hangs gives its slot back and is drafted again
on the next launch, because a failed session writes no ``result.json``. A retry launch of the slots that timed out
can give them longer with ``--session-timeout``."""


class Transport(StrEnum):
    BATCH = "batch"
    STREAM = "stream"


@dataclass(frozen=True)
class Slot:
    capability: str
    slot: int

    @property
    def name(self) -> str:
        return f"slot-{self.slot:02d}"


@dataclass(frozen=True)
class RunSettings:
    """What every shard of a run needs. The bulk token comes from the worker's environment, not from here."""

    proposals: str
    output: str
    relay_job: str
    transport: Transport
    sessions: int
    turns: int
    instances: int
    agent_reasoning_effort: str
    batch_size: int
    concurrent_batches: int
    session_timeout: float


def accepted_slots(proposals: StoragePath) -> list[Slot]:
    """The slots whose verdict is ``accept``, from the ``result.json`` of each capability of the proposal run."""
    results = sorted((proposals / "*" / RESULT_FILE).glob(), key=str)

    def read(path: StoragePath) -> list[Slot]:
        verdicts = json.loads(path.read_text())["slots"]
        return [Slot(path.parent.name, int(slot)) for slot, verdict in verdicts.items() if verdict == ACCEPT]

    with ThreadPoolExecutor(LISTING_THREADS) as pool:
        return [slot for slots in pool.map(read, results) for slot in slots]


def listed_slots(path: StoragePath) -> set[Slot]:
    """The slots that a JSONL file names, one ``{"slot": "<capability>/<n>"}`` object per line, such as a redraft
    list from validation."""
    lines = (json.loads(line)["slot"] for line in path.read_text().splitlines() if line.strip())
    return {Slot(capability, int(number)) for capability, number in (line.rsplit("/", 1) for line in lines)}


def draft_shard(slots: Sequence[Slot], settings: RunSettings) -> Iterator[dict[str, Any]]:
    """Draft the proposals of one shard, and give the summary line of each."""
    return iter(asyncio.run(_draft_all(slots, settings)))


class Models:
    """The agent models of a worker. Batch models of one endpoint share one transport, so that the turns of all the
    sessions share batches."""

    def __init__(self, settings: RunSettings):
        self.settings = settings
        self._transports: dict[str, GlmBatchTransport] = {}

    def connect(self, endpoint: GlmEndpoint, cache_key: str):
        http_client = None
        if self.settings.transport is Transport.BATCH:
            if endpoint.base_url not in self._transports:
                self._transports[endpoint.base_url] = GlmBatchTransport(
                    bulk_batch_api(endpoint.base_url.removesuffix("/v1"), endpoint.token),
                    batch_size=self.settings.batch_size,
                    flush=BATCH_FLUSH,
                    poll=BATCH_POLL,
                    concurrent_batches=self.settings.concurrent_batches,
                )
            http_client = httpx.AsyncClient(transport=self._transports[endpoint.base_url])
        return chat_model(
            endpoint,
            model=GLM_MODEL,
            reasoning_effort=self.settings.agent_reasoning_effort,
            cache_key=cache_key,
            http_client=http_client,
        )

    async def close(self) -> None:
        for transport in self._transports.values():
            await transport.aclose()


async def _draft_all(slots: Sequence[Slot], settings: RunSettings) -> list[dict[str, Any]]:
    source: EndpointSource = RelayEndpoint(settings.relay_job, pool_token(Pool.BULK), Pool.BULK)
    endpoint = source.current()
    models = Models(settings)
    workspace = workspace_factory()
    config = DraftConfig(
        agent_model=RelayModel(source, lambda endpoint: models.connect(endpoint, "unused")),
        workspace_factory=workspace,
        solver_factories={backend: solver_factory(backend) for backend in (Backend.SHELLSIM, Backend.LOCAL)},
        judge=JudgeConnection(endpoint.base_url, endpoint.token, default_model=DEEPSEEK_MODEL),
        agent_limits=agent_limits(settings.turns).model_copy(update={"attempt_timeout": settings.session_timeout}),
        solver_limits=SOLVER_LIMITS,
        command_timeout=COMMAND_TIMEOUT,
    )
    slots_open = asyncio.Semaphore(settings.sessions)
    model_failures = 0

    async def one(slot: Slot) -> dict[str, Any]:
        nonlocal model_failures
        directory = StoragePath(settings.output) / slot.capability / slot.name
        if (directory / RESULT_FILE).exists():
            return {"capability": slot.capability, "slot": slot.slot, "skipped": "done"}
        text = (StoragePath(settings.proposals) / slot.capability / f"{slot.name}.proposal.json").read_text()
        header = json.loads(text)["header"]
        if header["environment"] == "container":
            return {"capability": slot.capability, "slot": slot.slot, "skipped": "container"}
        proposal = DraftProposal.from_json(text)
        # Each session sends its whole conversation on every turn, so a cache key of its own lets the relay reuse
        # the prefix.
        cache_key = f"agentic-drafts-{proposal.task_id}-{uuid.uuid4().hex[:8]}"
        model = RelayModel(source, lambda endpoint: models.connect(endpoint, cache_key))
        async with slots_open:
            started = time.monotonic()
            result = await draft_task(proposal, dataclasses.replace(config, agent_model=model))
            ids = [f"{proposal.task_id}--{k}" for k in range(1, settings.instances + 1)]
            instances = (
                []
                if result.draft is None
                else await task_instances(result.draft, ids, workspace, timeout=PROGRAM_TIMEOUT)
            )
            seconds = time.monotonic() - started
        summary = write_draft(directory, slot, result, instances, seconds)
        logger.info("%s: %d instances, failure %s", proposal.id, len(instances), result.failure)
        failed_on_model = result.failure is not None and result.failure.startswith(f"{RolloutOperation.MODEL}:")
        model_failures = model_failures + 1 if failed_on_model else 0
        if model_failures >= settings.sessions:
            # Ortet is down, for example a relay whose router is offline: each session would fail on its first turn
            # and move on to the next proposal. Stop, so that a relaunch drafts the rest once Ortet is back.
            raise RuntimeError(f"{model_failures} sessions in a row failed on a model call: {result.failure}")
        return summary

    try:
        return await asyncio.gather(*(one(slot) for slot in slots))
    finally:
        await models.close()


def write_draft(
    directory: StoragePath, slot: Slot, result: DraftResult, instances: Sequence, seconds: float
) -> dict[str, Any]:
    """Write the files of one draft session under ``directory`` and return its summary line.

    A failed session without a draft writes no ``result.json``, so that a relaunch drafts the proposal again.
    """
    for task in instances:
        (directory / "tasks" / f"{task.id}.json").write_text(task.model_dump_json(indent=2))
    if result.draft is not None:
        (directory / "workspace.tgz").write_bytes(result.draft.workspace)
    conversations = [list(conversation.messages) for conversation in result.rollout.conversations]
    (directory / "agent.json.gz").write_bytes(gzip.compress(json.dumps(conversations, default=str).encode()))
    summary = {
        "capability": slot.capability,
        "slot": slot.slot,
        "proposal": result.proposal.id,
        "environment": result.proposal.environment,
        "verification": result.proposal.verification,
        "program": result.draft.program if result.draft is not None else None,
        "instances": [task.id for task in instances],
        "submits": result.submits,
        "turns": sum(
            1 for conversation in conversations for message in conversation if message.get("role") == "assistant"
        ),
        "agent_tokens": result.agent_tokens,
        "stop_reason": result.rollout.stop_reason,
        "report": result.report,
        "failure": result.failure,
        "seconds": round(seconds),
    }
    if result.draft is not None or result.failure is None:
        (directory / RESULT_FILE).write_text(json.dumps(summary, indent=2))
    return summary


@click.command()
@click.option("--run", "run_name", required=True, help="The name of this launch; a relaunch with the same name resumes.")
@click.option("--proposals", default=PROPOSALS, show_default=True, help="The run/ directory of the proposal run.")
@click.option("--output-root", default=OUTPUT_ROOT, show_default=True)
@click.option("--relay-job", default=RELAY_JOB, show_default=True)
@click.option(
    "--transport", type=click.Choice([t.value for t in Transport]), default=Transport.BATCH.value, show_default=True
)
@click.option(
    "--offset", type=click.IntRange(min=0), default=0, show_default=True, help="Skip the first N selected slots."
)
@click.option("--limit", type=click.IntRange(min=1), help="Draft only N accepted slots, for probes and ramps.")
@click.option(
    "--slots-from",
    help='A JSONL file of {"slot": "<capability>/<n>"} lines: draft only these accepted slots, for redrafts.',
)
@click.option(
    "--stride",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Take every Nth slot, to spread a probe over capabilities.",
)
@click.option("--workers", type=click.IntRange(min=1), required=True, help="Zephyr workers.")
@click.option("--shard-size", type=click.IntRange(min=1), default=64, show_default=True)
@click.option(
    "--sessions", type=click.IntRange(min=1), default=64, show_default=True, help="Concurrent sessions per worker."
)
@click.option("--turns", type=click.IntRange(min=1), default=30, show_default=True)
@click.option("--instances", type=click.IntRange(min=1), default=5, show_default=True)
@click.option("--agent-reasoning-effort", default="low", show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=64, show_default=True)
@click.option("--concurrent-batches", type=click.IntRange(min=1), default=4, show_default=True)
@click.option(
    "--session-timeout",
    type=click.FloatRange(min=60),
    default=SESSION_TIMEOUT,
    show_default=True,
    help="Seconds that one draft session can take.",
)
@click.option("--worker-cpu", type=float, default=8, show_default=True)
@click.option("--worker-ram", default="32g", show_default=True)
def main(
    run_name: str,
    proposals: str,
    output_root: str,
    relay_job: str,
    transport: str,
    offset: int,
    limit: int | None,
    slots_from: str | None,
    stride: int,
    workers: int,
    shard_size: int,
    sessions: int,
    turns: int,
    instances: int,
    agent_reasoning_effort: str,
    batch_size: int,
    concurrent_batches: int,
    session_timeout: float,
    worker_cpu: float,
    worker_ram: str,
) -> None:
    """Draft tasks from the accepted proposals with GLM, on a Zephyr pool."""
    configure_logging(logging.INFO)
    output = StoragePath(output_root) / run_name
    selected = accepted_slots(StoragePath(proposals))
    if slots_from is not None:
        listed = listed_slots(StoragePath(slots_from))
        selected = [slot for slot in selected if slot in listed]
        if len(selected) != len(listed):
            raise ValueError(f"{len(listed) - len(selected)} slots of {slots_from} are not accepted proposals")
    selected = selected[::stride][offset:]
    slots = selected[:limit] if limit is not None else selected
    logger.info("%d accepted slots to draft into %s", len(slots), output)
    settings = RunSettings(
        proposals,
        str(output),
        relay_job,
        Transport(transport),
        sessions,
        turns,
        instances,
        agent_reasoning_effort,
        batch_size,
        concurrent_batches,
        session_timeout,
    )
    shards = [slots[start : start + shard_size] for start in range(0, len(slots), shard_size)]
    dataset = (
        Dataset.from_list(shards)
        .flat_map(lambda shard: draft_shard(shard, settings))
        .write_jsonl(str(output / "summary" / "part-{shard:05d}.jsonl"))
    )
    resources = ResourceConfig(cpu=worker_cpu, ram=worker_ram, disk="50g", container_profile=PRIVILEGED)
    # A task may give a child job an elevated profile only when it has that profile itself, so the workers' parent,
    # the coordinator, gets it too, as the driver's own job does (--container-profile container_profile_privileged).
    coordinator = ResourceConfig(cpu=0.1, ram="4g", preemptible=False, container_profile=PRIVILEGED)
    context = ZephyrContext(
        resources=resources,
        coordinator_resources=coordinator,
        max_workers=min(workers, len(shards)),
        name="agentic-drafts",
    )
    context.execute(dataset, verbose=True)
    logger.info("Drafted %d shards: summaries in %s", math.ceil(len(slots) / shard_size), output / "summary")


if __name__ == "__main__":
    main()
