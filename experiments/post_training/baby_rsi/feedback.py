# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The feedback path of the RSI loop: failed rollouts become capability ideas, and ideas become new tasks.

``failed_runs`` keeps one failed attempt of each task that the policy mostly fails. ``analyze_failed_runs`` runs the
failed-run analysis on each batch of them and yields the capability ideas that it has not yielded before.
``generate_tasks`` runs one agentic Taskforge session for each idea, with the served policy as the solver, so that
the gate accepts tasks that the policy solves only sometimes. For each accepted task, its build program makes more
instances for training, and held-out instances that only the evaluation sees.
"""

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass

from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import GradedRollout
from taskcompendium.models import TaskSpec
from taskforge.agentic import agent as agentic
from taskforge.agentic.config import AgenticConfig
from taskforge.agentic.instances import task_instances
from taskforge.agentic.tools import AcceptedTask, content_digest
from taskforge.capability import CapabilityIdea
from taskforge.llm.policy import LLMPolicy
from zephyr.context import ZephyrContext

from experiments.post_training.baby_rsi.rollouts import PASS_REWARD, TaskRollouts, reward
from experiments.post_training.capability_driven_envs.analyze_failed_runs import pipeline as failed_run_analysis
from experiments.post_training.capability_driven_envs.analyze_failed_runs.pipeline import MakeClient
from experiments.post_training.capability_driven_envs.analyze_failed_runs.rollouts import policy_failed

logger = logging.getLogger(__name__)

SYNTHESIZED_TAG = "baby-rsi.synthesized"
CAPABILITY_TAG_PREFIX = "capability:"
SYNTHESIZED_LABEL_PREFIX = "synthesized:"
HELD_OUT_MARKER = "held-out"
"""The file in an idea's directory that marks a held-out idea, whose tasks the loop never trains on."""
"""The record label of a synthesized task: the prefix and its capability ID."""
INSTANCE_TIMEOUT = 120.0

FailedRun = tuple[TaskSpec, GradedRollout]


async def failed_runs(groups: AsyncIterable[TaskRollouts], max_pass_rate: float) -> AsyncIterator[FailedRun]:
    """One failed attempt of each task whose pass rate is at most ``max_pass_rate``: a capability that it lacks."""
    async for group in groups:
        rate = group.pass_rate
        if rate is None or rate > max_pass_rate:
            continue
        failed = [
            graded for graded in group.rollouts if policy_failed(graded, PASS_REWARD) and reward(graded) is not None
        ]
        if failed:
            yield group.task, failed[0]


@dataclass(frozen=True)
class AnalysisConfig:
    make_client: MakeClient
    max_new_ideas: int
    shard_size: int
    extract_policy: LLMPolicy
    merge_policy: LLMPolicy
    work_dir: str
    """Where Zephyr keeps the chunks of a run."""
    workers: int


async def analyze_failed_runs(
    batches: AsyncIterable[list[FailedRun]], config: AnalysisConfig, *, known: Iterable[str] = ()
) -> AsyncIterator[CapabilityIdea]:
    """The capability ideas of each batch of failed runs, each capability ID once and none of ``known``."""
    seen = set(known)
    analysis = 0
    async for batch in batches:
        analysis += 1
        plan = failed_run_analysis.analyze_failed_runs(
            batch,
            config.make_client,
            pass_threshold=PASS_REWARD,
            max_new_ideas=config.max_new_ideas,
            shard_size=config.shard_size,
            extract_ideas_llm_policy=config.extract_policy,
            merge_ideas_llm_policy=config.merge_policy,
            catalog_version=f"baby-rsi-analysis-{analysis}",
        )
        try:
            ideas = await asyncio.to_thread(_execute, plan, f"{config.work_dir}/analysis-{analysis}", config.workers)
        except Exception:
            # An analysis that fails, for example at a model outage, does not stop the loop; the next batch comes.
            logger.exception("Analysis %d failed", analysis)
            continue
        logger.info("Analysis %d of %d failed runs: %d ideas", analysis, len(batch), len(ideas))
        for idea in ideas:
            if idea.capability_id not in seen:
                seen.add(idea.capability_id)
                logger.info("Idea %s: %s", idea.capability_id, idea.capability.get("outcome"))
                yield idea


def _execute(plan, work_dir: str, workers: int) -> list[CapabilityIdea]:
    # The workers run in threads of this process: the policy and the model clients are local.
    client = LocalClient(max_threads=workers + 4)
    with (
        set_current_client(client),
        ZephyrContext(client=client, max_workers=workers, chunk_storage_prefix=work_dir) as context,
    ):
        return list(context.execute(plan).results)


@dataclass(frozen=True)
class SynthesisConfig:
    agentic: AgenticConfig
    tasks_per_idea: int
    instances_per_task: int
    """Training instances that the build program of each accepted task makes, beside the accepted task itself."""
    probes_per_task: int
    """Held-out instances of each accepted task, written for the evaluation and never trained on."""
    concurrency: int
    """Agent sessions at the same time."""
    output: StoragePath
    held_out_idea_share: float
    """The share of ideas whose tasks are never trained on: their probes measure transfer to unseen families."""


async def generate_tasks(ideas: AsyncIterable[CapabilityIdea], config: SynthesisConfig) -> AsyncIterator[TaskSpec]:
    """The training tasks that agentic Taskforge sessions make for ``ideas``, as each session ends."""
    made: asyncio.Queue[TaskSpec | None] = asyncio.Queue()
    slots = asyncio.Semaphore(config.concurrency)

    async def session(idea: CapabilityIdea) -> None:
        try:
            async with slots:
                result = await agentic.generate_tasks(idea, config.tasks_per_idea, config.agentic)
            logger.info("%s: %d tasks accepted, failure %s", idea.capability_id, len(result.tasks), result.failure)
            instances = [await _instances(accepted, config) for accepted in result.tasks]
            unseen = held_out_idea(idea, config.held_out_idea_share)
            write_session(config.output, result, [probes for _, probes in instances], held_out=unseen)
            if unseen:
                logger.info("%s is held out: its tasks are only probes", idea.capability_id)
                return
            for accepted, (training, _) in zip(result.tasks, instances, strict=True):
                for task in (accepted.task, *training):
                    made.put_nowait(synthesized(task, idea))
        except Exception:
            # One idea whose session fails, for example at a model outage, does not stop the loop.
            logger.exception("The synthesis session of %s failed", idea.capability_id)

    async def start_sessions() -> None:
        sessions: set[asyncio.Task[None]] = set()
        try:
            async for idea in ideas:
                sessions.add(asyncio.create_task(session(idea)))
            await asyncio.gather(*sessions)
        finally:
            for task in sessions:
                task.cancel()
            # The sessions release their machines before the caller closes the clients that they use.
            await asyncio.gather(*sessions, return_exceptions=True)
            made.put_nowait(None)

    runner = asyncio.create_task(start_sessions())
    try:
        while (task := await made.get()) is not None:
            yield task
        await runner
    finally:
        runner.cancel()
        with suppress(asyncio.CancelledError):
            await runner


def held_out_idea(idea: CapabilityIdea, share: float) -> bool:
    """Whether ``idea`` falls in the held-out ``share`` of ideas, by a stable hash of its capability ID."""
    return int(hashlib.sha256(idea.capability_id.encode()).hexdigest()[:8], 16) < share * 2**32


def training_ids(accepted: AcceptedTask, count: int) -> list[str]:
    return [f"{accepted.task.id}.train-{k}" for k in range(count)]


def probe_ids(accepted: AcceptedTask, count: int) -> list[str]:
    return [f"{accepted.task.id}.probe-{k}" for k in range(count)]


def held_out(probes: Sequence[TaskSpec], accepted: AcceptedTask, training: Sequence[TaskSpec]) -> list[TaskSpec]:
    """The probes whose content differs from the accepted task and every training instance.

    A program that seeds its data from a number in the task id writes the same task for ``train-0`` and ``probe-0``.
    """
    seen = {content_digest(task) for task in (accepted.task, *training)}
    return [probe for probe in probes if content_digest(probe) not in seen]


async def _instances(accepted: AcceptedTask, config: SynthesisConfig) -> tuple[list[TaskSpec], list[TaskSpec]]:
    """The training instances and the held-out probes of an accepted task."""
    factory = config.agentic.workspace_factory
    training = await task_instances(
        accepted, training_ids(accepted, config.instances_per_task), factory, timeout=INSTANCE_TIMEOUT
    )
    probes = await task_instances(
        accepted, probe_ids(accepted, config.probes_per_task), factory, timeout=INSTANCE_TIMEOUT
    )
    return training, held_out(probes, accepted, training)


def synthesized(task: TaskSpec, idea: CapabilityIdea) -> TaskSpec:
    """``task`` with the tags that mark it as synthesized for ``idea``, and an ID that is unique across ideas."""
    tags = (*task.tags, SYNTHESIZED_TAG, f"{CAPABILITY_TAG_PREFIX}{idea.capability_id}")
    return task.model_copy(update={"id": synthesized_id(idea, task.id), "tags": tuple(dict.fromkeys(tags))})


def synthesized_id(idea: CapabilityIdea, task_id: str) -> str:
    """Two sessions can give their tasks the same ID, so the loop prefixes the capability ID."""
    return f"{idea.capability_id}.{task_id}"


def capability_of(task: TaskSpec) -> str | None:
    return next(
        (tag.removeprefix(CAPABILITY_TAG_PREFIX) for tag in task.tags if tag.startswith(CAPABILITY_TAG_PREFIX)), None
    )


def write_session(
    output: StoragePath, result: agentic.AgentResult, probes: Sequence[Sequence[TaskSpec]], *, held_out: bool
) -> None:
    """The idea, the agent's report and spend, and each accepted task with its program, workspace and probes.

    A held-out idea gets the ``HELD_OUT_MARKER`` file. ``task.json`` comes last in a task's directory and
    ``idea.json`` last in the idea's, so that a resumed run reads only complete records.
    """
    directory = output / "ideas" / result.idea.capability_id
    summary = {
        "tasks": [accepted.task.id for accepted in result.tasks],
        "report": result.report,
        "failure": result.failure,
        "agent_tokens": result.agent_tokens,
        "spend": asdict(result.spend),
    }
    (directory / "session.json").write_text(_json(summary))
    conversations = [list(conversation.messages) for conversation in result.rollout.conversations]
    (directory / "agent.json").write_text(_json(conversations))
    for accepted, task_probes in zip(result.tasks, probes, strict=True):
        task_directory = directory / "tasks" / accepted.task.id
        (task_directory / "evidence.json").write_text(_json(asdict(accepted.evidence)))
        (task_directory / "review.json").write_text(accepted.review.model_dump_json(indent=1))
        (task_directory / "program.txt").write_text(accepted.program)
        (task_directory / "workspace.tgz").write_bytes(accepted.workspace)
        lines = "".join(synthesized(probe, result.idea).model_dump_json() + "\n" for probe in task_probes)
        (task_directory / "probes.jsonl").write_text(lines)
        (task_directory / "task.json").write_text(accepted.task.model_dump_json(indent=1))
    if held_out:
        (directory / HELD_OUT_MARKER).write_text("")
    (directory / "idea.json").write_text(result.idea.model_dump_json(indent=1))


def _json(value: object) -> str:
    return json.dumps(value, indent=1, default=str)
