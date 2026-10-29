# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The generator baseline of gate 5 as the loop's generate box, in place of agentic Taskforge sessions.

An idea of the failed-run analysis names a capability, not a dataset, but its evidence cites failed attempts of
evaluation tasks. For each Reasoning Gym type among the cited tasks, the box asks GLM for one program that prints
instances of that type, with every evaluation task of the type as its examples, so it uses only what the evaluation
box showed. The dataset's own scorer grades the instances, as it grades the seed's tasks. An idea that cites no
Reasoning Gym task makes no tasks.
"""

import asyncio
import base64
import json
import logging
from collections import defaultdict, deque
from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from dataclasses import dataclass

from taskcompendium.models import TaskSpec
from taskforge.capability import CapabilityIdea
from taskforge.llm.glm_client import GlmClient

from experiments.post_training.rsi.baseline_synth import generator, valid
from experiments.post_training.rsi.feedback import CAPABILITY_TAG_PREFIX, cited_tasks, synthesized
from experiments.post_training.rsi.gap import CONFIG_PATH, contract_resource, entry_task

logger = logging.getLogger(__name__)

REASONING_GYM_DATASET = "reasoning_gym"


@dataclass(frozen=True)
class GeneratorConfig:
    client: GlmClient
    evaluation: Sequence[TaskSpec]
    """The evaluation box's tasks, which the ideas' evidence cites."""
    templates: Mapping[str, str]
    """A seed row of each Reasoning Gym type, the form of its synthesized tasks."""
    instances_per_type: int
    concurrency: int
    """Ideas whose programs GLM writes at the same time."""


def reasoning_gym_entry(task: TaskSpec) -> tuple[str, dict] | None:
    """The Reasoning Gym type and entry of a task: a generated row's, or a Nemotron Reasoning Gym row's, which keeps
    the type in its metadata; ``None`` for any other task."""
    payload = json.loads(task.model_dump_json())
    if not any(file["path"] == CONFIG_PATH for file in payload.get("resources", {}).get("verifier") or ()):
        return None
    terms = json.loads(base64.b64decode(contract_resource(payload)["source"]["content_base64"])).get("contract")
    if terms is None:
        return None
    if "entry" in terms and "task" in terms:
        return terms["task"], terms["entry"]
    name = (terms.get("metadata") or {}).get("source_dataset")
    if REASONING_GYM_DATASET not in str(terms.get("dataset")) or name is None:
        return None
    return name, {"question": terms["question"], "answer": terms["answer"], "metadata": terms["metadata"]}


def examples_by_type(evaluation: Sequence[TaskSpec]) -> dict[str, dict[str, dict]]:
    """The entries of the evaluation's Reasoning Gym tasks, by type and then by task ID."""
    examples: defaultdict[str, dict[str, dict]] = defaultdict(dict)
    for task in evaluation:
        if (found := reasoning_gym_entry(task)) is not None:
            name, entry = found
            examples[name][task.id] = entry
    return dict(examples)


def cited_types(idea: CapabilityIdea, examples: Mapping[str, Mapping[str, dict]]) -> list[str]:
    """The Reasoning Gym types of the evaluation tasks that ``idea`` cites, the most cited first."""
    cited = cited_tasks(idea)
    counts = {name: sum(task_id in entries for task_id in cited) for name, entries in examples.items()}
    return sorted((name for name, count in counts.items() if count), key=lambda name: -counts[name])


def typed_idea(idea: CapabilityIdea, entries: Mapping[str, dict]) -> CapabilityIdea:
    """``idea`` with the evaluation entries of one type as its examples, the form that ``generator`` reads."""
    examples = list(entries.values())
    capability = {
        **idea.capability,
        "example_tasks": [entry["question"] for entry in examples],
        "example_entries": examples,
    }
    return idea.model_copy(update={"capability": capability})


async def round_robin[T](batches: asyncio.Queue[list[T] | None]) -> AsyncIterator[T]:
    """One item of each batch in turn, over the batches that have arrived by then, until ``None`` arrives and every
    batch is used up. Training takes the synthesized share a task at a time, and the first ideas' instances alone can
    fill it; taking them in turn gives the ideas that come later their share."""
    rotation: deque[deque[T]] = deque()
    finished = False

    def take(batch: list[T] | None) -> None:
        nonlocal finished
        if batch is None:
            finished = True
        elif batch:
            rotation.append(deque(batch))

    while True:
        while not batches.empty():
            take(batches.get_nowait())
        if not rotation:
            if finished:
                return
            take(await batches.get())
            continue
        batch = rotation.popleft()
        yield batch.popleft()
        if batch:
            rotation.append(batch)


async def generator_tasks(ideas: AsyncIterable[CapabilityIdea], config: GeneratorConfig) -> AsyncIterator[TaskSpec]:
    """The valid instances that one generator program per cited type makes for each idea, as each idea finishes."""
    examples = examples_by_type(config.evaluation)
    made: asyncio.Queue[list[TaskSpec] | None] = asyncio.Queue()
    slots = asyncio.Semaphore(config.concurrency)

    async def one(idea: CapabilityIdea) -> None:
        types = [name for name in cited_types(idea, examples) if name in config.templates]
        if not types:
            logger.info("%s cites no Reasoning Gym task of a seed type; it makes no tasks", idea.capability_id)
            return
        capability = f"{CAPABILITY_TAG_PREFIX}{idea.capability_id}"
        for name in types:
            try:
                async with slots:
                    pairs = await generator(config.client, typed_idea(idea, examples[name]), config.instances_per_type)
            except Exception:
                # A program that GLM fails to write, for example at a cut reply, stops neither the idea's other
                # types nor the loop.
                logger.exception("The generator of %s, %s failed", idea.capability_id, name)
                continue
            kept = [(question, entry) for question, entry in pairs if valid(name, question, entry)]
            logger.info("%s, %s: %d of %d instances are valid", idea.capability_id, name, len(kept), len(pairs))
            made.put_nowait(
                [
                    synthesized(
                        entry_task(config.templates[name], name, f"{name}.generator-{k}", question, entry, capability),
                        idea,
                    )
                    for k, (question, entry) in enumerate(kept)
                ]
            )

    async def run_all() -> None:
        running: set[asyncio.Task[None]] = set()
        try:
            async for idea in ideas:
                running.add(asyncio.create_task(one(idea)))
            await asyncio.gather(*running)
        finally:
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            made.put_nowait(None)

    runner = asyncio.create_task(run_all())
    try:
        async for task in round_robin(made):
            yield task
        await runner
    finally:
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
