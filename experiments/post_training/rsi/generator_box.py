# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The generator baseline of gate 5 as the loop's generate box, in place of agentic Taskforge sessions.

An idea of the failed-run analysis names a capability, not a dataset, but its evidence cites failed attempts of
evaluation tasks. For each generator type among the cited tasks (a Reasoning Gym type, or the calendar family), the
box asks GLM for one program that prints instances of that type, with evaluation tasks of the type as its examples, so
it uses only what the evaluation box showed. Each type's own scorer checks and grades the instances, as it grades the
seed's tasks. An idea that cites no task of a generator type makes no tasks.
"""

import asyncio
import base64
import json
import logging
from collections import defaultdict, deque
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass

from taskcompendium.models import TaskSpec
from taskforge.capability import CapabilityIdea
from taskforge.llm.glm_client import GlmClient

from experiments.post_training.rsi.baseline_synth import REASONING_GYM_ENTRY, generator, valid
from experiments.post_training.rsi.calendar_tasks import (
    CALENDAR,
    CalendarScorer,
    calendar_entry,
    calendar_task,
    valid_calendar,
)
from experiments.post_training.rsi.feedback import CAPABILITY_TAG_PREFIX, cited_tasks, synthesized
from experiments.post_training.rsi.gap import CONFIG_PATH, SYNTHESIZED_TAG, contract_resource, entry_task

logger = logging.getLogger(__name__)

REASONING_GYM_DATASET = "reasoning_gym"
CALENDAR_ENTRY = (
    "the conversation as messages, a list of role and content turns in the form of the examples, which starts with "
    "the same system prompt and ends with the user's last request, and exp_cal_state, the expected state of every "
    "event in the exact form of the examples', which the conversation must ask for"
)
CALENDAR_EXAMPLES = 3
"""Calendar conversations are long: a program sees this many."""


@dataclass(frozen=True)
class GeneratorConfig:
    client: GlmClient
    evaluation: Sequence[TaskSpec]
    """The evaluation box's tasks, which the ideas' evidence cites."""
    templates: Mapping[str, str]
    """A seed row of each Reasoning Gym type, the form of its synthesized tasks; calendar tasks take the form of an
    evaluation calendar task."""
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


Example = tuple[str, dict]
"""A task's question and its entry."""


@dataclass(frozen=True)
class Family:
    """How the box makes the tasks of one generator type."""

    entry_form: str
    """How the program's prompt describes an entry."""
    max_examples: int | None
    valid: Callable[[str, dict], bool]
    task: Callable[[str, str, dict, str], TaskSpec]
    """A task from its ID, question, entry and capability tag."""


def task_example(task: TaskSpec) -> tuple[str, Example] | None:
    """The generator type of a task and its example: a Reasoning Gym task's, or a calendar task's, whose question is
    the user's last request; ``None`` for any other task."""
    if (found := reasoning_gym_entry(task)) is not None:
        name, entry = found
        return name, (entry["question"], entry)
    if (entry := calendar_entry(task)) is not None:
        return CALENDAR, (entry["messages"][-1]["content"], entry)
    return None


def examples_by_type(evaluation: Sequence[TaskSpec]) -> dict[str, dict[str, Example]]:
    """The examples of the evaluation's tasks of generator types, by type and then by task ID."""
    examples: defaultdict[str, dict[str, Example]] = defaultdict(dict)
    for task in evaluation:
        if (found := task_example(task)) is not None:
            name, example = found
            examples[name][task.id] = example
    return dict(examples)


def families(evaluation: Sequence[TaskSpec], templates: Mapping[str, str]) -> dict[str, Family]:
    """The family of each Reasoning Gym type with a seed template, and of the calendar family when the evaluation has
    a calendar task, which is then the template of its tasks and the source of its scorer."""
    made = {
        name: Family(
            entry_form=REASONING_GYM_ENTRY,
            max_examples=None,
            valid=lambda question, entry, name=name: valid(name, question, entry),
            task=lambda task_id, question, entry, capability, name=name: entry_task(
                templates[name], name, task_id, question, entry, capability
            ),
        )
        for name in templates
    }
    calendar = next((task for task in evaluation if calendar_entry(task) is not None), None)
    if calendar is not None:
        scorer = CalendarScorer.of(calendar)
        made[CALENDAR] = Family(
            entry_form=CALENDAR_ENTRY,
            max_examples=CALENDAR_EXAMPLES,
            valid=lambda question, entry: valid_calendar(entry, scorer),
            task=lambda task_id, question, entry, capability: calendar_task(
                calendar, task_id, entry, capability, (SYNTHESIZED_TAG,)
            ),
        )
    return made


def cited_types(idea: CapabilityIdea, examples: Mapping[str, Mapping[str, Example]]) -> dict[str, int]:
    """The generator types of the evaluation tasks that ``idea`` cites, with how many it cites, the most first."""
    cited = cited_tasks(idea)
    counts = {name: sum(task_id in entries for task_id in cited) for name, entries in examples.items()}
    return {name: counts[name] for name in sorted(counts, key=lambda name: -counts[name]) if counts[name]}


def typed_idea(idea: CapabilityIdea, examples: Mapping[str, Example], limit: int | None = None) -> CapabilityIdea:
    """``idea`` with the evaluation examples of one type (at most ``limit``) as its examples, the form that
    ``generator`` reads."""
    chosen = list(examples.values())[:limit]
    capability = {
        **idea.capability,
        "example_tasks": [question for question, _ in chosen],
        "example_entries": [entry for _, entry in chosen],
    }
    return idea.model_copy(update={"capability": capability})


@dataclass
class Batch[T]:
    weight: int
    """Its share of the turns: the failed evaluation tasks of its type that its idea cites."""
    items: deque[T]
    credit: float = 0.0


async def in_turns[T](batches: asyncio.Queue[tuple[int, list[T]] | None]) -> AsyncIterator[T]:
    """The items of the weighted batches that have arrived, in turns in proportion to their weights (smooth weighted
    round robin), until ``None`` arrives and every batch is used up. Training takes the synthesized share a task at a
    time and the first ideas' instances alone can fill it, so the ideas that come later need turns; the weights give
    the most turns to the types that most failed evaluation tasks show."""
    rotation: list[Batch[T]] = []
    finished = False

    def take(batch: tuple[int, list[T]] | None) -> None:
        nonlocal finished
        if batch is None:
            finished = True
        elif batch[1]:
            rotation.append(Batch(batch[0], deque(batch[1])))

    while True:
        while not batches.empty():
            take(batches.get_nowait())
        if not rotation:
            if finished:
                return
            take(await batches.get())
            continue
        for batch in rotation:
            batch.credit += batch.weight
        chosen = max(rotation, key=lambda batch: batch.credit)
        chosen.credit -= sum(batch.weight for batch in rotation)
        yield chosen.items.popleft()
        if not chosen.items:
            rotation.remove(chosen)


async def generator_tasks(ideas: AsyncIterable[CapabilityIdea], config: GeneratorConfig) -> AsyncIterator[TaskSpec]:
    """The valid instances that one generator program per cited type makes for each idea, as each idea finishes."""
    examples = examples_by_type(config.evaluation)
    kinds = families(config.evaluation, config.templates)
    made: asyncio.Queue[tuple[int, list[TaskSpec]] | None] = asyncio.Queue()
    slots = asyncio.Semaphore(config.concurrency)

    async def one(idea: CapabilityIdea) -> None:
        types = {name: count for name, count in cited_types(idea, examples).items() if name in kinds}
        if not types:
            logger.info("%s cites no task of a generator type; it makes no tasks", idea.capability_id)
            return
        capability = f"{CAPABILITY_TAG_PREFIX}{idea.capability_id}"
        for name, count in types.items():
            family = kinds[name]
            try:
                async with slots:
                    pairs = await generator(
                        config.client,
                        typed_idea(idea, examples[name], family.max_examples),
                        config.instances_per_type,
                        family.entry_form,
                    )
            except Exception:
                # A program that GLM fails to write, for example at a cut reply, stops neither the idea's other
                # types nor the loop.
                logger.exception("The generator of %s, %s failed", idea.capability_id, name)
                continue
            # Validity checks can be slow (a calendar's schedule search), so they run off the event loop, which also
            # writes the training feed.
            kept = await asyncio.to_thread(
                lambda pairs=pairs, family=family: [(q, e) for q, e in pairs if family.valid(q, e)]
            )
            logger.info("%s, %s: %d of %d instances are valid", idea.capability_id, name, len(kept), len(pairs))
            tasks = [
                synthesized(family.task(f"{name}.generator-{k}", question, entry, capability), idea)
                for k, (question, entry) in enumerate(kept)
            ]
            made.put_nowait((count, tasks))

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
        async for task in in_turns(made):
            yield task
        await runner
    finally:
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
