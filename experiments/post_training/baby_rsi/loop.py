# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The RSI loop of "Post-training: Boxes/Systems", as streams.

The design document writes the loop as::

    task_prompts = seed_tasks(...)
    while True:
        tasks = generate_tasks(task_prompts, glm_inference, task_filter)
        model = train(model, tasks)
        eval_scores, rollouts = eval_policy(model)
        task_prompts = analyze_failed_runs(rollouts)

Here each box is a function from a stream to a stream, and the data drives the loop: the tasks flow into training,
training yields graded rollouts, the failed rollouts flow into the failed-run analysis, its ideas flow into task
synthesis, and the new tasks flow back into training beside the seed tasks. A box that needs enough data waits for a
batch, and a slow box never stops a fast one.
"""

from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass

from taskcompendium.models import TaskSpec
from taskforge.capability import CapabilityIdea

from experiments.post_training.baby_rsi.feedback import FailedRun, failed_runs
from experiments.post_training.baby_rsi.rollouts import TaskRollouts
from experiments.post_training.baby_rsi.streams import Broadcast, Source, batched, mix


@dataclass(frozen=True)
class Boxes:
    """The boxes of the loop, each with its configuration bound."""

    train: Callable[[AsyncIterable[TaskSpec]], AsyncIterator[TaskRollouts]]
    analyze_failed_runs: Callable[[AsyncIterable[list[FailedRun]]], AsyncIterator[CapabilityIdea]]
    generate_tasks: Callable[[AsyncIterable[CapabilityIdea]], AsyncIterator[TaskSpec]]


@dataclass(frozen=True)
class LoopConfig:
    max_pass_rate: float
    """A task whose pass rate is at most this shows a capability that the policy lacks."""
    failures_per_analysis: int
    synthesized_share: float
    """The share of training tasks that come from synthesis while synthesized tasks are ready."""
    seed: int


def rsi_loop(seed_tasks: AsyncIterable[TaskSpec], boxes: Boxes, config: LoopConfig) -> AsyncIterator[TaskRollouts]:
    """Train on the seed tasks and on the tasks that the loop makes from the policy's own failures."""
    rollouts = Broadcast[TaskRollouts]()
    failures = failed_runs(rollouts.listen(), config.max_pass_rate)
    ideas = boxes.analyze_failed_runs(batched(failures, config.failures_per_analysis))
    new_tasks = boxes.generate_tasks(ideas)
    tasks = mix(
        Source(seed_tasks, share=1 - config.synthesized_share),
        Source(new_tasks, share=config.synthesized_share),
        seed=config.seed,
    )
    return rollouts.publish(boxes.train(tasks))
