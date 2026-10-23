# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command line of the Baby RSI experiments.

- ``skyrl-rsi``: the RSI loop with a SkyRL job as its train box: seed tasks from the pool, and tasks synthesized from
  the policy's own failures. It runs in a privileged CPU task beside the SkyRL job of its feed.
- ``skyrl-control``: the same training on seed tasks only.
- ``baseline`` and ``evaluate`` run in an Iris GPU task through ``job.sh``: the pass rates of a base policy on a
  stratified pool sample, and of the base policy and trained checkpoints on held-out pool and synthesized tasks.
"""

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterable, AsyncIterator, Callable, Collection, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from functools import partial

import click
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from rolloutengine.models import Model
from shellbox.machine import Backend
from taskcompendium.models import TaskSpec
from taskforge.agentic.config import AgenticConfig, ToolBudget
from taskforge.agentic.parallel import ParallelWeb
from taskforge.capability import CapabilityIdea
from taskforge.llm.glm_client import GlmClient, Pool
from taskforge.llm.policy import LLMPolicy, ReasoningEffort
from transformers import AutoTokenizer

from experiments.post_training.capability_driven_envs.generate_tasks import GLM_PROVIDER, endpoint_source
from experiments.post_training.capability_driven_envs.generate_tasks_agentic import (
    COMMAND_TIMEOUT,
    GATE,
    PARALLEL_API_KEY_ENV,
    SOLVER_LIMITS,
    agent_limits,
    chat_model,
    solver_factory,
    workspace_factory,
)
from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_MODEL
from experiments.post_training.rsi.feedback import (
    SYNTHESIZED_LABEL_PREFIX,
    AnalysisConfig,
    SynthesisConfig,
    analyze_failed_runs,
    capability_of,
    generate_tasks,
)
from experiments.post_training.rsi.loop import Boxes, LoopConfig, rsi_loop
from experiments.post_training.rsi.policy import MAX_MODEL_LEN, LivePolicy, Sampling
from experiments.post_training.rsi.resume import (
    known_ideas,
    untrained_instances,
)
from experiments.post_training.rsi.rollouts import TaskRollouts, group_record
from experiments.post_training.rsi.seed import POOL_URL, TRAIN_FILE, TaskPool
from experiments.post_training.rsi.skyrl_box import FeedConfig, skyrl_train, written_tasks
from experiments.post_training.rsi.streams import Source, from_iterable, mix

logger = logging.getLogger(__name__)

SOLVER_ROLLOUTS_PER_TASK = 160
"""Solver rollouts that a synthesis session can spend for each task: the solver is the local policy, so they are
cheap, and a weak policy needs more rounds to bring a task into the gate's band."""
ANALYSIS_POLICY = LLMPolicy(reasoning_effort=ReasoningEffort.LOW, max_tokens=32768)
FEED_POLL_SECONDS = 10.0
UPLOAD_EVERY_GROUPS = 8


@click.group()
def main() -> None:
    configure_logging(logging.INFO)


def feedback_options(function):
    for option in reversed(
        [
            click.option("--relay-job", default=DEFAULT_GLM_RELAY_JOB, show_default=True),
            click.option(
                "--pool", type=click.Choice([pool.value for pool in Pool]), default=Pool.BULK.value, show_default=True
            ),
            click.option("--max-pass-rate", type=float, default=0.25, show_default=True),
            click.option("--failures-per-analysis", type=int, default=64, show_default=True),
            click.option("--max-new-ideas", type=int, default=8, show_default=True),
            click.option("--synthesized-share", type=float, default=0.5, show_default=True),
            click.option("--tasks-per-idea", type=int, default=2, show_default=True),
            click.option("--instances-per-task", type=int, default=32, show_default=True),
            click.option("--probes-per-task", type=int, default=8, show_default=True),
            click.option("--sessions", type=int, default=6, show_default=True, help="Agent sessions at a time."),
            click.option("--agent-turns", type=int, default=300, show_default=True),
            click.option(
                "--held-out-idea-share",
                type=click.FloatRange(0, 1),
                default=0.0,
                show_default=True,
                help=(
                    "Ideas whose tasks are only probes, to measure transfer to families that the policy never "
                    "trains on."
                ),
            ),
        ]
    ):
        function = option(function)
    return function


def _pop_feedback(arguments: dict) -> "FeedbackOptions":
    feedback = {name: arguments.pop(name) for name in FeedbackOptions.__dataclass_fields__}
    return FeedbackOptions(**{**feedback, "pool": Pool(feedback["pool"])})


@dataclass(frozen=True)
class FeedbackOptions:
    relay_job: str
    pool: Pool
    max_pass_rate: float
    failures_per_analysis: int
    max_new_ideas: int
    synthesized_share: float
    tasks_per_idea: int
    instances_per_task: int
    probes_per_task: int
    sessions: int
    agent_turns: int
    held_out_idea_share: float


def _remember_stratum(stream, strata: dict[str, str], *, written: Collection[str] = ()):
    """The tasks of ``stream`` that are not in ``written``; records the stratum of every task, also a written one."""
    for task in stream:
        strata[task.task.id] = task.stratum
        if task.task.id not in written:
            yield task.task


async def _feedback_boxes(
    stack: AsyncExitStack,
    feedback: FeedbackOptions,
    solver: Model,
    trainer: Callable[[AsyncIterable[TaskSpec]], AsyncIterator[TaskRollouts]],
    output: StoragePath,
    trained_ids: Collection[str],
) -> Boxes:
    source = endpoint_source(None, feedback.relay_job, feedback.pool)
    judge = await stack.enter_async_context(GlmClient(source, GLM_PROVIDER))
    web = await stack.enter_async_context(ParallelWeb(os.environ[PARALLEL_API_KEY_ENV]))
    endpoint = source.current()
    agentic = AgenticConfig(
        agent_model=chat_model(endpoint, model=GLM_MODEL, reasoning_effort="low", cache_key=None),
        workspace_factory=workspace_factory(),
        solver_model=solver,
        solver_factory=solver_factory(Backend.LOCAL),
        judge=judge,
        judge_policy=LLMPolicy(),
        web=web,
        agent_limits=agent_limits(feedback.agent_turns),
        solver_limits=SOLVER_LIMITS,
        gate=GATE,
        budget=ToolBudget(
            solver_rollouts=SOLVER_ROLLOUTS_PER_TASK * feedback.tasks_per_idea,
            judge_calls=6 * feedback.tasks_per_idea + 2,
            web_calls=10 + 2 * feedback.tasks_per_idea,
        ),
        command_timeout=COMMAND_TIMEOUT,
        max_concurrent_rollouts=8,
    )
    analysis = AnalysisConfig(
        make_client=partial(GlmClient, source, GLM_PROVIDER),
        max_new_ideas=feedback.max_new_ideas,
        shard_size=8,
        extract_policy=ANALYSIS_POLICY,
        merge_policy=ANALYSIS_POLICY,
        work_dir="/tmp/baby-rsi-zephyr",
        workers=16,
    )
    synthesis = SynthesisConfig(
        agentic=agentic,
        tasks_per_idea=feedback.tasks_per_idea,
        instances_per_task=feedback.instances_per_task,
        probes_per_task=feedback.probes_per_task,
        concurrency=feedback.sessions,
        output=output,
        held_out_idea_share=feedback.held_out_idea_share,
    )
    resumed = untrained_instances(output, trained_ids, feedback.instances_per_task, agentic.workspace_factory)

    def generate(ideas: AsyncIterable[CapabilityIdea]) -> AsyncIterator[TaskSpec]:
        # A resumed run first gives back the instances of the tasks that it accepted and did not train on.
        new = generate_tasks(ideas, synthesis)
        return mix(Source(resumed, share=1.0), Source(new, share=1.0), seed=0)

    return Boxes(
        train=trainer,
        analyze_failed_runs=partial(analyze_failed_runs, config=analysis, known=known_ideas(output)),
        generate_tasks=generate,
    )


def skyrl_options(function):
    for option in reversed(
        [
            click.option("--output-dir", required=True),
            click.option("--feed-root", required=True, help="The data.feed.root of the SkyRL job."),
            click.option("--model", required=True, help="The Hugging Face ID of the policy's tokenizer."),
            click.option("--thinking/--no-thinking", default=False, show_default=True, help="Qwen3 thinking mode."),
            click.option("--max-tokens", type=int, default=4096, show_default=True, help="Tokens of one gate turn."),
            click.option("--max-model-len", type=int, default=MAX_MODEL_LEN, show_default=True),
            click.option("--steps", type=int, required=True, help="The training steps of the SkyRL job."),
            click.option("--tasks-per-step", type=int, required=True, help="The prompts of one SkyRL step."),
            click.option("--lookahead-steps", type=int, default=3, show_default=True, help="Feed tasks ahead."),
            click.option("--seed", type=int, default=0, show_default=True),
        ]
    ):
        function = option(function)
    return function


@dataclass(frozen=True)
class SkyrlOptions:
    output_dir: str
    feed_root: str
    model: str
    thinking: bool
    max_tokens: int
    max_model_len: int
    steps: int
    tasks_per_step: int
    lookahead_steps: int
    seed: int

    def feed(self) -> FeedConfig:
        return FeedConfig(
            root=StoragePath(self.feed_root),
            lookahead=self.lookahead_steps * self.tasks_per_step,
            chunk_rows=self.tasks_per_step,
            poll_seconds=FEED_POLL_SECONDS,
            groups=self.steps * self.tasks_per_step,
        )


@main.command("skyrl-control")
@skyrl_options
def skyrl_control(**arguments) -> None:
    """Feed seed tasks only to a SkyRL job: the control of the RSI loop."""
    asyncio.run(_skyrl_run(SkyrlOptions(**arguments), feedback=None))


@main.command("skyrl-rsi")
@skyrl_options
@feedback_options
def skyrl_rsi(**arguments) -> None:
    """Run the RSI loop with a SkyRL job as its train box."""
    feedback = _pop_feedback(arguments)
    asyncio.run(_skyrl_run(SkyrlOptions(**arguments), feedback=feedback))


async def _skyrl_run(options: SkyrlOptions, *, feedback: FeedbackOptions | None) -> None:
    output = StoragePath(options.output_dir)
    feed = options.feed()
    written = {task.id for task in (await asyncio.to_thread(written_tasks, feed.root)).values()}
    pool = TaskPool.read(f"{POOL_URL}/{TRAIN_FILE}")
    strata: dict[str, str] = {}
    # A restarted loop does not write the pool tasks of its feed again, but labels their groups with their stratum.
    seed_tasks = from_iterable(_remember_stratum(pool.stream(options.seed), strata, written=written))
    trainer = partial(skyrl_train, config=feed)
    async with AsyncExitStack() as stack:
        if feedback is None:
            stream = trainer(seed_tasks)
        else:
            sampling = Sampling(temperature=1.0, top_p=1.0, max_tokens=options.max_tokens)
            chat_template = {"enable_thinking": options.thinking}
            tokenizer = AutoTokenizer.from_pretrained(options.model)
            solver = LivePolicy(feed.root, tokenizer, sampling, options.max_model_len, chat_template)
            boxes = await _feedback_boxes(stack, feedback, solver, trainer, output, written)
            config = LoopConfig(
                max_pass_rate=feedback.max_pass_rate,
                failures_per_analysis=feedback.failures_per_analysis,
                synthesized_share=feedback.synthesized_share,
                seed=options.seed,
            )
            stream = rsi_loop(seed_tasks, boxes, config)
        await _record_groups(stream, strata, output)


async def _record_groups(stream: AsyncIterator[TaskRollouts], strata: Mapping[str, str], output: StoragePath) -> None:
    """Append a record of each group to ``groups.jsonl``, after those of a restarted loop, and upload it often."""
    path = output / "groups.jsonl"
    lines = path.read_text().splitlines(keepends=True) if path.exists() else []
    async for group in stream:
        label = strata.get(group.task.id) or f"{SYNTHESIZED_LABEL_PREFIX}{capability_of(group.task)}"
        record = {**group_record(group, transcripts=False), "label": label}
        record["run_ids"] = [graded.rollout.run_id for graded in group.rollouts]
        lines.append(json.dumps(record, default=str) + "\n")
        if len(lines) % UPLOAD_EVERY_GROUPS == 0:
            await asyncio.to_thread(path.write_text, "".join(lines))
    await asyncio.to_thread(path.write_text, "".join(lines))


if __name__ == "__main__":
    main()
