# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command line of the Baby RSI experiments.

- ``skyrl-rsi``: the RSI loop with a SkyRL job as its train box: seed tasks from the pool, and tasks synthesized from
  the policy's own failures. It runs in a privileged CPU task beside the SkyRL job of its feed.
- ``skyrl-control``: the same training on seed tasks only.

The ``baseline`` and ``evaluate`` commands live in ``evaluate.py`` and run in an Iris GPU task through ``job.sh``.
"""

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterable, AsyncIterator, Callable, Collection, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
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
    FailedRun,
    SynthesisConfig,
    analyze_failed_runs,
    capability_of,
    generate_tasks,
)
from experiments.post_training.rsi.gap import (
    DIAGNOSTIC_CAPABILITY_PREFIX,
    GAP_CAPABILITY_PREFIX,
    diagnostic_indices,
    fresh_tasks,
    interleave,
    scan_types,
    training_indices,
    without_generators,
)
from experiments.post_training.rsi.live_eval import EvalSet, eval_policy, evaluator, policy_step
from experiments.post_training.rsi.loop import Boxes, LoopConfig, rsi_loop
from experiments.post_training.rsi.policy import MAX_MODEL_LEN, LivePolicy, Sampling
from experiments.post_training.rsi.resume import (
    known_ideas,
    untrained_instances,
)
from experiments.post_training.rsi.rollouts import (
    POLICY_LIMITS,
    PolicyRunner,
    TaskRollouts,
    group_record,
    read_tasks,
    rollout_engine,
)
from experiments.post_training.rsi.seed import TRAIN_URL, VALIDATION_URL, TaskPool
from experiments.post_training.rsi.skyrl_box import FeedConfig, skyrl_train, written_tasks
from experiments.post_training.rsi.streams import Source, from_iterable, mix

logger = logging.getLogger(__name__)

EVAL_CONCURRENCY = 64
EVAL_GRADER_SLOTS = 16
EVAL_GAP_INSTANCES = 20

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
            click.option(
                "--synthesis-max-wait",
                type=float,
                default=0.0,
                show_default=True,
                help="Seconds that training waits for synthesized tasks to keep their share.",
            ),
            click.option("--tasks-per-idea", type=int, default=2, show_default=True),
            click.option("--instances-per-task", type=int, default=32, show_default=True),
            click.option("--probes-per-task", type=int, default=8, show_default=True),
            click.option("--sessions", type=int, default=6, show_default=True, help="Agent sessions at a time."),
            click.option("--agent-turns", type=int, default=300, show_default=True),
            click.option(
                "--agent-reasoning-effort",
                type=click.Choice(["low", "medium", "high"]),
                default="low",
                show_default=True,
                help="The reasoning effort of the Taskforge agent's model.",
            ),
            click.option(
                "--solver-band",
                type=(float, float),
                default=GATE.solver_band,
                show_default=True,
                help="The lowest and the highest pass rate of the live policy that the gate accepts.",
            ),
            click.option(
                "--ideas-file",
                help="A JSON list of capability ideas that replace the failed-run analysis, to test synthesis alone.",
            ),
            click.option("--eval-every", type=int, default=10, show_default=True, help="Steps between evaluations."),
            click.option("--eval-samples", type=int, default=4, show_default=True, help="Attempts at each task."),
            click.option(
                "--eval-gap-types",
                default="",
                help="Comma-separated Reasoning Gym types whose unseen instances join the evaluation set.",
            ),
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
    synthesis_max_wait: float
    tasks_per_idea: int
    instances_per_task: int
    probes_per_task: int
    sessions: int
    agent_turns: int
    agent_reasoning_effort: str
    solver_band: tuple[float, float]
    ideas_file: str | None
    eval_every: int
    eval_samples: int
    eval_gap_types: str
    held_out_idea_share: float


def _remember_stratum(stream, strata: dict[str, str], *, written: Collection[str] = ()):
    """The tasks of ``stream`` that are not in ``written``; records the stratum of every task, also a written one."""
    for task in stream:
        strata[task.task.id] = task.stratum
        if task.task.id not in written:
            yield task.task


async def synthesis_config(
    stack: AsyncExitStack, feedback: FeedbackOptions, solver: Model, output: StoragePath
) -> SynthesisConfig:
    """The generate box's configuration: agentic Taskforge with ``solver`` as the solver of its gate."""
    source = endpoint_source(None, feedback.relay_job, feedback.pool)
    judge = await stack.enter_async_context(GlmClient(source, GLM_PROVIDER))
    web = await stack.enter_async_context(ParallelWeb(os.environ[PARALLEL_API_KEY_ENV]))
    endpoint = source.current()
    agentic = AgenticConfig(
        agent_model=chat_model(
            endpoint, model=GLM_MODEL, reasoning_effort=feedback.agent_reasoning_effort, cache_key=None
        ),
        workspace_factory=workspace_factory(),
        solver_model=solver,
        solver_factory=solver_factory(Backend.LOCAL),
        judge=judge,
        judge_policy=LLMPolicy(),
        web=web,
        agent_limits=agent_limits(feedback.agent_turns),
        solver_limits=SOLVER_LIMITS,
        gate=replace(GATE, solver_band=feedback.solver_band),
        budget=ToolBudget(
            solver_rollouts=SOLVER_ROLLOUTS_PER_TASK * feedback.tasks_per_idea,
            judge_calls=6 * feedback.tasks_per_idea + 2,
            web_calls=10 + 2 * feedback.tasks_per_idea,
        ),
        command_timeout=COMMAND_TIMEOUT,
        max_concurrent_rollouts=8,
    )
    return SynthesisConfig(
        agentic=agentic,
        tasks_per_idea=feedback.tasks_per_idea,
        instances_per_task=feedback.instances_per_task,
        probes_per_task=feedback.probes_per_task,
        concurrency=feedback.sessions,
        output=output,
        held_out_idea_share=feedback.held_out_idea_share,
    )


async def _feedback_boxes(
    stack: AsyncExitStack,
    feedback: FeedbackOptions,
    solver: Model,
    trainer: Callable[[AsyncIterable[TaskSpec]], AsyncIterator[TaskRollouts]],
    output: StoragePath,
    trained_ids: Collection[str],
) -> Boxes:
    source = endpoint_source(None, feedback.relay_job, feedback.pool)
    synthesis = await synthesis_config(stack, feedback, solver, output)
    analysis = AnalysisConfig(
        make_client=partial(GlmClient, source, GLM_PROVIDER),
        max_new_ideas=feedback.max_new_ideas,
        shard_size=8,
        extract_policy=ANALYSIS_POLICY,
        merge_policy=ANALYSIS_POLICY,
        work_dir="/tmp/baby-rsi-zephyr",
        workers=16,
    )
    resumed = untrained_instances(output, trained_ids, feedback.instances_per_task, synthesis.agentic.workspace_factory)

    def generate(ideas: AsyncIterable[CapabilityIdea]) -> AsyncIterator[TaskSpec]:
        # A resumed run first gives back the instances of the tasks that it accepted and did not train on.
        new = generate_tasks(ideas, synthesis)
        return mix(Source(resumed, share=1.0), Source(new, share=1.0), seed=0)

    eval_set = await asyncio.to_thread(evaluation_set, feedback.eval_gap_types)
    runner = PolicyRunner(rollout_engine(EVAL_GRADER_SLOTS), solver, POLICY_LIMITS, EVAL_CONCURRENCY)
    evaluate = evaluator(runner, eval_set, feedback.eval_samples, output)
    analyze = partial(analyze_failed_runs, config=analysis, known=known_ideas(output))
    if feedback.ideas_file is not None:
        analyze = partial(fixed_ideas, ideas=read_ideas(feedback.ideas_file), known=known_ideas(output))
    return Boxes(
        train=trainer,
        eval_policy=partial(eval_policy, evaluate=evaluate, every=feedback.eval_every, step_of=policy_step),
        analyze_failed_runs=analyze,
        generate_tasks=generate,
    )


def read_ideas(url: str) -> list[CapabilityIdea]:
    return [CapabilityIdea.model_validate(idea) for idea in json.loads(StoragePath(url).read_text())]


async def fixed_ideas(
    batches: AsyncIterable[list[FailedRun]], *, ideas: list[CapabilityIdea], known: Collection[str]
) -> AsyncIterator[CapabilityIdea]:
    """The given ideas, at once and without any failed run, except the ideas that a resumed run already had."""
    for idea in ideas:
        if idea.capability_id not in known:
            yield idea


def evaluation_set(gap_types: str) -> EvalSet:
    """The validation panel of the seed's selection, and unseen instances of ``gap_types`` when there are any."""
    panel = [task for task in TaskPool.read(VALIDATION_URL).stream(0)]
    tasks = [task.task for task in panel]
    labels = {task.task.id: f"panel:{task.stratum}" for task in panel}
    if gap_types:
        names = gap_types.split(",")
        types = scan_types()
        indices = {name: diagnostic_indices(types[name], EVAL_GAP_INSTANCES) for name in names}
        for name, fresh in fresh_tasks(types, indices, DIAGNOSTIC_CAPABILITY_PREFIX).items():
            # Without their tags, which name the type: the failed-run analysis reads a task's tags.
            untagged = [task.model_copy(update={"tags": ()}) for task in fresh]
            tasks += untagged
            labels.update({task.id: f"gap:{name}" for task in untagged})
    logger.info("The evaluation box evaluates %d tasks", len(tasks))
    return EvalSet(tasks, labels)


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
            click.option(
                "--exclude-generators",
                default="",
                help="Comma-separated Reasoning Gym task types dropped from the seed.",
            ),
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
    exclude_generators: str

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


@main.command("skyrl-oracle")
@skyrl_options
@click.option("--oracle-types", default="", help="Comma-separated Reasoning Gym types whose fresh instances train.")
@click.option("--oracle-tasks", type=int, default=1200, show_default=True, help="Fresh tasks made, shared by the types.")
@click.option("--tasks-file", help="A JSONL file of synthesized tasks (presynthesize.py) that train instead.")
@click.option("--synthesized-share", type=float, default=0.5, show_default=True)
def skyrl_oracle(
    oracle_types: str, oracle_tasks: int, tasks_file: str | None, synthesized_share: float, **arguments
) -> None:
    """Feed seed tasks and tasks that are ready from the start: fresh instances of chosen Reasoning Gym types (an
    oracle synthesizer, or a placebo for another type), or a file of tasks that synthesis made before training."""
    if bool(oracle_types) == (tasks_file is not None):
        raise click.UsageError("give exactly one of --oracle-types and --tasks-file")
    types = oracle_types.split(",") if oracle_types else []
    oracle = OracleOptions(types, oracle_tasks, tasks_file, synthesized_share)
    asyncio.run(_skyrl_run(SkyrlOptions(**arguments), feedback=None, oracle=oracle))


@main.command("skyrl-rsi")
@skyrl_options
@feedback_options
def skyrl_rsi(**arguments) -> None:
    """Run the RSI loop with a SkyRL job as its train box."""
    feedback = _pop_feedback(arguments)
    asyncio.run(_skyrl_run(SkyrlOptions(**arguments), feedback=feedback))


@dataclass(frozen=True)
class OracleOptions:
    types: list[str]
    tasks: int
    tasks_file: str | None
    share: float


async def _skyrl_run(
    options: SkyrlOptions, *, feedback: FeedbackOptions | None, oracle: OracleOptions | None = None
) -> None:
    output = StoragePath(options.output_dir)
    feed = options.feed()
    written = {task.id for task in (await asyncio.to_thread(written_tasks, feed.root)).values()}
    pool = TaskPool.read(TRAIN_URL)
    if options.exclude_generators:
        pool = without_generators(pool, frozenset(options.exclude_generators.split(",")))
    strata: dict[str, str] = {}
    # A restarted loop does not write the pool tasks of its feed again, but labels their groups with their stratum.
    seed_tasks = from_iterable(_remember_stratum(pool.stream(options.seed), strata, written=written))
    trainer = partial(skyrl_train, config=feed)
    async with AsyncExitStack() as stack:
        if oracle is not None:
            stream = trainer(await _with_oracle_tasks(seed_tasks, oracle, written, options.seed))
        elif feedback is None:
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
                synthesis_max_wait=feedback.synthesis_max_wait,
                seed=options.seed,
            )
            stream = rsi_loop(seed_tasks, boxes, config)
        await _record_groups(stream, strata, output)


async def _with_oracle_tasks(
    seed_tasks: AsyncIterable[TaskSpec], oracle: OracleOptions, written: Collection[str], seed: int
) -> AsyncIterator[TaskSpec]:
    """The seed tasks mixed with tasks that are ready from the start, the capabilities of the tasks in turn."""
    if oracle.tasks_file is not None:
        by_capability: dict[str, list[TaskSpec]] = {}
        for task in await asyncio.to_thread(read_tasks, oracle.tasks_file):
            by_capability.setdefault(capability_of(task) or "", []).append(task)
        ready = by_capability
    else:
        types = await asyncio.to_thread(scan_types)
        per_type = -(-oracle.tasks // len(oracle.types))
        indices = {name: training_indices(types[name])[:per_type] for name in oracle.types}
        ready = await asyncio.to_thread(fresh_tasks, types, indices, GAP_CAPABILITY_PREFIX)
    new_tasks = from_iterable(task for task in interleave(ready) if task.id not in written)
    return mix(Source(seed_tasks, share=1 - oracle.share), Source(new_tasks, share=oracle.share), seed=seed)


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
