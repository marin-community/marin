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
from collections import Counter
from collections.abc import AsyncIterable, AsyncIterator, Callable, Collection, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import click
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from rolloutengine.models import Model, RolloutLimits
from shellbox.machine import Backend
from taskcompendium.models import TaskSpec
from taskforge.agentic.config import AgenticConfig, ToolBudget
from taskforge.agentic.instances import task_instances
from taskforge.agentic.parallel import ParallelWeb
from taskforge.capability import CapabilityIdea
from taskforge.llm.glm_client import GlmClient, Pool
from taskforge.llm.policy import LLMPolicy, ReasoningEffort
from transformers import AutoTokenizer

from experiments.post_training.baby_rsi.evaluate import Evaluation, eval_policy, group_record, write_jsonl
from experiments.post_training.baby_rsi.feedback import (
    INSTANCE_TIMEOUT,
    SYNTHESIZED_LABEL_PREFIX,
    AnalysisConfig,
    SynthesisConfig,
    analyze_failed_runs,
    capability_of,
    generate_tasks,
    held_out,
    training_ids,
)
from experiments.post_training.baby_rsi.loop import Boxes, LoopConfig, rsi_loop
from experiments.post_training.baby_rsi.policy import BASE_MODEL_NAME, Sampling, VllmPolicy, VllmServer
from experiments.post_training.baby_rsi.resume import (
    accepted_tasks,
    known_ideas,
    untrained_instances,
)
from experiments.post_training.baby_rsi.rollouts import PolicyRunner, TaskRollouts, rollout_engine
from experiments.post_training.baby_rsi.seed import POOL_URL, TRAIN_FILE, VALIDATION_FILE, TaskPool
from experiments.post_training.baby_rsi.skyrl_box import FeedConfig, LivePolicy, skyrl_train, written_tasks
from experiments.post_training.baby_rsi.streams import Source, from_iterable, mix
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

logger = logging.getLogger(__name__)

VLLM_ENV = "BABY_RSI_VLLM"
CHECKPOINT_DIR = "/tmp/baby-rsi-checkpoints"
MAX_MODEL_LEN = 16384
MAX_LORA_RANK = 64
POLICY_LIMITS = RolloutLimits(
    max_turns=20,
    max_total_tokens=None,
    model_turn_timeout=600,
    tool_turn_timeout=120,
    total_turn_timeout=None,
    attempt_timeout=1800,
    verifier_timeout=360,
    cleanup_timeout=60,
)
REPEAT_NAME = "base-repeat"
SOLVER_ROLLOUTS_PER_TASK = 160
"""Solver rollouts that a synthesis session can spend for each task: the solver is the local policy, so they are
cheap, and a weak policy needs more rounds to bring a task into the gate's band."""
ANALYSIS_POLICY = LLMPolicy(reasoning_effort=ReasoningEffort.LOW, max_tokens=32768)
FEED_POLL_SECONDS = 10.0
UPLOAD_EVERY_GROUPS = 8


@dataclass(frozen=True)
class PolicyOptions:
    model: str
    thinking: bool
    max_tokens: int
    gpu_memory_utilization: float
    concurrency: int
    grader_slots: int
    vllm_args: tuple[str, ...]


def policy_options(function):
    for option in reversed(
        [
            click.option("--model", required=True, help="The Hugging Face ID of the policy."),
            click.option("--thinking/--no-thinking", default=False, show_default=True, help="Qwen3 thinking mode."),
            click.option("--max-tokens", type=int, default=4096, show_default=True, help="Tokens of one turn."),
            click.option("--gpu-memory-utilization", type=float, default=0.85, show_default=True),
            click.option("--concurrency", type=int, default=128, show_default=True, help="Attempts at a time."),
            click.option("--grader-slots", type=int, default=16, show_default=True, help="Script graders at a time."),
            click.option("--vllm-arg", "vllm_args", multiple=True, help="An argument of vllm serve; repeatable."),
        ]
    ):
        function = option(function)
    return function


@click.group()
def main() -> None:
    configure_logging(logging.INFO)


class Served:
    """The vLLM server and the policy runner of one task: the policy and its rollouts."""

    def __init__(self, options: PolicyOptions, server: VllmServer):
        self.server = server
        self.policy = VllmPolicy.for_model(
            server.base_url,
            options.model,
            Sampling(temperature=1.0, top_p=1.0, max_tokens=options.max_tokens),
            MAX_MODEL_LEN,
            chat_template_kwargs={"enable_thinking": options.thinking},
        )
        self.runner = PolicyRunner(rollout_engine(options.grader_slots), self.policy, POLICY_LIMITS, options.concurrency)


def serve(options: PolicyOptions, model: str | None = None):
    """A vLLM server of ``options.model``, or of the weights at ``model`` with the same tokenizer."""
    return VllmServer.start(
        model or options.model,
        executable=os.environ[VLLM_ENV],
        max_model_len=MAX_MODEL_LEN,
        max_lora_rank=MAX_LORA_RANK,
        gpu_memory_utilization=options.gpu_memory_utilization,
        extra_args=options.vllm_args,
    )


@main.command()
@policy_options
@click.option("--output-dir", required=True)
@click.option(
    "--pool-file", type=click.Choice([TRAIN_FILE, VALIDATION_FILE]), default=VALIDATION_FILE, show_default=True
)
@click.option("--tasks", type=int, default=200, show_default=True, help="Pool tasks, with the pool's strata weights.")
@click.option("--samples", type=int, default=4, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def baseline(output_dir: str, pool_file: str, tasks: int, samples: int, seed: int, **options) -> None:
    """Pass rates of a base policy on a stratified sample of a pool file."""
    asyncio.run(_baseline(PolicyOptions(**options), StoragePath(output_dir), pool_file, tasks, samples, seed))


async def _baseline(
    options: PolicyOptions, output: StoragePath, pool_file: str, tasks: int, samples: int, seed: int
) -> None:
    pool = TaskPool.read(f"{POOL_URL}/{pool_file}")
    logger.info("Pool strata weights: %s", pool.weights)
    sample = pool.sample(tasks, seed)
    logger.info("Sampled strata: %s", Counter(task.stratum for task in sample))
    with serve(options) as server:
        served = Served(options, server)
        labels = {task.task.id: task.stratum for task in sample}
        evaluation = await eval_policy(served.runner, [task.task for task in sample], labels, samples)
    write_evaluation(output, options.model, {BASE_MODEL_NAME: evaluation})


def write_evaluation(output: StoragePath, model: str, evaluations: Mapping[str, Evaluation]) -> None:
    summary = {"model": model, **{name: evaluation.summary() for name, evaluation in evaluations.items()}}
    logger.info("Summary: %s", json.dumps(summary, indent=1))
    (output / "summary.json").write_text(json.dumps(summary, indent=1))
    for name, evaluation in evaluations.items():
        records = (
            {**group_record(group, transcripts=True), "label": evaluation.labels.get(group.task.id)}
            for group in evaluation.groups
        )
        write_jsonl(output / f"groups-{name}.jsonl", records)


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


def _remember_stratum(stream, strata: dict[str, str]):
    for task in stream:
        strata[task.task.id] = task.stratum
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
    # A restarted loop does not write the pool tasks of its feed again.
    written = {task.id for task in (await asyncio.to_thread(written_tasks, feed.root)).values()}
    pool = TaskPool.read(f"{POOL_URL}/{TRAIN_FILE}")
    strata: dict[str, str] = {}
    seed_tasks = from_iterable(_remember_stratum(pool.stream(options.seed, exclude=frozenset(written)), strata))
    trainer = partial(skyrl_train, config=feed)
    async with AsyncExitStack() as stack:
        if feedback is None:
            stream = trainer(seed_tasks)
        else:
            sampling = Sampling(temperature=1.0, top_p=1.0, max_tokens=options.max_tokens)
            chat_template = {"enable_thinking": options.thinking}
            tokenizer = AutoTokenizer.from_pretrained(options.model)
            solver = LivePolicy(feed.root, tokenizer, sampling, MAX_MODEL_LEN, chat_template)
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


@main.command()
@policy_options
@click.option("--output-dir", required=True)
@click.option("--checkpoint", "checkpoints", multiple=True, help="NAME=URL of a trained HF checkpoint; repeatable.")
@click.option(
    "--base-runs",
    type=click.IntRange(0, 2),
    default=2,
    show_default=True,
    help="Evaluations of the base policy; 0 lets parallel jobs of other policies write to the same directory.",
)
@click.option("--probes", "probe_roots", multiple=True, help="An rsi output directory with held-out probes.")
@click.option("--instances-per-task", type=int, default=32, show_default=True, help="As the rsi run made them.")
@click.option("--pool-tasks", type=int, default=400, show_default=True, help="Held-out validation pool tasks.")
@click.option("--samples", type=int, default=8, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def evaluate(
    output_dir: str,
    checkpoints: tuple[str, ...],
    base_runs: int,
    probe_roots: tuple[str, ...],
    instances_per_task: int,
    pool_tasks: int,
    samples: int,
    seed: int,
    **options,
) -> None:
    """Evaluate the base policy and trained checkpoints on the same held-out tasks."""
    asyncio.run(
        _evaluate(
            PolicyOptions(**options),
            StoragePath(output_dir),
            dict(checkpoint.split("=", 1) for checkpoint in checkpoints),
            base_runs,
            probe_roots,
            instances_per_task,
            pool_tasks,
            samples,
            seed,
        )
    )


async def _evaluate(
    options: PolicyOptions,
    output: StoragePath,
    checkpoints: Mapping[str, str],
    base_runs: int,
    probe_roots: Sequence[str],
    instances_per_task: int,
    pool_tasks: int,
    samples: int,
    seed: int,
) -> None:
    validation = TaskPool.read(f"{POOL_URL}/{VALIDATION_FILE}").sample(pool_tasks, seed)
    tasks = [task.task for task in validation]
    labels = {task.task.id: f"pool:{task.stratum}" for task in validation}
    for probe in await read_probes(probe_roots, instances_per_task):
        tasks.append(probe)
        labels[probe.id] = f"probe:{capability_of(probe)}"
    logger.info("Evaluation tasks: %s", Counter(label.split(":")[0] for label in labels.values()))
    results = {}
    # A second evaluation of the base weights measures how often a task that the base never passes passes by chance
    # on new samples. One policy at a time, so that the graders of all policies see the same load.
    base_names = (BASE_MODEL_NAME, REPEAT_NAME)[:base_runs]
    if base_names:
        with serve(options) as server:
            runner = Served(options, server).runner
            for name in base_names:
                results[name] = await eval_policy(runner, tasks, labels, samples)
    for name, url in checkpoints.items():
        local = Path(CHECKPOINT_DIR) / name
        await asyncio.to_thread(_download_tree, url, local)
        with serve(options, str(local)) as server:
            results[name] = await eval_policy(Served(options, server).runner, tasks, labels, samples)
    write_evaluation(output, options.model, results)


async def read_probes(roots: Sequence[str], instances_per_task: int) -> list[TaskSpec]:
    """The held-out probes of the accepted tasks in ``roots``; a probe equal to a training instance is dropped."""
    factory = workspace_factory()
    probes = []
    for root in roots:
        for saved in accepted_tasks(StoragePath(root)):
            ids = training_ids(saved.accepted, instances_per_task)
            training = await task_instances(saved.accepted, ids, factory, timeout=INSTANCE_TIMEOUT)
            lines = (saved.directory / "probes.jsonl").read_text().splitlines()
            stored = [TaskSpec.model_validate_json(line) for line in lines if line]
            kept = held_out(stored, saved.accepted, training)
            logger.info("%s: %d of %d probes are held out", saved.accepted.task.id, len(kept), len(stored))
            probes += kept
    return probes


def _download_tree(url: str, local: Path) -> None:
    fs, path = url_to_fs(url)
    fs.get(path.rstrip("/") + "/", str(local), recursive=True)


if __name__ == "__main__":
    main()
