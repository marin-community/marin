# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command line of the Baby RSI experiments. Run it inside an Iris GPU task through ``job.sh``.

- ``baseline``: pass rates of a base policy on a stratified pool sample.
- ``rsi``: train with the RSI loop: seed tasks from the pool, and tasks synthesized from the policy's own failures.
- ``control``: the same training on seed tasks only.
- ``evaluate``: the base policy and trained adapters on the held-out pool tasks and on held-out synthesized tasks.
"""

import asyncio
import json
import logging
import os
from collections import Counter
from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from functools import partial
from itertools import islice
from pathlib import Path

import click
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from rolloutengine.models import RolloutLimits
from taskcompendium.models import TaskSpec
from taskforge.agentic.config import AgenticConfig, ToolBudget
from taskforge.agentic.instances import task_instances
from taskforge.agentic.parallel import ParallelWeb
from taskforge.capability import CapabilityIdea
from taskforge.llm.glm_client import GlmClient, Pool
from taskforge.llm.policy import LLMPolicy, ReasoningEffort

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
    OPTIMIZER_FILE,
    Progress,
    accepted_tasks,
    adapter_step,
    known_ideas,
    read_progress,
    untrained_instances,
)
from experiments.post_training.baby_rsi.rollouts import PolicyRunner, TaskRollouts, rollout_engine
from experiments.post_training.baby_rsi.seed import POOL_URL, TRAIN_FILE, VALIDATION_FILE, TaskPool
from experiments.post_training.baby_rsi.streams import Source, from_iterable, mix
from experiments.post_training.baby_rsi.trainer import LoraLearner, TrainerConfig, train
from experiments.post_training.capability_driven_envs.generate_tasks import GLM_PROVIDER, endpoint_source
from experiments.post_training.capability_driven_envs.generate_tasks_agentic import (
    COMMAND_TIMEOUT,
    GATE,
    PARALLEL_API_KEY_ENV,
    SOLVER_LIMITS,
    agent_limits,
    chat_model,
    workspace_factory,
)
from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_MODEL

logger = logging.getLogger(__name__)

VLLM_ENV = "BABY_RSI_VLLM"
ADAPTER_DIR = "/tmp/baby-rsi-adapters"
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


@dataclass(frozen=True)
class PolicyOptions:
    model: str
    thinking: bool
    max_tokens: int
    gpu_memory_utilization: float
    concurrency: int
    grader_slots: int


def policy_options(function):
    for option in reversed(
        [
            click.option("--model", required=True, help="The Hugging Face ID of the policy."),
            click.option("--thinking/--no-thinking", default=False, show_default=True, help="Qwen3 thinking mode."),
            click.option("--max-tokens", type=int, default=4096, show_default=True, help="Tokens of one turn."),
            click.option("--gpu-memory-utilization", type=float, default=0.85, show_default=True),
            click.option("--concurrency", type=int, default=128, show_default=True, help="Attempts at a time."),
            click.option("--grader-slots", type=int, default=16, show_default=True, help="Script graders at a time."),
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


def serve(options: PolicyOptions):
    return VllmServer.start(
        options.model,
        executable=os.environ[VLLM_ENV],
        max_model_len=MAX_MODEL_LEN,
        max_lora_rank=MAX_LORA_RANK,
        gpu_memory_utilization=options.gpu_memory_utilization,
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


def training_options(function):
    for option in reversed(
        [
            click.option("--output-dir", required=True),
            click.option("--steps", type=int, default=150, show_default=True),
            click.option("--tasks-per-step", type=int, default=32, show_default=True),
            click.option("--samples-per-task", type=int, default=8, show_default=True),
            click.option("--learning-rate", type=float, default=4e-5, show_default=True),
            click.option("--lora-rank", type=int, default=32, show_default=True),
            click.option("--micro-batch-tokens", type=int, default=32768, show_default=True),
            click.option("--seed", type=int, default=0, show_default=True),
            click.option("--adapter-every", type=int, default=10, show_default=True, help="Upload every N steps."),
        ]
    ):
        function = option(function)
    return function


@dataclass(frozen=True)
class TrainingOptions:
    output_dir: str
    steps: int
    tasks_per_step: int
    samples_per_task: int
    learning_rate: float
    lora_rank: int
    micro_batch_tokens: int
    seed: int
    adapter_every: int


@main.command()
@policy_options
@training_options
def control(**arguments) -> None:
    """Train on seed tasks only: the control of the RSI loop."""
    options, training = _split(arguments)
    asyncio.run(_train_run(options, training, feedback=None))


@main.command()
@policy_options
@training_options
@click.option("--relay-job", default=DEFAULT_GLM_RELAY_JOB, show_default=True)
@click.option("--pool", type=click.Choice([pool.value for pool in Pool]), default=Pool.BULK.value, show_default=True)
@click.option("--max-pass-rate", type=float, default=0.25, show_default=True)
@click.option("--failures-per-analysis", type=int, default=64, show_default=True)
@click.option("--max-new-ideas", type=int, default=8, show_default=True)
@click.option("--synthesized-share", type=float, default=0.5, show_default=True)
@click.option("--tasks-per-idea", type=int, default=2, show_default=True)
@click.option("--instances-per-task", type=int, default=32, show_default=True)
@click.option("--probes-per-task", type=int, default=8, show_default=True)
@click.option("--sessions", type=int, default=6, show_default=True, help="Agent sessions at a time.")
@click.option("--agent-turns", type=int, default=300, show_default=True)
def rsi(
    relay_job: str,
    pool: str,
    max_pass_rate: float,
    failures_per_analysis: int,
    max_new_ideas: int,
    synthesized_share: float,
    tasks_per_idea: int,
    instances_per_task: int,
    probes_per_task: int,
    sessions: int,
    agent_turns: int,
    **arguments,
) -> None:
    """Train with the RSI loop."""
    options, training = _split(arguments)
    feedback = FeedbackOptions(
        relay_job,
        Pool(pool),
        max_pass_rate,
        failures_per_analysis,
        max_new_ideas,
        synthesized_share,
        tasks_per_idea,
        instances_per_task,
        probes_per_task,
        sessions,
        agent_turns,
    )
    asyncio.run(_train_run(options, training, feedback=feedback))


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


def _split(arguments: dict) -> tuple[PolicyOptions, TrainingOptions]:
    policy = {name: arguments.pop(name) for name in PolicyOptions.__dataclass_fields__}
    return PolicyOptions(**policy), TrainingOptions(**arguments)


async def _train_run(options: PolicyOptions, training: TrainingOptions, *, feedback: FeedbackOptions | None) -> None:
    output = StoragePath(training.output_dir)
    # A preempted run continues from the last adapter that it uploaded.
    progress = read_progress(output, Path(ADAPTER_DIR))
    logger.info("Starting after step %d with %d groups done", progress.step, len(progress.records))
    pool = TaskPool.read(f"{POOL_URL}/{TRAIN_FILE}")
    strata: dict[str, str] = {}
    pool_tasks = _remember_stratum(pool.stream(training.seed), strata)
    for _ in islice(pool_tasks, progress.pool_tasks):
        pass
    seed_tasks = from_iterable(pool_tasks)
    with serve(options) as server:
        served = Served(options, server)
        learner = LoraLearner(
            TrainerConfig(
                model=options.model,
                lora_rank=training.lora_rank,
                learning_rate=training.learning_rate,
                tasks_per_step=training.tasks_per_step,
                samples_per_task=training.samples_per_task,
                max_steps=training.steps,
                micro_batch_tokens=training.micro_batch_tokens,
                adapter_dir=ADAPTER_DIR,
                device="cuda:0",
                seed=training.seed,
            )
        )
        trainer = partial(train, served.runner, server, learner, start_step=progress.step)
        async with AsyncExitStack() as stack:
            if feedback is None:
                stream = trainer(seed_tasks)  # The control: the same training on seed tasks only.
            else:
                boxes = await _feedback_boxes(stack, feedback, served, trainer, output, progress)
                config = LoopConfig(
                    max_pass_rate=feedback.max_pass_rate,
                    failures_per_analysis=feedback.failures_per_analysis,
                    synthesized_share=feedback.synthesized_share,
                    seed=training.seed,
                )
                stream = rsi_loop(seed_tasks, boxes, config)
            await _record(stream, strata, output, training, progress)


def _remember_stratum(stream, strata: dict[str, str]):
    for task in stream:
        strata[task.task.id] = task.stratum
        yield task.task


async def _feedback_boxes(
    stack: AsyncExitStack, feedback: FeedbackOptions, served: Served, trainer, output: StoragePath, progress: Progress
) -> Boxes:
    source = endpoint_source(None, feedback.relay_job, feedback.pool)
    judge = await stack.enter_async_context(GlmClient(source, GLM_PROVIDER))
    web = await stack.enter_async_context(ParallelWeb(os.environ[PARALLEL_API_KEY_ENV]))
    endpoint = source.current()
    agentic = AgenticConfig(
        agent_model=chat_model(endpoint.base_url, endpoint.token, GLM_MODEL, "low"),
        workspace_factory=workspace_factory(),
        solver_model=served.policy,
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
    resumed = untrained_instances(output, progress.trained_ids, feedback.instances_per_task, agentic.workspace_factory)

    def generate(ideas: AsyncIterable[CapabilityIdea]) -> AsyncIterator[TaskSpec]:
        # A resumed run first gives back the instances of the tasks that it accepted and did not train on.
        new = generate_tasks(ideas, synthesis)
        return mix(Source(resumed, share=1.0), Source(new, share=1.0), seed=0)

    return Boxes(
        train=trainer,
        analyze_failed_runs=partial(analyze_failed_runs, config=analysis, known=known_ideas(output)),
        generate_tasks=generate,
    )


async def _record(
    stream: AsyncIterator[TaskRollouts],
    strata: Mapping[str, str],
    output: StoragePath,
    training: TrainingOptions,
    progress: Progress,
) -> None:
    """Write a record of each group as it comes, after those of a resumed run, and upload the adapters every
    ``adapter_every`` steps and at the end."""
    local = Path("/tmp/baby-rsi-groups.jsonl")
    uploaded = set()
    with local.open("w") as handle:
        handle.writelines(json.dumps(record, default=str) + "\n" for record in progress.records)
        async for group in stream:
            label = strata.get(group.task.id) or f"{SYNTHESIZED_LABEL_PREFIX}{capability_of(group.task)}"
            record = {**group_record(group, transcripts=False), "label": label}
            record["run_ids"] = [graded.rollout.run_id for graded in group.rollouts]
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
            step = adapter_step(group.model)
            if step and step % training.adapter_every == 0 and group.model not in uploaded:
                uploaded.add(group.model)
                await asyncio.to_thread(_upload_progress, local, output, group.model)
    await asyncio.to_thread(_upload_progress, local, output, None)


def _upload_progress(local: Path, output: StoragePath, adapter: str | None) -> None:
    (output / "groups.jsonl").write_text(local.read_text())
    names = [adapter] if adapter else sorted(path.name for path in Path(ADAPTER_DIR).iterdir())[-1:]
    for name in names:
        # The optimizer state goes last: a resumed run takes only an adapter directory that has it.
        files = sorted(Path(ADAPTER_DIR, name).iterdir(), key=lambda file: file.name == OPTIMIZER_FILE)
        for file in files:
            (output / "adapters" / name / file.name).write_bytes(file.read_bytes())


@main.command()
@policy_options
@click.option("--output-dir", required=True)
@click.option("--adapter", "adapters", multiple=True, help="NAME=URL of a trained adapter directory; repeatable.")
@click.option("--probes", "probe_roots", multiple=True, help="An rsi output directory with held-out probes.")
@click.option("--instances-per-task", type=int, default=32, show_default=True, help="As the rsi run made them.")
@click.option("--pool-tasks", type=int, default=400, show_default=True, help="Held-out validation pool tasks.")
@click.option("--samples", type=int, default=8, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def evaluate(
    output_dir: str,
    adapters: tuple[str, ...],
    probe_roots: tuple[str, ...],
    instances_per_task: int,
    pool_tasks: int,
    samples: int,
    seed: int,
    **options,
) -> None:
    """Evaluate the base policy and trained adapters on the same held-out tasks."""
    named = dict(adapter.split("=", 1) for adapter in adapters)
    asyncio.run(
        _evaluate(
            PolicyOptions(**options),
            StoragePath(output_dir),
            named,
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
    adapters: Mapping[str, str],
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
    with serve(options) as server:
        names = [BASE_MODEL_NAME]
        for name, url in adapters.items():
            await server.load_adapter(name, _download(url, Path(ADAPTER_DIR) / name))
            names.append(name)
        # A second evaluation of the base weights measures how often a task that the base never passes passes by
        # chance on new samples.
        runs = {name: name for name in names} | {REPEAT_NAME: BASE_MODEL_NAME}
        served = Served(options, server)
        results = {}
        # One policy at a time, so that the graders of all policies see the same load.
        for name, served_name in runs.items():
            served.policy.model = served_name
            results[name] = await eval_policy(served.runner, tasks, labels, samples)
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


def _download(url: str, local: Path) -> str:
    local.mkdir(parents=True, exist_ok=True)
    for path in (StoragePath(url) / "*").glob():
        (local / path.name).write_bytes(path.read_bytes())
    return str(local)


if __name__ == "__main__":
    main()
