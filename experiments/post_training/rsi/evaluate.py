# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate policies on held-out tasks: the command line of the Baby RSI evaluation.

- ``baseline``: the pass rates of a base policy on a stratified pool sample.
- ``evaluate``: the base policy (twice) and trained checkpoints on held-out pool tasks and synthesized probes.

They run in an Iris GPU task through ``job.sh``, in the environment of marin-taskforge and marin-core: the server is
``marin.inference``'s vLLM backend.
"""

import asyncio
import hashlib
import json
import logging
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from marin.inference.backend import ModelSpec
from marin.inference.config import VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.vllm_backend import VllmBackend, VllmServedModel
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from rolloutengine.models import RolloutLimits
from taskcompendium.models import TaskSpec
from taskforge.agentic.instances import task_instances

from experiments.post_training.capability_driven_envs.generate_tasks_agentic import workspace_factory
from experiments.post_training.rsi.feedback import (
    INSTANCE_TIMEOUT,
    PROBES_FILE,
    capability_of,
    held_out,
    training_ids,
)
from experiments.post_training.rsi.policy import (
    BASE_MODEL_NAME,
    MAX_MODEL_LEN,
    OBJECT_STORE_SCHEMES,
    Sampling,
    VllmPolicy,
)
from experiments.post_training.rsi.resume import SavedTask, accepted_tasks
from experiments.post_training.rsi.rollouts import (
    PolicyRunner,
    TaskRollouts,
    group_record,
    rollout_engine,
    write_jsonl,
)
from experiments.post_training.rsi.seed import POOL_URL, TRAIN_FILE, VALIDATION_FILE, TaskPool

logger = logging.getLogger(__name__)

WEIGHTS_STREAM = "stream"
WEIGHTS_DOWNLOAD = "download"
WEIGHTS_DIR = "/tmp/baby-rsi-weights"
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
PROBE_BUILD_SLOTS = 16
STARTUP_TIMEOUT = 1800


@dataclass(frozen=True)
class Evaluation:
    """The attempts of the policy on each task of an evaluation set, with the task groups for the summary."""

    groups: tuple[TaskRollouts, ...]
    labels: Mapping[str, str]
    """The group label of each task ID, for example its stratum or its capability."""

    def summary(self) -> dict[str, Any]:
        """The mean pass rate over tasks, overall and for each label, with a 95% interval over tasks."""
        by_label: defaultdict[str, list[float]] = defaultdict(list)
        for group in self.groups:
            rate = group.pass_rate
            if rate is not None:
                by_label[self.labels.get(group.task.id, "")].append(rate)
        everything = [rate for rates in by_label.values() for rate in rates]
        return {
            "overall": _mean_interval(everything),
            "by_label": {label: _mean_interval(rates) for label, rates in sorted(by_label.items())},
            "ungraded_tasks": sum(group.pass_rate is None for group in self.groups),
        }


def _mean_interval(rates: Sequence[float]) -> dict[str, float | int]:
    if not rates:
        return {"tasks": 0}
    mean = sum(rates) / len(rates)
    variance = sum((rate - mean) ** 2 for rate in rates) / max(len(rates) - 1, 1)
    return {"tasks": len(rates), "pass_rate": mean, "ci95": 1.96 * math.sqrt(variance / len(rates))}


async def eval_policy(
    runner: PolicyRunner, tasks: Sequence[TaskSpec], labels: Mapping[str, str], samples: int
) -> Evaluation:
    """``samples`` attempts of each task with the policy's current weights."""
    return Evaluation(tuple(await runner.run_all(tasks, samples)), labels)


@dataclass(frozen=True)
class PolicyOptions:
    model: str
    weights: str | None
    thinking: bool
    max_tokens: int
    max_model_len: int
    gpu_memory_utilization: float
    concurrency: int
    grader_slots: int
    vllm_args: tuple[str, ...]
    weight_transfer: str


def policy_options(function):
    for option in reversed(
        [
            click.option("--model", required=True, help="The Hugging Face ID of the policy and its tokenizer."),
            click.option("--weights", help="The base weights to serve (for example an s3:// copy); --model if unset."),
            click.option("--thinking/--no-thinking", default=False, show_default=True, help="Thinking mode."),
            click.option("--max-tokens", type=int, default=4096, show_default=True, help="Tokens of one turn."),
            click.option("--max-model-len", type=int, default=MAX_MODEL_LEN, show_default=True),
            click.option("--gpu-memory-utilization", type=float, default=0.85, show_default=True),
            click.option("--concurrency", type=int, default=128, show_default=True, help="Attempts at a time."),
            click.option("--grader-slots", type=int, default=16, show_default=True, help="Script graders at a time."),
            click.option("--vllm-arg", "vllm_args", multiple=True, help="An argument of vllm serve; repeatable."),
            click.option(
                "--weight-transfer",
                type=click.Choice([WEIGHTS_STREAM, WEIGHTS_DOWNLOAD]),
                default=WEIGHTS_STREAM,
                show_default=True,
                help="Stream object-store weights into vLLM, or download them to the task's disk first.",
            ),
        ]
    ):
        function = option(function)
    return function


@click.group()
def main() -> None:
    configure_logging(logging.INFO)


class Served:
    """The vLLM server and the policy runner of one task: the policy and its rollouts."""

    def __init__(self, options: PolicyOptions, server: VllmServedModel):
        self.server = server
        self.policy = VllmPolicy.for_model(
            server.base_url,
            options.model,
            Sampling(temperature=1.0, top_p=1.0, max_tokens=options.max_tokens),
            options.max_model_len,
            chat_template_kwargs={"enable_thinking": options.thinking},
        )
        self.runner = PolicyRunner(rollout_engine(options.grader_slots), self.policy, POLICY_LIMITS, options.concurrency)


def serve(options: PolicyOptions, weights: str | None = None) -> AbstractContextManager[VllmServedModel]:
    """A vLLM server of the base weights, or of ``weights`` (a trained checkpoint) with the same tokenizer.

    The Marin vLLM fork serves Qwen3 and the GrugMoE architecture of Snowball, and streams s3:// weights.
    """
    model = weights or options.weights or options.model
    if options.weight_transfer == WEIGHTS_DOWNLOAD and model.startswith(OBJECT_STORE_SCHEMES):
        model = _download_weights(model)
    engine = VllmEngineConfig(
        launcher=VllmLauncherType.CUDA,
        source=VllmSource.MARIN_FORK,
        startup_timeout_seconds=STARTUP_TIMEOUT,
        extra_args=("--gpu-memory-utilization", str(options.gpu_memory_utilization), *options.vllm_args),
    )
    spec = ModelSpec(
        weights=model,
        api_model=BASE_MODEL_NAME,
        num_chips=None,
        tensor_parallel_size=None,
        dtype="auto",
        max_model_len=options.max_model_len,
        chat_template_content=None,
        tokenizer=options.model,
    )
    return VllmBackend(engine).serve(spec)


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
    for probe, unseen in await read_probes(probe_roots, instances_per_task):
        tasks.append(probe)
        # The probes of a held-out idea measure transfer: the policy never trained on any task of its family.
        labels[probe.id] = f"{'probe-heldout' if unseen else 'probe'}:{capability_of(probe)}"
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
        with serve(options, url) as server:
            results[name] = await eval_policy(Served(options, server).runner, tasks, labels, samples)
    write_evaluation(output, options.model, results)


def _download_weights(url: str) -> str:
    """A local copy of the weights at ``url``; the Run:ai stream of a 67B export sometimes fails on S3 errors."""
    local = Path(WEIGHTS_DIR) / hashlib.sha256(url.encode()).hexdigest()[:16]
    fs, path = url_to_fs(url)
    logger.info("Downloading %s to %s", url, local)
    fs.get(path.rstrip("/") + "/", str(local), recursive=True)
    return str(local)


async def read_probes(roots: Sequence[str], instances_per_task: int) -> list[tuple[TaskSpec, bool]]:
    """The held-out probes of the accepted tasks in ``roots``, each with whether its idea is held out; a probe equal
    to a training instance is dropped."""
    factory = workspace_factory()
    slots = asyncio.Semaphore(PROBE_BUILD_SLOTS)

    async def held_out_probes(saved: SavedTask) -> list[tuple[TaskSpec, bool]]:
        ids = training_ids(saved.accepted, instances_per_task)
        async with slots:
            training = await task_instances(saved.accepted, ids, factory, timeout=INSTANCE_TIMEOUT)
        lines = (saved.directory / PROBES_FILE).read_text().splitlines()
        stored = [TaskSpec.model_validate_json(line) for line in lines if line]
        kept = held_out(stored, saved.accepted, training)
        logger.info("%s: %d of %d probes are held out", saved.accepted.task.id, len(kept), len(stored))
        return [(probe, saved.held_out) for probe in kept]

    saved_tasks = [saved for root in roots for saved in accepted_tasks(StoragePath(root))]
    return [probe for kept in await asyncio.gather(*map(held_out_probes, saved_tasks)) for probe in kept]


if __name__ == "__main__":
    main()
