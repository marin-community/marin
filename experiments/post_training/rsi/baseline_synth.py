# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Baselines of the generate box for gate 5: tasks for capability ideas without the agentic Taskforge.

``direct`` asks GLM for one instance per call. ``generator`` asks GLM once per idea for a Python program that prints
instances, and runs it. Both write instances of a Reasoning Gym type (a question and an entry in the form of the
idea's ``example_entries``), keep only an instance whose own answer the dataset's scorer gives the full reward, and
grade them as ``gate5_grade.py`` does, like the ``grader`` variant of the agentic synthesis. There is no gate, review
or adversary. Run it in an Iris task that can resolve the GLM relay, with ``GLM_BULK_TOKEN`` set; it writes
``synthesized.jsonl`` for ``run.py skyrl-oracle --tasks-file``.
"""

import asyncio
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import click
import reasoning_gym
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from taskforge.capability import CapabilityIdea
from taskforge.llm.glm_client import GlmClient, Pool
from taskforge.llm.policy import LLMPolicy, ReasoningEffort

from experiments.post_training.capability_driven_envs.generate_tasks import GLM_PROVIDER, endpoint_source
from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB
from experiments.post_training.rsi.gap import GAP_CAPABILITY_PREFIX, entry_task, scan_types
from experiments.post_training.rsi.run import read_ideas

logger = logging.getLogger(__name__)

SYNTHESIZED_FILE = "synthesized.jsonl"
POLICY = LLMPolicy(reasoning_effort=ReasoningEffort.LOW, max_tokens=32768)
PROGRAM_TIMEOUT = 300.0
TYPE_BY_CAPABILITY = {
    "gap.search.shortest-state-machine-sequence": "quantum_lock",
    "gap.strings.exact-lexicographic-sorting": "word_sorting",
    "gap.calculus.antiderivatives-by-substitution-and-parts": "intermediate_integration",
    "gap.algebra.solve-linear-equations": "simple_equations",
    "gap.matrix.rotate-square-matrix": "rotate_matrix",
    "gap.arithmetic.exact-decimal-expressions": "decimal_arithmetic",
    "gap.strings.reverse-spelling": "spell_backward",
    "gap.induction.causal-detector": "acre",
    "gap.arithmetic.countdown-expression": "countdown",
}


class Instance(BaseModel):
    question: str
    entry_json: str
    """The entry as JSON: ``answer`` and ``metadata`` in the form of the example entries."""


class Program(BaseModel):
    source: str
    """A Python program; ``python program.py N SEED`` prints N lines, each a JSON object with question and entry."""


def brief(idea: CapabilityIdea) -> str:
    capability = dict(idea.capability)
    examples = capability.pop("example_tasks", [])[:8]
    entries = capability.pop("example_entries", [])
    for key in ("example_task_spec", "grader_rule", "evidence", "example_task"):
        capability.pop(key, None)
    pairs = [{"question": q, "entry": e} for q, e in zip(examples, entries, strict=False)] or examples
    return json.dumps({"capability": capability, "examples": pairs}, indent=1)


def valid(name: str, question: str, entry: dict) -> bool:
    """Whether the instance is whole and the dataset's scorer gives its own answer the full reward."""
    if not question.strip() or not isinstance(entry.get("answer"), str) or not isinstance(entry.get("metadata"), dict):
        return False
    try:
        if name == "quantum_lock" and not reaches_target(entry):
            return False
        return float(reasoning_gym.get_score_answer_fn(name)(entry["answer"], entry)) == 1.0
    except Exception:
        # A malformed entry that the scorer cannot read is an invalid instance.
        return False


def reaches_target(entry: dict) -> bool:
    """For ``quantum_lock``, whose scorer reads only the letters A to C: the answer is a sequence that reaches the
    target, so that an empty or garbled answer does not pass as its own reference."""
    sequence = re.findall(r"[A-C]", entry["answer"].upper())
    lock = reasoning_gym.create_dataset("quantum_lock", size=1, seed=0)
    return bool(sequence) and lock.simulate_sequence(entry["metadata"], sequence) == entry["metadata"]["target_value"]


async def direct(client: GlmClient, idea: CapabilityIdea, count: int, concurrency: int) -> list[tuple[str, dict]]:
    slots = asyncio.Semaphore(concurrency)
    request = (
        "Write one new task of the capability below, of the same kind and form as the examples, with a different "
        "instance. Give its question as the user would see it, and its entry (answer and metadata, in the exact form "
        "of the examples' entries).\n\n" + brief(idea)
    )

    async def one(k: int) -> tuple[str, dict] | None:
        async with slots:
            message = f"{request}\n\nInstance number {k}: vary the values, and cover the range of the examples."
            try:
                result = await client.complete_structured([{"role": "user", "content": message}], Instance, POLICY)
                return result.value.question, json.loads(result.value.entry_json)
            except Exception as error:
                logger.warning("Instance %d failed: %s", k, error)
                return None

    return [pair for pair in await asyncio.gather(*(one(k) for k in range(count))) if pair is not None]


async def generator(client: GlmClient, idea: CapabilityIdea, count: int) -> list[tuple[str, dict]]:
    request = (
        "Write a Python program that generates new tasks of the capability below, of the same kind and form as the "
        "examples. Run as `python program.py N SEED`, it prints N lines, each a JSON object with the keys question "
        "and entry (answer and metadata, in the exact form of the examples' entries). Its parameters must span the "
        "range of the examples, including their hardest instances. Use only the standard library.\n\n" + brief(idea)
    )
    result = await client.complete_structured([{"role": "user", "content": request}], Program, POLICY)
    with tempfile.TemporaryDirectory() as directory:
        program = Path(directory) / "program.py"
        program.write_text(result.value.source)
        try:
            run = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, str(program), str(count), "0"],
                capture_output=True,
                text=True,
                timeout=PROGRAM_TIMEOUT,
                cwd=directory,
            )
            stdout, returncode, stderr = run.stdout, run.returncode, run.stderr
        except subprocess.TimeoutExpired as expired:
            # A program that hangs keeps the instances it printed before the timeout, like one that fails.
            stdout = expired.stdout.decode() if isinstance(expired.stdout, bytes) else expired.stdout or ""
            returncode, stderr = None, f"timed out after {PROGRAM_TIMEOUT} seconds"
    pairs = []
    for line in stdout.splitlines():
        try:
            record = json.loads(line)
            pairs.append((record["question"], record["entry"]))
        except (ValueError, KeyError, TypeError):
            continue
    if returncode != 0:
        logger.warning("The generator of %s exited %s: %s", idea.capability_id, returncode, stderr[-2000:])
    return pairs


@click.command()
@click.option("--ideas-file", required=True)
@click.option("--mode", type=click.Choice(["direct", "generator"]), required=True)
@click.option("--instances-per-idea", type=int, default=256, show_default=True)
@click.option("--concurrency", type=int, default=32, show_default=True, help="GLM calls at a time (direct).")
@click.option("--relay-job", default=DEFAULT_GLM_RELAY_JOB, show_default=True)
@click.option("--output-dir", required=True)
def main(ideas_file: str, mode: str, instances_per_idea: int, concurrency: int, relay_job: str, output_dir: str) -> None:
    configure_logging(logging.INFO)
    asyncio.run(_main(ideas_file, mode, instances_per_idea, concurrency, relay_job, StoragePath(output_dir)))


async def _main(ideas_file: str, mode: str, count: int, concurrency: int, relay_job: str, output: StoragePath) -> None:
    ideas = read_ideas(ideas_file)
    types = await asyncio.to_thread(scan_types)
    lines = []
    async with GlmClient(endpoint_source(None, relay_job, Pool.BULK), GLM_PROVIDER) as client:
        for idea in ideas:
            name = TYPE_BY_CAPABILITY[idea.capability_id]
            pairs = await (
                direct(client, idea, count, concurrency) if mode == "direct" else generator(client, idea, count)
            )
            kept = [(q, e) for q, e in pairs if valid(name, q, e)]
            logger.info("%s: %d of %d instances are valid", idea.capability_id, len(kept), len(pairs))
            capability = f"{GAP_CAPABILITY_PREFIX.removesuffix('gap-')}{idea.capability_id}"
            for k, (question, entry) in enumerate(kept):
                task = entry_task(
                    types[name].template, name, f"{idea.capability_id}.{mode}-{k}", question, entry, capability
                )
                lines.append(task.model_dump_json() + "\n")
    (output / SYNTHESIZED_FILE).write_text("".join(lines))
    logger.info("Wrote %d tasks", len(lines))


if __name__ == "__main__":
    main()
