# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate TaskSpecs from the capability catalog: one GLM-written program per task.

For each selected capability and sample, GLM gets ``program_prompt.program_messages`` and submits a
Python program; the program runs in this driver's Python environment, as a child process, and must
write a TaskSpec to the path it is given. Every item records its own failure, from the model or the
program, so the run always writes. Everything stays in memory until every item is done; then
``OUTPUT_DIR/<item id>/`` gets ``request.json``, ``completions.json`` (content, reasoning, usage and
every HTTP attempt), ``program.py``, ``run.json`` and, when the output parses, ``task.json``, and
``OUTPUT_DIR/summary.json`` maps each item id to ``task`` or its failure. Item ids are
``<capability id>--<sample>``. On a laptop, with the GLM router port-forwarded::

    uv run --package marin-taskforge --frozen python -m \\
        experiments.post_training.capability_driven_envs.generate_tasks \\
        --output-dir /tmp/capability-tasks --base-url http://127.0.0.1:18000/v1 \\
        --token-file <file with a GLM_API_TOKEN= line> --pool high

Inside an Iris task, ``--relay-job`` names the GLM relay job instead of ``--base-url``.
"""

import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import ValidationError
from taskcompendium.models import TaskSpec
from taskforge.content_hash import pretty_json
from taskforge.llm.client import Completion, ModelClient, ModelRequestRejected, ModelUnavailable
from taskforge.llm.endpoint import GLM_MODEL
from taskforge.llm.glm_client import GlmClient, GlmEndpoint, Pool, endpoint_in_task
from taskforge.llm.policy import LLMPolicy, LLMProviderConfig, Message
from taskforge.llm.structured import StructuredOutputError

from experiments.post_training.capability_driven_envs.catalog import CapabilityIdea, load_capability_ideas
from experiments.post_training.capability_driven_envs.program_prompt import (
    ProgramSubmission,
    program_messages,
)

logger = logging.getLogger(__name__)

CATALOG = Path(__file__).resolve().parent / "data" / "catalog.json"
FILE_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
TOKEN_KEY = "GLM_API_TOKEN"
PROGRAM_FILE = "program.py"
TASK_FILE = "task.json"
SUMMARY_FILE = "summary.json"
TASK_STATUS = "task"
OUTPUT_TAIL_CHARS = 20_000
PROGRAM_ENV = {"PATH": os.defpath}
"""The program's whole environment: none of the driver's variables (credentials among them) reach it."""

GLM_PROVIDER = LLMProviderConfig(
    model=GLM_MODEL,
    max_attempts=8,
    hold_timeout=3600.0,
    backoff_initial=1.0,
    backoff_maximum=60.0,
    backoff_factor=2.0,
    stall_timeout=600.0,
    max_connections=512,
    structured_repairs=1,
)


@dataclass(frozen=True)
class ProgramRun:
    """One run of a task program. ``exit_code`` is None when it timed out; ``failure`` is None only with ``task``."""

    exit_code: int | None
    stdout: str
    stderr: str
    task: TaskSpec | None
    failure: str | None


@dataclass(frozen=True)
class GeneratedTask:
    """One item: the request, every completion, and the program and its run when GLM submitted a valid one."""

    item_id: str
    messages: list[Message]
    completions: tuple[Completion, ...]
    source: str | None
    run: ProgramRun | None
    failure: str | None


def _tail(output: bytes) -> str:
    return output.decode(errors="replace")[-OUTPUT_TAIL_CHARS:]


async def run_program(source: str, task_id: str, timeout: float) -> ProgramRun:
    """Run ``source`` as ``python3 program.py <task_id> <output path>`` and parse the TaskSpec it writes.

    The program runs in its own session; on timeout its whole process group is killed, so nothing it
    spawned outlives it.
    """
    with tempfile.TemporaryDirectory(prefix="task-program-") as workdir:
        program, output = Path(workdir) / PROGRAM_FILE, Path(workdir) / TASK_FILE
        program.write_text(source)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(program),
            task_id,
            str(output),
            cwd=workdir,
            env=PROGRAM_ENV,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except TimeoutError:
            # The group may have exited between the timeout and the kill; that is still a timeout.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            return ProgramRun(None, "", "", None, f"the program did not finish within {timeout}s")
        code = process.returncode
        if code != 0:
            return ProgramRun(code, _tail(stdout), _tail(stderr), None, f"the program exited with status {code}")
        if not output.is_file():
            return ProgramRun(code, _tail(stdout), _tail(stderr), None, "the program wrote no file at the output path")
        try:
            task = TaskSpec.model_validate_json(output.read_bytes())
        except ValidationError as error:
            return ProgramRun(code, _tail(stdout), _tail(stderr), None, f"the output is not a TaskSpec: {error}")
        return ProgramRun(code, _tail(stdout), _tail(stderr), task, None)


async def generate_task(
    client: ModelClient, idea: CapabilityIdea, item_id: str, policy: LLMPolicy, program_timeout: float
) -> GeneratedTask:
    """Ask GLM for a program for ``idea`` and run it.

    Every model error (an invalid submission, a rejected request, an exhausted context, the model
    unavailable) and every program failure is recorded as this item's failure, never raised, so the
    other items and the run's outputs survive it.
    """
    messages = program_messages(idea)
    try:
        result = await client.complete_structured(messages, ProgramSubmission, policy)
    except StructuredOutputError as error:
        return GeneratedTask(item_id, messages, error.completions, None, None, str(error))
    except (ModelRequestRejected, ModelUnavailable) as error:
        return GeneratedTask(item_id, messages, (), None, None, str(error))
    run = await run_program(result.value.source, item_id, program_timeout)
    return GeneratedTask(item_id, messages, result.completions, result.value.source, run, run.failure)


async def generate_tasks(
    client: ModelClient,
    items: Sequence[tuple[str, CapabilityIdea]],
    policy: LLMPolicy,
    program_timeout: float,
    concurrency: int,
) -> list[GeneratedTask]:
    """Generate every ``(item id, idea)`` item, at most ``concurrency`` at a time."""
    slots = asyncio.Semaphore(concurrency)

    async def bounded(item_id: str, idea: CapabilityIdea) -> GeneratedTask:
        async with slots:
            generated = await generate_task(client, idea, item_id, policy, program_timeout)
        logger.info("%s: %s", item_id, generated.failure or TASK_STATUS)
        return generated

    return await asyncio.gather(*(bounded(item_id, idea) for item_id, idea in items))


def task_items(ideas: Mapping[str, CapabilityIdea], samples: int) -> list[tuple[str, CapabilityIdea]]:
    """``samples`` items per idea, ids ``<capability id>--<sample>``; an id that is not file-safe raises ValueError."""
    items = [(f"{idea_id}--{sample:02d}", idea) for idea_id, idea in ideas.items() for sample in range(samples)]
    for item_id, _ in items:
        if FILE_SAFE_ID.match(item_id) is None:
            raise ValueError(f"item id {item_id!r} is not file-safe")
    return items


def write_outputs(output_dir: Path, generated: Sequence[GeneratedTask]) -> None:
    """Write every item's files and ``summary.json`` into ``output_dir``, which must not exist yet."""
    output_dir.mkdir(parents=True)
    for item in generated:
        item_dir = output_dir / item.item_id
        item_dir.mkdir()
        (item_dir / "request.json").write_text(json.dumps(item.messages, indent=2))
        (item_dir / "completions.json").write_text(pretty_json(item.completions))
        if item.source is not None:
            (item_dir / PROGRAM_FILE).write_text(item.source)
        if item.run is not None:
            record = {key: value for key, value in asdict(item.run).items() if key != "task"}
            (item_dir / "run.json").write_text(json.dumps(record, indent=2))
            if item.run.task is not None:
                (item_dir / TASK_FILE).write_text(item.run.task.model_dump_json(indent=2))
    summary = {item.item_id: item.failure or TASK_STATUS for item in generated}
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))


def read_token(path: Path) -> str:
    """The value of the ``GLM_API_TOKEN=`` line in ``path``; the value is never echoed."""
    for line in path.read_text().splitlines():
        name, separator, value = line.strip().partition("=")
        if separator and name.strip() == TOKEN_KEY and value.strip():
            return value.strip()
    raise ValueError(f"{path} has no non-empty {TOKEN_KEY}= line")


def selected_ideas(catalog: Path, capability_ids: Sequence[str]) -> dict[str, CapabilityIdea]:
    """The catalog's ideas, or only ``capability_ids`` when any are named."""
    ideas = load_capability_ideas(catalog)
    unknown = sorted(set(capability_ids) - set(ideas))
    if unknown:
        raise ValueError(f"capabilities not in {catalog}: {unknown}")
    return {idea_id: ideas[idea_id] for idea_id in capability_ids} if capability_ids else ideas


async def _generate(
    endpoint: GlmEndpoint, items: Sequence[tuple[str, CapabilityIdea]], args: argparse.Namespace
) -> list[GeneratedTask]:
    async with GlmClient(endpoint, GLM_PROVIDER) as client:
        return await generate_tasks(client, items, LLMPolicy(), args.program_timeout, args.concurrency)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="where the run is written; must not exist")
    parser.add_argument("--catalog", type=Path, default=CATALOG, help="the capability catalog JSON file")
    parser.add_argument("--capability", action="append", default=[], help="only this capability id; repeatable")
    parser.add_argument("--samples", type=positive_int, default=1, help="tasks per capability")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--base-url", help="the GLM router's OpenAI-compatible base URL, ending in /v1")
    where.add_argument("--relay-job", help="the Iris GLM relay job to resolve the endpoint from (inside Iris only)")
    parser.add_argument("--token-file", type=Path, required=True, help=f"a file with a {TOKEN_KEY}= line")
    parser.add_argument("--pool", type=Pool, choices=list(Pool), default=Pool.BULK, help="the router pool")
    parser.add_argument("--program-timeout", type=float, default=300.0, help="seconds each program may run")
    parser.add_argument("--concurrency", type=positive_int, default=16, help="items generated at once")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.output_dir.exists():
        raise FileExistsError(f"{args.output_dir} already exists")
    items = task_items(selected_ideas(args.catalog, args.capability), args.samples)
    token = read_token(args.token_file.expanduser())
    if args.relay_job is not None:
        endpoint = endpoint_in_task(args.relay_job, token, args.pool)
    else:
        endpoint = GlmEndpoint(base_url=args.base_url.rstrip("/"), token=token, pool=args.pool)
    generated = asyncio.run(_generate(endpoint, items, args))
    write_outputs(args.output_dir, generated)
    print(args.output_dir / SUMMARY_FILE)


if __name__ == "__main__":
    main()
