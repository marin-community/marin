# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate TaskSpecs from the capability catalog, with GLM writing a program for each task.

This driver runs the unstaged path: a catalog entry goes to a program, and the program writes a
TaskSpec, with none of Taskforge's stages in between. ``propose_tasks`` runs the staged path
instead, from the catalog to proposals, to ``stage20_triage`` and on to the build.

For each selected capability and sample, the driver sends GLM the messages from
``program_prompt.program_messages`` and receives a Python program. The driver runs the program as a
child process in the driver's own Python environment, passing it the path for the TaskSpec it
writes. A failure from the model or the program goes into that item's record alone, so the run
always writes its outputs. The driver holds everything in memory until every item finishes, then
writes:

* ``OUTPUT_DIR/<item id>/`` with ``request.json``, ``completions.json`` (content, reasoning, usage
  and every HTTP attempt), ``program.py``, ``run.json`` and, when the output parses, ``task.json``.
* ``OUTPUT_DIR/summary.json``, mapping each item id to ``task`` or to its failure.

Item ids are ``<capability id>--<sample>``.

``--pool`` selects the environment variable for the GLM bearer token: ``GLM_BULK_TOKEN`` for
``bulk``, ``GLM_HIGH_TOKEN`` for ``high``. The token in turn selects the router pool serving each
request, so ``--pool`` must match the token's pool. ``--pool`` also picks the pool whose workers the
client's ``/health`` check counts. On a laptop, with the GLM router port-forwarded and
``GLM_BULK_TOKEN`` exported::

    uv run --package marin-taskforge --frozen python -m \\
        experiments.post_training.capability_driven_envs.generate_tasks \\
        --output-dir /tmp/capability-tasks --base-url http://127.0.0.1:18000/v1 --pool bulk

Inside an Iris task, ``--relay-job`` gives the GLM relay job in place of ``--base-url``. The client
resolves the endpoint from the relay again while it recovers from a missing route or a refused
connection, following a relay that moved.
"""

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
from dataclasses import asdict, dataclass, field
from pathlib import Path

import click
from pydantic import ValidationError
from rigging.log_setup import configure_logging
from taskcompendium.models import TaskSpec
from taskforge.content_hash import pretty_json
from taskforge.llm.client import Completion, ModelClient, ModelRequestRejected, ModelUnavailable
from taskforge.llm.glm_client import EndpointSource, FixedEndpoint, GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy, LLMProviderConfig, Message
from taskforge.llm.structured import StructuredOutputError, schema_instruction

from experiments.post_training.capability_driven_envs.catalog import (
    CapabilityIdea,
    example_catalog_path,
    load_capability_ideas,
)
from experiments.post_training.capability_driven_envs.program_prompt import (
    ProgramSubmission,
    program_messages,
)
from experiments.post_training.glm import (
    DEFAULT_GLM_RELAY_JOB,
    GLM_BULK_TOKEN_ENV,
    GLM_MODEL,
    resolve_glm_base_url,
)

logger = logging.getLogger(__name__)

FILE_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
GLM_HIGH_TOKEN_ENV = "GLM_HIGH_TOKEN"
POOL_TOKEN_ENV = {Pool.BULK: GLM_BULK_TOKEN_ENV, Pool.HIGH: GLM_HIGH_TOKEN_ENV}
"""The environment variable holding each pool's bearer token; the token selects the pool."""
PROGRAM_FILE = "program.py"
TASK_FILE = "task.json"
SUMMARY_FILE = "summary.json"
TASK_STATUS = "task"
PROGRAM_ENV = {"PATH": os.defpath}
"""The program's entire environment, ``PATH`` alone, so the driver's variables and credentials stay in the driver."""

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


# NOTE: The driver holds stdout and stderr in memory and writes them into run.json, so a program printing
# without limit can exhaust memory. Truncate them or stream them to files if that happens.
@dataclass(frozen=True)
class ProgramRun:
    """A run of a task program.

    ``exit_code`` is None after a timeout. ``failure`` is None exactly when ``task`` is set.
    """

    exit_code: int | None
    stdout: str
    stderr: str
    task: TaskSpec | None
    failure: str | None


@dataclass(frozen=True)
class GeneratedTask:
    """An item's request and completions, with the program and its run when GLM submitted a valid program."""

    item_id: str
    messages: list[Message]
    completions: tuple[Completion, ...]
    source: str | None
    run: ProgramRun | None
    failure: str | None


async def run_program(source: str, task_id: str, timeout: float) -> ProgramRun:
    """Run ``source`` as ``python3 program.py <task_id> <output path>`` and parse the TaskSpec it writes.

    The program starts in a new session. On timeout the driver kills the whole process group, taking
    the program's own children down with it.
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
        out, err = stdout.decode(errors="replace"), stderr.decode(errors="replace")
        if code != 0:
            return ProgramRun(code, out, err, None, f"the program exited with status {code}")
        if not output.is_file():
            return ProgramRun(code, out, err, None, "the program wrote no file at the output path")
        try:
            task = TaskSpec.model_validate_json(output.read_bytes())
        except ValidationError as error:
            return ProgramRun(code, out, err, None, f"the output is not a TaskSpec: {error}")
        return ProgramRun(code, out, err, task, None)


async def generate_task(
    client: ModelClient, idea: CapabilityIdea, item_id: str, policy: LLMPolicy, program_timeout: float
) -> GeneratedTask:
    """Ask GLM for a program for ``idea`` and run it.

    An invalid submission, a rejected request, an exhausted context, an unavailable model and a failed
    program all end up as this item's ``failure``. The function returns that failure in place of
    raising it, so the other items proceed and the run still writes its outputs.
    """
    messages = program_messages(idea)
    # The library appends the field instruction as the final user turn; the record keeps what was sent.
    request: list[Message] = [*messages, {"role": "user", "content": schema_instruction(ProgramSubmission)}]
    try:
        result = await client.complete_structured(messages, ProgramSubmission, policy)
    except StructuredOutputError as error:
        return GeneratedTask(item_id, request, error.completions, None, None, str(error))
    except (ModelRequestRejected, ModelUnavailable) as error:
        return GeneratedTask(item_id, request, (), None, None, str(error))
    run = await run_program(result.value.source, item_id, program_timeout)
    return GeneratedTask(item_id, request, result.completions, result.value.source, run, run.failure)


async def generate_tasks(
    client: ModelClient,
    items: Sequence[tuple[str, CapabilityIdea]],
    policy: LLMPolicy,
    program_timeout: float,
    concurrency: int,
) -> list[GeneratedTask]:
    """Generate every ``(item id, idea)`` item, at most ``concurrency`` at a time."""
    slots = asyncio.Semaphore(concurrency)

    async def limited(item_id: str, idea: CapabilityIdea) -> GeneratedTask:
        async with slots:
            generated = await generate_task(client, idea, item_id, policy, program_timeout)
        logger.info("%s: %s", item_id, generated.failure or TASK_STATUS)
        return generated

    return await asyncio.gather(*(limited(item_id, idea) for item_id, idea in items))


def task_items(ideas: Mapping[str, CapabilityIdea], samples: int) -> list[tuple[str, CapabilityIdea]]:
    """Build ``samples`` items per idea with ids ``<capability id>--<sample>``.

    Raises ValueError for an id unsafe as a file name.
    """
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


@dataclass(frozen=True)
class RelayEndpoint:
    """The endpoint an Iris GLM relay job registers, looked up through the Iris client.

    The client calls ``current`` once at construction, then again only while it recovers from a
    missing route, an empty pool or a refused connection. Requests in normal operation skip the Iris
    lookup, which works only inside an Iris task.
    """

    relay_job: str
    token: str = field(repr=False)
    pool: Pool

    def current(self) -> GlmEndpoint:
        return GlmEndpoint(resolve_glm_base_url(self.relay_job), self.token, self.pool)


def pool_token(pool: Pool) -> str:
    """The bearer token for ``pool``, from its ``POOL_TOKEN_ENV`` variable."""
    name = POOL_TOKEN_ENV[pool]
    token = os.environ.get(name, "")
    if not token:
        raise click.ClickException(f"--pool {pool} reads its GLM bearer token from {name}, which is not set")
    return token


def endpoint_source(base_url: str | None, relay_job: str | None, pool: Pool) -> EndpointSource:
    """Build a fixed endpoint for ``base_url`` or a relay-resolved one for ``relay_job``; give exactly one."""
    if base_url is not None and relay_job is None:
        return FixedEndpoint(GlmEndpoint(base_url.rstrip("/"), pool_token(pool), pool))
    if relay_job is not None and base_url is None:
        return RelayEndpoint(relay_job, pool_token(pool), pool)
    raise click.UsageError("give exactly one of --base-url and --relay-job")


def selected_ideas(catalog: Path, capability_ids: Sequence[str]) -> dict[str, CapabilityIdea]:
    """Load the catalog's ideas, keeping only ``capability_ids`` when that list is non-empty."""
    ideas = load_capability_ideas(catalog)
    unknown = sorted(set(capability_ids) - set(ideas))
    if unknown:
        raise ValueError(f"capabilities not in {catalog}: {unknown}")
    return {idea_id: ideas[idea_id] for idea_id in capability_ids} if capability_ids else ideas


async def _generate(
    source: EndpointSource,
    items: Sequence[tuple[str, CapabilityIdea]],
    program_timeout: float,
    concurrency: int,
) -> list[GeneratedTask]:
    async with GlmClient(source, GLM_PROVIDER) as client:
        return await generate_tasks(client, items, LLMPolicy(), program_timeout, concurrency)


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory for the run's outputs; it must not exist yet.",
)
@click.option(
    "--catalog",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The capability catalog JSON file. Default: the package's data/example_catalog.json.",
)
@click.option("--capability", "capabilities", multiple=True, help="Only this capability id; repeatable.")
@click.option("--samples", type=click.IntRange(min=1), default=1, show_default=True, help="Tasks per capability.")
@click.option("--base-url", help="The GLM router's OpenAI-compatible base URL, ending in /v1.")
@click.option(
    "--relay-job",
    help=f"The Iris GLM relay job to resolve the endpoint from, e.g. {DEFAULT_GLM_RELAY_JOB} (inside Iris only).",
)
@click.option(
    "--pool",
    type=click.Choice([pool.value for pool in Pool]),
    default=Pool.BULK.value,
    show_default=True,
    help=(
        f"The router pool. It selects the token variable ({GLM_BULK_TOKEN_ENV} for bulk, {GLM_HIGH_TOKEN_ENV} "
        "for high) and the pool for the /health check. The token selects the pool serving requests, so this "
        "must match the token's pool."
    ),
)
@click.option("--program-timeout", type=float, default=300.0, show_default=True, help="Seconds each program may run.")
@click.option(
    "--concurrency", type=click.IntRange(min=1), default=16, show_default=True, help="Items generated at once."
)
def main(
    output_dir: Path,
    catalog: Path | None,
    capabilities: tuple[str, ...],
    samples: int,
    base_url: str | None,
    relay_job: str | None,
    pool: str,
    program_timeout: float,
    concurrency: int,
) -> None:
    """Generate TaskSpecs from the capability catalog, with GLM writing a program for each task.

    Give exactly one of --base-url (a port-forwarded router) and --relay-job (inside an Iris task).
    --pool selects the environment variable for the GLM bearer token. The token selects the pool
    serving the requests, so --pool must match the token's pool.
    """
    configure_logging(logging.INFO)
    source = endpoint_source(base_url, relay_job, Pool(pool))
    if output_dir.exists():
        raise click.BadParameter(f"{output_dir} already exists", param_hint="'--output-dir'")
    items = task_items(selected_ideas(catalog or example_catalog_path(), capabilities), samples)
    generated = asyncio.run(_generate(source, items, program_timeout, concurrency))
    write_outputs(output_dir, generated)
    click.echo(output_dir / SUMMARY_FILE)


if __name__ == "__main__":
    main()
