# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An independent audit of the synthesized tasks by a model of another family than the GLM agent and reviewer.

For each task that a synthesis session accepted, DeepSeek reads the capability idea, the task, its grader with the
reference answer, and two held-out probe instances, and judges whether the task tests the capability, is well
posed, and has a correct reference answer. The agreement with the GLM reviewer, which accepted every audited task,
measures how much the acceptance depends on the reviewer's model. Inside an Iris task on the GLM relay's cluster::

    uv run --package marin-taskforge --frozen python -m experiments.post_training.baby_rsi.audit \\
        s3://<bucket>/<prefix>/rsi --relay-job /muchanem/glm53-relay
"""

import asyncio
import json
import logging
from dataclasses import dataclass

import click
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.runtime.resources import resource_bytes
from taskforge.llm.glm_client import GlmClient, Pool
from taskforge.llm.policy import LLMPolicy, LLMProviderConfig, Message

from experiments.post_training.capability_driven_envs.generate_tasks import GLM_PROVIDER, endpoint_source
from experiments.post_training.capability_driven_envs.generate_tasks_agentic import DEEPSEEK_MODEL

logger = logging.getLogger(__name__)

PROBES_SHOWN = 2
# Without a mark at the cut, the auditor reports the files as broken and their references as wrong.
FILE_CHARS = 20000
AUDIT_PROMPT = f"""\
You audit a reinforcement-learning task that another model made to train a capability. You get the capability, the \
task's prompt and files, its grader with the reference answer, and two more instances that the same build program \
made from other task ids. Judge the task as written, and say what is wrong.

The harness writes the agent's final reply to the grader's output path (for example /app/answer.txt), so the prompt \
does not need to name that file. A file longer than {FILE_CHARS} characters ends with a [CUT ...] mark here, but the \
agent sees the whole file: do not report the cut as a problem, or a reference as wrong because of data after the cut."""


class TaskAudit(BaseModel):
    """The audit of one task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tests_capability: bool = Field(description="The task needs the capability, not only general knowledge.")
    well_posed: bool = Field(description="The prompt is unambiguous and has exactly one correct answer.")
    reference_correct: bool = Field(description="The reference answer is the correct answer of the prompt.")
    instances_consistent: bool = Field(description="The other instances are the same kind of task and also valid.")
    problems: tuple[str, ...] = Field(description="Each problem that you found, or none.")


@dataclass(frozen=True)
class DeepSeekPolicy(LLMPolicy):
    """DeepSeek switches its reasoning on with ``thinking``; it has no reasoning effort."""

    def template_kwargs(self) -> dict[str, object]:
        return {"thinking": True}


def file_text(content: bytes) -> str:
    text = content.decode(errors="replace")
    if len(text) <= FILE_CHARS:
        return text
    return f"{text[:FILE_CHARS]}\n[CUT: the first {FILE_CHARS} of {len(text)} characters]"


def task_view(task: TaskSpec) -> dict[str, object]:
    return {
        "prompt": [event.content for event in task.context.events if isinstance(event, TextMessage)],
        "answer_format": task.answer_format.model_dump(mode="json"),
        "files": {resource.path: file_text(resource_bytes(resource)) for resource in task.resources.worker},
        "grader": task.grader.model_dump(mode="json"),
    }


async def audit_all(root: str, client: GlmClient) -> list[dict[str, object]]:
    records = []
    for task_path in sorted((StoragePath(root) / "ideas/*/tasks/*/task.json").glob(), key=str):
        directory = task_path.parent
        idea = json.loads((directory.parent.parent / "idea.json").read_text())
        task = TaskSpec.model_validate_json(task_path.read_text())
        probes = [TaskSpec.model_validate_json(line) for line in (directory / "probes.jsonl").read_text().splitlines()]
        content = {
            "capability": idea["capability"],
            "task": task_view(task),
            "other_instances": [task_view(probe) for probe in probes[:PROBES_SHOWN]],
        }
        messages: list[Message] = [
            {"role": "system", "content": AUDIT_PROMPT},
            {"role": "user", "content": json.dumps(content, indent=1)},
        ]
        result = await client.complete_structured(messages, TaskAudit, DeepSeekPolicy(max_tokens=32768))
        records.append({"capability_id": idea["capability_id"], "task_id": task.id, **result.value.model_dump()})
        logger.info("%s: %s", task.id, result.value)
    return records


@click.command()
@click.argument("rsi_dir")
@click.option("--relay-job", required=True)
@click.option("--pool", type=click.Choice([pool.value for pool in Pool]), default=Pool.BULK.value, show_default=True)
def main(rsi_dir: str, relay_job: str, pool: str) -> None:
    """Audit the accepted tasks of an ``rsi`` output directory and write audit.jsonl and audit.json there."""
    configure_logging(logging.INFO)
    provider = LLMProviderConfig(**{**vars(GLM_PROVIDER), "model": DEEPSEEK_MODEL})

    async def run() -> list[dict[str, object]]:
        async with GlmClient(endpoint_source(None, relay_job, Pool(pool)), provider) as client:
            return await audit_all(rsi_dir, client)

    records = asyncio.run(run())
    root = StoragePath(rsi_dir)
    (root / "audit.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    fields = ("tests_capability", "well_posed", "reference_correct", "instances_consistent")
    summary = {"tasks": len(records), **{name: sum(bool(r[name]) for r in records) for name in fields}}
    summary["all_pass"] = sum(all(record[name] for name in fields) for record in records)
    (root / "audit.json").write_text(json.dumps(summary, indent=1))
    click.echo(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
