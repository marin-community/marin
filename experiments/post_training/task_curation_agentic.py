# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise NeMo actions, a TaskTrove shell exemplar, and generated shell/calendar tasks.

Requires GLM_BULK_TOKEN and an existing local Docker image. No image is pulled.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import JobName
from marin.inference.openai_batch import OpenAIBatchClient
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.machine import DockerImage, MachineSpec, NetworkPolicy
from taskcompendium.grading import GradingAttempt
from taskcompendium.harbor.protocol import assistant_message
from taskcompendium.models import ConversationInput, TaskSpec
from taskcompendium.pipeline.batches import BatchClient
from taskcompendium.pipeline.datasets import calendar, nemo_actions, shell_files
from taskcompendium.pipeline.models import DatasetRecipe, SnapshotSource
from taskcompendium.pipeline.review import BatchReviewer
from taskcompendium.pipeline.runner import run_pipeline
from taskcompendium.pipeline.sources import source_rows
from taskcompendium.pipeline.verification import PLAIN
from taskcompendium.runtime.calendar import CalendarFactory
from taskcompendium.runtime.checks import episode_suite
from taskcompendium.runtime.episode import run_episode
from taskcompendium.runtime.models import ActorTask, EnvironmentFactory, Termination
from taskcompendium.runtime.shell import ShellFactory
from taskcompendium.submission import conversation_messages
from taskcompendium.verifier_registry import resolve_verifier

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL
from experiments.post_training.tasktrove.converters.converted_task import ConvertedTask
from experiments.post_training.tasktrove.converters.nl2bash import DATA_NAME, convert_nl2bash
from experiments.post_training.tasktrove.taskbinary import read_task_binary

FIXTURE = Path(__file__).parent / "tasktrove/fixtures/nl2bash.tar.gz"


@dataclass(frozen=True)
class PilotSource:
    recipe: DatasetRecipe
    limit: int | None
    factory: EnvironmentFactory | None


def nl2bash_snapshot(path: Path) -> DatasetRecipe:
    """Adapt the existing cleanup output without importing experiments from the library."""
    binary = FIXTURE.read_bytes()
    converted = convert_nl2bash(read_task_binary(binary))
    if not isinstance(converted, ConvertedTask):
        raise ValueError(f"TaskTrove exemplar conversion failed: {converted}")
    row = {
        "instruction": converted.instruction,
        "reference_script": converted.solution_files["solution/solve.sh"].decode(),
        "expected_output": json.loads(converted.data_files[f"tests/{DATA_NAME}"])["expected_output"],
        "public_files": {
            f"/{name}": base64.b64encode(data).decode()
            for name, data in converted.data_files.items()
            if name.startswith("setup_files/")
        },
        "control_files": {
            f"/{name}": base64.b64encode(data).decode()
            for name, data in converted.data_files.items()
            if name.startswith("tests/setup_files/")
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n")
    return replace(
        shell_files.recipe,
        name="tasktrove-nl2bash-exemplar",
        version="tasktrove-nl2bash-v1",
        source=SnapshotSource(
            "TaskTrove/DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
            hashlib.sha256(binary).hexdigest(),
            "fixture",
            "train",
            str(path),
        ),
    )


def pilot_recipes(output: Path, image: str, max_steps: int) -> tuple[PilotSource, ...]:
    digest = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image], check=True, text=True, capture_output=True
    ).stdout.strip()
    factory = ShellFactory(
        machine_factory=DockerMachineFactory(),
        machine_spec=MachineSpec(DockerImage(digest), network=NetworkPolicy.DENY, memory_mb=512),
        backend_identity={"backend": "docker", "image": digest},
        command_timeout=15,
        output_limit_bytes=65536,
    )
    shell_suite = episode_suite(factory, max_steps=max_steps)
    return (
        PilotSource(nemo_actions.recipe, None, None),
        PilotSource(replace(shell_files.recipe, check_suite=shell_suite), None, factory),
        PilotSource(
            replace(calendar.recipe, check_suite=episode_suite(CalendarFactory(), max_steps=max_steps)),
            None,
            CalendarFactory(),
        ),
        PilotSource(replace(nl2bash_snapshot(output / "sources/nl2bash.jsonl"), check_suite=shell_suite), 1, factory),
    )


@dataclass
class GLMActor:
    """One bounded solving probe through the existing bulk transport."""

    client: BatchClient
    output: Path
    max_tokens: int
    turn: int = field(default=0, init=False)

    def respond(self, task: ActorTask, events):
        directory = self.output / f"turn-{self.turn}"
        directory.mkdir(parents=True, exist_ok=True)
        body = {
            "model": GLM_MODEL,
            "messages": conversation_messages(ConversationInput(events=tuple(events))),
            "tools": [{"type": "function", "function": tool.model_dump(exclude_none=True)} for tool in task.tools],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"reasoning_effort": "low"},
        }
        request = {"custom_id": f"{task.id}-{self.turn}", "method": "POST", "url": "/v1/chat/completions", "body": body}
        (directory / "request.json").write_text(json.dumps(request) + "\n")
        submission = self.client.submit([request], "task-curation-solving-probe.jsonl")
        (directory / "batch-state.json").write_text(
            json.dumps({"batch_id": submission.batch_id, "file_id": submission.file_id})
        )
        batch = self.client.wait(submission.batch_id, poll_seconds=5)
        output = self.client.output(batch)
        (directory / "output.jsonl").write_text(output.output)
        if output.errors is not None:
            (directory / "errors.jsonl").write_text(output.errors)
        rows = [json.loads(line) for line in output.output.split("\n") if line.strip()]
        if len(rows) != 1 or rows[0]["custom_id"] != request["custom_id"]:
            raise RuntimeError("Solving probe batch membership mismatch")
        response = rows[0].get("response")
        if response is None or response["status_code"] != 200:
            raise RuntimeError("Solving probe request failed")
        choices = response["body"]["choices"]
        if len(choices) != 1 or choices[0]["finish_reason"] not in {"tool_calls", "stop"}:
            raise RuntimeError("Solving probe response incomplete")
        self.turn += 1
        try:
            return assistant_message(choices[0]["message"])
        except ValueError as error:
            raise RuntimeError(f"Malformed solving probe assistant response: {error}") from error


def solve_probe(
    task: TaskSpec, factory: EnvironmentFactory, reviewer: BatchReviewer, output: Path, max_steps: int
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    actor = GLMActor(reviewer.client, output, reviewer.max_tokens)
    rollout = asyncio.run(run_episode(task, actor, factory, max_steps=max_steps, control="glm-independent"))
    (output / "rollout.json").write_text(rollout.model_dump_json(indent=2) + "\n")
    result = {
        "task_id": task.id,
        "termination": rollout.termination.value,
        "turns": actor.turn,
        "reward": None,
        "model": GLM_MODEL,
        "model_revision": reviewer.model_revision,
        "runtime": factory.identity,
        "max_steps": max_steps,
        "max_tokens": actor.max_tokens,
        "detail": rollout.detail,
    }
    if rollout.termination == Termination.FINAL_MESSAGE:
        grade = resolve_verifier(task.verifier).grade(GradingAttempt(PLAIN, rollout.events, rollout.evidence()))
        result.update({"status": grade.status.value, "reward": grade.reward, "detail": grade.error})
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--relay-job", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--image", required=True, help="Existing local Docker image, resolved to an immutable ID")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--solve-per-source",
        type=int,
        default=0,
        help="Independent GLM episodes per executable source; kept separate from filtering",
    )
    args = parser.parse_args()
    recipes = pilot_recipes(args.output, args.image, args.max_steps)
    with connect_controller(cluster_name=args.cluster) as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials) as client:
            serving = client.resolver_for_job(JobName.from_string(args.relay_job)).resolve(GLM_MODEL).first()
            base_url = (args.base_url or serving.url).rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            reviewer = BatchReviewer(
                OpenAIBatchClient(base_url, os.environ[GLM_BULK_TOKEN_ENV]),
                GLM_MODEL,
                args.relay_job,
                max_tokens=args.max_tokens,
            )
            for source in recipes:
                recipe = source.recipe
                limit = source.limit or args.limit
                manifest = run_pipeline(
                    recipe,
                    source_rows(recipe.source, limit),
                    output_path=args.output / recipe.name,
                    limit=limit,
                    reviewer=reviewer,
                )
                print(
                    json.dumps(
                        {
                            "dataset": recipe.name,
                            "input": manifest["input_rows"],
                            "normalized": manifest["normalized_rows"],
                            "checks": manifest["check_statuses"],
                            "reviewed": manifest["reviewed_rows"],
                            "dispositions": manifest["dispositions"],
                        }
                    ),
                    flush=True,
                )
                if source.factory is not None and args.solve_per_source:
                    with (args.output / recipe.name / "normalized.jsonl").open() as stream:
                        for index, line in enumerate(stream):
                            if index >= args.solve_per_source:
                                break
                            task = TaskSpec.model_validate_json(line)
                            result = solve_probe(
                                task,
                                source.factory,
                                reviewer,
                                args.output / recipe.name / "solver" / str(index),
                                args.max_steps,
                            )
                            print(json.dumps({"dataset": recipe.name, "solver": result}), flush=True)


if __name__ == "__main__":
    main()
