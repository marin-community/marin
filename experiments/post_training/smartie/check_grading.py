# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay labeled actions through the deployed Smartie judge route without GPUs."""

import asyncio
import hashlib
import json
import os
from dataclasses import asdict
from functools import partial

import click
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.contracts import ModelRequest, ModelTurn
from rolloutengine.engine import ShellboxRolloutEngine
from rolloutengine.spec import LoweredTaskSpec
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from taskcompendium.models import VerifyitGrader
from verifyit.modes.grade_judge import JudgeConnection

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, resolve_glm_base_url
from experiments.post_training.task_curation.datasets.nemotron_ultra.calibrate_tool_judge import CASES


async def replay(message: dict, request: ModelRequest) -> ModelTurn:
    # Tokens are placeholders: this probe measures grading, not generation or loss.
    return ModelTurn(message, (1,), (2,), None, "stop")


async def check_grading(data: StoragePath, output: StoragePath, connection: JudgeConnection) -> None:
    """Check admitted task identities and grade positive and negative recorded actions."""
    if output.exists():
        raise FileExistsError(f"Grading evidence must be new: {output}")
    manifest_bytes = (data / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    tasks = {}
    split_counts = {}
    for split in ("train", "validation"):
        with (data / f"{split}.parquet").open("rb") as stream:
            rows = pq.read_table(stream).to_pylist()
        lowered = [LoweredTaskSpec.model_validate_json(row["lowered_task_spec"]) for row in rows]
        ids = [task.task.id for task in lowered]
        if len(set(ids)) != len(ids) or set(ids) != set(manifest[f"{split}_ids"]):
            raise ValueError(f"{split} identities differ from the conversion manifest")
        split_counts[split] = len(ids)
        for task in lowered:
            if isinstance(task.task.grader, VerifyitGrader) and task.task.grader.mode == "judge":
                row = int(task.task.source.row.rsplit(":", 1)[-1])
                if row in tasks:
                    raise ValueError(f"Duplicate tool-call source row {row}")
                tasks[row] = task

    results = []
    for case in (case for case in CASES if case.source_row == 582):
        task = tasks[case.source_row]
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": case.name,
                    "type": "function",
                    "function": {"name": case.function, "arguments": json.dumps(case.arguments)},
                }
            ],
        }

        record = await ShellboxRolloutEngine(
            partial(replay, message), {"daytona": DaytonaMachineFactory()}, judge_connection=connection
        ).run(task)
        result = {
            "case": asdict(case),
            "task_id": task.task.id,
            "grade": asdict(record.grade),
            "messages": record.messages,
            "expected_reward": case.expected_reward,
            "correct": record.grade.reward == case.expected_reward,
        }
        results.append(result)
        click.echo(json.dumps({"case": case.name, "reward": record.grade.reward, "correct": result["correct"]}))
    report = {
        "data": str(data),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "split_counts": split_counts,
        "generation": "replayed labeled actions; no model inference",
        "results": results,
    }
    output.write_text(json.dumps(report, indent=2))
    if len(results) < 2 or not all(result["correct"] for result in results):
        raise ValueError(f"Live grading check failed; evidence: {output}")


@click.command()
@click.option("--data", required=True)
@click.option("--output", required=True)
@click.option("--relay-job", required=True)
def main(data: str, output: str, relay_job: str) -> None:
    connection = JudgeConnection(resolve_glm_base_url(relay_job), os.environ[GLM_BULK_TOKEN_ENV])
    asyncio.run(check_grading(StoragePath(data), StoragePath(output), connection))


if __name__ == "__main__":
    main()
