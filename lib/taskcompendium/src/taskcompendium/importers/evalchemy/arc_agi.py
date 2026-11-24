# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ARC-AGI-2 public evaluation prompts, one task per test input.

The prompt is ARC Prize's ``prompts/system_prompt.txt`` filled as in ``prompts/prompt_manager.py``
(arcprize/arc-agi-benchmarking, MIT, at 78f2d3eacb86587d1438ab14f08fb56811eb1f1c): every training pair,
then one test input, with grids written by ``json.dumps``. Each row carries a weight of one over its
task's test-input count, so the weighted mean averages over tasks as ARC Prize's score does.
"""

import json
import tarfile
from collections.abc import Iterator
from pathlib import PurePosixPath
from typing import Any

from rigging.filesystem.storage_path import StoragePath

from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

ARC_AGI_PROMPT = (
    "You are participating in a puzzle solving competition. You are an expert at solving puzzles.\n\n"
    "Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or "
    "transformation in the training examples that maps the input to the output, then apply that pattern to "
    "the test input to give a final output.\n\n"
    "Respond in the format of the training output examples\n\n"
    "--Training Examples--\n{training_examples}\n--End of Training Examples--\n\n"
    "--Test Input--\n{test_input}\n--End of Test Input--\n\n"
    "Your response:"
)
ARC_AGI_EVALUATION_DIR = ("data", "evaluation")


def arc_agi_prompt(train: list[dict[str, Any]], test_input: list[list[int]]) -> str:
    examples = "".join(
        f"--Example {index}-- \n\n INPUT: \n\n" + json.dumps(pair["input"]) + "\n\n"
        "OUTPUT: \n\n" + json.dumps(pair["output"]) + "\n\n"
        for index, pair in enumerate(train)
    )
    return ARC_AGI_PROMPT.format(training_examples=examples, test_input=json.dumps(test_input))


def _member_json(archive: tarfile.TarFile, member: tarfile.TarInfo) -> Any:
    handle = archive.extractfile(member)
    assert handle is not None, f"{member.name} is not a regular file"
    return json.load(handle)


def read_arc_agi_2(file: StoragePath, _context: ConversionContext) -> Iterator[dict[str, Any]]:
    """Yield one row per test input of the repository archive's public evaluation tasks, in task-id order.

    Grids stay JSON text until conversion so integer cells survive tabular staging.
    """
    with file.open("rb") as stream, tarfile.open(fileobj=stream, mode="r:gz") as archive:
        tasks = {
            PurePosixPath(member.name).stem: _member_json(archive, member)
            for member in archive.getmembers()
            if member.isfile()
            and PurePosixPath(member.name).parent.parts[1:] == ARC_AGI_EVALUATION_DIR
            and member.name.endswith(".json")
        }
    for task_id in sorted(tasks):
        task = tasks[task_id]
        for index, test in enumerate(task["test"]):
            yield {
                "task_id": task_id,
                "test_index": index,
                "test_count": len(task["test"]),
                "train": json.dumps(task["train"]),
                "test_input": json.dumps(test["input"]),
                "test_output": json.dumps(test["output"]),
            }


def convert_arc_agi_2(row: RawRow, context: ConversionContext) -> TaskSpec:
    task = conversation_task(
        row,
        events=(
            TextMessage(
                role="user", content=arc_agi_prompt(json.loads(row.data["train"]), json.loads(row.data["test_input"]))
            ),
        ),
        package=source_package(
            "arc_agi",
            {"expected": json.loads(row.data["test_output"]), "weight": 1 / row.data["test_count"]},
            environment=required_grader_environment(context),
        ),
        evidence=row.data,
    )
    return task.model_copy(update={"tags": ("eval", "source:arcprize", "benchmark:ARC-AGI-2")})


def arc_agi_2_item(task: TaskSpec) -> EvaluationItem:
    return chat_item(task, response_selection="final_only")
