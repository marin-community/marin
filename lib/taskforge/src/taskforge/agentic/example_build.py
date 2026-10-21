# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An example build program that the agent gets in its workspace. The ``build`` tool runs a program like it.

Run as ``python3 example_build.py <task id> <output path>``, the program draws a stock-ledger problem from the task
id, computes the answer, and writes a TaskSpec: the ledger is a file in the solver's ShellSim workspace, and a
verifyit grader holds the answer, which the solver never sees.
"""

import random
import sys
from pathlib import Path

from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import NumericSpec

WORKDIR = "/workspace"
ITEMS = ("bolts", "nuts", "washers", "hinges", "brackets")


def task_spec(task_id: str) -> TaskSpec:
    rng = random.Random(task_id)
    item = rng.choice(ITEMS)
    lines, stock = ["day,item,change"], 0
    for day in range(1, rng.randint(12, 20)):
        name = rng.choice(ITEMS)
        change = rng.randint(-30, 60)
        lines.append(f"{day},{name},{change:+d}")
        if name == item:
            stock += change
    instruction = (
        f"{WORKDIR}/ledger.csv lists the stock changes of a store, one line for each change. The stock of every "
        f"item was 0 before day 1. What is the stock of {item} after the last day? Reply with the number alone."
    )
    grader = verifyit_package(NumericSpec(expected=str(stock), tolerance_abs=0.0, tolerance_rel=0.0))
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem"),
            command_semantics=CommandSemantics.SHELL_SIMULATOR,
            working_directory=WORKDIR,
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader.grader,
        source=Source(dataset="taskforge-agentic", revision="example", row=task_id, importer_revision="example"),
        resources=ResourceGroups(
            worker=(inline_resource("workspace/ledger.csv", "\n".join(lines).encode() + b"\n"),),
            verifier=grader.resources,
        ),
        tags=("example.stock-ledger",),
    )


if __name__ == "__main__":
    task_id, output_path = sys.argv[1:]
    Path(output_path).write_text(task_spec(task_id).model_dump_json())
