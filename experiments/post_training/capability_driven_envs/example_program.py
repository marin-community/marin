# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A task program for ``d43.culinary.scaling``, shown to GLM as the example of what to write.

Run as ``python3 example_program.py <task id> <output path>``: it draws a recipe-scaling problem from
the task id, computes the answer, and writes a TaskSpec whose verifyit grader checks that answer.
"""

import base64
import random
import sys
from pathlib import Path

from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    InlineFile,
    PlainText,
    ResourceGroups,
    Source,
    TaskResource,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import NumericSpec

WORKDIR = "/workspace"
RECIPE_PATH = "workspace/recipe.txt"
INGREDIENTS = ("flour", "butter", "sugar", "milk", "onions", "rice")


def task_spec(task_id: str) -> TaskSpec:
    rng = random.Random(task_id)
    portions = rng.choice((4, 6, 8, 10, 12))
    target = rng.choice([n for n in range(2, 61) if n != portions])
    names = rng.sample(INGREDIENTS, 3)
    # Every weight is a whole multiple of the yield, so the scaled weight is exact.
    grams = {name: portions * rng.randint(5, 60) for name in names}
    asked = rng.choice(names)
    answer = grams[asked] * target // portions

    recipe = "\n".join([f"Yield: {portions} portions", *(f"{name}: {weight} g" for name, weight in grams.items())])
    instruction = (
        f"The recipe in {WORKDIR}/recipe.txt states its yield. Scale it to {target} portions, keeping "
        f"every ratio. How many grams of {asked} does the scaled recipe use? Reply with the number alone."
    )
    grader = verifyit_package(NumericSpec(expected=str(answer), tolerance_abs=0.0, tolerance_rel=0.0))
    recipe_file = TaskResource(
        path=RECIPE_PATH, source=InlineFile(content_base64=base64.b64encode(f"{recipe}\n".encode()).decode())
    )
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
        source=Source(dataset="taskforge", revision="example", row=task_id, importer_revision="example"),
        resources=ResourceGroups(worker=(recipe_file,), verifier=grader.resources),
        tags=("d43.culinary.scaling",),
    )


if __name__ == "__main__":
    task_id, output_path = sys.argv[1:]
    Path(output_path).write_text(task_spec(task_id).model_dump_json())
