# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron competitive coding stdin/stdout tasks."""

from taskcompendium.convert.tasktrove import DOCKERFILE, INSTRUCTION, TaskFiles
from taskcompendium.convert.tasktrove_converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from taskcompendium.convert.tasktrove_nemotron_data import verifier_data
from taskcompendium.convert.tasktrove_stdio_cases import SOLUTION_COMMAND, case_files, hidden_case_rejection
from verifyit.spec import Compare, StdioSpec


def convert_nemotron_competitive(task: TaskFiles) -> ConvertedTask | Rejected:
    """Stdin/stdout competitive programming: ``{"inputs": [...], "outputs": [...]}`` case pairs.

    The old grader ran ``python3 /app/solution.py`` once per case, fed each ``inputs[i]`` on stdin,
    and compared stdout to ``outputs[i]`` after stripping trailing whitespace per line and trailing
    blank lines — exactly ``Compare.EXACT``. All cases must pass, matching the old all-or-nothing
    scoring.
    """
    data = verifier_data(task)
    inputs, outputs = data.get("inputs"), data.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs):
        got_in = len(inputs) if isinstance(inputs, list) else type(inputs).__name__
        got_out = len(outputs) if isinstance(outputs, list) else type(outputs).__name__
        return Rejected(ConvertStatus.NULL_GRADER, f"invalid inputs/outputs: {got_in} inputs, {got_out} outputs")
    if not all(isinstance(x, str) for x in [*inputs, *outputs]):
        return Rejected(ConvertStatus.NULL_GRADER, "inputs/outputs must be strings")
    cases = case_files(inputs, outputs)
    instruction = task.text(INSTRUCTION)
    rejection = hidden_case_rejection(cases, instruction)
    if rejection is not None:
        return rejection
    return ConvertedTask(
        instruction=instruction,
        spec=StdioSpec(command=SOLUTION_COMMAND, compare=Compare.EXACT),
        dockerfile=task.text(DOCKERFILE),
        tags=("code", "competitive-programming", "stdio", "nemotron"),
        language="python",
        data_files=cases,
    )


CONVERTER = Converter(
    name="nemotron_competitive",
    keys=(
        ConverterKey(
            "competitive-programming",
            frozenset({"tests/test.sh", "tests/validate_verifier_data.py", "tests/verifier.py"}),
        ),
    ),
    convert=convert_nemotron_competitive,
)
