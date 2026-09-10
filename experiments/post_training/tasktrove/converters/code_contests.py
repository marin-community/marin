# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CodeContests stdin/stdout problems.

The only per-task file is ``tests/test_data.json``: parallel ``inputs``/``outputs`` string lists.
The old grader ran ``python3 /app/solution.py`` once per case with the case's input on stdin and
compared stdout to the expected output line for line, so this maps directly onto ``stdio``.
"""

import json

from tasktrove_verify.spec import Compare, StdioSpec

from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.converters.stdio_cases import case_files, hidden_case_rejection
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, TaskFiles

TEST_DATA = "tests/test_data.json"
PER_CASE_TIMEOUT = 30.0
"""The old grader's SOLUTION_TIMEOUT_SEC, uniform across the source."""


def convert_code_contests(task: TaskFiles) -> ConvertedTask | Rejected:
    raw = task.get_text(TEST_DATA)
    if raw is None:
        return Rejected(ConvertStatus.NULL_GRADER, f"{TEST_DATA} missing")
    data = json.loads(raw)
    inputs, outputs = data.get("inputs", []), data.get("outputs", [])
    if len(inputs) != len(outputs):
        return Rejected(ConvertStatus.NULL_GRADER, f"{len(inputs)} inputs but {len(outputs)} outputs")
    cases = case_files([str(i) for i in inputs], [str(o) for o in outputs])
    instruction = task.text(INSTRUCTION)
    rejection = hidden_case_rejection(cases, instruction)
    if rejection is not None:
        return rejection
    return ConvertedTask(
        instruction=instruction,
        spec=StdioSpec(command="python3 /app/solution.py", compare=Compare.EXACT, per_case_timeout=PER_CASE_TIMEOUT),
        dockerfile=task.text(DOCKERFILE),
        tags=("code", "competitive-programming", "stdio", "code-contests"),
        language="python",
        data_files=cases,
    )


CONVERTER = Converter(
    name="code_contests",
    keys=(ConverterKey("competitive-programming", frozenset({"tests/test.sh"})),),
    convert=convert_code_contests,
)
