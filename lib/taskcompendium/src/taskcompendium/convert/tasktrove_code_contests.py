# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CodeContests stdin/stdout problems.

The only per-task file is ``tests/test_data.json``: parallel ``inputs``/``outputs`` string lists.
The old grader ran ``python3 /app/solution.py`` once per case with the case's input on stdin and
compared stdout to the expected output line for line after stripping outer whitespace from both.
A source-specific stdio checker preserves that comparison; prompts with an explicit numeric-error
tolerance use float comparison so the grader honors the task contract.
"""

import ast
import json
import re

from verifyit.spec import Compare, StdioSpec

from taskcompendium.convert.tasktrove import DOCKERFILE, INSTRUCTION, TaskFiles
from taskcompendium.convert.tasktrove_converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from taskcompendium.convert.tasktrove_stdio_cases import (
    SOLUTION_COMMAND,
    case_files,
    comparison_from_instruction,
    hidden_case_rejection,
)

TEST_DATA = "tests/test_data.json"
PER_CASE_TIMEOUT = 30.0
"""The old grader's timeout for tasks whose time limit is unavailable."""
NANOSECONDS_PER_SECOND = 1_000_000_000
TIME_LIMIT = re.compile(r"(?m)^- \*\*Time Limit\*\*:\s*(\{[^\n]+\}|None)\s+seconds\s*$")
OUTPUT_CHECKER = "compare_output.py"
OUTPUT_CHECKER_PY = """import sys
from pathlib import Path

_, _, expected_path, actual_path = sys.argv
expected = Path(expected_path).read_text().strip().split("\\n")
actual = Path(actual_path).read_text().strip().split("\\n")
print(int(actual == expected))
"""


def timeout_from_instruction(instruction: str) -> float:
    """Use the stated protobuf Duration, retaining the source policy when it is absent."""
    match = TIME_LIMIT.search(instruction)
    if match is None or match.group(1) == "None":
        return PER_CASE_TIMEOUT
    duration = ast.literal_eval(match.group(1))
    seconds, nanos = duration["seconds"], duration["nanos"]
    if (
        not isinstance(seconds, int)
        or not isinstance(nanos, int)
        or seconds < 0
        or not 0 <= nanos < NANOSECONDS_PER_SECOND
    ):
        raise ValueError(f"Invalid CodeContests time limit: {duration}")
    timeout = seconds + nanos / NANOSECONDS_PER_SECOND
    if timeout <= 0:
        raise ValueError(f"CodeContests time limit must be positive: {duration}")
    return timeout


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
    compare, float_tolerance = comparison_from_instruction(instruction, Compare.EXACT)
    special_judge = None
    if compare == Compare.EXACT:
        special_judge = OUTPUT_CHECKER
        cases[f"tests/{OUTPUT_CHECKER}"] = OUTPUT_CHECKER_PY.encode()
    return ConvertedTask(
        instruction=instruction,
        spec=StdioSpec(
            command=SOLUTION_COMMAND,
            compare=compare,
            per_case_timeout=timeout_from_instruction(instruction),
            float_tolerance=float_tolerance,
            special_judge=special_judge,
        ),
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
