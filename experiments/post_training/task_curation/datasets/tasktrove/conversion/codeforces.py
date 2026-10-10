# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Codeforces and CodeElo stdin/stdout problems, some with a per-task special judge.

Each task ships ``tests/inputs/input_<n>.txt`` / ``tests/outputs/output_<n>.txt`` pairs.
Tasks without ``tests/checker.py`` use whitespace-token comparison, or float comparison
when the instruction specifies numeric-error tolerance. Tasks with a checker preserve its
decision: the launcher supports positional ``main(input, expected, got)`` and CLI-style
``main()`` reading those paths from argv. A crashing or malformed checker scores zero;
it must not silently become an exact-output comparison that rejects other valid outputs.

The agent may submit ``solution.py`` or ``solution.cpp``. The Dockerfile has no JDK, so the
converter removes the source's stale ``Solution.java`` boilerplate. :data:`_BUILD` compiles the
C++ case once and :data:`_COMMAND` runs whichever file the workspace has.
"""

from verifyit.spec import Compare, StdioSpec

from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import DOCKERFILE, INSTRUCTION, TaskFiles
from experiments.post_training.task_curation.datasets.tasktrove.conversion.result import (
    ConvertedTask,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.stdio_cases import (
    case_files_from_dirs,
    comparison_from_instruction,
    hidden_case_rejection,
)

CHECKER_PATH = "tests/checker.py"
JUDGE_PATH = "tests/judge.py"

_SUBMISSION_REPLACEMENTS = {
    "`/app/solution.py` (Python 3), `/app/solution.cpp` (C++17), or `/app/Solution.java` (Java)": (
        "`/app/solution.py` (Python 3) or `/app/solution.cpp` (C++17)"
    ),
    "`/app/solution.py` (or solution.cpp/Solution.java for C++/Java)": "`/app/solution.py` or `/app/solution.cpp`",
}

_BUILD = (
    "if [ -f solution.py ]; then exit 0; "
    "elif [ -f solution.cpp ]; then g++ -O2 -std=c++17 -o solution_bin solution.cpp; "
    "else exit 1; fi"
)
_COMMAND = (
    "bash -c 'if [ -f solution.py ]; then exec python3 solution.py; "
    "elif [ -f solution_bin ]; then exec ./solution_bin; else exit 1; fi'"
)

_JUDGE_PY = r"""import importlib.util
import inspect
import sys
from pathlib import Path


def main() -> None:
    paths = sys.argv[1:]
    checker_path = Path(__file__).with_name("checker.py")
    sys.argv = [str(checker_path), *paths]
    spec = importlib.util.spec_from_file_location("cf_checker", checker_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    signature = inspect.signature(module.main)
    try:
        signature.bind(*paths)
    except TypeError:
        signature.bind()
        module.main()
    else:
        module.main(*paths)


main()
"""


def convert_codeforces(task: TaskFiles) -> ConvertedTask | Rejected:
    data_files = case_files_from_dirs(task)
    instruction = task.text(INSTRUCTION)
    for source, replacement in _SUBMISSION_REPLACEMENTS.items():
        instruction = instruction.replace(source, replacement)
    rejection = hidden_case_rejection(data_files, instruction)
    if rejection is not None:
        return rejection

    checker = task.files.get(CHECKER_PATH)
    special_judge = None
    if checker is not None:
        data_files[JUDGE_PATH] = _JUDGE_PY.encode()
        data_files[CHECKER_PATH] = checker
        special_judge = "judge.py"
    else:
        outputs = [v for path, v in data_files.items() if path.rsplit("/", 1)[-1].startswith("output_")]
        if outputs and all(not v.split() for v in outputs):
            return Rejected(ConvertStatus.NULL_GRADER, "every expected output is empty and there is no special judge")

    tags = ("code", "competitive-programming", "stdio", "codeforces")
    if special_judge is not None:
        tags = (*tags, "special-judge")
    compare, float_tolerance = comparison_from_instruction(instruction, Compare.TOKENS)

    return ConvertedTask(
        instruction=instruction,
        spec=StdioSpec(
            command=_COMMAND,
            build=_BUILD,
            compare=compare,
            special_judge=special_judge,
            float_tolerance=float_tolerance,
        ),
        dockerfile=task.text(DOCKERFILE),
        tags=tags,
        language="python",
        data_files=data_files,
    )
