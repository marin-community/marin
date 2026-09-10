# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Codeforces and CodeElo stdin/stdout problems, some with a per-task special judge.

Each task ships ``tests/inputs/input_<n>.txt`` / ``tests/outputs/output_<n>.txt`` pairs and a
fixed ``tests/judge.py`` launcher. The launcher always falls back to a whitespace-normalized exact
match, which is what plain ``stdio`` token comparison already does, so only the ~16% of tasks that
also ship a per-task ``tests/checker.py`` need ``special_judge``: :data:`_JUDGE_PY` adapts the
shipped launcher to the ``(input, expected, got)`` argv the special-judge contract calls it with,
by locating ``checker.py`` next to itself instead of taking it as a fourth argument. About half of
the shipped checkers define ``main()`` with no parameters and always raise when called positionally
with three arguments; :data:`_JUDGE_PY` catches that and falls back to the normalized match, same
as the original launcher did.

The agent may submit ``solution.py`` or ``solution.cpp`` (the Dockerfile has no JDK, so
``Solution.java`` never worked in the original grader either); :data:`_BUILD` compiles the C++
case once and :data:`_COMMAND` runs whichever the workspace has.
"""

from tasktrove_verify.spec import Compare, StdioSpec

from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.converters.nemotron_data import metadata
from experiments.post_training.tasktrove.converters.stdio_cases import case_files_from_dirs, hidden_case_rejection
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, TaskFiles

"""Below this many provided input/output files a stdio grader is too easy to game by guessing;
~30% of laion__codeforces-v3 tasks fall short, most shipping only the problem statement's own
worked example. This is also the source audit's own cutoff (see ``source_verdicts.json``)."""

CHECKER_PATH = "tests/checker.py"
JUDGE_PATH = "tests/judge.py"

_BUILD = (
    "if [ -f solution.py ]; then exit 0; "
    "elif [ -f solution.cpp ]; then g++ -O2 -std=c++17 -o solution_bin solution.cpp; "
    "else exit 1; fi"
)
_COMMAND = (
    "bash -c 'if [ -f solution.py ]; then exec python3 solution.py; "
    "elif [ -f solution_bin ]; then exec ./solution_bin; else exit 1; fi'"
)

_JUDGE_PY = r"""import contextlib
import importlib.util
import io
import re
import sys
from pathlib import Path


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def main() -> None:
    input_path, expected_path, got_path = sys.argv[1], sys.argv[2], sys.argv[3]
    got = open(got_path, errors="replace").read()
    expected = open(expected_path, errors="replace").read()
    checker_path = Path(__file__).with_name("checker.py")
    if checker_path.is_file():
        try:
            spec = importlib.util.spec_from_file_location("cf_checker", checker_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                module.main(input_path, expected_path, got_path)
            lines = [line for line in buffer.getvalue().strip().splitlines() if line.strip()]
            if lines and lines[-1].strip() in ("0", "1"):
                print(lines[-1].strip())
                return
        except Exception:
            pass  # broken or argv-style checker: fall back to the normalized match
    print("1" if _norm(got) == _norm(expected) else "0")


main()
"""


def convert_codeforces(task: TaskFiles) -> ConvertedTask | Rejected:
    data_files = case_files_from_dirs(task)
    instruction = task.text(INSTRUCTION)
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

    return ConvertedTask(
        instruction=instruction,
        spec=StdioSpec(
            command=_COMMAND,
            build=_BUILD,
            compare=Compare.TOKENS,
            special_judge=special_judge,
        ),
        dockerfile=task.text(DOCKERFILE),
        tags=tags,
        language="python",
        data_files=data_files,
        metadata={k: v for k, v in metadata(task).items() if v is not None},
    )


CONVERTER = Converter(
    name="codeforces",
    keys=(ConverterKey("competitive-programming", frozenset({"tests/test.sh", "tests/judge.py"})),),
    convert=convert_codeforces,
)
