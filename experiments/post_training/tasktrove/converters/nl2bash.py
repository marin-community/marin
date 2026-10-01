# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Natural language to bash, compared against an oracle command's captured output.

The agent runs a shell command of its own choosing and writes its combined stdout/stderr to
``/output/command_capture.txt``; the old grader (``tests/verifier.py``) compared that against a
pre-captured ``expected_output`` from ``tests/verifier_data.json`` as a normalized,
order-insensitive multiset of "records" (one per output line): ANSI codes, a leading
``/workspace/`` or ``./`` path prefix, and a trailing size unit are stripped, and every expected
record must appear in the actual output, with no extra record that looks like an error. That is
not a plain normalized string match (records are compared as a multiset, extra non-error output
is tolerated), so this maps onto :class:`ScriptSpec`: a checker under ``tests/`` calls
the shared comparison in verifyit. NUL-delimited captures are compared as records too.

Every task in this source also ships a root ``setup_files/`` directory instruction.md tells the
agent to run (``bash /setup_files/setup_seeds.sh``) before starting, mirrored under
``tests/setup_files/`` for the oracle. Both are forwarded unchanged.
"""

import json

from verifyit.spec import ScriptSpec

from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.converters.nemotron_data import verifier_data
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, SOLVE_SH, TaskFiles

CHECKER_NAME = "nl2bash_check.py"
DATA_NAME = "nl2bash_expected.json"
OUTPUT_PATH = "/output/command_capture.txt"
WORKSPACE = "/workspace"
"""Every task's Dockerfile sets ``WORKDIR /workspace``; ``ScriptSpec.workspace`` must match it
exactly (rather than the default ``/app``, which nothing creates in this image) so the checker
process has a cwd that exists."""

_BROKEN_SEED_CALL = "bash /tests/setup_seeds.sh"
_FIXED_SEED_CALL = "bash /tests/setup_files/setup_seeds.sh"
"""Every shipped oracle calls the seed script at the wrong path (it lives at
``tests/setup_files/setup_seeds.sh``, not ``tests/setup_seeds.sh``); without this fix the oracle
solution silently fails to seed the workspace before capturing its output."""

_CHECKER_TEMPLATE = '''\
#!/usr/bin/env python3
"""Score a captured shell session's output against one task's oracle output.

Reads the expected output from __DATA_NAME__ beside this script (under
``$VERIFYIT_TESTS_DIR``), compares it against the capture file named by its one argument,
and reports the reward through ``$VERIFYIT_LOGS_DIR/reward.json``. The comparison is a
normalized, order-insensitive multiset of "records" (one per output line; ANSI codes, a leading
``/workspace/`` or ``./`` prefix, a trailing size unit, and repeated whitespace are stripped):
every expected record must appear in the actual output, and no extra record may look like an
error. It uses verifyit rather than the original dataset's grader.
"""

import json
import os
import sys
from pathlib import Path

from verifyit.modes.grade_nl2bash import score_capture

OUTPUT = Path(sys.argv[1])


def main():
    tests_dir = Path(os.environ["VERIFYIT_TESTS_DIR"])
    logs_dir = Path(os.environ["VERIFYIT_LOGS_DIR"])
    data = json.loads((tests_dir / "__DATA_NAME__").read_text())
    expected = data["expected_output"]

    if not OUTPUT.exists():
        reward, errors = 0, [f"missing output: {OUTPUT}"]
    else:
        reward, errors = score_capture(OUTPUT.read_text(errors="replace"), expected)

    for error in errors:
        print(error, file=sys.stderr)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "reward.json").write_text(json.dumps({"reward": reward}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

CHECKER_PY = _CHECKER_TEMPLATE.replace("__DATA_NAME__", DATA_NAME)


def convert_nl2bash(task: TaskFiles) -> ConvertedTask | Rejected:
    """NL-to-bash: ``{"expected_output": "..."}`` captured from the oracle command's stdout+stderr."""
    data = verifier_data(task)
    expected = data.get("expected_output")
    if not isinstance(expected, str):
        return Rejected(ConvertStatus.NULL_GRADER, f"expected_output missing or not a string: {type(expected)}")
    solve = task.get_text(SOLVE_SH)
    if not solve:
        return Rejected(ConvertStatus.NULL_GRADER, "no oracle solution/solve.sh shipped")

    data_files: dict[str, bytes] = {
        f"tests/{CHECKER_NAME}": CHECKER_PY.encode(),
        f"tests/{DATA_NAME}": json.dumps({"expected_output": expected}).encode(),
    }
    data_files.update(task.under("setup_files/"))
    data_files.update(task.under("tests/setup_files/"))

    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=ScriptSpec(path=CHECKER_NAME, args=(OUTPUT_PATH,), workspace=WORKSPACE),
        dockerfile=task.text(DOCKERFILE),
        tags=("shell", "bash", "nl2bash", "terminal", "dcagent2"),
        language="bash",
        data_files=data_files,
        solution_files={SOLVE_SH: solve.replace(_BROKEN_SEED_CALL, _FIXED_SEED_CALL).encode()},
    )


CONVERTER = Converter(
    name="nl2bash",
    keys=(ConverterKey("shell-cmd", frozenset({"tests/test.sh", "tests/verifier.py"})),),
    convert=convert_nl2bash,
)
