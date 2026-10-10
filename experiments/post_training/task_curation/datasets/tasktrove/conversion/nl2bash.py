# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Natural language to bash, compared against an oracle command's captured output.

The agent runs a shell command of its own choosing and writes its combined stdout/stderr to
``/output/command_capture.txt``; the old grader (``tests/verifier.py``) compared that against a
pre-captured ``expected_output`` from ``tests/verifier_data.json`` as a normalized,
order-insensitive multiset of "records" (one per output line): ANSI codes, a leading
``/workspace/`` or ``./`` path prefix, and a trailing size unit are stripped, and every expected
record must appear in the actual output. A self-contained :class:`ScriptSpec` checker preserves
that comparison and its tolerance for harmless extra records. It rejects extra error records and
contradictory integer counts for the same normalized filename, total, or standalone count.

Every task in this source also ships a root ``setup_files/`` directory instruction.md tells the
agent to run (``bash /setup_files/setup_seeds.sh``) before starting, mirrored under
``tests/setup_files/`` for the oracle. Both are forwarded unchanged.
"""

import json

from verifyit.spec import ScriptSpec

from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import (
    DOCKERFILE,
    INSTRUCTION,
    SOLVE_SH,
    TaskFiles,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.nemotron_data import verifier_data
from experiments.post_training.task_curation.datasets.tasktrove.conversion.result import (
    ConvertedTask,
    ConvertStatus,
    Rejected,
)

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
every expected record must appear in the actual output, and extra records may neither look like
errors nor contradict expected counts for the same target. Self-contained: it does not import the
original dataset's grader.
"""

import collections
import json
import os
import re
import sys
from pathlib import Path

OUTPUT = Path(sys.argv[1])
ANSI = re.compile(r"\\x1b\\[[0-?]*[ -/]*[@-~]")
ERROR = re.compile(r"(?i)\\b(?:error|failed|failure|no such file|not found|permission denied|traceback)\\b")
UNIT = re.compile(r"(?i)\\s+(?:bytes?|kb|kib|mb|mib|gb|gib)\\s*$")
COUNT = re.compile(r"^([+-]?\\d+)(?: (.+))?$")


def _record(line):
    value = ANSI.sub("", line).strip()
    value = re.sub(r"(?<!\\S)/workspace/", "", value)
    value = re.sub(r"(?<!\\S)\\./", "", value)
    value = UNIT.sub("", value)
    return re.sub(r"\\s+", " ", value).strip()


def _records(text):
    return [record for line in text.replace("\\0", "\\n").splitlines() if (record := _record(line))]


def _normalized_count(value):
    digits = value.lstrip("+-").lstrip("0") or "0"
    return ("-" if value.startswith("-") and digits != "0" else "") + digits


def _score(actual, expected):
    expected_records = collections.Counter(_records(expected))
    actual_records = collections.Counter(_records(actual))
    if not expected_records:
        return (1, []) if not actual_records else (0, ["expected empty output"])
    missing = expected_records - actual_records
    if missing:
        return 0, [f"missing expected records: {dict(missing)}"]
    extras = actual_records - expected_records
    standalone_count = expected_records.total() == 1
    expected_counts = collections.defaultdict(set)
    for record in expected_records:
        match = COUNT.fullmatch(record)
        if match:
            count, target = match.groups()
            if target is None and not standalone_count:
                continue
            expected_counts[target].add(_normalized_count(count))
    for record in extras:
        if ERROR.search(record):
            return 0, [f"unexpected error record: {record}"]
        match = COUNT.fullmatch(record)
        if match:
            count, target = match.groups()
            if target in expected_counts and _normalized_count(count) not in expected_counts[target]:
                return 0, [f"contradictory count record: {record}"]
    return 1, []


def main():
    tests_dir = Path(os.environ["VERIFYIT_TESTS_DIR"])
    logs_dir = Path(os.environ["VERIFYIT_LOGS_DIR"])
    data = json.loads((tests_dir / "__DATA_NAME__").read_text())
    expected = data["expected_output"]

    if not OUTPUT.exists():
        reward, errors = 0, [f"missing output: {OUTPUT}"]
    else:
        reward, errors = _score(OUTPUT.read_text(errors="replace"), expected)

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
        solution_files={SOLVE_SH: solve.replace(_BROKEN_SEED_CALL, _FIXED_SEED_CALL).encode()} if solve else {},
    )
