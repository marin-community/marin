# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron reasoning-gym and ARC-AGI tasks.

One template (``family`` ``other`` in ``source_verdicts.json``) covers three Nemotron-Gym sources
that all ship the same old-grader code files (``tests/test.sh``,
``tests/validate_verifier_data.py``, ``tests/verifier.py``) but different ``tests/verifier_data.json``
shapes, so this converter routes on that shape:

- ``laion__nemotron-gym-reasoning-gym-v2``: ``{"answer": ..., "metadata": {"source_dataset": ...},
  "question": ...}``. The old grader tried the reasoning-gym library's own scorer first and fell
  back to normalized exact-match; the new task drops the fallback and grades with
  :class:`ReasoningGymSpec` directly, the library's own scoring semantics.
- ``laion__nemotron-gym-arc-agi-python-inductive-v2``: ``{"test_cases": [{"input": [[int,...]],
  "output": [[int,...]]}, ...]}``. The agent writes a ``transform()`` function; the old grader ran
  it in a subprocess against the held-out case(s). No built-in mode runs agent-authored code
  against a data file, so this ships a new grading script under ``tests/`` for :class:`ScriptSpec`
  rather than adding a pytest toolchain the task's own Dockerfile does not install.
- ``laion__nemotron-gym-arc-agi-transductive-v3``: ``{"expected_output": [[int,...], ...]}``. The
  instruction already demands a plain-text grid, one row per line, cells space-separated, so
  :class:`ExactSpec` matches it after whitespace normalization.
"""

import json
import re

from tasktrove_verify.spec import ExactSpec, ReasoningGymSpec, ScriptSpec

from experiments.post_training.tasktrove.contract import drop_dockerfile_lines
from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.converters.nemotron_data import verifier_data
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, TaskFiles

# reasoning-gym's own scorer for these two datasets compares a JSON-deserialized ``list`` (the
# entry's ``metadata["output"]``) against ``parse_board()``'s ``tuple``-of-``tuple``s return; the
# comparison is never equal regardless of the candidate, so even the gold answer scores 0.05
# instead of 1.0. Confirmed against every occurrence in the local corpus: 149 ``arc_agi`` and 133
# ``rearc`` rows out of 14259, both 100% unscorable, while every other of the 99 datasets present
# scores its own gold answer 1.0.
_UNSCORABLE_REASONING_GYM_DATASETS = frozenset({"arc_agi", "rearc"})

_OLD_REASONING_GYM_PIP_INSTALL = re.compile(r"^RUN pip install --no-cache-dir reasoning-gym")
"""The old grader imported ``reasoning_gym`` directly in the task's system Python; the new grader
installs its own copy through ``tasktrove-verify[reasoning-gym]``, so this line is dead weight."""

TRANSFORM_SCRIPT = "run_transform.py"
CASES_FILE = "cases.json"
MIN_TRANSFORM_CASES = 1


def convert_nemotron_reasoning(task: TaskFiles) -> ConvertedTask | Rejected:
    data = verifier_data(task)
    if "test_cases" in data:
        return _convert_grid_transform(task, data)
    if "expected_output" in data:
        return _convert_grid_match(task, data)
    if "answer" in data and "metadata" in data:
        return _convert_reasoning_gym(task, data)
    return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, f"unrecognized verifier_data shape: {sorted(data)}")


def _convert_reasoning_gym(task: TaskFiles, data: dict) -> ConvertedTask | Rejected:
    entry_metadata = data.get("metadata")
    source_dataset = entry_metadata.get("source_dataset") if isinstance(entry_metadata, dict) else None
    if not isinstance(source_dataset, str) or not source_dataset:
        return Rejected(ConvertStatus.NULL_GRADER, "verifier_data.metadata.source_dataset missing")
    if source_dataset in _UNSCORABLE_REASONING_GYM_DATASETS:
        return Rejected(
            ConvertStatus.UNSUPPORTED_VARIANT,
            f"reasoning-gym dataset {source_dataset!r} cannot score even its own gold answer (library bug)",
        )
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return Rejected(ConvertStatus.NULL_GRADER, "verifier_data.answer is empty")
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=ReasoningGymSpec(dataset=source_dataset),
        dockerfile=drop_dockerfile_lines(task.text(DOCKERFILE), _OLD_REASONING_GYM_PIP_INSTALL),
        tags=("reasoning", "reasoning-gym", source_dataset.replace("_", "-"), "nemotron"),
        data_files={"tests/entry.json": json.dumps(data).encode()},
    )


def _convert_grid_match(task: TaskFiles, data: dict) -> ConvertedTask | Rejected:
    """ARC-AGI transductive: the agent writes the output grid directly, matched with :class:`ExactSpec`."""
    grid = data.get("expected_output")
    if not (isinstance(grid, list) and grid and all(isinstance(row, list) and row for row in grid)):
        return Rejected(ConvertStatus.NULL_GRADER, "expected_output missing or malformed")
    if not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for row in grid for v in row):
        return Rejected(ConvertStatus.UNSUPPORTED_VARIANT, "expected_output has a non single-digit (0-9) cell")
    expected = "\n".join(" ".join(str(v) for v in row) for row in grid)
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=ExactSpec(expected=(expected,)),
        dockerfile=task.text(DOCKERFILE),
        tags=("reasoning", "arc-agi", "grid-match", "nemotron"),
    )


def _convert_grid_transform(task: TaskFiles, data: dict) -> ConvertedTask | Rejected:
    """ARC-AGI python-inductive: the agent writes ``transform()``, run on held-out grid pairs."""
    cases = data.get("test_cases")
    if not isinstance(cases, list):
        return Rejected(ConvertStatus.NULL_GRADER, f"test_cases is not a list: {type(cases).__name__}")
    if len(cases) < MIN_TRANSFORM_CASES:
        return Rejected(ConvertStatus.TOO_FEW_CASES, f"{len(cases)} held-out cases, need at least {MIN_TRANSFORM_CASES}")
    for case in cases:
        valid = (
            isinstance(case, dict)
            and isinstance(case.get("input"), list)
            and case["input"]
            and isinstance(case.get("output"), list)
            and case["output"]
        )
        if not valid:
            return Rejected(ConvertStatus.NULL_GRADER, "a held-out case is missing its input/output grid")
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=ScriptSpec(path=TRANSFORM_SCRIPT, timeout=120.0),
        dockerfile=task.text(DOCKERFILE),
        tags=("reasoning", "arc-agi", "grid-transform", "code", "nemotron"),
        language="python",
        data_files={
            f"tests/{TRANSFORM_SCRIPT}": _TRANSFORM_RUNNER.encode(),
            f"tests/{CASES_FILE}": json.dumps(cases).encode(),
        },
    )


_TRANSFORM_RUNNER = '''#!/usr/bin/env python3
"""Run the agent's transform() against the held-out ARC-AGI grid pair(s).

Reads test cases from cases.json beside this script. Looks for a ``def transform(...)`` in
/app/solution.py first, then falls back to a fenced ```python code block in /app/answer.txt (the
instruction allows either). Runs the extracted function once per case in a fresh subprocess and
compares the returned grid cell-by-cell against the gold output. Prints the reward, 1.0 or 0.0, on
the last line of stdout, per the script mode's stdout-channel contract.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(os.environ["TASKTROVE_TESTS_DIR"])
WORKSPACE = Path(os.environ["TASKTROVE_WORKSPACE"])
CASES = json.loads((TESTS_DIR / "cases.json").read_text())
TIMEOUT = 30

_FENCE = re.compile(r"```(?:python|py)?\\s*\\n(.*?)```", re.DOTALL | re.IGNORECASE)

_RUNNER_TEMPLATE = """
import json, sys

{{TRANSFORM_CODE}}

def _main():
    cases = json.load(open(sys.argv[1]))
    results = []
    for case in cases:
        out = transform([list(row) for row in case["input"]])
        results.append([[int(v) for v in row] for row in out])
    print(json.dumps(results))

_main()
"""


def _extract_code(text: str) -> str | None:
    blocks = _FENCE.findall(text)
    for block in reversed(blocks):
        if "def transform" in block:
            return block
    if "def transform" in text:
        return text
    return None


def _agent_code() -> str | None:
    solution = WORKSPACE / "solution.py"
    if solution.is_file():
        code = _extract_code(solution.read_text(errors="replace"))
        if code is not None:
            return code
    answer = WORKSPACE / "answer.txt"
    if answer.is_file():
        code = _extract_code(answer.read_text(errors="replace"))
        if code is not None:
            return code
    return None


def main() -> float:
    code = _agent_code()
    if code is None:
        print("no transform() found in solution.py or answer.txt", file=sys.stderr)
        return 0.0
    runner = _RUNNER_TEMPLATE.replace("{{TRANSFORM_CODE}}", code)
    with tempfile.TemporaryDirectory() as scratch:
        runner_path = Path(scratch) / "runner.py"
        cases_path = Path(scratch) / "cases.json"
        runner_path.write_text(runner)
        cases_path.write_text(json.dumps(CASES))
        try:
            result = subprocess.run(
                [sys.executable, str(runner_path), str(cases_path)],
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                cwd=scratch,
            )
        except subprocess.TimeoutExpired:
            print("transform() timed out", file=sys.stderr)
            return 0.0
    if result.returncode != 0:
        print(f"runner failed (rc={result.returncode}): {result.stderr[-500:]}", file=sys.stderr)
        return 0.0
    try:
        got = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as error:
        print(f"could not parse runner output: {error}", file=sys.stderr)
        return 0.0
    if len(got) != len(CASES):
        print(f"case count mismatch: {len(got)} vs {len(CASES)}", file=sys.stderr)
        return 0.0
    for actual, case in zip(got, CASES):
        expected = [[int(v) for v in row] for row in case["output"]]
        if actual != expected:
            print("case mismatch", file=sys.stderr)
            return 0.0
    return 1.0


if __name__ == "__main__":
    print(main())
'''


CONVERTER = Converter(
    name="nemotron_reasoning",
    keys=(ConverterKey("other", frozenset({"tests/test.sh", "tests/validate_verifier_data.py", "tests/verifier.py"})),),
    convert=convert_nemotron_reasoning,
)
