# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private R2E-Gym verifier: preserve the source status-map reward contract."""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from source_parser import decolor_dict_keys, parse_log_pytest

tests = Path(os.environ["TASKTROVE_TESTS_DIR"])
workspace = Path(os.environ["TASKTROVE_WORKSPACE"])
command = [
    "xvfb-run",
    "--auto-servernum",
    ".venv/bin/python",
    "-W",
    "ignore",
    "-m",
    "pytest",
    "--color=no",
    "-rA",
    str(tests / "r2e_tests"),
]
completed = subprocess.run(
    command,
    cwd=workspace,
    env={**os.environ, "QT_QPA_PLATFORM": "minimal", "PYTHONWARNINGS": "ignore::UserWarning,ignore::SyntaxWarning"},
    text=True,
    capture_output=True,
)
actual = decolor_dict_keys(parse_log_pytest(completed.stdout + completed.stderr))
with open(tests / "r2e_assets/expected_output.json") as stream:
    expected = decolor_dict_keys(json.load(stream))
actual = {key.split(" - ")[0]: actual[key] for key in sorted(actual)}
expected = {key.split(" - ")[0]: expected[key] for key in sorted(expected)}
match = len(actual) == len(expected) and all(
    not key or (key in expected and actual[key] == expected[key]) for key in actual
)
print("1" if match else "0")
