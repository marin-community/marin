# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The staged verifyit runtime runs on the grader machine's own interpreter.

The catalog declares Python 3.10 workspace images, so the package must grade with the
compat backports instead of the 3.11 standard library, and 3.11 machines must keep
working (issue #11721).
"""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from verifyit.spec import NumericSpec, render_spec

VERIFYIT_ROOT = Path(__file__).resolve().parents[1]

# The declared grader machines below the repo's 3.12 floor. 3.12 is exercised by every
# other test in this suite.
GRADER_INTERPRETERS = ("3.10", "3.11")

# Extras too heavy for the ephemeral venv; their modules may raise ImportError on the
# missing third-party root, never on the staged package's own imports.
UNINSTALLED_EXTRA_ROOTS = ("reasoning_gym", "sympy", "math_verify")

# Cold uv caches download the interpreter and wheels; the pytest and subprocess budgets stay in step.
TIMEOUT_SECONDS = 300

SWEEP_SCRIPT = f"""
import importlib
import pkgutil
import re
import sys

import verifyit

uninstalled = {UNINSTALLED_EXTRA_ROOTS}
broken = {{}}
for module in pkgutil.walk_packages(verifyit.__path__, "verifyit."):
    try:
        importlib.import_module(module.name)
    except ImportError as error:
        message = str(error)
        missing = re.search(r"No module named '([^'.]+)", message)
        if missing and missing.group(1) in uninstalled:
            continue
        broken[module.name] = message
if broken:
    for name, message in broken.items():
        print(name + ": " + message, file=sys.stderr)
    raise SystemExit(1)
"""


@pytest.fixture(scope="session", params=GRADER_INTERPRETERS, ids=lambda version: f"python-{version}")
def grader_venv(tmp_path_factory, request):
    """An installed verifyit in a uv-managed venv on one grader interpreter version.

    The install resolves ``requires-python`` and the conditional tomli dependency.
    """
    interpreter = request.param
    venv = tmp_path_factory.mktemp(f"grader-{interpreter.replace('.', '')}")
    for command in (
        ["uv", "venv", "--python", interpreter, str(venv)],
        ["uv", "pip", "install", "--python", str(venv / "bin" / "python"), f"{VERIFYIT_ROOT}[judge,schema]"],
    ):
        subprocess.run(command, check=True, capture_output=True, text=True)
    return venv


@pytest.mark.timeout(TIMEOUT_SECONDS)
def test_staged_imports_survive_the_grader_interpreter(grader_venv):
    result = subprocess.run(
        [str(grader_venv / "bin" / "python"), "-c", SWEEP_SCRIPT],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.timeout(TIMEOUT_SECONDS)
def test_entry_grades_on_the_grader_interpreter(tmp_path, grader_venv):
    config = tmp_path / "verifier.toml"
    config.write_text(
        render_spec(
            replace(NumericSpec("0.3", tolerance_abs=0.0, tolerance_rel=0.0), output=str(tmp_path / "answer.txt"))
        )
    )
    (tmp_path / "answer.txt").write_text("0.3")
    logs = tmp_path / "logs"
    result = subprocess.run(
        [
            str(grader_venv / "bin" / "verifyit"),
            str(config),
            "--logs-dir",
            str(logs),
            "--workspace",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((logs / "reward.json").read_text()) == {"reward": 1.0}
