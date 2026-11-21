# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The staged verifyit runtime imports and grades on the grader machine's own interpreter.

Staging copies package directories onto the machine and puts them on ``sys.path``; no
dependency metadata ships with them. ``RUNTIME_PACKAGES`` in ``taskcompendium.runtime.local``
is the production counterpart of the staging below. The declared grader environments include
Python 3.10 images, so the runtime must import with the compat backports and the staged
tomli, and 3.11 machines must keep working (issue #11721).
"""

import json
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import tomli
import tomli_w
import verifyit
from verifyit.spec import NumericSpec, render_spec

GRADER_INTERPRETERS = ("3.10", "3.11")

# Extras too heavy for the ephemeral venv; their modules may raise ImportError on the
# missing third-party root, never on the staged package's own imports.
UNINSTALLED_EXTRA_ROOTS = ("reasoning_gym", "sympy", "math_verify")

# The third-party roots of the judge and schema extras, installed into the grader venv the
# way an environment's own package lock provides a mode's dependencies.
EXTRA_ROOT_PACKAGES = ("openai", "jsonschema", "pyyaml")

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

# Hand-written TOML with integer literals, the way task authors write label tables.
JUDGE_SPEC_TOML = """\
mode = "judge"
model = "pinned-judge"
label_scores = { yes = 1, no = 0 }
label_metrics = { yes = { score = 2 } }
"""

JUDGE_ROUND_TRIP_SCRIPT = """
import sys

from verifyit.spec import parse_spec, render_spec

with open(sys.argv[1]) as handle:
    text = handle.read()
parsed = parse_spec(text)
scores = parsed.label_scores
if scores != {"yes": 1.0, "no": 0.0} or any(type(score) is not float for score in scores.values()):
    raise SystemExit(1)
metrics = parsed.label_metrics
if metrics != {"yes": {"score": 2.0}} or any(
    type(metric) is not float for table in metrics.values() for metric in table.values()
):
    raise SystemExit(1)
replayed = render_spec(parsed)
if parse_spec(replayed) != parsed or render_spec(parse_spec(replayed)) != replayed:
    raise SystemExit(1)
"""


@pytest.fixture(scope="session", params=GRADER_INTERPRETERS, ids=lambda version: f"python-{version}")
def staged_runtime(tmp_path_factory, request):
    """A staged runtime directory and a bare uv-managed venv, the way a runtime build stages them."""
    interpreter = request.param
    base = tmp_path_factory.mktemp(f"grader-{interpreter.replace('.', '')}")
    venv = base / "venv"
    for command in (
        ["uv", "venv", "--python", interpreter, str(venv)],
        ["uv", "pip", "install", "--python", str(venv / "bin" / "python"), *EXTRA_ROOT_PACKAGES],
    ):
        subprocess.run(command, check=True, capture_output=True, text=True)
    runtime = base / "runtime"
    for package in (verifyit, tomli, tomli_w):
        shutil.copytree(
            Path(package.__file__).parent,
            runtime / package.__name__,
            # LocalRuntime._build stages the same pure-Python sources on every consumer.
            ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.pyd"),
        )
    return runtime, venv


def _run_on_grader(staged_runtime, script, *arguments, cwd="/"):
    """Run one command on the grader interpreter with the staged packages importable."""
    runtime, venv = staged_runtime
    return subprocess.run(
        [str(venv / "bin" / "python"), "-c", script, *arguments],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(runtime)},
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )


@pytest.mark.timeout(TIMEOUT_SECONDS)
def test_staged_imports_survive_the_grader_interpreter(staged_runtime):
    runtime, _ = staged_runtime
    # The runtime identity must stay platform-neutral, so staging ships pure-Python sources only.
    assert not [path for path in runtime.rglob("*") if path.suffix in (".so", ".pyd")]
    result = _run_on_grader(staged_runtime, SWEEP_SCRIPT)
    assert result.returncode == 0, result.stderr


@pytest.mark.timeout(TIMEOUT_SECONDS)
def test_entry_grades_on_the_grader_interpreter(tmp_path, staged_runtime):
    config = tmp_path / "verifier.toml"
    config.write_text(
        render_spec(
            replace(NumericSpec("0.3", tolerance_abs=0.0, tolerance_rel=0.0), output=str(tmp_path / "answer.txt"))
        )
    )
    (tmp_path / "answer.txt").write_text("0.3")
    logs = tmp_path / "logs"
    result = _run_on_grader(
        staged_runtime,
        "from verifyit.grade import main; raise SystemExit(main())",
        str(config),
        "--logs-dir",
        str(logs),
        "--workspace",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((logs / "reward.json").read_text()) == {"reward": 1.0}


@pytest.mark.timeout(TIMEOUT_SECONDS)
def test_judge_spec_round_trips_on_the_grader_interpreter(tmp_path, staged_runtime):
    config = tmp_path / "verifier.toml"
    config.write_text(JUDGE_SPEC_TOML)
    result = _run_on_grader(staged_runtime, JUDGE_ROUND_TRIP_SCRIPT, str(config))
    assert result.returncode == 0, result.stderr
