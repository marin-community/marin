# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Python-test sources retain their archived agent environments and separate graders."""

import os
import subprocess

import pytest

from experiments.post_training.task_curation.datasets.tasktrove import python_tests
from experiments.post_training.task_curation.environment_fidelity import PINNED_BASES
from experiments.post_training.task_curation.tests.conversion import convert_row, converted_task, tasktrove_row

SOURCES = python_tests.sources()


def python_archive(dockerfile: bytes) -> dict:
    return tasktrove_row(
        {
            "instruction.md": b"Implement add(a, b) in /app/solution.py.",
            "environment/Dockerfile": dockerfile,
            "tests/test.sh": b"#!/bin/sh\nexit 1\n",
            "tests/test_solution.py": b"from solution import add\ndef test_add():\n    assert add(2, 3) == 5\n",
            "solution/solution.py": b"def add(a, b):\n    return a + b\n",
        }
    )


@pytest.mark.parametrize("source", SOURCES, ids=lambda source: source.name)
def test_python_test_sources_keep_recipe_steps_and_runtime_variables(source, tmp_path):
    recipe = source.config
    task = converted_task(
        recipe,
        python_archive(
            b"FROM python:3.11-slim-bookworm\n"
            b"ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\n"
            b"RUN pip install --no-cache-dir pytest==8.3.5 numpy==2.1.3\n"
            b"WORKDIR /app\n"
        ),
    )
    environment = task.environment_requirements
    assert environment.docker_image == PINNED_BASES["python:3.11-slim-bookworm"]
    assert environment.working_directory == "/app"
    capture = tmp_path / "pip-call.txt"
    pip = tmp_path / "pip"
    pip.write_text('#!/bin/sh\nprintf "%s\\n" "$PYTHONDONTWRITEBYTECODE|$PYTHONUNBUFFERED|$PWD|$*" > "$PIP_CAPTURE"\n')
    pip.chmod(0o755)
    subprocess.run(
        ["sh", "-c", environment.setup_commands[0]],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "PIP_CAPTURE": str(capture)},
        check=True,
    )
    assert capture.read_text().strip() == "1|1|/|install --no-cache-dir pytest==8.3.5 numpy==2.1.3"
    assert environment.environment_variables == {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}


@pytest.mark.parametrize("source", SOURCES, ids=lambda source: source.name)
@pytest.mark.parametrize(
    "dockerfile, match",
    [
        (b"FROM python:3.9-slim\n", "unmapped base"),
        (b"FROM python:3.10-slim\n#\xff\n", "not UTF-8"),
        (b"FROM python:3.10-slim\nCOPY fixture /app/fixture\n", "unsupported step: COPY"),
    ],
)
def test_python_test_sources_refuse_an_environment_they_cannot_lower(source, dockerfile, match):
    converted = convert_row(source.config, python_archive(dockerfile))
    assert converted.reason == "environment_unlowerable"
    assert match in converted.detail
