# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Python-test sources retain their archived agent environments and separate graders."""

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
    workdir = tmp_path / "workspace"
    task = converted_task(
        recipe,
        python_archive(
            (
                "FROM python:3.11-slim-bookworm\n"
                "ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\n"
                f"WORKDIR {workdir}\n"
                'RUN mkdir -p dependency && printf "%s:%s" "$PYTHONDONTWRITEBYTECODE" '
                '"$PYTHONUNBUFFERED" > dependency/runtime.txt\n'
            ).encode()
        ),
    )
    environment = task.environment_requirements
    assert environment.docker_image == PINNED_BASES["python:3.11-slim-bookworm"]
    subprocess.run(["sh", "-c", " && ".join(environment.setup_commands)], check=True)
    assert (workdir / "dependency/runtime.txt").read_text() == "1:1"
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
