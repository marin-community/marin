# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Python-test sources retain their archived agent environments and separate graders."""

import pytest
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.conversion import row_source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow

from experiments.post_training.task_curation.datasets.tasktrove import python_tests
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import unpack_task_binary
from experiments.post_training.task_curation.environment_fidelity import PINNED_BASES, UnlowerableRecipe
from experiments.post_training.task_curation.pipeline import planned_lock_url, source_recipe
from experiments.post_training.task_curation.tests.conversion import converted_task, tasktrove_row

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
def test_python_test_sources_keep_recipe_steps_and_runtime_variables(source):
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
    assert environment.setup_commands == (
        "(cd / && PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 "
        "pip install --no-cache-dir pytest==8.3.5 numpy==2.1.3)",
        "mkdir -p /app",
    )
    assert environment.environment_variables == {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}


@pytest.mark.parametrize("source", SOURCES, ids=lambda source: source.name)
def test_python_test_sources_keep_the_production_grader_lock(source):
    recipe = source.config
    lock = planned_lock_url(recipe.grader)
    grader = EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock=lock)
    context = ConversionContext({}, grader)
    pipeline = source_recipe(recipe, {}, grader)
    data = unpack_task_binary(python_archive(b"FROM python:3.10-slim\nWORKDIR /app\n"), context)
    row = RawRow("fixture", source=row_source(pipeline, "fixture"), data=data)
    converted = recipe.convert(row, context)
    assert converted.task.grader.environment == grader
    assert converted.task.environment_requirements.docker_image == PINNED_BASES["python:3.10-slim"]


@pytest.mark.parametrize("source", SOURCES, ids=lambda source: source.name)
@pytest.mark.parametrize(
    "dockerfile, error, match",
    [
        (b"FROM python:3.9-slim\n", UnlowerableRecipe, "unmapped base"),
        (b"FROM python:3.10-slim\nCOPY fixture /app/fixture\n", UnlowerableRecipe, "unsupported step: COPY"),
    ],
)
def test_python_test_sources_refuse_an_environment_they_cannot_lower(source, dockerfile, error, match):
    with pytest.raises(error, match=match):
        converted_task(source.config, python_archive(dockerfile))
