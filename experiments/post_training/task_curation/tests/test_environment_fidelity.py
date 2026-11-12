# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The environment-fidelity checks reject substitution and missing environments."""

import os
import subprocess

import pytest

from experiments.post_training.task_curation.environment_fidelity import (
    PINNED_BASES,
    UnlowerableRecipe,
    environment_unused,
    lower_environment,
    missing_environment,
)

AGENT = "ghcr.io/marin-community/iris-task@sha256:" + "a" * 64


def test_a_spec_that_replaces_its_archive_environment_is_flagged():
    dockerfile = b"FROM ubuntu:24.04\nRUN pip install pandas\n"
    replaced = {"environment_requirements": {"docker_image": AGENT}}
    # The archive lowers onto ubuntu:24.04, so an iris-task image is not its environment.
    assert "does not use the archive's base image" in environment_unused(replaced, dockerfile)
    faithful = {"environment_requirements": {"docker_build": {"files": [{"path": "Dockerfile", "source": {}}]}}}
    assert environment_unused(faithful, dockerfile) is None


def test_an_agentic_task_without_an_environment_is_rejected():
    assert missing_environment({"environment_requirements": {"capabilities": ["agent:pi"]}})
    assert missing_environment({"environment_requirements": {"docker_image": AGENT}}) is None


def test_a_recipe_lowers_onto_its_pinned_base():
    recipe = (
        "FROM ubuntu:24.04\n"
        "ENV DEBIAN_FRONTEND=noninteractive\n"
        "WORKDIR /workspace\n"
        "RUN mkdir -p /out && chmod 777 /out\n"
    )
    lowered = lower_environment(recipe, PINNED_BASES)
    assert lowered.base_ref == PINNED_BASES["ubuntu:24.04"]
    assert lowered.working_directory == "/workspace"
    assert lowered.setup_commands == (
        "mkdir -p /workspace",
        "(cd /workspace && export DEBIAN_FRONTEND=noninteractive; mkdir -p /out && chmod 777 /out)",
    )


@pytest.mark.parametrize(
    "recipe",
    [
        "FROM python:3.9-slim\n",  # unmapped base
        "FROM ubuntu:24.04\nCOPY files/ /app/\n",  # build context
        "FROM ubuntu:24.04\nUSER app\n",  # identity change
    ],
)
def test_a_recipe_is_refused_when_it_cannot_be_expressed(recipe):
    with pytest.raises(UnlowerableRecipe):
        lower_environment(recipe, PINNED_BASES)


def test_replaying_the_archive_steps_counts_as_using_its_environment():
    dockerfile = b"FROM ubuntu:24.04\nWORKDIR /workspace\nRUN mkdir -p /out\n"
    lowered = lower_environment(dockerfile.decode(), PINNED_BASES)
    faithful = {
        "environment_requirements": {
            "docker_image": lowered.base_ref,
            "working_directory": lowered.working_directory,
            "setup_commands": list(lowered.setup_commands),
        }
    }
    assert environment_unused(faithful, dockerfile) is None
    stale = {**faithful, "environment_requirements": {**faithful["environment_requirements"], "setup_commands": []}}
    assert "does not replay" in environment_unused(stale, dockerfile)


def test_lowering_retains_runtime_environment_and_working_directory():
    dockerfile = b'FROM python:3.11-slim-bookworm\nENV LABEL="two words" PYTHONUNBUFFERED=1\nWORKDIR /app\n'
    lowered = lower_environment(dockerfile.decode(), PINNED_BASES)
    assert lowered.environment_variables == {"LABEL": "two words", "PYTHONUNBUFFERED": "1"}
    requirements = {
        "docker_image": lowered.base_ref,
        "working_directory": "/app",
        "setup_commands": list(lowered.setup_commands),
        "environment_variables": lowered.environment_variables,
    }
    assert environment_unused({"environment_requirements": requirements}, dockerfile) is None
    missing_variables = {**requirements, "environment_variables": {}}
    assert "environment variables" in environment_unused({"environment_requirements": missing_variables}, dockerfile)
    wrong_directory = {**requirements, "working_directory": "/workspace"}
    assert "working directory" in environment_unused({"environment_requirements": wrong_directory}, dockerfile)


def test_env_applies_to_every_command_in_a_run(tmp_path):
    recipe = (
        f"FROM ubuntu:24.04\nWORKDIR {tmp_path}\nENV LABEL=expected\nRUN true && printf '%s' \"$LABEL\" > observed\n"
    )
    lowered = lower_environment(recipe, PINNED_BASES)
    subprocess.run(
        ["sh", "-c", " && ".join(lowered.setup_commands)],
        env={key: value for key, value in os.environ.items() if key != "LABEL"},
        check=True,
    )
    assert (tmp_path / "observed").read_text() == "expected"


def test_env_dependent_on_the_base_image_is_deferred():
    with pytest.raises(UnlowerableRecipe, match="ENV"):
        lower_environment("FROM ubuntu:24.04\nENV PATH=/tool/bin:$PATH\n", PINNED_BASES)
