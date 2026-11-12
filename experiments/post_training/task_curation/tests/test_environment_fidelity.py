# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The environment-fidelity checks reject substitution and missing environments."""

import os
import subprocess

import pytest
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)

from experiments.post_training.task_curation.environment_fidelity import (
    PINNED_BASES,
    UnlowerableRecipe,
    environment_unused,
    lower_environment,
    missing_environment,
)

AGENT = "ghcr.io/marin-community/iris-task@sha256:" + "a" * 64


def spec(requirements: dict) -> TaskSpec:
    """A minimal agentic task carrying exactly these environment requirements."""
    declared = {"docker_image", "docker_build", "packages_lock", "package_spec"}
    if any(key in requirements for key in declared):
        # The model requires process semantics whenever a native environment is declared.
        requirements = {"command_semantics": "linux_process", **requirements}
    return TaskSpec(
        id="fixture",
        context=ConversationInput(events=(TextMessage(role="user", content="do the task"),)),
        environment_requirements=EnvironmentRequirements.model_validate(requirements),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="no verifier in this fixture"),
        source=Source(dataset="fixture", revision="1", row="fixture", importer_revision="1"),
    )


def test_a_spec_that_replaces_its_archive_environment_is_flagged():
    dockerfile = b"FROM ubuntu:24.04\nRUN pip install pandas\n"
    replaced = spec({"docker_image": AGENT})
    # The archive lowers onto ubuntu:24.04, so an iris-task image is not its environment.
    assert "does not use the archive's base image" in environment_unused(replaced, dockerfile)
    faithful = spec({"docker_build": {"files": [{"path": "Dockerfile", "source": {"content_base64": ""}}]}})
    assert environment_unused(faithful, dockerfile) is None


def test_an_agentic_task_without_an_environment_is_rejected():
    assert missing_environment(spec({"capabilities": ["agent:pi"]}))
    assert missing_environment(spec({"docker_image": AGENT})) is None


def test_a_recipe_lowers_onto_its_pinned_base(tmp_path):
    recipe = (
        "FROM ubuntu:24.04\n"
        "ENV DEBIAN_FRONTEND=noninteractive\n"
        f"WORKDIR {tmp_path}\n"
        "RUN mkdir -p out && touch out/ready\n"
    )
    lowered = lower_environment(recipe, PINNED_BASES)
    assert lowered.base_ref == PINNED_BASES["ubuntu:24.04"]
    subprocess.run(["sh", "-c", " && ".join(lowered.setup_commands)], check=True)
    assert (tmp_path / "out/ready").is_file()


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
        "docker_image": lowered.base_ref,
        "working_directory": lowered.working_directory,
        "setup_commands": list(lowered.setup_commands),
    }
    assert environment_unused(spec(faithful), dockerfile) is None
    assert "does not replay" in environment_unused(spec({**faithful, "setup_commands": []}), dockerfile)


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
    assert environment_unused(spec(requirements), dockerfile) is None
    missing_variables = {**requirements, "environment_variables": {}}
    assert "environment variables" in environment_unused(spec(missing_variables), dockerfile)
    wrong_directory = {**requirements, "working_directory": "/workspace"}
    assert "working directory" in environment_unused(spec(wrong_directory), dockerfile)


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
