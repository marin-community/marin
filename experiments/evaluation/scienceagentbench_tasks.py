# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize prepared ScienceAgentBench tasks for the Harbor + Pi launch path."""

import hashlib
import json
from pathlib import Path

from harbor.models.task.config import AgentConfig, EnvironmentConfig, TaskConfig, VerifierConfig, VerifierEnvironmentMode
from harbor.models.task.task import Task
from shellbox.image import image_source_key
from taskcompendium.importers.scienceagentbench import SUBMISSION_PATH, WORKING_DIRECTORY
from taskcompendium.models import ScriptGrader, TaskSpec, TextMessage
from taskcompendium.runtime.resources import materialize_build_context

TASK_SPEC_FILENAME = "grading.json"
ACTOR_BUILD_KEY = "actor_build_key"
TASK_SPEC_HASH_KEY = "grading_sha256"
VERIFIER_IMPORT_PATH = "experiments.evaluation.scienceagentbench_verifier:ScienceAgentBenchVerifier"


def write_harbor_task(
    task: TaskSpec,
    directory: Path,
    *,
    environment: EnvironmentConfig,
    agent_timeout: float,
) -> Task:
    """Write one private task bundle, preserving the prompt and host-only grading inputs.

    Input data and Pi must already be installed in the pinned actor image.
    The caller owns resource sizing and image validation. The output bundle
    includes restricted reference material and must not be published publicly.
    """
    if not isinstance(task.grader, ScriptGrader) or task.grader.replay is None:
        raise ValueError("ScienceAgentBench Harbor tasks require isolated replay")
    actor = task.environment_requirements
    if actor != task.grader.replay.environment:
        raise ValueError("The actor and replay environments must match")
    if actor.docker_build is None and actor.docker_image is None:
        raise ValueError("ScienceAgentBench requires a registry image or a Docker build recipe")
    if environment.docker_image != actor.docker_image:
        raise ValueError("The prepared Harbor actor image must match the TaskSpec")
    if (
        environment.workdir != WORKING_DIRECTORY
        or task.environment_requirements.working_directory != WORKING_DIRECTORY
        or task.output_paths != (SUBMISSION_PATH,)
    ):
        raise ValueError("ScienceAgentBench requires its standard workspace and Python submission path")
    if environment.env != task.environment_requirements.environment_variables:
        raise ValueError("Harbor and TaskSpec must declare the same actor environment variables")
    if task.resources.all or task.resources.worker or task.environment_requirements.setup_commands:
        raise ValueError("Prepare actor inputs and setup in the image before exporting Harbor tasks")
    events = task.context.events
    if len(events) != 1 or not isinstance(events[0], TextMessage) or events[0].role != "user":
        raise ValueError("ScienceAgentBench requires a single task instruction")
    specification_bytes = (task.model_dump_json() + "\n").encode("utf-8")
    config = TaskConfig(
        source=task.source.dataset,
        metadata={
            "task_id": task.id,
            "source_revision": task.source.revision,
            "source_row": task.source.row,
            **task.labels,
            TASK_SPEC_HASH_KEY: hashlib.sha256(specification_bytes).hexdigest(),
        },
        environment=environment,
        agent=AgentConfig(timeout_sec=agent_timeout),
        verifier=VerifierConfig(timeout_sec=task.grader.timeout, environment_mode=VerifierEnvironmentMode.SHARED),
    )
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "tests").mkdir()
    (directory / "instruction.md").write_text(events[0].content, encoding="utf-8")
    if actor.docker_build is not None:
        source = materialize_build_context(actor.docker_build, directory / "environment")
        config.metadata[ACTOR_BUILD_KEY] = image_source_key(source)
    else:
        (directory / "environment").mkdir()
        (directory / "environment" / "shellbox-images.json").write_text(
            json.dumps({"main": {"image": environment.docker_image, "workdir": environment.workdir}}) + "\n",
            encoding="utf-8",
        )
    (directory / "task.toml").write_text(config.model_dump_toml(), encoding="utf-8")
    specification = directory / TASK_SPEC_FILENAME
    specification.write_bytes(specification_bytes)
    specification.chmod(0o600)
    # Harbor's default verifier uploads tests/. Keep private material outside it,
    # and fail if a launch omits the required custom-verifier selection.
    (directory / "tests" / "test.sh").write_text(
        f"#!/bin/sh\necho 'Select verifier.import_path={VERIFIER_IMPORT_PATH}' >&2\nexit 1\n", encoding="utf-8"
    )
    return Task(directory)
