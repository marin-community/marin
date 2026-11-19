# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The frozen upstream parser dependency context for a fresh grading machine."""

from pathlib import Path

from taskcompendium.models import CommandSemantics, DockerBuildContext, EnvironmentRequirements
from taskcompendium.runtime.resources import inline_resource

HERE = Path(__file__).parent / "parser"


def report_environment() -> EnvironmentRequirements:
    """Bind the pinned base and dependency lock without publishing a baked image."""
    return EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        working_directory="/",
        docker_build=DockerBuildContext(
            files=tuple(
                inline_resource(name, (HERE / name).read_bytes()).model_copy(update={"mode": "0644", "mtime_ns": 0})
                for name in ("Dockerfile", "requirements.lock", "provenance.json")
            )
        ),
    )
