# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Registry outputs of the archived CodeContests and TACO build contexts."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from taskcompendium.models import CommandSemantics, DockerBuildContext, EnvironmentRequirements
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.environments import VERIFYIT_PACKAGE
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import DOCKERFILE, archive_resources
from experiments.post_training.task_curation.datasets.tasktrove.conversion.verifyit_build import (
    VERIFYIT_CONTEXT,
    verifyit_build_context,
)

# Keys cover the archived context and verifier-install recipe, with paths, modes and bytes.
# The images freeze the bundled Verifyit from d8cdda47f4; current checkout changes do not
# update those installed bytes. Full build-context hashes are recorded with the release evidence.
BUILT_IMAGES = {
    "fdaccdb4fa7b05b556da31096829ace404f9d01d6bd6758fb634b1a6e8cee2b7": (
        "ghcr.io/marin-community/task-curation-tasktrove"
        "@sha256:0923f913e119e75183a4496f423e63e2db7ebf04beb28f31542442b687415ecd"
    ),
    "3c03c0d808b01cb286491d5bbfc940a524176a807545533a9d3df828a7cc70a3": (
        "ghcr.io/marin-community/task-curation-tasktrove"
        "@sha256:dce167c396b99d6e978d675d79bf0c84d3c170c6e7ab9033cb3a9602c9988bf1"
    ),
}


def build_context_sha256(build: DockerBuildContext) -> str:
    entries = sorted(
        (resource.path, int(resource.mode or "644", 8), hashlib.sha256(resource_bytes(resource)).hexdigest())
        for resource in build.files
    )
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def archive_build_environment(data: Mapping[str, Any]) -> EnvironmentRequirements:
    """Use an exact archived build's pin; retain unrecognized contexts as builds."""
    resources = archive_resources(data).oracle
    dockerfile = next(resource_bytes(resource).decode() for resource in resources if resource.path == DOCKERFILE)
    build = verifyit_build_context(dockerfile, resources, package=VERIFYIT_PACKAGE)
    source = DockerBuildContext(
        files=tuple(resource for resource in build.files if not resource.path.startswith(f"{VERIFYIT_CONTEXT}/"))
    )
    image = BUILT_IMAGES.get(build_context_sha256(source))
    return EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image=image,
        docker_build=None if image is not None else build,
    )
