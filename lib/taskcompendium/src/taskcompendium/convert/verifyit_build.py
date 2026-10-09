# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source image recipes with the checked-out Verifyit package."""

import re
from functools import cache
from pathlib import Path

from taskcompendium.convert.tasktrove import DOCKERFILE, UV_IMAGE
from taskcompendium.models import DockerBuildContext, TaskResource
from taskcompendium.runtime.local import context_paths
from taskcompendium.runtime.resources import inline_resource

VERIFYIT_CONTEXT = "taskcompendium-verifyit"
VERIFYIT_INSTALL = (
    "# --- verifyit ---\n"
    "RUN command -v git >/dev/null || (apt-get update && apt-get install -y --no-install-recommends git"
    " && rm -rf /var/lib/apt/lists/*)\n"
    f"COPY --from={UV_IMAGE} /uv /usr/local/bin/uv\n"
    f"COPY {VERIFYIT_CONTEXT}/ /opt/taskcompendium-verifyit/\n"
    'RUN UV_TOOL_BIN_DIR=/usr/local/bin uv tool install --python ">=3.11" /opt/taskcompendium-verifyit\n'
)


@cache
def verifyit_build_files(package: Path) -> tuple[TaskResource, ...]:
    """Bundle the checked-out package used by the source recipe."""
    paths = [package / "pyproject.toml", package / "README.md", *context_paths(package / "src/verifyit")]
    return tuple(
        inline_resource(f"{VERIFYIT_CONTEXT}/{path.relative_to(package).as_posix()}", path.read_bytes())
        for path in paths
    )


def verifyit_build_context(
    dockerfile: str, archive_resources: tuple[TaskResource, ...], *, package: Path
) -> DockerBuildContext:
    """Retain source environment files and append the bundled verifier installation."""
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(line.rstrip() for line in dockerfile.splitlines())).strip("\n")
    dockerfile = body + "\n\n" + VERIFYIT_INSTALL
    recipe = inline_resource("Dockerfile", dockerfile.encode())
    original = next((resource for resource in archive_resources if resource.path == DOCKERFILE), None)
    if original is not None:
        recipe = original.model_copy(update={"path": "Dockerfile", "source": recipe.source})
    files = (
        recipe,
        *(
            resource.model_copy(update={"path": resource.path.removeprefix("environment/")})
            for resource in archive_resources
            if resource.path.startswith("environment/") and resource.path != DOCKERFILE
        ),
    )
    if any(resource.path.startswith(VERIFYIT_CONTEXT) for resource in files):
        raise ValueError("Source build context occupies the bundled verifier path")
    return DockerBuildContext(files=(*files, *verifyit_build_files(package)))
