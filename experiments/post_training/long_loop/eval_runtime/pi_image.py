# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add the frozen Pi runtime to the experiment's restored Harbor images."""

import tomllib
from enum import StrEnum
from pathlib import Path

import tomli_w
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import inline_resource, resource_bytes

RECIPE = Path(__file__).with_suffix("")
NODE_IMAGE = "node@sha256:5ec23b7c622b00547753bc6ffd78085dd868a68aeb41c8e0a1418f033e13d96d"


class PiPathScope(StrEnum):
    IMAGE = "image"
    AGENT = "agent"


def with_pi_runtime(task: TaskSpec, *, path_scope: PiPathScope) -> TaskSpec:
    """Preserve the task image and files, adding an isolated locked Pi install."""
    files = {resource.path: resource for resource in task.resources.verifier}
    config = tomllib.loads(resource_bytes(files["task.toml"]).decode())
    environment = config.setdefault("environment", {})
    if "environment/docker-compose.yaml" in files or "environment/docker-compose.yml" in files:
        return task  # Compose tasks are excluded by the smoke selection.
    if task.source.dataset == "bixbench-pi" or "bixbench" in task.id:
        environment["storage_mb"] = 10240
        config["verifier"]["env"] = {
            "OPENAI_API_KEY": "${OPENAI_API_KEY}",
            "OPENAI_BASE_URL": "${OPENAI_BASE_URL}",
            "MODEL_NAME": "MiniMaxAI/MiniMax-M3-MXFP8",
        }
    original = environment.pop("docker_image", None)
    dockerfile = f"FROM {original}\n" if original else resource_bytes(files["environment/Dockerfile"]).decode()
    dockerfile = (
        f"FROM {NODE_IMAGE} AS long_loop_pi_runtime\n"
        "WORKDIR /opt/long-loop-pi\n"
        "COPY long-loop-pi/package.json long-loop-pi/package-lock.json ./\n"
        "RUN npm ci --omit=dev --no-audit --no-fund\n"
        + dockerfile
        + "\nCOPY --from=long_loop_pi_runtime /usr/local/bin/node /opt/long-loop-node/node\n"
        "COPY --from=long_loop_pi_runtime /opt/long-loop-pi /opt/long-loop-pi\n"
        + (
            "ENV PATH=/opt/long-loop-node:/opt/long-loop-pi/node_modules/.bin:${PATH}\n"
            if path_scope is PiPathScope.IMAGE
            else ""
        )
        + "RUN PATH=/opt/long-loop-node:/opt/long-loop-pi/node_modules/.bin:${PATH} "
        'bash -c \'test "$(pi --version)" = "0.87.0"\'\n'
    )
    replacements = {
        "task.toml": tomli_w.dumps(config).encode(),
        "environment/Dockerfile": dockerfile.encode(),
        **{
            f"environment/long-loop-pi/{name}": (RECIPE / name).read_bytes()
            for name in ("package.json", "package-lock.json")
        },
    }
    resources = tuple(r for r in task.resources.verifier if r.path not in replacements) + tuple(
        inline_resource(path, content) for path, content in replacements.items()
    )
    return task.model_copy(update={"resources": task.resources.model_copy(update={"verifier": resources})})
