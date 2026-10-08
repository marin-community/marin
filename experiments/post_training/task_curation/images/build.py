# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build task-curation images, push them, and record each as an artifact under MARIN_PREFIX.

An image's identity is the SHA-256 of its recipe: every file in its context and package directories
with its mode, the digest-pinned base image named by the Dockerfile's FROM line, and the target
platform. The artifact ``images/<name>-<identity[:16]>`` records the pushed digest, so a run whose
recipe is unchanged finds the artifact and starts no Docker build.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import click
from marin.execution.artifact import Artifact, read_record
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext, run
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.task_curation.images.recipes import RECIPES, ImageRecipe

DEFAULT_REGISTRY = "ghcr.io/marin-community"
PLATFORM = "linux/amd64"
IMAGE_ARTIFACT_VERSION = "2026.10.07"
IDENTITY_CHARS = 16
LOCK_FILE = "requirements.lock"
REGULAR_MODE = "100644"
EXECUTABLE_MODE = "100755"
FROM_LINE = re.compile(r"^FROM\s+(\S+)", re.MULTILINE)
PINNED_BASE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
BUILD_COMMAND = "uv run python -m experiments.post_training.task_curation.images.build --recipe {name}"


class MissingImageArtifact(RuntimeError):
    """A declaration names an image recipe whose current identity has no built artifact."""


class ContextFile(BaseModel):
    """A file in a build context, by path relative to the context, git-style mode and content digest."""

    model_config = ConfigDict(frozen=True)

    path: str
    mode: str
    sha256: str


class ImageArtifact(Artifact):
    """The pushed image a recipe produced and the inputs it was built from."""

    name: str
    identity: str
    tag: str
    image: str
    base_image: str
    lock_sha256: str
    context_files: list[ContextFile]
    package_files: dict[str, list[ContextFile]]
    platform: str


@dataclass(frozen=True)
class ImageBuild:
    identity: str
    registry: str
    output_path: str


def context_paths(root: Path) -> list[Path]:
    """Every file below ``root`` in path order, skipping bytecode caches."""
    return sorted(path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts)


def _file_mode(path: Path) -> str:
    # Iris workspace bundles drop exec bits, so a consumer sees only these two modes.
    return EXECUTABLE_MODE if path.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) else REGULAR_MODE


def context_files(root: Path) -> list[ContextFile]:
    return [
        ContextFile(
            path=path.relative_to(root).as_posix(),
            mode=_file_mode(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in context_paths(root)
    ]


def base_image(dockerfile: Path) -> str:
    """The digest-pinned image a single-stage Dockerfile builds from."""
    bases = FROM_LINE.findall(dockerfile.read_text())
    if len(bases) != 1 or PINNED_BASE.fullmatch(bases[0]) is None:
        raise ValueError(f"{dockerfile} must have one FROM line naming an image pinned by digest; found {bases}")
    return bases[0]


def recipe_identity(recipe: ImageRecipe) -> dict[str, Any]:
    """Everything that determines the image a recipe builds."""
    return {
        "name": recipe.name,
        "context": [file.model_dump() for file in context_files(recipe.context)],
        "packages": {
            package.name: [file.model_dump() for file in context_files(package)] for package in recipe.packages
        },
        "base_image": base_image(recipe.context / "Dockerfile"),
        "platform": PLATFORM,
    }


def identity_digest(recipe: ImageRecipe) -> str:
    return hashlib.sha256(canonical_json(recipe_identity(recipe)).encode()).hexdigest()


def image_repository(recipe: ImageRecipe, registry: str) -> str:
    return f"{registry}/task-curation-{recipe.name}"


def _image_build(identity: str, ctx: StepContext) -> ImageBuild:
    return ImageBuild(identity=identity, registry=ctx.runtime_arg("registry"), output_path=ctx.output_path)


def image_artifact(recipe: ImageRecipe, registry: str = DEFAULT_REGISTRY) -> ArtifactStep[ImageArtifact]:
    """The ``images/<name>-<identity[:16]>`` artifact; the registry is where a build pushes, not identity."""
    identity = identity_digest(recipe)
    return ArtifactStep(
        name=f"images/{recipe.name}-{identity[:IDENTITY_CHARS]}",
        version=IMAGE_ARTIFACT_VERSION,
        artifact_type=ImageArtifact,
        run=partial(build_image, recipe),
        build_config=partial(_image_build, identity),
        runtime_args={"registry": registry},
    )


def _docker_config() -> dict[str, Any]:
    directory = Path(os.environ.get("DOCKER_CONFIG", Path.home() / ".docker"))
    path = directory / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _require_login(registry: str) -> None:
    host = registry.split("/", 1)[0]
    config = _docker_config()
    if host in config.get("auths", {}) or host in config.get("credHelpers", {}) or config.get("credsStore"):
        return
    raise RuntimeError(f"Docker has no credentials for {host}; run `docker login {host}` before building")


def _require_tracked(root: Path) -> None:
    """The files the identity hashes must be exactly the git-tracked files, without exec bits.

    Consumers recompute the identity from a workspace bundle that has no git metadata and no exec bits,
    so an untracked or executable file here would give them a different identity.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", "."], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    expected = {path for path in tracked.split("\0") if path}
    files = context_files(root)
    found = {file.path for file in files}
    if found != expected:
        raise ValueError(
            f"{root} must contain exactly its git-tracked files; "
            f"untracked: {sorted(found - expected)}, missing: {sorted(expected - found)}"
        )
    executable = [file.path for file in files if file.mode == EXECUTABLE_MODE]
    if executable:
        raise ValueError(f"{root} has executable files {executable}; set modes inside the Dockerfile instead")


def _preflight(recipe: ImageRecipe, registry: str) -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("Building an image requires the docker CLI with buildx")
    subprocess.run(["docker", "buildx", "version"], check=True, capture_output=True)
    _require_login(registry)
    for root in (recipe.context, *recipe.packages):
        _require_tracked(root)


def build_image(recipe: ImageRecipe, build: ImageBuild) -> ImageArtifact:
    """Build ``recipe`` for linux/amd64, push it under its identity tag, and resolve the pushed digest."""
    if identity_digest(recipe) != build.identity:
        raise RuntimeError(f"The {recipe.name} recipe changed after its artifact was planned; rerun the build")
    _preflight(recipe, build.registry)
    repository = image_repository(recipe, build.registry)
    tag = f"{repository}:{build.identity[:IDENTITY_CHARS]}"
    package_contexts = [f"--build-context={package.name}={package}" for package in recipe.packages]
    # Without provenance the pushed reference is one platform manifest rather than an attestation index.
    subprocess.run(
        [
            "docker",
            "buildx",
            "build",
            f"--platform={PLATFORM}",
            "--provenance=false",
            "--push",
            f"--tag={tag}",
            *package_contexts,
            str(recipe.context),
        ],
        check=True,
    )
    inspected = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", tag, "--format", "{{json .Manifest}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    digest = json.loads(inspected)["digest"]
    return ImageArtifact(
        path=build.output_path,
        name=recipe.name,
        identity=build.identity,
        tag=tag,
        image=f"{repository}@{digest}",
        base_image=base_image(recipe.context / "Dockerfile"),
        lock_sha256=hashlib.sha256((recipe.context / LOCK_FILE).read_bytes()).hexdigest(),
        context_files=context_files(recipe.context),
        package_files={package.name: context_files(package) for package in recipe.packages},
        platform=PLATFORM,
    )


# Built artifacts by path; every declaration that names a recipe reads the same record.
_built_images: dict[str, ImageArtifact] = {}


def built_image(recipe: ImageRecipe) -> ImageArtifact:
    """The built artifact for the recipe's current identity under MARIN_PREFIX."""
    path = image_artifact(recipe).path()
    if path not in _built_images:
        if read_record(path) is None:
            raise MissingImageArtifact(
                f"The {recipe.name} image for this recipe is not built ({path} has no artifact); "
                f"run: {BUILD_COMMAND.format(name=recipe.name)}"
            )
        _built_images[path] = ImageArtifact.raw_load(path)
    return _built_images[path]


@click.command(help=__doc__)
@click.option(
    "--recipe",
    "names",
    type=click.Choice(sorted(RECIPES)),
    multiple=True,
    help="Recipe to build; repeat to select several. Defaults to every recipe.",
)
@click.option("--registry", default=DEFAULT_REGISTRY, show_default=True, help="Registry the build pushes to.")
def main(names: tuple[str, ...], registry: str) -> None:
    logging.basicConfig(level=logging.INFO)
    # A workstation reaches the CoreWeave artifact prefix through its ambient CW_KEY_* pair.
    configure_coreweave_s3()
    recipes = [RECIPES[name] for name in names or sorted(RECIPES)]
    for image in run(*(image_artifact(recipe, registry) for recipe in recipes)):
        click.echo(f"{image.name}: {image.image} ({image.path})")


if __name__ == "__main__":
    main()
