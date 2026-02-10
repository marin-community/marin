# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Image build commands."""

import subprocess
from pathlib import Path

import click


def _find_marin_root() -> Path:
    """Find the marin monorepo root (contains pyproject.toml + lib/iris)."""
    iris_root = _find_iris_root()
    # iris root is lib/iris, marin root is two levels up
    marin_root = iris_root.parent.parent
    if (marin_root / "pyproject.toml").exists() and (marin_root / "lib" / "iris").is_dir():
        return marin_root
    raise click.ClickException("Cannot find marin repo root. Expected lib/iris to be inside a marin workspace.")


def _find_iris_root() -> Path:
    """Find the iris package root directory containing Dockerfiles.

    Searches in order:
    1. Relative to this file (cli/build.py -> iris root is 4 levels up from src/iris/cli/build.py)
    2. Current working directory
    3. Walking up from cwd until Dockerfile.worker is found
    """
    build_path = Path(__file__).resolve()
    # build.py is at src/iris/cli/build.py, so iris root is 4 levels up
    iris_root = build_path.parent.parent.parent.parent
    if (iris_root / "Dockerfile.worker").exists() and (iris_root / "Dockerfile.controller").exists():
        return iris_root

    cwd = Path.cwd()
    if (cwd / "Dockerfile.worker").exists():
        return cwd

    for parent in cwd.parents:
        if (parent / "Dockerfile.worker").exists():
            return parent

    raise click.ClickException(
        "Cannot find Dockerfile.worker. Run from the iris directory or specify --dockerfile and --context."
    )


def _push_to_registries(
    source_tag: str,
    regions: tuple[str, ...],
    project: str,
    image_name: str | None = None,
    version: str | None = None,
) -> None:
    """Push a local Docker image to multiple GCP Artifact Registry regions."""
    if not image_name or not version:
        parts = source_tag.split(":")
        if not image_name:
            image_name = parts[0].split("/")[-1]
        if not version:
            version = parts[1] if len(parts) > 1 else "latest"

    click.echo(f"Pushing {source_tag} to {len(regions)} region(s)...")

    for r in regions:
        dest_tag = f"{r}-docker.pkg.dev/{project}/marin/{image_name}:{version}"

        click.echo(f"\nConfiguring {r}-docker.pkg.dev...")
        result = subprocess.run(
            ["gcloud", "auth", "configure-docker", f"{r}-docker.pkg.dev", "-q"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            click.echo(
                f"Warning: Failed to configure docker auth for {r}-docker.pkg.dev: {result.stderr.strip()}", err=True
            )

        click.echo(f"Tagging as {dest_tag}")
        result = subprocess.run(["docker", "tag", source_tag, dest_tag], check=False)
        if result.returncode != 0:
            click.echo(f"Failed to tag image for {r}", err=True)
            continue

        click.echo(f"Pushing to {r}...")
        result = subprocess.run(["docker", "push", dest_tag], check=False)
        if result.returncode != 0:
            click.echo(f"Failed to push to {r}", err=True)
            continue

        click.echo(f"Successfully pushed to {dest_tag}")

    click.echo("\nDone!")


def _build_image(
    image_type: str,
    tag: str,
    push: bool,
    dockerfile: str | None,
    context: str | None,
    platform: str,
    region: tuple[str, ...],
    project: str,
) -> None:
    """Build a Docker image for Iris (worker or controller)."""
    dockerfile_name = f"Dockerfile.{image_type}"

    iris_root = _find_iris_root()
    dockerfile_path = Path(dockerfile) if dockerfile else iris_root / dockerfile_name
    context_path = Path(context) if context else iris_root

    if not dockerfile_path.exists():
        raise click.ClickException(f"Dockerfile not found: {dockerfile_path}")

    click.echo(f"Using Dockerfile: {dockerfile_path}")

    cmd = ["docker", "buildx", "build", "--platform", platform]
    cmd.extend(["-t", tag])
    cmd.extend(["-f", str(dockerfile_path)])
    cmd.extend(["--output", f"type=docker,compression=zstd,compression-level=1,name={tag}"])
    cmd.append(str(context_path))

    click.echo(f"Building image: {tag}")
    click.echo(f"Platform: {platform}")
    click.echo(f"Context: {context_path}")
    click.echo()

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        click.echo("Build failed", err=True)
        raise SystemExit(1)

    click.echo()
    click.echo("Build successful!")
    click.echo(f"Image available locally as: {tag}")

    if push:
        _push_to_registries(tag, region, project)
    elif region:
        click.echo()
        click.echo("To push to registries, run again with --push flag")


@click.group()
def build():
    """Image build commands."""


@build.command("worker-image")
@click.option("--tag", "-t", default="iris-worker:latest", help="Image tag")
@click.option("--push", is_flag=True, help="Push image to registry after building")
@click.option("--dockerfile", type=click.Path(exists=True), help="Custom Dockerfile path")
@click.option("--context", type=click.Path(exists=True), help="Build context directory")
@click.option("--platform", default="linux/amd64", help="Target platform")
@click.option("--region", multiple=True, help="GCP Artifact Registry regions to push to")
@click.option("--project", default="hai-gcp-models", help="GCP project ID for registry")
def build_worker_image(
    tag: str,
    push: bool,
    dockerfile: str | None,
    context: str | None,
    platform: str,
    region: tuple[str, ...],
    project: str,
):
    """Build Docker image for Iris worker."""
    _build_image("worker", tag, push, dockerfile, context, platform, region, project)


@build.command("controller-image")
@click.option("--tag", "-t", default="iris-controller:latest", help="Image tag")
@click.option("--push", is_flag=True, help="Push image to registry after building")
@click.option("--dockerfile", type=click.Path(exists=True), help="Custom Dockerfile path")
@click.option("--context", type=click.Path(exists=True), help="Build context directory")
@click.option("--platform", default="linux/amd64", help="Target platform")
@click.option("--region", multiple=True, help="GCP Artifact Registry regions to push to")
@click.option("--project", default="hai-gcp-models", help="GCP project ID for registry")
def build_controller_image(
    tag: str,
    push: bool,
    dockerfile: str | None,
    context: str | None,
    platform: str,
    region: tuple[str, ...],
    project: str,
):
    """Build Docker image for Iris controller."""
    _build_image("controller", tag, push, dockerfile, context, platform, region, project)


@build.command("task-image")
@click.option("--tag", "-t", default="iris-task:latest", help="Image tag")
@click.option("--push", is_flag=True, help="Push image to registry after building")
@click.option("--dockerfile", type=click.Path(exists=True), help="Custom Dockerfile path")
@click.option("--platform", default="linux/amd64", help="Target platform")
@click.option("--region", multiple=True, help="GCP Artifact Registry regions to push to")
@click.option("--project", default="hai-gcp-models", help="GCP project ID for registry")
def build_task_image(
    tag: str,
    push: bool,
    dockerfile: str | None,
    platform: str,
    region: tuple[str, ...],
    project: str,
):
    """Build base task image with system deps and pre-synced marin core deps.

    The build context is the marin repo root so that pyproject.toml and uv.lock
    are available for COPY. The Dockerfile lives at lib/iris/Dockerfile.task.
    """
    marin_root = _find_marin_root()
    iris_root = _find_iris_root()
    dockerfile_path = Path(dockerfile) if dockerfile else iris_root / "Dockerfile.task"

    if not dockerfile_path.exists():
        raise click.ClickException(f"Dockerfile not found: {dockerfile_path}")

    _build_image(
        "task",
        tag,
        push,
        str(dockerfile_path),
        str(marin_root),
        platform,
        region,
        project,
    )


@build.command("push")
@click.argument("source_tag")
@click.option("--region", "-r", multiple=True, required=True, help="GCP Artifact Registry region")
@click.option("--project", default="hai-gcp-models", help="GCP project ID")
@click.option("--image-name", default="iris-worker", help="Image name in registry")
@click.option("--version", default="latest", help="Version tag")
def build_push(source_tag: str, region: tuple[str, ...], project: str, image_name: str, version: str):
    """Push a local Docker image to GCP Artifact Registry."""
    _push_to_registries(
        source_tag,
        region,
        project,
        image_name=image_name,
        version=version,
    )
