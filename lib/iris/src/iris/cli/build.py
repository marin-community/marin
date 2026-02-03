# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image build commands."""

import subprocess
from pathlib import Path

import click


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
        subprocess.run(
            ["gcloud", "auth", "configure-docker", f"{r}-docker.pkg.dev", "-q"],
            check=False,
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
    cmd.extend(["--load"])
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
