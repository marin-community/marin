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

"""Docker image build utilities for Iris.

This module provides shared utilities for building and pushing Docker images
for both the Iris worker and controller services.
"""

import subprocess
from pathlib import Path

import click


def find_iris_root() -> Path:
    """Find the iris package root directory containing Dockerfiles.

    Searches for the directory containing Dockerfile.worker and Dockerfile.controller
    in the following order:
    1. Relative to this file (build.py is at src/iris/build.py, so root is 3 levels up)
    2. Current working directory
    3. Walking up from cwd until Dockerfile.worker is found

    Returns:
        Path to the iris root directory.

    Raises:
        click.ClickException: If the iris root cannot be found.
    """
    # Try relative to this file first
    build_path = Path(__file__).resolve()
    # build.py is at src/iris/build.py, so iris root is 3 levels up
    iris_root = build_path.parent.parent.parent
    if (iris_root / "Dockerfile.worker").exists() and (iris_root / "Dockerfile.controller").exists():
        return iris_root

    # Try current working directory
    cwd = Path.cwd()
    if (cwd / "Dockerfile.worker").exists():
        return cwd

    # Walk up from cwd looking for Dockerfile.worker
    for parent in cwd.parents:
        if (parent / "Dockerfile.worker").exists():
            return parent

    raise click.ClickException(
        "Cannot find Dockerfile.worker. Run from the iris directory or specify --dockerfile and --context."
    )


def push_to_registries(
    source_tag: str,
    regions: tuple[str, ...],
    project: str,
    image_name: str | None = None,
    version: str | None = None,
) -> None:
    """Push a local Docker image to multiple GCP Artifact Registry regions.

    Args:
        source_tag: Local Docker image tag to push (e.g., "iris-worker:latest")
        regions: Tuple of GCP Artifact Registry regions (e.g., ("us-central1", "europe-west4"))
        project: GCP project ID
        image_name: Image name in registry (derived from source_tag if None)
        version: Version tag in registry (derived from source_tag if None)
    """
    # Derive defaults if not provided
    if not image_name or not version:
        parts = source_tag.split(":")
        if not image_name:
            image_name = parts[0].split("/")[-1]
        if not version:
            version = parts[1] if len(parts) > 1 else "latest"

    click.echo(f"Pushing {source_tag} to {len(regions)} region(s)...")

    for r in regions:
        dest_tag = f"{r}-docker.pkg.dev/{project}/marin/{image_name}:{version}"

        # Configure docker for this registry
        click.echo(f"\nConfiguring {r}-docker.pkg.dev...")
        subprocess.run(
            ["gcloud", "auth", "configure-docker", f"{r}-docker.pkg.dev", "-q"],
            check=False,
        )

        # Tag image
        click.echo(f"Tagging as {dest_tag}")
        result = subprocess.run(["docker", "tag", source_tag, dest_tag], check=False)
        if result.returncode != 0:
            click.echo(f"Failed to tag image for {r}", err=True)
            continue

        # Push image
        click.echo(f"Pushing to {r}...")
        result = subprocess.run(["docker", "push", dest_tag], check=False)
        if result.returncode != 0:
            click.echo(f"Failed to push to {r}", err=True)
            continue

        click.echo(f"Successfully pushed to {dest_tag}")

    click.echo("\nDone!")


def build_image(
    image_type: str,
    tag: str,
    push: bool,
    dockerfile: str | None,
    context: str | None,
    platform: str,
    region: tuple[str, ...],
    project: str,
) -> None:
    """Build a Docker image for Iris (worker or controller).

    Args:
        image_type: Either "worker" or "controller"
        tag: Docker image tag (e.g., "iris-worker:latest")
        push: Whether to push to registry after building
        dockerfile: Custom Dockerfile path, or None to use default
        context: Build context directory, or None to use iris root
        platform: Target platform (e.g., "linux/amd64")
        region: Tuple of GCP Artifact Registry regions to push to
        project: GCP project ID for registry
    """
    dockerfile_name = f"Dockerfile.{image_type}"

    iris_root = find_iris_root()
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
        push_to_registries(tag, region, project)
    elif region:
        click.echo()
        click.echo("To push to registries, run again with --push flag")
