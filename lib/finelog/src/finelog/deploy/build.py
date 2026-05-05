# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the finelog container image.

Wraps ``docker buildx build`` against ``lib/finelog/deploy/Dockerfile``.
The ``--build`` flag on ``finelog deploy {up,restart}`` calls into this
module so that local edits land in the deployed image without having to
hop through the GitHub Actions release workflow.

The image must be pushed to a registry the deployment can pull from
(default: ``ghcr.io/marin-community/finelog:latest`` — what
``config/marin*.yaml`` references). Use ``docker login ghcr.io`` first.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

DEFAULT_IMAGE = "ghcr.io/marin-community/finelog:latest"


def find_marin_root() -> Path:
    """Locate the marin monorepo root by looking for ``lib/finelog/deploy/Dockerfile``.

    The finelog package may be installed in editable mode (so module path
    points at the repo) or as a wheel (no Dockerfile alongside). We try the
    in-repo location first, then walk up from cwd.
    """
    here = Path(__file__).resolve()
    # finelog/deploy/build.py → ../../.. = lib/finelog/src → up two more = marin root.
    candidate = here.parent.parent.parent.parent.parent
    if (candidate / "lib" / "finelog" / "deploy" / "Dockerfile").is_file():
        return candidate

    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        if (parent / "lib" / "finelog" / "deploy" / "Dockerfile").is_file():
            return parent

    raise click.ClickException(
        "Cannot find marin repo root (lib/finelog/deploy/Dockerfile). " "Run from a marin checkout."
    )


def build_image(
    *,
    image: str = DEFAULT_IMAGE,
    push: bool = True,
    platform: str = "linux/amd64",
) -> None:
    """Build the finelog Docker image and (by default) push it to the registry.

    ``image`` should match what the cluster config references; otherwise the
    cluster will keep pulling the old digest. ``push=False`` is useful for
    smoke-testing the Dockerfile locally without registry access.
    """
    marin_root = find_marin_root()
    dockerfile = marin_root / "lib" / "finelog" / "deploy" / "Dockerfile"

    cmd = [
        "docker",
        "buildx",
        "build",
        "--platform",
        platform,
        "--file",
        str(dockerfile),
        "--tag",
        image,
        "--provenance=false",
    ]
    if push:
        cmd.extend(["--output", "type=image,compression=zstd,compression-level=3,push=true"])
    else:
        cmd.extend(["--output", f"type=docker,name={image}"])
    cmd.append(str(marin_root))

    click.echo(f"Building finelog image: {image}")
    click.echo(f"Context: {marin_root}")
    click.echo(f"Push: {'enabled' if push else 'disabled (local only)'}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise click.ClickException("docker build failed")
    click.echo("Build successful.")
