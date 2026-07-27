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

import subprocess
from collections.abc import Sequence
from pathlib import Path

import click

DEFAULT_IMAGE = "ghcr.io/marin-community/finelog:latest"
DEFAULT_PLATFORM = "linux/amd64"
REGISTRY_COMPRESSION = "compression=zstd,compression-level=3"


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
    additional_tags: Sequence[str] = (),
    push: bool = True,
    platform: str = DEFAULT_PLATFORM,
    cargo_profile: str = "release",
    cache_image: str | None = None,
) -> None:
    """Build the finelog Docker image and (by default) push it to the registry.

    ``image`` should match what the cluster config references; otherwise the
    cluster will keep pulling the old digest. ``push=False`` is useful for
    smoke-testing the Dockerfile locally without registry access.

    Finelog deployments are pinned to amd64 control nodes, so builds default to
    amd64. Callers may request multiple platforms explicitly. Docker cannot load
    a multi-platform image into the local engine, so ``push=False`` builds the
    first requested platform only.

    ``cargo_profile`` selects the Rust build profile baked into the image.
    ``release`` (default) is the optimized fat-LTO production build; ``fast``
    skips LTO for a much quicker final link, suited to dev/test deploys.
    """
    marin_root = find_marin_root()
    dockerfile = marin_root / "lib" / "finelog" / "deploy" / "Dockerfile"
    effective_platform = platform if push else platform.split(",", maxsplit=1)[0]

    cmd = [
        "docker",
        "buildx",
        "build",
        "--platform",
        effective_platform,
        "--file",
        str(dockerfile),
        "--build-arg",
        f"CARGO_PROFILE={cargo_profile}",
        "--tag",
        image,
        "--provenance=false",
    ]
    for tag in additional_tags:
        cmd.extend(["--tag", tag])
    if push:
        if cache_image is not None:
            cmd.extend(
                [
                    "--cache-from",
                    f"type=registry,ref={cache_image}",
                    "--cache-to",
                    f"type=registry,ref={cache_image},mode=max,{REGISTRY_COMPRESSION},"
                    "oci-mediatypes=true,image-manifest=true",
                ]
            )
        cmd.extend(["--output", f"type=image,{REGISTRY_COMPRESSION},push=true"])
    else:
        cmd.extend(["--output", f"type=docker,name={image}"])
    cmd.append(str(marin_root))

    click.echo(f"Building finelog image: {image}")
    click.echo(f"Context: {marin_root}")
    click.echo(f"Cargo profile: {cargo_profile}")
    click.echo(f"Platform: {effective_platform}")
    click.echo(f"Push: {'enabled' if push else 'disabled (local only)'}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise click.ClickException("docker build failed")
    click.echo("Build successful.")
