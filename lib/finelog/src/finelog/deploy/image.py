# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a container image tag to its immutable content digest.

Deploy backends call ``resolve_image_digest`` to pin ``cfg.image`` (often a
mutable ``:latest``) to a ``...@sha256:...`` reference at deploy time, so a
redeploy lands the exact image the tag points to right now — no node-cache
staleness, and the rendered manifest records precisely what is running.
"""

import json
import subprocess

import click


def resolve_image_digest(image: str) -> str:
    """Pin a tag to its content digest via ``docker buildx imagetools inspect``.

    Returns ``ghcr.io/...@sha256:...`` on success, or the original tag with a
    warning on any failure (no docker CLI, no network, private registry, etc.).
    """
    if "@sha256:" in image:
        return image
    if ":" not in image.rsplit("/", 1)[-1]:
        # No tag at all — leave it alone; the deploy bootstrap will resolve.
        return image
    repo, _, _ = image.rpartition(":")
    try:
        result = subprocess.run(
            [
                "docker",
                "buildx",
                "imagetools",
                "inspect",
                image,
                "--format",
                "{{json .Manifest}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        click.echo(f"warning: could not resolve digest for {image} ({e}); using tag", err=True)
        return image
    if result.returncode != 0:
        stderr_msg = result.stderr.strip()[:200]
        click.echo(
            f"warning: `docker buildx imagetools inspect` failed for {image}: {stderr_msg}; using tag",
            err=True,
        )
        return image
    digest = _extract_digest(result.stdout)
    if not digest:
        click.echo(f"warning: could not parse digest from manifest of {image}; using tag", err=True)
        return image
    return f"{repo}@{digest}"


def _extract_digest(manifest_json: str) -> str | None:
    """Pull the deployable manifest or image-index digest from Buildx JSON output."""
    try:
        parsed = json.loads(manifest_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed.get("digest")
