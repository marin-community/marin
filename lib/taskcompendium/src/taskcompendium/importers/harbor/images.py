# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve image tags to the registry digests they name now."""

import hashlib
import subprocess
from collections.abc import Iterable

from taskcompendium.importers.harbor.dockerfile import PINNED_IMAGE

DOCKER_HUB = "docker.io"


def registry_reference(reference: str) -> str:
    """Qualify a Docker short name, such as ``python:3.12``, with Docker Hub's registry and namespace."""
    first, separator, _ = reference.partition("/")
    if not separator:
        return f"{DOCKER_HUB}/library/{reference}"
    if "." in first or ":" in first or first == "localhost":
        return reference
    return f"{DOCKER_HUB}/{reference}"


def resolve_image_digest(reference: str) -> str:
    """The ``sha256:`` digest of the manifest or image index ``reference`` names, read with ``skopeo``.

    Hashing the raw manifest bytes yields the content digest, so a multi-platform tag pins its index.
    """
    pinned = PINNED_IMAGE.fullmatch(reference)
    if pinned is not None:
        return reference.rsplit("@", 1)[1]
    try:
        manifest = subprocess.run(
            ("skopeo", "inspect", "--raw", f"docker://{registry_reference(reference)}"),
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Cannot resolve image {reference!r}: {error.stderr.decode(errors='replace')}") from error
    return f"sha256:{hashlib.sha256(manifest).hexdigest()}"


def resolve_image_digests(references: Iterable[str]) -> dict[str, str]:
    """Map each distinct reference to its current digest, for ``lower_package``'s ``image_digests``."""
    return {reference: resolve_image_digest(reference) for reference in dict.fromkeys(references)}
