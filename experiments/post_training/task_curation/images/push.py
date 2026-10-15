# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curation image publication settings shared by native and generated Docker contexts."""

import json
import re
import subprocess
from pathlib import Path

# Iris workers pull anonymously; new GHCR packages default to org-internal visibility.
DEFAULT_REPOSITORY = "ghcr.io/marin-community/iris-task"
PLATFORM = "linux/amd64"
IMAGE_COMPRESSION = "gzip"


def push_image(context: Path, tag: str, packages: tuple[Path, ...] = ()) -> str:
    """Build ``context`` for linux/amd64, push it as ``tag``, and return the pushed manifest digest."""
    package_contexts = [f"--build-context={package.name}={package}" for package in packages]
    # Without provenance the pushed reference is one platform manifest rather than an attestation index.
    subprocess.run(
        [
            "docker",
            "buildx",
            "build",
            f"--platform={PLATFORM}",
            "--provenance=false",
            # Export inherited layers with the selected compression too, so OCI
            # consumers can convert the image to Docker's schema2 format.
            f"--output=type=registry,compression={IMAGE_COMPRESSION},force-compression=true",
            f"--tag={tag}",
            *package_contexts,
            str(context),
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
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise ValueError(f"Registry returned an invalid image digest for {tag}")
    return digest
