# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish preserved FACET OCI images without exposing them publicly."""

import hashlib
import subprocess
from collections.abc import Callable
from pathlib import Path


def require_nonpublic_visibility(visibility: str) -> str:
    """Accept the user-authorized organization or private package scope."""
    if visibility not in {"internal", "private"}:
        raise ValueError(f"FACET image package must be internal or private, observed {visibility!r}")
    return visibility


def publish_image(
    layout: Path,
    repository: str,
    tag: str,
    manifest_digest: str,
    auth_file: Path,
    visibility: Callable[[], str],
) -> str:
    """Check package scope before copying, then verify scope and the exact OCI digest."""
    require_nonpublic_visibility(visibility())
    subprocess.run(
        [
            "skopeo",
            "copy",
            "--preserve-digests",
            "--authfile",
            str(auth_file),
            f"oci:{layout}:image",
            f"docker://{repository}:{tag}",
        ],
        check=True,
    )
    observed = require_nonpublic_visibility(visibility())
    manifest = subprocess.check_output(
        ["skopeo", "inspect", "--raw", "--authfile", str(auth_file), f"docker://{repository}@{manifest_digest}"]
    )
    if "sha256:" + hashlib.sha256(manifest).hexdigest() != manifest_digest:
        raise ValueError("Published FACET OCI manifest differs from its preserved digest")
    return observed
