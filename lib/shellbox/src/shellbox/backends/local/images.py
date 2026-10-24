# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare private writable OCI root filesystems before local guest execution."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from shellbox.image import OCI_TAG, PreparedImage, prepared_image_config


@dataclass(frozen=True)
class PreparedLocalImage:
    """One caller-owned unpacked image and its OCI process defaults."""

    root: Path
    environment: dict[str, str]
    workdir: str
    entrypoint: tuple[str, ...]
    arguments: tuple[str, ...]
    user: str | None


def prepare_local_image(image: PreparedImage, directory: Path, *, umoci: Path) -> PreparedLocalImage:
    """Unpack a pinned image into a new private bundle using rootless umoci.

    The caller owns the entire bundle directory, including cleanup after failure.
    Image resolution and builds belong to trusted host preparation, before this
    step; no container daemon or image fetching runs in the guest.
    """
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(directory)
    config = prepared_image_config(image)
    if config["os"] != "linux" or config["architecture"] != "amd64":
        raise ValueError("Only Linux amd64 images are supported")
    process = config.get("config") or {}
    environment = {}
    for entry in process.get("Env") or ():
        name, separator, value = entry.partition("=")
        if not separator or not name or "\0" in entry:
            raise ValueError("OCI environment entries must be NAME=value strings")
        environment[name] = value
    workdir = process.get("WorkingDir") or "/"
    if not PurePosixPath(workdir).is_absolute() or "\0" in workdir:
        raise ValueError("OCI working directory must be absolute")
    subprocess.run(
        [str(umoci), "unpack", "--rootless", "--image", f"{image.layout}:{OCI_TAG}", str(directory)],
        check=True,
        capture_output=True,
    )
    metadata = json.loads((directory / "umoci.json").read_text())
    unpacked_digest = metadata["from_descriptor_path"]["descriptor_walk"][-1]["digest"]
    if unpacked_digest != image.manifest_digest:
        raise ValueError("Unpacked image differs from its pinned manifest")
    return PreparedLocalImage(
        root=directory / "rootfs",
        environment=environment,
        workdir=workdir,
        entrypoint=tuple(process.get("Entrypoint") or ()),
        arguments=tuple(process.get("Cmd") or ()),
        user=process.get("User") or None,
    )
