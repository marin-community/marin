# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare private writable OCI root filesystems before local guest execution."""

import json
import os
import re
import shutil
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


def prepare_broker_runtime(directory: Path, *, python: Path, bwrap: Path, identity: tuple[int, int]) -> None:
    """Stage Python, bubblewrap and their ELF dependencies at guest-private paths."""
    stdlib = Path(
        subprocess.run(
            [str(python), "-I", "-S", "-c", "import sysconfig; print(sysconfig.get_path('stdlib'))"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if not stdlib.is_dir():
        raise FileNotFoundError(f"Python standard library is missing: {stdlib}")
    (directory / "bin").mkdir(parents=True)
    (directory / "libdeps").mkdir()
    shutil.copy2(python.resolve(), directory / "bin/python3")
    shutil.copy2(bwrap.resolve(), directory / "bin/bwrap")
    shutil.copytree(
        stdlib,
        directory / "lib" / stdlib.name,
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", "site-packages", "dist-packages", "test", "tests"),
    )
    loader = None
    sources = (python.resolve(), bwrap.resolve(), *stdlib.glob("lib-dynload/*.so"))
    for source in sources:
        dependencies = subprocess.run(["ldd", str(source)], check=True, capture_output=True, text=True).stdout
        if "not found" in dependencies:
            raise FileNotFoundError(f"Missing shared library for {source}: {dependencies}")
        for name in re.findall(r"(?<!\S)/[^\s()]+", dependencies):
            library = Path(name)
            if not library.is_file():
                continue
            if library.name.startswith("ld-linux"):
                loader = library
                continue
            target = directory / "libdeps" / library.name
            if not target.exists():
                shutil.copy2(library, target)
    if loader is None:
        raise RuntimeError("Cannot locate the Python ELF loader")
    shutil.copy2(loader, directory / "ld-linux")
    for current, children, files in os.walk(directory):
        os.chown(current, *identity)
        for name in (*children, *files):
            os.lchown(os.path.join(current, name), *identity)


def prepare_local_image(
    image: PreparedImage, directory: Path, *, umoci: Path, owner: tuple[int, int] | None = None
) -> PreparedLocalImage:
    """Unpack a pinned image into a new private bundle using rootless umoci.

    The caller owns the entire bundle directory, including cleanup after failure.
    Image resolution and builds belong to trusted host preparation, before this
    step; no container daemon or image fetching runs in the guest. ``owner`` maps
    the image's root account to a chosen host UID and GID.
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
    arguments = [str(umoci), "unpack", "--rootless"]
    if owner is not None:
        uid, gid = owner
        arguments.extend((f"--uid-map=0:{uid}:1", f"--gid-map=0:{gid}:1"))
    arguments.extend(("--image", f"{image.layout}:{OCI_TAG}", str(directory)))
    try:
        subprocess.run(
            arguments,
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"OCI image preparation requires umoci ({umoci}); install it on the host "
            "or pass LocalMachineFactory(umoci=Path(...))"
        ) from error
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
