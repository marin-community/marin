# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge guest downloads into host destinations without following guest links."""

import io
import os
import shutil
import stat
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath

DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _guest_member(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo:
    if member.islnk() or member.isdev():
        return tarfile.data_filter(member, path)
    filtered = tarfile.tar_filter(member, path)
    if filtered.isdir() and filtered.mode is not None:
        # Keep the staging tree traversable and writable for repeated merges,
        # without granting group or other write access from guest metadata.
        filtered = filtered.replace(mode=(filtered.mode & 0o755) | 0o700, deep=False)
    return filtered


def _metadata(source: Path, descriptor: int) -> None:
    metadata = source.stat()
    os.fchmod(descriptor, stat.S_IMODE(metadata.st_mode))
    os.utime(descriptor, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))


def _merge(source: Path, parent: int, name: str) -> None:
    metadata = source.lstat()
    if stat.S_ISDIR(metadata.st_mode):
        try:
            destination = os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            destination = None
        if destination is not None and stat.S_ISLNK(destination.st_mode):
            os.unlink(name, dir_fd=parent)
        created = False
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent)
        except FileExistsError:
            pass
        else:
            created = True
        descriptor = os.open(name, DIRECTORY_FLAGS, dir_fd=parent)
        try:
            for entry in source.iterdir():
                _merge(entry, descriptor, entry.name)
            if created:
                _metadata(source, descriptor)
        finally:
            os.close(descriptor)
        return

    temporary = f".shellbox-download-{uuid.uuid4().hex}"
    try:
        if stat.S_ISLNK(metadata.st_mode):
            os.symlink(os.readlink(source), temporary, dir_fd=parent)
        elif stat.S_ISREG(metadata.st_mode):
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600, dir_fd=parent)
            with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
                output.flush()
                _metadata(source, output.fileno())
        else:
            raise ValueError("Guest downloads may contain only directories, regular files, and symlinks")
        # Replacing the directory entry also avoids writing through an existing
        # regular file that has a hard link elsewhere on the host.
        os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
    finally:
        try:
            os.unlink(temporary, dir_fd=parent)
        except FileNotFoundError:
            pass


def _merge_destination(source: Path, target: Path) -> None:
    target = target.absolute()
    if not target.name:
        raise ValueError("A download destination must not be the host filesystem root")
    descriptor = os.open(target.anchor, DIRECTORY_FLAGS)
    try:
        for component in target.parent.parts[1:]:
            try:
                os.mkdir(component, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(component, DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        try:
            destination = os.stat(target.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            destination = None
        if destination is not None and stat.S_ISLNK(destination.st_mode):
            raise ValueError("A download destination must not be a symlink")
        if source.is_file() and destination is not None and stat.S_ISDIR(destination.st_mode):
            child = os.open(target.name, DIRECTORY_FLAGS, dir_fd=descriptor)
            try:
                _merge(source, child, source.name)
            finally:
                os.close(child)
            return
        _merge(source, descriptor, target.name)
    finally:
        os.close(descriptor)


def unpack_local_archive(stream: bytes, target: Path) -> None:
    """Merge a guest tar stream into a caller-selected host file or directory.

    The destination and its ancestors must not be symlinks. Interior symlinks
    remain links when downloaded, and later downloads replace rather than follow
    them. Existing unrelated destination files remain in place.
    Existing directory modes stay unchanged; new directories retain owner access
    and cannot acquire group or other write permissions from guest metadata.
    """
    with tempfile.TemporaryDirectory() as staging, tarfile.open(fileobj=io.BytesIO(stream)) as archive:
        members = archive.getmembers()
        directory = bool(members) and members[0].name == "." and members[0].isdir()
        if not members or not (directory or members[0].isfile()):
            raise ValueError("The machine's archive must start with a regular file or directory")
        if not directory and PurePosixPath(members[0].name).name != members[0].name:
            raise ValueError("A single-file archive must start with its file's basename")
        archive.extractall(staging, filter=_guest_member)
        source = Path(staging) if directory else Path(staging, members[0].name)
        if source.is_symlink() or not (source.is_dir() if directory else source.is_file()):
            raise ValueError("The machine's archive changed its selected root type")
        _merge_destination(source, target)
