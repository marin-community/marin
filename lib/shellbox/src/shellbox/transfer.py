# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded file and directory downloads across shell machine backends."""

import io
import tarfile
from collections.abc import Sequence
from pathlib import PurePosixPath

from shellbox.machine import Command, Machine

_MISSING_PATH_EXIT = 44
_LINKED_PATH_EXIT = 45
_METADATA_BYTES = 1024 * 1024


async def download_files(
    machine: Machine,
    paths: Sequence[str],
    *,
    timeout: float | None,
    max_files: int,
    max_bytes: int,
    max_file_bytes: int,
) -> dict[str, bytes]:
    """Download regular files, recursively expanding directories into absolute-path keys.

    Missing paths are omitted. Explicit selections must not contain symlinks; interior
    links are skipped. Limits apply to the complete result, and overflow returns no partial
    result. The machine's execution user reads the files.
    """
    files: dict[str, bytes] = {}
    total_bytes = 0
    archive_limit = max_bytes + _METADATA_BYTES + max_files * 1024
    for path in paths:
        root = PurePosixPath(path)
        if not root.is_absolute() or ".." in root.parts or root.as_posix() != path:
            raise ValueError(f"Download path must be normalized and absolute: {path}")
        inspected = await machine.run(
            Command(
                (
                    "sh",
                    "-c",
                    'path=${1%/}; while [ "$path" ] && [ "$path" != / ]; do '
                    f'[ ! -L "$path" ] || exit {_LINKED_PATH_EXIT}; '
                    'case "$path" in */*) path=${path%/*};; *) break;; esac; done; '
                    'if [ -d "$1" ]; then printf directory; elif [ -f "$1" ]; then printf file; '
                    f"else exit {_MISSING_PATH_EXIT}; fi",
                    "download-kind",
                    path,
                ),
                timeout=timeout,
            )
        )
        if inspected.exit_code == _MISSING_PATH_EXIT:
            continue
        if inspected.exit_code != 0 or inspected.stdout_truncated:
            raise RuntimeError(f"Cannot inspect download path (links are not allowed): {path}")
        if inspected.stdout not in (b"directory", b"file"):
            raise RuntimeError(f"Invalid download path kind: {path}")
        directory = inspected.stdout == b"directory"
        base = root if directory else root.parent
        # Tar interprets backslash escapes even in argv filenames.
        archive_name = "." if directory else root.name.replace("\\", "\\\\")
        result = await machine.run(
            Command(
                ("tar", "-cf", "-", "-C", str(base), "--", archive_name),
                timeout=timeout,
                output_limit_bytes=archive_limit + 1,
            )
        )
        if result.exit_code != 0 or result.stdout_truncated or len(result.stdout) > archive_limit:
            raise RuntimeError(
                f"Download unavailable or exceeds archive budget: {path}; "
                f"{result.stderr.decode(errors='replace')[-2000:]}"
            )
        try:
            with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
                archived_files: dict[str, bytes] = {}
                for member in archive:
                    relative = PurePosixPath(member.name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"Invalid download archive member: {member.name}")
                    if member.isdir() or member.issym():
                        continue
                    if not (member.isfile() or member.islnk()) or member.size > max_file_bytes:
                        raise RuntimeError(f"Download contains an invalid or oversized file: {member.name}")
                    target = (base / relative).as_posix()
                    if not directory and target != path:
                        raise ValueError(f"Download archive escaped selected file: {target}")
                    data = None
                    size = member.size
                    if member.islnk():
                        linked = PurePosixPath(member.linkname)
                        if linked.is_absolute() or ".." in linked.parts:
                            raise ValueError(f"Invalid download archive link: {member.linkname}")
                        linked_path = (base / linked).as_posix()
                        if linked_path not in archived_files:
                            raise RuntimeError(f"Download archive links to an unavailable file: {member.linkname}")
                        data = archived_files[linked_path]
                        size = len(data)
                    if target not in files and (len(files) >= max_files or total_bytes + size > max_bytes):
                        raise RuntimeError("Download exceeds its file or byte budget")
                    if data is None:
                        stream = archive.extractfile(member)
                        assert stream is not None
                        data = stream.read()
                    archived_files[target] = data
                    if target not in files:
                        files[target] = data
                        total_bytes += size
        except tarfile.TarError as error:
            raise RuntimeError(f"Invalid download archive: {path}") from error
    return files
