"""Inspect captured root filesystems without extracting or executing their contents.

Compressed-byte integrity is handled by ``oci_artifact.validate_layer``. This
separate gate checks the files that will actually be published, including the
public task copies and provider state that must not survive a capture.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import posixpath
import re
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import BinaryIO


class RootfsReviewError(ValueError):
    pass


# Empty mountpoint directories may exist, but no file, link or populated subtree
# from these locations may enter a portable task image. File bind mounts require
# explicit exclusion: a directory-recursion flag alone is not sufficient proof.
PROVIDER_STATE_PATHS = (
    "/proc",
    "/sys",
    "/dev",
    "/run",
    "/tmp",
    "/root/.daytona",
    "/.dockerenv",
    "/etc/hostname",
    "/etc/hosts",
    "/etc/resolv.conf",
    "/usr/local/bin/daytona",
    "/usr/local/lib/daytona-computer-use",
    "/etc/daytona",
    "/var/lib/docker",
    "/var/lib/kubelet",
    "/var/lib/k0s",
    "/var/lib/rancher",
    "/var/lib/buildkit",
    "/var/lib/containerd",
    "/var/lib/postgresql/data",
)


def _path(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or "\0" in name:
        raise RootfsReviewError("unsafe archive member path")
    return "/" if str(path) == "." else "/" + str(path)


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _rejected(path: str, patterns: Sequence[str]) -> bool:
    return any(
        fnmatch.fnmatchcase(path, pattern)
        or (not any(char in pattern for char in "*?[") and _under(path, pattern))
        for pattern in patterns
    )


def review_rootfs(
    stream: BinaryIO,
    *,
    expected_files: Mapping[str, str],
    reject_paths: Sequence[str],
    task_roots: Sequence[str],
    content_markers: Sequence[str] = (),
    reject_hashes: Sequence[str] = (),
    excluded_paths: Sequence[str] = PROVIDER_STATE_PATHS,
    max_members: int = 500_000,
    max_file_bytes: int = 20 << 30,
) -> dict:
    """Return a content inventory receipt, or reject the captured gzip tar.

    All expected files must be regular members with exactly the reviewed contents.
    Task namespaces are closed to additional payload files and links; base-image
    files outside those namespaces are inventoried without being executed. Marker
    checks are scoped to task namespaces to avoid confusing base package examples
    with task-private artifacts. Memory is bounded by member metadata and 1 MiB.
    """
    if not expected_files or any(
        not path.startswith("/")
        or str(PurePosixPath(path)) != path
        or ".." in PurePosixPath(path).parts
        or not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for path, value in expected_files.items()
    ):
        raise RootfsReviewError("invalid required file hashes")
    if (
        type(max_members) is not int
        or max_members <= 0
        or type(max_file_bytes) is not int
        or max_file_bytes <= 0
    ):
        raise RootfsReviewError("invalid archive limits")
    if any(not marker for marker in content_markers):
        raise RootfsReviewError("empty forbidden content marker")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in reject_hashes):
        raise RootfsReviewError("invalid private content hashes")
    markers = [marker.encode() for marker in content_markers]
    overlap = max((len(marker) - 1 for marker in markers), default=0)
    seen: dict[str, str] = {}
    matched: dict[str, str] = {}
    inventory = hashlib.sha256()
    total_bytes = 0
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                path = _path(member.name)
                if path in seen:
                    raise RootfsReviewError(f"duplicate archive member: {path}")
                if len(seen) >= max_members:
                    raise RootfsReviewError("archive member limit exceeded")
                if _rejected(path, reject_paths):
                    raise RootfsReviewError(f"private path in captured image: {path}")
                for root in excluded_paths:
                    if _under(path, root) and not (path == root and member.isdir()):
                        raise RootfsReviewError(
                            f"runtime or provider state in image: {path}"
                        )
                in_task = any(_under(path, root) for root in task_roots)
                if in_task and not member.isdir() and path not in expected_files:
                    raise RootfsReviewError(f"undeclared task payload: {path}")
                if member.isdir():
                    kind = "directory"
                elif member.isfile():
                    kind = "file"
                elif member.issym() or member.islnk():
                    kind = "symlink" if member.issym() else "hardlink"
                else:
                    raise RootfsReviewError(f"unsupported archive member: {path}")
                seen[path] = kind
                record = {
                    "path": path,
                    "kind": kind,
                    "size": member.size,
                    "mode": member.mode,
                    "uid": member.uid,
                    "gid": member.gid,
                }
                if path in expected_files and kind != "file":
                    raise RootfsReviewError(
                        f"required file is not a regular member: {path}"
                    )
                if kind in {"symlink", "hardlink"}:
                    target = member.linkname
                    if "\0" in target or "\\" in target:
                        raise RootfsReviewError(f"unsafe link: {path}")
                    # Absolute symlinks are ordinary in Linux root filesystems.
                    # Resolve them within the image, never on the host.
                    resolved = posixpath.normpath(
                        target
                        if target.startswith("/")
                        else posixpath.join(
                            posixpath.dirname(path) if kind == "symlink" else "/",
                            target,
                        )
                    )
                    if _rejected(resolved, reject_paths):
                        raise RootfsReviewError(f"link to private path: {path}")
                    record["target"] = target
                if kind == "file":
                    total_bytes += member.size
                    if total_bytes > max_file_bytes:
                        raise RootfsReviewError("archive file-byte limit exceeded")
                    payload = archive.extractfile(member)
                    if payload is None:
                        raise RootfsReviewError(f"unreadable archive member: {path}")
                    checksum = hashlib.sha256()
                    size = 0
                    tail = b""
                    while chunk := payload.read(1 << 20):
                        checksum.update(chunk)
                        size += len(chunk)
                        if in_task and markers:
                            window = tail + chunk
                            if any(marker in window for marker in markers):
                                raise RootfsReviewError(
                                    f"private marker in task payload: {path}"
                                )
                            tail = window[-overlap:] if overlap else b""
                    if size != member.size:
                        raise RootfsReviewError(f"truncated archive member: {path}")
                    record["sha256"] = checksum.hexdigest()
                    if record["sha256"] in reject_hashes:
                        raise RootfsReviewError(f"private content in captured image: {path}")
                    if path in expected_files:
                        if record["sha256"] != expected_files[path]:
                            raise RootfsReviewError(
                                f"required content mismatch: {path}"
                            )
                        matched[path] = record["sha256"]
                inventory.update(
                    json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                    + b"\n"
                )
    except (tarfile.TarError, EOFError, OSError) as error:
        raise RootfsReviewError("invalid rootfs tar archive") from error
    if set(matched) != set(expected_files):
        raise RootfsReviewError("captured image lacks required files")
    for path in seen:
        for parent in PurePosixPath(path).parents:
            if seen.get(str(parent)) in {"file", "symlink", "hardlink"}:
                raise RootfsReviewError(
                    f"archive member traverses a nondirectory: {path}"
                )
    return {
        "schema_version": "capability-rootfs-review-v1",
        "state": "passed",
        "members": len(seen),
        "file_bytes": total_bytes,
        "inventory_sha256": inventory.hexdigest(),
        "required_file_hashes": matched,
        "compressed_integrity": "requires_separate_validate_layer",
        "task_acceptance": "not_evaluated",
    }
