# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor task packages staged as one archive per package, and read back as one row each.

A package archive holds the package's files at its root, as Harbor Hub archives do, so packages
from a Hugging Face repository, a repository archive and a Hub release share one reader.
"""

import gzip
import hashlib
import io
import tarfile
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO

from huggingface_hub import snapshot_download
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.inputs import ConversionContext

PACKAGE_ARCHIVES = "tasks/*.tar.gz"
TASK_CONFIG = "task.toml"
ARCHIVE_SUFFIX = ".tar.gz"
PACKAGE_ARCHIVE_REVISION = "1"
"""The format revision of ``package_archive``; bump it when the archive bytes for a package change."""
READ_BYTES = 1024 * 1024


def package_archive(directory: Path) -> bytes:
    """A reproducible gzipped tar of ``directory``: sorted entries with no owners, times or group bits.

    Package stores keep only the executable bit, so file modes become 0755 or 0644. Symlinks are
    rejected, as the importer rejects them.
    """
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for path in sorted(directory.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"Harbor package {directory.name} has a symlink: {path.relative_to(directory)}")
                if not path.is_file():
                    continue
                info = tarfile.TarInfo(path.relative_to(directory).as_posix())
                info.size = path.stat().st_size
                info.mode = 0o755 if path.stat().st_mode & 0o100 else 0o644
                with path.open("rb") as content:
                    archive.addfile(info, content)
    return buffer.getvalue()


def stage_repository_packages(
    repo: str, revision: str, output: StoragePath, *, root: str = "", selected: frozenset[str] | None = None
) -> None:
    """Archive packages directly under ``root`` of a pinned Hugging Face repository.

    A directory is a package when it holds ``task.toml``. ``selected`` restricts the package names.
    """
    with TemporaryDirectory(prefix="harbor-repository-") as directory:
        snapshot = Path(snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=directory))
        package_root = (snapshot / root).resolve()
        if not package_root.is_relative_to(snapshot.resolve()):
            raise ValueError(f"Package root escapes the repository: {root!r}")
        packages = sorted(path.parent for path in package_root.glob(f"*/{TASK_CONFIG}"))
        if not packages:
            raise ValueError(f"{repo}@{revision} has no Harbor packages")
        if selected is not None:
            missing = selected - {package.name for package in packages}
            if missing:
                raise ValueError(f"{repo}@{revision} has no Harbor packages {sorted(missing)} under {root!r}")
            packages = [package for package in packages if package.name in selected]
        write_package_archives(packages, output)


class DigestReader:
    """A readable stream that computes the SHA-256 of the bytes read through it."""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def stage_archive_packages(
    archive: BinaryIO, sha256: str, root: str, output: StoragePath, selected: frozenset[str] | None
) -> None:
    """Archive every package directly under ``root`` of a gzipped repository tarball, or the ``selected`` ones.

    The tarball holds the repository under one top-level directory, as a GitHub commit archive does;
    ``root=""`` is the repository's top level. A directory is a package when it holds ``task.toml``.
    The tarball is read as a stream and only ``root`` is kept, because a repository archive can be far
    larger than the packages, and nothing is written unless the tarball's SHA-256 is ``sha256``.
    """
    prefix = f"{root}/" if root else ""
    reader = DigestReader(archive)
    with TemporaryDirectory(prefix="harbor-archive-packages-") as directory:
        extracted = Path(directory)
        with tarfile.open(fileobj=reader, mode="r|gz") as repository:
            for member in repository:
                _, _, path = member.name.partition("/")
                if path.startswith(prefix) and path != prefix:
                    repository.extract(member.replace(name=path.removeprefix(prefix)), extracted, filter="data")
        # The tar reader stops at the end-of-archive marker; the digest covers every byte.
        while reader.read(READ_BYTES):
            pass
        if reader.digest.hexdigest() != sha256:
            raise ValueError(f"The repository archive has SHA-256 {reader.digest.hexdigest()}; expected {sha256}")
        packages = sorted(path.parent for path in extracted.glob(f"*/{TASK_CONFIG}"))
        if not packages:
            raise ValueError(f"The repository archive has no Harbor packages under {root!r}")
        if selected is not None:
            missing = selected - {package.name for package in packages}
            if missing:
                raise ValueError(f"The repository archive has no Harbor packages {sorted(missing)} under {root!r}")
            packages = [package for package in packages if package.name in selected]
        write_package_archives(packages, output)


def write_package_archives(packages: list[Path], output: StoragePath) -> None:
    for package in packages:
        (output / "tasks" / f"{package.name}{ARCHIVE_SUFFIX}").write_bytes(package_archive(package), auto_mkdir=True)


def package_rows(path: StoragePath, _context: ConversionContext) -> Iterator[dict[str, Any]]:
    """One row per package archive: its name and archive bytes."""
    yield {"name": path.name.removesuffix(ARCHIVE_SUFFIX), "archive": path.read_bytes()}


def extract_package(archive: bytes, destination: Path) -> None:
    """Unpack a package archive without allowing files or links outside ``destination``."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
        package.extractall(destination, filter="data")
