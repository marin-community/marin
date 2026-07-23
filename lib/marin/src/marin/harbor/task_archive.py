# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Parquet archives for Harbor task directories."""

import io
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from gzip import GzipFile
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

TASK_MARKER = "instruction.md"


@dataclass(frozen=True)
class TaskArchive:
    """A materialized task archive and its task count."""

    parquet_path: Path
    task_count: int


def discover_task_dirs(root: Path, recursive: bool = True) -> list[Path]:
    """Return task directories marked by instruction.md in stable path order."""

    candidates = root.rglob(TASK_MARKER) if recursive else root.glob(f"*/{TASK_MARKER}")
    return sorted(marker.parent for marker in candidates if marker.is_file())


def _archive_task(task_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for path in sorted(task_dir.rglob("*"), key=lambda item: item.relative_to(task_dir).as_posix()):
                if path.is_symlink():
                    raise ValueError(f"Task archives do not permit symlinks: {path}")
                relative = path.relative_to(task_dir).as_posix()
                archive.add(path, arcname=relative, recursive=False, filter=_normalize_tar_metadata)
    return buffer.getvalue()


def _normalize_tar_metadata(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Remove host-specific metadata so identical tasks have identical archives."""

    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def write_task_archive(tasks_dir: Path, parquet_path: Path, recursive: bool = True) -> TaskArchive:
    """Write task directories into path/task_binary Parquet rows."""

    task_dirs = discover_task_dirs(tasks_dir, recursive=recursive)
    rows = [
        {"path": task_dir.relative_to(tasks_dir).as_posix(), "task_binary": _archive_task(task_dir)}
        for task_dir in task_dirs
    ]
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=pa.schema([("path", pa.string()), ("task_binary", pa.binary())]))
    pq.write_table(table, parquet_path, compression="zstd")
    return TaskArchive(parquet_path=parquet_path, task_count=len(rows))


def _safe_member_path(destination: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe task archive member: {name}")
    target = destination.joinpath(*relative.parts)
    if destination not in target.resolve().parents and target.resolve() != destination:
        raise ValueError(f"Unsafe task archive member: {name}")
    return target


def _extract_task(task_binary: bytes, destination: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            _safe_member_path(destination, member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Unsafe task archive member type: {member.name}")
        archive.extractall(destination, members=members, filter="data")


def extract_task_archive(parquet_path: Path, destination: Path, replace: bool = False) -> list[Path]:
    """Safely materialize a task Parquet archive under destination."""

    table = pq.read_table(parquet_path, columns=["path", "task_binary"])
    rows = table.to_pylist()
    extracted: list[Path] = []
    destination.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = PurePosixPath(row["path"])
        if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
            raise ValueError(f"Unsafe task path: {row['path']}")
        task_destination = destination.joinpath(*path.parts)
        if task_destination.exists() and not replace:
            raise FileExistsError(f"Task destination already exists: {task_destination}")
        if task_destination.exists():
            shutil.rmtree(task_destination)
        task_destination.mkdir(parents=True, exist_ok=True)
        _extract_task(row["task_binary"], task_destination)
        if not (task_destination / TASK_MARKER).is_file():
            raise ValueError(f"Task archive row lacks {TASK_MARKER}: {row['path']}")
        extracted.append(task_destination)
    return extracted


def upload_task_archive(
    archive: TaskArchive,
    *,
    repo_id: str,
    private: bool,
    token: str | None = None,
) -> str:
    """Publish a completed task archive as a Hugging Face dataset."""

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    result = api.upload_file(
        path_or_fileobj=str(archive.parquet_path),
        path_in_repo="tasks.parquet",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload Harbor task archive",
    )
    return str(result)


def archive_tasks_in_temporary_directory(tasks_dir: Path) -> tuple[TaskArchive, tempfile.TemporaryDirectory[str]]:
    """Create an archive whose caller owns the returned temporary directory."""

    temporary_directory = tempfile.TemporaryDirectory(prefix="marin-harbor-tasks-")
    archive = write_task_archive(tasks_dir, Path(temporary_directory.name) / "tasks.parquet")
    return archive, temporary_directory
