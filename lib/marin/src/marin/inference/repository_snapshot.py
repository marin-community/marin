# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Clone bounded public GitHub history for ShellSim workspaces."""

import asyncio
import json
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from marin.inference.shell_workspace import (
    MAX_COMMIT_METADATA_BYTES,
    MAX_IMPORTED_COMMITS,
    MAX_IMPORTED_HISTORY_BYTES,
    MAX_WORKSPACE_FILE_BYTES,
    MAX_WORKSPACE_FILES,
    MAX_WORKSPACE_TOTAL_BYTES,
    ImportedGitCommit,
    imported_git_history_bytes,
    is_workspace_path,
)

MAX_REPOSITORY_CLONE_BYTES = 64 * 1024 * 1024
REPOSITORY_CLONE_TIMEOUT = 30
_GITHUB_HOST = "github.com"
_IGNORED_PARTS = frozenset({".git", ".venv", "venv", "node_modules", "target", "__pycache__"})

CloneRepository = Callable[[str, Path], None]


class RepositorySnapshotTooLarge(ValueError):
    """Raised when a repository snapshot exceeds a workspace bound."""


class RepositoryCloneError(RuntimeError):
    """Raised when Git cannot clone or inspect a public repository."""


@dataclass(frozen=True)
class RepositorySnapshot:
    """Text files and recent commits from a repository's default branch."""

    files: dict[str, str]
    commits: tuple[ImportedGitCommit, ...]
    skipped_files: int
    truncated_history: bool


@dataclass(frozen=True)
class _CommitSnapshot:
    message: str
    author_name: str
    author_email: str
    files: dict[str, str]


@dataclass(frozen=True)
class _TextSnapshot:
    files: dict[str, str]
    skipped_files: int


@dataclass(frozen=True)
class _BoundedHistory:
    commits: tuple[ImportedGitCommit, ...]
    truncated: bool


def github_repository_slug(value: object) -> tuple[str, str]:
    """Return the owner and repository from a canonical public GitHub URL."""
    if not isinstance(value, str):
        raise ValueError("Repository URL must be a string")
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme != "https"
        or parsed.hostname != _GITHUB_HOST
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Repository URL must have the form https://github.com/owner/repository")
    parts = parsed.path.strip("/").split("/")
    if len(parts) != 2 or not all(_valid_github_name(part) for part in parts):
        raise ValueError("Repository URL must have the form https://github.com/owner/repository")
    owner, repository = parts
    if repository.endswith(".git"):
        repository = repository[:-4]
    if not repository:
        raise ValueError("Repository name may not be empty")
    return owner, repository


def _valid_github_name(value: str) -> bool:
    return bool(value) and all(character.isalnum() or character in "-_." for character in value)


async def clone_repository_snapshot(
    url: object, *, clone_repository: CloneRepository | None = None
) -> RepositorySnapshot:
    """Clone one public GitHub repository and extract bounded recent history."""
    owner, repository = github_repository_slug(url)
    clone_url = f"https://{_GITHUB_HOST}/{owner}/{repository}.git"
    clone = clone_repository or _clone_repository
    return await asyncio.to_thread(_clone_and_extract, clone_url, clone)


def _clone_and_extract(clone_url: str, clone_repository: CloneRepository) -> RepositorySnapshot:
    with tempfile.TemporaryDirectory(prefix="marin-chat-repository-") as temporary_directory:
        repository_path = Path(temporary_directory, "repository")
        clone_repository(clone_url, repository_path)
        clone_bytes = sum(path.stat().st_size for path in repository_path.rglob("*") if path.is_file())
        if clone_bytes > MAX_REPOSITORY_CLONE_BYTES:
            raise RepositorySnapshotTooLarge(f"Repository clone exceeds {MAX_REPOSITORY_CLONE_BYTES} bytes")
        return repository_snapshot_from_git(repository_path)


def _clone_repository(clone_url: str, destination: Path) -> None:
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    environment.update(
        {
            "GIT_ASKPASS": "/bin/false",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    try:
        process = subprocess.Popen(
            [
                "git",
                "-c",
                "protocol.version=2",
                "-c",
                "credential.helper=",
                "-c",
                "http.extraHeader=",
                "clone",
                "--quiet",
                "--no-checkout",
                "--single-branch",
                "--no-tags",
                f"--depth={MAX_IMPORTED_COMMITS + 1}",
                f"--filter=blob:limit={MAX_WORKSPACE_FILE_BYTES + 1}",
                "--",
                clone_url,
                str(destination),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise RepositoryCloneError("Git is unavailable on the dashboard server") from exc

    deadline = time.monotonic() + REPOSITORY_CLONE_TIMEOUT
    while process.poll() is None:
        if _directory_bytes(destination) > MAX_REPOSITORY_CLONE_BYTES:
            _kill_clone(process)
            raise RepositorySnapshotTooLarge(f"Repository clone exceeds {MAX_REPOSITORY_CLONE_BYTES} bytes")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_clone(process)
            raise RepositoryCloneError(f"GitHub repository clone exceeded {REPOSITORY_CLONE_TIMEOUT} seconds")
        try:
            _, stderr = process.communicate(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            continue
        break
    else:
        _, stderr = process.communicate()

    if process.returncode != 0:
        detail = stderr.decode(errors="replace").strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise RepositoryCloneError(f"Could not clone public GitHub repository{suffix}")


def _directory_bytes(root: Path) -> int:
    total_bytes = 0
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            try:
                total_bytes += os.stat(Path(directory, filename), follow_symlinks=False).st_size
            except FileNotFoundError:
                continue
            if total_bytes > MAX_REPOSITORY_CLONE_BYTES:
                return total_bytes
    return total_bytes


def _kill_clone(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.communicate()


def repository_snapshot_from_git(repository_path: Path) -> RepositorySnapshot:
    """Extract a bounded first-parent text history from a local Git repository."""
    revisions = _git(repository_path, "rev-list", "--first-parent", f"--max-count={MAX_IMPORTED_COMMITS + 1}", "HEAD")
    commit_ids = revisions.decode().splitlines()
    truncated_history = len(commit_ids) > MAX_IMPORTED_COMMITS
    commit_ids = list(reversed(commit_ids[:MAX_IMPORTED_COMMITS]))
    blob_cache: dict[str, str | None] = {}
    if not commit_ids:
        return RepositorySnapshot(files={}, commits=(), skipped_files=0, truncated_history=False)

    head = _text_snapshot(repository_path, commit_ids[-1], blob_cache)
    reversed_snapshots: list[_CommitSnapshot] = []
    for commit_id in reversed(commit_ids):
        try:
            reversed_snapshots.append(_commit_snapshot(repository_path, commit_id, blob_cache))
        except RepositorySnapshotTooLarge:
            if commit_id == commit_ids[-1]:
                raise
            truncated_history = True
            break
    snapshots = list(reversed(reversed_snapshots))
    history = _bounded_history(snapshots)
    return RepositorySnapshot(
        files=head.files,
        commits=history.commits,
        skipped_files=head.skipped_files,
        truncated_history=truncated_history or history.truncated,
    )


def _commit_snapshot(repository_path: Path, commit_id: str, blob_cache: dict[str, str | None]) -> _CommitSnapshot:
    metadata = _git(repository_path, "show", "-s", "--format=%s%x00%an%x00%ae", commit_id)
    try:
        message, author_name, author_email = metadata.decode(errors="replace").rstrip("\n").split("\x00")
    except ValueError as exc:
        raise RepositoryCloneError("Git returned invalid commit metadata") from exc
    snapshot = _text_snapshot(repository_path, commit_id, blob_cache)
    return _CommitSnapshot(
        message=_bounded_metadata(message, commit_id[:12]),
        author_name=_bounded_metadata(
            author_name.translate(str.maketrans({"<": "(", ">": ")", "\n": " "})), "Imported Author"
        ),
        author_email=_bounded_metadata(
            author_email.translate(str.maketrans({"<": "", ">": "", "\n": ""})), "imported@example.invalid"
        ),
        files=snapshot.files,
    )


def _text_snapshot(repository_path: Path, commit_id: str, blob_cache: dict[str, str | None]) -> _TextSnapshot:
    entries = _git(repository_path, "ls-tree", "-r", "-z", "--long", commit_id).split(b"\x00")
    files: dict[str, str] = {}
    skipped_files = 0
    total_bytes = 0
    for entry in entries:
        if not entry:
            continue
        try:
            metadata, raw_path = entry.split(b"\t", maxsplit=1)
            mode, kind, object_id, size_text = metadata.decode().split()
        except (UnicodeDecodeError, ValueError) as exc:
            raise RepositoryCloneError("Git returned an invalid tree entry") from exc
        try:
            path = raw_path.decode()
        except UnicodeDecodeError:
            skipped_files += 1
            continue
        if kind != "blob" or mode not in {"100644", "100755"} or not _is_repository_path(path):
            skipped_files += 1
            continue
        try:
            size = int(size_text)
        except ValueError as exc:
            raise RepositoryCloneError("Git returned an invalid blob size") from exc
        if size > MAX_WORKSPACE_FILE_BYTES:
            skipped_files += 1
            continue
        if object_id not in blob_cache:
            content_bytes = _git(repository_path, "cat-file", "blob", object_id)
            try:
                blob_cache[object_id] = content_bytes.decode()
            except UnicodeDecodeError:
                blob_cache[object_id] = None
        content = blob_cache[object_id]
        if content is None:
            skipped_files += 1
            continue
        if len(files) >= MAX_WORKSPACE_FILES:
            raise RepositorySnapshotTooLarge(f"Repository contains more than {MAX_WORKSPACE_FILES} text files")
        total_bytes += len(path.encode()) + size
        if total_bytes > MAX_WORKSPACE_TOTAL_BYTES:
            raise RepositorySnapshotTooLarge(f"Repository text files exceed {MAX_WORKSPACE_TOTAL_BYTES} bytes")
        files[path] = content
    return _TextSnapshot(files=files, skipped_files=skipped_files)


def _is_repository_path(value: str) -> bool:
    path = PurePosixPath(value)
    return is_workspace_path(value) and not any(part in _IGNORED_PARTS for part in path.parts)


def _bounded_metadata(value: str, fallback: str) -> str:
    value = value or fallback
    if len(value.encode()) <= MAX_COMMIT_METADATA_BYTES:
        return value
    return value.encode()[:MAX_COMMIT_METADATA_BYTES].decode(errors="ignore")


def _bounded_history(snapshots: list[_CommitSnapshot]) -> _BoundedHistory:
    for first_index in range(len(snapshots)):
        commits = _imported_commits(snapshots[first_index:])
        if imported_git_history_bytes(commits) <= MAX_IMPORTED_HISTORY_BYTES:
            return _BoundedHistory(commits=commits, truncated=first_index > 0)
    return _BoundedHistory(commits=(), truncated=True)


def _imported_commits(snapshots: list[_CommitSnapshot]) -> tuple[ImportedGitCommit, ...]:
    commits: list[ImportedGitCommit] = []
    previous_files: dict[str, str] = {}
    for snapshot in snapshots:
        changes: dict[str, str | None] = {
            path: content for path, content in snapshot.files.items() if previous_files.get(path) != content
        }
        changes.update({path: None for path in previous_files.keys() - snapshot.files.keys()})
        if changes:
            commits.append(
                ImportedGitCommit(
                    message=snapshot.message,
                    author_name=snapshot.author_name,
                    author_email=snapshot.author_email,
                    changes=changes,
                )
            )
        previous_files = snapshot.files
    return tuple(commits)


def _git(repository_path: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_path), *arguments],
            check=True,
            capture_output=True,
            timeout=REPOSITORY_CLONE_TIMEOUT,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        raise RepositoryCloneError(f"Could not inspect cloned Git repository with {' '.join(arguments[:2])}") from exc
    return result.stdout


async def repository_snapshot_response(request: Request) -> Response:
    """Clone a public GitHub repository snapshot for the dashboard."""
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Repository snapshot request must be an object")
        snapshot = await clone_repository_snapshot(payload.get("url"))
    except json.JSONDecodeError:
        return JSONResponse({"error": "Request body must be JSON"}, status_code=400)
    except RepositorySnapshotTooLarge as exc:
        return JSONResponse({"error": str(exc)}, status_code=413)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except RepositoryCloneError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    return JSONResponse(asdict(snapshot))
