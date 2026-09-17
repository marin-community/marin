# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch bounded public GitHub snapshots for ShellSim workspaces."""

import io
import json
import stat
import zipfile
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from marin.inference.shell_workspace import (
    MAX_WORKSPACE_FILE_BYTES,
    MAX_WORKSPACE_FILES,
    MAX_WORKSPACE_TOTAL_BYTES,
)

MAX_REPOSITORY_ARCHIVE_BYTES = 4 * 1024 * 1024
MAX_REPOSITORY_ARCHIVE_ENTRIES = 10_000
_GITHUB_API_HOST = "api.github.com"
_GITHUB_ARCHIVE_HOSTS = frozenset({_GITHUB_API_HOST, "codeload.github.com"})
_IGNORED_PARTS = frozenset({".git", ".venv", "venv", "node_modules", "target", "__pycache__"})


class RepositorySnapshotTooLarge(ValueError):
    """Raised when a repository snapshot exceeds a workspace bound."""


@dataclass(frozen=True)
class RepositorySnapshot:
    """Text files extracted from a repository's default branch."""

    files: dict[str, str]
    skipped_files: int


def github_repository_slug(value: object) -> tuple[str, str]:
    """Return the owner and repository from a canonical public GitHub URL."""
    if not isinstance(value, str):
        raise ValueError("Repository URL must be a string")
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
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


async def fetch_repository_snapshot(
    url: object, *, transport: httpx.AsyncBaseTransport | None = None
) -> RepositorySnapshot:
    """Download and extract the default branch of one public GitHub repository."""
    owner, repository = github_repository_slug(url)
    api_url = f"https://{_GITHUB_API_HOST}/repos/{owner}/{repository}/zipball"
    async with httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(30, connect=10),
        headers={"Accept": "application/vnd.github+json", "User-Agent": "marin-serve-dashboard"},
    ) as client:
        async with client.stream("GET", api_url) as response:
            if not response.is_redirect:
                return repository_snapshot_from_zip(await _bounded_archive(response))
            location = response.headers.get("location")
            if location is None:
                raise ValueError("GitHub repository archive redirect omitted its location")
            archive_url = response.url.join(location)
            if archive_url.scheme != "https" or archive_url.host not in _GITHUB_ARCHIVE_HOSTS:
                raise ValueError(f"GitHub archive redirected to an unexpected host: {archive_url.host}")
        async with client.stream("GET", archive_url) as response:
            return repository_snapshot_from_zip(await _bounded_archive(response))


async def _bounded_archive(response: httpx.Response) -> bytes:
    response.raise_for_status()
    if response.url.scheme != "https" or response.url.host not in _GITHUB_ARCHIVE_HOSTS:
        raise ValueError(f"Unexpected GitHub archive host: {response.url.host}")
    archive = bytearray()
    async for chunk in response.aiter_bytes():
        archive.extend(chunk)
        if len(archive) > MAX_REPOSITORY_ARCHIVE_BYTES:
            raise RepositorySnapshotTooLarge(
                f"Repository archive exceeds {MAX_REPOSITORY_ARCHIVE_BYTES} compressed bytes"
            )
    return bytes(archive)


def repository_snapshot_from_zip(archive: bytes) -> RepositorySnapshot:
    """Extract UTF-8 regular files from a GitHub repository archive."""
    try:
        repository_zip = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as exc:
        raise ValueError("GitHub returned an invalid repository archive") from exc

    with repository_zip:
        entries = repository_zip.infolist()
        if len(entries) > MAX_REPOSITORY_ARCHIVE_ENTRIES:
            raise RepositorySnapshotTooLarge(
                f"Repository archive contains more than {MAX_REPOSITORY_ARCHIVE_ENTRIES} entries"
            )
        files: dict[str, str] = {}
        skipped_files = 0
        total_bytes = 0
        for entry in entries:
            path = _archive_path(entry)
            if path is None:
                if not entry.is_dir():
                    skipped_files += 1
                continue
            if entry.file_size > MAX_WORKSPACE_FILE_BYTES:
                skipped_files += 1
                continue
            content_bytes = repository_zip.read(entry)
            try:
                content = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                skipped_files += 1
                continue
            if len(files) >= MAX_WORKSPACE_FILES:
                raise RepositorySnapshotTooLarge(f"Repository contains more than {MAX_WORKSPACE_FILES} text files")
            total_bytes += len(path.encode()) + len(content_bytes)
            if total_bytes > MAX_WORKSPACE_TOTAL_BYTES:
                raise RepositorySnapshotTooLarge(
                    f"Repository text files exceed {MAX_WORKSPACE_TOTAL_BYTES} uncompressed bytes"
                )
            files[path] = content
    return RepositorySnapshot(files=files, skipped_files=skipped_files)


def _archive_path(entry: zipfile.ZipInfo) -> str | None:
    """Return a safe workspace path, or ``None`` for entries that must be skipped."""
    mode = entry.external_attr >> 16
    if entry.is_dir() or stat.S_ISLNK(mode):
        return None
    raw_path = PurePosixPath(entry.filename)
    if raw_path.is_absolute() or ".." in raw_path.parts or len(raw_path.parts) < 2:
        return None
    path = PurePosixPath(*raw_path.parts[1:])
    if path == PurePosixPath(".") or any(part in _IGNORED_PARTS for part in path.parts):
        return None
    return path.as_posix()


async def repository_snapshot_response(request: Request) -> Response:
    """Load a public GitHub repository snapshot for the dashboard."""
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Repository snapshot request must be an object")
        snapshot = await fetch_repository_snapshot(payload.get("url"))
    except json.JSONDecodeError:
        return JSONResponse({"error": "Request body must be JSON"}, status_code=400)
    except RepositorySnapshotTooLarge as exc:
        return JSONResponse({"error": str(exc)}, status_code=413)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except httpx.HTTPStatusError as exc:
        return JSONResponse(
            {"error": f"GitHub repository snapshot returned {exc.response.status_code}"},
            status_code=502,
        )
    except httpx.HTTPError as exc:
        return JSONResponse({"error": f"Could not fetch GitHub repository snapshot: {exc}"}, status_code=502)
    return JSONResponse(asdict(snapshot))
