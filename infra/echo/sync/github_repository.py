# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build and atomically publish Echo's rolling GitHub repository index."""

import base64
import json
import tarfile
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import PurePosixPath
from urllib.parse import quote

import repository_files
import schema
import search_config
import sqlalchemy
from fastembed import TextEmbedding
from sqlalchemy.dialects.postgresql import insert as pg_insert

REPOSITORY_BATCH = 100
REPOSITORY_CHECK_INTERVAL = timedelta(hours=1)
MAX_COMPARE_FILES = 300
REPOSITORY_SYNC_LOCK_KEY = 0x65636872  # "echr"


@dataclass(frozen=True)
class RepositoryTarget:
    repository: str
    branch: str


@dataclass(frozen=True)
class RepositoryState:
    commit_sha: str
    checked_at: datetime


@dataclass(frozen=True)
class RepositoryChangeSet:
    replaced_paths: frozenset[str]
    files: tuple[repository_files.IndexedFile, ...]


def github_open(path: str, token: str, accept: str = "application/vnd.github+json"):
    request = urllib.request.Request(
        f"https://api.github.com/repos/{path}",
        headers={
            "Accept": accept,
            "Authorization": f"Bearer {token}",
            "User-Agent": "echo-sync",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    return urllib.request.urlopen(request, timeout=600)


def github_json(path: str, token: str) -> dict[str, object]:
    with github_open(path, token) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"GitHub returned a non-object for {path}")
    return {str(key): item for key, item in value.items()}


def github_head(target: RepositoryTarget, token: str) -> str:
    value = github_json(f"{target.repository}/commits/{quote(target.branch, safe='')}", token)
    sha = value.get("sha")
    if not isinstance(sha, str):
        raise ValueError("GitHub commit response has no SHA")
    return sha


def github_blob(repository: str, sha: str, token: str) -> bytes:
    value = github_json(f"{repository}/git/blobs/{quote(sha, safe='')}", token)
    content = value.get("content")
    if value.get("encoding") != "base64" or not isinstance(content, str):
        raise ValueError(f"GitHub blob {sha} is not base64 encoded")
    return base64.b64decode("".join(content.split()), validate=True)


def incremental_repository_files(
    comparison: dict[str, object],
    load_blob: Callable[[str], bytes],
) -> RepositoryChangeSet | None:
    """Translate a GitHub compare response into changed indexed files."""
    raw_files = comparison.get("files")
    if comparison.get("status") != "ahead" or not isinstance(raw_files, list) or len(raw_files) >= MAX_COMPARE_FILES:
        return None

    replaced_paths: set[str] = set()
    files: list[repository_files.IndexedFile] = []
    for value in raw_files:
        if not isinstance(value, dict):
            raise ValueError("GitHub compare file must be an object")
        filename = value.get("filename")
        if not isinstance(filename, str):
            raise ValueError("GitHub compare file has no filename")
        path = repository_files.repository_path(filename)
        if path is None:
            continue
        replaced_paths.add(str(path))

        previous_filename = value.get("previous_filename")
        if isinstance(previous_filename, str):
            previous_path = repository_files.repository_path(previous_filename)
            if previous_path is not None:
                replaced_paths.add(str(previous_path))

        if value.get("status") == "removed" or not repository_files.eligible_path(path):
            continue
        size = value.get("size")
        if isinstance(size, int) and size > repository_files.MAX_FILE_BYTES:
            continue
        sha = value.get("sha")
        if not isinstance(sha, str):
            raise ValueError(f"GitHub compare file {filename} has no blob SHA")
        indexed = repository_files.indexed_file(path, load_blob(sha))
        if indexed is not None:
            files.append(indexed)
    return RepositoryChangeSet(frozenset(replaced_paths), tuple(files))


def archive_repository_files(response) -> tuple[repository_files.IndexedFile, ...]:
    """Read eligible files from a GitHub tarball without extracting it."""
    files: list[repository_files.IndexedFile] = []
    with tarfile.open(fileobj=response, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or member.size > repository_files.MAX_FILE_BYTES:
                continue
            archive_path = PurePosixPath(member.name)
            if len(archive_path.parts) < 2:
                continue
            path = repository_files.repository_path(str(PurePosixPath(*archive_path.parts[1:])))
            if path is None or not repository_files.eligible_path(path):
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            data = source.read(repository_files.MAX_FILE_BYTES + 1)
            indexed = repository_files.indexed_file(path, data)
            if indexed is not None:
                files.append(indexed)
    return tuple(files)


def github_archive_files(
    target: RepositoryTarget,
    commit_sha: str,
    token: str,
) -> tuple[repository_files.IndexedFile, ...]:
    with github_open(
        f"{target.repository}/tarball/{quote(commit_sha, safe='')}",
        token,
        accept="application/vnd.github+json",
    ) as response:
        return archive_repository_files(response)


def repository_state(conn: sqlalchemy.Connection, target: RepositoryTarget) -> RepositoryState | None:
    row = conn.execute(
        sqlalchemy.select(
            schema.repository_index_state.c.commit_sha,
            schema.repository_index_state.c.checked_at,
        ).where(
            schema.repository_index_state.c.repository == target.repository,
            schema.repository_index_state.c.branch == target.branch,
        )
    ).first()
    if row is None:
        return None
    return RepositoryState(row.commit_sha, row.checked_at)


def repository_check_due(state: RepositoryState | None, now: datetime) -> bool:
    return state is None or now - state.checked_at >= REPOSITORY_CHECK_INTERVAL


def repository_chunk_record(
    target: RepositoryTarget,
    chunk: repository_files.EmbeddedChunk,
) -> dict[str, object]:
    embedding = repository_files.decode_embedding(chunk.embedding)
    if len(embedding) != schema.EMBED_DIM:
        raise ValueError(f"expected {schema.EMBED_DIM}-d repository embedding, got {len(embedding)}")
    return {
        "repository": target.repository,
        "branch": target.branch,
        "path": chunk.path,
        "title": chunk.title,
        "chunk_index": chunk.chunk_index,
        "start_line": chunk.start_line,
        "text": chunk.text,
        "embedding": list(embedding),
    }


def repository_scope(target: RepositoryTarget) -> tuple[sqlalchemy.ColumnElement[bool], ...]:
    return (
        schema.repository_file_chunks.c.repository == target.repository,
        schema.repository_file_chunks.c.branch == target.branch,
    )


def publish_repository_update(
    engine: sqlalchemy.Engine,
    target: RepositoryTarget,
    expected_sha: str | None,
    commit_sha: str,
    checked_at: datetime,
    chunks: list[repository_files.EmbeddedChunk],
    delete_statement: sqlalchemy.Delete | None,
) -> bool:
    """Publish prepared chunks, returning False when another run advanced the index."""
    with engine.begin() as conn:
        current = repository_state(conn, target)
        current_sha = current.commit_sha if current is not None else None
        if current_sha != expected_sha:
            print(f"repository index advanced from {expected_sha or 'empty'} to {current_sha}; discarding stale update")
            return False

        if delete_statement is not None:
            conn.execute(delete_statement)

        records = [repository_chunk_record(target, chunk) for chunk in chunks]
        for start in range(0, len(records), REPOSITORY_BATCH):
            conn.execute(pg_insert(schema.repository_file_chunks).values(records[start : start + REPOSITORY_BATCH]))

        statement = pg_insert(schema.repository_index_state).values(
            repository=target.repository,
            branch=target.branch,
            commit_sha=commit_sha,
            checked_at=checked_at,
            indexed_at=checked_at,
        )
        conn.execute(
            statement.on_conflict_do_update(
                index_elements=[
                    schema.repository_index_state.c.repository,
                    schema.repository_index_state.c.branch,
                ],
                set_={
                    "commit_sha": commit_sha,
                    "checked_at": checked_at,
                    "indexed_at": checked_at,
                },
            )
        )
    return True


def publish_full_repository(
    engine: sqlalchemy.Engine,
    target: RepositoryTarget,
    expected_sha: str | None,
    commit_sha: str,
    checked_at: datetime,
    chunks: list[repository_files.EmbeddedChunk],
) -> bool:
    deletion = sqlalchemy.delete(schema.repository_file_chunks).where(*repository_scope(target))
    return publish_repository_update(engine, target, expected_sha, commit_sha, checked_at, chunks, deletion)


def publish_changed_repository(
    engine: sqlalchemy.Engine,
    target: RepositoryTarget,
    expected_sha: str,
    commit_sha: str,
    checked_at: datetime,
    changes: RepositoryChangeSet,
    chunks: list[repository_files.EmbeddedChunk],
) -> bool:
    deletion = None
    if changes.replaced_paths:
        deletion = sqlalchemy.delete(schema.repository_file_chunks).where(
            *repository_scope(target),
            schema.repository_file_chunks.c.path.in_(changes.replaced_paths),
        )
    return publish_repository_update(engine, target, expected_sha, commit_sha, checked_at, chunks, deletion)


@contextmanager
def repository_sync_lock(engine: sqlalchemy.Engine) -> Iterator[bool]:
    """Yield whether this session acquired the repository sync lock."""
    with engine.connect() as conn:
        locked = bool(
            conn.execute(sqlalchemy.select(sqlalchemy.func.pg_try_advisory_lock(REPOSITORY_SYNC_LOCK_KEY))).scalar()
        )
        conn.commit()
        try:
            yield locked
        finally:
            if locked:
                conn.execute(sqlalchemy.select(sqlalchemy.func.pg_advisory_unlock(REPOSITORY_SYNC_LOCK_KEY)))
                conn.commit()


def record_unchanged_repository(
    engine: sqlalchemy.Engine,
    target: RepositoryTarget,
    commit_sha: str,
    checked_at: datetime,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            sqlalchemy.update(schema.repository_index_state)
            .where(
                schema.repository_index_state.c.repository == target.repository,
                schema.repository_index_state.c.branch == target.branch,
                schema.repository_index_state.c.commit_sha == commit_sha,
            )
            .values(checked_at=checked_at)
        )


def sync_repository(
    engine: sqlalchemy.Engine,
    target: RepositoryTarget,
    token: str,
    now: datetime,
) -> None:
    with repository_sync_lock(engine) as locked:
        if not locked:
            print("another repository sync is running; exiting")
            return
        sync_repository_locked(engine, target, token, now)


def sync_repository_locked(
    engine: sqlalchemy.Engine,
    target: RepositoryTarget,
    token: str,
    now: datetime,
) -> None:
    with engine.connect() as conn:
        state = repository_state(conn, target)
    if not repository_check_due(state, now):
        return

    head_sha = github_head(target, token)
    if state is not None and state.commit_sha == head_sha:
        record_unchanged_repository(engine, target, head_sha, now)
        print(f"repository up to date: {target.repository}@{target.branch} {head_sha[:12]}")
        return

    full_rebuild = state is None
    if state is None:
        files = github_archive_files(target, head_sha, token)
        changes = RepositoryChangeSet(frozenset(), files)
    else:
        comparison = github_json(
            f"{target.repository}/compare/{quote(f'{state.commit_sha}...{head_sha}', safe='.')}",
            token,
        )
        incremental = incremental_repository_files(
            comparison,
            lambda sha: github_blob(target.repository, sha, token),
        )
        if incremental is None:
            full_rebuild = True
            files = github_archive_files(target, head_sha, token)
            changes = RepositoryChangeSet(frozenset(), files)
        else:
            changes = incremental

    if full_rebuild and not changes.files:
        raise RuntimeError(f"GitHub archive for {target.repository}@{head_sha} contained no eligible files")

    started = time.time()
    chunks: list[repository_files.EmbeddedChunk] = []
    if changes.files:
        model = TextEmbedding(search_config.EMBED_MODEL)
        chunks = repository_files.embed_files(changes.files, model.passage_embed)
    expected_sha = state.commit_sha if state is not None else None
    if full_rebuild:
        published = publish_full_repository(engine, target, expected_sha, head_sha, now, chunks)
    else:
        assert expected_sha is not None
        published = publish_changed_repository(engine, target, expected_sha, head_sha, now, changes, chunks)
    if published:
        mode = "full" if full_rebuild else "incremental"
        print(
            f"repository {mode} sync {head_sha[:12]}: "
            f"{len(changes.files)} files, {len(chunks)} chunks, {time.time() - started:.0f}s"
        )
