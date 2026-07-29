# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build Echo's visible, resumable rolling GitHub repository index."""

import base64
import json
import tarfile
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import PurePosixPath
from urllib.parse import quote

import repository_files
import schema
import search_config
import sqlalchemy
from fastembed import TextEmbedding
from sqlalchemy.dialects.postgresql import insert as pg_insert

REPOSITORY_FILE_BATCH = 10
DATABASE_INSERT_BATCH = 100
REPOSITORY_CHECK_INTERVAL = timedelta(hours=1)
MAX_COMPARE_FILES = 300
REPOSITORY_SYNC_LOCK_KEY = 0x65636872  # "echr"
GITHUB_JSON_MEDIA_TYPE = "application/vnd.github+json"


@dataclass(frozen=True)
class RepositoryTarget:
    repository: str
    branch: str


@dataclass(frozen=True)
class RepositoryState:
    commit_sha: str
    checked_at: datetime


class RepositoryBuildMode(StrEnum):
    FULL = "full"
    INCREMENTAL = "incremental"


@dataclass(frozen=True)
class RepositoryBuild:
    commit_sha: str
    base_sha: str | None
    mode: RepositoryBuildMode
    total_files: int
    completed_files: int
    started_at: datetime


@dataclass(frozen=True)
class RepositoryChangeSet:
    replaced_paths: frozenset[str]
    files: tuple[repository_files.IndexedFile, ...]


def github_open(path: str, token: str, accept: str = GITHUB_JSON_MEDIA_TYPE):
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
    """Translate a GitHub comparison, or return None when a full rebuild is required."""
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
    """Return eligible repository files from a streamed GitHub archive."""
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


def repository_build(conn: sqlalchemy.Connection, target: RepositoryTarget) -> RepositoryBuild | None:
    row = conn.execute(
        sqlalchemy.select(
            schema.repository_index_builds.c.commit_sha,
            schema.repository_index_builds.c.base_sha,
            schema.repository_index_builds.c.mode,
            schema.repository_index_builds.c.total_files,
            schema.repository_index_builds.c.completed_files,
            schema.repository_index_builds.c.started_at,
        ).where(
            schema.repository_index_builds.c.repository == target.repository,
            schema.repository_index_builds.c.branch == target.branch,
        )
    ).first()
    if row is None:
        return None
    return RepositoryBuild(
        commit_sha=row.commit_sha,
        base_sha=row.base_sha,
        mode=RepositoryBuildMode(row.mode),
        total_files=row.total_files,
        completed_files=row.completed_files,
        started_at=row.started_at,
    )


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


def start_repository_build(
    engine: sqlalchemy.Engine,
    target: RepositoryTarget,
    build: RepositoryBuild,
    delete_statement: sqlalchemy.Delete | None,
) -> None:
    """Start a visible, resumable repository generation."""
    with engine.begin() as conn:
        if delete_statement is not None:
            conn.execute(delete_statement)
        conn.execute(
            pg_insert(schema.repository_index_builds).values(
                repository=target.repository,
                branch=target.branch,
                commit_sha=build.commit_sha,
                base_sha=build.base_sha,
                mode=build.mode.value,
                total_files=build.total_files,
                completed_files=0,
                started_at=build.started_at,
            )
        )


def completed_repository_paths(
    conn: sqlalchemy.Connection,
    target: RepositoryTarget,
    paths: tuple[str, ...],
) -> frozenset[str]:
    if not paths:
        return frozenset()
    rows = conn.execute(
        sqlalchemy.select(schema.repository_file_chunks.c.path)
        .where(*repository_scope(target), schema.repository_file_chunks.c.path.in_(paths))
        .distinct()
    )
    return frozenset(row.path for row in rows)


def remaining_repository_files(
    files: tuple[repository_files.IndexedFile, ...],
    completed_paths: frozenset[str],
) -> tuple[repository_files.IndexedFile, ...]:
    return tuple(file for file in files if file.path not in completed_paths)


def publish_repository_batch(
    engine: sqlalchemy.Engine,
    target: RepositoryTarget,
    commit_sha: str,
    files: tuple[repository_files.IndexedFile, ...],
    chunks: list[repository_files.EmbeddedChunk],
) -> int:
    """Publish one file batch and return the durable completed-file count."""
    records = [repository_chunk_record(target, chunk) for chunk in chunks]
    with engine.begin() as conn:
        for start in range(0, len(records), DATABASE_INSERT_BATCH):
            conn.execute(pg_insert(schema.repository_file_chunks).values(records[start : start + DATABASE_INSERT_BATCH]))
        row = conn.execute(
            sqlalchemy.update(schema.repository_index_builds)
            .where(
                schema.repository_index_builds.c.repository == target.repository,
                schema.repository_index_builds.c.branch == target.branch,
                schema.repository_index_builds.c.commit_sha == commit_sha,
            )
            .values(completed_files=schema.repository_index_builds.c.completed_files + len(files))
            .returning(schema.repository_index_builds.c.completed_files)
        ).first()
        if row is None:
            raise RuntimeError(f"repository build {commit_sha} disappeared while publishing")
    return row.completed_files


def finish_repository_build(
    engine: sqlalchemy.Engine,
    target: RepositoryTarget,
    build: RepositoryBuild,
    completed_at: datetime,
) -> bool:
    """Mark a complete generation ready, or return False if its build disappeared."""
    with engine.begin() as conn:
        current = repository_build(conn, target)
        if current is None or current.commit_sha != build.commit_sha:
            return False
        if current.completed_files != current.total_files:
            raise RuntimeError(
                f"repository build {build.commit_sha} has "
                f"{current.completed_files}/{current.total_files} completed files"
            )
        statement = pg_insert(schema.repository_index_state).values(
            repository=target.repository,
            branch=target.branch,
            commit_sha=build.commit_sha,
            checked_at=completed_at,
            indexed_at=completed_at,
        )
        conn.execute(
            statement.on_conflict_do_update(
                index_elements=[
                    schema.repository_index_state.c.repository,
                    schema.repository_index_state.c.branch,
                ],
                set_={
                    "commit_sha": build.commit_sha,
                    "checked_at": completed_at,
                    "indexed_at": completed_at,
                },
            )
        )
        conn.execute(
            sqlalchemy.delete(schema.repository_index_builds).where(
                schema.repository_index_builds.c.repository == target.repository,
                schema.repository_index_builds.c.branch == target.branch,
                schema.repository_index_builds.c.commit_sha == build.commit_sha,
            )
        )
    return True


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


def load_repository_changes(
    target: RepositoryTarget,
    token: str,
    commit_sha: str,
    base_sha: str | None,
    mode: RepositoryBuildMode,
) -> tuple[RepositoryBuildMode, str | None, RepositoryChangeSet]:
    if mode is RepositoryBuildMode.FULL:
        changes = RepositoryChangeSet(frozenset(), github_archive_files(target, commit_sha, token))
    else:
        assert base_sha is not None
        comparison = github_json(
            f"{target.repository}/compare/{quote(f'{base_sha}...{commit_sha}', safe='.')}",
            token,
        )
        incremental = incremental_repository_files(
            comparison,
            lambda sha: github_blob(target.repository, sha, token),
        )
        if incremental is None:
            mode = RepositoryBuildMode.FULL
            base_sha = None
            changes = RepositoryChangeSet(frozenset(), github_archive_files(target, commit_sha, token))
        else:
            changes = incremental
    if mode is RepositoryBuildMode.FULL and not changes.files:
        raise RuntimeError(f"GitHub archive for {target.repository}@{commit_sha} contained no eligible files")
    return mode, base_sha, changes


def initialize_repository_build(
    engine: sqlalchemy.Engine,
    target: RepositoryTarget,
    commit_sha: str,
    base_sha: str | None,
    mode: RepositoryBuildMode,
    changes: RepositoryChangeSet,
    now: datetime,
) -> RepositoryBuild:
    build = RepositoryBuild(
        commit_sha=commit_sha,
        base_sha=base_sha,
        mode=mode,
        total_files=len(changes.files),
        completed_files=0,
        started_at=now,
    )
    if mode is RepositoryBuildMode.FULL:
        deletion = sqlalchemy.delete(schema.repository_file_chunks).where(*repository_scope(target))
    elif changes.replaced_paths:
        deletion = sqlalchemy.delete(schema.repository_file_chunks).where(
            *repository_scope(target),
            schema.repository_file_chunks.c.path.in_(changes.replaced_paths),
        )
    else:
        deletion = None
    start_repository_build(engine, target, build, deletion)
    print(
        f"repository {build.mode} build {commit_sha[: search_config.DISPLAY_SHA_CHARACTERS]}: "
        f"0/{build.total_files} files (partial results are searchable)",
        flush=True,
    )
    return build


def validate_resumed_build(
    build: RepositoryBuild,
    mode: RepositoryBuildMode,
    changes: RepositoryChangeSet,
) -> None:
    if build.total_files != len(changes.files) or build.mode is not mode:
        raise RuntimeError(
            f"repository build {build.commit_sha} changed shape while resuming: "
            f"{build.mode} {build.total_files} files became {mode} {len(changes.files)} files"
        )


def run_repository_build(
    engine: sqlalchemy.Engine,
    target: RepositoryTarget,
    build: RepositoryBuild,
    changes: RepositoryChangeSet,
) -> None:
    paths = tuple(file.path for file in changes.files)
    with engine.connect() as conn:
        completed_paths = completed_repository_paths(conn, target, paths)
    remaining = remaining_repository_files(changes.files, completed_paths)
    display_sha = build.commit_sha[: search_config.DISPLAY_SHA_CHARACTERS]
    if build.completed_files:
        print(
            f"repository {build.mode} build {display_sha}: "
            f"resuming at {build.completed_files}/{build.total_files} files",
            flush=True,
        )

    run_started = time.time()
    total_chunks = 0
    if remaining:
        model = TextEmbedding(search_config.EMBED_MODEL)
        for start in range(0, len(remaining), REPOSITORY_FILE_BATCH):
            batch = remaining[start : start + REPOSITORY_FILE_BATCH]
            chunks = repository_files.embed_files(batch, model.passage_embed)
            total_chunks += len(chunks)
            completed = publish_repository_batch(engine, target, build.commit_sha, batch, chunks)
            print(
                f"repository {build.mode} build {display_sha}: "
                f"{completed}/{build.total_files} files, {total_chunks} chunks this run, "
                f"{time.time() - run_started:.0f}s",
                flush=True,
            )

    if finish_repository_build(engine, target, build, datetime.now(UTC)):
        print(
            f"repository {build.mode} sync {display_sha} complete: "
            f"{build.total_files} files, {time.time() - run_started:.0f}s this run",
            flush=True,
        )


def sync_repository_locked(
    engine: sqlalchemy.Engine,
    target: RepositoryTarget,
    token: str,
    now: datetime,
) -> None:
    with engine.connect() as conn:
        state = repository_state(conn, target)
        build = repository_build(conn, target)
    if build is None and not repository_check_due(state, now):
        return

    if build is None:
        head_sha = github_head(target, token)
        if state is not None and state.commit_sha == head_sha:
            record_unchanged_repository(engine, target, head_sha, now)
            print(
                f"repository up to date: {target.repository}@{target.branch} "
                f"{head_sha[: search_config.DISPLAY_SHA_CHARACTERS]}",
                flush=True,
            )
            return
        base_sha = state.commit_sha if state is not None else None
        build_mode = RepositoryBuildMode.FULL if state is None else RepositoryBuildMode.INCREMENTAL
    else:
        head_sha = build.commit_sha
        base_sha = build.base_sha
        build_mode = build.mode

    build_mode, base_sha, changes = load_repository_changes(
        target,
        token,
        head_sha,
        base_sha,
        build_mode,
    )
    if build is None:
        build = initialize_repository_build(engine, target, head_sha, base_sha, build_mode, changes, now)
    else:
        validate_resumed_build(build, build_mode, changes)
    run_repository_build(engine, target, build, changes)
