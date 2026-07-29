# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sync MarinMirror activity and GitHub repository files into Echo.

The MarinMirror phase refreshes GitHub and Discord activity. An independently
watermarked hourly phase indexes the configured GitHub branch head, fetching only
changed blobs when GitHub can compare it with the previously indexed commit.

This mirror duplicates what marinmirror itself could push; it is the interim answer
until marinmirror runs as a service in this project (see README.md).

Runs as a Cloud Run job: Postgres is reached over the Cloud SQL connector socket mounted
at /cloudsql, marinmirror over its bearer-token HTTP API. Configuration comes from env
vars (see ``infra/echo/__main__.py``).
"""

import base64
import hashlib
import json
import os
import sqlite3
import struct
import sys
import tarfile
import tempfile
import time
import urllib.request
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import repository_files
import schema
import search_config
import sqlalchemy
from fastembed import TextEmbedding
from google.cloud.sql.connector import Connector
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert as pg_insert

MARINMIRROR_URL = os.environ.get("MARINMIRROR_URL", "https://marinmirror.exe.xyz")
SOURCES = ("github", "discord")
BATCH = 400
REPOSITORY_BATCH = 100
REPOSITORY_CHECK_INTERVAL = timedelta(hours=1)
MAX_COMPARE_FILES = 300
# Session advisory lock so overlapping executions don't convoy on row locks: a full sync
# outlasts the 10-minute schedule, and Cloud Run jobs have no concurrency limit of their own.
SYNC_LOCK_KEY = 0x6563686F  # "echo"
REPOSITORY_SYNC_LOCK_KEY = 0x65636872  # "echr"

CHUNK_COLUMNS = [c.name for c in schema.chunks.columns]


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


def mirror_open(path: str, timeout: int = 600):
    req = urllib.request.Request(
        MARINMIRROR_URL + path,
        headers={"Authorization": f"Bearer {os.environ['MARINMIRROR_TOKEN']}", "User-Agent": "echo-sync"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


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


def download_corpus(dest: Path, expected_sha: str) -> None:
    digest = hashlib.sha256()
    with mirror_open("/corpus-index.db") as response, open(dest, "wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
            digest.update(block)
    if digest.hexdigest() != expected_sha:
        raise RuntimeError(f"corpus sha256 mismatch: got {digest.hexdigest()}, manifest says {expected_sha}")


def make_engine(connector: Connector) -> sqlalchemy.Engine:
    """Engine authenticating as the job's service account via Cloud SQL IAM auth.

    The connector mints a short-lived OAuth token from the job's ADC identity — no
    password. PGUSER is the SA's database username (its email minus the
    `.gserviceaccount.com` suffix).
    """
    return sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=lambda: connector.connect(
            os.environ["CLOUDSQL_CONNECTION"],
            "pg8000",
            user=os.environ["PGUSER"],
            db=os.environ["PGDATABASE"],
            enable_iam_auth=True,
        ),
    )


def decode_embedding(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    count = len(blob) // 4
    assert count == schema.EMBED_DIM, f"expected {schema.EMBED_DIM}-d embedding, got {count}"
    return list(struct.unpack(f"<{count}f", blob))


def chunk_record(row: tuple) -> dict:
    record = dict(zip(CHUNK_COLUMNS, row, strict=True))
    record["date"] = datetime.fromisoformat(record["date"]) if record["date"] else None
    record["embedding"] = decode_embedding(record["embedding"])
    return record


def upsert_chunks(conn: sqlalchemy.Connection, corpus: Path) -> tuple[int, int]:
    """Upsert changed github/discord chunks; delete rows gone upstream. Returns (upserted, deleted).

    Rows whose (id, hash) already match the database are skipped (the hash is the corpus's
    incremental-embed key; every re-upserted row pays an HNSW graph insertion). Rows stream
    one BATCH at a time — decoded embeddings are ~30x their sqlite blobs and the full corpus
    exceeds the job's memory — with each batch a single multi-row VALUES statement.
    """
    existing = dict(conn.execute(sqlalchemy.select(schema.chunks.c.id, schema.chunks.c.hash)).fetchall())
    placeholders = ",".join("?" * len(SOURCES))
    cursor = sqlite3.connect(corpus).execute(
        f"SELECT {', '.join(CHUNK_COLUMNS)} FROM chunks WHERE source IN ({placeholders}) ORDER BY id",
        SOURCES,
    )
    ids: list[int] = []
    upserted = 0
    started = time.time()
    while rows := cursor.fetchmany(BATCH):
        records = [chunk_record(row) for row in rows]
        ids.extend(record["id"] for record in records)
        changed = [r for r in records if existing.get(r["id"], object()) != r["hash"] or r["hash"] is None]
        if not changed:
            continue
        statement = pg_insert(schema.chunks).values(changed)
        statement = statement.on_conflict_do_update(
            index_elements=[schema.chunks.c.id],
            set_={name: statement.excluded[name] for name in CHUNK_COLUMNS if name != "id"},
        )
        conn.execute(statement)
        upserted += len(changed)
        if upserted % 8000 < len(changed):
            print(f"  upserted {upserted} rows ({upserted / (time.time() - started):.0f}/s)")

    # One array bind, not id.not_in(ids): expanding 73k+ ids into individual parameters
    # exceeds pg8000's 65535-parameter wire-protocol limit.
    id_array = sqlalchemy.cast(ids, postgresql.ARRAY(sqlalchemy.BigInteger))
    deleted = conn.execute(
        sqlalchemy.delete(schema.chunks)
        .where(schema.chunks.c.source.in_(SOURCES))
        .where(sqlalchemy.not_(schema.chunks.c.id == sqlalchemy.any_(id_array)))
    ).rowcount
    return upserted, deleted


def fetch_manifest() -> dict:
    with mirror_open("/manifest.json", timeout=30) as response:
        return json.load(response)


def corpus_build_epoch(path: Path) -> int:
    """The build epoch a corpus file claims for itself, after an integrity check."""
    db = sqlite3.connect(path)
    try:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("corpus failed sqlite integrity check")
        return int(db.execute("SELECT value FROM meta WHERE key = 'built_at_epoch'").fetchone()[0])
    finally:
        db.close()


def fetch_corpus(dest: Path, manifest: dict, attempts: int = 2) -> int:
    """Download the corpus and return the build epoch actually downloaded.

    The manifest sha is the fast path. marinmirror rebuilds every ~90 minutes, so a
    mismatch usually means the corpus was replaced mid-download — refetch and retry.
    The manifest and corpus can also skew for a whole build (observed: a manifest ahead
    of a regressed corpus file), so after the retries a mismatched file is still
    accepted if it passes sqlite integrity and self-reports its build epoch.
    """
    error: RuntimeError | None = None
    for _ in range(attempts):
        try:
            download_corpus(dest, manifest["corpus_index"]["sha256"])
            return manifest["built_at_epoch"]
        except RuntimeError as caught:
            error = caught
            print(f"{caught}; refetching manifest")
            manifest = fetch_manifest()
    built = corpus_build_epoch(dest)
    print(f"{error}; accepting intact corpus self-reporting build {built} (manifest/corpus skew)")
    return built


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
        "digest": chunk.digest,
        "title": chunk.title,
        "chunk_index": chunk.chunk_index,
        "start_line": chunk.start_line,
        "text": chunk.text,
        "embedding": list(embedding),
    }


def publish_repository_update(
    engine: sqlalchemy.Engine,
    target: RepositoryTarget,
    expected_sha: str | None,
    commit_sha: str,
    checked_at: datetime,
    changes: RepositoryChangeSet,
    chunks: list[repository_files.EmbeddedChunk],
    *,
    full_rebuild: bool,
) -> bool:
    """Atomically publish one prepared repository update."""
    with engine.begin() as conn:
        current = repository_state(conn, target)
        current_sha = current.commit_sha if current is not None else None
        if current_sha != expected_sha:
            print(f"repository index advanced from {expected_sha or 'empty'} to {current_sha}; discarding stale update")
            return False

        scope = (
            schema.repository_file_chunks.c.repository == target.repository,
            schema.repository_file_chunks.c.branch == target.branch,
        )
        if full_rebuild:
            conn.execute(sqlalchemy.delete(schema.repository_file_chunks).where(*scope))
        elif changes.replaced_paths:
            conn.execute(
                sqlalchemy.delete(schema.repository_file_chunks).where(
                    *scope,
                    schema.repository_file_chunks.c.path.in_(changes.replaced_paths),
                )
            )

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


@contextmanager
def repository_sync_lock(engine: sqlalchemy.Engine):
    """Hold the repository advisory lock for fetch, embedding, and publication."""
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
    published = publish_repository_update(
        engine,
        target,
        state.commit_sha if state is not None else None,
        head_sha,
        now,
        changes,
        chunks,
        full_rebuild=full_rebuild,
    )
    if published:
        mode = "full" if full_rebuild else "incremental"
        print(
            f"repository {mode} sync {head_sha[:12]}: "
            f"{len(changes.files)} files, {len(chunks)} chunks, {time.time() - started:.0f}s"
        )


def sync_corpus(engine: sqlalchemy.Engine) -> int:
    manifest = fetch_manifest()
    built = manifest["built_at_epoch"]

    with engine.connect() as conn:
        watermark = conn.execute(sqlalchemy.select(schema.sync_state.c.built_at_epoch)).scalar()
    if watermark is not None and watermark >= built:
        print(f"up to date: corpus build {built} already synced")
        return 0

    start = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus-index.db"
        built = fetch_corpus(corpus, manifest)
        if watermark is not None and watermark >= built:
            print(f"up to date: downloaded corpus is build {built}, already synced")
            return 0
        print(f"downloaded corpus build {built} ({corpus.stat().st_size >> 20} MB)")
        with engine.begin() as conn:
            locked = conn.execute(sqlalchemy.select(sqlalchemy.func.pg_try_advisory_xact_lock(SYNC_LOCK_KEY))).scalar()
            if not locked:
                print("another sync is already running; exiting")
                return 0
            # Re-read under the lock: this run may hold an older build than what the
            # previous lock holder committed, and must not roll the mirror backward.
            watermark = conn.execute(sqlalchemy.select(schema.sync_state.c.built_at_epoch)).scalar()
            if watermark is not None and watermark >= built:
                print(f"up to date: build {built} superseded while waiting to sync")
                return 0
            upserted, deleted = upsert_chunks(conn, corpus)
            watermark_insert = pg_insert(schema.sync_state).values(built_at_epoch=built)
            conn.execute(
                watermark_insert.on_conflict_do_update(
                    index_elements=[schema.sync_state.c.singleton],
                    set_={"built_at_epoch": built, "synced_at": sqlalchemy.func.now()},
                )
            )
    print(f"synced build {built}: {upserted} chunks upserted, {deleted} deleted, {time.time() - start:.0f}s")
    return 0


def run(engine: sqlalchemy.Engine) -> int:
    target = RepositoryTarget(
        repository=os.environ["GITHUB_REPOSITORY"],
        branch=os.environ["GITHUB_BRANCH"],
    )
    sync_repository(engine, target, os.environ["MARINMIRROR_TOKEN"], datetime.now(UTC))
    sync_corpus(engine)
    return 0


def main() -> int:
    with Connector() as connector:
        return run(make_engine(connector))


if __name__ == "__main__":
    sys.exit(main())
