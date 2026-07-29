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

import hashlib
import json
import os
import sqlite3
import struct
import sys
import tempfile
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import github_repository
import schema
import sqlalchemy
from google.cloud.sql.connector import Connector
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert as pg_insert

MARINMIRROR_URL = os.environ.get("MARINMIRROR_URL", "https://marinmirror.exe.xyz")
SOURCES = ("github", "discord")
BATCH = 400
# Session advisory lock so overlapping executions don't convoy on row locks: a full sync
# outlasts the 10-minute schedule, and Cloud Run jobs have no concurrency limit of their own.
SYNC_LOCK_KEY = 0x6563686F  # "echo"

MIRRORED_CHUNK_COLUMNS = tuple(column.name for column in schema.chunks.columns if column.computed is None)


def mirror_open(path: str, timeout: int = 600):
    req = urllib.request.Request(
        MARINMIRROR_URL + path,
        headers={"Authorization": f"Bearer {os.environ['MARINMIRROR_TOKEN']}", "User-Agent": "echo-sync"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


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
    record = dict(zip(MIRRORED_CHUNK_COLUMNS, row, strict=True))
    record["date"] = datetime.fromisoformat(record["date"]) if record["date"] else None
    record["embedding"] = decode_embedding(record["embedding"])
    return record


def corpus_chunk_cursor(database: sqlite3.Connection) -> sqlite3.Cursor:
    placeholders = ",".join("?" * len(SOURCES))
    return database.execute(
        f"SELECT {', '.join(MIRRORED_CHUNK_COLUMNS)} FROM chunks WHERE source IN ({placeholders}) ORDER BY id",
        SOURCES,
    )


def upsert_chunks(conn: sqlalchemy.Connection, corpus: Path) -> tuple[int, int]:
    """Upsert changed github/discord chunks; delete rows gone upstream. Returns (upserted, deleted).

    Rows whose (id, hash) already match the database are skipped (the hash is the corpus's
    incremental-embed key; every re-upserted row pays an HNSW graph insertion). Rows stream
    one BATCH at a time — decoded embeddings are ~30x their sqlite blobs and the full corpus
    exceeds the job's memory — with each batch a single multi-row VALUES statement.
    """
    existing = dict(conn.execute(sqlalchemy.select(schema.chunks.c.id, schema.chunks.c.hash)).fetchall())
    ids: list[int] = []
    upserted = 0
    started = time.time()
    with sqlite3.connect(corpus) as database:
        cursor = corpus_chunk_cursor(database)
        while rows := cursor.fetchmany(BATCH):
            records = [chunk_record(row) for row in rows]
            ids.extend(record["id"] for record in records)
            changed = [r for r in records if existing.get(r["id"], object()) != r["hash"] or r["hash"] is None]
            if not changed:
                continue
            statement = pg_insert(schema.chunks).values(changed)
            statement = statement.on_conflict_do_update(
                index_elements=[schema.chunks.c.id],
                set_={name: statement.excluded[name] for name in MIRRORED_CHUNK_COLUMNS if name != "id"},
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
    target = github_repository.RepositoryTarget(
        repository=os.environ["GITHUB_REPOSITORY"],
        branch=os.environ["GITHUB_BRANCH"],
    )
    github_repository.sync_repository(engine, target, os.environ["MARINMIRROR_TOKEN"], datetime.now(UTC))
    sync_corpus(engine)
    return 0


def main() -> int:
    with Connector() as connector:
        return run(make_engine(connector))


if __name__ == "__main__":
    sys.exit(main())
