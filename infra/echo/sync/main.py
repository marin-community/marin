# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sync github+discord chunks from the marinmirror corpus into the context database.

One idempotent pass: fetch the marinmirror manifest and compare its ``built_at_epoch``
against the watermark in ``sync_state`` (exit early when unchanged), download and
sha-verify the corpus SQLite index, upsert every github/discord chunk on its id, delete
rows whose ids vanished upstream, and advance the watermark in the same transaction.

This mirror duplicates what marinmirror itself could push; it is the interim answer
until marinmirror runs as a service in this project (see README.md).

Runs as a Cloud Run job: Postgres is reached over the Cloud SQL connector socket mounted
at /cloudsql, marinmirror over its bearer-token HTTP API. Configuration comes from env
vars (see ``infra/echo/__main__.py``): CLOUDSQL_CONNECTION, PGDATABASE, PGUSER,
PGPASSWORD, MARINMIRROR_TOKEN, and optionally MARINMIRROR_URL.
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
from datetime import datetime
from pathlib import Path

import pg8000.dbapi
import schema
import sqlalchemy
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert as pg_insert

MARINMIRROR_URL = os.environ.get("MARINMIRROR_URL", "https://marinmirror.exe.xyz")
SOURCES = ("github", "discord")
BATCH = 400
# Session advisory lock so overlapping executions don't convoy on row locks: a full sync
# outlasts the 10-minute schedule, and Cloud Run jobs have no concurrency limit of their own.
SYNC_LOCK_KEY = 0x6563686F  # "echo"

CHUNK_COLUMNS = [c.name for c in schema.chunks.columns]


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


def make_engine() -> sqlalchemy.Engine:
    socket_dir = f"/cloudsql/{os.environ['CLOUDSQL_CONNECTION']}"
    return sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=lambda: pg8000.dbapi.connect(
            user=os.environ["PGUSER"],
            password=os.environ["PGPASSWORD"],
            database=os.environ["PGDATABASE"],
            unix_sock=f"{socket_dir}/.s.PGSQL.5432",
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

    Rows whose (id, hash) already match the database are skipped — the hash is the corpus's
    own incremental-embed key, and re-upserting an unchanged row still pays an HNSW graph
    insertion (~100 rows/s), so a typical sync touches hundreds of rows, not all 73k.

    Streams one BATCH of decoded rows at a time: the decoded embeddings are ~30x larger
    than their sqlite blobs (73k rows of boxed floats exceed the job's memory), so only
    the id list is held for the whole corpus. Each batch is one multi-row VALUES statement
    — executemany with ON CONFLICT falls back to a round-trip per row.
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


def main() -> int:
    manifest = fetch_manifest()
    built = manifest["built_at_epoch"]

    engine = make_engine()
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


if __name__ == "__main__":
    sys.exit(main())
