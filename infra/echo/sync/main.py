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
from sqlalchemy.dialects.postgresql import insert as pg_insert

MARINMIRROR_URL = os.environ.get("MARINMIRROR_URL", "https://marinmirror.exe.xyz")
SOURCES = ("github", "discord")
BATCH = 400

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


def chunk_rows(corpus: Path):
    src = sqlite3.connect(corpus)
    placeholders = ",".join("?" * len(SOURCES))
    rows = src.execute(
        f"SELECT {', '.join(CHUNK_COLUMNS)} FROM chunks WHERE source IN ({placeholders}) ORDER BY id",
        SOURCES,
    ).fetchall()
    for row in rows:
        record = dict(zip(CHUNK_COLUMNS, row, strict=True))
        record["date"] = datetime.fromisoformat(record["date"]) if record["date"] else None
        record["embedding"] = decode_embedding(record["embedding"])
        yield record


def upsert_chunks(conn: sqlalchemy.Connection, corpus: Path) -> tuple[int, int]:
    """Upsert every github/discord chunk; delete rows gone upstream. Returns (upserted, deleted)."""
    records = list(chunk_rows(corpus))
    statement = pg_insert(schema.chunks)
    statement = statement.on_conflict_do_update(
        index_elements=[schema.chunks.c.id],
        set_={name: statement.excluded[name] for name in CHUNK_COLUMNS if name != "id"},
    )
    for start in range(0, len(records), BATCH):
        conn.execute(statement, records[start : start + BATCH])

    deleted = conn.execute(
        sqlalchemy.delete(schema.chunks)
        .where(schema.chunks.c.source.in_(SOURCES))
        .where(schema.chunks.c.id.not_in([r["id"] for r in records]))
    ).rowcount
    return len(records), deleted


def main() -> int:
    with mirror_open("/manifest.json", timeout=30) as response:
        manifest = json.load(response)
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
        download_corpus(corpus, manifest["corpus_index"]["sha256"])
        print(f"downloaded corpus build {built} ({corpus.stat().st_size >> 20} MB)")
        with engine.begin() as conn:
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
