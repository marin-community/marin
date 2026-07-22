# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sync github+discord chunks from the marinmirror corpus into the marin-context database.

One idempotent pass: fetch the marinmirror manifest and compare its ``built_at_epoch``
against the watermark stored in ``sync_state`` (exit early when unchanged), download and
sha-verify the corpus SQLite index, upsert every github/discord chunk on its id, delete
rows whose ids vanished upstream, and advance the watermark in the same transaction.

Runs as a Cloud Run job: Postgres is reached over the Cloud SQL connector socket mounted
at /cloudsql, marinmirror over its bearer-token HTTP API. Configuration comes from env
vars (see ``infra/context/__main__.py``): CLOUDSQL_CONNECTION, PGDATABASE, PGUSER,
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

MARINMIRROR_URL = os.environ.get("MARINMIRROR_URL", "https://marinmirror.exe.xyz")
SOURCES = ("github", "discord")
EMBED_DIM = 384
BATCH = 400

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
  id bigint PRIMARY KEY,
  source text NOT NULL,
  kind text NOT NULL,
  ref text,
  parent text,
  title text,
  author text,
  date timestamptz,
  url text NOT NULL,
  text text,
  hash text,
  embedding vector(384),
  part int NOT NULL DEFAULT 0,
  n_parts int NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_chunks_source_kind ON chunks(source, kind);
CREATE INDEX IF NOT EXISTS idx_chunks_date ON chunks(date);
CREATE INDEX IF NOT EXISTS idx_chunks_url ON chunks(url);
CREATE INDEX IF NOT EXISTS idx_chunks_key ON chunks(source, kind, ref);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS sync_state (
  singleton bool PRIMARY KEY DEFAULT true CHECK (singleton),
  built_at_epoch bigint NOT NULL,
  synced_at timestamptz NOT NULL DEFAULT now()
);
"""

COLS = "id, source, kind, ref, parent, title, author, date, url, text, hash, embedding, part, n_parts"
ROW_SQL = "(" + ", ".join(["%s"] * 11) + ", CAST(%s AS vector), %s, %s)"
UPSERT_TAIL = """
ON CONFLICT (id) DO UPDATE SET
  source=EXCLUDED.source, kind=EXCLUDED.kind, ref=EXCLUDED.ref, parent=EXCLUDED.parent,
  title=EXCLUDED.title, author=EXCLUDED.author, date=EXCLUDED.date, url=EXCLUDED.url,
  text=EXCLUDED.text, hash=EXCLUDED.hash, embedding=EXCLUDED.embedding,
  part=EXCLUDED.part, n_parts=EXCLUDED.n_parts
"""


def mirror_open(path: str, timeout: int = 600):
    req = urllib.request.Request(
        MARINMIRROR_URL + path,
        headers={"Authorization": f"Bearer {os.environ['MARINMIRROR_TOKEN']}", "User-Agent": "marin-context-sync"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def download_corpus(dest: Path, expected_sha: str) -> None:
    h = hashlib.sha256()
    with mirror_open("/corpus-index.db") as r, open(dest, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)
            h.update(block)
    if h.hexdigest() != expected_sha:
        raise RuntimeError(f"corpus sha256 mismatch: got {h.hexdigest()}, manifest says {expected_sha}")


def connect() -> pg8000.dbapi.Connection:
    socket_dir = f"/cloudsql/{os.environ['CLOUDSQL_CONNECTION']}"
    return pg8000.dbapi.connect(
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        database=os.environ["PGDATABASE"],
        unix_sock=f"{socket_dir}/.s.PGSQL.5432",
    )


def embedding_literal(blob: bytes | None) -> str | None:
    """float32 LE blob -> pgvector text literal '[f1,f2,...]'."""
    if blob is None:
        return None
    n = len(blob) // 4
    assert n == EMBED_DIM, f"expected {EMBED_DIM}-d embedding, got {n}"
    return "[" + ",".join(repr(v) for v in struct.unpack(f"<{n}f", blob)) + "]"


def parse_date(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def upsert_chunks(cur, corpus: Path) -> tuple[int, int]:
    """Upsert every github/discord chunk; delete rows gone upstream. Returns (upserted, deleted)."""
    src = sqlite3.connect(corpus)
    placeholders = ",".join("?" * len(SOURCES))
    rows = src.execute(f"SELECT {COLS} FROM chunks WHERE source IN ({placeholders}) ORDER BY id", SOURCES).fetchall()
    for i in range(0, len(rows), BATCH):
        batch = rows[i : i + BATCH]
        params: list[object] = []
        for r in batch:
            params.extend(
                [
                    r[0],
                    r[1],
                    r[2],
                    r[3],
                    r[4],
                    r[5],
                    r[6],
                    parse_date(r[7]),
                    r[8],
                    r[9],
                    r[10],
                    embedding_literal(r[11]),
                    r[12],
                    r[13],
                ]
            )
        cur.execute(f"INSERT INTO chunks ({COLS}) VALUES " + ", ".join([ROW_SQL] * len(batch)) + UPSERT_TAIL, params)

    cur.execute(
        "DELETE FROM chunks WHERE source = ANY(CAST(%s AS text[])) AND NOT (id = ANY(CAST(%s AS bigint[])))",
        [list(SOURCES), [r[0] for r in rows]],
    )
    return len(rows), cur.rowcount


def main() -> int:
    with mirror_open("/manifest.json", timeout=30) as r:
        manifest = json.load(r)
    built = manifest["built_at_epoch"]

    conn = connect()
    cur = conn.cursor()
    for stmt in DDL.strip().split(";"):
        if stmt.strip():
            cur.execute(stmt)
    conn.commit()

    cur.execute("SELECT built_at_epoch FROM sync_state")
    row = cur.fetchone()
    if row and row[0] >= built:
        print(f"up to date: corpus build {built} already synced")
        return 0

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus-index.db"
        download_corpus(corpus, manifest["corpus_index"]["sha256"])
        print(f"downloaded corpus build {built} ({corpus.stat().st_size >> 20} MB)")
        upserted, deleted = upsert_chunks(cur, corpus)

    cur.execute(
        "INSERT INTO sync_state (built_at_epoch) VALUES (%s) "
        "ON CONFLICT (singleton) DO UPDATE SET built_at_epoch=EXCLUDED.built_at_epoch, synced_at=now()",
        [built],
    )
    conn.commit()
    print(f"synced build {built}: {upserted} chunks upserted, {deleted} deleted, {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
