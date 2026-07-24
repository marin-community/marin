#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cloud-sql-python-connector[pg8000]>=1.9",
#     "fastembed>=0.3",
# ]
# ///
"""Search the echo corpus: GitHub issues/PRs/comments and Discord messages.

The `chunks` table on echo (the `context` database on the shared `marin-metadata`
instance) mirrors the github+discord slice of the marinmirror corpus (re-synced every
10 minutes), with pgvector embeddings for semantic
search. Every hit carries a canonical, citable URL. See
`.agents/skills/context-search/SKILL.md` for when to use it, and
`infra/echo/README.md` for the database itself.

    scripts/context_search.py search "<natural language query>" [--source github|discord]
        [--kind issue|pr|comment|message] [--since YYYY-MM-DD] [--limit 10]
    scripts/context_search.py grep "<substring>" [--source ...] [--limit 20]
    scripts/context_search.py show <id>

`search` embeds the query with the corpus's own model (BAAI/bge-small-en-v1.5, downloaded
on first use) and ranks by cosine distance; `grep` is a plain ILIKE substring scan ordered
newest-first — use it for identifiers and exact strings.

Auth: Cloud SQL IAM — you connect as your own ADC identity (a member of the
`echo@openathena.ai` group, granted read), so you need roles/cloudsql.instanceUser and
roles/cloudsql.client, no password.
"""

import argparse
import json
import os
import textwrap
import urllib.request

import google.auth
from fastembed import TextEmbedding
from google.auth.transport.requests import Request
from google.cloud.sql.connector import Connector

PROJECT = "hai-gcp-models"
REGION = "us-central1"
INSTANCE = f"{PROJECT}:{REGION}:marin-metadata"
DATABASE = "context"
# Must match the corpus's embedding space (see infra/echo/schema.py).
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def db_user() -> str:
    """The Postgres username for the caller's ADC identity under Cloud SQL IAM auth.

    A service account's DB name is its email minus `.gserviceaccount.com`; a user's is
    their email. `MARIN_DB_USER` overrides for principals ADC cannot resolve locally.
    """
    if override := os.environ.get("MARIN_DB_USER"):
        return override
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    sa_email = getattr(credentials, "service_account_email", None)
    if sa_email and sa_email != "default":
        return sa_email.removesuffix(".gserviceaccount.com")
    credentials.refresh(Request())
    with urllib.request.urlopen(f"https://oauth2.googleapis.com/tokeninfo?access_token={credentials.token}") as resp:
        return json.load(resp)["email"]


def chunk_filters(args: argparse.Namespace) -> tuple[str, list[str]]:
    where: list[str] = ["true"]
    params: list[str] = []
    if args.source:
        where.append("source = %s")
        params.append(args.source)
    if getattr(args, "kind", None):
        where.append("kind = %s")
        params.append(args.kind)
    if getattr(args, "since", None):
        where.append("date >= %s")
        params.append(args.since)
    return " AND ".join(where), params


def print_hits(rows) -> None:
    for chunk_id, dist, source, kind, date, author, title, text, url in rows:
        snippet = " ".join((text or "").split())[:110]
        prefix = f"{dist:.3f} " if dist is not None else ""
        # For discord, title is just the channel name — the snippet carries the content.
        headline = f"{title}: {snippet}" if source == "discord" else (title or snippet)
        print(f"#{chunk_id} {prefix}[{source}/{kind}] {date} {author or '?'} — {headline}")
        print(f"    {url}")


def cmd_search(cur, args: argparse.Namespace) -> None:
    model = TextEmbedding(EMBED_MODEL)
    query_vector = "[" + ",".join(repr(float(v)) for v in next(iter(model.embed([args.query])))) + "]"
    # Without iterative scans, HNSW returns ef_search candidates before WHERE applies —
    # a selective filter (e.g. --source discord) can discard all of them and return nothing.
    cur.execute("SET hnsw.iterative_scan = relaxed_order")
    where, params = chunk_filters(args)
    cur.execute(
        f"SELECT id, embedding <=> CAST(%s AS vector) AS dist, source, kind, date::date, author, title, text, url "
        f"FROM chunks WHERE {where} ORDER BY dist LIMIT %s",
        [query_vector, *params, args.limit],
    )
    print_hits(cur.fetchall())


def cmd_grep(cur, args: argparse.Namespace) -> None:
    where, params = chunk_filters(args)
    # Escape ILIKE wildcards so the pattern is an exact substring: unescaped, the _ in
    # e.g. "ragged_all_to_all" matches any character and % matches everything.
    escaped = args.pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    cur.execute(
        f"SELECT id, NULL, source, kind, date::date, author, title, text, url "
        f"FROM chunks WHERE {where} AND text ILIKE %s ESCAPE '\\' ORDER BY chunks.date DESC LIMIT %s",
        [*params, f"%{escaped}%", args.limit],
    )
    print_hits(cur.fetchall())


def cmd_show(cur, args: argparse.Namespace) -> None:
    cur.execute("SELECT source, kind, date, author, title, url, text FROM chunks WHERE id = %s", [args.id])
    row = cur.fetchone()
    if row is None:
        raise SystemExit(f"no chunk #{args.id}")
    source, kind, date, author, title, url, text = row
    print(f"[{source}/{kind}] {date} {author or '?'} {title or ''}\n{url}\n")
    print(textwrap.fill(text or "", width=100, replace_whitespace=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the echo github+discord corpus.")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="semantic search, ranked by cosine distance")
    search.add_argument("query")
    search.add_argument("--source", choices=["github", "discord"])
    search.add_argument("--kind", choices=["issue", "pr", "comment", "message"])
    search.add_argument("--since", help="YYYY-MM-DD")
    search.add_argument("--limit", type=int, default=10)
    search.set_defaults(func=cmd_search)

    grep = sub.add_parser("grep", help="substring scan, newest first")
    grep.add_argument("pattern")
    grep.add_argument("--source", choices=["github", "discord"])
    grep.add_argument("--kind", choices=["issue", "pr", "comment", "message"])
    grep.add_argument("--since", help="YYYY-MM-DD")
    grep.add_argument("--limit", type=int, default=20)
    grep.set_defaults(func=cmd_grep)

    show = sub.add_parser("show", help="print one chunk in full")
    show.add_argument("id", type=int)
    show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    connector = Connector(quota_project=PROJECT)
    try:
        conn = connector.connect(INSTANCE, "pg8000", user=db_user(), enable_iam_auth=True, db=DATABASE)
        try:
            args.func(conn.cursor(), args)
        finally:
            conn.close()
    finally:
        connector.close()


if __name__ == "__main__":
    main()
