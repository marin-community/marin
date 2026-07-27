#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cloud-sql-python-connector[pg8000]>=1.9",
#     "google-auth>=2.30",
#     "fastembed>=0.8",
#     "sqlalchemy>=2",
# ]
# ///
"""Search the echo corpus over Cloud SQL IAM authentication.

See ``infra/echo/README.md`` for access requirements and usage.
"""

import argparse
import datetime
import json
import os
import sys
import urllib.request
from pathlib import Path

import google.auth
import sqlalchemy
from google.auth.transport.requests import Request

sys.path.insert(0, str(Path(__file__).parents[1]))

import hybrid_search

INSTANCE = "hai-gcp-models:us-central1:marin-metadata"
DATABASE = "context"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # must match the corpus's embedding space
SOURCES = ["github", "discord"]
KINDS = ["issue", "pr", "comment", "message"]
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"


def db_user() -> str:
    """The Postgres username for the caller's ADC identity under Cloud SQL IAM auth.

    A service account's is its email minus `.gserviceaccount.com`; a user's is their email,
    read from the OpenID userinfo endpoint. `MARIN_DB_USER` overrides for principals that
    cannot resolve (impersonated, external-account, or workforce credentials).
    """
    if override := os.environ.get("MARIN_DB_USER"):
        return override
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    sa_email = getattr(credentials, "service_account_email", None)
    if sa_email and sa_email != "default":
        return sa_email.removesuffix(".gserviceaccount.com")
    credentials.refresh(Request())
    request = urllib.request.Request(USERINFO_URL, headers={"Authorization": f"Bearer {credentials.token}"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            email = json.load(response).get("email")
    except Exception as error:
        raise SystemExit(f"cannot resolve your database identity from ADC ({error}); set MARIN_DB_USER") from error
    if not email:
        raise SystemExit("your ADC identity exposes no email; set MARIN_DB_USER to your database username")
    return email


def escape_like(pattern: str) -> str:
    """Escape LIKE wildcards so `pattern` matches as an exact substring under ILIKE."""
    return pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def chunk_filters(args: argparse.Namespace) -> tuple[list[str], dict[str, object]]:
    """Return safe fixed predicates and their named values."""
    predicates: list[str] = []
    params: dict[str, object] = {}
    for column, value in (("source", args.source), ("kind", getattr(args, "kind", None))):
        if value:
            predicates.append(f"c.{column} = :{column}")
            params[column] = value
    if getattr(args, "since", None):
        predicates.append("c.date >= :since")
        params["since"] = args.since
    return predicates, params


def where_clause(predicates: list[str]) -> str:
    return f"WHERE {' AND '.join(predicates)}" if predicates else ""


def print_hits(rows) -> None:
    for row in rows:
        snippet = " ".join((row.text or "").split())[:110]
        prefix = f"{row.score:.4f} " if row.score is not None else ""
        # For discord, title is just the channel name — the snippet carries the content.
        headline = f"{row.title}: {snippet}" if row.source == "discord" else (row.title or snippet)
        print(f"#{row.id} {prefix}[{row.source}/{row.kind}] {row.date} {row.author or '?'} — {headline}")
        print(f"    {row.url}")


def cmd_search(conn: sqlalchemy.Connection, args: argparse.Namespace) -> None:
    # Imported here, not at module top: fastembed is heavy (only `search` needs it) and
    # keeping it out of import lets the pure-logic unit tests run without it.
    from fastembed import TextEmbedding  # noqa: PLC0415

    model = TextEmbedding(EMBED_MODEL)
    query_vector = "[" + ",".join(repr(float(v)) for v in next(iter(model.query_embed([args.query])))) + "]"
    # Without iterative scans, HNSW returns ef_search candidates before WHERE applies —
    # a selective filter (e.g. --source discord) can discard all of them and return nothing.
    conn.execute(hybrid_search.HNSW_ITERATIVE_SCAN)
    predicates, params = chunk_filters(args)
    params.update(
        q=args.query,
        embedding=query_vector,
        candidate_limit=hybrid_search.candidate_limit(args.limit),
        limit=args.limit,
    )
    print_hits(conn.execute(hybrid_search.chunk_search_statement(predicates), params))


def cmd_grep(conn: sqlalchemy.Connection, args: argparse.Namespace) -> None:
    predicates, params = chunk_filters(args)
    predicates.append("c.text ILIKE :pattern ESCAPE '\\'")
    params.update(pattern=f"%{escape_like(args.pattern)}%", limit=args.limit)
    statement = sqlalchemy.text(
        "SELECT c.*, 0.0 AS score, NULL AS distance, NULL AS lexical_score "
        f"FROM chunks AS c {where_clause(predicates)} ORDER BY c.date DESC LIMIT :limit"
    )
    print_hits(conn.execute(statement, params))


def cmd_show(conn: sqlalchemy.Connection, args: argparse.Namespace) -> None:
    row = conn.execute(
        sqlalchemy.text("SELECT source, kind, date, author, title, url, text FROM chunks WHERE id = :id"),
        {"id": args.id},
    ).first()
    if row is None:
        raise SystemExit(f"no chunk #{args.id}")
    print(f"[{row.source}/{row.kind}] {row.date} {row.author or '?'} {row.title or ''}\n{row.url}\n")
    print(row.text or "")


def bounded_limit(value: str) -> int:
    limit = int(value)
    if not 1 <= limit <= 100:
        raise argparse.ArgumentTypeError("must be between 1 and 100")
    return limit


def iso_date(value: str) -> str:
    datetime.date.fromisoformat(value)  # raises ValueError -> argparse rejects
    return value


def nonblank(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be blank")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search the echo github+discord corpus.")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text, default_limit, func in (
        ("search", "semantic search, ranked by cosine distance", 10, cmd_search),
        ("grep", "substring scan, newest first", 20, cmd_grep),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("query" if name == "search" else "pattern", type=nonblank)
        p.add_argument("--source", choices=SOURCES)
        p.add_argument("--kind", choices=KINDS)
        p.add_argument("--since", type=iso_date, help="YYYY-MM-DD")
        p.add_argument("--limit", type=bounded_limit, default=default_limit)
        p.set_defaults(func=func)

    show = sub.add_parser("show", help="print one chunk verbatim")
    show.add_argument("id", type=int)
    show.set_defaults(func=cmd_show)
    return parser


def main() -> None:
    from google.cloud.sql.connector import Connector  # noqa: PLC0415  (heavy; kept off import for tests)

    args = build_parser().parse_args()
    with Connector(quota_project=os.environ.get("GOOGLE_CLOUD_QUOTA_PROJECT")) as connector:
        engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=lambda: connector.connect(INSTANCE, "pg8000", user=db_user(), enable_iam_auth=True, db=DATABASE),
            pool_pre_ping=True,
        )
        try:
            with engine.connect() as conn:
                args.func(conn, args)
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()
