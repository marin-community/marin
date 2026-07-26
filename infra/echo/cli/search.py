#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cloud-sql-python-connector[pg8000]>=1.9",
#     "google-auth>=2.30",
#     "fastembed>=0.3",
# ]
# ///
"""Search the echo corpus over Cloud SQL IAM authentication.

See ``infra/echo/README.md`` for access requirements and usage.
"""

import argparse
import datetime
import json
import os
import urllib.request

import google.auth
from google.auth.transport.requests import Request

INSTANCE = "hai-gcp-models:us-central1:marin-metadata"
DATABASE = "context"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # must match the corpus's embedding space
SOURCES = ["github", "discord"]
KINDS = ["issue", "pr", "comment", "message"]
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
OPENATHENA_USER_SUFFIX = "@openathena.ai"
CLOUD_SQL_SERVICE_ACCOUNT_SUFFIX = ".iam"


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


def validate_db_user(user: str) -> str:
    """Reject ADC identities that cannot inherit Echo's database grants."""
    if user.endswith((OPENATHENA_USER_SUFFIX, CLOUD_SQL_SERVICE_ACCOUNT_SUFFIX)):
        return user
    raise SystemExit(
        f"Echo resolved the database identity as {user!r}, which does not have access. "
        "Run `gcloud auth application-default login` and select your @openathena.ai account. "
        "Application Default Credentials are separate from the active `gcloud auth list` account."
    )


def escape_like(pattern: str) -> str:
    """Escape LIKE wildcards so `pattern` matches as an exact substring under ILIKE."""
    return pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def chunk_filters(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """The (predicate, param) pairs common to search and grep."""
    predicates: list[str] = []
    params: list[str] = []
    for column, value in (("source", args.source), ("kind", getattr(args, "kind", None))):
        if value:
            predicates.append(f"{column} = %s")
            params.append(value)
    if getattr(args, "since", None):
        predicates.append("date >= %s")
        params.append(args.since)
    return predicates, params


def where_clause(predicates: list[str]) -> str:
    return f"WHERE {' AND '.join(predicates)}" if predicates else ""


def print_hits(rows) -> None:
    for chunk_id, dist, source, kind, date, author, title, text, url in rows:
        snippet = " ".join((text or "").split())[:110]
        prefix = f"{dist:.3f} " if dist is not None else ""
        # For discord, title is just the channel name — the snippet carries the content.
        headline = f"{title}: {snippet}" if source == "discord" else (title or snippet)
        print(f"#{chunk_id} {prefix}[{source}/{kind}] {date} {author or '?'} — {headline}")
        print(f"    {url}")


def cmd_search(cur, args: argparse.Namespace) -> None:
    # Imported here, not at module top: fastembed is heavy (only `search` needs it) and
    # keeping it out of import lets the pure-logic unit tests run without it.
    from fastembed import TextEmbedding  # noqa: PLC0415

    model = TextEmbedding(EMBED_MODEL)
    query_vector = "[" + ",".join(repr(float(v)) for v in next(iter(model.embed([args.query])))) + "]"
    # Without iterative scans, HNSW returns ef_search candidates before WHERE applies —
    # a selective filter (e.g. --source discord) can discard all of them and return nothing.
    cur.execute("SET hnsw.iterative_scan = relaxed_order")
    predicates, params = chunk_filters(args)
    cur.execute(
        "SELECT id, embedding <=> CAST(%s AS vector) AS dist, source, kind, date::date, author, title, text, url "
        f"FROM chunks {where_clause(predicates)} ORDER BY dist LIMIT %s",
        [query_vector, *params, args.limit],
    )
    print_hits(cur.fetchall())


def cmd_grep(cur, args: argparse.Namespace) -> None:
    predicates, params = chunk_filters(args)
    predicates.append("text ILIKE %s ESCAPE '\\'")
    params.append(f"%{escape_like(args.pattern)}%")
    cur.execute(
        "SELECT id, NULL, source, kind, date::date, author, title, text, url "
        f"FROM chunks {where_clause(predicates)} ORDER BY chunks.date DESC LIMIT %s",
        [*params, args.limit],
    )
    print_hits(cur.fetchall())


def cmd_show(cur, args: argparse.Namespace) -> None:
    cur.execute("SELECT source, kind, date, author, title, url, text FROM chunks WHERE id = %s", [args.id])
    row = cur.fetchone()
    if row is None:
        raise SystemExit(f"no chunk #{args.id}")
    source, kind, date, author, title, url, text = row
    print(f"[{source}/{kind}] {date} {author or '?'} {title or ''}\n{url}\n")
    print(text or "")


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
    connector = Connector(quota_project=os.environ.get("GOOGLE_CLOUD_QUOTA_PROJECT"))
    try:
        conn = connector.connect(
            INSTANCE,
            "pg8000",
            user=validate_db_user(db_user()),
            enable_iam_auth=True,
            db=DATABASE,
        )
        try:
            args.func(conn.cursor(), args)
        finally:
            conn.close()
    finally:
        connector.close()


if __name__ == "__main__":
    main()
