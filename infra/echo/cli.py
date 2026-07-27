#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Search Echo activity, read chunks, and manage wiki notes through the echo-api service.

Run inside the repo environment, e.g. ``uv run infra/echo/cli.py search "..."``. This shares
Marin's IAP login helpers from ``marin-rigging`` (not on PyPI), so it is not a standalone
``uv --script``.

Every command calls the IAP-gated ``echo-api`` Cloud Run service with a Google-signed ID
token whose audience is the shared Marin desktop OAuth client — the same identity that
reaches iris. There is no separate echo login: humans authenticate once with ``iris login``
(cached under ``~/.config/marin/credentials``) and this reuses that token; agents and CI use
ambient service-account credentials with no login. See ``infra/echo/README.md``.

    uv run infra/echo/cli.py search "expert parallel MoE MFU on B200" --limit 10
    uv run infra/echo/cli.py grep ragged_all_to_all --source discord
    uv run infra/echo/cli.py show 12345
    uv run infra/echo/cli.py wiki search "grafana access"
    uv run infra/echo/cli.py wiki add --title "Grafana access" --use-when "inspecting dashboards" --body note.md
    uv run infra/echo/cli.py wiki edit 12 --title "Grafana access" --use-when "inspecting dashboards" --body -
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import requests
from rigging.auth import (
    MARIN_DESKTOP_OAUTH_CLIENT,
    IapCredentialsUnavailable,
    IapLoginRequired,
    IapServiceAccountTokenProvider,
    TokenProvider,
)
from rigging.credential_store import credentials_dir
from rigging.credentials import iap_edge_provider

# echo.oa.dev maps to the echo-api Cloud Run service; the IAP token audience is the shared
# Marin desktop OAuth client, which echo-api's IAP settings admit as a programmatic client.
API_URL = os.environ.get("ECHO_API_URL", "https://echo.oa.dev").rstrip("/")
AUDIENCE = MARIN_DESKTOP_OAUTH_CLIENT.client_id
# The rigging cluster whose cached login to prefer; every Marin login shares the desktop
# client, so any cached record works and `iris login` alone is enough.
LOGIN_CLUSTER = os.environ.get("ECHO_LOGIN_CLUSTER", "marin")
LOGIN_HINT = "run `iris login`"
SOURCES = ["github", "discord"]
KINDS = ["issue", "pr", "comment", "message"]


def cached_login_provider() -> TokenProvider | None:
    """An IAP provider from a cached Marin login: the preferred cluster, else any cached one."""
    provider = iap_edge_provider(LOGIN_CLUSTER)
    if provider is not None:
        return provider
    for path in sorted(credentials_dir().glob("*.json")):
        candidate = iap_edge_provider(path.stem)
        if candidate is not None:
            return candidate
    return None


def bearer_token() -> str:
    """A Google ID token for echo-api's IAP: cached human login first, else ambient SA creds."""
    provider = cached_login_provider() or IapServiceAccountTokenProvider(AUDIENCE)
    try:
        token = provider.get_token()
    except (IapCredentialsUnavailable, IapLoginRequired) as error:
        raise SystemExit(f"{error}\nHuman callers: {LOGIN_HINT}.") from error
    if not token:
        raise SystemExit("could not obtain an IAP token for echo-api")
    return token


def request(method: str, path: str, *, params: dict | None = None, body: dict | None = None) -> object:
    """Call echo-api and return the decoded JSON, or exit with a message on any HTTP error."""
    response = requests.request(
        method,
        f"{API_URL}{path}",
        params={k: v for k, v in (params or {}).items() if v is not None},
        json=body,
        headers={"Authorization": f"Bearer {bearer_token()}"},
        timeout=30,
        allow_redirects=False,
    )
    if response.status_code == 401:
        raise SystemExit(f"echo-api rejected the token (401). Confirm IAP access and that {LOGIN_HINT} succeeded.")
    if response.status_code >= 400:
        detail = (
            response.json().get("detail", response.text)
            if "json" in response.headers.get("content-type", "")
            else response.text
        )
        raise SystemExit(f"{method} {path} -> {response.status_code}: {detail}")
    return response.json()


def print_hits(hits: list[dict]) -> None:
    for hit in hits:
        score = f"{hit['score']:.4f} " if hit.get("score") else ""
        # For discord, title is only the channel; the snippet carries the content.
        headline = (
            f"{hit['title']}: {hit['snippet']}" if hit["source"] == "discord" else (hit["title"] or hit["snippet"])
        )
        print(f"#{hit['id']} {score}[{hit['source']}/{hit['kind']}] {hit['date']} {hit['author'] or '?'} — {headline}")
        print(f"    {hit['url']}")


def print_wiki(entries: list[dict]) -> None:
    for entry in entries:
        print(f"#{entry['id']} {entry['title']} (refs={entry.get('reference_count', 0)}, by {entry['author']})")
        print(f"    use when: {entry['use_when']}")
        if entry.get("snippet"):
            print(f"    {entry['snippet']}")


def read_body(value: str) -> str:
    """Body text inline, from stdin (``-``), or from a file path."""
    if value == "-":
        return sys.stdin.read()
    if os.path.exists(value):
        return Path(value).read_text()
    return value


def cmd_search(args: argparse.Namespace) -> None:
    print_hits(
        request(
            "GET",
            "/search",
            params={"q": args.query, "source": args.source, "kind": args.kind, "since": args.since, "limit": args.limit},
        )
    )


def cmd_grep(args: argparse.Namespace) -> None:
    print_hits(
        request(
            "GET",
            "/grep",
            params={"pattern": args.pattern, "source": args.source, "kind": args.kind, "limit": args.limit},
        )
    )


def cmd_show(args: argparse.Namespace) -> None:
    chunk = request("GET", f"/chunks/{args.id}")
    print(f"[{chunk['source']}/{chunk['kind']}] {chunk['date']} {chunk['author'] or '?'} {chunk['title'] or ''}")
    print(f"{chunk['url']}\n")
    print(chunk.get("text") or "")


def cmd_wiki_search(args: argparse.Namespace) -> None:
    print_wiki(request("GET", "/wiki/search", params={"q": args.query, "limit": args.limit}))


def cmd_wiki_show(args: argparse.Namespace) -> None:
    entry = request("GET", f"/wiki/{args.id}")
    print(f"#{entry['id']} {entry['title']} (refs={entry['reference_count']}, by {entry['author']})")
    print(f"use when: {entry['use_when']}\n")
    print(entry["body"])


def wiki_link(entry_id: int) -> str:
    """Browseable URL for a wiki entry — the dashboard deep-links off ?wiki=<id>."""
    return f"{API_URL}/?wiki={entry_id}"


def cmd_wiki_add(args: argparse.Namespace) -> None:
    entry = request("POST", "/wiki", body={"title": args.title, "use_when": args.use_when, "body": read_body(args.body)})
    print(f"created wiki #{entry['id']}: {entry['title']}")
    print(wiki_link(entry["id"]))


def cmd_wiki_edit(args: argparse.Namespace) -> None:
    entry = request(
        "PUT", f"/wiki/{args.id}", body={"title": args.title, "use_when": args.use_when, "body": read_body(args.body)}
    )
    print(f"updated wiki #{entry['id']}: {entry['title']}")
    print(wiki_link(entry["id"]))


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


def add_wiki_write_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--title", type=nonblank, required=True)
    parser.add_argument(
        "--use-when",
        dest="use_when",
        type=nonblank,
        required=True,
        help="one sentence: when an agent should load this note",
    )
    parser.add_argument("--body", required=True, help="text inline, a file path, or - for stdin")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query Echo and manage wiki notes via echo-api.")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text, default_limit, func in (
        ("search", "hybrid semantic + lexical activity search", 10, cmd_search),
        ("grep", "exact substring scan over activity, newest first", 20, cmd_grep),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("query" if name == "search" else "pattern", type=nonblank)
        p.add_argument("--source", choices=SOURCES)
        p.add_argument("--kind", choices=KINDS)
        if name == "search":
            p.add_argument("--since", type=iso_date, help="YYYY-MM-DD lower bound on chunk date")
        p.add_argument("--limit", type=bounded_limit, default=default_limit)
        p.set_defaults(func=func)

    show = sub.add_parser("show", help="print one activity chunk verbatim")
    show.add_argument("id", type=int)
    show.set_defaults(func=cmd_show)

    wiki = sub.add_parser("wiki", help="search, read, add, or edit wiki notes").add_subparsers(
        dest="wiki", required=True
    )
    wiki_search = wiki.add_parser("search", help="search wiki notes")
    wiki_search.add_argument("query", nargs="?", default="", help="blank returns recently updated notes")
    wiki_search.add_argument("--limit", type=bounded_limit, default=10)
    wiki_search.set_defaults(func=cmd_wiki_search)
    wiki_show = wiki.add_parser("show", help="print one wiki note verbatim")
    wiki_show.add_argument("id", type=int)
    wiki_show.set_defaults(func=cmd_wiki_show)
    wiki_add = wiki.add_parser("add", help="create a wiki note (server embeds and attributes it)")
    add_wiki_write_args(wiki_add)
    wiki_add.set_defaults(func=cmd_wiki_add)
    wiki_edit = wiki.add_parser("edit", help="replace a wiki note's text and re-embed it")
    wiki_edit.add_argument("id", type=int)
    add_wiki_write_args(wiki_edit)
    wiki_edit.set_defaults(func=cmd_wiki_edit)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
