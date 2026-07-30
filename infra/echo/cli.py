#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Search Echo wiki, GitHub, Discord, and repository files.

Run inside the repo environment, e.g. ``uv run infra/echo/cli.py search "..."``. This shares
Marin's IAP login helpers from ``marin-rigging`` (not on PyPI), so it is not a standalone
``uv --script``.

Commands call the IAP-gated ``echo-api`` Cloud Run service with a Google-signed ID token.
There is no separate echo login: humans authenticate once with ``iris login`` (cached
under ``~/.config/marin/credentials``) and this reuses that token; agents and CI use
ambient service-account credentials with no login. See ``infra/echo/README.md``.

    uv run infra/echo/cli.py search "expert parallel MoE MFU on B200" --limit 10
    uv run infra/echo/cli.py get file:lib/iris/OPS.md
    uv run infra/echo/cli.py grep ragged_all_to_all --source discord
    uv run infra/echo/cli.py wiki search "grafana access" --tag ops
    uv run infra/echo/cli.py wiki add --file note.md          # OKF: frontmatter title/use_when + body
    uv run infra/echo/cli.py wiki show 12 > note.md           # export as OKF, edit, then:
    uv run infra/echo/cli.py wiki edit 12 --file note.md
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import quote

import okf
import requests
import search_config
from rigging.auth import (
    MARIN_DESKTOP_OAUTH_CLIENT,
    IapCredentialsUnavailable,
    IapLoginRequired,
    IapServiceAccountTokenProvider,
    TokenProvider,
)
from rigging.credential_store import credentials_dir
from rigging.credentials import iap_edge_provider
from search_result import SearchResult

# echo.oa.dev maps to the echo-api Cloud Run service; the IAP token audience is the shared
# Marin desktop OAuth client, which echo-api's IAP settings admit as a programmatic client.
API_URL = os.environ.get("ECHO_API_URL", search_config.PUBLIC_URL).rstrip("/")
# Data endpoints live under /api on the service; the dashboard SPA owns the bare paths.
API_BASE = f"{API_URL}/api"
AUDIENCE = MARIN_DESKTOP_OAUTH_CLIENT.client_id
# The rigging cluster whose cached login to prefer; every Marin login shares the desktop
# client, so any cached record works and `iris login` alone is enough.
LOGIN_CLUSTER = os.environ.get("ECHO_LOGIN_CLUSTER", "marin")
LOGIN_HINT = "run `iris login`"
SOURCES = ("github", "discord")
KINDS = ("issue", "pr", "comment", "message")
DOMAINS = search_config.SEARCH_DOMAINS
DEFAULT_DOMAINS = search_config.DEFAULT_SEARCH_DOMAINS
SEARCH_DETAIL_INSTRUCTION = "Detail: uv run infra/echo/cli.py get <domain:id>"
MISSING_EMAIL_SCOPE_WARNING = "Not all requested scopes were granted by the authorization server, missing scopes email."


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


def keep_oauth_log(record: logging.LogRecord) -> bool:
    return record.getMessage() != MISSING_EMAIL_SCOPE_WARNING


def bearer_token() -> str:
    """A Google ID token for echo-api's IAP: cached human login first, else ambient SA creds."""
    provider = cached_login_provider() or IapServiceAccountTokenProvider(AUDIENCE)
    oauth_logger = logging.getLogger("google.oauth2.credentials")
    oauth_logger.addFilter(keep_oauth_log)
    try:
        try:
            token = provider.get_token()
        finally:
            oauth_logger.removeFilter(keep_oauth_log)
    except (IapCredentialsUnavailable, IapLoginRequired) as error:
        raise SystemExit(f"{error}\nHuman callers: {LOGIN_HINT}.") from error
    if not token:
        raise SystemExit("could not obtain an IAP token for echo-api")
    return token


def request(method: str, path: str, *, params: dict | None = None, body: dict | None = None) -> object:
    """Call echo-api and return the decoded JSON, or exit with a message on any HTTP error."""
    response = requests.request(
        method,
        f"{API_BASE}{path}",
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


def response_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SystemExit("echo-api returned a non-object response")
    return value


def response_objects(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise SystemExit("echo-api returned a non-list response")
    return value


def print_hits(hits: list[dict]) -> None:
    for hit in hits:
        score = f"{hit['score']:.4f} " if hit.get("score") else ""
        # For discord, title is only the channel; the snippet carries the content.
        headline = (
            f"{hit['title']}: {hit['snippet']}" if hit["source"] == "discord" else (hit["title"] or hit["snippet"])
        )
        print(f"#{hit['id']} {score}[{hit['source']}/{hit['kind']}] {hit['date']} {hit['author'] or '?'} — {headline}")
        print(f"    {hit['url']}")


def print_search_results(results: list[SearchResult]) -> None:
    print(SEARCH_DETAIL_INSTRUCTION)
    if not results:
        print("No results.")
        return

    ids = [result.id for result in results]
    titles = [one_line(result.title) for result in results]
    id_width = max(len("ID"), *(len(value) for value in ids))
    available = max(40, shutil.get_terminal_size(fallback=(160, 24)).columns - id_width - 4)
    title_width = min(max(len("TITLE"), *(len(value) for value in titles)), 36, max(16, available // 3))
    detail_width = max(20, available - title_width)
    print(f"{'ID':<{id_width}}  {'TITLE':<{title_width}}  DETAIL")
    for result in results:
        if result.references:
            detail = " · ".join(f"L{reference.line} {reference.text}" for reference in result.references)
        else:
            detail = result.subtitle if result.domain == "wiki" else result.snippet
        print(
            f"{result.id:<{id_width}}  "
            f"{truncate_cell(one_line(result.title), title_width):<{title_width}}  "
            f"{truncate_cell(one_line(detail), detail_width)}"
        )


def one_line(value: str) -> str:
    return " ".join(value.split())


def truncate_cell(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return f"{value[: width - 1]}…"


def print_wiki(entries: list[dict]) -> None:
    for entry in entries:
        print(f"#{entry['id']} {entry['title']} (refs={entry.get('reference_count', 0)}, by {entry['author']})")
        print(f"    use when: {entry['use_when']}")
        if entry.get("tags"):
            print(f"    tags: {', '.join(entry['tags'])}")
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
    domains = list(dict.fromkeys(args.domain or DEFAULT_DOMAINS))
    remote_value = response_objects(
        request(
            "GET",
            "/federated-search",
            params={"q": args.query, "domain": domains, "limit": args.limit},
        )
    )
    results = [SearchResult.from_json(result) for result in remote_value]
    print_search_results(results)


def cmd_grep(args: argparse.Namespace) -> None:
    print_hits(
        response_objects(
            request(
                "GET",
                "/grep",
                params={"pattern": args.pattern, "source": args.source, "kind": args.kind, "limit": args.limit},
            )
        )
    )


def cmd_get(args: argparse.Namespace) -> None:
    domain, _, value = args.id.partition(":")
    if domain == "wiki":
        entry = response_object(request("GET", f"/wiki/{value}"))
        title = entry["title"]
        subtitle = entry["use_when"]
        url = wiki_link(int(value))
        text = entry["body"]
    elif domain == "file":
        file = response_object(request("GET", f"/repository-files/{quote(value, safe='/')}"))
        title, subtitle, url, text = file["title"], file["subtitle"], file["url"], file["text"]
    else:
        chunk = response_object(request("GET", f"/chunks/{value}"))
        actual_domain = chunk_domain(chunk)
        if actual_domain != domain:
            raise SystemExit(f"{args.id} identifies a {actual_domain} result")
        title = chunk.get("title") or chunk.get("snippet") or chunk["url"]
        details = [domain]
        if chunk.get("author"):
            details.append(str(chunk["author"]))
        if chunk.get("date"):
            details.append(str(chunk["date"]))
        subtitle = " · ".join(details)
        url, text = chunk["url"], chunk.get("text") or ""
    print(f"[{args.id}] {title}")
    if subtitle:
        print(subtitle)
    print(f"{url}\n")
    print(text)


def chunk_domain(chunk: dict[str, object]) -> str:
    if chunk["source"] == "discord":
        return "discord"
    if chunk["kind"] == "pr" or "/pull/" in str(chunk["url"]):
        return "pr"
    return "issue"


def cmd_wiki_search(args: argparse.Namespace) -> None:
    print_wiki(
        response_objects(request("GET", "/wiki/search", params={"q": args.query, "tag": args.tag, "limit": args.limit}))
    )


def cmd_wiki_show(args: argparse.Namespace) -> None:
    # Emit the entry as an OKF document so it round-trips through a file: `wiki show > note.md`,
    # edit, `wiki edit --file note.md`.
    entry = response_object(request("GET", f"/wiki/{args.id}"))
    print(okf.wiki_to_okf(entry, resource=wiki_link(args.id)), end="")


def wiki_link(entry_id: int) -> str:
    """Browseable URL for a wiki entry — the dashboard's client-side route, not the API path."""
    return f"{API_URL}/wiki/{entry_id}"


def wiki_write_body(args: argparse.Namespace) -> dict[str, object]:
    """The fields for a wiki write, from an OKF ``--file`` or individual flags."""
    if args.file:
        text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text()
        try:
            fields = okf.parse_wiki(text)
        except ValueError as error:
            raise SystemExit(f"{args.file}: {error}") from error
        return {"title": fields.title, "use_when": fields.use_when, "tags": list(fields.tags), "body": fields.body}
    missing = [name for name in ("title", "use_when", "body") if not getattr(args, name)]
    if missing:
        raise SystemExit(f"provide --file (OKF) or all of --title/--use-when/--body (missing: {', '.join(missing)})")
    return {"title": args.title, "use_when": args.use_when, "tags": args.tag, "body": read_body(args.body)}


def cmd_wiki_add(args: argparse.Namespace) -> None:
    entry = response_object(request("POST", "/wiki", body=wiki_write_body(args)))
    print(f"created wiki #{entry['id']}: {entry['title']}")
    print(wiki_link(entry["id"]))


def cmd_wiki_edit(args: argparse.Namespace) -> None:
    entry = response_object(request("PUT", f"/wiki/{args.id}", body=wiki_write_body(args)))
    print(f"updated wiki #{entry['id']}: {entry['title']}")
    print(wiki_link(entry["id"]))


def bounded_limit(value: str) -> int:
    limit = int(value)
    if not 1 <= limit <= 100:
        raise argparse.ArgumentTypeError("must be between 1 and 100")
    return limit


def nonblank(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be blank")
    return value


def artifact_id(value: str) -> str:
    domain, separator, detail = value.partition(":")
    if not separator or domain not in DOMAINS or not detail:
        raise argparse.ArgumentTypeError("must be <wiki|file|discord|pr|issue>:<id>")
    if domain != "file" and not detail.isdecimal():
        raise argparse.ArgumentTypeError(f"{domain} result IDs are numeric")
    return value


def add_wiki_write_args(parser: argparse.ArgumentParser) -> None:
    # Either an OKF document (--file) or the three fields as flags.
    parser.add_argument("--file", help="OKF markdown file (frontmatter title/use_when + body), or - for stdin")
    parser.add_argument("--title", type=nonblank)
    parser.add_argument(
        "--use-when",
        dest="use_when",
        type=nonblank,
        help="one sentence: when an agent should load this note",
    )
    parser.add_argument("--body", help="text inline, a file path, or - for stdin")
    parser.add_argument("--tag", action="append", default=[], help="lowercase kebab-case tag; repeat as needed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query Echo and manage wiki notes via echo-api.")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="federated semantic + lexical search")
    search.add_argument("query", type=nonblank)
    search.add_argument(
        "--domain",
        action="append",
        choices=DOMAINS,
        help="search this domain; repeat to select several (default: wiki,file,pr,issue)",
    )
    search.add_argument("--limit", type=bounded_limit, default=10)
    search.set_defaults(func=cmd_search)

    grep = sub.add_parser("grep", help="exact substring scan over activity, newest first")
    grep.add_argument("pattern", type=nonblank)
    grep.add_argument("--source", choices=SOURCES)
    grep.add_argument("--kind", choices=KINDS)
    grep.add_argument("--limit", type=bounded_limit, default=20)
    grep.set_defaults(func=cmd_grep)

    get = sub.add_parser("get", help="print full detail for a federated-search result ID")
    get.add_argument("id", type=artifact_id)
    get.set_defaults(func=cmd_get)

    wiki = sub.add_parser("wiki", help="search, read, add, or edit wiki notes").add_subparsers(
        dest="wiki", required=True
    )
    wiki_search = wiki.add_parser("search", help="search wiki notes")
    wiki_search.add_argument("query", nargs="?", default="", help="blank returns recently updated notes")
    wiki_search.add_argument("--tag", action="append", default=[], help="require this tag; repeat to require several")
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
