# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the per-scenario fixture generation scripts.

Imported by the ``generate_*.py`` scripts under the same directory. This
module is not a PEP 723 script itself — the importing scripts pull in the
runtime deps via their ``# /// script`` blocks.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import sys
from collections.abc import Callable, Iterable, Iterator
from urllib.parse import quote
from urllib.request import Request, urlopen

from datasets import Dataset, Features

DEFAULT_REPO = "ravwojdyla/marin-test-data-fixtures"

# Recent CC monthly indexes to probe in order (newest-first). Each generation
# script lookup walks this list until it finds a 200/text-html capture for
# the URL, then range-fetches the WARC record. Hardcoding the list (rather
# than pulling collinfo.json) keeps fetches deterministic and fast.
CC_INDEXES = (
    "CC-MAIN-2026-17",
    "CC-MAIN-2026-12",
    "CC-MAIN-2026-08",
    "CC-MAIN-2026-04",
    "CC-MAIN-2025-51",
    "CC-MAIN-2025-46",
    "CC-MAIN-2025-42",
    "CC-MAIN-2025-38",
)


def _cc_index_lookup(url: str) -> dict | None:
    """Return the first CC capture row matching *url* across CC_INDEXES, or None."""
    for index_id in CC_INDEXES:
        api = (
            f"https://index.commoncrawl.org/{index_id}-index"
            f"?url={quote(url, safe='')}&output=json&limit=1&filter=%3Dstatus%3A200"
        )
        try:
            body = urlopen(Request(api), timeout=30).read().decode()
        except Exception:
            continue
        text = body.strip()
        if not text or "No Captures" in text:
            continue
        try:
            return json.loads(text.split("\n", 1)[0])
        except json.JSONDecodeError:
            continue
    return None


def fetch_cc_html(url: str) -> tuple[str, dict]:
    """Pull the most recent CC capture for *url* via direct CC index + WARC range."""
    cap = _cc_index_lookup(url)
    if cap is None:
        raise RuntimeError(f"No Common Crawl capture found for {url} in recent indexes")
    warc_url = f"https://data.commoncrawl.org/{cap['filename']}"
    offset = int(cap["offset"])
    length = int(cap["length"])
    req = Request(warc_url, headers={"Range": f"bytes={offset}-{offset + length - 1}"})
    warc_chunk = urlopen(req, timeout=120).read()
    with gzip.GzipFile(fileobj=io.BytesIO(warc_chunk)) as gz:
        record_bytes = gz.read()
    # WARC record: WARC headers \r\n\r\n HTTP headers \r\n\r\n HTTP body.
    parts = record_bytes.split(b"\r\n\r\n", 2)
    if len(parts) < 3:
        raise RuntimeError(f"Unexpected WARC record format for {url}")
    html = parts[2].decode("utf-8", errors="replace")
    metadata = {
        "source_url": url,
        "warc_filename": cap["filename"],
        "warc_offset": offset,
        "warc_length": length,
        "capture_timestamp": cap["timestamp"],
    }
    return html, metadata


def _http_get_with_retry(url: str, *, timeout: int = 60, attempts: int = 5) -> bytes:
    """GET *url* with exponential backoff on 5xx / transient network errors.

    Decompresses gzip-encoded responses (Wayback's ``id_`` modifier
    preserves the original capture's Content-Encoding, so snapshots crawled
    from gzip-serving CDNs come back gzipped regardless of request headers).
    """
    import time

    delay = 2.0
    last_err: Exception | None = None
    for _ in range(attempts):
        try:
            req = Request(url, headers={"User-Agent": "marin-test-fixtures/0.1"})
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                encoding = (resp.headers.get("Content-Encoding") or "").lower()
            if encoding == "gzip" or body[:2] == b"\x1f\x8b":
                body = gzip.decompress(body)
            return body
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError(f"GET failed after {attempts} attempts: {url} ({last_err})")


def fetch_wayback_html(url: str, *, year: int) -> tuple[str, dict]:
    """Fetch one Wayback Machine snapshot of *url* from a given calendar *year*.

    Wayback's CDX search returns the earliest 200/text-html capture in the
    requested year; the snapshot is then fetched with the ``id_`` modifier
    so we get the original page bytes (no Wayback toolbar HTML injected).
    Used by configs that need temporally-spaced captures of the same URL —
    Common Crawl rarely has multiple captures of one URL while Wayback
    typically has hundreds. Wayback 5xx errors are retried with backoff
    since the service is occasionally rate-limited.
    """
    cdx = (
        f"https://web.archive.org/cdx/search/cdx?url={quote(url, safe='')}"
        f"&output=json&limit=1&from={year}0101&to={year}1231"
        f"&filter=statuscode:200&filter=mimetype:text/html"
    )
    # Wayback CDX is consistently slow (typically 30-90s per query); allow plenty of headroom.
    rows = json.loads(_http_get_with_retry(cdx, timeout=180).decode())
    if len(rows) <= 1:
        raise RuntimeError(f"No Wayback capture for {url} in {year}")
    timestamp = rows[1][1]
    snapshot_url = f"https://web.archive.org/web/{timestamp}id_/{url}"
    html = _http_get_with_retry(snapshot_url, timeout=180).decode("utf-8", errors="replace")
    metadata = {
        "source_url": url,
        "capture_timestamp": timestamp,
        "archive_source": "wayback",
    }
    return html, metadata


def fetch_live_html(url: str) -> tuple[str, dict]:
    """Fetch *url* directly via HTTP.

    Fallback for when Common Crawl is unavailable or rate-limited (CC's index
    API is occasionally flaky). The returned ``metadata`` matches the shape
    ``fetch_cc_html`` returns so downstream schemas stay uniform; WARC fields
    are empty / 0 to signal the source was the live web.
    """
    from datetime import UTC, datetime

    req = Request(url, headers={"User-Agent": "marin-test-fixtures/0.1 (+https://github.com/marin-community/marin)"})
    with urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    metadata = {
        "source_url": url,
        "warc_filename": "",
        "warc_offset": 0,
        "warc_length": 0,
        "capture_timestamp": datetime.now(UTC).strftime("%Y%m%d%H%M%S"),
    }
    return html, metadata


def parse_with_trafilatura(html: str) -> str:
    import trafilatura

    return trafilatura.extract(html, include_comments=False, favor_recall=True) or ""


def parse_with_html2text(html: str) -> str:
    import html2text as html2text_mod

    h = html2text_mod.HTML2Text()
    h.ignore_links = False
    h.body_width = 0
    return h.handle(html)


def parse_with_readability(html: str) -> str:
    from readability import Document

    doc = Document(html)
    summary_html = doc.summary()
    text = re.sub(r"<[^>]+>", " ", summary_html)
    text = re.sub(r"\s+", " ", text).strip()
    title = (doc.short_title() or "").strip()
    return f"{title}\n\n{text}" if title else text


def push_config(rows: Iterable[dict], *, repo: str, config_name: str, features: Features):
    """Push *rows* as a HF dataset config; return huggingface_hub CommitInfo."""
    ds = Dataset.from_list(list(rows), features=features)
    print(f"Pushing to {repo} (config: {config_name})...", file=sys.stderr)
    return ds.push_to_hub(repo, config_name=config_name, split="train")


def cli_main(*, config_name: str, build_rows: Callable[[], Iterator[dict]], features: Features) -> None:
    """Standard CLI entry point for the per-scenario generation scripts.

    Parses ``--repo`` / ``--dry-run``, builds rows via *build_rows*, and either
    pushes to HF or prints a summary. Prints the resulting commit SHA so the
    caller can paste it into the corresponding ``*_REVISION`` constant in the
    dedup conftest.
    """
    cli = argparse.ArgumentParser(description=f"Refresh the {config_name!r} fixture config.")
    cli.add_argument("--repo", default=DEFAULT_REPO, help=f"target HF dataset repo (default: {DEFAULT_REPO})")
    cli.add_argument("--dry-run", action="store_true", help="fetch and parse but skip the HF push")
    args = cli.parse_args()

    if not args.dry_run and not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN env var is required for non-dry-run uploads.", file=sys.stderr)
        sys.exit(2)

    rows = list(build_rows())
    print(f"\nBuilt {len(rows)} rows for config {config_name!r}", file=sys.stderr)

    if args.dry_run:
        print("(dry run; skipping push)", file=sys.stderr)
        return

    info = push_config(rows, repo=args.repo, config_name=config_name, features=features)
    print(f"\nDone. Commit: {info.oid}")
    print(f"      URL:    {info.commit_url}")
    print(f"\nNext: bump the revision for {config_name!r} in the dedup conftest to:")
    print(f"      {info.oid}")
