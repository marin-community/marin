# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cdx-toolkit>=0.9",
#     "trafilatura>=2.0",
#     "html2text>=2024.2",
#     "readability-lxml>=0.8",
#     "lxml[html_clean]>=5.3",
#     "datasets>=3.0",
#     "huggingface_hub>=0.25",
# ]
# ///
"""Refresh the parser-variant fixtures HuggingFace dataset.

For each entry in ``ARTICLES``: fetch the most recent Common Crawl capture,
extract text with trafilatura, html2text, and readability-lxml, and build
one row per (article, parser). Then push all rows as the
``parser_variants`` config of the target HF dataset and print the new
commit SHA so you can pin it in the dedup conftest.

Run with an HF token that can push to the target repo::

    HF_TOKEN=... uv run tests/.../parser_variants/generate_test_examples.py [--repo NAME] [--dry-run]

Adding a new article: append ``(slug, url)`` to ``ARTICLES``, run the
script, bump ``PARSER_VARIANTS_REVISION`` in the dedup conftest to the
printed commit SHA. PEP 723 inline ``# /// script`` deps keep all parser
and HF libraries out of ``pyproject.toml`` — ``uv run`` resolves them in
a transient environment.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterator

import cdx_toolkit
import html2text as html2text_mod
import trafilatura
from datasets import Dataset, Features, Value
from readability import Document

DEFAULT_REPO = "ravwojdyla/marin-test-data-fixtures"
CONFIG_NAME = "parser_variants"

ARTICLES: list[tuple[str, str]] = [
    ("wikipedia_isaac_newton", "https://en.wikipedia.org/wiki/Isaac_Newton"),
    ("wikipedia_georg_cantor", "https://en.wikipedia.org/wiki/Georg_Cantor"),
]

FEATURES = Features(
    {
        "article_slug": Value("string"),
        "parser": Value("string"),
        "text": Value("string"),
        "source_url": Value("string"),
        "warc_filename": Value("string"),
        "warc_offset": Value("int64"),
        "warc_length": Value("int64"),
        "capture_timestamp": Value("string"),
    }
)


def fetch_cc_html(url: str) -> tuple[str, dict]:
    """Pull the most recent CC capture for *url*; return (html, metadata)."""
    cdx = cdx_toolkit.CDXFetcher(source="cc")
    captures = list(cdx.iter(url, limit=1, filter=["status:200", "mime:text/html"]))
    if not captures:
        raise RuntimeError(f"No Common Crawl capture found for {url}")
    cap = captures[0]
    record = cap.fetch_warc_record()
    html_bytes = record.content_stream().read()
    html = html_bytes.decode("utf-8", errors="replace")
    metadata = {
        "source_url": url,
        "warc_filename": cap.data.get("filename", ""),
        "warc_offset": int(cap.data.get("offset", 0) or 0),
        "warc_length": int(cap.data.get("length", 0) or 0),
        "capture_timestamp": cap.data.get("timestamp", ""),
    }
    return html, metadata


def parse_with_trafilatura(html: str) -> str:
    return trafilatura.extract(html, include_comments=False, favor_recall=True) or ""


def parse_with_html2text(html: str) -> str:
    h = html2text_mod.HTML2Text()
    h.ignore_links = False
    h.body_width = 0
    return h.handle(html)


def parse_with_readability(html: str) -> str:
    doc = Document(html)
    summary_html = doc.summary()
    text = re.sub(r"<[^>]+>", " ", summary_html)
    text = re.sub(r"\s+", " ", text).strip()
    title = (doc.short_title() or "").strip()
    return f"{title}\n\n{text}" if title else text


PARSERS = {
    "trafilatura": parse_with_trafilatura,
    "html2text": parse_with_html2text,
    "readability": parse_with_readability,
}


def build_rows() -> Iterator[dict]:
    """Fetch every article in ``ARTICLES`` and yield one row per (article, parser)."""
    for slug, url in ARTICLES:
        print(f"=== {slug}: {url}", file=sys.stderr)
        html, metadata = fetch_cc_html(url)
        for parser_name, parser_fn in PARSERS.items():
            text = parser_fn(html)
            print(f"  {parser_name}: {len(text)} chars", file=sys.stderr)
            yield {"article_slug": slug, "parser": parser_name, "text": text, **metadata}


def main() -> None:
    cli = argparse.ArgumentParser(description="Refresh the parser-variant fixtures HF dataset.")
    cli.add_argument("--repo", default=DEFAULT_REPO, help=f"target HF dataset repo (default: {DEFAULT_REPO})")
    cli.add_argument("--dry-run", action="store_true", help="fetch and parse but skip the HF push")
    args = cli.parse_args()

    if not args.dry_run and not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN env var is required for non-dry-run uploads.", file=sys.stderr)
        sys.exit(2)

    rows = list(build_rows())
    print(f"\nBuilt {len(rows)} rows from {len(ARTICLES)} articles", file=sys.stderr)

    if args.dry_run:
        print("(dry run; skipping push)", file=sys.stderr)
        return

    ds = Dataset.from_list(rows, features=FEATURES)
    print(f"\nPushing to {args.repo} (config: {CONFIG_NAME})...", file=sys.stderr)
    info = ds.push_to_hub(args.repo, config_name=CONFIG_NAME, split="train")
    print(f"\nDone. Commit: {info.oid}")
    print(f"      URL:    {info.commit_url}")
    print(f"\nNext: bump PARSER_VARIANTS_REVISION in the dedup conftest to:\n      {info.oid}")


if __name__ == "__main__":
    main()
