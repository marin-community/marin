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
# ]
# ///
"""Regenerate parser-variant regression fixtures from Common Crawl HTML.

Run from anywhere::

    uv run tests/processing/classification/deduplication/resources/parser_variants/generate_test_examples.py

Fetches the most recent Common Crawl capture for each entry in ``ARTICLES``,
runs trafilatura, html2text, and readability-lxml on the raw HTML, and writes
one fixture directory per article containing:

* ``metadata.json`` — source URL, CC record locator, capture timestamp
* ``trafilatura.txt`` / ``html2text.txt`` / ``readability.txt`` — parser output

(Raw HTML is intentionally not committed: it would add ~1.5 MB per article
and is recoverable from CC via the WARC locator in ``metadata.json``.)

The committed fixtures are the source of truth for
``test_html_parser_variants_cluster_per_article``. This script is only run
when adding new articles or refreshing parser outputs after a parser
upgrade. Inline ``# /// script`` deps keep the parsers out of the project's
runtime requirements; ``uv run`` resolves them in a transient environment.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cdx_toolkit
import html2text as html2text_mod
import trafilatura
from readability import Document

ARTICLES: list[tuple[str, str]] = [
    ("wikipedia_isaac_newton", "https://en.wikipedia.org/wiki/Isaac_Newton"),
    ("wikipedia_georg_cantor", "https://en.wikipedia.org/wiki/Georg_Cantor"),
]

OUT_ROOT = Path(__file__).parent


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
        "cc_index": cap.data.get("cdx_api"),
        "warc_filename": cap.data.get("filename"),
        "warc_offset": cap.data.get("offset"),
        "warc_length": cap.data.get("length"),
        "capture_timestamp": cap.data.get("timestamp"),
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


def main() -> None:
    for slug, url in ARTICLES:
        print(f"=== {slug}: {url}")
        article_dir = OUT_ROOT / slug
        article_dir.mkdir(exist_ok=True)
        html, metadata = fetch_cc_html(url)
        (article_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for name, fn in [
            ("trafilatura", parse_with_trafilatura),
            ("html2text", parse_with_html2text),
            ("readability", parse_with_readability),
        ]:
            text = fn(html)
            (article_dir / f"{name}.txt").write_text(text, encoding="utf-8")
            print(f"  {name}: {len(text)} chars")


if __name__ == "__main__":
    main()
