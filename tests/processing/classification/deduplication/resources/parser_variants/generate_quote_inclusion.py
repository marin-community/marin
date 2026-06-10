# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "trafilatura>=2.0",
#     "lxml[html_clean]>=5.3",
#     "datasets>=3.0",
#     "huggingface_hub>=0.25",
# ]
# ///
"""Build and push the ``quote_inclusion`` config.

Stores two real Wikipedia articles. The third "host article with an
embedded long quote of the other" document is constructed at *test time*
inside the test rather than committed to HF — that keeps the fixture
minimal (two rows) and makes the construction explicit at the assertion
site.

Drives the precision regression "a host article that quotes another
article must not cluster with the quoted source." Common case in
citation patterns: a blog post quoting a paper, an article quoting
another article. Should stay distinct unless the quote dominates the
host document.

Run with::

    HF_TOKEN=... uv run tests/.../parser_variants/generate_quote_inclusion.py [--dry-run]
"""
from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _utils import cli_main, fetch_live_html, parse_with_trafilatura
from datasets import Features, Value

CONFIG_NAME = "quote_inclusion"

ARTICLES: list[tuple[str, str]] = [
    ("article_a", "https://en.wikipedia.org/wiki/Climate_change"),
    ("article_b", "https://en.wikipedia.org/wiki/Industrial_Revolution"),
]

FEATURES = Features(
    {
        "doc_id": Value("string"),
        "text": Value("string"),
        "source_url": Value("string"),
        "capture_timestamp": Value("string"),
        "archive_source": Value("string"),
    }
)


def build_rows() -> Iterator[dict]:
    for slug, url in ARTICLES:
        print(f"=== {slug}: {url}", file=sys.stderr)
        html, metadata = fetch_live_html(url)
        text = parse_with_trafilatura(html)
        print(f"  trafilatura: {len(text)} chars", file=sys.stderr)
        yield {
            "doc_id": slug,
            "text": text,
            "source_url": metadata["source_url"],
            "capture_timestamp": metadata["capture_timestamp"],
            "archive_source": "live",
        }


if __name__ == "__main__":
    cli_main(config_name=CONFIG_NAME, build_rows=build_rows, features=FEATURES)
