# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "trafilatura>=2.0",
#     "html2text>=2024.2",
#     "readability-lxml>=0.8",
#     "lxml[html_clean]>=5.3",
#     "datasets>=3.0",
#     "huggingface_hub>=0.25",
# ]
# ///
"""Build and push the ``parser_variants`` config of the test-data dataset.

For each Wikipedia article in ``ARTICLES``: fetch the most recent CC capture,
extract text with trafilatura, html2text, and readability-lxml, and emit
one row per (article, parser).

Run with::

    HF_TOKEN=... uv run tests/.../parser_variants/generate_parser_variants.py [--dry-run]
"""
from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

# Make sibling _utils importable when invoked via ``uv run --script``.
sys.path.insert(0, str(Path(__file__).parent))

from _utils import (
    cli_main,
    fetch_cc_html,
    parse_with_html2text,
    parse_with_readability,
    parse_with_trafilatura,
)
from datasets import Features, Value

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


def build_rows() -> Iterator[dict]:
    parsers = {
        "trafilatura": parse_with_trafilatura,
        "html2text": parse_with_html2text,
        "readability": parse_with_readability,
    }
    for slug, url in ARTICLES:
        print(f"=== {slug}: {url}", file=sys.stderr)
        html, metadata = fetch_cc_html(url)
        for parser_name, parser_fn in parsers.items():
            text = parser_fn(html)
            print(f"  {parser_name}: {len(text)} chars", file=sys.stderr)
            yield {"article_slug": slug, "parser": parser_name, "text": text, **metadata}


if __name__ == "__main__":
    cli_main(config_name=CONFIG_NAME, build_rows=build_rows, features=FEATURES)
