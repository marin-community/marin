# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "html2text>=2024.2",
#     "lxml[html_clean]>=5.3",
#     "datasets>=3.0",
#     "huggingface_hub>=0.25",
# ]
# ///
"""Build and push the ``same_site_distinct_bodies`` config.

Drives the precision regression "shared site chrome must not cluster
distinct article bodies." Uses ``html2text`` (which preserves chrome) so
the test exercises the realistic precision pressure — trafilatura would
strip the chrome and trivialise the assertion.

Articles are Wikipedia pages on disjoint topics. Wikipedia is the only
sufficiently archived site for this experiment: BBC, Guardian, Reuters,
NPR, etc. all block Common Crawl via ``robots.txt``. Distinct Wikipedia
topics still share the wiki chrome (sidebar, infobox, citations format,
edit links), exercising the "shared template, distinct bodies" case.

Fetches HTML directly from the live wiki (``fetch_live_html``) rather than
CC: the CC index API is occasionally rate-limited / flaky, and once the
rows are pushed they are pinned by an HF dataset commit SHA, so the source
method does not affect downstream reproducibility.

Run with::

    HF_TOKEN=... uv run tests/.../parser_variants/generate_same_site_distinct_bodies.py [--dry-run]
"""
from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _utils import cli_main, fetch_live_html, parse_with_html2text
from datasets import Features, Value

CONFIG_NAME = "same_site_distinct_bodies"

ARTICLES: list[tuple[str, str]] = [
    ("wikipedia_photography", "https://en.wikipedia.org/wiki/Photography"),
    ("wikipedia_photosynthesis", "https://en.wikipedia.org/wiki/Photosynthesis"),
    ("wikipedia_quantum_mechanics", "https://en.wikipedia.org/wiki/Quantum_mechanics"),
    ("wikipedia_roman_empire", "https://en.wikipedia.org/wiki/Roman_Empire"),
]

FEATURES = Features(
    {
        "doc_id": Value("string"),
        "text": Value("string"),
        "source_url": Value("string"),
        "warc_filename": Value("string"),
        "warc_offset": Value("int64"),
        "warc_length": Value("int64"),
        "capture_timestamp": Value("string"),
    }
)


def build_rows() -> Iterator[dict]:
    for slug, url in ARTICLES:
        print(f"=== {slug}: {url}", file=sys.stderr)
        html, metadata = fetch_live_html(url)
        text = parse_with_html2text(html)
        print(f"  html2text: {len(text)} chars", file=sys.stderr)
        yield {"doc_id": slug, "text": text, **metadata}


if __name__ == "__main__":
    cli_main(config_name=CONFIG_NAME, build_rows=build_rows, features=FEATURES)
