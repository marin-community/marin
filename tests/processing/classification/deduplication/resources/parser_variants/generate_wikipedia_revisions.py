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
"""Build and push the ``wikipedia_revisions`` config.

Drives the recall regression "minor temporal drift on the same article
must still cluster." For each article URL in ``ARTICLES``, fetches one
Wayback Machine snapshot from each year in ``YEARS`` (typically 3 years
spanning 2-4 calendar years apart). Wikipedia bodies on stable topics
drift slightly between revisions — paragraphs added, citations refreshed
— so char-Jaccard between captures stays high but not 1.0.

Wayback (rather than CC) because CC rarely has multiple captures of one
URL while Wayback typically has hundreds. Trafilatura output keeps the
fixture lean and the recall threshold loose.

Run with::

    HF_TOKEN=... uv run tests/.../parser_variants/generate_wikipedia_revisions.py [--dry-run]
"""
from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _utils import cli_main, fetch_wayback_html, parse_with_trafilatura
from datasets import Features, Value

CONFIG_NAME = "wikipedia_revisions"

ARTICLES: list[tuple[str, str]] = [
    ("mount_everest", "https://en.wikipedia.org/wiki/Mount_Everest"),
    ("world_war_ii", "https://en.wikipedia.org/wiki/World_War_II"),
]

YEARS = (2020, 2022, 2024)

FEATURES = Features(
    {
        "doc_id": Value("string"),
        "article_slug": Value("string"),
        "text": Value("string"),
        "source_url": Value("string"),
        "capture_timestamp": Value("string"),
        "archive_source": Value("string"),
    }
)


def build_rows() -> Iterator[dict]:
    for slug, url in ARTICLES:
        for year in YEARS:
            print(f"=== {slug} @ {year}: {url}", file=sys.stderr)
            html, metadata = fetch_wayback_html(url, year=year)
            text = parse_with_trafilatura(html)
            print(f"  trafilatura: {len(text)} chars (ts={metadata['capture_timestamp']})", file=sys.stderr)
            yield {
                "doc_id": f"{slug}__{metadata['capture_timestamp']}",
                "article_slug": slug,
                "text": text,
                **metadata,
            }


if __name__ == "__main__":
    cli_main(config_name=CONFIG_NAME, build_rows=build_rows, features=FEATURES)
