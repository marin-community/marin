# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
from zephyr.readers import load_jsonl, load_parquet

# Pinned HF dataset for data_integration test fixtures. The generation
# scripts under resources/parser_variants/ each push one config and print
# a commit SHA — paste that SHA into the corresponding ``*_REVISION``
# constant below.
DATASET_REPO = "ravwojdyla/marin-test-data-fixtures"
PARSER_VARIANTS_REVISION = "b4410029dd8fd57171283c681912adc3a5092e88"
SAME_SITE_DISTINCT_REVISION = "88401e16e3ae1f03e23a9ca3b48d453727af41ad"

# Back-compat alias used by existing parser-variants fixtures below.
PARSER_VARIANTS_REPO = DATASET_REPO
PARSER_VARIANTS_CONFIG = "parser_variants"


@pytest.fixture(scope="module")
def docs():
    test_resources = Path(__file__).parent.joinpath("resources", "docs")
    docs = {}
    for doc_file in test_resources.glob("*.txt"):
        docs[doc_file.stem] = doc_file.read_text()
    return docs


@pytest.fixture(scope="session")
def parser_variants_corpus():
    """Pinned ``parser_variants`` config from the marin test-fixtures HF dataset.

    Tests using this fixture must be marked ``@pytest.mark.data_integration``
    so they only run from CI workflows that have HF access (``HF_TOKEN``
    is unnecessary for the public dataset, but the marker keeps these
    network-touching tests off the unit-test job).
    """
    from datasets import load_dataset

    return load_dataset(
        PARSER_VARIANTS_REPO,
        PARSER_VARIANTS_CONFIG,
        revision=PARSER_VARIANTS_REVISION,
        split="train",
    )


@pytest.fixture
def parser_variants_docs(parser_variants_corpus) -> list[dict]:
    """Parser-variant rows reshaped as ingestible ``{id, text}`` records."""
    return [{"id": f"{r['article_slug']}__{r['parser']}", "text": r["text"]} for r in parser_variants_corpus]


@pytest.fixture
def parser_variants_articles(parser_variants_corpus) -> list[str]:
    """Sorted distinct article slugs present in the corpus."""
    return sorted({r["article_slug"] for r in parser_variants_corpus})


@pytest.fixture(scope="session")
def same_site_distinct_corpus():
    """Pinned ``same_site_distinct_bodies`` config from the marin test-fixtures HF dataset.

    Wikipedia pages on disjoint topics, html2text-extracted so site chrome
    is preserved. Drives the precision regression that distinct article
    bodies must not cluster despite shared template.
    """
    from datasets import load_dataset

    return load_dataset(
        DATASET_REPO,
        "same_site_distinct_bodies",
        revision=SAME_SITE_DISTINCT_REVISION,
        split="train",
    )


@pytest.fixture
def same_site_distinct_docs(same_site_distinct_corpus) -> list[dict]:
    """Same-site rows reshaped as ingestible ``{id, text}`` records."""
    return [{"id": r["doc_id"], "text": r["text"]} for r in same_site_distinct_corpus]


def load_dedup_outputs(output_dir: str) -> dict[str, dict]:
    """Load all dedupe output files and return as id->doc mapping.

    Args:
        output_dir: Directory containing .jsonl.gz output files

    Returns:
        Dictionary mapping document IDs to document records
    """
    output_files = list(Path(output_dir).glob("**/*.jsonl.gz"))
    results = []
    for output_file in output_files:
        results.extend(load_jsonl(str(output_file)))
    return {r["id"]: r for r in results}


def load_dedup_parquet_outputs(output_dir: str) -> dict[str, list[dict]]:
    """Load all dedup parquet output files keyed by output filename stem.

    Returns:
        Dictionary mapping output file stem (e.g. "test_shard_0") to list of records.
    """
    output_files = sorted(Path(output_dir).glob("**/*.parquet"))
    by_file: dict[str, list[dict]] = {}
    for output_file in output_files:
        by_file[output_file.stem] = list(load_parquet(str(output_file)))
    return by_file
