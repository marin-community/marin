# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize the pre-staged Focus Crawl (CC-SUPPLEMENTAL-2026-22) extraction.

A one-off run outside this repository pulled the indexed HTML response byte ranges
out of all 4,573 CC-SUPPLEMENTAL-2026-22 WARCs with XenonMolecule/jusText and wrote
the text and WARC provenance as Parquet. Those shards are the input here, not the
finished source: they hold duplicate ids and no row order, so the chain sends them
through ``normalize_step`` like every other datakit source.

The extraction is read from ``{MARIN_PREFIX}/<_FOCUS_CRAWL_EXTRACTION>``, so each
cluster reads it from its own bucket and it must be copied in before a run. The
canonical copy lives on CoreWeave at
``s3://marin-us-east-02a/marin/data/datakit/normalized/common_crawl_focus_2026_22_ed4b8bc9``
(4,573 Parquet shards, 89.4 GB, 36,327,068 documents).
"""

from rigging.filesystem import StoragePath, prefix_join

from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

# Path of the extraction relative to MARIN_PREFIX. The ``normalized/`` segment is a
# leftover from the one-off run that wrote the tree; the shards below it are raw
# extractor output.
_FOCUS_CRAWL_EXTRACTION = "data/datakit/normalized/common_crawl_focus_2026_22_ed4b8bc9"
_EXTRACTION_SHARDS = "outputs/main"
# One shard per WARC. An interrupted copy leaves fewer, and normalizing that
# publishes a valid-looking source built from part of the crawl.
_EXTRACTION_SHARD_COUNT = 4573


def _validate_focus_crawl_extraction(output_path: str) -> None:
    shard_glob = prefix_join(output_path, f"{_EXTRACTION_SHARDS}/*.parquet")
    found = sum(1 for _ in StoragePath(shard_glob).glob())
    if found != _EXTRACTION_SHARD_COUNT:
        raise FileNotFoundError(f"Expected {_EXTRACTION_SHARD_COUNT} Parquet shards under {shard_glob}, found {found}")


def common_crawl_focus_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(extraction, normalize)`` chain for the Focus Crawl source."""
    extraction = StepSpec(
        name="raw/common-crawl-focus-2026-22",
        override_output_path=_FOCUS_CRAWL_EXTRACTION,
        fn=_validate_focus_crawl_extraction,
        hash_attrs={
            "crawl": "CC-SUPPLEMENTAL-2026-22",
            "justext_repository": "https://github.com/XenonMolecule/jusText",
            "justext_commit": "1652a1497b36c4b9941c609ffa1714eeefedc70b",
            "justext_model": "sklearn",
            "justext_stoplist": "English",
            "format": "parquet",
            "shards": _EXTRACTION_SHARD_COUNT,
            "documents": 36_327_068,
        },
    )
    normalized = normalize_step(
        name="normalized/common-crawl-focus-2026-22",
        download=extraction,
        relative_input_path=_EXTRACTION_SHARDS,
        # The extractor already keys rows by xxh3_128 of the text, so normalize
        # regenerates the same id and keeps the WARC record UUID as source_id.
        id_field="source_id",
        file_extensions=(".parquet",),
    )
    return extraction, normalized
