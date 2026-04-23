# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""common-pile/* download + normalize helpers.

27 filtered subsets of the common-pile v0.1 corpora. Each entry is a standalone
HF repo (``common-pile/<subset>_filtered``, or ``common-pile/stackv2`` as the
one exception); no shared family download, no custom preprocessing beyond what
normalize does. Thin wrapper over :func:`hf_normalize_steps`.

The download HF ids end in ``_filtered`` because Marin re-publishes curated
variants of the canonical common-pile repos; the token-count-viewer's CSV
shows the user-facing unfiltered names, but what actually downloads (and what
the staged dirs on GCS hold) is the ``_filtered`` version.
"""

from __future__ import annotations

from marin.datakit.download.hf_simple_util import hf_normalize_steps
from marin.execution.step_spec import StepSpec

# (marin_name, hf_dataset_id, revision, staged_path)
# Staged paths verified present under gs://marin-us-central1/raw/ as of the
# canonical registry sweep.
_COMMON_PILE_ENTRIES: tuple[tuple[str, str, str, str], ...] = (
    (
        "cp/arxiv_abstracts",
        "common-pile/arxiv_abstracts_filtered",
        "f1d7a9a",
        "raw/common_pile/arxiv_abstracts_filtered-f1d7a9a",
    ),
    ("cp/arxiv_papers", "common-pile/arxiv_papers_filtered", "033cf7f", "raw/common_pile/arxiv_papers_filtered-033cf7f"),
    (
        "cp/biodiversity",
        "common-pile/biodiversity_heritage_library_filtered",
        "0486ed6",
        "raw/common_pile/biodiversity_heritage_library_filtered-0486ed6",
    ),
    (
        "cp/caselaw",
        "common-pile/caselaw_access_project_filtered",
        "50e1961",
        "raw/common_pile/caselaw_access_project_filtered-50e1961",
    ),
    (
        "cp/data_provenance",
        "common-pile/data_provenance_initiative_filtered",
        "8f5afcf",
        "raw/common_pile/data_provenance_initiative_filtered-8f5afcf",
    ),
    ("cp/doab", "common-pile/doab_filtered", "defb24c", "raw/common_pile/doab_filtered-defb24c"),
    ("cp/foodista", "common-pile/foodista_filtered", "bf2c7aa", "raw/common_pile/foodista_filtered-bf2c7aa"),
    (
        "cp/github_archive",
        "common-pile/github_archive_filtered",
        "52282fe",
        "raw/common_pile/github_archive_filtered-52282fe",
    ),
    (
        "cp/library_of_congress",
        "common-pile/library_of_congress_filtered",
        "56725c7",
        "raw/common_pile/library_of_congress_filtered-56725c7",
    ),
    ("cp/libretexts", "common-pile/libretexts_filtered", "70388bc", "raw/common_pile/libretexts_filtered-70388bc"),
    ("cp/news", "common-pile/news_filtered", "59aaa8f", "raw/common_pile/news_filtered-59aaa8f"),
    ("cp/oercommons", "common-pile/oercommons_filtered", "506b615", "raw/common_pile/oercommons_filtered-506b615"),
    ("cp/peS2o", "common-pile/peS2o_filtered", "2977475", "raw/common_pile/peS2o_filtered-2977475"),
    (
        "cp/peps",
        "common-pile/python_enhancement_proposals_filtered",
        "5821709",
        "raw/common_pile/python_enhancement_proposals_filtered-5821709",
    ),
    (
        "cp/pre_1929_books",
        "common-pile/pre_1929_books_filtered",
        "23f9d96",
        "raw/common_pile/pre_1929_books_filtered-23f9d96",
    ),
    ("cp/pressbooks", "common-pile/pressbooks_filtered", "1a1d3b5", "raw/common_pile/pressbooks_filtered-1a1d3b5"),
    (
        "cp/project_gutenberg",
        "common-pile/project_gutenberg_filtered",
        "3cdf687",
        "raw/common_pile/project_gutenberg_filtered-3cdf687",
    ),
    (
        "cp/public_domain_review",
        "common-pile/public_domain_review_filtered",
        "efc7f21",
        "raw/common_pile/public_domain_review_filtered-efc7f21",
    ),
    ("cp/pubmed", "common-pile/pubmed_filtered", "c156f05", "raw/common_pile/pubmed_filtered-c156f05"),
    ("cp/regulations", "common-pile/regulations_filtered", "3327364", "raw/common_pile/regulations_filtered-3327364"),
    (
        "cp/stackexchange",
        "common-pile/stackexchange_filtered",
        "c0ac737",
        "raw/common_pile/stackexchange_filtered-c0ac737",
    ),
    # Note: stackv2 is unfiltered — the only exception in the common-pile set.
    ("cp/stackv2_code", "common-pile/stackv2", "d0e3266", "raw/common_pile/stackv2-d0e3266"),
    ("cp/ubuntu_irc", "common-pile/ubuntu_irc_filtered", "84f88c9", "raw/common_pile/ubuntu_irc_filtered-84f88c9"),
    ("cp/uk_hansard", "common-pile/uk_hansard_filtered", "c88adc4", "raw/common_pile/uk_hansard_filtered-c88adc4"),
    ("cp/usgpo", "common-pile/usgpo_filtered", "b150cc2", "raw/common_pile/usgpo_filtered-b150cc2"),
    ("cp/uspto", "common-pile/uspto_filtered", "13894c5", "raw/common_pile/uspto_filtered-13894c5"),
    ("cp/wikiteam", "common-pile/wikiteam_filtered", "f4ed055", "raw/common_pile/wikiteam_filtered-f4ed055"),
    ("cp/youtube", "common-pile/youtube_filtered", "dff8c8a", "raw/common_pile/youtube_filtered-dff8c8a"),
)


def common_pile_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return ``(download, normalize)`` chains for every common-pile entry.

    common-pile is published in the Dolma ``.json.gz`` format (JSON-lines,
    gzipped) rather than parquet, so we override the normalize input
    extension. The default ``text``/``id`` fields match Dolma's schema.
    """
    return {
        marin_name: hf_normalize_steps(
            marin_name=marin_name,
            hf_dataset_id=hf_id,
            revision=revision,
            staged_path=staged,
            file_extensions=(".json.gz",),
        )
        for marin_name, hf_id, revision, staged in _COMMON_PILE_ENTRIES
    }
