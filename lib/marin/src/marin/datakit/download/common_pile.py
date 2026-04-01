# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Common Pile download definitions for HuggingFace-hosted subsets.

Two catalogs live here:

- COMMON_PILE_DATASETS (raw): Unfiltered source snapshots with explicit URL globs.
  Use these when you need the original data for a new tokenization/processing
  pipeline (e.g. experiments/pretraining_datasets/common_pile.py).

- COMMON_PILE_FILTERED_DATASETS (filtered): Quality-filtered repos published by
  common-pile (e.g. "common-pile/pubmed_filtered"). Use these for the legacy
  COMMA mixture (experiments/common_pile/tokenize_common_pile.py) or when a
  subset has no raw download and must be tokenized from filtered output.

Keys overlap between the two catalogs (e.g. "pubmed" appears in both) but point
to different HF repos and revisions.
"""

from dataclasses import dataclass

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec


@dataclass(frozen=True)
class CommonPileDataset:
    hf_dataset_id: str
    revision: str
    hf_urls_glob: str
    output_name: str


COMMON_PILE_DATASETS: dict[str, CommonPileDataset] = {
    "peS2o": CommonPileDataset("common-pile/peS2o", "2caeba1", "v0/documents/*.json.gz", "peS2o"),
    "pubmed": CommonPileDataset("common-pile/pubmed", "648b8cf", "data/*.jsonl.gz", "pubmed"),
    "arxiv_papers": CommonPileDataset("common-pile/arxiv_papers", "963fe98", "*.jsonl.gz", "arxiv_papers"),
    "arxiv_abstracts": CommonPileDataset("common-pile/arxiv_abstracts", "828e35d", "*.jsonl.gz", "arxiv_abstracts"),
    "caselaw": CommonPileDataset(
        "common-pile/caselaw_access_project",
        "3c2cb50",
        "*.jsonl.gz",
        "caselaw_access_project",
    ),
    "doab": CommonPileDataset("common-pile/doab", "89e7a35", "v0/*.json.gz", "doab"),
    "uk_hansard": CommonPileDataset("common-pile/uk_hansard", "05eeb43", "uk_hansard/*.jsonl.gz", "uk_hansard"),
    "peps": CommonPileDataset(
        "common-pile/python_enhancement_proposals",
        "f932757",
        "raw/documents/*.jsonl.gz",
        "python_enhancement_proposals",
    ),
    "public_domain_review": CommonPileDataset(
        "common-pile/public_domain_review",
        "e9c7669",
        "v0/*.jsonl.gz",
        "public_domain_review",
    ),
}


def download_common_pile_step(name: str) -> StepSpec:
    dataset = COMMON_PILE_DATASETS[name]
    return download_hf_step(
        f"raw/common_pile/{dataset.output_name}",
        hf_dataset_id=dataset.hf_dataset_id,
        revision=dataset.revision,
        hf_urls_glob=[dataset.hf_urls_glob],
    )


@dataclass(frozen=True)
class CommonPileFilteredDataset:
    hf_dataset_id: str
    revision: str
    override_output_path: str


COMMON_PILE_FILTERED_DATASETS: dict[str, CommonPileFilteredDataset] = {
    "arxiv_abstracts": CommonPileFilteredDataset(
        "common-pile/arxiv_abstracts_filtered", "f1d7a9a", "raw/common_pile/arxiv_abstracts_filtered-f1d7a9a"
    ),
    "arxiv_papers": CommonPileFilteredDataset(
        "common-pile/arxiv_papers_filtered", "033cf7f", "raw/common_pile/arxiv_papers_filtered-033cf7f"
    ),
    "biodiversity_heritage_library": CommonPileFilteredDataset(
        "common-pile/biodiversity_heritage_library_filtered",
        "0486ed6",
        "raw/common_pile/biodiversity_heritage_library_filtered-0486ed6",
    ),
    "caselaw_access_project": CommonPileFilteredDataset(
        "common-pile/caselaw_access_project_filtered",
        "50e1961",
        "raw/common_pile/caselaw_access_project_filtered-50e1961",
    ),
    "cccc": CommonPileFilteredDataset("common-pile/cccc_filtered", "03a3de5", "raw/common_pile/cccc_filtered-03a3de5"),
    "data_provenance_initiative": CommonPileFilteredDataset(
        "common-pile/data_provenance_initiative_filtered",
        "8f5afcf",
        "raw/common_pile/data_provenance_initiative_filtered-8f5afcf",
    ),
    "doab": CommonPileFilteredDataset("common-pile/doab_filtered", "defb24c", "raw/common_pile/doab_filtered-defb24c"),
    "foodista": CommonPileFilteredDataset(
        "common-pile/foodista_filtered", "bf2c7aa", "raw/common_pile/foodista_filtered-bf2c7aa"
    ),
    "github_archive": CommonPileFilteredDataset(
        "common-pile/github_archive_filtered", "52282fe", "raw/common_pile/github_archive_filtered-52282fe"
    ),
    "library_of_congress": CommonPileFilteredDataset(
        "common-pile/library_of_congress_filtered", "56725c7", "raw/common_pile/library_of_congress_filtered-56725c7"
    ),
    "libretexts": CommonPileFilteredDataset(
        "common-pile/libretexts_filtered", "70388bc", "raw/common_pile/libretexts_filtered-70388bc"
    ),
    "news": CommonPileFilteredDataset("common-pile/news_filtered", "59aaa8f", "raw/common_pile/news_filtered-59aaa8f"),
    "oercommons": CommonPileFilteredDataset(
        "common-pile/oercommons_filtered", "506b615", "raw/common_pile/oercommons_filtered-506b615"
    ),
    "peS2o": CommonPileFilteredDataset(
        "common-pile/peS2o_filtered", "2977475", "raw/common_pile/peS2o_filtered-2977475"
    ),
    "pre_1929_books": CommonPileFilteredDataset(
        "common-pile/pre_1929_books_filtered", "23f9d96", "raw/common_pile/pre_1929_books_filtered-23f9d96"
    ),
    "pressbooks": CommonPileFilteredDataset(
        "common-pile/pressbooks_filtered", "1a1d3b5", "raw/common_pile/pressbooks_filtered-1a1d3b5"
    ),
    "project_gutenberg": CommonPileFilteredDataset(
        "common-pile/project_gutenberg_filtered", "3cdf687", "raw/common_pile/project_gutenberg_filtered-3cdf687"
    ),
    "public_domain_review": CommonPileFilteredDataset(
        "common-pile/public_domain_review_filtered", "efc7f21", "raw/common_pile/public_domain_review_filtered-efc7f21"
    ),
    "pubmed": CommonPileFilteredDataset(
        "common-pile/pubmed_filtered", "c156f05", "raw/common_pile/pubmed_filtered-c156f05"
    ),
    "python_enhancement_proposals": CommonPileFilteredDataset(
        "common-pile/python_enhancement_proposals_filtered",
        "5821709",
        "raw/common_pile/python_enhancement_proposals_filtered-5821709",
    ),
    "regulations": CommonPileFilteredDataset(
        "common-pile/regulations_filtered", "3327364", "raw/common_pile/regulations_filtered-3327364"
    ),
    "stackexchange": CommonPileFilteredDataset(
        "common-pile/stackexchange_filtered", "c0ac737", "raw/common_pile/stackexchange_filtered-c0ac737"
    ),
    "stackv2_edu": CommonPileFilteredDataset(
        "common-pile/stackv2_edu_filtered", "c354dbe", "raw/common_pile/stackv2_edu_filtered-c354dbe"
    ),
    "stackv2_html": CommonPileFilteredDataset(
        "common-pile/stackv2_html_filtered", "92c9fa8", "raw/common_pile/stackv2_html_filtered-92c9fa8"
    ),
    "stackv2": CommonPileFilteredDataset("common-pile/stackv2", "d0e3266", "raw/common_pile/stackv2-d0e3266"),
    "ubuntu_irc": CommonPileFilteredDataset(
        "common-pile/ubuntu_irc_filtered", "84f88c9", "raw/common_pile/ubuntu_irc_filtered-84f88c9"
    ),
    "uk_hansard": CommonPileFilteredDataset(
        "common-pile/uk_hansard_filtered", "c88adc4", "raw/common_pile/uk_hansard_filtered-c88adc4"
    ),
    "usgpo": CommonPileFilteredDataset(
        "common-pile/usgpo_filtered", "b150cc2", "raw/common_pile/usgpo_filtered-b150cc2"
    ),
    "uspto": CommonPileFilteredDataset(
        "common-pile/uspto_filtered", "13894c5", "raw/common_pile/uspto_filtered-13894c5"
    ),
    "wikimedia": CommonPileFilteredDataset(
        "common-pile/wikimedia_filtered", "0641bb8", "raw/common_pile/wikimedia_filtered-0641bb8"
    ),
    "wikiteam": CommonPileFilteredDataset(
        "common-pile/wikiteam_filtered", "f4ed055", "raw/common_pile/wikiteam_filtered-f4ed055"
    ),
    "youtube": CommonPileFilteredDataset(
        "common-pile/youtube_filtered", "dff8c8a", "raw/common_pile/youtube_filtered-dff8c8a"
    ),
}


def download_common_pile_filtered_step(name: str) -> StepSpec:
    dataset = COMMON_PILE_FILTERED_DATASETS[name]
    # Derive step name from the HF dataset ID (e.g. "common-pile/stackv2" -> "stackv2")
    step_name = dataset.hf_dataset_id.split("/")[-1]
    return download_hf_step(
        f"raw/common_pile/{step_name}",
        hf_dataset_id=dataset.hf_dataset_id,
        revision=dataset.revision,
        override_output_path=dataset.override_output_path,
    )
