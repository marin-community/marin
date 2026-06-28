# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Common Pile v0.1 filtered splits as lazy ``Dataset`` handles (the dataset catalog).

One handle per Common Pile split: each tokenizes from a pinned ``download_hf`` of the
filtered HuggingFace split, reusing the existing ``raw/common_pile/<name>-<revision>``
download instead of re-fetching it. This is the catalog only — handles plus the
published mixture weights; assembling a mixture from ``{handle: weight}`` is the
experiment's job (via :func:`marin.experiment.data.mixture`).
"""

from dataclasses import dataclass

from fray.types import ResourceConfig
from marin.execution.lazy import Dataset
from marin.experiment.data import hf_download, tokenized

from experiments.llama import llama3_tokenizer

_RAW_PREFIX = "raw/common_pile"

# Tokenizable shard extensions. The download holds the whole HF repo tree, so each split
# is globbed recursively for every extension the tokenizer reads (json/jsonl + parquet);
# brace expansion fans this single glob out to one pattern per extension.
_TOKENIZE_GLOB = "**/*.{json.gz,json.zst,json.zstd,jsonl.gz,jsonl.zst,jsonl.zstd,parquet}"

# CPU tokenize step; per-shard zephyr workers use the tokenize config's own defaults.
_TOKENIZE_RESOURCES = ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g")


@dataclass(frozen=True)
class CommonPileSource:
    """A Common Pile split's HuggingFace id and pinned download revision."""

    hf_id: str
    revision: str

    @property
    def _basename(self) -> str:
        return self.hf_id.split("/", 1)[1]

    @property
    def download_name(self) -> str:
        """The raw download step's name, ``raw/common_pile/<hf-basename>``."""
        return f"{_RAW_PREFIX}/{self._basename}"

    @property
    def raw_path(self) -> str:
        """The pinned download location, ``raw/common_pile/<hf-basename>-<revision>``."""
        return f"{_RAW_PREFIX}/{self._basename}-{self.revision}"


# HF id + pinned commit revision per Common Pile split, keyed by the short split name.
COMMON_PILE_SOURCES: dict[str, CommonPileSource] = {
    "arxiv_abstracts": CommonPileSource("common-pile/arxiv_abstracts_filtered", "f1d7a9a"),
    "arxiv_papers": CommonPileSource("common-pile/arxiv_papers_filtered", "033cf7f"),
    "biodiversity_heritage_library": CommonPileSource("common-pile/biodiversity_heritage_library_filtered", "0486ed6"),
    "caselaw_access_project": CommonPileSource("common-pile/caselaw_access_project_filtered", "50e1961"),
    "cccc": CommonPileSource("common-pile/cccc_filtered", "03a3de5"),
    "data_provenance_initiative": CommonPileSource("common-pile/data_provenance_initiative_filtered", "8f5afcf"),
    "doab": CommonPileSource("common-pile/doab_filtered", "defb24c"),
    "foodista": CommonPileSource("common-pile/foodista_filtered", "bf2c7aa"),
    "github_archive": CommonPileSource("common-pile/github_archive_filtered", "52282fe"),
    "library_of_congress": CommonPileSource("common-pile/library_of_congress_filtered", "56725c7"),
    "libretexts": CommonPileSource("common-pile/libretexts_filtered", "70388bc"),
    "news": CommonPileSource("common-pile/news_filtered", "59aaa8f"),
    "oercommons": CommonPileSource("common-pile/oercommons_filtered", "506b615"),
    "peS2o": CommonPileSource("common-pile/peS2o_filtered", "2977475"),
    "pre_1929_books": CommonPileSource("common-pile/pre_1929_books_filtered", "23f9d96"),
    "pressbooks": CommonPileSource("common-pile/pressbooks_filtered", "1a1d3b5"),
    "project_gutenberg": CommonPileSource("common-pile/project_gutenberg_filtered", "3cdf687"),
    "public_domain_review": CommonPileSource("common-pile/public_domain_review_filtered", "efc7f21"),
    "pubmed": CommonPileSource("common-pile/pubmed_filtered", "c156f05"),
    "python_enhancement_proposals": CommonPileSource("common-pile/python_enhancement_proposals_filtered", "5821709"),
    "regulations": CommonPileSource("common-pile/regulations_filtered", "3327364"),
    "stackexchange": CommonPileSource("common-pile/stackexchange_filtered", "c0ac737"),
    "stackv2_edu": CommonPileSource("common-pile/stackv2_edu_filtered", "c354dbe"),
    "stackv2_html": CommonPileSource("common-pile/stackv2_html_filtered", "92c9fa8"),
    "ubuntu_irc": CommonPileSource("common-pile/ubuntu_irc_filtered", "84f88c9"),
    "uk_hansard": CommonPileSource("common-pile/uk_hansard_filtered", "c88adc4"),
    "usgpo": CommonPileSource("common-pile/usgpo_filtered", "b150cc2"),
    "uspto": CommonPileSource("common-pile/uspto_filtered", "13894c5"),
    "wikimedia": CommonPileSource("common-pile/wikimedia_filtered", "0641bb8"),
    "wikiteam": CommonPileSource("common-pile/wikiteam_filtered", "f4ed055"),
    "youtube": CommonPileSource("common-pile/youtube_filtered", "dff8c8a"),
}

# Effective token counts for the main training stage (in teratokens).
# Weights from https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset under Main stage.
COMMA_MAIN_MIXTURE_WEIGHTS = {
    "common_pile/arxiv_abstracts": 0.00342,
    "common_pile/arxiv_papers": 0.036,
    "common_pile/biodiversity_heritage_library": 0.00245,
    "common_pile/caselaw_access_project": 0.0197,
    "common_pile/cccc": 0.0912,
    "common_pile/data_provenance_initiative": 0.00552,
    "common_pile/doab": 0.018,
    "common_pile/foodista": 0.00015,
    "common_pile/github_archive": 0.066,
    "common_pile/library_of_congress": 0.00237,
    "common_pile/libretexts": 0.00056,
    "common_pile/news": 0.00038,
    "common_pile/oercommons": 0.00007,
    "common_pile/peS2o": 0.2598,
    "common_pile/pre_1929_books": 0.0124,
    "common_pile/pressbooks": 0.00084,
    "common_pile/project_gutenberg": 0.0057,
    "common_pile/public_domain_review": 0.00001,
    "common_pile/pubmed": 0.0366,
    "common_pile/python_enhancement_proposals": 0.00002,
    "common_pile/regulations": 0.0084,
    "common_pile/stackexchange": 0.1434,
    "common_pile/stackv2_edu": 0.1356,
    "common_pile/stackv2_html": 0.0024,
    "common_pile/ubuntu_irc": 0.0114,
    "common_pile/uk_hansard": 0.0138,
    "common_pile/usgpo": 0.0022,
    "common_pile/uspto": 0.03935,
    "common_pile/wikimedia": 0.0948,
    "common_pile/wikiteam": 0.0172,
    "common_pile/youtube": 0.0047,
}

# Effective token counts for the cooldown stage (in teratokens).
# Weights from https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset under Cooldown stage.
COMMA_COOLDOWN_MIXTURE_WEIGHTS = {
    "common_pile/arxiv_papers": 0.003,
    "common_pile/cccc": 0.00456,
    "common_pile/data_provenance_initiative": 0.00184,
    "common_pile/doab": 0.006,
    "common_pile/foodista": 0.00005,
    "common_pile/libretexts": 0.00019,
    "common_pile/news": 0.00013,
    "common_pile/oercommons": 0.00002,
    "common_pile/peS2o": 0.00433,
    "common_pile/pressbooks": 0.00028,
    "common_pile/public_domain_review": 0.0,
    "common_pile/python_enhancement_proposals": 0.00001,
    "common_pile/stackexchange": 0.00597,
    "common_pile/stackv2_edu": 0.00678,
    "common_pile/wikimedia": 0.00632,
}


def _download_handle(source: CommonPileSource) -> Dataset:
    """A HuggingFace-download handle for ``source``, pinned to its existing raw download."""
    return hf_download(source.download_name, hf_id=source.hf_id, revision=source.revision, pin=source.raw_path)


def stackv2_edu_filtered_download() -> Dataset:
    """Raw download handle for the Common Pile stackv2_edu filtered split."""
    return _download_handle(COMMON_PILE_SOURCES["stackv2_edu"])


def common_pile_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, Dataset]:
    """One tokenized :class:`Dataset` handle per Common Pile split, keyed ``common_pile/<name>``.

    Each handle tokenizes from a pinned ``download_hf`` of the filtered HuggingFace split, so
    referencing it reuses the existing raw download rather than re-fetching the corpus.
    """
    return {
        f"common_pile/{name}": tokenized(
            f"common_pile/{name}",
            tokenizer=tokenizer,
            raw=_download_handle(source),
            glob=_TOKENIZE_GLOB,
            resources=_TOKENIZE_RESOURCES,
        )
        for name, source in COMMON_PILE_SOURCES.items()
    }
