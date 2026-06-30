# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Common Pile v0.1 filtered splits as lazy ``Dataset`` handles (the dataset catalog).

:func:`common_pile_slice` builds one split's tokenized handle from its name, pinned download
revision, and (optionally) an existing llama3 cache to reuse — call it directly for a single
split, or :func:`common_pile_datasets` for the whole catalog. Each slice tokenizes from a pinned
``download_hf`` of the filtered HuggingFace split, reusing the existing
``raw/common_pile/<name>_filtered-<revision>`` download instead of re-fetching it. This is the
catalog only — handles plus the published mixture weights; assembling a mixture from
``{handle: weight}`` is the experiment's job (via :func:`marin.experiment.data.mixture`).
"""

from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.llama import llama3_tokenizer

_RAW_PREFIX = "raw/common_pile"

# Tokenizable shard extensions. The download holds the whole HF repo tree, so each split
# is globbed recursively for every extension the tokenizer reads (json/jsonl + parquet);
# brace expansion fans this single glob out to one pattern per extension.
_TOKENIZE_GLOB = "**/*.{json.gz,json.zst,json.zstd,jsonl.gz,jsonl.zst,jsonl.zstd,parquet}"

# CPU tokenize step; per-shard zephyr workers use the tokenize config's own defaults.
_TOKENIZE_RESOURCES = ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g")


def _common_pile_download(name: str, revision: str) -> ArtifactStep[TokenizedCache]:
    """Pinned raw download of the filtered split ``common-pile/<name>_filtered`` at ``revision``."""
    basename = f"{name}_filtered"
    return hf_download(
        f"{_RAW_PREFIX}/{basename}",
        hf_id=f"common-pile/{basename}",
        revision=revision,
        pin=f"{_RAW_PREFIX}/{basename}-{revision}",
        version="2026.06.28",
    )


def common_pile_slice(
    name: str, revision: str, pin: str | None = None, *, tokenizer: str = llama3_tokenizer
) -> ArtifactStep[TokenizedCache]:
    """One Common Pile split as a tokenized handle, keyed ``common_pile/<name>``.

    Downloads the filtered HuggingFace split ``common-pile/<name>_filtered`` at ``revision`` (reusing
    the existing raw download) and tokenizes it. With the llama3 tokenizer and ``pin`` set, resolves
    to that marin-executor cache instead of re-tokenizing; any other tokenizer tokenizes fresh.
    """
    return tokenized(
        f"common_pile/{name}",
        tokenizer=tokenizer,
        raw=_common_pile_download(name, revision),
        glob=_TOKENIZE_GLOB,
        resources=_TOKENIZE_RESOURCES,
        pin=pin if tokenizer == llama3_tokenizer else None,
        version="2026.06.28",
    )


# split -> (pinned download revision, existing llama3 tokenized cache). The pin reuses the
# marin-executor llama3 cache so a llama3 run resolves to it instead of re-tokenizing these
# multi-billion-token corpora.
COMMON_PILE_SLICES: dict[str, tuple[str, str]] = {
    "arxiv_abstracts": ("f1d7a9a", "tokenized/common_pile/arxiv_abstracts-fa99b2"),
    "arxiv_papers": ("033cf7f", "tokenized/common_pile/arxiv_papers-75f8c0"),
    "biodiversity_heritage_library": ("0486ed6", "tokenized/common_pile/biodiversity_heritage_library-c141ed"),
    "caselaw_access_project": ("50e1961", "tokenized/common_pile/caselaw_access_project-ba2bc9"),
    "cccc": ("03a3de5", "tokenized/common_pile/cccc-fd5797"),
    "data_provenance_initiative": ("8f5afcf", "tokenized/common_pile/data_provenance_initiative-f0f8e6"),
    "doab": ("defb24c", "tokenized/common_pile/doab-cab67a"),
    "foodista": ("bf2c7aa", "tokenized/common_pile/foodista-904225"),
    "github_archive": ("52282fe", "tokenized/common_pile/github_archive-ed0971"),
    "library_of_congress": ("56725c7", "tokenized/common_pile/library_of_congress-8cd324"),
    "libretexts": ("70388bc", "tokenized/common_pile/libretexts-46297d"),
    "news": ("59aaa8f", "tokenized/common_pile/news-8f5d41"),
    "oercommons": ("506b615", "tokenized/common_pile/oercommons-728289"),
    "peS2o": ("2977475", "tokenized/common_pile/peS2o-2e6500"),
    "pre_1929_books": ("23f9d96", "tokenized/common_pile/pre_1929_books-c33f75"),
    "pressbooks": ("1a1d3b5", "tokenized/common_pile/pressbooks-6d36ee"),
    "project_gutenberg": ("3cdf687", "tokenized/common_pile/project_gutenberg-4ae24e"),
    "public_domain_review": ("efc7f21", "tokenized/common_pile/public_domain_review-e73382"),
    "pubmed": ("c156f05", "tokenized/common_pile/pubmed-7986e4"),
    "python_enhancement_proposals": ("5821709", "tokenized/common_pile/python_enhancement_proposals-cfa465"),
    "regulations": ("3327364", "tokenized/common_pile/regulations-9d6cae"),
    "stackexchange": ("c0ac737", "tokenized/common_pile/stackexchange-1ba844"),
    "stackv2_edu": ("c354dbe", "tokenized/common_pile/stackv2_edu-fdc0ad"),
    "stackv2_html": ("92c9fa8", "tokenized/common_pile/stackv2_html-2b653d"),
    "ubuntu_irc": ("84f88c9", "tokenized/common_pile/ubuntu_irc-ffc7af"),
    "uk_hansard": ("c88adc4", "tokenized/common_pile/uk_hansard-67e776"),
    "usgpo": ("b150cc2", "tokenized/common_pile/usgpo-86324c"),
    "uspto": ("13894c5", "tokenized/common_pile/uspto-674f8f"),
    "wikimedia": ("0641bb8", "tokenized/common_pile/wikimedia-53a667"),
    "wikiteam": ("f4ed055", "tokenized/common_pile/wikiteam-174e57"),
    "youtube": ("dff8c8a", "tokenized/common_pile/youtube-6fb6c3"),
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


def stackv2_edu_filtered_download() -> ArtifactStep[TokenizedCache]:
    """Raw download handle for the Common Pile stackv2_edu filtered split."""
    return _common_pile_download("stackv2_edu", COMMON_PILE_SLICES["stackv2_edu"][0])


def common_pile_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, ArtifactStep[TokenizedCache]]:
    """One tokenized :class:`Dataset` handle per Common Pile split, keyed ``common_pile/<name>``."""
    return {
        f"common_pile/{name}": common_pile_slice(name, revision, pin, tokenizer=tokenizer)
        for name, (revision, pin) in COMMON_PILE_SLICES.items()
    }
