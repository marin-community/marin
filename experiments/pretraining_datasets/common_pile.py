# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Common Pile dataset definitions for the pretraining dataset CLI.

Uses two catalogs from marin.datakit.download.common_pile:
- COMMON_PILE_DATASETS (raw, unfiltered snapshots with URL globs) for subsets
  where we run our own tokenization on the raw data.
- COMMON_PILE_FILTERED_DATASETS (quality-filtered repos) for subsets where we
  tokenize the filtered output directly.

Two subsets are derived transforms rather than direct downloads:
- "biodiversity": BHL pages stitched into full books (stitch_bhl_books.py)
- "stackv2_code": stackv2 filtered to code extensions (filter_stackv2_code.py)
"""

from functools import partial

from experiments.common_pile.filter_stackv2_code import stackv2_code_filtered
from experiments.common_pile.stitch_bhl_books import bhl_full_books
from experiments.defaults import default_tokenize
from experiments.marin_models import marin_tokenizer
from fray.cluster import ResourceConfig
from marin.datakit.download.common_pile import (
    COMMON_PILE_DATASETS,
    download_common_pile_filtered_step,
    download_common_pile_step,
)

_R20 = ResourceConfig(ram="20g", disk="10g")
_R40 = ResourceConfig(ram="40g", disk="10g")

tokenize_cp = partial(default_tokenize, tokenizer=marin_tokenizer)


def _filtered(name: str):
    return download_common_pile_filtered_step(name).as_executor_step()


# Downloads keyed by short subset name.
# Raw downloads (from COMMON_PILE_DATASETS) for subsets with explicit URL globs:
cp_downloads = {
    **{name: download_common_pile_step(name).as_executor_step() for name in COMMON_PILE_DATASETS},
    # Filtered downloads for subsets tokenized directly from filtered output:
    "wikiteam": _filtered("wikiteam"),
    "pre_1929_books": _filtered("pre_1929_books"),
    "ubuntu_irc": _filtered("ubuntu_irc"),
    "regulations": _filtered("regulations"),
    "project_gutenberg": _filtered("project_gutenberg"),
    "data_provenance": _filtered("data_provenance_initiative"),
    "youtube": _filtered("youtube"),
    "library_of_congress": _filtered("library_of_congress"),
    "usgpo": _filtered("usgpo"),
    "pressbooks": _filtered("pressbooks"),
    "libretexts": _filtered("libretexts"),
    "news": _filtered("news"),
    "foodista": _filtered("foodista"),
    "oercommons": _filtered("oercommons"),
    "uspto": _filtered("uspto"),
    "stackexchange": _filtered("stackexchange"),
    "github_archive": _filtered("github_archive"),
    # Derived transform steps (depend on their own filtered downloads):
    "biodiversity": bhl_full_books,
    "stackv2_code": stackv2_code_filtered,
}

# Tokenization steps keyed as "cp/<subset>".
# Every entry references its download via cp_downloads to keep keys in sync.
cp_tokenized = {
    "cp/peS2o": tokenize_cp("common_pile/peS2o", cp_downloads["peS2o"] / "v0/documents/*.json.gz"),
    "cp/pubmed": tokenize_cp("common_pile/pubmed", cp_downloads["pubmed"] / "data/*.jsonl.gz", worker_resources=_R20),
    "cp/arxiv_papers": tokenize_cp(
        "common_pile/arxiv_papers", cp_downloads["arxiv_papers"] / "*.jsonl.gz", worker_resources=_R20
    ),
    "cp/arxiv_abstracts": tokenize_cp("common_pile/arxiv_abstracts", cp_downloads["arxiv_abstracts"] / "*.jsonl.gz"),
    "cp/caselaw": tokenize_cp(
        "common_pile/caselaw_access_project", cp_downloads["caselaw"] / "*.jsonl.gz", worker_resources=_R20
    ),
    "cp/doab": tokenize_cp("common_pile/doab", cp_downloads["doab"] / "v0/*.json.gz"),
    "cp/uk_hansard": tokenize_cp(
        "common_pile/uk_hansard", cp_downloads["uk_hansard"] / "uk_hansard/*.jsonl.gz", worker_resources=_R20
    ),
    "cp/peps": tokenize_cp(
        "common_pile/python_enhancement_proposals", cp_downloads["peps"] / "raw/documents/*.jsonl.gz"
    ),
    "cp/public_domain_review": tokenize_cp(
        "common_pile/public_domain_review", cp_downloads["public_domain_review"] / "v0/*.jsonl.gz"
    ),
    "cp/wikiteam": tokenize_cp("common_pile/wikiteam", cp_downloads["wikiteam"] / "**/*.json*"),
    "cp/pre_1929_books": tokenize_cp(
        "common_pile/pre_1929_books", cp_downloads["pre_1929_books"] / "**/*.json*", worker_resources=_R40
    ),
    "cp/ubuntu_irc": tokenize_cp(
        "common_pile/ubuntu_irc", cp_downloads["ubuntu_irc"] / "**/*.json*", worker_resources=_R20
    ),
    "cp/regulations": tokenize_cp(
        "common_pile/regulations", cp_downloads["regulations"] / "**/*.json*", worker_resources=_R40
    ),
    "cp/project_gutenberg": tokenize_cp(
        "common_pile/project_gutenberg", cp_downloads["project_gutenberg"] / "**/*.json*", worker_resources=_R40
    ),
    "cp/data_provenance": tokenize_cp(
        "common_pile/data_provenance_initiative", cp_downloads["data_provenance"] / "**/*.json*"
    ),
    "cp/youtube": tokenize_cp("common_pile/youtube", cp_downloads["youtube"] / "**/*.json*"),
    "cp/biodiversity": tokenize_cp(
        "common_pile/biodiversity_heritage_library_books",
        cp_downloads["biodiversity"] / "**/*.jsonl.gz",
        worker_resources=_R20,
    ),
    "cp/library_of_congress": tokenize_cp(
        "common_pile/library_of_congress", cp_downloads["library_of_congress"] / "**/*.json*", worker_resources=_R40
    ),
    "cp/usgpo": tokenize_cp("common_pile/usgpo", cp_downloads["usgpo"] / "**/*.json*", worker_resources=_R40),
    "cp/pressbooks": tokenize_cp("common_pile/pressbooks", cp_downloads["pressbooks"] / "**/*.json*"),
    "cp/libretexts": tokenize_cp("common_pile/libretexts", cp_downloads["libretexts"] / "**/*.json*"),
    "cp/news": tokenize_cp("common_pile/news", cp_downloads["news"] / "**/*.json*"),
    "cp/foodista": tokenize_cp("common_pile/foodista", cp_downloads["foodista"] / "**/*.json*"),
    "cp/oercommons": tokenize_cp("common_pile/oercommons", cp_downloads["oercommons"] / "**/*.json*"),
    "cp/uspto": tokenize_cp("common_pile/uspto", cp_downloads["uspto"] / "**/*.json*", worker_resources=_R20),
    "cp/stackexchange": tokenize_cp("common_pile/stackexchange", cp_downloads["stackexchange"] / "**/*.json*"),
    "cp/github_archive": tokenize_cp("common_pile/github_archive", cp_downloads["github_archive"] / "**/*.json*"),
    "cp/stackv2_code": tokenize_cp(
        "common_pile/stackv2_code", cp_downloads["stackv2_code"] / "**/*.jsonl.gz", worker_resources=_R20
    ),
}

_tokenized_keys = {name.removeprefix("cp/") for name in cp_tokenized}
assert cp_downloads.keys() == _tokenized_keys, (
    f"cp_downloads and cp_tokenized keys out of sync: "
    f"downloads only: {cp_downloads.keys() - _tokenized_keys}, "
    f"tokenized only: {_tokenized_keys - cp_downloads.keys()}"
)
