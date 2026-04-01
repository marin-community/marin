# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Common Pile dataset definitions for the pretraining dataset CLI."""

from functools import partial

from experiments.common_pile.filter_stackv2_code import stackv2_code_filtered
from experiments.common_pile.stitch_bhl_books import bhl_full_books
from experiments.common_pile.tokenize_common_pile import (
    data_provenance_initiative_filtered,
    foodista_filtered,
    github_archive_filtered,
    library_of_congress_filtered,
    libretexts_filtered,
    news_filtered,
    oercommons_filtered,
    pre_1929_books_filtered,
    pressbooks_filtered,
    project_gutenberg_filtered,
    regulations_filtered,
    stackexchange_filtered,
    ubuntu_irc_filtered,
    usgpo_filtered,
    uspto_filtered,
    wikiteam_filtered,
    youtube_filtered,
)
from experiments.defaults import default_tokenize
from experiments.marin_models import marin_tokenizer
from fray.cluster import ResourceConfig
from marin.datakit.download.common_pile import COMMON_PILE_DATASETS, download_common_pile_step

_R20 = ResourceConfig(ram="20g", disk="10g")
_R40 = ResourceConfig(ram="40g", disk="10g")

tokenize_cp = partial(default_tokenize, tokenizer=marin_tokenizer)

cp_downloads = {
    **{name: download_common_pile_step(name).as_executor_step() for name in COMMON_PILE_DATASETS},
    "wikiteam": wikiteam_filtered,
    "pre_1929_books": pre_1929_books_filtered,
    "ubuntu_irc": ubuntu_irc_filtered,
    "regulations": regulations_filtered,
    "project_gutenberg": project_gutenberg_filtered,
    "data_provenance": data_provenance_initiative_filtered,
    "youtube": youtube_filtered,
    "biodiversity": bhl_full_books,
    "library_of_congress": library_of_congress_filtered,
    "usgpo": usgpo_filtered,
    "pressbooks": pressbooks_filtered,
    "libretexts": libretexts_filtered,
    "news": news_filtered,
    "foodista": foodista_filtered,
    "oercommons": oercommons_filtered,
    "uspto": uspto_filtered,
    "stackexchange": stackexchange_filtered,
    "github_archive": github_archive_filtered,
    "stackv2_code": stackv2_code_filtered,
}


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
    "cp/wikiteam": tokenize_cp("common_pile/wikiteam", wikiteam_filtered / "**/*.json*"),
    "cp/pre_1929_books": tokenize_cp(
        "common_pile/pre_1929_books", pre_1929_books_filtered / "**/*.json*", worker_resources=_R40
    ),
    "cp/ubuntu_irc": tokenize_cp("common_pile/ubuntu_irc", ubuntu_irc_filtered / "**/*.json*", worker_resources=_R20),
    "cp/regulations": tokenize_cp("common_pile/regulations", regulations_filtered / "**/*.json*", worker_resources=_R40),
    "cp/project_gutenberg": tokenize_cp(
        "common_pile/project_gutenberg", project_gutenberg_filtered / "**/*.json*", worker_resources=_R40
    ),
    "cp/data_provenance": tokenize_cp(
        "common_pile/data_provenance_initiative", data_provenance_initiative_filtered / "**/*.json*"
    ),
    "cp/youtube": tokenize_cp("common_pile/youtube", youtube_filtered / "**/*.json*"),
    "cp/biodiversity": tokenize_cp(
        "common_pile/biodiversity_heritage_library_books", bhl_full_books / "**/*.jsonl.gz", worker_resources=_R20
    ),
    "cp/library_of_congress": tokenize_cp(
        "common_pile/library_of_congress", library_of_congress_filtered / "**/*.json*", worker_resources=_R40
    ),
    "cp/usgpo": tokenize_cp("common_pile/usgpo", usgpo_filtered / "**/*.json*", worker_resources=_R40),
    "cp/pressbooks": tokenize_cp("common_pile/pressbooks", pressbooks_filtered / "**/*.json*"),
    "cp/libretexts": tokenize_cp("common_pile/libretexts", libretexts_filtered / "**/*.json*"),
    "cp/news": tokenize_cp("common_pile/news", news_filtered / "**/*.json*"),
    "cp/foodista": tokenize_cp("common_pile/foodista", foodista_filtered / "**/*.json*"),
    "cp/oercommons": tokenize_cp("common_pile/oercommons", oercommons_filtered / "**/*.json*"),
    "cp/uspto": tokenize_cp("common_pile/uspto", uspto_filtered / "**/*.json*", worker_resources=_R20),
    "cp/stackexchange": tokenize_cp("common_pile/stackexchange", stackexchange_filtered / "**/*.json*"),
    "cp/github_archive": tokenize_cp("common_pile/github_archive", github_archive_filtered / "**/*.json*"),
    "cp/stackv2_code": tokenize_cp(
        "common_pile/stackv2_code", stackv2_code_filtered / "**/*.jsonl.gz", worker_resources=_R20
    ),
}
