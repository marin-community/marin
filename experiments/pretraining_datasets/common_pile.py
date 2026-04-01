# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Common Pile dataset definitions for the pretraining dataset CLI."""

from levanter.data.text import TextLmDatasetFormat

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
from experiments.marin_models import marin_tokenizer
from fray.cluster import ResourceConfig
from marin.datakit.download.huggingface import download_hf_step
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize


def _tokenize_step(
    name: str,
    train_paths: list,
    *,
    worker_ram: str = "10g",
    text_key: str = "text",
) -> ExecutorStep:
    kwargs = {}
    if worker_ram != "10g":
        kwargs["worker_resources"] = ResourceConfig(ram=worker_ram, disk="10g")
    return ExecutorStep(
        name=f"tokenized/{name}",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=train_paths,
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(marin_tokenizer),
            format=TextLmDatasetFormat(text_key=text_key),
            **kwargs,
        ),
    )


def _raw_download(name: str, hf_dataset_id: str, revision: str, glob: str) -> ExecutorStep:
    return download_hf_step(
        f"raw/common_pile/{name}",
        hf_dataset_id=hf_dataset_id,
        revision=revision,
        hf_urls_glob=[glob],
    ).as_executor_step()


cp_downloads = {
    "peS2o": _raw_download("peS2o", "common-pile/peS2o", "2caeba1", "v0/documents/*.json.gz"),
    "pubmed": _raw_download("pubmed", "common-pile/pubmed", "648b8cf", "data/*.jsonl.gz"),
    "arxiv_papers": _raw_download("arxiv_papers", "common-pile/arxiv_papers", "963fe98", "*.jsonl.gz"),
    "arxiv_abstracts": _raw_download("arxiv_abstracts", "common-pile/arxiv_abstracts", "828e35d", "*.jsonl.gz"),
    "caselaw": _raw_download("caselaw_access_project", "common-pile/caselaw_access_project", "3c2cb50", "*.jsonl.gz"),
    "doab": _raw_download("doab", "common-pile/doab", "89e7a35", "v0/*.json.gz"),
    "uk_hansard": _raw_download("uk_hansard", "common-pile/uk_hansard", "05eeb43", "uk_hansard/*.jsonl.gz"),
    "peps": _raw_download(
        "python_enhancement_proposals",
        "common-pile/python_enhancement_proposals",
        "f932757",
        "raw/documents/*.jsonl.gz",
    ),
    "public_domain_review": _raw_download(
        "public_domain_review",
        "common-pile/public_domain_review",
        "e9c7669",
        "v0/*.jsonl.gz",
    ),
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
    "cp/peS2o": _tokenize_step("common_pile/peS2o", [cp_downloads["peS2o"] / "v0/documents/*.json.gz"]),
    "cp/pubmed": _tokenize_step("common_pile/pubmed", [cp_downloads["pubmed"] / "data/*.jsonl.gz"], worker_ram="20g"),
    "cp/arxiv_papers": _tokenize_step(
        "common_pile/arxiv_papers",
        [cp_downloads["arxiv_papers"] / "*.jsonl.gz"],
        worker_ram="20g",
    ),
    "cp/arxiv_abstracts": _tokenize_step(
        "common_pile/arxiv_abstracts",
        [cp_downloads["arxiv_abstracts"] / "*.jsonl.gz"],
    ),
    "cp/caselaw": _tokenize_step(
        "common_pile/caselaw_access_project",
        [cp_downloads["caselaw"] / "*.jsonl.gz"],
        worker_ram="20g",
    ),
    "cp/doab": _tokenize_step("common_pile/doab", [cp_downloads["doab"] / "v0/*.json.gz"]),
    "cp/uk_hansard": _tokenize_step(
        "common_pile/uk_hansard",
        [cp_downloads["uk_hansard"] / "uk_hansard/*.jsonl.gz"],
        worker_ram="20g",
    ),
    "cp/peps": _tokenize_step(
        "common_pile/python_enhancement_proposals",
        [cp_downloads["peps"] / "raw/documents/*.jsonl.gz"],
    ),
    "cp/public_domain_review": _tokenize_step(
        "common_pile/public_domain_review",
        [cp_downloads["public_domain_review"] / "v0/*.jsonl.gz"],
    ),
    "cp/wikiteam": _tokenize_step("common_pile/wikiteam", [wikiteam_filtered / "**/*.json*"]),
    "cp/pre_1929_books": _tokenize_step(
        "common_pile/pre_1929_books",
        [pre_1929_books_filtered / "**/*.json*"],
        worker_ram="40g",
    ),
    "cp/ubuntu_irc": _tokenize_step("common_pile/ubuntu_irc", [ubuntu_irc_filtered / "**/*.json*"], worker_ram="20g"),
    "cp/regulations": _tokenize_step(
        "common_pile/regulations",
        [regulations_filtered / "**/*.json*"],
        worker_ram="40g",
    ),
    "cp/project_gutenberg": _tokenize_step(
        "common_pile/project_gutenberg",
        [project_gutenberg_filtered / "**/*.json*"],
        worker_ram="40g",
    ),
    "cp/data_provenance": _tokenize_step(
        "common_pile/data_provenance_initiative",
        [data_provenance_initiative_filtered / "**/*.json*"],
    ),
    "cp/youtube": _tokenize_step("common_pile/youtube", [youtube_filtered / "**/*.json*"]),
    "cp/biodiversity": _tokenize_step(
        "common_pile/biodiversity_heritage_library_books",
        [bhl_full_books / "**/*.jsonl.gz"],
        worker_ram="20g",
    ),
    "cp/library_of_congress": _tokenize_step(
        "common_pile/library_of_congress",
        [library_of_congress_filtered / "**/*.json*"],
        worker_ram="40g",
    ),
    "cp/usgpo": _tokenize_step("common_pile/usgpo", [usgpo_filtered / "**/*.json*"], worker_ram="40g"),
    "cp/pressbooks": _tokenize_step("common_pile/pressbooks", [pressbooks_filtered / "**/*.json*"]),
    "cp/libretexts": _tokenize_step("common_pile/libretexts", [libretexts_filtered / "**/*.json*"]),
    "cp/news": _tokenize_step("common_pile/news", [news_filtered / "**/*.json*"]),
    "cp/foodista": _tokenize_step("common_pile/foodista", [foodista_filtered / "**/*.json*"]),
    "cp/oercommons": _tokenize_step("common_pile/oercommons", [oercommons_filtered / "**/*.json*"]),
    "cp/uspto": _tokenize_step("common_pile/uspto", [uspto_filtered / "**/*.json*"], worker_ram="20g"),
    "cp/stackexchange": _tokenize_step("common_pile/stackexchange", [stackexchange_filtered / "**/*.json*"]),
    "cp/github_archive": _tokenize_step("common_pile/github_archive", [github_archive_filtered / "**/*.json*"]),
    "cp/stackv2_code": _tokenize_step(
        "common_pile/stackv2_code",
        [stackv2_code_filtered / "**/*.jsonl.gz"],
        worker_ram="20g",
    ),
}
