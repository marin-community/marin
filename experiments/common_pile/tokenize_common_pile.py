# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenization and mixture configs for the Common Pile v0.1 dataset."""

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from marin.datakit.download.common_pile import (
    COMMON_PILE_FILTERED_DATASETS,
    download_common_pile_filtered_step,
)
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import TokenizerStep, lm_mixture_data_config

# Common Pile v0.1 filtered dataset download steps — sourced from datakit
_filtered_steps = {
    name: download_common_pile_filtered_step(name).as_executor_step() for name in COMMON_PILE_FILTERED_DATASETS
}

# Map filtered dataset names to their corresponding download steps.
# Used by common_pile_tokenized() below for the COMMA llama3 mixture.
COMMON_PILE_TOKENIZED: dict[str, TokenizerStep] = _filtered_steps

# Effective token counts for the main training stage (in teratokens)
# Weights pulled from https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset under Main stage
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

# Effective token counts for the cooldown stage (in teratokens)
# Weights pulled from https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset under Cooldown stage
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


def common_pile_tokenized(*, tokenizer: str = llama3_tokenizer) -> dict[str, TokenizerStep]:
    """Return tokenization steps for the Common Pile filtered datasets."""
    tokenized: dict[str, TokenizerStep] = {}
    for dataset, step in COMMON_PILE_TOKENIZED.items():
        tokenized[f"common_pile/{dataset}"] = default_tokenize(
            name=f"common_pile/{dataset}",
            dataset=step,
            tokenizer=tokenizer,
        )
    return tokenized


def comma_main_mixture(*, tokenizer: str = llama3_tokenizer):
    """LmMixtureDatasetConfig for the main training stage."""
    tokenized = common_pile_tokenized(tokenizer=tokenizer)
    components = {f"common_pile/{dataset}": tokenized[f"common_pile/{dataset}"] for dataset in COMMON_PILE_TOKENIZED}
    return lm_mixture_data_config(
        components=components,
        weights=COMMA_MAIN_MIXTURE_WEIGHTS,
    )


def comma_cooldown_mixture(*, tokenizer: str = llama3_tokenizer):
    """LmMixtureDatasetConfig for the cooldown stage."""
    tokenized = common_pile_tokenized(tokenizer=tokenizer)
    components = {f"common_pile/{dataset}": tokenized[f"common_pile/{dataset}"] for dataset in COMMON_PILE_TOKENIZED}
    return lm_mixture_data_config(
        components=components,
        weights=COMMA_COOLDOWN_MIXTURE_WEIGHTS,
    )


if __name__ == "__main__":
    steps = list(common_pile_tokenized().values())
    executor_main(steps=steps)
