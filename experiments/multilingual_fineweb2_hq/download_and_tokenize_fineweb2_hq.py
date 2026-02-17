# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tokenizes the Fineweb2-HQ dataset splits.

This module defines a function that returns tokenization steps for each dataset split available in the
Fineweb2 dataset.
"""

import os.path

import dataclasses

from experiments.defaults import default_download
from experiments.llama import llama3_tokenizer
from experiments.multilingual_fineweb2_hq.constants import FINEWEB2_DATASETS
from marin.execution.step_model import StepSpec
from marin.execution.step_runner import StepRunner
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

fineweb2_raw = dataclasses.replace(
    default_download(
        name="raw/fineweb2_hq",
        hf_dataset_id="epfml/FineWeb2-HQ",
        revision="c0c06e94fd3a44ae9e802b2b0fc533817601eb5e",
    ),
    override_output_path="raw/fineweb2-hq",
)


def _get_fineweb2_split_paths(split):
    patterns = FINEWEB2_DATASETS[split]
    fineweb2_split_paths = [os.path.join(fineweb2_raw.output_path, pattern) for pattern in patterns]
    return fineweb2_split_paths


def tokenize_fineweb2hq_steps(*, base_path="tokenized/", tokenizer=llama3_tokenizer) -> dict[str, TokenizerStep]:
    """Return a mapping from dataset key to tokenization step for Fineweb2-HQ.

    Keys follow the pattern "fineweb2_hq/<split>", aligning with mixture naming conventions
    in other datasets (e.g., "dolma/...", "nemotron_cc/...").
    """
    steps: dict[str, StepSpec] = {}
    for split in FINEWEB2_DATASETS.keys():
        fineweb2_split_output_path = os.path.join(base_path, "fineweb2_hq", split)
        fineweb2_split_paths = _get_fineweb2_split_paths(split)
        step = StepSpec(
            name=fineweb2_split_output_path,
            hash_attrs={
                "train_paths": fineweb2_split_paths,
                "validation_paths": [],
                "tokenizer": tokenizer,
            },
            deps=[fineweb2_raw],
            fn=lambda output_path, _paths=fineweb2_split_paths, _tok=tokenizer: tokenize(
                TokenizeConfig(
                    train_paths=_paths,
                    validation_paths=[],
                    cache_path=output_path,
                    tokenizer=_tok,
                )
            ),
        )
        steps[f"fineweb2_hq/{split}"] = step
    return steps


if __name__ == "__main__":
    StepRunner().run(list(tokenize_fineweb2hq_steps().values()))
