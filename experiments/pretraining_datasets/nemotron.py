# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NEMOTRON CC dataset definitions and tokenization."""

import dataclasses
import os.path

from fray.v2.types import ResourceConfig

from experiments.defaults import DEFAULT_NEW_RUN_DATA_SHUFFLE
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from marin.datakit.download.nemotron_v1 import download_nemotron_v1_step
from marin.execution.executor import ExecutorStep, InputName, this_output_path, versioned
from marin.execution.remote import remote
from marin.processing.tokenize import TokenizeConfig, lm_mixture_data_config, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

# Fray resources for running a single Nemotron tokenize split as a remote job.
# TODO (rav): debug why this needs 32g - probably levanter store consolidation
NEMOTRON_SPLIT_TOKENIZE_RESOURCES = ResourceConfig(ram="32g", cpu=2)


def nemotron_cc_download() -> ExecutorStep:
    return download_nemotron_v1_step().as_executor_step()


NEMOTRON_DATASETS = {
    "hq_actual": ["quality=high/kind=actual/**/*.jsonl.*"],
    "hq_synth": ["quality=high/kind=synthetic/**/*.jsonl.*"],
    "medium_high": ["quality=medium-high/**/*.jsonl.*"],
    "medium": ["quality=medium/**/*.jsonl.*"],
    "medium_low": ["quality=medium-low/**/*.jsonl.*"],
    "low_actual": ["quality=low/kind=actual/**/*.jsonl.*"],
    "low_synth": ["quality=low/kind=synthetic/**/*.jsonl.*"],
}

# Weights for each split based on their size in TiB
NEMOTRON_WEIGHTS = {
    "nemotron_cc/hq_actual": 0.91351,  # TiB
    "nemotron_cc/hq_synth": 2.72,  # TiB
    "nemotron_cc/medium_high": 0.82471,  # TiB
    "nemotron_cc/medium": 3.38,  # TiB
    "nemotron_cc/medium_low": 1.54,  # TiB
    "nemotron_cc/low_actual": 0.70123,  # TiB
    "nemotron_cc/low_synth": 0.62771,  # TiB
}

# NB: we changed how hashes were computed for this corpus and we'd like to avoid recomputing them
NEMOTRON_LLAMA3_OVERRIDES = {
    "hq_actual": "tokenized/nemotron_cc/hq_actual-5af4cc",
    "hq_synth": "tokenized/nemotron_cc/hq_synth-3525e2",
    "low_actual": "tokenized/nemotron_cc/low_actual-cb3f2c",
    "low_synth": "tokenized/nemotron_cc/low_synth-3c57b3",
    "medium": "tokenized/nemotron_cc/medium-d86506",
    "medium_high": "tokenized/nemotron_cc/medium_high-d21701",
    "medium_low": "tokenized/nemotron_cc/medium_low-0fdb07",
}


# Hardcoded path to the nemotron download output so that glob or download
# step changes don't alter the tokenize step's version hash.
_NEMOTRON_CC_DATA_PATH = InputName.hardcoded("raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl/")


def _get_nemotron_split_paths(split: str):
    """Helper to get file paths for a nemotron split."""
    return [_NEMOTRON_CC_DATA_PATH / pattern for pattern in NEMOTRON_DATASETS[split]]


def tokenize_nemotron(
    *,
    tokenizer: str | None = None,
    max_workers: int = 4096,
    cache_copy_max_workers: int = 128,
) -> dict[str, TokenizerStep]:
    """Generate tokenization steps for all Nemotron CC dataset splits.

    Each split's tokenize function is wrapped with ``@remote`` so it runs as
    its own Fray job (see ``NEMOTRON_SPLIT_TOKENIZE_RESOURCES``). This keeps the
    entrypoint pod lightweight and lets the tokenize+consolidate work survive
    entrypoint restarts.
    """
    if tokenizer is None:
        from experiments.llama import llama3_tokenizer

        tokenizer = llama3_tokenizer

    tokenize_fn = remote(tokenize, resources=NEMOTRON_SPLIT_TOKENIZE_RESOURCES)

    nemotron_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for split in NEMOTRON_DATASETS:
        nemotron_split_output_path = os.path.join("tokenized", "nemotron_cc", split)
        nemotron_split_paths = _get_nemotron_split_paths(split)
        step = ExecutorStep(
            name=nemotron_split_output_path,
            fn=tokenize_fn,
            config=TokenizeConfig(
                train_paths=nemotron_split_paths,
                validation_paths=versioned([]),
                cache_path=this_output_path(),
                tokenizer=versioned(tokenizer),
                max_workers=max_workers,
                cache_copy_max_workers=cache_copy_max_workers,
            ),
        )

        # Check if we need to use override path for llama3
        from experiments.llama import llama3_tokenizer as _llama3_tokenizer

        if tokenizer == _llama3_tokenizer and split in NEMOTRON_LLAMA3_OVERRIDES:
            step = step.with_output_path(NEMOTRON_LLAMA3_OVERRIDES[split])

        nemotron_steps[os.path.join("nemotron_cc", split)] = step

    assert nemotron_steps.keys() == NEMOTRON_WEIGHTS.keys()
    return nemotron_steps


nemotron_mix = lm_mixture_data_config(
    components={
        **tokenize_nemotron(),
        "starcoderdata": dclm_components_llama3["starcoderdata"],
        "proofpile_2": dclm_components_llama3["proofpile_2"],
    },
    weights={
        **NEMOTRON_WEIGHTS,
        "starcoderdata": 0.25,
        "proofpile_2": 0.055,
    },
)

nemotron_mix_block_shuffle = dataclasses.replace(nemotron_mix, shuffle=DEFAULT_NEW_RUN_DATA_SHUFFLE)


def tokenize_nemotron_subset(name: str, tokenizer: str | None = None) -> ExecutorStep[TokenizeConfig]:
    """Get a specific nemotron split tokenization step."""
    assert name in NEMOTRON_DATASETS, f"Split {name} not found in NEMOTRON_DATASETS"
    return tokenize_nemotron(tokenizer=tokenizer)[f"nemotron_cc/{name}"]
