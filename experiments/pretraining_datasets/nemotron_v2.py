# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Nemotron v2 pre-training dataset tokenization.

Download definitions live in marin.datakit.download.nemotron_v2.
This file wires them into normalize + tokenization steps for experiment
pipelines.
"""

import os.path

from marin.datakit.download.nemotron_v2 import (
    NEMOTRON_V2_DATASETS,
    download_nemotron_v2_step,
    normalize_nemotron_v2_step,
)
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

from experiments.llama import llama3_tokenizer

# ============================================================================
# RAW DATASET DOWNLOADS AND NORMALIZED OUTPUTS
# ============================================================================

# TODO (rav): remove these in favor of the datakit sources


def downloads() -> dict[str, ExecutorStep]:
    """Raw download step per Nemotron v2 family."""
    return {family: download_nemotron_v2_step(family).as_executor_step() for family in NEMOTRON_V2_DATASETS}


def normalized() -> dict[str, dict[str, ExecutorStep]]:
    """One normalize step per (family, subset) — normalize processes a single directory."""
    result: dict[str, dict[str, ExecutorStep]] = {}
    for family in NEMOTRON_V2_DATASETS:
        download = download_nemotron_v2_step(family)
        result[family] = {
            subset: normalize_nemotron_v2_step(download, family=family, subset=subset).as_executor_step()
            for subset in NEMOTRON_V2_DATASETS[family].subsets
        }
    return result


# ============================================================================
# TOKENIZATION
# ============================================================================


def tokenize_nemotron_v2_family(
    family: str,
    *,
    tokenizer: str | None = None,
) -> dict[str, TokenizerStep]:
    """Generate tokenization steps for all subsets of a Nemotron HF dataset family.

    Each subset has its own normalize step; tokenize reads from its
    ``outputs/main/`` directory.
    """
    if tokenizer is None:
        tokenizer = llama3_tokenizer

    info = NEMOTRON_V2_DATASETS[family]
    family_normalized = normalized()[family]

    steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for subset in info.subsets:
        output_name = os.path.join("tokenized", family, subset)
        normalized_step = family_normalized[subset]
        step = ExecutorStep(
            name=output_name,
            fn=tokenize,
            config=TokenizeConfig(
                train_paths=[normalized_step / "outputs/main/*.parquet"],
                validation_paths=versioned([]),
                cache_path=this_output_path(),
                tokenizer=versioned(tokenizer),
            ),
        )
        steps[f"{family}/{subset}"] = step

    return steps


def tokenize_all_nemotron_v2(*, tokenizer: str | None = None) -> dict[str, TokenizerStep]:
    """Generate tokenization steps for all Nemotron HF datasets."""
    all_steps: dict[str, TokenizerStep] = {}
    for family in NEMOTRON_V2_DATASETS:
        all_steps.update(tokenize_nemotron_v2_family(family, tokenizer=tokenizer))
    return all_steps
