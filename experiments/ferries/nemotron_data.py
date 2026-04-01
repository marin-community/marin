# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run only the data-preparation steps upstream of the Nemotron canary ferry.

Tokenizes all Nemotron CC splits, the DCLM code/math components, and the
default validation sets — everything that NEMOTRON_MIX_WITH_DEFAULT_VALIDATION
depends on.
"""

import dataclasses

from experiments.defaults import default_validation_sets
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.pretraining_datasets.nemotron import nemotron_mix_block_shuffle, tokenize_nemotron
from marin.execution.executor import ExecutorStep, executor_main
from marin.processing.tokenize.tokenize import TokenizeConfigBase

COREWEAVE_CACHE_COPY_MAX_WORKERS = 12


def _with_cache_copy_max_workers(step: ExecutorStep, *, cache_copy_max_workers: int) -> ExecutorStep:
    config = step.config
    if not isinstance(config, TokenizeConfigBase):
        return step

    return dataclasses.replace(
        step,
        config=dataclasses.replace(config, cache_copy_max_workers=cache_copy_max_workers),
    )


def main() -> None:
    nemotron_steps = list(tokenize_nemotron().values())
    dclm_steps = [
        dclm_components_llama3["starcoderdata"],
        dclm_components_llama3["proofpile_2"],
    ]
    validation_steps = list(default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer).values())
    steps = nemotron_steps + dclm_steps + validation_steps

    executor_main(
        steps=[
            _with_cache_copy_max_workers(step, cache_copy_max_workers=COREWEAVE_CACHE_COPY_MAX_WORKERS) for step in steps
        ]
    )


if __name__ == "__main__":
    main()
