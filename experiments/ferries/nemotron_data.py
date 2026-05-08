# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run only the data-preparation steps upstream of the Nemotron canary ferry.

Runs v2 normalize for every Nemotron CC split, then tokenizes all Nemotron CC
splits, the DCLM code/math components, and the default validation sets —
everything that NEMOTRON_MIX_WITH_DEFAULT_VALIDATION depends on.
"""

import dataclasses

from marin.datakit.download.nemotron_v1 import (
    NEMOTRON_V1_SPLITS,
    download_nemotron_v1_step,
    normalize_nemotron_v1_step,
)
from marin.execution.executor import ExecutorStep, executor_main
from marin.processing.tokenize.tokenize import TokenizeConfigBase

from experiments.defaults import default_validation_sets
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.pretraining_datasets.nemotron import nemotron_mix, tokenize_nemotron

S3_PREFIX = "s3://marin-na/marin/tmp/rav"

MAX_WORKERS = 42
CACHE_COPY_MAX_WORKERS = 42


def _with_worker_caps(step: ExecutorStep) -> ExecutorStep:
    """Override max_workers and cache_copy_max_workers on a TokenizeConfig step."""
    config = step.config
    if not isinstance(config, TokenizeConfigBase):
        return step
    return dataclasses.replace(
        step,
        config=dataclasses.replace(
            config,
            max_workers=MAX_WORKERS,
            cache_copy_max_workers=CACHE_COPY_MAX_WORKERS,
        ),
    )


def main() -> None:
    nemotron_download = download_nemotron_v1_step()
    normalize_by_split = {
        split: (
            normalize_nemotron_v1_step(nemotron_download, split=split, max_workers=MAX_WORKERS)
            .as_executor_step()
            .with_output_path(f"{S3_PREFIX}/normalized/nemotron_v1/{split}")
        )
        for split in NEMOTRON_V1_SPLITS
    }
    normalize_steps = list(normalize_by_split.values())
    # Tokenize splits read the normalized parquet output instead of the raw
    # jsonl.zst dump under data-jsonl/.
    input_paths_by_split = {split: [step / "outputs/main/*.parquet"] for split, step in normalize_by_split.items()}
    nemotron_steps = [
        step.with_output_path(f"{S3_PREFIX}/{step.name}")
        for step in tokenize_nemotron(
            max_workers=MAX_WORKERS,
            cache_copy_max_workers=CACHE_COPY_MAX_WORKERS,
            input_paths_by_split=input_paths_by_split,
        ).values()
    ]
    dclm_steps = [
        _with_worker_caps(step).with_output_path(f"{S3_PREFIX}/{step.name}")
        for step in [dclm_components_llama3["starcoderdata"], dclm_components_llama3["proofpile_2"]]
    ]
    validation_steps = [
        _with_worker_caps(step).with_output_path(f"{S3_PREFIX}/{step.name}")
        for step in default_validation_sets(tokenizer=nemotron_mix.tokenizer).values()
    ]
    steps = normalize_steps + nemotron_steps + dclm_steps + validation_steps

    executor_main(steps=steps)


if __name__ == "__main__":
    main()
