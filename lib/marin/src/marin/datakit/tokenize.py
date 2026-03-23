# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit tokenize stage — convert normalized Parquet datasets into Levanter cache format.

This is the final stage of the datakit pipeline. It reads normalized Parquet
files and produces tokenized training data in Levanter's TreeStore format.

Tokenization is the boundary where per-document structure ends. The tokenizer
concatenates documents into fixed-size token sequences for efficient training.
"""

import logging

from marin.execution.step_spec import StepSpec
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize

logger = logging.getLogger(__name__)


def tokenize_step(
    name: str,
    *,
    input_path: str,
    tokenizer: str,
    max_workers: int = 4096,
    deps: list[StepSpec] | None = None,
    output_path_prefix: str | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a StepSpec that tokenizes a normalized dataset.

    Reads normalized Parquet files and produces Levanter cache format output
    suitable for training.

    Args:
        name: Step name (e.g. "fineweb/tokenize").
        input_path: Path to normalized Parquet files (output of normalize step).
        tokenizer: HuggingFace tokenizer name (e.g. "meta-llama/Llama-3.1-8B").
        max_workers: Maximum Zephyr worker parallelism.
        deps: Upstream dependencies (typically the normalize or consolidate step).
        output_path_prefix: Override the default output path prefix.
        override_output_path: Override the computed output path entirely.

    Returns:
        A StepSpec whose output_path contains the tokenized Levanter cache.
    """

    def _run(output_path: str) -> None:
        tokenize(
            TokenizeConfig(
                train_paths=[input_path],
                validation_paths=[],
                cache_path=output_path,
                tokenizer=tokenizer,
                max_workers=max_workers,
                allow_test_in_train=True,
            )
        )

    return StepSpec(
        name=name,
        fn=_run,
        deps=deps or [],
        hash_attrs={
            "input_path": input_path,
            "tokenizer": tokenizer,
        },
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )
