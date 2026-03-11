# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared data config helpers for Grug MoE perf experiments."""

import dataclasses

from levanter.data.text import LmDataConfig

from experiments.pretraining_datasets import nemotron_mix, nemotron_mix_block_shuffle


def qwen3_moe_perf_mix() -> LmDataConfig:
    """Return the standard Nemotron mixture using step-result-backed tokenizer outputs.

    Perf launchers should follow the executor's step graph instead of hardcoding region-specific
    cache buckets. That keeps dataset resolution aligned with the current run environment and
    lets the executor reuse any existing step outputs naturally.
    """

    return dataclasses.replace(
        nemotron_mix,
        shuffle_before_trainval_split=False,
    )


def qwen3_moe_perf_mix_block_shuffle() -> LmDataConfig:
    """Return the standard Nemotron perf mix with block shuffle enabled."""

    return dataclasses.replace(
        nemotron_mix_block_shuffle,
        shuffle_before_trainval_split=False,
    )
