# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw Hero data views for bounded pipeline trials."""

import dataclasses

from levanter.data.text.datasets import LmDataConfig
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import harrier_mix_2026_08_18_data_config


def raw_hero_data_config(
    *, ctx: StepContext, schedule_steps: int, batch_size: int, max_seq_len: int, experiment_flops: float
) -> LmDataConfig:
    """Use the Hero mixture schedule without shrinking its component datasets.

    A diagnostic execution limit must not simulate an entire training budget:
    shrinking rare components for a few steps can leave empty dataset views.
    """
    config = harrier_mix_2026_08_18_data_config(
        ctx=ctx,
        total_steps=schedule_steps,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        experiment_flops=experiment_flops,
        validation=(),
    )
    return dataclasses.replace(config, target_budget=None, experiment_budget=None)
