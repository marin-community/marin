# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""State-preserving proposal transactions at hero step boundaries."""

from dataclasses import replace

import jax

from experiments.grug.fast_track.model import Transformer
from experiments.grug.fast_track.train import GrugTrainState


def snapshot_state(state: GrugTrainState) -> GrugTrainState:
    """Copy buffers so future donated optimizer steps cannot invalidate the snapshot."""
    copied = jax.tree.map(lambda x: jax.device_put(x, x.sharding, may_alias=False), state)
    return jax.block_until_ready(copied)


def commit_parameters(state: GrugTrainState, candidate: Transformer) -> GrugTrainState:
    """Commit only parameter values; retain moments, clock, EMA and router state."""
    if state.master_params is not None:
        raise ValueError("This experiment requires the native FP32 parameter state without separate master weights")
    return replace(state, params=candidate)
