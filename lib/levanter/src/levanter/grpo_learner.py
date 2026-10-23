# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared synchronous GRPO steps and SPMD execution."""

from dataclasses import dataclass
from functools import partial
from typing import Generic, Protocol, TypeVar

import equinox as eqx
import haliax as hax
import jax
from haliax.partitioning import named_jit

from levanter.grpo_model import GrpoExample, response_logprobs
from levanter.trainer import Trainer
from levanter.trainer_state import TrainerState

S = TypeVar("S")
TrainerStateT = TypeVar("TrainerStateT", bound=TrainerState)


@dataclass(frozen=True)
class GrpoStep(Generic[S]):
    state: S
    loss: float


class GrpoBackend(Protocol[S]):
    """Execute scoring and updates on batches with global objective weights."""

    def score(self, state: S, batch: GrpoExample) -> GrpoExample: ...

    def update(self, state: S, batch: GrpoExample) -> GrpoStep[S]: ...


def grpo_step(backend: GrpoBackend[S], state: S, batch: GrpoExample) -> GrpoStep[S]:
    """Score the old policy, then update it once on the accepted batch."""
    return backend.update(state, backend.score(state, batch))


@named_jit
def _score_microbatch(model, batch, *, mp, block_size):
    return response_logprobs(mp.cast_to_compute(model), batch, block_size=block_size)


def _slice_batch(value, *, start: int, size: int):
    if not isinstance(value, hax.NamedArray):
        return value
    return value["batch", hax.ds(start, size)]


@dataclass(frozen=True)
class SpmdGrpoBackend:
    """Use a Trainer configured with grpo_model_loss for SPMD GRPO updates.

    The caller supplies its loss partial with accumulation_steps equal to the
    number of execution microbatches. This compensates Trainer's gradient mean
    while preserving the batch's precomputed global objective weights.
    """

    trainer: Trainer
    block_size: int

    def score(self, state: TrainerStateT, batch: GrpoExample) -> GrpoExample:
        """Recompute old-policy scores with the update's execution shape."""
        Batch = batch.tokens.resolve_axis("batch")
        microbatch_size = self.trainer.config.microbatch_size or Batch.size
        if Batch.size % microbatch_size:
            raise ValueError("Complete GRPO batch must divide evenly into execution microbatches")
        batch = hax.shard(batch, self.trainer.compute_axis_mapping)
        scores = []
        for start in range(0, Batch.size, microbatch_size):
            microbatch = jax.tree.map(
                partial(_slice_batch, start=start, size=microbatch_size),
                batch,
                is_leaf=lambda x: isinstance(x, hax.NamedArray),
            )
            with hax.axis_mapping(self.trainer.compute_axis_mapping):
                scores.append(
                    _score_microbatch(state.model, microbatch, mp=self.trainer.mp, block_size=self.block_size)
                )
        return eqx.tree_at(lambda x: x.old_logprobs, batch, hax.concatenate(Batch, scores))

    def update(self, state: TrainerStateT, batch: GrpoExample) -> GrpoStep[TrainerStateT]:
        """Apply the Trainer's optimizer and model averaging to a scored batch."""
        info = self.trainer.train_step(state, batch)
        return GrpoStep(info.state, info.loss)
