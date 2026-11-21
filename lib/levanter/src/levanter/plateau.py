# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Plateau dropping: remove a token from the training loss once its loss has stopped changing.

Reconstructed from Harshit Varma's write-up (marin issue #10028 and gist hrshtv/76c17ac7).
Every supervised token keeps its loss at its last two visits and a visit count. At the third
visit and later, if both visit-to-visit relative changes (including the current visit) are below
``relative_change``, the token's term is removed from the training loss. It stays in the input,
its loss is still measured on every visit, and it is trained again as soon as its loss moves.
The training loss is the mean over the kept supervised tokens.

Rows are identified by a hash of their ``input_ids`` and ``loss_weights``, so the state does not
depend on how the loader orders rows, and survives per-epoch permutations and resume.
"""

import hashlib
import io
import logging
import typing
from dataclasses import dataclass

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import NamedArray, NamedOrNumeric
from jax.experimental import multihost_utils

from levanter.callbacks import StepInfo
from levanter.checkpoint import is_checkpoint_path
from levanter.metrics import Metric, ReductionType
from levanter.models.lm_model import LmExample, LmHeadModel, split_activations
from levanter.models.loss import maybe_fused_next_token_loss, next_token_loss_weight
from levanter.trainer import S, Trainer
from rigging.filesystem.storage_path import StoragePath, prefix_join

logger = logging.getLogger(__name__)

TOKEN_LOSS_METRIC = "plateau_token_loss"


@dataclass(frozen=True)
class PlateauDroppingConfig:
    relative_change: float = 0.1
    """A token plateaus when both of its last two relative loss changes are below this."""
    denominator_floor: float = 0.05
    """Floor on the earlier loss in each relative change, in nats."""
    min_visits: int = 3
    """The first visit at which a token can be dropped; the test needs that many losses."""


class HostHistory(typing.NamedTuple):
    """A batch's history on the host, laid out like ``example.tokens``; fields as in PlateauHistory."""

    previous: np.ndarray
    last: np.ndarray
    visits: np.ndarray


class PlateauHistory(typing.NamedTuple):
    """Per-token history for one batch, laid out like ``example.tokens``."""

    previous: NamedArray
    """Loss at the second-to-last visit."""
    last: NamedArray
    """Loss at the last visit."""
    visits: NamedArray
    """Completed visits before this one."""


def plateau_mask(config: PlateauDroppingConfig, history: PlateauHistory, current: NamedArray) -> NamedArray:
    """Return True where the token's loss has plateaued over its last three visits."""

    def change(earlier, later):
        return hax.abs(later - earlier) / hax.maximum(earlier, config.denominator_floor)

    current = jax.lax.stop_gradient(current)
    stable = (change(history.previous, history.last) < config.relative_change) & (
        change(history.last, current) < config.relative_change
    )
    return stable & (history.visits >= config.min_visits - 1)


def token_cross_entropy(model: LmHeadModel, example: LmExample, *, key=None) -> tuple[NamedArray, NamedOrNumeric]:
    """Next-token cross-entropy at every position (the last is 0), and the model's aux loss."""
    activations, aux_loss = split_activations(model.activations(example.tokens, example.attn_mask, key=key))
    token_loss = maybe_fused_next_token_loss(
        model.Pos,
        model.Embed,
        model.Vocab,
        activations,
        model.get_lm_head(),
        example.tokens,
        loss_weight=None,
        reduction=None,
        dtype=jnp.float32,
    )
    return token_loss, aux_loss


def plateau_training_loss(
    model: LmHeadModel,
    example: LmExample,
    history: PlateauHistory | None = None,
    *,
    config: PlateauDroppingConfig,
    key=None,
):
    """Supervised-token mean over kept tokens, plus totals for the epoch statistics.

    Without ``history`` (evaluation) every supervised token is kept, which is plain SFT.
    """
    token_loss, aux_loss = token_cross_entropy(model, example, key=key)
    position = example.tokens.resolve_axis(model.Pos.name)
    weights = hax.broadcast_axis(next_token_loss_weight(position, example.loss_weight), example.tokens.axes)
    weights = weights.astype(jnp.float32)
    if history is None:
        dropped = hax.zeros_like(weights)
    else:
        dropped = plateau_mask(config, history, token_loss).astype(jnp.float32) * weights
    kept = weights - dropped
    kept_tokens = hax.sum(kept).scalar()
    kept_loss_sum = hax.sum(token_loss * kept).scalar()
    loss = jnp.where(kept_tokens > 0, kept_loss_sum / jnp.maximum(kept_tokens, 1.0), 0.0) + aux_loss
    supervised_tokens = hax.sum(weights).scalar()
    loss_sum = hax.sum(token_loss * weights).scalar()
    return loss, {
        "loss_sum": Metric.from_value(loss_sum, ReductionType.SUM),
        "supervised_tokens": Metric.from_value(supervised_tokens, ReductionType.SUM),
        "plateau_kept_tokens": Metric.from_value(kept_tokens, ReductionType.SUM),
        "plateau_kept_loss_sum": Metric.from_value(kept_loss_sum, ReductionType.SUM),
        "plateau_dropped_tokens": Metric.from_value(supervised_tokens - kept_tokens, ReductionType.SUM),
        "plateau_dropped_loss_sum": Metric.from_value(loss_sum - kept_loss_sum, ReductionType.SUM),
        TOKEN_LOSS_METRIC: Metric.from_value(jax.lax.stop_gradient(token_loss.array), ReductionType.LAST),
    }


def row_key(input_ids: np.ndarray, loss_weights: np.ndarray) -> str:
    digest = hashlib.sha256(np.ascontiguousarray(input_ids, dtype=np.int32).tobytes())
    digest.update(np.ascontiguousarray(loss_weights, dtype=np.float32).tobytes())
    return digest.hexdigest()


@dataclass
class RowHistory:
    positions: np.ndarray
    previous: np.ndarray
    last: np.ndarray
    visits: int = 0


class PlateauState:
    """Host-side loss history, keyed by (row hash, token position)."""

    def __init__(self):
        self.rows: dict[str, RowHistory] = {}

    def lookup(self, input_ids: np.ndarray, loss_weights: np.ndarray) -> tuple[list[str], HostHistory]:
        """Return each row's key and the batch's history arrays."""
        keys = [row_key(ids, weights) for ids, weights in zip(input_ids, loss_weights, strict=True)]
        previous = np.zeros(input_ids.shape, dtype=np.float32)
        last = np.zeros(input_ids.shape, dtype=np.float32)
        visits = np.zeros(input_ids.shape, dtype=np.int32)
        for index, key in enumerate(keys):
            row = self.rows.get(key)
            if row is None:
                continue
            previous[index, row.positions] = row.previous
            last[index, row.positions] = row.last
            visits[index, row.positions] = row.visits
        return keys, HostHistory(previous, last, visits)

    def record(self, keys: list[str], loss_weights: np.ndarray, token_loss: np.ndarray) -> None:
        """Add one visit per row. Positions are the supervised next-token targets."""
        for index, key in enumerate(keys):
            weights = loss_weights[index].copy()
            weights[-1] = 0
            positions = np.flatnonzero(weights)
            row = self.rows.get(key)
            if row is None:
                row = RowHistory(positions, np.zeros(len(positions), np.float32), np.zeros(len(positions), np.float32))
                self.rows[key] = row
            row.previous = row.last
            row.last = token_loss[index, positions].astype(np.float32)
            row.visits += 1

    def dumps(self) -> bytes:
        keys = sorted(self.rows)
        rows = [self.rows[key] for key in keys]
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            keys=np.array(keys),
            lengths=np.array([len(row.positions) for row in rows], dtype=np.int64),
            visits=np.array([row.visits for row in rows], dtype=np.int64),
            positions=np.concatenate([row.positions for row in rows]) if rows else np.zeros(0, np.int64),
            previous=np.concatenate([row.previous for row in rows]) if rows else np.zeros(0, np.float32),
            last=np.concatenate([row.last for row in rows]) if rows else np.zeros(0, np.float32),
        )
        return buffer.getvalue()

    @classmethod
    def loads(cls, data: bytes) -> "PlateauState":
        arrays = np.load(io.BytesIO(data))
        state = cls()
        bounds = np.concatenate([[0], np.cumsum(arrays["lengths"])])
        for index, key in enumerate(arrays["keys"]):
            start, end = bounds[index], bounds[index + 1]
            state.rows[str(key)] = RowHistory(
                arrays["positions"][start:end],
                arrays["previous"][start:end],
                arrays["last"][start:end],
                int(arrays["visits"][index]),
            )
        return state


def _host(array: jax.Array) -> np.ndarray:
    if jax.process_count() == 1:
        return np.asarray(array)
    return np.asarray(multihost_utils.process_allgather(array, tiled=True))


def _device(values: np.ndarray, like: NamedArray) -> NamedArray:
    sharding = like.array.sharding
    array = jax.make_array_from_callback(values.shape, sharding, lambda index: values[index])
    return hax.named(array, like.axes)


class PlateauTrainer(Trainer):
    """A Trainer that feeds each batch its loss history and records the visit after the step.

    The history is written next to every checkpoint the run saves, as
    ``plateau-updates-<N>.npz`` in the checkpoint's parent directory, where N is the completed
    update count. A resumed run refuses to start without the file for its restored count.
    """

    def __init__(self, config, optimizer, loss_fn, *, add_default_hooks: bool = True):
        """``loss_fn`` is ``plateau_training_loss`` with its config bound, optionally wrapped to add metrics."""
        super().__init__(config, optimizer, loss_fn, add_default_hooks=add_default_hooks)
        if config.microbatch_size is not None and config.microbatch_size != config.train_batch_size:
            raise ValueError("Plateau dropping normalizes over the global batch; microbatches are not supported")
        self.plateau = PlateauState()
        self._plateau_saved_updates = -1
        self._pending: tuple[list[str], np.ndarray] = ([], np.zeros(0))

    @staticmethod
    def plateau_file(directory: str, updates: int) -> StoragePath:
        return StoragePath(prefix_join(directory, f"plateau-updates-{updates}.npz"))

    def load_plateau_state(self, updates: int) -> None:
        """Load the history saved with the checkpoint this run resumed from."""
        if updates == 0:
            return
        directories = [
            str(StoragePath(path).parent) if is_checkpoint_path(path) else path
            for path in self.checkpoint_search_paths
        ]
        for directory in directories:
            path = self.plateau_file(directory, updates)
            if path.exists():
                self.plateau = PlateauState.loads(path.read_bytes())
                self._plateau_saved_updates = updates
                logger.info("Loaded plateau history for %d rows from %s", len(self.plateau.rows), path)
                return
        raise FileNotFoundError(
            f"Resuming at {updates} updates requires plateau-updates-{updates}.npz in {directories}"
        )

    def _save_plateau_state(self, updates: int) -> None:
        """Write the history next to the checkpoint directories, once per saved update count."""
        if updates == self._plateau_saved_updates:
            return
        if jax.process_index() == 0:
            data = self.plateau.dumps()
            for directory in {self._checkpointer.base_path, self._checkpointer.temporary_base_path} - {None}:
                self.plateau_file(directory, updates).write_bytes(data)
        self._plateau_saved_updates = updates

    def training_steps(self, state: S, train_loader) -> typing.Iterator[StepInfo[S]]:
        self.load_plateau_state(int(state.step))
        return super().training_steps(state, train_loader)

    def _step_inputs(self, batch: tuple) -> tuple:
        """Look up the batch's history; the compiled loss receives ``(example, history)``."""
        (example,) = batch
        loss_weights = _host(example.loss_weight.array)
        keys, host = self.plateau.lookup(_host(example.tokens.array), loss_weights)
        self._pending = (keys, loss_weights)
        history = PlateauHistory(*(_device(values, example.tokens) for values in host))
        return (example, history)

    def _step_metrics(self, result) -> dict[str, typing.Any]:
        """Record this step's token losses before the hooks run, so a checkpoint saved now includes them."""
        metrics = super()._step_metrics(result)
        keys, loss_weights = self._pending
        self.plateau.record(keys, loss_weights, _host(metrics.pop(f"train/{TOKEN_LOSS_METRIC}")))
        return metrics

    def run_hooks(self, info: StepInfo, force: bool = False):
        """Run the hooks; if the checkpointer saved, save the history that matches it."""
        saved_before = self._checkpointer._last_save_step
        super().run_hooks(info, force=force)
        if self._checkpointer._last_save_step != saved_before:
            self._save_plateau_state(int(info.state.step))


__all__ = [
    "PlateauDroppingConfig",
    "PlateauHistory",
    "PlateauState",
    "PlateauTrainer",
    "plateau_mask",
    "plateau_training_loss",
    "row_key",
    "token_cross_entropy",
]
