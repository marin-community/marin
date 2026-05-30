# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Random-tensor data loader for isolating the RL trainer + weight export path.

`NoiseRolloutLoader` replaces `StreamingRolloutLoader` in `TrainWorker` to feed
the train loop synthetic batches that match the production `TrainingBatch`
contract bit-for-bit (named axes, dtypes, sharding). This lets us exercise
forward, backward, optimizer step, and weight export without any rollout
worker, replay buffer, or vLLM dependency.

Used to bisect the multi-host weight-sync bug described in marin#4287: the
crash is on the export path, not on training compute, so swapping in a noise
loader is the cheapest way to iterate on the export implementation.

A companion `NoopReplayBuffer` provides the `set_current_step` / `size`
surface that the train worker calls every step, so the rest of the hook
plumbing stays unchanged.
"""

from dataclasses import dataclass, field
from typing import Any

import haliax as hax
import jax.numpy as jnp
import numpy as np
from levanter.layers.attention import DEFAULT_SPLASH_BLOCK_SIZE, AttentionBackend
from levanter.models.flash_attention import BLOCK_SIZE as DEFAULT_FLASH_BLOCK_SIZE

from .batch_prep_timing import BatchPrepTiming
from .replay_buffer import RolloutWithCount
from .types import TrainingBatch


@dataclass(frozen=True)
class NoiseRolloutConfig:
    """Configuration for the noise rollout loader.

    Most fields default to ``None`` and are resolved against the surrounding
    ``TrainWorkerConfig`` at loader-construction time. Override the resolved
    values only when isolating a specific shape from the production curriculum.
    """

    seed: int = 0
    """Numpy RNG seed for deterministic noise generation."""

    max_seq_len: int | None = None
    """Total prompt + response sequence length. Defaults to
    ``train_worker_config.curriculum_config.max_seq_len``."""

    vocab_size: int | None = None
    """Vocab size for synthetic ``input_ids``. Defaults to
    ``train_worker_config.vocab_size`` or ``len(tokenizer)`` if missing."""

    advantage_scale: float = 0.1
    """Standard deviation for synthetic per-token loss weights ("advantages")."""

    policy_logprob_mean: float = -10.0
    """Mean of synthetic policy log-probabilities."""

    policy_logprob_std: float = 0.5
    """Standard deviation of synthetic policy log-probabilities."""


class NoopReplayBuffer:
    """Minimal `ReplayBuffer` stand-in for noise-trainer mode.

    Exposes only the methods that `TrainWorker` calls (`set_current_step`,
    `size`, `get_stats`, plus the `_current_step` attribute used by the debug
    snapshot). Returning `size() > 0` lets `_wait_for_initial_rollouts` exit
    immediately if a caller forgets to bypass it.
    """

    def __init__(self) -> None:
        self._current_step: int = 0

    def set_current_step(self, step: int) -> None:
        self._current_step = step

    def size(self) -> int:
        # Report a single rollout so any defensive wait-for-rollouts caller
        # treats the buffer as non-empty.
        return 1

    def get_stats(self) -> dict:
        return {
            "total_size": 1,
            "env_sizes": {},
            "num_environments": 0,
            "total_batches_added": 0,
            "total_batches_sampled": 0,
        }


@dataclass
class NoiseRolloutLoader:
    """Yield synthetic, already-sharded `TrainingBatch` instances.

    Mirrors the public surface of `StreamingRolloutLoader` so `TrainWorker`
    can swap it in transparently:

    - ``__iter__`` yields sharded `TrainingBatch` objects.
    - ``_last_batch_prep_timing`` exposes a `BatchPrepTiming` matching the
      production data loader contract.
    - ``_last_rollouts`` stays ``None`` (no real rollouts exist), and the
      `_log_samples_hook` already no-ops on `None`.

    Tensor shapes track the production rollout pipeline: per-process batch is
    ``train_batch_size // data_axis_size`` rows of length
    ``round_up(max_seq_len, pad_to_multiple)``. The first half of each row
    has `loss_masks == 0` (prompt region) and the second half has
    `loss_masks == 1` (response region), so the assert at
    ``train_batch.py:150`` never trips and the RLOO loss sees a non-zero
    denominator on every row.
    """

    config: Any
    """The owning `TrainWorkerConfig`. Annotated as `Any` to avoid a
    train_worker -> noise_rollout_loader -> train_worker import cycle. We only
    read field values; no method calls."""

    noise_config: NoiseRolloutConfig

    _batch_size_per_process: int = field(init=False)
    _seq_len: int = field(init=False)
    _vocab_size: int = field(init=False)
    _rng: np.random.Generator = field(init=False)

    _cached_input_ids: np.ndarray = field(init=False)
    _cached_position_ids: np.ndarray = field(init=False)
    _cached_loss_masks: np.ndarray = field(init=False)
    _cached_temperature: np.ndarray = field(init=False)
    _cached_top_k: np.ndarray = field(init=False)
    _cached_truncated: np.ndarray = field(init=False)

    _last_batch_prep_timing: BatchPrepTiming = field(default_factory=BatchPrepTiming)
    _last_rollouts: list[RolloutWithCount] | None = field(default=None)

    def __post_init__(self) -> None:
        config = self.config
        noise_config = self.noise_config

        # Resolve per-process batch size: production splits the global batch
        # across the trainer's data axis (one process per chip group).
        trainer = config.trainer
        data_axis_size = trainer.data_axis_size
        global_batch_size = trainer.train_batch_size
        if global_batch_size % data_axis_size != 0:
            raise ValueError(
                "train_batch_size must be divisible by data_axis_size for noise rollouts: "
                f"{global_batch_size=} {data_axis_size=}"
            )
        self._batch_size_per_process = global_batch_size // data_axis_size

        max_seq_len = noise_config.max_seq_len or config.curriculum_config.max_seq_len
        if max_seq_len <= 0:
            raise ValueError(f"NoiseRolloutLoader requires max_seq_len > 0, got {max_seq_len}")

        # Match StreamingRolloutLoader's pad_to_multiple rounding so the sharded
        # batch shape matches what the real loader would emit.
        is_splash = getattr(config.model, "attn_backend", None) == AttentionBackend.SPLASH
        flash_block_size = getattr(config.model, "flash_attention_block_size", None)
        pad_to_multiple = flash_block_size or (DEFAULT_SPLASH_BLOCK_SIZE if is_splash else DEFAULT_FLASH_BLOCK_SIZE)
        seq_len = ((max_seq_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        self._seq_len = seq_len

        if noise_config.vocab_size is not None:
            vocab_size = noise_config.vocab_size
        elif config.vocab_size is not None:
            vocab_size = config.vocab_size
        else:
            vocab_size = len(config.tokenizer)
        self._vocab_size = vocab_size

        self._rng = np.random.default_rng(seed=noise_config.seed)

        # Pre-build the deterministic, batch-invariant tensors once. We rebuild
        # the per-iteration noise (input_ids, loss_weights, policy_logprobs) on
        # every __next__ to keep gradients non-degenerate over time.
        local_batch = self._batch_size_per_process
        half_seq = seq_len // 2
        loss_masks = np.zeros((local_batch, seq_len), dtype=np.float32)
        loss_masks[:, half_seq:] = 1.0
        self._cached_loss_masks = loss_masks

        position_ids = np.broadcast_to(np.arange(seq_len, dtype=np.int32), (local_batch, seq_len)).copy()
        self._cached_position_ids = position_ids

        self._cached_temperature = np.ones((local_batch,), dtype=np.float32)
        self._cached_top_k = -np.ones((local_batch,), dtype=np.int32)
        self._cached_truncated = np.zeros((local_batch,), dtype=bool)
        # input_ids is regenerated per iter to keep the trainer from compiling
        # against a single fixed prompt.
        self._cached_input_ids = np.zeros((local_batch, seq_len), dtype=np.int32)

    def _build_batch(self) -> TrainingBatch:
        local_batch = self._batch_size_per_process
        seq_len = self._seq_len
        loss_masks = self._cached_loss_masks

        input_ids = self._rng.integers(
            low=0,
            high=self._vocab_size,
            size=(local_batch, seq_len),
            dtype=np.int32,
        )

        advantages = self._rng.normal(
            loc=0.0,
            scale=self.noise_config.advantage_scale,
            size=(local_batch, 1),
        ).astype(np.float32)
        loss_weights = (loss_masks * advantages).astype(np.float32)

        logprob_noise = self._rng.normal(
            loc=self.noise_config.policy_logprob_mean,
            scale=self.noise_config.policy_logprob_std,
            size=(local_batch, seq_len),
        ).astype(np.float32)
        policy_logprobs = (loss_masks * logprob_noise).astype(np.float32)

        return TrainingBatch(
            input_ids=hax.named(jnp.asarray(input_ids), ["batch", "position"]),
            position_ids=hax.named(jnp.asarray(self._cached_position_ids), ["batch", "position"]),
            loss_weights=hax.named(jnp.asarray(loss_weights), ["batch", "position"]),
            loss_masks=hax.named(jnp.asarray(loss_masks), ["batch", "position"]),
            policy_logprobs=hax.named(jnp.asarray(policy_logprobs), ["batch", "position"]),
            temperature=hax.named(jnp.asarray(self._cached_temperature), ["batch"]),
            top_k=hax.named(jnp.asarray(self._cached_top_k), ["batch"]),
            truncated=jnp.asarray(self._cached_truncated),
            max_output_tokens=seq_len // 2,
        )

    def __iter__(self):
        """Yield sharded synthetic batches indefinitely.

        Sharding mirrors `StreamingRolloutLoader.__iter__`: we wrap the
        per-process numpy arrays as Haliax named arrays, then call
        `hax.shard(batch, compute_axis_mapping)` inside `hax.set_mesh(device_mesh)`
        so the resulting batch is on-device and matches the contract Levanter's
        `Trainer.train` expects.
        """
        import time

        compute_axis_mapping = self.config.trainer.compute_axis_mapping
        device_mesh = self.config.trainer.device_mesh
        while True:
            batch_start = time.time()
            batch = self._build_batch()
            batch_time = time.time() - batch_start

            shard_start = time.time()
            with hax.set_mesh(device_mesh):
                sharded_batch = hax.shard(batch, compute_axis_mapping)
            shard_time = time.time() - shard_start

            self._last_batch_prep_timing = BatchPrepTiming(
                fetch_time=0.0,
                batch_time=batch_time,
                shard_time=shard_time,
            )
            yield sharded_batch
