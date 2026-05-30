# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `NoiseRolloutLoader` and `NoopReplayBuffer`.

These tests verify the contract that lets `TrainWorker` drop in a synthetic
data loader in place of the real rollout pipeline. They do not require a TPU
and run under the standard CPU JAX backend.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from marin.rl.noise_rollout_loader import NoiseRolloutConfig, NoiseRolloutLoader, NoopReplayBuffer
from marin.rl.types import TrainingBatch


def _single_device_mesh() -> Mesh:
    devices = np.asarray(jax.local_devices()[:1])
    return Mesh(devices, axis_names=("data",))


def _make_train_worker_config_stub(
    *,
    train_batch_size: int = 8,
    max_seq_len: int = 32,
    vocab_size: int = 257,
    pad_to_multiple: int = 16,
    attn_backend=None,
) -> SimpleNamespace:
    """Build a `TrainWorkerConfig`-shaped stub the loader can read fields off.

    Only fields the loader touches are populated; nothing else is required
    because the loader only reads attributes.
    """
    mesh = _single_device_mesh()
    compute_axis_mapping = {"batch": "data"}

    trainer = SimpleNamespace(
        train_batch_size=train_batch_size,
        data_axis_size=1,
        device_mesh=mesh,
        compute_axis_mapping=compute_axis_mapping,
    )
    curriculum_config = SimpleNamespace(max_seq_len=max_seq_len)
    model = SimpleNamespace(attn_backend=attn_backend, flash_attention_block_size=pad_to_multiple)
    tokenizer = SimpleNamespace()
    # vocab_size on the worker config; loader will fall back to noise_config first.
    return SimpleNamespace(
        trainer=trainer,
        curriculum_config=curriculum_config,
        model=model,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
    )


def test_noise_rollout_loader_yields_valid_training_batches():
    config = _make_train_worker_config_stub(train_batch_size=8, max_seq_len=32, vocab_size=257)
    noise_config = NoiseRolloutConfig(seed=42)

    loader = NoiseRolloutLoader(config=config, noise_config=noise_config)

    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) == 3:
            break

    assert len(batches) == 3
    for batch in batches:
        assert isinstance(batch, TrainingBatch)
        # Named axes match the production contract.
        assert batch.input_ids.axes[0].name == "batch"
        assert batch.input_ids.axes[1].name == "position"
        assert batch.loss_masks.axes[0].name == "batch"
        assert batch.loss_masks.axes[1].name == "position"
        assert batch.temperature.axes[0].name == "batch"
        # Shapes: 8 rows, padded sequence length.
        batch_size = batch.input_ids.axis_size("batch")
        seq_len = batch.input_ids.axis_size("position")
        assert batch_size == 8
        # Default pad_to_multiple=16 (we set flash_attention_block_size=16), max_seq_len=32 -> 32.
        assert seq_len == 32

        # Every row must have a non-zero loss mask so the train_batch.py:150 assert is satisfied.
        per_row_mask_sum = batch.loss_masks.array.sum(axis=1)
        assert jnp.all(per_row_mask_sum > 0), per_row_mask_sum

        # Dtypes match what the loss expects.
        assert batch.input_ids.dtype == jnp.int32
        assert batch.position_ids.dtype == jnp.int32
        assert batch.loss_weights.dtype == jnp.float32
        assert batch.loss_masks.dtype == jnp.float32
        assert batch.policy_logprobs.dtype == jnp.float32
        assert batch.temperature.dtype == jnp.float32

        # Vocab range respected.
        assert int(batch.input_ids.array.max()) < 257
        assert int(batch.input_ids.array.min()) >= 0

        # truncated is per-row bool.
        assert batch.truncated.shape == (8,)
        assert batch.truncated.dtype == jnp.bool_


def test_noise_rollout_loader_records_batch_prep_timing():
    config = _make_train_worker_config_stub()
    loader = NoiseRolloutLoader(config=config, noise_config=NoiseRolloutConfig(seed=0))

    it = iter(loader)
    _ = next(it)

    timing = loader._last_batch_prep_timing
    assert timing.fetch_time == 0.0
    assert timing.batch_time >= 0.0
    assert timing.shard_time >= 0.0
    assert timing.total_time >= 0.0
    # _last_rollouts stays None so _log_samples_hook becomes a no-op.
    assert loader._last_rollouts is None


def test_noise_rollout_loader_rejects_indivisible_batch_size():
    config = _make_train_worker_config_stub(train_batch_size=7)
    # data_axis_size = 1, so 7 is divisible — we instead override data_axis_size.
    config.trainer.data_axis_size = 2

    with pytest.raises(ValueError, match="divisible"):
        NoiseRolloutLoader(config=config, noise_config=NoiseRolloutConfig(seed=0))


def test_noise_rollout_loader_respects_vocab_override():
    config = _make_train_worker_config_stub(vocab_size=1000)
    loader = NoiseRolloutLoader(
        config=config,
        noise_config=NoiseRolloutConfig(seed=0, vocab_size=100),
    )
    it = iter(loader)
    batch = next(it)
    assert int(batch.input_ids.array.max()) < 100


def test_noise_rollout_loader_loss_masks_have_response_region_only():
    config = _make_train_worker_config_stub(train_batch_size=4, max_seq_len=16, pad_to_multiple=8)
    loader = NoiseRolloutLoader(config=config, noise_config=NoiseRolloutConfig(seed=0))
    it = iter(loader)
    batch = next(it)

    # First half is prompt (mask 0), second half is response (mask 1).
    seq_len = batch.input_ids.axis_size("position")
    mask = batch.loss_masks.array
    half = seq_len // 2
    assert jnp.all(mask[:, :half] == 0.0)
    assert jnp.all(mask[:, half:] == 1.0)


def test_noop_replay_buffer_surface():
    buf = NoopReplayBuffer()
    assert buf.size() == 1  # bootstrap-bypass for any defensive checks
    buf.set_current_step(7)
    assert buf._current_step == 7
    stats = buf.get_stats()
    assert "total_size" in stats
    assert "env_sizes" in stats


def test_noise_rollout_loader_produces_distinct_batches():
    """Two successive iterations should not produce identical input_ids.

    Otherwise the trainer would be compiling and stepping against a single
    fixed prompt across every step, which defeats the purpose of stressing
    the optimizer + weight-export path.
    """
    config = _make_train_worker_config_stub()
    loader = NoiseRolloutLoader(config=config, noise_config=NoiseRolloutConfig(seed=123))
    it = iter(loader)
    b0 = next(it)
    b1 = next(it)
    assert not jnp.array_equal(b0.input_ids.array, b1.input_ids.array)
    assert not jnp.array_equal(b0.loss_weights.array, b1.loss_weights.array)
