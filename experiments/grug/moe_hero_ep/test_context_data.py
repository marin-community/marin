# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import BlockShuffleConfig
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import (
    HARRIER_MIX_2026_08_18_STORE,
    harrier_mix_2026_08_18_data_config,
)


def _hero_data(sequence_length):
    return harrier_mix_2026_08_18_data_config(
        ctx=StepContext.for_fingerprint(runtime_arg_keys=(), deps=(HARRIER_MIX_2026_08_18_STORE,)),
        total_steps=390_251,
        batch_size=46_137_344 // sequence_length,
        max_seq_len=sequence_length,
        experiment_flops=2.7e24,
        validation=(),
    )


@pytest.mark.parametrize("sequence_length", [8192, 16384, 65536])
def test_context_switch_preserves_completed_shuffle_window_token_coverage(sequence_length):
    # Represent tokens by their 4K chunk IDs so the test reads no training storage. There
    # are more than two windows: covering the first window is not covering the whole dataset.
    chunk_count = 1025 * 256
    coverage = []
    for length in (4096, sequence_length):
        shuffle = _hero_data(length).shuffle
        assert isinstance(shuffle, BlockShuffleConfig)
        chunks_per_sequence = length // 4096
        dataset = ListAsyncDataset(
            [range(i, i + chunks_per_sequence) for i in range(0, chunk_count, chunks_per_sequence)]
        ).block_shuffle(
            io_block_size=shuffle.io_block_size,
            window_blocks=shuffle.window_blocks,
            key=jax.random.PRNGKey(17),
            perm_type=shuffle.perm_type,
        )
        sequences = dataset.as_sync_dataset().get_batch(range(shuffle.io_block_size * shuffle.window_blocks))
        coverage.append({chunk for sequence in sequences for chunk in sequence})

    assert len(coverage[0]) == 512 * 256
    assert coverage[0] == coverage[1]


@pytest.mark.parametrize(
    ("sequence_length", "expected_starts"),
    [(4096, [0, 108000, 312192]), (8192, [0, 108000, 312192]), (16384, [0, 108096, 312192])],
)
def test_context_switch_exposes_historical_mixture_boundary_shift(sequence_length, expected_starts):
    # These are persisted hero lineage milestones. The 16K shift is an operational
    # difference that must be accepted explicitly, not mistaken for schedule equality.
    data = _hero_data(sequence_length)
    assert isinstance(data.train_weights, list)
    assert [step for step, _ in data.train_weights] == expected_starts
    assert data.mixture_block_size == 49_152
