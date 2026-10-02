# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared paged-model correctness scenarios for inference tests."""

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis

from levanter.inference.page_table import PageBatchInfo, PageTableSpec


def assert_mixed_paged_decode_matches_forward(model):
    """Check packed chunked requests, noncontiguous pages, and padding against full forward."""
    # Keep token buffers divisible by the data mesh, including padding after the valid prefix.
    capacity = 8 * jax.device_count()
    token_axis = Axis("position", capacity)
    sequences = [np.array([2, 8, 3, 7, 1, 9]), np.array([6, 4, 10, 5, 11])]
    page_indices = np.array([[2, 4, 1], [3, 0, 5]], dtype=np.int32)
    phases = [((0, 3), (0, 2)), ((3, 2), (2, 1)), ((5, 1), (3, 2))]
    full_forward = hax.named_jit(lambda m, ids: m(ids))
    reference = []
    for seq in sequences:
        ids = hax.named(
            jnp.broadcast_to(jnp.asarray(seq, dtype=jnp.int32), (jax.device_count(), len(seq))),
            (Axis("batch", jax.device_count()), Axis("position", len(seq))),
        )
        reference.append(np.asarray(full_forward(model, ids).array)[0])
    cache = model.initial_cache(PageTableSpec(num_pages=6, page_size=2, max_seqs=2), dtype=jnp.float32)
    decode = hax.named_jit(lambda m, ids, state, info, pos: m.decode(ids, state, info, pos))
    for phase in phases:
        lengths = [length for _, length in phase]
        n = sum(lengths)
        ids = np.zeros(capacity, dtype=np.int32)
        positions = np.zeros(capacity, dtype=np.int32)
        dests = np.full(capacity, -1, dtype=np.int32)
        expected = []
        offset = 0
        for seq_id, (start, length) in enumerate(phase):
            pos = np.arange(start, start + length)
            ids[offset : offset + length] = sequences[seq_id][pos]
            positions[offset : offset + length] = pos
            dests[offset : offset + length] = page_indices[seq_id, pos // 2] * 2 + pos % 2
            expected.append(reference[seq_id][pos])
            offset += length
        info = PageBatchInfo(
            slot_ids=hax.named(jnp.array([0, 1], dtype=jnp.int32), "seq"),
            page_indices=hax.named(jnp.asarray(page_indices), ("seq", "page")),
            seq_lens=hax.named(jnp.array([start + length for start, length in phase], dtype=jnp.int32), "seq"),
            cu_q_lens=hax.named(jnp.array([0, lengths[0], n], dtype=jnp.int32), "seq"),
            num_seqs=jnp.array(2, dtype=jnp.int32),
            new_token_dests=hax.named(jnp.asarray(dests), token_axis),
            page_size=2,
        )
        logits, cache = decode(
            model,
            hax.named(jnp.asarray(ids), token_axis),
            cache,
            info,
            hax.named(jnp.asarray(positions), token_axis),
        )
        np.testing.assert_allclose(np.asarray(logits.array)[:n], np.concatenate(expected), rtol=1e-4, atol=1e-4)
        assert np.isfinite(np.asarray(logits.array)).all()
