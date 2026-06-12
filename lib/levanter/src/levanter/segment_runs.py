# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-shape contiguous segment-run metadata."""

import equinox as eqx
import jax
import jax.numpy as jnp


class SegmentRunMetadata(eqx.Module):
    """Contiguous segment lengths padded to a fixed max segment count."""

    segment_lengths: jax.Array
    num_segments: jax.Array


def segment_run_metadata_from_segment_ids(
    segment_ids: jax.Array,
    *,
    max_segments: int,
) -> SegmentRunMetadata:
    if max_segments <= 0:
        raise ValueError(f"max_segments must be positive, got {max_segments}.")

    segment_ids = jnp.asarray(segment_ids)
    if segment_ids.ndim == 0:
        raise ValueError("segment_ids must include a sequence dimension.")

    seq_len = segment_ids.shape[-1]
    flat_segment_ids = segment_ids.reshape((-1, seq_len))

    def one_row(row_segment_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
        first = jnp.zeros((seq_len,), dtype=jnp.bool_).at[0].set(True)
        previous = jnp.concatenate([row_segment_ids[:1], row_segment_ids[:-1]], axis=0)
        starts = first | (row_segment_ids != previous)
        num_segments = jnp.sum(starts, dtype=jnp.int32)
        start_positions = jnp.nonzero(starts, size=max_segments, fill_value=seq_len)[0].astype(jnp.int32)
        end_positions = jnp.concatenate([start_positions[1:], jnp.asarray([seq_len], dtype=jnp.int32)], axis=0)
        lengths = jnp.maximum(end_positions - start_positions, 0)
        lengths = jnp.where(jnp.arange(max_segments, dtype=jnp.int32) < num_segments, lengths, 0)
        lengths = eqx.error_if(
            lengths,
            num_segments > max_segments,
            "packed segment_ids contain more contiguous runs than max_segments.",
        )
        return lengths.astype(jnp.int32), num_segments

    lengths, num_segments = jax.vmap(one_row)(flat_segment_ids)
    out_shape = segment_ids.shape[:-1]
    return SegmentRunMetadata(
        segment_lengths=lengths.reshape((*out_shape, max_segments)),
        num_segments=num_segments.reshape(out_shape),
    )
