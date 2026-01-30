#!/usr/bin/env python3
"""Simple test script to verify the debug logging in _compute_batched_embeddings."""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Use CPU to avoid TPU issues

import jax
import jax.numpy as jnp
import numpy as np

print("Testing debug logging in _compute_batched_embeddings simulation...")

# Simulate the relevant parts of _compute_batched_embeddings
def simulate_batched_embeddings_debug():
    # Simulate parameters similar to actual run
    num_tokens = 2000  # Total packed tokens
    num_seqs = 2  # Number of sequences in batch
    num_batched_segments = 2

    # Simulate cu_q_lens
    seq_lens = [1000, 1000]
    max_seqs = 8
    cu_q_lens_full = jnp.zeros(max_seqs + 1, dtype=jnp.int32)
    cu_q_lens_full = cu_q_lens_full.at[0].set(0)
    cu_q_lens_full = cu_q_lens_full.at[1].set(seq_lens[0])
    cu_q_lens_full = cu_q_lens_full.at[2].set(sum(seq_lens))

    # Use only valid portion
    cu_q_lens = cu_q_lens_full[:num_seqs + 1]

    # Compute segment_ids
    positions = jnp.arange(num_tokens)
    segment_indices = jnp.searchsorted(cu_q_lens, positions, side='right') - 1
    segment_ids = jnp.clip(segment_indices, 0, num_seqs - 1)

    # Simulate image token mask
    # For SigLIP 384x384 patch16: 576 features per image
    # Simulate 576 image tokens per sequence = 1152 total
    features_per_image = 576
    is_image_token = jnp.zeros(num_tokens, dtype=jnp.bool_)
    # Put image tokens at the beginning of each sequence
    for seg_id in range(num_seqs):
        start = cu_q_lens[seg_id]
        for i in range(features_per_image):
            is_image_token = is_image_token.at[start + i].set(True)

    # Simulate batched image features
    # With 1 patch per image (disable_anyres), features_per_patch = 576
    patches_per_request = 1  # Only base patch with disable_anyres
    features_per_patch = 576
    total_patches_axis = num_batched_segments * patches_per_request
    total_features = total_patches_axis * features_per_patch

    # Compute segment feature starts (uniform assumption)
    patches_per_segment = total_patches_axis // num_batched_segments
    features_per_segment_padded = patches_per_segment * features_per_patch
    segment_feature_starts = jnp.arange(num_batched_segments) * features_per_segment_padded

    # Compute within_seg_counts
    def segment_cumsum_fn(carry, x):
        mask_val, seg_id = x
        count = carry[seg_id]
        new_carry = carry.at[seg_id].add(mask_val)
        return new_carry, count

    init_counts = jnp.zeros(num_batched_segments, dtype=jnp.int32)
    _, within_seg_counts = jax.lax.scan(
        segment_cumsum_fn,
        init_counts,
        (is_image_token.astype(jnp.int32), segment_ids)
    )

    # Compute feature indices
    segment_starts_per_token = segment_feature_starts[segment_ids]
    feature_indices = segment_starts_per_token + within_seg_counts

    # Print debug info (simulating jax.debug.print)
    image_token_count = jnp.sum(is_image_token)
    max_within_seg = jnp.max(within_seg_counts)

    print(f"=== Batched Embeddings Debug (Simulated) ===")
    print(f"num_seqs={num_seqs}")
    print(f"num_batched_segments={num_batched_segments}")
    print(f"image_token_count={int(image_token_count)}")
    print(f"total_features={total_features}")
    print(f"features_per_segment_padded={features_per_segment_padded}")
    print(f"max_within_seg_count={int(max_within_seg)}")
    print(f"segment_feature_starts={segment_feature_starts}")

    # Check for out-of-bounds
    potential_max_idx = int(jnp.max(segment_feature_starts)) + int(max_within_seg)
    print(f"\nmax possible feature_index = {potential_max_idx}")
    print(f"total_features = {total_features}")

    if potential_max_idx >= total_features:
        print(f"⚠ WARNING: Feature index overflow! {potential_max_idx} >= {total_features}")
        print(f"   This means some image placeholders will gather WRONG features!")
    else:
        print(f"✓ Feature indices within bounds")

    # Count actual out-of-bounds
    oob_count = jnp.sum(feature_indices >= total_features)
    if oob_count > 0:
        print(f"⚠ {int(oob_count)} feature indices are out of bounds BEFORE clipping")

        # Show which segment has the issue
        for seg_id in range(num_batched_segments):
            seg_mask = segment_ids == seg_id
            seg_within_counts = within_seg_counts * seg_mask
            max_in_seg = jnp.max(seg_within_counts)
            seg_start = segment_feature_starts[seg_id]
            print(f"   Segment {seg_id}: start={int(seg_start)}, max_within={int(max_in_seg)}, "
                  f"max_idx={int(seg_start + max_in_seg)}, limit={int(seg_start + features_per_segment_padded)}")

if __name__ == "__main__":
    simulate_batched_embeddings_debug()
