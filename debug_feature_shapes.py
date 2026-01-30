#!/usr/bin/env python3
"""Debug script to check actual feature shapes in VLM batched inference."""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Use CPU to avoid TPU issues

import jax
import jax.numpy as jnp
import numpy as np

# Simulate the actual _compute_batched_embeddings logic with realistic values
def debug_feature_computation():
    """
    Debug the feature index computation to find where the mismatch occurs.
    """
    print("="*80)
    print("DEBUG: Feature Index Computation Analysis")
    print("="*80)

    # Configuration from eval_vlm.py
    image_size = 384
    patch_size = 16
    vision_feature_height = image_size // patch_size  # 24

    # Number of features per image (patches per image after vision encoder)
    features_per_patch = vision_feature_height * vision_feature_height  # 576

    # Number of image tokens in the tokenized text (from processor)
    num_image_tokens = vision_feature_height * vision_feature_height  # 576

    print(f"\nConfiguration:")
    print(f"  image_size: {image_size}")
    print(f"  patch_size: {patch_size}")
    print(f"  vision_feature_height: {vision_feature_height}")
    print(f"  features_per_patch (from vision encoder): {features_per_patch}")
    print(f"  num_image_tokens (in tokenized text): {num_image_tokens}")

    # Simulate batched mode with 2 sequences
    num_seqs = 2

    # In disable_anyres mode with single image per request:
    # - Each request has 1 patch (the base image tile)
    # - After padding, all requests have max_total_patches patches
    patches_per_request_original = [1, 1]  # disable_anyres = only base patch
    max_total_patches = max(patches_per_request_original)  # 1

    # After stacking and padding
    num_batched_segments = num_seqs
    total_patches_axis = num_batched_segments * max_total_patches  # 2 * 1 = 2 patches total

    # Features available
    total_features = total_patches_axis * features_per_patch  # 2 * 576 = 1152

    print(f"\nBatched mode with {num_seqs} sequences:")
    print(f"  patches_per_request_original: {patches_per_request_original}")
    print(f"  max_total_patches: {max_total_patches}")
    print(f"  total_patches_axis: {total_patches_axis}")
    print(f"  features_per_patch: {features_per_patch}")
    print(f"  total_features: {total_features}")

    # Segment computation
    patches_per_segment = total_patches_axis // num_batched_segments  # 2 // 2 = 1
    features_per_segment_padded = patches_per_segment * features_per_patch  # 1 * 576 = 576

    print(f"\nSegment computation:")
    print(f"  patches_per_segment: {patches_per_segment}")
    print(f"  features_per_segment_padded: {features_per_segment_padded}")

    # Segment feature starts
    segment_feature_starts = np.arange(num_batched_segments) * features_per_segment_padded
    print(f"  segment_feature_starts: {segment_feature_starts}")

    # Now simulate image placeholders in tokens
    # Each sequence has num_image_tokens (576) image placeholder tokens
    # Total tokens = seq0_len + seq1_len
    seq_lens = [600, 600]  # Each sequence has ~600 tokens (576 image + 24 text)
    num_tokens = sum(seq_lens)

    # Simulate cu_q_lens (cumulative sequence lengths)
    cu_q_lens = np.array([0, seq_lens[0], seq_lens[0] + seq_lens[1]])

    # Compute segment_ids for each token
    positions = np.arange(num_tokens)
    segment_ids = np.searchsorted(cu_q_lens, positions, side='right') - 1
    segment_ids = np.clip(segment_ids, 0, num_seqs - 1)

    print(f"\nToken analysis:")
    print(f"  seq_lens: {seq_lens}")
    print(f"  num_tokens: {num_tokens}")
    print(f"  cu_q_lens: {cu_q_lens}")

    # Simulate image tokens - first 576 tokens of each sequence are image tokens
    is_image_token = np.zeros(num_tokens, dtype=bool)
    for seg in range(num_seqs):
        start = cu_q_lens[seg]
        end = start + num_image_tokens
        is_image_token[start:end] = True

    image_token_count = np.sum(is_image_token)
    print(f"  image_token_count: {image_token_count}")
    print(f"  expected: {num_seqs * num_image_tokens}")

    # Compute within_seg_counts using scan-like logic
    within_seg_counts = np.zeros(num_tokens, dtype=np.int32)
    seg_counters = np.zeros(num_batched_segments, dtype=np.int32)

    for i in range(num_tokens):
        if is_image_token[i]:
            seg = segment_ids[i]
            within_seg_counts[i] = seg_counters[seg]
            seg_counters[seg] += 1

    # Check the maximum within_seg_count per segment
    for seg in range(num_batched_segments):
        seg_mask = (segment_ids == seg) & is_image_token
        max_in_seg = np.max(within_seg_counts[seg_mask]) if np.any(seg_mask) else 0
        seg_start = segment_feature_starts[seg]
        max_idx = seg_start + max_in_seg

        print(f"\n  Segment {seg}:")
        print(f"    segment_feature_start: {seg_start}")
        print(f"    max_within_seg_count: {max_in_seg}")
        print(f"    max_feature_index: {max_idx}")
        print(f"    segment_feature_limit: {seg_start + features_per_segment_padded}")

        if max_idx >= seg_start + features_per_segment_padded:
            print(f"    ⚠ OVERFLOW: max_feature_index >= segment_feature_limit!")
        else:
            print(f"    ✓ Within bounds")

    # Global check
    max_within_seg = np.max(within_seg_counts * is_image_token.astype(np.int32))
    max_feature_idx = np.max(segment_feature_starts) + max_within_seg

    print(f"\nGlobal analysis:")
    print(f"  max_within_seg_count (across all segments): {max_within_seg}")
    print(f"  max possible feature_index: {max_feature_idx}")
    print(f"  total_features: {total_features}")

    if max_feature_idx >= total_features:
        print(f"  ⚠ GLOBAL OVERFLOW: {max_feature_idx} >= {total_features}")
    else:
        print(f"  ✓ All feature indices within bounds")

    # THE KEY CHECK
    print(f"\n" + "="*80)
    print("KEY ANALYSIS:")
    print(f"  num_image_tokens per sequence: {num_image_tokens}")
    print(f"  features_per_segment_padded: {features_per_segment_padded}")

    if num_image_tokens == features_per_segment_padded:
        print(f"  ✓ MATCH: Image tokens ({num_image_tokens}) == features per segment ({features_per_segment_padded})")
    elif num_image_tokens > features_per_segment_padded:
        print(f"  ⚠ MISMATCH: Image tokens ({num_image_tokens}) > features per segment ({features_per_segment_padded})")
        print(f"     This causes {num_image_tokens - features_per_segment_padded} tokens to have out-of-bounds indices!")
    else:
        print(f"  ⚠ MISMATCH: Image tokens ({num_image_tokens}) < features per segment ({features_per_segment_padded})")
        print(f"     Some features will never be used.")
    print("="*80)


if __name__ == "__main__":
    debug_feature_computation()
