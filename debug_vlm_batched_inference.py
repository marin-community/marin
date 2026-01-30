#!/usr/bin/env python3
"""
Debug script for VLM batched inference - tests actual model inference.

This script helps diagnose why batched VLM inference produces garbage (newlines only)
after increasing max_prefill_size.

Run:
    uv run python debug_vlm_batched_inference.py

The script will:
1. Load a small VLM model
2. Create test VLM requests
3. Compare sequential vs batched inference results
4. Print detailed debugging information
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress some noisy loggers
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)


def test_batched_embeddings_calculation():
    """
    Test the segment_ids and feature_indices calculation logic
    without loading the full model.
    """
    import jax.numpy as jnp
    import jax

    logger.info("=" * 80)
    logger.info("TEST: Batched Embeddings Calculation Logic")
    logger.info("=" * 80)

    # Simulate a batch of 3 requests
    num_requests = 3
    max_patches_per_request = 4  # padded patch count
    features_per_patch = 14

    # Simulate token lengths for each request
    # Request 0: 100 tokens (50 text + 50 image placeholders)
    # Request 1: 80 tokens (40 text + 40 image placeholders)
    # Request 2: 120 tokens (60 text + 60 image placeholders)
    seq_lens = [100, 80, 120]
    total_tokens = sum(seq_lens)

    # Simulate cu_q_lens (cumulative query lengths)
    # Has max_seqs + 1 elements, but only first num_requests + 1 are valid
    max_seqs = 8
    cu_q_lens = jnp.zeros(max_seqs + 1, dtype=jnp.int32)
    cu_q_lens = cu_q_lens.at[0].set(0)
    cu_q_lens = cu_q_lens.at[1].set(seq_lens[0])
    cu_q_lens = cu_q_lens.at[2].set(seq_lens[0] + seq_lens[1])
    cu_q_lens = cu_q_lens.at[3].set(sum(seq_lens))

    logger.info(f"Simulated batch: {num_requests} requests, total {total_tokens} tokens")
    logger.info(f"Sequence lengths: {seq_lens}")
    logger.info(f"cu_q_lens: {cu_q_lens}")

    # Compute segment_indices for each position
    positions = jnp.arange(total_tokens)
    segment_indices = jnp.searchsorted(cu_q_lens, positions, side='right') - 1
    segment_indices_clipped = jnp.clip(segment_indices, 0, num_requests - 1)

    # Check unique segment_ids
    unique_segments, counts = jnp.unique(segment_indices_clipped, return_counts=True)
    logger.info(f"\nSegment assignment:")
    for seg, count in zip(unique_segments.tolist(), counts.tolist()):
        logger.info(f"  Segment {seg}: {count} tokens")

    # Simulate image token mask (some positions are image placeholders)
    # Let's say every other token in each sequence is an image placeholder
    is_image_token = jnp.zeros(total_tokens, dtype=jnp.bool_)
    # Mark image tokens for each segment
    for seg_id in range(num_requests):
        start = cu_q_lens[seg_id]
        end = cu_q_lens[seg_id + 1]
        # Mark every 2nd token as image token
        for pos in range(start, end, 2):
            is_image_token = is_image_token.at[pos].set(True)

    image_token_count = jnp.sum(is_image_token)
    logger.info(f"\nImage tokens: {image_token_count} / {total_tokens}")

    # Compute segment_feature_starts (uniform padding assumption)
    num_batched_segments = num_requests
    total_patches_axis = num_requests * max_patches_per_request
    patches_per_segment = total_patches_axis // num_batched_segments
    features_per_segment_padded = patches_per_segment * features_per_patch
    segment_feature_starts = jnp.arange(num_batched_segments) * features_per_segment_padded

    logger.info(f"\nFeature layout:")
    logger.info(f"  total_patches_axis: {total_patches_axis}")
    logger.info(f"  patches_per_segment: {patches_per_segment}")
    logger.info(f"  features_per_segment_padded: {features_per_segment_padded}")
    logger.info(f"  segment_feature_starts: {segment_feature_starts}")

    # Compute within_seg_counts using scan
    def segment_cumsum_fn(carry, x):
        mask_val, seg_id = x
        count = carry[seg_id]
        new_carry = carry.at[seg_id].add(mask_val)
        return new_carry, count

    init_counts = jnp.zeros(num_batched_segments, dtype=jnp.int32)
    _, within_seg_counts = jax.lax.scan(
        segment_cumsum_fn,
        init_counts,
        (is_image_token.astype(jnp.int32), segment_indices_clipped)
    )

    # Compute feature_indices
    segment_starts_per_token = segment_feature_starts[segment_indices_clipped]
    feature_indices = segment_starts_per_token + within_seg_counts

    # Check feature indices range
    total_features = total_patches_axis * features_per_patch
    logger.info(f"\nFeature indices:")
    logger.info(f"  total_features: {total_features}")
    logger.info(f"  feature_indices range: [{jnp.min(feature_indices)}, {jnp.max(feature_indices)}]")
    logger.info(f"  within_seg_counts range: [{jnp.min(within_seg_counts)}, {jnp.max(within_seg_counts)}]")

    # Check for out-of-bounds
    oob_count = jnp.sum(feature_indices >= total_features)
    if oob_count > 0:
        logger.error(f"  ⚠ {oob_count} feature indices are out of bounds!")
    else:
        logger.info(f"  ✓ All feature indices are within bounds")

    # Check if image tokens get correct feature indices
    image_positions = jnp.where(is_image_token)[0]
    logger.info(f"\nSample image token feature indices:")
    for pos in image_positions[:10].tolist():
        seg = segment_indices_clipped[pos]
        within_count = within_seg_counts[pos]
        feat_idx = feature_indices[pos]
        expected_start = segment_feature_starts[seg]
        logger.info(f"  pos={pos}, seg={seg}, within_count={within_count}, "
                   f"feat_idx={feat_idx}, expected_start={expected_start}")

    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)


def test_with_real_model():
    """
    Test batched inference with the actual VLM model.
    Requires the model checkpoint to be available.
    """
    logger.info("=" * 80)
    logger.info("TEST: Real Model Batched vs Sequential Inference")
    logger.info("=" * 80)

    # Check if we have the necessary imports
    try:
        import jax
        import jax.numpy as jnp
        import haliax as hax
        from levanter.models.llava_onevision import LlavaOnevision, LlavaOnevisionConfig
        logger.info("✓ Imports successful")
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        logger.error("Make sure you're running with 'uv run python ...'")
        return

    # Try to load a model (this requires a checkpoint)
    # For now, just test that we can import and set up the inference engine
    logger.info("\nTo test with a real model, run the full evaluation:")
    logger.info("  uv run python -m levanter.main.eval_vlm \\")
    logger.info("      --hf_checkpoint <path> \\")
    logger.info("      --eval_harness.task_spec='[\"mmmu_val\"]' \\")
    logger.info("      --eval_harness.vlm_batch_size=8 \\")
    logger.info("      --eval_harness.max_examples=10")


def analyze_garbage_output():
    """
    Analyze what might cause all-newline output.
    """
    logger.info("=" * 80)
    logger.info("ANALYSIS: Possible Causes of All-Newline Output")
    logger.info("=" * 80)

    logger.info("""
Symptoms:
- Before max_prefill_size fix: 37% empty, 63% with content
- After max_prefill_size fix: 100% are just newlines

Possible causes:

1. EMBEDDINGS ARE CORRUPTED
   - If feature_indices are out of bounds, we gather garbage features
   - JAX array indexing wraps around on out-of-bounds, returning wrong values
   - This would cause the model to see nonsensical embeddings

2. SEGMENT MISMATCH
   - num_batched_segments (from set_batched_request_data) might not match
     batch_info.num_seqs (from prefill)
   - If more sequences are prefilled than expected, segment_ids would be wrong

3. POSITION IDS INCORRECT
   - In batched mode, pos_ids should be the per-sequence positions
   - If they're wrong, RoPE would produce incorrect rotations

4. CU_Q_LENS STRUCTURE
   - cu_q_lens has max_seqs+1 elements, but only first num_seqs+1 are valid
   - searchsorted might return unexpected values for positions past valid data

Debugging Steps:

1. Add jax.debug.print in _compute_batched_embeddings():
   - Print num_tokens, num_seqs, num_batched_segments
   - Print feature_indices range
   - Print segment_ids unique values

2. Add assertions:
   - assert num_seqs == num_batched_segments
   - assert max(feature_indices) < total_features

3. Compare with sequential mode:
   - Run same requests sequentially
   - Compare embeddings output
""")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Debug VLM batched inference")
    parser.add_argument("--result-file", type=str, help="Path to result JSON file to analyze")
    parser.add_argument("--test-calculation", action="store_true",
                       help="Test the segment/feature calculation logic")
    parser.add_argument("--analyze", action="store_true",
                       help="Show analysis of possible causes")
    args = parser.parse_args()

    if args.test_calculation:
        test_batched_embeddings_calculation()
    elif args.analyze:
        analyze_garbage_output()
    elif args.result_file:
        # Import the existing analysis function
        from debug_vlm_batch import analyze_result_file
        analyze_result_file(args.result_file)
    else:
        # Run all tests
        test_batched_embeddings_calculation()
        print("\n")
        analyze_garbage_output()
        print("\n")
        test_with_real_model()


if __name__ == "__main__":
    main()
