# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for VLM (Vision-Language Model) Sequence Packing.

Tests cover:
- Basic packing of single-image samples
- Variable length sample packing
- Multi-image sample packing
- Segment ID assignment
- Position ID computation
- Pack assignment determinism
- End-to-end tests with real data (HF vs Levanter packed loss comparison)
"""

import asyncio
import tempfile
from typing import Dict, List, Sequence

import numpy as np
import pytest

from levanter.data.vlm_packing import (
    VLMPackerConfig,
    VLMPrepackedDataset,
)

# Import test utilities for real data tests
from test_image_utils import (
    get_real_data,
    prepare_test_data,
    QWEN3_TOKENIZER,
    SINGLE_PATCH_GRID_PINPOINTS,
)
from test_utils import skip_if_no_torch


class MockAsyncDataset:
    """Mock AsyncDataset for testing VLMPrepackedDataset."""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    async def async_len(self) -> int:
        return len(self.samples)

    async def get_batch(self, indices: Sequence[int]) -> List[Dict]:
        return [self.samples[i] for i in indices]

    def is_finite(self) -> bool:
        return True


def create_mock_sample(
    input_ids: List[int],
    pixel_values: np.ndarray = None,
    attention_mask: List[int] = None,
    loss_mask: List[float] = None,
    grid_mask: List[bool] = None,
) -> Dict:
    """Create a mock ImageTextDict sample for testing."""
    sample = {
        "input_ids": np.array(input_ids, dtype=np.int32),
    }

    if pixel_values is not None:
        sample["pixel_values"] = pixel_values

    if attention_mask is not None:
        sample["attention_mask"] = np.array(attention_mask, dtype=np.int32)

    if loss_mask is not None:
        sample["loss_mask"] = np.array(loss_mask, dtype=np.float32)

    if grid_mask is not None:
        sample["grid_mask"] = np.array(grid_mask, dtype=bool)

    return sample


# =============================================================================
# End-to-End Tests with Real Data
# =============================================================================
# These tests use real VLM data from HuggingFace and compare:
# - HF model processing samples individually (golden reference)
# - Levanter with samples packed together
# Verifies that packing does not affect loss computation correctness.
# =============================================================================

MODEL_NAME = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
MAX_LENGTH = 8192


@pytest.mark.entry
@skip_if_no_torch
def test_vlm_prepacked_dataset_loss_consistency():
    """
    Test VLMPrepackedDataset loss consistency with HF model.

    This test verifies that when multiple samples are packed together,
    the per-segment loss matches the HF single-sample loss.

    Steps:
    1. Load real data from HuggingFace
    2. Create VLMPrepackedDataset (in-memory packing)
    3. Get a pack containing multiple samples
    4. Compute HF loss for each sample individually (golden reference)
    5. Compute Levanter packed loss per segment
    6. Compare: each segment's loss should match corresponding HF sample's loss
    """
    import torch
    import transformers.models.llava_onevision.modeling_llava_onevision as llava_modeling
    from transformers import AutoModelForVision2Seq, AutoTokenizer
    import haliax as hax
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from levanter.models.llava_onevision import LlavaOnevisionModel
    from levanter.trainer import TrainerConfig
    from levanter.main.train_vlm import compute_vlm_loss
    from test_train_image_anyres import _load_levanter_config

    # Set JAX to use float32 matmul precision
    jax.config.update("jax_default_matmul_precision", "float32")

    grid_pinpoints = SINGLE_PATCH_GRID_PINPOINTS
    num_samples = 2  # Pack 2 samples together

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Load real data
        hf_dataset = get_real_data(num_samples=num_samples)
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        # Prepare test data pairs (HF format + Levanter format)
        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=list(range(num_samples)),
            model_name=MODEL_NAME,
            max_length=MAX_LENGTH,
            max_num_patches=1,  # disable_anyres mode
            grid_pinpoints=grid_pinpoints,
            disable_anyres=True,
        )

    # Load tokenizer
    qwen3_tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)
    image_token_id = qwen3_tokenizer.convert_tokens_to_ids("<|image_pad|>")

    # ==================== HF: Compute golden losses ====================
    hf_model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    hf_model.model.config.image_grid_pinpoints = grid_pinpoints
    hf_model.model.config.vision_aspect_ratio = "single"
    hf_model.model.image_newline = None
    hf_model.config.image_token_index = image_token_id
    hf_model.eval()

    # Monkey-patch for disable_anyres
    original_image_size_to_num_patches = llava_modeling.image_size_to_num_patches
    llava_modeling.image_size_to_num_patches = lambda *args, **kwargs: 1

    hf_golden_losses = []
    try:
        for i, pair in enumerate(test_pairs):
            lev_seq_len = len(pair.lev.input_ids)
            hf_input_ids = torch.from_numpy(np.array(pair.lev.input_ids[:lev_seq_len])).unsqueeze(0)
            hf_pixel_values = torch.from_numpy(pair.hf.pixel_values).unsqueeze(0)
            hf_image_sizes = torch.from_numpy(pair.hf.image_sizes).unsqueeze(0)

            if hf_pixel_values.dim() == 5:
                hf_pixel_values = hf_pixel_values[:, 0:1, :, :, :]

            hf_labels = hf_input_ids.clone().long()
            seq_len = hf_input_ids.shape[1]
            loss_mask_np = np.array(pair.lev.loss_mask)[:seq_len]
            mask_tensor = torch.from_numpy(loss_mask_np).unsqueeze(0)
            hf_labels[mask_tensor == 0] = -100

            with torch.no_grad():
                hf_output = hf_model(
                    input_ids=hf_input_ids,
                    pixel_values=hf_pixel_values,
                    image_sizes=hf_image_sizes,
                    labels=hf_labels,
                )
                hf_golden_losses.append(hf_output.loss.item())
    finally:
        llava_modeling.image_size_to_num_patches = original_image_size_to_num_patches

    print(f"\n=== HF Golden Losses ===")
    for i, loss in enumerate(hf_golden_losses):
        print(f"  Sample {i}: {loss:.6f}")

    # ==================== Levanter: Pack samples and compute loss ====================
    # Create VLMPrepackedDataset with both samples
    samples_for_packing = [
        create_mock_sample(
            input_ids=pair.lev.input_ids.tolist(),
            pixel_values=pair.lev.pixel_values,
            attention_mask=pair.lev.attention_mask.tolist(),
            loss_mask=pair.lev.loss_mask.tolist(),
            grid_mask=pair.lev.grid_mask.tolist(),
        )
        for pair in test_pairs
    ]
    mock_dataset = MockAsyncDataset(samples_for_packing)

    config = VLMPackerConfig(
        max_length=MAX_LENGTH,
        max_patches=num_samples,  # 1 patch per sample
        max_segments=num_samples,
        features_per_patch=576,
    )

    packed_dataset = VLMPrepackedDataset(
        base_dataset=mock_dataset,
        config=config,
    )

    # Get the packed sample
    packed_batch = asyncio.get_event_loop().run_until_complete(packed_dataset.get_batch([0]))
    packed = packed_batch[0]

    print(f"\n=== Packed Sample Info ===")
    print(f"  num_segments: {packed['num_segments']}")
    print(f"  segment_ids unique: {np.unique(packed['segment_ids'])}")
    print(f"  input_ids shape: {packed['input_ids'].shape}")

    # Verify packing worked correctly
    assert packed['num_segments'] == num_samples, f"Expected {num_samples} segments, got {packed['num_segments']}"

    # Verify segment IDs are assigned correctly
    segment_ids = packed['segment_ids']
    for seg_id in range(num_samples):
        seg_mask = segment_ids == seg_id
        assert seg_mask.sum() > 0, f"Segment {seg_id} has no tokens"

    # ==================== Load Levanter model and compute per-segment loss ====================
    lev_config = _load_levanter_config(MODEL_NAME, enable_flash_attention=False, gradient_checkpointing=False)
    lev_config = lev_config.with_token_ids(image_token_id=image_token_id)
    trainer_config = TrainerConfig()

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=MODEL_NAME)
        lev_model = converter.load_pretrained(
            LlavaOnevisionModel,
            ref=MODEL_NAME,
            config=lev_config,
            axis_mapping=trainer_config.parameter_axis_mapping,
            dtype=jnp.float32,
            resize_vocab_to_match_tokenizer=False,
        )

        # Create JAX tensors from packed sample
        Batch = hax.Axis("batch", 1)
        Position = hax.Axis("position", len(packed['input_ids']))
        NumPatches = hax.Axis("num_patches", packed['pixel_values'].shape[0])
        Channels = hax.Axis("channels", 3)
        Height = hax.Axis("height", 384)
        Width = hax.Axis("width", 384)
        GridMask = hax.Axis("grid_mask", packed['pixel_values'].shape[0])

        input_ids = hax.named(jnp.array(packed['input_ids']).reshape(1, -1), (Batch, Position))
        pixel_values = hax.named(
            jnp.array(packed['pixel_values']).reshape(1, -1, 3, 384, 384),
            (Batch, NumPatches, Channels, Height, Width)
        )
        grid_mask = hax.named(
            jnp.array(packed['image_segment_ids'] >= 0).reshape(1, -1),
            (Batch, GridMask)
        )
        loss_mask = hax.named(jnp.array(packed['loss_mask']).reshape(1, -1), (Batch, Position))
        segment_ids_jax = hax.named(jnp.array(packed['segment_ids']).reshape(1, -1), (Batch, Position))
        position_ids = hax.named(jnp.array(packed['position_ids']).reshape(1, -1), (Batch, Position))

        from levanter.data.image import ImageTextExample as ImgTextEx

        batch_example = ImgTextEx(
            pixel_values=pixel_values,
            input_ids=input_ids,
            loss_mask=loss_mask,
            grid_mask=grid_mask,
            unpad_indices=None,
            combined_mask=None,
            position_ids=position_ids,
        )

        # Compute per-token loss
        @eqx.filter_jit
        def compute_per_token_loss(model, example):
            from levanter.models.loss import fused_cross_entropy_loss_and_logsumexp_penalty

            grid_mask = getattr(example, "grid_mask", None)
            position_ids = getattr(example, "position_ids", None)

            activations, lm_head = model.forward_with_activations(
                example.input_ids,
                pixel_values=example.pixel_values,
                grid_mask=grid_mask,
                unpad_indices=None,
                combined_mask=None,
                position_ids=position_ids,
                key=None,
            )

            Pos = example.input_ids.resolve_axis("position")
            Embed = model.config.TextEmbed
            Vocab = model.Vocab

            targets = hax.roll(example.input_ids, -1, Pos)

            per_token_loss = fused_cross_entropy_loss_and_logsumexp_penalty(
                pred_embeddings=activations,
                pred_lm_head=lm_head,
                Contract=Embed,
                Label=Vocab,
                target_y=targets,
                reduction=None,
                weight=None,
                logsumexp_weight=0.0,
                block_size=4096,
            )

            return per_token_loss

        per_token_loss = compute_per_token_loss(lev_model, batch_example)
        per_token_loss_np = np.array(per_token_loss.array)[0]  # (seq_len,)

        # Compute per-segment loss
        segment_ids_np = packed['segment_ids']
        loss_mask_np = packed['loss_mask']
        shifted_loss_mask = np.roll(loss_mask_np, -1)
        shifted_loss_mask[-1] = 0

        lev_segment_losses = []
        for seg_id in range(num_samples):
            seg_mask = (segment_ids_np == seg_id) & (shifted_loss_mask > 0)
            if seg_mask.sum() > 0:
                seg_loss = per_token_loss_np[seg_mask].mean()
                lev_segment_losses.append(float(seg_loss))
            else:
                lev_segment_losses.append(0.0)

        print(f"\n=== Levanter Per-Segment Losses ===")
        for i, loss in enumerate(lev_segment_losses):
            print(f"  Segment {i}: {loss:.6f}")

        # ==================== Compare losses ====================
        print(f"\n=== Loss Comparison ===")
        all_passed = True
        for i in range(num_samples):
            hf_loss = hf_golden_losses[i]
            lev_loss = lev_segment_losses[i]
            diff = abs(hf_loss - lev_loss)
            rel_diff = diff / hf_loss if hf_loss != 0 else diff

            status = "PASS" if rel_diff < 0.01 else "FAIL"
            print(f"  Sample {i}: HF={hf_loss:.6f}, Lev={lev_loss:.6f}, rel_diff={rel_diff*100:.2f}% [{status}]")

            if rel_diff >= 0.01:
                all_passed = False

        assert all_passed, "Per-segment losses do not match HF golden losses!"
        print("\n VLMPrepackedDataset loss consistency test passed!")


@pytest.mark.entry
@skip_if_no_torch
def test_position_ids_reset_with_real_data():
    """
    Test position ID reset at segment boundaries using real data.

    Verifies:
    - Segment 0: positions [0, 1, 2, ...]
    - Segment 1: positions [0, 1, 2, ...] (reset from 0)
    """
    num_samples = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_dataset = get_real_data(num_samples=num_samples)
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=list(range(num_samples)),
            model_name=MODEL_NAME,
            max_length=MAX_LENGTH,
            max_num_patches=1,
            grid_pinpoints=SINGLE_PATCH_GRID_PINPOINTS,
            disable_anyres=True,
        )

    samples = [
        create_mock_sample(
            input_ids=pair.lev.input_ids.tolist(),
            pixel_values=pair.lev.pixel_values,
            attention_mask=pair.lev.attention_mask.tolist(),
            loss_mask=pair.lev.loss_mask.tolist(),
            grid_mask=pair.lev.grid_mask.tolist(),
        )
        for pair in test_pairs
    ]
    mock_dataset = MockAsyncDataset(samples)

    config = VLMPackerConfig(
        max_length=MAX_LENGTH,
        max_patches=num_samples,
        max_segments=num_samples,
        features_per_patch=576,
    )

    packed_dataset = VLMPrepackedDataset(
        base_dataset=mock_dataset,
        config=config,
    )

    packed_batch = asyncio.get_event_loop().run_until_complete(packed_dataset.get_batch([0]))
    packed = packed_batch[0]

    segment_ids = packed['segment_ids']
    position_ids = packed['position_ids']

    print(f"\n=== Position ID Verification ===")

    for seg_id in range(num_samples):
        seg_mask = segment_ids == seg_id
        seg_positions = position_ids[seg_mask]

        # First position should be 0
        assert seg_positions[0] == 0, f"Segment {seg_id} first position should be 0, got {seg_positions[0]}"

        # Positions should be consecutive
        expected_positions = np.arange(len(seg_positions))
        np.testing.assert_array_equal(
            seg_positions, expected_positions,
            err_msg=f"Segment {seg_id} positions not consecutive"
        )

        print(f"  Segment {seg_id}: positions 0-{len(seg_positions)-1} (length={len(seg_positions)})")

    print("\n Position ID reset verification passed!")


@pytest.mark.entry
@skip_if_no_torch
def test_pack_assignment_from_parquet():
    """
    Test compute_pack_assignments() correctly computes pack assignments from parquet.

    Verifies:
    - shard_info correctly records each shard's position
    - assignments sample indices are contiguous and cover all samples
    - JSON save/load roundtrip preserves data
    """
    from transformers import AutoTokenizer
    from levanter.data.vlm_packing import (
        compute_pack_assignments,
        load_pack_assignment_result,
        PackAssignmentConfig,
    )

    num_samples = 4

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Load real data and save to parquet
        hf_dataset = get_real_data(num_samples=num_samples)
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        # 2. Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)

        # 3. Create config
        config = PackAssignmentConfig(
            max_length=2048,
            max_patches=10,
            max_segments=64,
            features_per_patch=576,
            image_column="images",
            text_column="messages",
        )

        # 4. Compute pack assignments
        output_file = f"{tmpdir}/pack_assignments.json"
        result = compute_pack_assignments(
            parquet_paths=[parquet_path],
            output_file=output_file,
            tokenizer=tokenizer,
            config=config,
        )

        print(f"\n=== Pack Assignment Results ===")
        print(f"  Total samples: {result.num_samples}")
        print(f"  Total packs: {result.num_packs}")
        print(f"  Number of shards: {len(result.shard_info)}")
        print(f"  Compression ratio: {result.num_samples / result.num_packs:.2f}x")

        # 5. Verify shard_info
        assert len(result.shard_info) == 1, "Should have 1 shard"
        shard = result.shard_info[0]
        assert shard.path == parquet_path, "Shard path mismatch"
        assert shard.start_idx == 0, "First shard should start at 0"
        assert shard.num_samples == num_samples, f"Expected {num_samples} samples, got {shard.num_samples}"

        # 6. Verify assignments cover all samples
        all_sample_indices = set()
        for start, end in result.assignments:
            for idx in range(start, end):
                all_sample_indices.add(idx)

        expected_indices = set(range(num_samples))
        assert all_sample_indices == expected_indices, f"Assignments don't cover all samples: {all_sample_indices} vs {expected_indices}"

        # 7. Verify JSON roundtrip
        loaded_result = load_pack_assignment_result(output_file)
        assert loaded_result.num_samples == result.num_samples
        assert loaded_result.num_packs == result.num_packs
        assert loaded_result.assignments == result.assignments

        print("\n Pack assignment from parquet test passed!")


@pytest.mark.entry
@skip_if_no_torch
def test_loss_mask_per_segment_with_real_data():
    """
    Test loss mask correctness per segment using real data.

    Verifies:
    - Only assistant response tokens have loss_mask == 1
    - User prompt / image tokens have loss_mask == 0
    """
    num_samples = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_dataset = get_real_data(num_samples=num_samples)
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=list(range(num_samples)),
            model_name=MODEL_NAME,
            max_length=MAX_LENGTH,
            max_num_patches=1,
            grid_pinpoints=SINGLE_PATCH_GRID_PINPOINTS,
            disable_anyres=True,
        )

    samples = [
        create_mock_sample(
            input_ids=pair.lev.input_ids.tolist(),
            pixel_values=pair.lev.pixel_values,
            attention_mask=pair.lev.attention_mask.tolist(),
            loss_mask=pair.lev.loss_mask.tolist(),
            grid_mask=pair.lev.grid_mask.tolist(),
        )
        for pair in test_pairs
    ]
    mock_dataset = MockAsyncDataset(samples)

    config = VLMPackerConfig(
        max_length=MAX_LENGTH,
        max_patches=num_samples,
        max_segments=num_samples,
        features_per_patch=576,
    )

    packed_dataset = VLMPrepackedDataset(
        base_dataset=mock_dataset,
        config=config,
    )

    packed_batch = asyncio.get_event_loop().run_until_complete(packed_dataset.get_batch([0]))
    packed = packed_batch[0]

    segment_ids = packed['segment_ids']
    loss_mask = packed['loss_mask']

    print(f"\n=== Loss Mask Verification ===")

    for seg_id in range(num_samples):
        seg_mask = segment_ids == seg_id
        seg_loss_mask = loss_mask[seg_mask]

        # Original sample's loss mask
        original_loss_mask = np.array(test_pairs[seg_id].lev.loss_mask)
        # Truncate to actual tokens (remove padding from original)
        original_valid_len = int(np.sum(test_pairs[seg_id].lev.attention_mask))
        original_loss_mask = original_loss_mask[:original_valid_len]

        # Compare
        np.testing.assert_array_equal(
            seg_loss_mask[:len(original_loss_mask)],
            original_loss_mask,
            err_msg=f"Segment {seg_id} loss mask mismatch"
        )

        valid_loss_tokens = (seg_loss_mask > 0).sum()
        total_tokens = len(seg_loss_mask)
        print(f"  Segment {seg_id}: {valid_loss_tokens}/{total_tokens} tokens have loss_mask=1")

    print("\n Loss mask verification passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
