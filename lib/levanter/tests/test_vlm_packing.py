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
    PackedVLMDataset,
    compute_pack_assignments,
    PackAssignmentConfig,
)

# Import test utilities for real data tests
from test_image_utils import (
    get_real_data,
    get_interleaved_data,
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


def create_streaming_packed_dataset(
    parquet_path: str,
    tmpdir: str,
    processor,
    tokenizer,
    max_length: int = 2048,
    max_patches: int = 10,
    max_segments: int = 64,
    features_per_patch: int = 729,
    # BatchImageProcessor configuration
    disable_anyres: bool = False,
    grid_pinpoints=None,
    vision_feature_height: int = 27,
    patch_size: int = 384,
    pad_token_id: int = 151643,
) -> PackedVLMDataset:
    """
    Helper to create a streaming PackedVLMDataset from parquet file.

    This is the recommended way to test VLM packing - it uses the same
    streaming pipeline as production training.

    Steps:
    1. Compute pack assignments from parquet
    2. Create PackedVLMDataset with assignments and processor

    Args:
        parquet_path: Path to parquet file with raw data
        tmpdir: Temporary directory to store pack assignments
        processor: HuggingFace processor for image processing
        tokenizer: HuggingFace tokenizer for text tokenization
        max_length: Maximum sequence length for packed samples
        max_patches: Maximum number of image patches per pack
        max_segments: Maximum number of samples per pack
        features_per_patch: Number of features per image patch (27*27=729 for llava-onevision)
        disable_anyres: If True, disable anyres processing (use single resolution)
        grid_pinpoints: Grid resolutions for anyres processing
        vision_feature_height: Vision encoder output tokens per spatial dim (e.g., 27 for 384/14)
        patch_size: Size of each image patch (default 384)

    Returns:
        PackedVLMDataset configured for streaming packing
    """
    # Compute pack assignments
    config = PackAssignmentConfig(
        max_length=max_length,
        max_patches=max_patches,
        max_segments=max_segments,
        features_per_patch=features_per_patch,
        image_column="images",
        text_column="messages",
    )

    output_file = f"{tmpdir}/pack_assignments.json"
    compute_pack_assignments(
        parquet_paths=[parquet_path],
        output_file=output_file,
        tokenizer=tokenizer,
        config=config,
        processor=processor,  # Pass processor for chat template on conversation format
    )

    # Create streaming dataset with full BatchImageProcessor configuration
    return PackedVLMDataset(
        pack_assignments_file=output_file,
        processor=processor,
        max_length=max_length,
        tokenizer=tokenizer,
        disable_anyres=disable_anyres,
        grid_pinpoints=grid_pinpoints,
        vision_feature_height=vision_feature_height,
        max_num_patches=max_patches,
        patch_size=patch_size,
        pad_token_id=pad_token_id,
    )


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
def test_vlm_streaming_packed_dataset_loss_consistency():
    """
    Test PackedVLMDataset (streaming) loss consistency with HF model.

    This test verifies that when multiple samples are packed together using
    the streaming PackedVLMDataset, the per-segment loss matches the HF single-sample loss.

    Steps:
    1. Load real data from HuggingFace and save to parquet
    2. Compute pack assignments using compute_pack_assignments()
    3. Create PackedVLMDataset (streaming packing with pre-computed assignments)
    4. Compute HF loss for each sample individually (golden reference)
    5. Compute Levanter packed loss per segment
    6. Compare: each segment's loss should match corresponding HF sample's loss
    """
    import torch
    import transformers.models.llava_onevision.modeling_llava_onevision as llava_modeling
    from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
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

    # Load tokenizer and processor upfront
    qwen3_tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)
    image_token_id = qwen3_tokenizer.convert_tokens_to_ids("<|image_pad|>")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Load real data and save to parquet
        hf_dataset = get_real_data(num_samples=num_samples)
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        # Prepare test data pairs (HF format + Levanter format) for HF golden loss computation
        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=list(range(num_samples)),
            model_name=MODEL_NAME,
            max_length=MAX_LENGTH,
            max_num_patches=1,  # disable_anyres mode
            grid_pinpoints=grid_pinpoints,
            disable_anyres=True,
        )

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

        # ==================== Levanter: Create streaming packed dataset ====================
        # Use PackedVLMDataset with pre-computed pack assignments (streaming mode)
        # In disable_anyres mode, each sample has exactly 1 patch (base patch only)
        # So max_patches = num_samples (one patch per sample)
        max_patches = num_samples  # 2 samples with 1 patch each = 2 patches

        packed_dataset = create_streaming_packed_dataset(
            parquet_path=parquet_path,
            tmpdir=tmpdir,
            processor=processor,
            tokenizer=qwen3_tokenizer,
            max_length=MAX_LENGTH,
            max_patches=max_patches,
            max_segments=num_samples,
            features_per_patch=729,  # 27*27 for llava-onevision (384/14 ≈ 27)
            disable_anyres=True,
            grid_pinpoints=grid_pinpoints,
            vision_feature_height=27,
        )

        # Get the packed sample
        packed_batch = asyncio.get_event_loop().run_until_complete(packed_dataset.get_batch([0]))
        packed = packed_batch[0]

        print(f"\n=== Packed Sample Info (Streaming) ===")
        print(f"  num_segments: {packed['num_segments']}")
        print(f"  segment_ids unique: {np.unique(packed['segment_ids'])}")
        print(f"  input_ids shape: {packed['input_ids'].shape}")
        print(f"  pixel_values shape: {packed['pixel_values'].shape}")
        print(f"  image_segment_ids: {packed['image_segment_ids']}")

        # Verify packing worked correctly
        assert packed['num_segments'] == num_samples, f"Expected {num_samples} segments, got {packed['num_segments']}"

        # Verify segment IDs are assigned correctly
        segment_ids = packed['segment_ids']
        for seg_id in range(num_samples):
            seg_mask = segment_ids == seg_id
            assert seg_mask.sum() > 0, f"Segment {seg_id} has no tokens"

        # ==================== Load Levanter model and compute per-segment loss ====================
        # IMPORTANT: Must use vision_aspect_ratio="single" and image_grid_pinpoints for disable_anyres mode
        lev_config = _load_levanter_config(
            MODEL_NAME,
            enable_flash_attention=False,
            gradient_checkpointing=False,
            vision_aspect_ratio="single",
            image_grid_pinpoints=grid_pinpoints,
        )
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
            image_segment_ids_jax = hax.named(jnp.array(packed['image_segment_ids']).reshape(1, -1), (Batch, GridMask))

            # Use combined_mask from packed data (computed by data pipeline)
            # This is required for the model to use precomputed position_ids
            combined_mask = hax.named(jnp.array(packed['combined_mask']).reshape(1, -1), (Batch, Position))

            from levanter.data.image import ImageTextExample as ImgTextEx

            batch_example = ImgTextEx(
                pixel_values=pixel_values,
                input_ids=input_ids,
                loss_mask=loss_mask,
                grid_mask=grid_mask,
                unpad_indices=None,
                combined_mask=combined_mask,  # Pass combined_mask to enable precomputed position_ids
                position_ids=position_ids,
            )

            # Compute per-token loss
            # NOTE: segment_ids and image_segment_ids are needed for packing mode
            # to correctly map image tokens to per-segment image features
            @eqx.filter_jit
            def compute_per_token_loss(model, example, segment_ids, image_segment_ids):
                from levanter.models.loss import fused_cross_entropy_loss_and_logsumexp_penalty

                grid_mask = getattr(example, "grid_mask", None)
                position_ids = getattr(example, "position_ids", None)
                combined_mask = getattr(example, "combined_mask", None)

                # Packing mode: pass segment_ids for attention masking and image_segment_ids for image indexing
                # IMPORTANT: combined_mask must be provided along with position_ids to use precomputed values
                activations, lm_head = model.forward_with_activations(
                    example.input_ids,
                    pixel_values=example.pixel_values,
                    grid_mask=grid_mask,
                    unpad_indices=None,
                    combined_mask=combined_mask,  # Pass combined_mask to use precomputed position_ids
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    image_segment_ids=image_segment_ids,
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

            per_token_loss = compute_per_token_loss(lev_model, batch_example, segment_ids_jax, image_segment_ids_jax)
            per_token_loss_np = np.array(per_token_loss.array)[0]  # (seq_len,)

            # Compute per-segment loss
            # HF convention: labels[i+1] controls whether loss at position i is included
            # (shift_labels[i] = labels[i+1], ignore if == -100)
            # So we use shifted_loss_mask: include position i if loss_mask[i+1] > 0
            segment_ids_np = packed['segment_ids']
            loss_mask_np = packed['loss_mask']
            shifted_loss_mask = np.roll(loss_mask_np, -1)
            shifted_loss_mask[-1] = 0  # Last position has no valid target

            lev_segment_losses = []
            for seg_id in range(num_samples):
                # Include position i if: in this segment AND loss_mask[i+1] > 0
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

                # Strict threshold: 0.02% to ensure numerical consistency
                status = "PASS" if rel_diff < 0.0002 else "FAIL"
                print(f"  Sample {i}: HF={hf_loss:.6f}, Lev={lev_loss:.6f}, rel_diff={rel_diff*100:.4f}% [{status}]")

                if rel_diff >= 0.0002:
                    all_passed = False

            assert all_passed, "Per-segment losses do not match HF golden losses!"
            print("\n PackedVLMDataset (streaming) loss consistency test passed!")


@pytest.mark.entry
@skip_if_no_torch
def test_position_ids_reset_with_real_data():
    """
    Test position ID reset at segment boundaries using real data.

    Uses PackedVLMDataset (streaming) with pre-computed pack assignments.

    Verifies:
    - Segment 0: positions [0, 1, 2, ...]
    - Segment 1: positions [0, 1, 2, ...] (reset from 0)
    """
    from transformers import AutoTokenizer, AutoProcessor

    num_samples = 2

    # Load tokenizer and processor upfront
    qwen3_tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_dataset = get_real_data(num_samples=num_samples)
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        # Get test_pairs for computing max_patches
        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=list(range(num_samples)),
            model_name=MODEL_NAME,
            max_length=MAX_LENGTH,
            max_num_patches=1,
            grid_pinpoints=SINGLE_PATCH_GRID_PINPOINTS,
            disable_anyres=True,
        )

        # In disable_anyres mode, each sample has exactly 1 patch (base patch only)
        max_patches = num_samples  # 2 samples with 1 patch each = 2 patches

        # Create streaming packed dataset
        packed_dataset = create_streaming_packed_dataset(
            parquet_path=parquet_path,
            tmpdir=tmpdir,
            processor=processor,
            tokenizer=qwen3_tokenizer,
            max_length=MAX_LENGTH,
            max_patches=max_patches,
            max_segments=num_samples,
            features_per_patch=729,  # 27*27 for llava-onevision (384/14 ≈ 27)
            disable_anyres=True,
            grid_pinpoints=SINGLE_PATCH_GRID_PINPOINTS,
            vision_feature_height=27,
        )

        packed_batch = asyncio.get_event_loop().run_until_complete(packed_dataset.get_batch([0]))
        packed = packed_batch[0]

        segment_ids = packed['segment_ids']
        position_ids = packed['position_ids']

        print(f"\n=== Position ID Verification (Streaming) ===")

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
            features_per_patch=729,  # 27*27 for llava-onevision (384/14 ≈ 27)
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

    Uses PackedVLMDataset (streaming) with pre-computed pack assignments.

    Verifies:
    - Only assistant response tokens have loss_mask == 1
    - User prompt / image tokens have loss_mask == 0
    """
    from transformers import AutoTokenizer, AutoProcessor

    num_samples = 2

    # Load tokenizer and processor upfront
    qwen3_tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_dataset = get_real_data(num_samples=num_samples)
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        # Get test_pairs for computing max_patches and reference loss_mask
        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=list(range(num_samples)),
            model_name=MODEL_NAME,
            max_length=MAX_LENGTH,
            max_num_patches=1,
            grid_pinpoints=SINGLE_PATCH_GRID_PINPOINTS,
            disable_anyres=True,
        )

        # In disable_anyres mode, each sample has exactly 1 patch (base patch only)
        max_patches = num_samples  # 2 samples with 1 patch each = 2 patches

        # Create streaming packed dataset
        packed_dataset = create_streaming_packed_dataset(
            parquet_path=parquet_path,
            tmpdir=tmpdir,
            processor=processor,
            tokenizer=qwen3_tokenizer,
            max_length=MAX_LENGTH,
            max_patches=max_patches,
            max_segments=num_samples,
            features_per_patch=729,  # 27*27 for llava-onevision (384/14 ≈ 27)
            disable_anyres=True,
            grid_pinpoints=SINGLE_PATCH_GRID_PINPOINTS,
            vision_feature_height=27,
        )

        packed_batch = asyncio.get_event_loop().run_until_complete(packed_dataset.get_batch([0]))
        packed = packed_batch[0]

        segment_ids = packed['segment_ids']
        loss_mask = packed['loss_mask']

        print(f"\n=== Loss Mask Verification (Streaming) ===")

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


@pytest.mark.entry
@skip_if_no_torch
def test_vlm_batched_packing_loss_consistency_interleaved():
    """
    Test batched VLM packing with interleaved (multi-image) data.

    Uses PackedVLMDataset (streaming) with pre-computed pack assignments.

    This test:
    1. Loads interleaved data with 1-4 images per sample
    2. Creates multiple packs using streaming PackedVLMDataset
    3. Batches multiple packs together
    4. Computes HF loss for each sample individually (golden reference)
    5. Computes Levanter batched packed loss per segment
    6. Compares: weighted loss per segment should match HF sample loss
    """
    import torch
    import transformers.models.llava_onevision.modeling_llava_onevision as llava_modeling
    from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
    import haliax as hax
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from levanter.models.llava_onevision import LlavaOnevisionModel
    from levanter.trainer import TrainerConfig
    from test_train_image_anyres import _load_levanter_config

    # Set JAX to use float32 matmul precision
    jax.config.update("jax_default_matmul_precision", "float32")

    grid_pinpoints = SINGLE_PATCH_GRID_PINPOINTS
    num_samples = 4  # 4 samples total, will be packed into 2 packs of 2 samples each
    samples_per_pack = 2
    batch_size = num_samples // samples_per_pack  # 2 packs per batch

    # Load tokenizer and processor upfront
    qwen3_tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)
    image_token_id = qwen3_tokenizer.convert_tokens_to_ids("<|image_pad|>")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Load interleaved data (contains samples with 1-4 images)
        hf_dataset = get_interleaved_data(num_samples=num_samples)
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        # Get num_images for each sample to set max_num_patches
        num_images_per_sample = [len(hf_dataset[i]["images"]) for i in range(num_samples)]
        max_images = max(num_images_per_sample)

        # Prepare test data pairs (HF format + Levanter format) for HF golden loss computation
        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=list(range(num_samples)),
            model_name=MODEL_NAME,
            max_length=MAX_LENGTH,
            max_num_patches=max_images,  # One patch per image in disable_anyres
            grid_pinpoints=grid_pinpoints,
            disable_anyres=True,
            max_images_per_sample=max_images,
        )

        # ==================== HF: Compute golden losses for each sample ====================
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
                num_images = num_images_per_sample[i]
                lev_seq_len = len(pair.lev.input_ids)
                hf_input_ids = torch.from_numpy(np.array(pair.lev.input_ids[:lev_seq_len])).unsqueeze(0).long()
                hf_image_sizes = torch.from_numpy(pair.hf.image_sizes)

                # For multi-image disable_anyres: pixel_values shape is (num_images, C, H, W)
                hf_pixel_values = torch.from_numpy(pair.hf.pixel_values)
                if hf_pixel_values.dim() == 4 and num_images > 1:
                    # Multi-image: keep 4D format (num_images, C, H, W) for HF model
                    hf_pixel_values = hf_pixel_values[:num_images]
                else:
                    # Single image: add batch dimension
                    hf_pixel_values = hf_pixel_values.unsqueeze(0)

                # Create labels
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
                        batch_num_images=torch.tensor([num_images]),
                    )
                    hf_golden_losses.append(hf_output.loss.item())
        finally:
            llava_modeling.image_size_to_num_patches = original_image_size_to_num_patches

        print(f"\n=== HF Golden Losses (Interleaved) ===")
        for i, loss in enumerate(hf_golden_losses):
            print(f"  Sample {i} ({num_images_per_sample[i]} images): {loss:.6f}")

        # ==================== Create batched packed data using streaming ====================
        # In disable_anyres mode, each image contributes 1 patch
        # For interleaved data, max_patches = total images in all samples
        total_images = sum(num_images_per_sample)  # Total images across all samples
        max_patches = total_images  # Each image = 1 patch in disable_anyres mode

        # Create streaming packed dataset with max_segments=2 to get 2 samples per pack
        packed_dataset = create_streaming_packed_dataset(
            parquet_path=parquet_path,
            tmpdir=tmpdir,
            processor=processor,
            tokenizer=qwen3_tokenizer,
            max_length=MAX_LENGTH,
            max_patches=max_patches,
            max_segments=samples_per_pack,  # 2 samples per pack
            features_per_patch=729,  # 27*27 for llava-onevision
            disable_anyres=True,
            grid_pinpoints=grid_pinpoints,
            vision_feature_height=27,
        )

        # Get all packs (should be 2 packs with 2 samples each)
        all_packs = asyncio.get_event_loop().run_until_complete(
            packed_dataset.get_batch(list(range(batch_size)))
        )

        # Find the maximum dimensions across all packs for batching
        max_seq_len = max(len(p['input_ids']) for p in all_packs)
        max_patches = max(p['pixel_values'].shape[0] for p in all_packs)

        # Pad all packs to the same dimensions and stack into batch
        def pad_to_shape(arr, target_shape, pad_value=0):
            result = np.full(target_shape, pad_value, dtype=arr.dtype)
            slices = tuple(slice(0, s) for s in arr.shape)
            result[slices] = arr
            return result

        batched_input_ids = np.stack([
            pad_to_shape(p['input_ids'], (max_seq_len,), pad_value=0)
            for p in all_packs
        ])
        batched_pixel_values = np.stack([
            pad_to_shape(p['pixel_values'], (max_patches,) + all_packs[0]['pixel_values'].shape[1:], pad_value=0)
            for p in all_packs
        ])
        batched_loss_mask = np.stack([
            pad_to_shape(p['loss_mask'], (max_seq_len,), pad_value=0)
            for p in all_packs
        ])
        batched_segment_ids = np.stack([
            pad_to_shape(p['segment_ids'], (max_seq_len,), pad_value=-1)
            for p in all_packs
        ])
        batched_image_segment_ids = np.stack([
            pad_to_shape(p['image_segment_ids'], (max_patches,), pad_value=-1)
            for p in all_packs
        ])
        batched_position_ids = np.stack([
            pad_to_shape(p['position_ids'], (max_seq_len,), pad_value=0)
            for p in all_packs
        ])
        batched_combined_mask = np.stack([
            pad_to_shape(p['combined_mask'], (max_seq_len,), pad_value=0)
            for p in all_packs
        ])

        print(f"\n=== Batched Packed Info (Streaming) ===")
        print(f"  batch_size: {batch_size}")
        print(f"  max_seq_len: {max_seq_len}")
        print(f"  max_patches: {max_patches}")
        print(f"  batched_input_ids shape: {batched_input_ids.shape}")
        print(f"  batched_pixel_values shape: {batched_pixel_values.shape}")

        # ==================== Load Levanter model and compute batched loss ====================
        # IMPORTANT: Must use vision_aspect_ratio="single" and image_grid_pinpoints for disable_anyres mode
        lev_config = _load_levanter_config(
            MODEL_NAME,
            enable_flash_attention=False,
            gradient_checkpointing=False,
            vision_aspect_ratio="single",
            image_grid_pinpoints=grid_pinpoints,
        )
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

            # Create JAX tensors from batched packed sample
            Batch = hax.Axis("batch", batch_size)
            Position = hax.Axis("position", max_seq_len)
            NumPatches = hax.Axis("num_patches", max_patches)
            Channels = hax.Axis("channels", 3)
            Height = hax.Axis("height", 384)
            Width = hax.Axis("width", 384)
            GridMask = hax.Axis("grid_mask", max_patches)

            input_ids = hax.named(jnp.array(batched_input_ids), (Batch, Position))
            pixel_values = hax.named(
                jnp.array(batched_pixel_values),
                (Batch, NumPatches, Channels, Height, Width)
            )
            grid_mask = hax.named(
                jnp.array(batched_image_segment_ids >= 0),
                (Batch, GridMask)
            )
            loss_mask = hax.named(jnp.array(batched_loss_mask), (Batch, Position))
            segment_ids_jax = hax.named(jnp.array(batched_segment_ids), (Batch, Position))
            position_ids = hax.named(jnp.array(batched_position_ids), (Batch, Position))
            image_segment_ids_jax = hax.named(jnp.array(batched_image_segment_ids), (Batch, GridMask))
            combined_mask = hax.named(jnp.array(batched_combined_mask), (Batch, Position))

            from levanter.data.image import ImageTextExample as ImgTextEx

            batch_example = ImgTextEx(
                pixel_values=pixel_values,
                input_ids=input_ids,
                loss_mask=loss_mask,
                grid_mask=grid_mask,
                unpad_indices=None,
                combined_mask=combined_mask,
                position_ids=position_ids,
            )

            # Compute per-token loss for the batch
            @eqx.filter_jit
            def compute_per_token_loss(model, example, segment_ids, image_segment_ids):
                from levanter.models.loss import fused_cross_entropy_loss_and_logsumexp_penalty

                grid_mask = getattr(example, "grid_mask", None)
                position_ids = getattr(example, "position_ids", None)
                combined_mask = getattr(example, "combined_mask", None)

                activations, lm_head = model.forward_with_activations(
                    example.input_ids,
                    pixel_values=example.pixel_values,
                    grid_mask=grid_mask,
                    unpad_indices=None,
                    combined_mask=combined_mask,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    image_segment_ids=image_segment_ids,
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

            per_token_loss = compute_per_token_loss(lev_model, batch_example, segment_ids_jax, image_segment_ids_jax)
            per_token_loss_np = np.array(per_token_loss.array)  # (batch_size, seq_len)

            # Compute per-segment loss for each pack in the batch
            print(f"\n=== Levanter Per-Segment Losses (Batched Streaming) ===")
            all_passed = True
            global_sample_idx = 0

            for pack_idx in range(batch_size):
                pack_loss_np = per_token_loss_np[pack_idx]
                pack_segment_ids = batched_segment_ids[pack_idx]
                pack_loss_mask = batched_loss_mask[pack_idx]

                shifted_loss_mask = np.roll(pack_loss_mask, -1)
                shifted_loss_mask[-1] = 0

                print(f"\n  Pack {pack_idx}:")

                for seg_id in range(samples_per_pack):
                    seg_mask = (pack_segment_ids == seg_id) & (shifted_loss_mask > 0)
                    if seg_mask.sum() > 0:
                        seg_loss = pack_loss_np[seg_mask].mean()
                    else:
                        seg_loss = 0.0

                    hf_loss = hf_golden_losses[global_sample_idx]
                    diff = abs(hf_loss - seg_loss)
                    rel_diff = diff / hf_loss if hf_loss != 0 else diff

                    status = "PASS" if rel_diff < 0.0002 else "FAIL"
                    print(f"    Segment {seg_id} (Sample {global_sample_idx}, {num_images_per_sample[global_sample_idx]} img): "
                          f"HF={hf_loss:.6f}, Lev={seg_loss:.6f}, rel_diff={rel_diff*100:.4f}% [{status}]")

                    if rel_diff >= 0.0002:
                        all_passed = False

                    global_sample_idx += 1

            assert all_passed, "Batched packed losses do not match HF golden losses!"
            print("\n Batched VLM packing loss consistency test (interleaved/streaming) passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
