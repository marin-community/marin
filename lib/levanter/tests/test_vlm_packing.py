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
"""

import asyncio
import tempfile
from typing import Dict, List, Sequence

import numpy as np
import pytest

from levanter.data.vlm_packing import (
    PackedImageTextDict,
    VLMPackerConfig,
    VLMPrepackedDataset,
    compute_vlm_sample_lengths,
    load_pack_assignments,
    pack_vlm_samples,
    save_pack_assignments,
)


class MockImageTextDict(dict):
    """Mock ImageTextDict for testing."""
    pass


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


def create_mock_pixel_values(num_patches: int, channels: int = 3, height: int = 384, width: int = 384) -> np.ndarray:
    """Create mock pixel values for testing."""
    return np.random.randn(num_patches, channels, height, width).astype(np.float32)


class TestVLMPackerConfig:
    """Tests for VLMPackerConfig."""

    def test_default_values(self):
        config = VLMPackerConfig(max_length=2048, max_patches=10)
        assert config.max_length == 2048
        assert config.max_patches == 10
        assert config.max_segments == 64
        assert config.features_per_patch == 576
        assert config.pad_token_id == 0

    def test_custom_values(self):
        config = VLMPackerConfig(
            max_length=1024,
            max_patches=5,
            max_segments=8,
            features_per_patch=256,
            pad_token_id=1,
            image_token_id=151646,
        )
        assert config.max_length == 1024
        assert config.max_patches == 5
        assert config.max_segments == 8
        assert config.features_per_patch == 256
        assert config.pad_token_id == 1


class TestComputeVLMSampleLengths:
    """Tests for compute_vlm_sample_lengths function."""

    def test_basic_lengths(self):
        samples = [
            create_mock_sample(
                input_ids=[1, 2, 3, 4, 5],
                pixel_values=create_mock_pixel_values(2),
                attention_mask=[1, 1, 1, 1, 1],
            ),
            create_mock_sample(
                input_ids=[1, 2, 3],
                pixel_values=create_mock_pixel_values(1),
                attention_mask=[1, 1, 1],
            ),
        ]
        dataset = MockAsyncDataset(samples)

        # features_per_patch = 4 for easy calculation
        # Sample 0: 5 tokens + 2 patches * (4-1) = 5 + 6 = 11 effective tokens
        # Sample 1: 3 tokens + 1 patch * (4-1) = 3 + 3 = 6 effective tokens
        lengths = compute_vlm_sample_lengths(dataset, features_per_patch=4, batch_size=10)

        assert len(lengths["tokens"]) == 2
        assert len(lengths["patches"]) == 2
        assert lengths["tokens"][0] == 11  # 5 + 2 * 3
        assert lengths["tokens"][1] == 6   # 3 + 1 * 3
        assert lengths["patches"][0] == 2
        assert lengths["patches"][1] == 1

    def test_no_images(self):
        """Test samples with no images."""
        samples = [
            create_mock_sample(
                input_ids=[1, 2, 3, 4, 5],
                attention_mask=[1, 1, 1, 1, 1],
            ),
        ]
        dataset = MockAsyncDataset(samples)

        lengths = compute_vlm_sample_lengths(dataset, features_per_patch=4)

        assert lengths["tokens"][0] == 5  # No image expansion
        assert lengths["patches"][0] == 0

    def test_with_grid_mask(self):
        """Test that grid_mask is used to count valid patches."""
        samples = [
            create_mock_sample(
                input_ids=[1, 2, 3],
                pixel_values=create_mock_pixel_values(5),
                attention_mask=[1, 1, 1],
                grid_mask=[True, True, True, False, False],  # Only 3 valid patches
            ),
        ]
        dataset = MockAsyncDataset(samples)

        lengths = compute_vlm_sample_lengths(dataset, features_per_patch=4)

        assert lengths["patches"][0] == 3  # Only valid patches counted
        assert lengths["tokens"][0] == 3 + 3 * 3  # 3 + 9 = 12


class TestPackVLMSamples:
    """Tests for pack_vlm_samples function."""

    def test_basic_packing(self):
        lengths = {
            "tokens": np.array([100, 200, 150, 300]),
            "patches": np.array([1, 1, 1, 2]),
        }

        packs = pack_vlm_samples(
            lengths=lengths,
            max_tokens=400,
            max_patches=3,
            max_segments=4,
        )

        # Verify all samples are assigned to some pack
        all_indices = set()
        for pack in packs:
            for idx in pack:
                all_indices.add(idx)
        assert all_indices == {0, 1, 2, 3}

        # Verify no pack exceeds limits
        for pack in packs:
            pack_tokens = sum(lengths["tokens"][i] for i in pack)
            pack_patches = sum(lengths["patches"][i] for i in pack)
            assert pack_tokens <= 400
            assert pack_patches <= 3

    def test_single_large_sample(self):
        """Test that oversized samples become solo packs."""
        lengths = {
            "tokens": np.array([100, 500, 150]),  # Sample 1 exceeds max
            "patches": np.array([1, 1, 1]),
        }

        packs = pack_vlm_samples(
            lengths=lengths,
            max_tokens=400,
            max_patches=3,
            max_segments=4,
        )

        # Sample with 500 tokens should be in its own pack
        for pack in packs:
            if 1 in pack:
                assert len(pack) == 1  # Should be solo


class TestSaveLoadPackAssignments:
    """Tests for save_pack_assignments and load_pack_assignments."""

    def test_save_and_load(self):
        assignments = [range(0, 3), range(3, 5), range(5, 8)]
        config = VLMPackerConfig(max_length=2048, max_patches=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            path = save_pack_assignments(assignments, tmpdir, config)
            assert path.endswith("vlm_pack_assignments.json")

            # Load
            loaded = load_pack_assignments(tmpdir)

            assert len(loaded) == len(assignments)
            for orig, loaded_range in zip(assignments, loaded):
                assert list(orig) == list(loaded_range)

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_pack_assignments(tmpdir)


class TestVLMPrepackedDataset:
    """Tests for VLMPrepackedDataset."""

    def test_basic_dataset(self):
        """Test basic dataset creation and access."""
        # Create samples
        samples = [
            create_mock_sample(
                input_ids=[1, 2, 3, 4, 5],
                pixel_values=create_mock_pixel_values(1),
                attention_mask=[1, 1, 1, 1, 1],
                loss_mask=[0, 0, 1, 1, 1],
            ),
            create_mock_sample(
                input_ids=[6, 7, 8],
                pixel_values=create_mock_pixel_values(1),
                attention_mask=[1, 1, 1],
                loss_mask=[0, 1, 1],
            ),
        ]
        base_dataset = MockAsyncDataset(samples)

        config = VLMPackerConfig(
            max_length=20,
            max_patches=5,
            max_segments=4,
            features_per_patch=4,
        )

        dataset = VLMPrepackedDataset(
            base_dataset=base_dataset,
            config=config,
        )

        # Check dataset properties
        assert dataset.is_finite()
        assert asyncio.get_event_loop().run_until_complete(dataset.async_len()) >= 1

    def test_get_batch(self):
        """Test get_batch returns correct packed format."""
        samples = [
            create_mock_sample(
                input_ids=[1, 2, 3],
                pixel_values=create_mock_pixel_values(1),
                attention_mask=[1, 1, 1],
                loss_mask=[0, 1, 1],
            ),
            create_mock_sample(
                input_ids=[4, 5],
                pixel_values=create_mock_pixel_values(1),
                attention_mask=[1, 1],
                loss_mask=[1, 1],
            ),
        ]
        base_dataset = MockAsyncDataset(samples)

        config = VLMPackerConfig(
            max_length=10,
            max_patches=5,
            max_segments=4,
            features_per_patch=4,
        )

        dataset = VLMPrepackedDataset(
            base_dataset=base_dataset,
            config=config,
        )

        # Get first pack
        batch = asyncio.get_event_loop().run_until_complete(dataset.get_batch([0]))
        packed = batch[0]

        # Verify packed format
        assert "input_ids" in packed
        assert "segment_ids" in packed
        assert "position_ids" in packed
        assert "loss_mask" in packed
        assert len(packed["input_ids"]) == config.max_length

    def test_cache_persistence(self):
        """Test that pack assignments are saved and loaded from cache."""
        samples = [
            create_mock_sample(
                input_ids=[1, 2, 3],
                pixel_values=create_mock_pixel_values(1),
                attention_mask=[1, 1, 1],
            ),
            create_mock_sample(
                input_ids=[4, 5],
                pixel_values=create_mock_pixel_values(1),
                attention_mask=[1, 1],
            ),
        ]
        base_dataset = MockAsyncDataset(samples)

        config = VLMPackerConfig(
            max_length=10,
            max_patches=5,
            max_segments=4,
            features_per_patch=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # First creation - computes and saves pack assignments
            dataset1 = VLMPrepackedDataset(
                base_dataset=base_dataset,
                config=config,
                cache_dir=tmpdir,
            )
            len1 = asyncio.get_event_loop().run_until_complete(dataset1.async_len())

            # Second creation - should load from cache
            dataset2 = VLMPrepackedDataset(
                base_dataset=base_dataset,
                config=config,
                cache_dir=tmpdir,
            )
            len2 = asyncio.get_event_loop().run_until_complete(dataset2.async_len())

            assert len1 == len2


class TestAssemblePack:
    """Tests for _assemble_pack method."""

    def test_segment_ids_assignment(self):
        """Test that segment IDs are correctly assigned to each sample.

        Uses _assemble_pack directly to ensure multiple samples are packed together.
        """
        samples = [
            create_mock_sample(
                input_ids=[1, 2, 3],
                pixel_values=create_mock_pixel_values(1),
                attention_mask=[1, 1, 1],
            ),
            create_mock_sample(
                input_ids=[4, 5],
                pixel_values=create_mock_pixel_values(1),
                attention_mask=[1, 1],
            ),
        ]

        config = VLMPackerConfig(
            max_length=10,
            max_patches=5,
            max_segments=4,
            features_per_patch=4,
        )

        # Use _assemble_pack directly to test packing logic
        base_dataset = MockAsyncDataset(samples)
        dataset = VLMPrepackedDataset(
            base_dataset=base_dataset,
            config=config,
        )

        # Directly call _assemble_pack with both samples
        packed = dataset._assemble_pack(samples)

        segment_ids = packed["segment_ids"]

        # First 3 tokens should have segment_id=0
        # Next 2 tokens should have segment_id=1
        # Rest should be padding (-1)
        assert segment_ids[0] == 0
        assert segment_ids[1] == 0
        assert segment_ids[2] == 0
        assert segment_ids[3] == 1
        assert segment_ids[4] == 1
        assert all(segment_ids[5:] == -1)

    def test_image_segment_ids(self):
        """Test that image segment IDs are correctly assigned.

        Uses _assemble_pack directly to ensure multiple samples are packed together.
        """
        samples = [
            create_mock_sample(
                input_ids=[1, 2],
                pixel_values=create_mock_pixel_values(2),  # 2 patches
                attention_mask=[1, 1],
            ),
            create_mock_sample(
                input_ids=[3, 4],
                pixel_values=create_mock_pixel_values(1),  # 1 patch
                attention_mask=[1, 1],
            ),
        ]

        config = VLMPackerConfig(
            max_length=10,
            max_patches=5,
            max_segments=4,
            features_per_patch=4,
        )

        base_dataset = MockAsyncDataset(samples)
        dataset = VLMPrepackedDataset(
            base_dataset=base_dataset,
            config=config,
        )

        # Directly call _assemble_pack with both samples
        packed = dataset._assemble_pack(samples)

        image_segment_ids = packed["image_segment_ids"]

        # First 2 patches -> segment 0
        # Next 1 patch -> segment 1
        # Rest -> padding (-1)
        assert image_segment_ids[0] == 0
        assert image_segment_ids[1] == 0
        assert image_segment_ids[2] == 1
        assert all(image_segment_ids[3:] == -1)

    def test_position_ids_reset_per_segment(self):
        """Test that position IDs reset for each segment.

        Uses _assemble_pack directly to ensure multiple samples are packed together.
        """
        samples = [
            create_mock_sample(
                input_ids=[1, 2, 3],
                pixel_values=create_mock_pixel_values(1),
                attention_mask=[1, 1, 1],
            ),
            create_mock_sample(
                input_ids=[4, 5],
                pixel_values=create_mock_pixel_values(1),
                attention_mask=[1, 1],
            ),
        ]

        config = VLMPackerConfig(
            max_length=10,
            max_patches=5,
            max_segments=4,
            features_per_patch=4,
        )

        base_dataset = MockAsyncDataset(samples)
        dataset = VLMPrepackedDataset(
            base_dataset=base_dataset,
            config=config,
        )

        # Directly call _assemble_pack with both samples
        packed = dataset._assemble_pack(samples)

        position_ids = packed["position_ids"]

        # First segment: positions 0, 1, 2
        assert position_ids[0] == 0
        assert position_ids[1] == 1
        assert position_ids[2] == 2

        # Second segment: positions 0, 1 (reset)
        assert position_ids[3] == 0
        assert position_ids[4] == 1

        # Padding positions should be 0
        assert all(position_ids[5:] == 0)

    def test_attention_mask_from_segment_ids(self):
        """Test that attention_mask is derived from segment_ids."""
        samples = [
            create_mock_sample(
                input_ids=[1, 2, 3],
                pixel_values=create_mock_pixel_values(1),
                attention_mask=[1, 1, 1],
            ),
        ]
        base_dataset = MockAsyncDataset(samples)

        config = VLMPackerConfig(
            max_length=10,
            max_patches=5,
            max_segments=4,
            features_per_patch=4,
        )

        dataset = VLMPrepackedDataset(
            base_dataset=base_dataset,
            config=config,
        )

        batch = asyncio.get_event_loop().run_until_complete(dataset.get_batch([0]))
        packed = batch[0]

        attention_mask = packed["attention_mask"]
        segment_ids = packed["segment_ids"]

        # attention_mask should be 1 where segment_ids >= 0
        expected_mask = (segment_ids >= 0).astype(np.int32)
        np.testing.assert_array_equal(attention_mask, expected_mask)

    def test_loss_mask_concatenation(self):
        """Test that loss masks are correctly concatenated and padded.

        Uses _assemble_pack directly to ensure multiple samples are packed together.
        """
        samples = [
            create_mock_sample(
                input_ids=[1, 2, 3],
                pixel_values=create_mock_pixel_values(1),
                attention_mask=[1, 1, 1],
                loss_mask=[0, 1, 1],  # Don't compute loss for first token
            ),
            create_mock_sample(
                input_ids=[4, 5],
                pixel_values=create_mock_pixel_values(1),
                attention_mask=[1, 1],
                loss_mask=[1, 1],
            ),
        ]

        config = VLMPackerConfig(
            max_length=10,
            max_patches=5,
            max_segments=4,
            features_per_patch=4,
        )

        base_dataset = MockAsyncDataset(samples)
        dataset = VLMPrepackedDataset(
            base_dataset=base_dataset,
            config=config,
        )

        # Directly call _assemble_pack with both samples
        packed = dataset._assemble_pack(samples)

        loss_mask = packed["loss_mask"]

        # Expected: [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        assert loss_mask[0] == 0  # First sample, first token
        assert loss_mask[1] == 1
        assert loss_mask[2] == 1
        assert loss_mask[3] == 1  # Second sample
        assert loss_mask[4] == 1
        assert all(loss_mask[5:] == 0)  # Padding


class TestPackDeterminism:
    """Tests for pack assignment determinism."""

    def test_deterministic_packing(self):
        """Test that same input produces same pack assignments."""
        lengths = {
            "tokens": np.array([100, 200, 150, 300, 50, 80]),
            "patches": np.array([1, 2, 1, 3, 1, 1]),
        }

        packs1 = pack_vlm_samples(
            lengths=lengths,
            max_tokens=400,
            max_patches=4,
            max_segments=4,
        )

        packs2 = pack_vlm_samples(
            lengths=lengths,
            max_tokens=400,
            max_patches=4,
            max_segments=4,
        )

        assert len(packs1) == len(packs2)
        for p1, p2 in zip(packs1, packs2):
            assert list(p1) == list(p2)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_no_pixel_values(self):
        """Test packing samples with no images."""
        samples = [
            create_mock_sample(
                input_ids=[1, 2, 3],
                attention_mask=[1, 1, 1],
            ),
            create_mock_sample(
                input_ids=[4, 5],
                attention_mask=[1, 1],
            ),
        ]
        base_dataset = MockAsyncDataset(samples)

        config = VLMPackerConfig(
            max_length=10,
            max_patches=5,
            max_segments=4,
            features_per_patch=4,
        )

        dataset = VLMPrepackedDataset(
            base_dataset=base_dataset,
            config=config,
        )

        batch = asyncio.get_event_loop().run_until_complete(dataset.get_batch([0]))
        packed = batch[0]

        # Should have placeholder pixel_values and all -1 image_segment_ids
        assert packed["pixel_values"].shape[0] == config.max_patches
        assert all(packed["image_segment_ids"] == -1)

    def test_single_sample_pack(self):
        """Test packing a single sample."""
        samples = [
            create_mock_sample(
                input_ids=[1, 2, 3, 4, 5],
                pixel_values=create_mock_pixel_values(2),
                attention_mask=[1, 1, 1, 1, 1],
            ),
        ]
        base_dataset = MockAsyncDataset(samples)

        config = VLMPackerConfig(
            max_length=10,
            max_patches=5,
            max_segments=4,
            features_per_patch=4,
        )

        dataset = VLMPrepackedDataset(
            base_dataset=base_dataset,
            config=config,
        )

        batch = asyncio.get_event_loop().run_until_complete(dataset.get_batch([0]))
        packed = batch[0]

        # All valid tokens should have segment_id=0
        assert packed["segment_ids"][0] == 0
        assert packed["segment_ids"][4] == 0
        assert packed["num_segments"] == 1

    def test_padding_removal(self):
        """Test that padding from original samples is removed before packing."""
        samples = [
            create_mock_sample(
                input_ids=[1, 2, 3, 0, 0],  # Original has padding
                pixel_values=create_mock_pixel_values(1),
                attention_mask=[1, 1, 1, 0, 0],  # Only first 3 valid
            ),
        ]
        base_dataset = MockAsyncDataset(samples)

        config = VLMPackerConfig(
            max_length=10,
            max_patches=5,
            max_segments=4,
            features_per_patch=4,
        )

        dataset = VLMPrepackedDataset(
            base_dataset=base_dataset,
            config=config,
        )

        batch = asyncio.get_event_loop().run_until_complete(dataset.get_batch([0]))
        packed = batch[0]

        # Only 3 valid tokens should be packed
        assert packed["segment_ids"][0] == 0
        assert packed["segment_ids"][1] == 0
        assert packed["segment_ids"][2] == 0
        assert packed["segment_ids"][3] == -1  # Padding


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
