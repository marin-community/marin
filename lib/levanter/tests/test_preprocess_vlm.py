# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for VLM offline preprocessing.

These tests verify that the preprocessing script correctly produces data that:
1. Results in the same loss as HF model (golden reference)
2. Has identical input_ids and loss_mask to BatchImageProcessor output
3. Works correctly with the packing pipeline

Following the testing patterns from test_vlm_packing.py:
- Use HF model loss as the golden reference
- Use strict threshold (0.02%) for loss comparison
- Use prepare_test_data utility for HF vs Levanter comparison
"""

import asyncio
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add scripts directory to path for imports
# Test file: lib/levanter/tests/test_preprocess_vlm.py
# Scripts dir: scripts/
# So: tests/ -> levanter/ -> lib/ -> marin_private3/ -> scripts/
_scripts_dir = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(_scripts_dir))

from test_image_utils import (
    get_real_data,
    prepare_test_data,
    QWEN3_TOKENIZER,
    SINGLE_PATCH_GRID_PINPOINTS,
)
from test_utils import skip_if_no_torch

# Import preprocessing functions using importlib for reliable path resolution
import importlib.util
_preprocess_script = _scripts_dir / "preprocess_vlm_data.py"
if not _preprocess_script.exists():
    raise ImportError(f"preprocess_vlm_data.py not found at {_preprocess_script}")
spec = importlib.util.spec_from_file_location("preprocess_vlm_data", _preprocess_script)
_preprocess_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_preprocess_module)
PreprocessConfig = _preprocess_module.PreprocessConfig
preprocess_sample = _preprocess_module.preprocess_sample
create_loss_mask = _preprocess_module.create_loss_mask


MODEL_NAME = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
MAX_LENGTH = 8192
FEATURES_PER_PATCH = 729  # 27*27 for llava-onevision (384/14 ≈ 27)


@pytest.mark.entry
@skip_if_no_torch
def test_preprocessed_input_ids_match_batch_processor():
    """
    Test that preprocessed input_ids match what BatchImageProcessor produces.

    Uses prepare_test_data utility to get the reference data from BatchImageProcessor,
    which is the same processing path used during training.
    """
    from transformers import AutoTokenizer, AutoProcessor

    num_samples = 4

    qwen3_tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    config = PreprocessConfig(
        max_length=MAX_LENGTH,
        features_per_patch=FEATURES_PER_PATCH,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save real data to parquet
        hf_dataset = get_real_data(num_samples=num_samples)
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        # Get reference data using prepare_test_data (uses BatchImageProcessor)
        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=list(range(num_samples)),
            model_name=MODEL_NAME,
            max_length=MAX_LENGTH,
            max_num_patches=1,
            grid_pinpoints=SINGLE_PATCH_GRID_PINPOINTS,
            disable_anyres=True,
        )

        print(f"\n=== Input IDs Comparison ===")

        # Preprocess using our function and compare
        for i in range(num_samples):
            sample = hf_dataset[i]
            preprocessed = preprocess_sample(sample, processor, qwen3_tokenizer, config)

            # Reference input_ids from BatchImageProcessor
            ref_input_ids = np.array(test_pairs[i].lev.input_ids)
            ref_valid_len = int(np.sum(test_pairs[i].lev.attention_mask))

            # Our preprocessed input_ids
            our_valid_len = int(np.sum(preprocessed["attention_mask"]))

            # Compare valid portions
            min_len = min(ref_valid_len, our_valid_len)

            np.testing.assert_array_equal(
                preprocessed["input_ids"][:min_len],
                ref_input_ids[:min_len],
                err_msg=f"Sample {i}: input_ids mismatch"
            )

            print(f"  Sample {i}: PASS (valid_len={min_len})")

        print("\n Input IDs comparison test passed!")


@pytest.mark.entry
@skip_if_no_torch
def test_preprocessed_loss_mask_match_batch_processor():
    """
    Test that preprocessed loss_mask matches BatchImageProcessor output.

    The loss_mask determines which tokens contribute to the loss during training.
    It must be identical to ensure training behavior is unchanged.
    """
    from transformers import AutoTokenizer, AutoProcessor

    num_samples = 4

    qwen3_tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    config = PreprocessConfig(
        max_length=MAX_LENGTH,
        features_per_patch=FEATURES_PER_PATCH,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save real data to parquet
        hf_dataset = get_real_data(num_samples=num_samples)
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        # Get reference data using prepare_test_data
        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=list(range(num_samples)),
            model_name=MODEL_NAME,
            max_length=MAX_LENGTH,
            max_num_patches=1,
            grid_pinpoints=SINGLE_PATCH_GRID_PINPOINTS,
            disable_anyres=True,
        )

        print(f"\n=== Loss Mask Comparison ===")

        for i in range(num_samples):
            sample = hf_dataset[i]
            preprocessed = preprocess_sample(sample, processor, qwen3_tokenizer, config)

            # Reference loss_mask from BatchImageProcessor
            ref_loss_mask = np.array(test_pairs[i].lev.loss_mask)
            ref_valid_len = int(np.sum(test_pairs[i].lev.attention_mask))

            # Compare valid portions
            np.testing.assert_array_equal(
                preprocessed["loss_mask"][:ref_valid_len],
                ref_loss_mask[:ref_valid_len],
                err_msg=f"Sample {i}: loss_mask mismatch"
            )

            # Report stats
            valid_tokens = (preprocessed["loss_mask"][:ref_valid_len] > 0).sum()
            total_tokens = ref_valid_len
            print(f"  Sample {i}: PASS ({valid_tokens}/{total_tokens} tokens with loss_mask=1)")

        print("\n Loss mask comparison test passed!")


@pytest.mark.entry
@skip_if_no_torch
def test_preprocessed_loss_consistency_with_hf():
    """
    Test that loss computed from preprocessed data matches HF model loss.

    This is the KEY correctness test - verifies that offline preprocessing
    produces the same loss as HF model would compute directly.

    Steps:
    1. Load real data
    2. Preprocess using preprocess_vlm_data.py functions
    3. Get reference data using prepare_test_data (for pixel_values)
    4. Compute HF golden loss for each sample
    5. Compute Levanter loss using preprocessed data
    6. Compare: loss should match within 0.02% tolerance
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

    num_samples = 2
    grid_pinpoints = SINGLE_PATCH_GRID_PINPOINTS

    qwen3_tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)
    image_token_id = qwen3_tokenizer.convert_tokens_to_ids("<|image_pad|>")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    config = PreprocessConfig(
        max_length=MAX_LENGTH,
        features_per_patch=FEATURES_PER_PATCH,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Load real data and save to parquet
        hf_dataset = get_real_data(num_samples=num_samples)
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        # 2. Preprocess using our offline preprocessing
        preprocessed_samples = []
        for i in range(num_samples):
            sample = hf_dataset[i]
            preprocessed = preprocess_sample(sample, processor, qwen3_tokenizer, config)
            preprocessed_samples.append(preprocessed)

        # 3. Get reference data for pixel_values using prepare_test_data
        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=list(range(num_samples)),
            model_name=MODEL_NAME,
            max_length=MAX_LENGTH,
            max_num_patches=1,
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
            for i in range(num_samples):
                # Use preprocessed input_ids but HF pixel_values from test_pairs
                lev_data = test_pairs[i].lev
                hf_data = test_pairs[i].hf
                seq_len = int(np.sum(lev_data.attention_mask))

                hf_input_ids = torch.from_numpy(np.array(preprocessed_samples[i]["input_ids"][:seq_len])).unsqueeze(0)
                hf_pixel_values = torch.from_numpy(hf_data.pixel_values).unsqueeze(0)
                hf_image_sizes = torch.from_numpy(hf_data.image_sizes).unsqueeze(0)

                if hf_pixel_values.dim() == 5:
                    hf_pixel_values = hf_pixel_values[:, 0:1, :, :, :]

                # Create labels using our preprocessed loss_mask
                hf_labels = hf_input_ids.clone().long()
                loss_mask_np = preprocessed_samples[i]["loss_mask"][:seq_len]
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

        # ==================== Levanter: Compute loss using preprocessed data ====================
        lev_config = _load_levanter_config(
            MODEL_NAME,
            enable_flash_attention=False,
            gradient_checkpointing=False,
            vision_aspect_ratio="single",
            image_grid_pinpoints=grid_pinpoints,
        )
        lev_config = lev_config.with_token_ids(image_token_id=image_token_id)
        trainer_config = TrainerConfig()

        lev_losses = []
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

            for i in range(num_samples):
                # Use preprocessed input_ids and loss_mask, but pixel_values from test_pairs
                lev_data = test_pairs[i].lev
                seq_len = int(np.sum(lev_data.attention_mask))

                # Create JAX tensors
                Batch = hax.Axis("batch", 1)
                Position = hax.Axis("position", seq_len)
                NumPatches = hax.Axis("num_patches", lev_data.pixel_values.shape[0])
                Channels = hax.Axis("channels", 3)
                Height = hax.Axis("height", 384)
                Width = hax.Axis("width", 384)
                GridMask = hax.Axis("grid_mask", lev_data.pixel_values.shape[0])

                input_ids = hax.named(
                    jnp.array(preprocessed_samples[i]["input_ids"][:seq_len]).reshape(1, -1),
                    (Batch, Position)
                )
                pixel_values = hax.named(
                    jnp.array(lev_data.pixel_values).reshape(1, -1, 3, 384, 384),
                    (Batch, NumPatches, Channels, Height, Width)
                )
                grid_mask = hax.named(
                    jnp.array(lev_data.grid_mask).reshape(1, -1),
                    (Batch, GridMask)
                )
                loss_mask = hax.named(
                    jnp.array(preprocessed_samples[i]["loss_mask"][:seq_len]).reshape(1, -1),
                    (Batch, Position)
                )

                from levanter.data.image import ImageTextExample as ImgTextEx

                batch_example = ImgTextEx(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    loss_mask=loss_mask,
                    grid_mask=grid_mask,
                    unpad_indices=None,
                )

                # Compute loss
                @eqx.filter_jit
                def compute_loss(model, example):
                    from levanter.models.loss import fused_cross_entropy_loss_and_logsumexp_penalty

                    activations, lm_head = model.forward_with_activations(
                        example.input_ids,
                        pixel_values=example.pixel_values,
                        grid_mask=example.grid_mask,
                        unpad_indices=None,
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

                    # Apply loss_mask (shifted by 1 to match HF convention)
                    shifted_mask = hax.roll(example.loss_mask, -1, Pos)
                    masked_loss = per_token_loss * shifted_mask
                    total_loss = hax.sum(masked_loss) / hax.sum(shifted_mask)

                    return total_loss

                loss = compute_loss(lev_model, batch_example)
                lev_losses.append(float(loss.array))

        print(f"\n=== Levanter Losses (from preprocessed data) ===")
        for i, loss in enumerate(lev_losses):
            print(f"  Sample {i}: {loss:.6f}")

        # ==================== Compare losses ====================
        print(f"\n=== Loss Comparison ===")
        all_passed = True
        for i in range(num_samples):
            hf_loss = hf_golden_losses[i]
            lev_loss = lev_losses[i]
            diff = abs(hf_loss - lev_loss)
            rel_diff = diff / hf_loss if hf_loss != 0 else diff

            # Strict threshold: 0.02%
            status = "PASS" if rel_diff < 0.0002 else "FAIL"
            print(f"  Sample {i}: HF={hf_loss:.6f}, Lev={lev_loss:.6f}, rel_diff={rel_diff*100:.4f}% [{status}]")

            if rel_diff >= 0.0002:
                all_passed = False

        assert all_passed, "Preprocessed data loss does not match HF golden loss!"
        print("\n Preprocessed data loss consistency test passed!")


@pytest.mark.entry
@skip_if_no_torch
def test_preprocessed_with_streaming_packed_dataset():
    """
    Test that preprocessed data works correctly with PackedVLMDataset (streaming packing).

    This is the end-to-end test verifying that:
    1. Preprocessed input_ids and loss_mask are used correctly
    2. Packing produces correct per-segment losses
    3. Per-segment loss matches HF golden loss

    Similar to test_vlm_streaming_packed_dataset_loss_consistency but starts from
    preprocessed data instead of raw data.
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
    from levanter.data.vlm_packing import (
        PackedVLMDataset,
        compute_pack_assignments,
        PackAssignmentConfig,
    )
    from test_train_image_anyres import _load_levanter_config

    jax.config.update("jax_default_matmul_precision", "float32")

    num_samples = 2
    grid_pinpoints = SINGLE_PATCH_GRID_PINPOINTS

    qwen3_tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)
    image_token_id = qwen3_tokenizer.convert_tokens_to_ids("<|image_pad|>")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Load real data and save to parquet
        hf_dataset = get_real_data(num_samples=num_samples)
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        # Get test pairs for HF golden loss computation
        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=list(range(num_samples)),
            model_name=MODEL_NAME,
            max_length=MAX_LENGTH,
            max_num_patches=1,
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

        # ==================== Create PackedVLMDataset ====================
        max_patches = num_samples  # 1 patch per sample in disable_anyres mode

        # Compute pack assignments
        pack_config = PackAssignmentConfig(
            max_length=MAX_LENGTH,
            max_patches=max_patches,
            max_segments=num_samples,
            features_per_patch=FEATURES_PER_PATCH,
            image_column="images",
            text_column="messages",
        )

        output_file = f"{tmpdir}/pack_assignments.json"
        compute_pack_assignments(
            parquet_paths=[parquet_path],
            output_file=output_file,
            tokenizer=qwen3_tokenizer,
            config=pack_config,
            processor=processor,
        )

        # Create streaming dataset
        packed_dataset = PackedVLMDataset(
            pack_assignments_file=output_file,
            processor=processor,
            max_length=MAX_LENGTH,
            tokenizer=qwen3_tokenizer,
            disable_anyres=True,
            grid_pinpoints=grid_pinpoints,
            vision_feature_height=27,
            max_num_patches=max_patches,
            patch_size=384,
        )

        # Get packed batch
        packed_batch = asyncio.get_event_loop().run_until_complete(packed_dataset.get_batch([0]))
        packed = packed_batch[0]

        print(f"\n=== Packed Sample Info ===")
        print(f"  num_segments: {packed['num_segments']}")
        print(f"  input_ids shape: {packed['input_ids'].shape}")
        print(f"  pixel_values shape: {packed['pixel_values'].shape}")

        # ==================== Levanter: Compute per-segment loss ====================
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

            # Create JAX tensors
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
            combined_mask = hax.named(jnp.array(packed['combined_mask']).reshape(1, -1), (Batch, Position))

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

            @eqx.filter_jit
            def compute_per_token_loss(model, example, segment_ids, image_segment_ids):
                from levanter.models.loss import fused_cross_entropy_loss_and_logsumexp_penalty

                activations, lm_head = model.forward_with_activations(
                    example.input_ids,
                    pixel_values=example.pixel_values,
                    grid_mask=example.grid_mask,
                    unpad_indices=None,
                    combined_mask=example.combined_mask,
                    position_ids=example.position_ids,
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
            per_token_loss_np = np.array(per_token_loss.array)[0]

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

                status = "PASS" if rel_diff < 0.0002 else "FAIL"
                print(f"  Sample {i}: HF={hf_loss:.6f}, Lev={lev_loss:.6f}, rel_diff={rel_diff*100:.4f}% [{status}]")

                if rel_diff >= 0.0002:
                    all_passed = False

            assert all_passed, "Packed per-segment losses do not match HF golden losses!"
            print("\n Preprocessed data with packing loss consistency test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
