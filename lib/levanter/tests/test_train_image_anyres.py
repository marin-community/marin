# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Test training for vision-language models (LLaVA OneVision).

This test validates the training pipeline for image-text models.
"""

import dataclasses
import tempfile

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from transformers import AutoConfig

from levanter.main.train_vlm import compute_vlm_loss
from levanter.models.llava_onevision import LlavaOnevisionConfig
from test_image_utils import (
    prepare_test_data,
    prepare_batched_test_data,
    compare_logits_by_region,
    create_lev_jax_tensors,
    DEFAULT_GRID_PINPOINTS,
    SINGLE_PATCH_GRID_PINPOINTS,
    QWEN3_TOKENIZER,
    get_real_data,
    get_interleaved_data,
)
from levanter.utils.mesh import DEFAULT_DP_AXES, MeshConfig
from test_utils import skip_if_no_torch

MODEL_NAME = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
MODEL_NAME_7B = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
MAX_LENGTH = 8192


def _load_levanter_config(
    model_name=MODEL_NAME,
    enable_flash_attention=False,
    gradient_checkpointing=True,
    vision_aspect_ratio=None,
    image_grid_pinpoints=None,
):
    """Load and configure LlavaOnevisionConfig."""
    from levanter.layers.attention import AttentionBackend

    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    lev_config = LlavaOnevisionConfig.from_hf_config(hf_config)

    vision_config_updated = dataclasses.replace(
        lev_config.vision_config,
        use_flash_attention=False,
        gradient_checkpointing=gradient_checkpointing,
    )

    attn_backend = AttentionBackend.DEFAULT if enable_flash_attention else AttentionBackend.VANILLA
    text_config_updated = dataclasses.replace(
        lev_config.text_config,
        attn_backend=attn_backend,
        gradient_checkpointing=gradient_checkpointing,
    )

    config_updates = {
        "vision_config": vision_config_updated,
        "text_config": text_config_updated,
        "gradient_checkpointing": gradient_checkpointing,
    }

    # Override vision_aspect_ratio and image_grid_pinpoints if provided
    if vision_aspect_ratio is not None:
        config_updates["vision_aspect_ratio"] = vision_aspect_ratio
    if image_grid_pinpoints is not None:
        config_updates["image_grid_pinpoints"] = image_grid_pinpoints

    return dataclasses.replace(lev_config, **config_updates)


@pytest.mark.entry
@skip_if_no_torch
def test_vlm_numerical_correctness():
    """Verify numerical correctness of Levanter VLM vs HuggingFace implementation (disable_anyres mode)."""
    import torch
    import transformers.models.llava_onevision.modeling_llava_onevision as llava_modeling
    from transformers import AutoModelForVision2Seq
    from levanter.models.llava_onevision import LlavaOnevisionModel
    from levanter.trainer import TrainerConfig

    # Set JAX to use float32 matmul precision
    jax.config.update("jax_default_matmul_precision", "float32")

    model_name = MODEL_NAME
    grid_pinpoints = SINGLE_PATCH_GRID_PINPOINTS  # Use single patch for disable_anyres
    num_samples = 4

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_dataset = get_real_data()
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=list(range(num_samples)),
            model_name=model_name,
            max_length=MAX_LENGTH,
            max_num_patches=1,  # Only 1 patch for disable_anyres
            grid_pinpoints=grid_pinpoints,
            disable_anyres=True,
        )

    # Load Qwen3 tokenizer and get image token ID (used by both HF and Levanter)
    from transformers import AutoTokenizer
    qwen3_tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)
    image_token_id = qwen3_tokenizer.convert_tokens_to_ids("<|image_pad|>")

    hf_model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    hf_model.model.config.image_grid_pinpoints = grid_pinpoints
    hf_model.model.config.vision_aspect_ratio = "single"  # disable_anyres mode
    hf_model.model.image_newline = None
    hf_model.config.image_token_id = image_token_id  # Use Qwen3's image token ID
    hf_model.eval()

    lev_config = _load_levanter_config(model_name, enable_flash_attention=False, gradient_checkpointing=False)

    # Update config with correct image_token_id for Qwen3 tokenizer
    lev_config = lev_config.with_token_ids(image_token_id=image_token_id)

    trainer_config = TrainerConfig()

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        compute_dtype = jnp.float32
        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=model_name)
        parameter_axis_mapping = trainer_config.parameter_axis_mapping

        lev_model = converter.load_pretrained(
            LlavaOnevisionModel,
            ref=model_name,
            config=lev_config,
            axis_mapping=parameter_axis_mapping,
            dtype=compute_dtype,
            resize_vocab_to_match_tokenizer=False,
        )

        @eqx.filter_jit
        def compute_forward(model, input_ids, pixel_values, grid_mask, unpad_indices):
            return model(
                input_ids, pixel_values=pixel_values, grid_mask=grid_mask, unpad_indices=unpad_indices, key=None
            )

        all_passed = []

        # Monkey-patch image_size_to_num_patches to return 1 for disable_anyres mode
        original_image_size_to_num_patches = llava_modeling.image_size_to_num_patches

        def patched_image_size_to_num_patches(*args, **kwargs):
            return 1  # Force 1 patch per image for disable_anyres

        llava_modeling.image_size_to_num_patches = patched_image_size_to_num_patches

        try:
            for sample_idx, pair in enumerate(test_pairs):
                # Both processors now use same tokenizer, so input_ids should match
                hf_input_ids = torch.from_numpy(np.array(pair.hf.input_ids)).unsqueeze(0)
                hf_pixel_values = torch.from_numpy(pair.hf.pixel_values).unsqueeze(0)
                hf_image_sizes = torch.from_numpy(pair.hf.image_sizes).unsqueeze(0)

                # For disable_anyres mode: truncate to base patch only
                if hf_pixel_values.dim() == 5:
                    hf_pixel_values = hf_pixel_values[:, 0:1, :, :, :]

                with torch.no_grad():
                    hf_output = hf_model(
                        input_ids=hf_input_ids,
                        pixel_values=hf_pixel_values,
                        image_sizes=hf_image_sizes,
                    )
                    hf_logits = hf_output.logits[0].numpy()

                jax_tensors = create_lev_jax_tensors(pair.lev, batch_size=1)
                lev_logits = compute_forward(
                    lev_model,
                    jax_tensors.input_ids,
                    jax_tensors.pixel_values,
                    jax_tensors.grid_mask,
                    jax_tensors.unpad_indices,
                )
                lev_logits_np = np.array(lev_logits.array)[0]

                result = compare_logits_by_region(
                    hf_logits=hf_logits,
                    lev_logits=lev_logits_np,
                    input_ids=np.array(pair.hf.input_ids),
                    image_token_id=image_token_id,
                    tolerance=1e-3,
                    verbose=False,
                )
                all_passed.append(result.passed)
        finally:
            # Restore original function
            llava_modeling.image_size_to_num_patches = original_image_size_to_num_patches

        assert all(all_passed), f"Not all samples passed: {sum(all_passed)}/{len(all_passed)}"


@pytest.mark.entry
@skip_if_no_torch
def test_vlm_loss_and_gradients():
    """Test loss computation and gradient flow, comparing HF and Levanter implementations (disable_anyres mode)."""
    import torch
    import transformers.models.llava_onevision.modeling_llava_onevision as llava_modeling
    from transformers import AutoModelForVision2Seq
    from levanter.models.llava_onevision import LlavaOnevisionModel
    from levanter.trainer import TrainerConfig

    # Set JAX to use float32 matmul precision
    jax.config.update("jax_default_matmul_precision", "float32")

    model_name = MODEL_NAME
    grid_pinpoints = SINGLE_PATCH_GRID_PINPOINTS

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_dataset = get_real_data()
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=[0],
            model_name=model_name,
            max_length=MAX_LENGTH,
            max_num_patches=1,  # Only 1 patch for disable_anyres
            grid_pinpoints=grid_pinpoints,
            disable_anyres=True,
        )
        pair = test_pairs[0]

    # Load Qwen3 tokenizer and get image token ID (used by both HF and Levanter)
    from transformers import AutoTokenizer
    qwen3_tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)
    image_token_id = qwen3_tokenizer.convert_tokens_to_ids("<|image_pad|>")

    # ==================== Load HF Model ====================
    hf_model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    hf_model.model.config.image_grid_pinpoints = grid_pinpoints
    hf_model.model.config.vision_aspect_ratio = "single"  # disable_anyres mode
    hf_model.model.image_newline = None
    hf_model.config.image_token_id = image_token_id  # Use Qwen3's image token ID
    hf_model.eval()

    # Prepare HF inputs - use Levanter's input_ids (Qwen3 tokenized)
    lev_seq_len = len(pair.lev.input_ids)
    hf_input_ids = torch.from_numpy(np.array(pair.lev.input_ids[:lev_seq_len])).unsqueeze(0)
    hf_pixel_values = torch.from_numpy(pair.hf.pixel_values).unsqueeze(0)
    hf_image_sizes = torch.from_numpy(pair.hf.image_sizes).unsqueeze(0)

    # For disable_anyres mode: truncate to base patch only
    if hf_pixel_values.dim() == 5:
        hf_pixel_values = hf_pixel_values[:, 0:1, :, :, :]

    # Create labels: copy input_ids, set masked positions to -100
    # NOTE: Do NOT shift the loss_mask here! HF's ForCausalLMLoss already shifts labels internally.
    # Levanter's loss_mask[i]=1 means compute loss for predicting token[i].
    # HF's labels[i]!=-100 means compute loss for predicting token[i] (after HF's internal shift).
    hf_labels = hf_input_ids.clone().long()  # Cast to int64 for PyTorch cross_entropy
    seq_len = hf_input_ids.shape[1]
    loss_mask_np = np.array(pair.lev.loss_mask)[:seq_len]  # Truncate to actual sequence length
    mask_tensor = torch.from_numpy(loss_mask_np).unsqueeze(0)
    hf_labels[mask_tensor == 0] = -100  # Ignore masked positions

    # Monkey-patch image_size_to_num_patches to return 1 for disable_anyres mode
    original_image_size_to_num_patches = llava_modeling.image_size_to_num_patches
    llava_modeling.image_size_to_num_patches = lambda *args, **kwargs: 1

    try:
        # Compute HF loss
        with torch.no_grad():
            hf_output = hf_model(
                input_ids=hf_input_ids,
                pixel_values=hf_pixel_values,
                image_sizes=hf_image_sizes,
                labels=hf_labels,
            )
            hf_loss = hf_output.loss.item()
    finally:
        llava_modeling.image_size_to_num_patches = original_image_size_to_num_patches

    # ==================== Load Levanter Model ====================
    # Use flash attention and gradient checkpointing to avoid OOM with long sequences
    lev_config = _load_levanter_config(model_name, enable_flash_attention=True, gradient_checkpointing=True)

    # Update config with correct image_token_id for Qwen3 tokenizer (already loaded above)
    lev_config = lev_config.with_token_ids(image_token_id=image_token_id)

    # Use tensor parallelism instead of data parallelism for batch_size=1
    # Mesh: model=-1 absorbs all 8 devices, data=1 (auto), replica=1
    # Sharding strategy:
    #   - mlp -> model (tensor parallelism for MLP layers)
    #   - heads -> replica (size=1, no sharding; avoids heads=14 not divisible by 8)
    #   - embed -> data (default, size=1, no sharding)
    # This avoids conflicts: heads won't clash with embed (both would be 'data') or mlp ('model')
    mesh_config = MeshConfig(
        axes={"model": -1},
        shared_mapping={"mlp": "model", "heads": "replica"},
    )
    trainer_config = TrainerConfig(mesh=mesh_config)

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=model_name)
        lev_model = converter.load_pretrained(
            LlavaOnevisionModel,
            ref=model_name,
            config=lev_config,
            axis_mapping=trainer_config.parameter_axis_mapping,
            dtype=jnp.float32,
            resize_vocab_to_match_tokenizer=False,
        )

        # Use batch_size=1 for accurate HF comparison (single sample)
        jax_tensors = create_lev_jax_tensors(pair.lev, batch_size=1)
        from levanter.data.image import ImageTextExample as ImgTextEx

        batch_example = ImgTextEx(
            pixel_values=jax_tensors.pixel_values,
            input_ids=jax_tensors.input_ids,
            loss_mask=jax_tensors.loss_mask,
            grid_mask=jax_tensors.grid_mask,
            unpad_indices=jax_tensors.unpad_indices,
            combined_mask=jax_tensors.combined_mask,
            position_ids=jax_tensors.position_ids,
        )

        def compute_loss(model):
            loss = compute_vlm_loss(model, batch_example, key=None)
            return loss.scalar()

        lev_loss, grads = eqx.filter_value_and_grad(compute_loss)(lev_model)
        lev_loss_value = float(lev_loss)

        # ==================== Compare HF and Levanter Losses ====================
        print(f"\n=== Loss Comparison ===")
        print(f"  HF loss: {hf_loss:.6f}")
        print(f"  Levanter loss: {lev_loss_value:.6f}")
        loss_diff = abs(hf_loss - lev_loss_value)
        loss_rtol = abs(loss_diff / hf_loss) if hf_loss != 0 else loss_diff
        print(f"  Absolute diff: {loss_diff:.6f}")
        print(f"  Relative diff: {loss_rtol:.6f}")

        # Verify loss consistency between HF and Levanter
        assert np.isclose(hf_loss, lev_loss_value, rtol=1e-3, atol=1e-3), (
            f"Loss mismatch: HF={hf_loss:.6f}, Levanter={lev_loss_value:.6f}, "
            f"diff={loss_diff:.6f}, rtol={loss_rtol:.6f}"
        )

        # Verify loss is reasonable
        assert not np.isnan(lev_loss_value), "Levanter loss is NaN"
        assert 0.0 < lev_loss_value < 100.0, f"Levanter loss out of range: {lev_loss_value}"

        # ==================== Verify Gradient Flow ====================
        grad_dict = hax.state_dict.to_torch_compatible_state_dict(grads)
        has_vision_grads = any("vision_tower" in k for k in grad_dict if grad_dict[k] is not None)
        has_projector_grads = any("multi_modal_projector" in k for k in grad_dict if grad_dict[k] is not None)
        has_lm_grads = any("language_model" in k for k in grad_dict if grad_dict[k] is not None)

        assert has_vision_grads, "Vision tower should have gradients"
        assert has_projector_grads, "Projector should have gradients"
        assert has_lm_grads, "Language model should have gradients"

        # Check gradient norms are reasonable (filter out zero gradients from disabled params like biases)
        all_norms = [float(np.linalg.norm(g)) for g in grad_dict.values() if g is not None]
        nonzero_norms = [n for n in all_norms if n > 0]
        assert max(all_norms) < 1e6, f"Gradient explosion: max norm = {max(all_norms)}"
        assert min(nonzero_norms) > 1e-12, f"Gradient vanishing: min norm = {min(nonzero_norms)}"

        print("\n Loss and gradient consistency test passed!")


@pytest.mark.entry
@skip_if_no_torch
def test_vlm_training_reproducibility():
    """Verify training steps are reproducible with same seed (disable_anyres mode)."""
    from levanter.models.llava_onevision import LlavaOnevisionModel
    from levanter.trainer import TrainerConfig

    # Set JAX to use float32 matmul precision
    jax.config.update("jax_default_matmul_precision", "float32")

    model_name = MODEL_NAME

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_dataset = get_real_data()
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=[0],
            model_name=model_name,
            max_length=MAX_LENGTH,
            max_num_patches=1,  # Only 1 patch for disable_anyres
            grid_pinpoints=SINGLE_PATCH_GRID_PINPOINTS,
            disable_anyres=True,
        )
        pair = test_pairs[0]

    lev_config = _load_levanter_config(model_name, enable_flash_attention=True, gradient_checkpointing=True)

    # Update config with correct image_token_id for Qwen3 tokenizer
    from transformers import AutoTokenizer
    qwen3_tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)
    image_token_id = qwen3_tokenizer.convert_tokens_to_ids("<|image_pad|>")
    lev_config = lev_config.with_token_ids(image_token_id=image_token_id)

    trainer_config = TrainerConfig(per_device_parallelism=1)

    def run_training_step():
        with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
            converter = lev_config.hf_checkpoint_converter(ref_checkpoint=model_name)
            model = converter.load_pretrained(
                LlavaOnevisionModel,
                ref=model_name,
                config=lev_config,
                axis_mapping=trainer_config.parameter_axis_mapping,
                dtype=jnp.float32,
                resize_vocab_to_match_tokenizer=False,
            )

            jax_tensors = create_lev_jax_tensors(pair.lev, batch_size=8)
            from levanter.data.image import ImageTextExample as ImgTextEx

            batch_example = ImgTextEx(
                pixel_values=jax_tensors.pixel_values,
                input_ids=jax_tensors.input_ids,
                loss_mask=jax_tensors.loss_mask,
                grid_mask=jax_tensors.grid_mask,
                unpad_indices=jax_tensors.unpad_indices,
            )

            def compute_loss(model):
                return compute_vlm_loss(model, batch_example, key=None).scalar()

            loss, grads = eqx.filter_value_and_grad(compute_loss)(model)
            grad_dict = hax.state_dict.to_torch_compatible_state_dict(grads)
            sample_grad = grad_dict.get("language_model.lm_head.weight", None)
            return float(loss), sample_grad

    loss1, grads1 = run_training_step()
    loss2, grads2 = run_training_step()

    assert loss1 == loss2, f"Losses not identical: {loss1} vs {loss2}"
    if grads1 is not None and grads2 is not None:
        assert np.max(np.abs(grads1 - grads2)) == 0.0, "Gradients not identical"


@pytest.mark.entry
@skip_if_no_torch
def test_vlm_loss_mask():
    """Verify loss masking correctly excludes user prompts (disable_anyres mode)."""
    model_name = MODEL_NAME

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_dataset = get_real_data()
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=[0],
            model_name=model_name,
            max_length=MAX_LENGTH,
            max_num_patches=1,  # Only 1 patch for disable_anyres
            grid_pinpoints=SINGLE_PATCH_GRID_PINPOINTS,
            disable_anyres=True,
        )
        pair = test_pairs[0]

    loss_mask_np = np.array(pair.lev.loss_mask)
    total_positions = len(loss_mask_np)
    unmasked_positions = np.sum(loss_mask_np == 1.0)

    # Verify unmasked positions exist (assistant response)
    assert unmasked_positions > 0, "No unmasked positions - training would have no signal"
    # Verify most positions are masked (user prompt + image tokens + padding)
    assert unmasked_positions < total_positions * 0.5, "Too many unmasked positions"


@pytest.mark.entry
@skip_if_no_torch
def test_vlm_loss_and_gradients_interleved():
    """Test loss computation and gradient flow for interleaved multi-image data, comparing HF and Levanter."""
    import torch
    import transformers.models.llava_onevision.modeling_llava_onevision as llava_modeling
    from transformers import AutoModelForVision2Seq
    from levanter.models.llava_onevision import LlavaOnevisionModel
    from levanter.trainer import TrainerConfig

    jax.config.update("jax_default_matmul_precision", "float32")
    model_name = MODEL_NAME
    grid_pinpoints = SINGLE_PATCH_GRID_PINPOINTS

    with tempfile.TemporaryDirectory() as tmpdir:
        # Load interleaved data (contains samples with 1-4 images)
        hf_dataset = get_interleaved_data()
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        # Find a sample with multiple images for better testing
        sample_idx = 0
        for i, sample in enumerate(hf_dataset):
            if len(sample["images"]) > 1:
                sample_idx = i
                break

        num_images = len(hf_dataset[sample_idx]["images"])

        test_pairs = prepare_test_data(
            parquet_path=parquet_path,
            sample_indices=[sample_idx],
            model_name=model_name,
            max_length=MAX_LENGTH,
            max_num_patches=num_images,  # One patch per image in disable_anyres
            grid_pinpoints=grid_pinpoints,
            disable_anyres=True,
        )
        pair = test_pairs[0]

    # Load Qwen3 tokenizer and get image token ID (used by both HF and Levanter)
    from transformers import AutoTokenizer
    qwen3_tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)
    image_token_id = qwen3_tokenizer.convert_tokens_to_ids("<|image_pad|>")

    # ==================== Load HF Model ====================
    hf_model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    hf_model.model.config.image_grid_pinpoints = grid_pinpoints
    hf_model.model.config.vision_aspect_ratio = "single"  # disable_anyres mode
    hf_model.model.image_newline = None
    hf_model.config.image_token_id = image_token_id  # Use Qwen3's image token ID
    hf_model.eval()

    # Prepare HF inputs - use Levanter's input_ids (Qwen3 tokenized)
    lev_seq_len = len(pair.lev.input_ids)
    hf_input_ids = torch.from_numpy(np.array(pair.lev.input_ids[:lev_seq_len])).unsqueeze(0).long()
    hf_image_sizes = torch.from_numpy(pair.hf.image_sizes)

    # For multi-image disable_anyres: pixel_values shape is (num_images, C, H, W)
    # HF model expects 4D tensor for this case, NOT 5D with batch dimension
    # (5D format is for single-image anyres with multiple patches per image)
    hf_pixel_values = torch.from_numpy(pair.hf.pixel_values)
    if hf_pixel_values.dim() == 4 and num_images > 1:
        # Multi-image: keep 4D format (num_images, C, H, W) for HF model
        hf_pixel_values = hf_pixel_values[:num_images]
    else:
        # Single image: add batch dimension
        hf_pixel_values = hf_pixel_values.unsqueeze(0)

    # Create labels: copy input_ids, set masked positions to -100
    # NOTE: Do NOT shift the loss_mask here! HF's ForCausalLMLoss already shifts labels internally.
    # Levanter's loss_mask[i]=1 means compute loss for predicting token[i].
    # HF's labels[i]!=-100 means compute loss for predicting token[i] (after HF's internal shift).
    hf_labels = hf_input_ids.clone().long()  # Cast to int64 for PyTorch cross_entropy
    seq_len = hf_input_ids.shape[1]
    loss_mask_np = np.array(pair.lev.loss_mask)[:seq_len]  # Truncate to actual sequence length
    mask_tensor = torch.from_numpy(loss_mask_np).unsqueeze(0)
    hf_labels[mask_tensor == 0] = -100

    # ==================== Debug: Loss Mask and Input IDs Alignment ====================
    print(f"\n=== Loss Mask Debug ===")
    print(f"  HF input_ids length: {seq_len}")
    print(f"  Levanter loss_mask shape: {pair.lev.loss_mask.shape}")
    print(f"  Levanter loss_mask sum: {np.sum(pair.lev.loss_mask)}")
    print(f"  Truncated loss_mask sum: {np.sum(loss_mask_np)}")
    print(f"  HF labels with -100 count: {(hf_labels[0] == -100).sum().item()}")
    print(f"  HF labels valid count: {(hf_labels[0] != -100).sum().item()}")

    # Monkey-patch image_size_to_num_patches to return 1 for disable_anyres mode
    original_image_size_to_num_patches = llava_modeling.image_size_to_num_patches
    llava_modeling.image_size_to_num_patches = lambda *args, **kwargs: 1

    try:
        # ==================== Debug: Extract and compare image features ====================
        with torch.no_grad():
            # Get HF image features
            hf_image_features = hf_model.get_image_features(
                pixel_values=hf_pixel_values,
                image_sizes=hf_image_sizes,
            )
            print(f"\n=== HF Image Features Debug ===")
            print(f"  Number of feature tensors: {len(hf_image_features)}")
            for i, feat in enumerate(hf_image_features):
                print(f"  Feature {i}: shape={feat.shape}, mean={feat.mean():.6f}, std={feat.std():.6f}")

            # Concatenate HF features to single tensor for comparison
            hf_features_cat = torch.cat(hf_image_features, dim=0)  # (total_tokens, embed)
            print(f"  HF concatenated features: shape={hf_features_cat.shape}, mean={hf_features_cat.mean():.6f}, std={hf_features_cat.std():.6f}")

        # Compute HF loss with batch_num_images for multi-image
        with torch.no_grad():
            hf_output = hf_model(
                input_ids=hf_input_ids,
                pixel_values=hf_pixel_values,
                image_sizes=hf_image_sizes,
                labels=hf_labels,
                batch_num_images=torch.tensor([num_images]),
            )
            hf_loss = hf_output.loss.item()
    finally:
        llava_modeling.image_size_to_num_patches = original_image_size_to_num_patches

    # ==================== Load Levanter Model ====================
    # Use flash attention and gradient checkpointing to avoid OOM with long sequences
    # Match HF model's disable_anyres settings: vision_aspect_ratio="single" and custom grid_pinpoints
    lev_config = _load_levanter_config(
        model_name,
        enable_flash_attention=True,
        gradient_checkpointing=True,
        vision_aspect_ratio="single",
        image_grid_pinpoints=grid_pinpoints,
    )

    # Update config with correct image_token_id for Qwen3 tokenizer (already loaded above)
    lev_config = lev_config.with_token_ids(image_token_id=image_token_id)

    # Use tensor parallelism instead of data parallelism for batch_size=1
    # Mesh: model=-1 absorbs all 8 devices, data=1 (auto), replica=1
    # Sharding: mlp->model, heads->replica (avoids divisibility and conflict issues)
    mesh_config = MeshConfig(
        axes={"model": -1},
        shared_mapping={"mlp": "model", "heads": "replica"},
    )
    trainer_config = TrainerConfig(mesh=mesh_config)

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=model_name)
        lev_model = converter.load_pretrained(
            LlavaOnevisionModel,
            ref=model_name,
            config=lev_config,
            axis_mapping=trainer_config.parameter_axis_mapping,
            dtype=jnp.float32,
            resize_vocab_to_match_tokenizer=False,
        )

        # Use batch_size=1 to match HF
        jax_tensors = create_lev_jax_tensors(pair.lev, batch_size=1)
        from levanter.data.image import ImageTextExample as ImgTextEx

        batch_example = ImgTextEx(
            pixel_values=jax_tensors.pixel_values,
            input_ids=jax_tensors.input_ids,
            loss_mask=jax_tensors.loss_mask,
            grid_mask=jax_tensors.grid_mask,
            unpad_indices=jax_tensors.unpad_indices,
            combined_mask=jax_tensors.combined_mask,
            position_ids=jax_tensors.position_ids,
        )

        # ==================== Debug: Extract Levanter image features ====================
        print(f"\n=== Levanter Image Features Debug ===")
        print(f"  pixel_values shape: {jax_tensors.pixel_values.axes}")
        print(f"  grid_mask shape: {jax_tensors.grid_mask.axes}")
        print(f"  grid_mask values: {jax_tensors.grid_mask.array}")
        print(f"  unpad_indices: {jax_tensors.unpad_indices}")
        if jax_tensors.unpad_indices is not None:
            print(f"    unpad_indices shape: {jax_tensors.unpad_indices.axes}")
            print(f"    unpad_indices values: {np.array(jax_tensors.unpad_indices.array)}")

        print(f"\n=== Position IDs Debug ===")
        print(f"  position_ids shape: {jax_tensors.position_ids.axes if jax_tensors.position_ids is not None else None}")
        if jax_tensors.position_ids is not None:
            pos_np = np.array(jax_tensors.position_ids.array)[0]  # First sample
            print(f"  position_ids unique values: {np.unique(pos_np)[:20]}...")
            print(f"  position_ids max: {pos_np.max()}")
            print(f"  position_ids[:50]: {pos_np[:50]}")

        print(f"\n=== Combined Mask Debug ===")
        print(f"  combined_mask shape: {jax_tensors.combined_mask.axes if jax_tensors.combined_mask is not None else None}")
        if jax_tensors.combined_mask is not None:
            cmask_np = np.array(jax_tensors.combined_mask.array)[0]
            print(f"  combined_mask sum: {cmask_np.sum()}")
            print(f"  combined_mask[:50]: {cmask_np[:50]}")

        lev_features, _ = lev_model.get_image_features(
            jax_tensors.pixel_values,
            jax_tensors.grid_mask,
            key=None,
        )
        print(f"  Levanter features shape: {lev_features.axes}")
        # Get only valid patches based on grid_mask
        grid_mask_np = np.array(jax_tensors.grid_mask.array)
        lev_features_np = np.array(lev_features.array)  # (batch, num_patches, features_per_patch, embed)
        # Flatten to (batch, num_patches * features_per_patch, embed) then extract valid
        batch_size_np, num_patches, features_per_patch, embed_dim = lev_features_np.shape
        # Repeat mask to match features_per_patch
        mask_expanded = np.repeat(grid_mask_np, features_per_patch, axis=1)  # (batch, num_patches * features_per_patch)
        lev_features_flat = lev_features_np.reshape(batch_size_np, num_patches * features_per_patch, embed_dim)

        print(f"  Levanter features flat shape: {lev_features_flat.shape}")
        print(f"  mask_expanded sum per sample: {mask_expanded.sum(axis=1)}")

        # Extract valid features for first sample
        valid_features = lev_features_flat[0][mask_expanded[0].astype(bool)]
        print(f"  Levanter valid features: shape={valid_features.shape}, mean={valid_features.mean():.6f}, std={valid_features.std():.6f}")

        # Compare with HF features
        print(f"\n=== Feature Comparison ===")
        hf_feat_np = hf_features_cat.numpy()
        print(f"  HF features: shape={hf_feat_np.shape}, mean={hf_feat_np.mean():.6f}, std={hf_feat_np.std():.6f}")
        print(f"  Levanter features: shape={valid_features.shape}, mean={valid_features.mean():.6f}, std={valid_features.std():.6f}")

        # Check if shapes match
        if hf_feat_np.shape == valid_features.shape:
            # Element-wise comparison
            abs_diff = np.abs(hf_feat_np - valid_features)
            print(f"  Max abs diff: {abs_diff.max():.6f}")
            print(f"  Mean abs diff: {abs_diff.mean():.6f}")
            # Check first few elements
            print(f"  HF[0,:5]: {hf_feat_np[0, :5]}")
            print(f"  Lev[0,:5]: {valid_features[0, :5]}")

            # Check per-image feature alignment
            print(f"\n=== Per-Image Feature Comparison ===")
            for img_idx in range(num_images):
                start = img_idx * 729
                end = (img_idx + 1) * 729
                hf_img = hf_feat_np[start:end]
                lev_img = valid_features[start:end]
                img_diff = np.abs(hf_img - lev_img)
                print(f"  Image {img_idx}: max_diff={img_diff.max():.6f}, mean_diff={img_diff.mean():.6f}")
                print(f"    HF mean={hf_img.mean():.6f}, std={hf_img.std():.6f}")
                print(f"    Lev mean={lev_img.mean():.6f}, std={lev_img.std():.6f}")
        else:
            print(f"  WARNING: Shape mismatch! HF={hf_feat_np.shape}, Lev={valid_features.shape}")

        # ==================== Debug: Compare Merged Embeddings ====================
        print(f"\n=== Merged Embeddings Debug ===")

        # Get HF merged embeddings
        with torch.no_grad():
            # Get text embeddings for non-image positions
            text_embed = hf_model.get_input_embeddings()(hf_input_ids)
            print(f"  HF text embed shape: {text_embed.shape}")

            # Get image token positions
            image_token_id = hf_model.config.image_token_index
            image_positions = (hf_input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
            print(f"  Number of image token positions in HF: {len(image_positions)}")
            print(f"  Expected: {num_images * 729} = {num_images} images * 729 tokens/image")

        # Get Levanter merged embeddings
        lev_merged, lev_pos_ids, lev_validity = lev_model._merge_embeddings(
            input_ids=batch_example.input_ids,
            inputs_embeds=None,
            pixel_values=batch_example.pixel_values,
            grid_mask=batch_example.grid_mask,
            unpad_indices=batch_example.unpad_indices,
            precomputed_combined_mask=batch_example.combined_mask,
            precomputed_position_ids=batch_example.position_ids,
        )
        lev_merged_np = np.array(lev_merged.array)[0]  # (seq_len, embed)
        print(f"  Levanter merged embed shape: {lev_merged_np.shape}")

        # Get Levanter image token positions
        lev_input_ids_np = np.array(batch_example.input_ids.array)[0]
        lev_image_positions = np.where(lev_input_ids_np == image_token_id)[0]
        print(f"  Number of image token positions in Levanter: {len(lev_image_positions)}")

        # Compare embeddings at text positions (before first image)
        if len(image_positions) > 0:
            first_image_pos = image_positions[0].item()
            print(f"  First image position: {first_image_pos}")

            # Compare text embeddings before first image
            if first_image_pos > 0:
                hf_text_before = text_embed[0, :first_image_pos].numpy()
                lev_text_before = lev_merged_np[:first_image_pos]
                text_diff = np.abs(hf_text_before - lev_text_before)
                print(f"  Text before first image: max_diff={text_diff.max():.6f}, mean_diff={text_diff.mean():.6f}")

            # Compare image embeddings at first few image positions
            # Note: lev_merged_np contains image features at image positions
            #       hf_features_cat contains the raw image features (not merged)
            # So we compare: lev_merged at image positions vs hf_features

            # First image (positions first_image_pos to first_image_pos + 728)
            lev_first_img_embeds = lev_merged_np[first_image_pos:first_image_pos + 729]
            hf_first_img_features = hf_feat_np[:729]  # First 729 features = first image
            img_embed_diff = np.abs(lev_first_img_embeds - hf_first_img_features)
            print(f"  First image embeddings at merged positions:")
            print(f"    max_diff={img_embed_diff.max():.6f}, mean_diff={img_embed_diff.mean():.6f}")
            print(f"    Lev[0,:5]: {lev_first_img_embeds[0, :5]}")
            print(f"    HF[0,:5]: {hf_first_img_features[0, :5]}")

        # ==================== Debug: Compare Per-Token Losses ====================
        print(f"\n=== Per-Token Loss Debug ===")

        # Get HF per-token logits
        with torch.no_grad():
            hf_full_output = hf_model(
                input_ids=hf_input_ids,
                pixel_values=hf_pixel_values,
                image_sizes=hf_image_sizes,
                batch_num_images=torch.tensor([num_images]),
            )
            hf_logits = hf_full_output.logits[0]  # (seq_len, vocab)
            print(f"  HF logits shape: {hf_logits.shape}")

            # Compute HF per-token CE loss
            # Shift: predict next token
            hf_shift_logits = hf_logits[:-1]  # (seq_len-1, vocab)
            hf_shift_labels = hf_input_ids[0, 1:]  # (seq_len-1,)
            hf_ce = torch.nn.functional.cross_entropy(hf_shift_logits, hf_shift_labels, reduction='none')
            print(f"  HF per-token CE shape: {hf_ce.shape}")

        # Get Levanter per-token logits
        lev_logits = lev_model(
            batch_example.input_ids,
            pixel_values=batch_example.pixel_values,
            grid_mask=batch_example.grid_mask,
            unpad_indices=batch_example.unpad_indices,
            combined_mask=batch_example.combined_mask,
            position_ids=batch_example.position_ids,
            key=None,
        )
        lev_logits_np = np.array(lev_logits.array)[0, :seq_len]  # (seq_len, vocab)
        print(f"  Levanter logits shape: {lev_logits_np.shape}")

        # Compute Levanter per-token CE loss
        lev_shift_logits = lev_logits_np[:-1]  # (seq_len-1, vocab)
        lev_shift_labels = np.array(batch_example.input_ids.array)[0, 1:seq_len]  # (seq_len-1,)
        # Compute CE: -log(softmax(logits)[label])
        lev_log_softmax = lev_shift_logits - np.log(np.exp(lev_shift_logits).sum(axis=-1, keepdims=True) + 1e-10)
        lev_ce = -lev_log_softmax[np.arange(len(lev_shift_labels)), lev_shift_labels]
        print(f"  Levanter per-token CE shape: {lev_ce.shape}")

        # Compare per-token losses at positions where Levanter computes loss
        # Levanter uses roll(loss_mask, -1) to shift the mask, matching the shifted logits/labels
        lev_loss_mask = np.roll(loss_mask_np, -1)  # Shift mask for next-token prediction
        lev_loss_mask[-1] = 0  # Last position has no target
        shifted_mask_for_compare = lev_loss_mask[:-1]  # Align with shifted logits
        valid_positions = np.where(shifted_mask_for_compare > 0)[0]
        print(f"  Number of valid loss positions: {len(valid_positions)}")

        hf_ce_valid = hf_ce.numpy()[valid_positions]
        lev_ce_valid = lev_ce[valid_positions]
        ce_diff = np.abs(hf_ce_valid - lev_ce_valid)
        print(f"  Per-token CE at valid positions:")
        print(f"    HF mean: {hf_ce_valid.mean():.6f}, std: {hf_ce_valid.std():.6f}")
        print(f"    Lev mean: {lev_ce_valid.mean():.6f}, std: {lev_ce_valid.std():.6f}")
        print(f"    Diff max: {ce_diff.max():.6f}, mean: {ce_diff.mean():.6f}")
        print(f"    First 10 HF CE: {hf_ce_valid[:10]}")
        print(f"    First 10 Lev CE: {lev_ce_valid[:10]}")

        # Debug: Verify HF loss matches our calculation
        print(f"\n  Verifying HF loss calculation:")
        print(f"    HF CE all positions mean: {hf_ce.mean().item():.6f}")
        print(f"    HF CE valid positions mean: {hf_ce_valid.mean():.6f}")
        print(f"    Actual HF loss from model: {hf_loss:.6f}")
        # HF uses shifted labels internally, let's compute using same method
        hf_labels_shifted = hf_labels[0, 1:].numpy()  # labels[1:] after padding is what HF uses
        hf_valid_from_labels = np.where(hf_labels_shifted != -100)[0]
        print(f"    HF valid positions from labels: {len(hf_valid_from_labels)}")
        if len(hf_valid_from_labels) > 0:
            hf_ce_from_labels = hf_ce.numpy()[hf_valid_from_labels]
            print(f"    HF CE from labels mean: {hf_ce_from_labels.mean():.6f}")

        # Compare valid_positions vs hf_valid_from_labels
        print(f"\n  Comparing valid position selection:")
        print(f"    Our shifted_mask valid positions: {len(valid_positions)}")
        print(f"    HF labels valid positions: {len(hf_valid_from_labels)}")
        positions_match = np.array_equal(valid_positions, hf_valid_from_labels)
        print(f"    Positions match: {positions_match}")
        if not positions_match:
            # Find differences
            our_set = set(valid_positions)
            hf_set = set(hf_valid_from_labels)
            only_in_ours = sorted(our_set - hf_set)
            only_in_hf = sorted(hf_set - our_set)
            print(f"    Positions only in our mask: {only_in_ours[:20]}...")
            print(f"    Positions only in HF labels: {only_in_hf[:20]}...")

        def compute_loss(model):
            loss = compute_vlm_loss(model, batch_example, key=None)
            return loss.scalar()

        lev_loss, grads = eqx.filter_value_and_grad(compute_loss)(lev_model)
        lev_loss_value = float(lev_loss)

        # ==================== Compare HF and Levanter Losses ====================
        print(f"\n=== Interleaved Multi-Image Loss Comparison ===")
        print(f"  Number of images: {num_images}")
        print(f"  HF loss: {hf_loss:.6f}")
        print(f"  Levanter loss: {lev_loss_value:.6f}")
        loss_diff = abs(hf_loss - lev_loss_value)
        loss_rtol = abs(loss_diff / hf_loss) if hf_loss != 0 else loss_diff
        print(f"  Absolute diff: {loss_diff:.6f}")
        print(f"  Relative diff: {loss_rtol:.6f}")

        # Verify loss consistency between HF and Levanter
        assert np.isclose(hf_loss, lev_loss_value, rtol=1e-3, atol=1e-3), (
            f"Loss mismatch: HF={hf_loss:.6f}, Levanter={lev_loss_value:.6f}, "
            f"diff={loss_diff:.6f}, rtol={loss_rtol:.6f}"
        )

        # Verify loss is reasonable
        assert not np.isnan(lev_loss_value), "Levanter loss is NaN"
        assert 0.0 < lev_loss_value < 100.0, f"Levanter loss out of range: {lev_loss_value}"

        # ==================== Verify Gradient Flow ====================
        grad_dict = hax.state_dict.to_torch_compatible_state_dict(grads)
        has_vision_grads = any("vision_tower" in k for k in grad_dict if grad_dict[k] is not None)
        has_projector_grads = any("multi_modal_projector" in k for k in grad_dict if grad_dict[k] is not None)
        has_lm_grads = any("language_model" in k for k in grad_dict if grad_dict[k] is not None)

        assert has_vision_grads, "Vision tower should have gradients"
        assert has_projector_grads, "Projector should have gradients"
        assert has_lm_grads, "Language model should have gradients"

        # Check gradient norms are reasonable
        all_norms = [float(np.linalg.norm(g)) for g in grad_dict.values() if g is not None]
        nonzero_norms = [n for n in all_norms if n > 0]
        assert max(all_norms) < 1e6, f"Gradient explosion: max norm = {max(all_norms)}"
        assert min(nonzero_norms) > 1e-12, f"Gradient vanishing: min norm = {min(nonzero_norms)}"

        print("\n Interleaved loss and gradient consistency test passed!")


def _compute_per_example_vlm_loss(
    model,
    example,
    *,
    key=None,
    block_size: int = 4096,
):
    """Compute per-example loss for a batch of VLM examples.

    Unlike compute_vlm_loss which returns batch-averaged loss, this function
    returns loss for each example in the batch separately.

    Args:
        model: The LlavaOnevisionModel
        example: Batched example with input_ids, pixel_values, loss_mask, etc.
        key: Random key for stochastic operations
        block_size: Block size for blockwise cross-entropy

    Returns:
        NamedArray of shape (batch,) with per-example losses
    """
    from levanter.models.loss import fused_cross_entropy_loss_and_logsumexp_penalty

    # Forward pass
    grid_mask = getattr(example, "grid_mask", None)
    unpad_indices = getattr(example, "unpad_indices", None)
    combined_mask = getattr(example, "combined_mask", None)
    position_ids = getattr(example, "position_ids", None)

    activations, lm_head = model.forward_with_activations(
        example.input_ids,
        pixel_values=example.pixel_values,
        grid_mask=grid_mask,
        unpad_indices=unpad_indices,
        combined_mask=combined_mask,
        position_ids=position_ids,
        key=key,
    )

    # Get axes
    Pos = example.input_ids.resolve_axis("position")
    Batch = example.input_ids.resolve_axis("batch")
    Embed = model.config.TextEmbed
    Vocab = model.Vocab

    # Get targets (shifted by 1)
    targets = hax.roll(example.input_ids, -1, Pos)

    # Get shifted loss mask
    loss_weight = hax.roll(example.loss_mask, -1, Pos)

    # Compute per-token cross-entropy
    per_token_loss = fused_cross_entropy_loss_and_logsumexp_penalty(
        pred_embeddings=activations,
        pred_lm_head=lm_head,
        Contract=Embed,
        Label=Vocab,
        target_y=targets,
        reduction=None,
        weight=None,
        logsumexp_weight=0.0,
        block_size=block_size,
    )

    # Apply mask
    masked_loss = per_token_loss * loss_weight

    # Sum over Position axis only (not Batch) - per-example total loss
    per_example_loss_sum = hax.sum(masked_loss, axis=Pos)  # (Batch,)
    per_example_mask_sum = hax.sum(loss_weight, axis=Pos)  # (Batch,)

    # Per-example average loss
    per_example_losses = per_example_loss_sum / (per_example_mask_sum + 1e-8)

    return per_example_losses  # Return NamedArray, convert to numpy outside JIT


@pytest.mark.entry
@skip_if_no_torch
def test_vlm_batch_loss_consistency():
    """Test that batch processing produces same loss as individual processing.

    This test verifies that Levanter's batch processing gives identical per-example
    losses compared to HuggingFace processing each example individually.

    HF side: Compute loss 1-by-1 (no batching) → golden losses
    Levanter side: Batch 4 examples together → per-example losses
    Compare: Each Levanter loss should match HF golden loss
    """
    import torch
    import transformers.models.llava_onevision.modeling_llava_onevision as llava_modeling
    from transformers import AutoModelForVision2Seq
    from levanter.models.llava_onevision import LlavaOnevisionModel
    from levanter.trainer import TrainerConfig
    from levanter.data.image import ImageTextExample as ImgTextEx

    # Set JAX to use float32 matmul precision
    jax.config.update("jax_default_matmul_precision", "float32")

    model_name = MODEL_NAME
    grid_pinpoints = SINGLE_PATCH_GRID_PINPOINTS
    BATCH_SIZE = 16

    print(f"\n=== Batch Loss Consistency Test ===")
    print(f"  Batch size: {BATCH_SIZE}")

    # ==================== Prepare test data (without batching yet) ====================
    # Use interleaved data which has variable number of images per sample (1-4 images)
    tmpdir = tempfile.mkdtemp()
    hf_dataset = get_interleaved_data(num_samples=BATCH_SIZE)
    parquet_path = f"{tmpdir}/test_data.parquet"
    hf_dataset.to_parquet(parquet_path)

    # Get individual test pairs first (for HF comparison)
    test_pairs = prepare_test_data(
        parquet_path=parquet_path,
        sample_indices=list(range(BATCH_SIZE)),
        model_name=model_name,
        max_length=MAX_LENGTH,
        max_num_patches=1,
        grid_pinpoints=grid_pinpoints,
        disable_anyres=True,
    )

    print(f"  Loaded {len(test_pairs)} test samples")
    for i, pair in enumerate(test_pairs):
        print(f"    Sample {i}: {len(pair.lev.input_ids)} tokens, {np.sum(pair.lev.loss_mask)} valid loss positions")

    # Load Qwen3 tokenizer and get image token ID (used by both HF and Levanter)
    from transformers import AutoTokenizer
    qwen3_tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, trust_remote_code=True)
    image_token_id = qwen3_tokenizer.convert_tokens_to_ids("<|image_pad|>")

    # ==================== Load HF model ====================
    hf_model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    hf_model.model.config.image_grid_pinpoints = grid_pinpoints
    hf_model.model.config.vision_aspect_ratio = "single"
    hf_model.model.image_newline = None
    hf_model.config.image_token_id = image_token_id  # Use Qwen3's image token ID
    hf_model.eval()

    # Monkey-patch for disable_anyres mode
    original_image_size_to_num_patches = llava_modeling.image_size_to_num_patches
    llava_modeling.image_size_to_num_patches = lambda *args, **kwargs: 1

    # ==================== HF: Compute golden losses one-by-one ====================
    print(f"\n=== HF Golden Losses (1-by-1) ===")
    hf_golden_losses = []

    try:
        for i, pair in enumerate(test_pairs):
            # Get number of images in this sample from image_sizes
            # image_sizes shape: (2,) for single image [height, width], (num_images, 2) for multi-image
            if pair.hf.image_sizes.ndim == 1:
                num_images = 1  # Single image: shape is (2,) = [height, width]
            else:
                num_images = pair.hf.image_sizes.shape[0]  # Multi-image: shape is (num_images, 2)

            # Convert to torch tensors - use Levanter's input_ids (Qwen3 tokenized)
            lev_seq_len = len(pair.lev.input_ids)
            hf_input_ids = torch.from_numpy(np.array(pair.lev.input_ids[:lev_seq_len])).unsqueeze(0)
            hf_pixel_values = torch.from_numpy(pair.hf.pixel_values)
            hf_image_sizes = torch.from_numpy(pair.hf.image_sizes)
            # Ensure image_sizes has shape (num_images, 2) for HF model
            if hf_image_sizes.dim() == 1:
                hf_image_sizes = hf_image_sizes.unsqueeze(0)  # (2,) -> (1, 2)

            # For disable_anyres with multi-image:
            # pixel_values shape is (num_images, C, H, W) - keep it as-is for HF model
            # HF model expects 4D tensor for multi-image, NOT 5D with batch dimension
            # (5D format is for single-image anyres with multiple patches per image)
            if hf_pixel_values.dim() == 4 and num_images > 1:
                # Multi-image: keep 4D format (num_images, C, H, W) for HF model
                hf_pixel_values = hf_pixel_values[:num_images]
            else:
                # Single image: add batch dimension
                hf_pixel_values = hf_pixel_values.unsqueeze(0)
                # For disable_anyres single image: truncate to base patch only
                if hf_pixel_values.dim() == 5:
                    hf_pixel_values = hf_pixel_values[:, 0:1, :, :, :]

            print(f"  Sample {i}: input_ids={hf_input_ids.shape}, pixel_values={hf_pixel_values.shape}, image_sizes={hf_image_sizes.shape}, num_images={num_images}")


            # Create labels from loss_mask (no shift - HF shifts internally)
            hf_labels = hf_input_ids.clone().long()  # Cast to int64 for PyTorch cross_entropy
            seq_len = hf_input_ids.shape[1]
            loss_mask_np = np.array(pair.lev.loss_mask)[:seq_len]
            mask_tensor = torch.from_numpy(loss_mask_np).unsqueeze(0)
            hf_labels[mask_tensor == 0] = -100

            # Compute HF loss
            with torch.no_grad():
                hf_output = hf_model(
                    input_ids=hf_input_ids,
                    pixel_values=hf_pixel_values,
                    image_sizes=hf_image_sizes,
                    labels=hf_labels,
                    batch_num_images=torch.tensor([num_images]),
                )
                hf_loss = hf_output.loss.item()
                hf_golden_losses.append(hf_loss)
                print(f"  Sample {i}: HF loss = {hf_loss:.6f}")
    finally:
        llava_modeling.image_size_to_num_patches = original_image_size_to_num_patches

    # ==================== Load Levanter model ====================
    lev_config = _load_levanter_config(model_name, enable_flash_attention=False, gradient_checkpointing=False)
    trainer_config = TrainerConfig()

    # Update config with correct image_token_id for Qwen3 tokenizer (already loaded above)
    lev_config = lev_config.with_token_ids(image_token_id=image_token_id)
    print(f"  Updated lev_config.image_token_index to {image_token_id} (<|image_pad|>)")

    with trainer_config.use_device_mesh(), hax.axis_mapping(trainer_config.compute_axis_mapping):
        compute_dtype = jnp.float32
        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=model_name)
        parameter_axis_mapping = trainer_config.parameter_axis_mapping

        lev_model = converter.load_pretrained(
            LlavaOnevisionModel,
            ref=model_name,
            config=lev_config,
            axis_mapping=parameter_axis_mapping,
            dtype=compute_dtype,
            resize_vocab_to_match_tokenizer=False,
        )

        # ==================== Levanter: Create batched tensors inside mesh context ====================
        from jax._src.mesh import get_concrete_mesh
        mesh = get_concrete_mesh()

        _, batched_tensors = prepare_batched_test_data(
            parquet_path=parquet_path,
            sample_indices=list(range(BATCH_SIZE)),
            model_name=model_name,
            max_length=MAX_LENGTH,
            max_num_patches=1,
            grid_pinpoints=grid_pinpoints,
            disable_anyres=True,
            mesh=mesh,
            axis_resources=trainer_config.compute_axis_mapping,
        )

        print(f"\n=== Levanter Batch Processing ===")
        print(f"  Batched input_ids shape: {batched_tensors.input_ids.shape}")
        print(f"  Batched pixel_values shape: {batched_tensors.pixel_values.shape}")

        # batched_tensors is already an ImageTextExample, use it directly
        batch_example = batched_tensors

        # ==================== Levanter: Compute per-example losses ====================
        @eqx.filter_jit
        def compute_batch_losses(model, example):
            return _compute_per_example_vlm_loss(model, example, key=None)

        lev_per_example_losses_named = compute_batch_losses(lev_model, batch_example)
        lev_per_example_losses = np.array(lev_per_example_losses_named.array)
        print(f"  Levanter per-example losses: {lev_per_example_losses}")

        # ==================== Compare losses ====================
        print(f"\n=== Loss Comparison ===")
        all_passed = True
        max_rel_diff = 0.0

        for i in range(BATCH_SIZE):
            hf_loss = hf_golden_losses[i]
            lev_loss = float(lev_per_example_losses[i])
            abs_diff = abs(lev_loss - hf_loss)
            rel_diff = abs_diff / hf_loss

            status = "PASS" if rel_diff < 0.001 else "FAIL"
            print(f"  Sample {i}: HF={hf_loss:.6f}, Lev={lev_loss:.6f}, rel_diff={rel_diff*100:.4f}% [{status}]")

            if rel_diff >= 0.001:
                all_passed = False
            max_rel_diff = max(max_rel_diff, rel_diff)

        print(f"\n  Max relative difference: {max_rel_diff*100:.4f}%")

        assert all_passed, f"Batch loss consistency test failed! Max rel_diff: {max_rel_diff*100:.4f}%"
        print("\n Batch loss consistency test passed!")


@skip_if_no_torch
def test_text_only_conversation():
    """Test BatchImageProcessor with text-only conversations."""
    from transformers import AutoProcessor
    from levanter.data.image import BatchImageProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    bp = BatchImageProcessor(processor, max_length=2048, padding=True)

    messages = [
        {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "The capital of France is Paris."}]},
    ]

    results = bp([{"messages": messages, "images": []}])
    result = results[0]

    assert result["pixel_values"] is None, "Text-only should have None pixel_values"
    assert np.sum(result["loss_mask"] == 1.0) > 0, "Should have unmasked positions for assistant response"


@skip_if_no_torch
def test_replace_tokenizer_qwen3():
    """Test tokenizer replacement with Qwen3 for thinking tokens and image tokens."""
    from transformers import AutoProcessor, AutoTokenizer
    from levanter.data.image import BatchImageProcessor, CustomVLMProcessor
    from PIL import Image

    processor = AutoProcessor.from_pretrained(MODEL_NAME_7B)
    llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    # Create BatchImageProcessor with Qwen3 tokenizer
    bp = BatchImageProcessor(processor, tokenizer=llm_tokenizer, max_length=2048)

    # Verify CustomVLMProcessor is used with new tokenizer
    assert isinstance(bp.processor, CustomVLMProcessor)
    assert bp.processor.tokenizer is llm_tokenizer

    # Verify thinking tokens encode correctly (Qwen3-specific)
    text_with_thinking = "<think>Let me think...</think>Answer is 42."
    original_encoding = processor.tokenizer.encode(text_with_thinking, add_special_tokens=False)
    new_encoding = bp.processor.tokenizer.encode(text_with_thinking, add_special_tokens=False)

    think_token_id = 151667
    end_think_token_id = 151668

    assert think_token_id not in original_encoding, "Original should not have <think> as single token"
    assert think_token_id in new_encoding, "Qwen3 should have <think> as single token"
    assert end_think_token_id in new_encoding, "Qwen3 should have </think> as single token"

    # Verify image token uses Qwen3's <|image_pad|>
    assert bp.processor.image_token == "<|image_pad|>"
    assert bp.processor.image_token_id == llm_tokenizer.convert_tokens_to_ids("<|image_pad|>")

    # Verify processing works with thinking tokens
    test_image = Image.new("RGB", (100, 100), color="blue")
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "<think>Analyzing...</think>A blue square."}]},
    ]

    results = bp([{"messages": messages, "images": [test_image]}])
    result = results[0]

    assert result["pixel_values"] is not None
    assert think_token_id in result["input_ids"]

    # Verify get_token_ids returns updated values
    token_ids = bp.get_token_ids()
    assert token_ids["image_token_id"] == bp.processor.image_token_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
