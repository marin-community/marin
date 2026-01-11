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
    compare_logits_by_region,
    create_lev_jax_tensors,
    DEFAULT_GRID_PINPOINTS,
    get_real_data,
)

from test_utils import skip_if_no_torch

MODEL_NAME = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
MODEL_NAME_7B = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
MAX_LENGTH = 8192


def _load_levanter_config(model_name=MODEL_NAME, enable_flash_attention=False, gradient_checkpointing=True):
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

    return dataclasses.replace(
        lev_config,
        vision_config=vision_config_updated,
        text_config=text_config_updated,
        gradient_checkpointing=gradient_checkpointing,
    )


@pytest.mark.entry
@skip_if_no_torch
def test_vlm_numerical_correctness():
    """Verify numerical correctness of Levanter VLM vs HuggingFace implementation."""
    import torch
    from transformers import AutoModelForVision2Seq
    from levanter.models.llava_onevision import LlavaOnevisionModel
    from levanter.trainer import TrainerConfig

    # Set JAX to use float32 matmul precision
    jax.config.update("jax_default_matmul_precision", "float32")

    model_name = MODEL_NAME
    grid_pinpoints = DEFAULT_GRID_PINPOINTS
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
            max_num_patches=9,
            grid_pinpoints=grid_pinpoints,
        )

    hf_model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    hf_model.model.config.image_grid_pinpoints = grid_pinpoints
    hf_model.model.image_newline = None
    hf_model.eval()

    lev_config = _load_levanter_config(model_name, enable_flash_attention=False, gradient_checkpointing=False)
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

        for sample_idx, pair in enumerate(test_pairs):
            hf_input_ids = torch.from_numpy(pair.hf.input_ids).unsqueeze(0)
            hf_pixel_values = torch.from_numpy(pair.hf.pixel_values).unsqueeze(0)
            hf_image_sizes = torch.from_numpy(pair.hf.image_sizes).unsqueeze(0)

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

            image_token_id = hf_model.config.image_token_index
            result = compare_logits_by_region(
                hf_logits=hf_logits,
                lev_logits=lev_logits_np,
                input_ids=pair.hf.input_ids,
                image_token_id=image_token_id,
                tolerance=1e-3,
                verbose=False,
            )
            all_passed.append(result.passed)

        assert all(all_passed), f"Not all samples passed: {sum(all_passed)}/{len(all_passed)}"


@pytest.mark.entry
@skip_if_no_torch
def test_vlm_loss_and_gradients():
    """Test loss computation and gradient flow through all VLM components."""
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
            max_num_patches=9,
            grid_pinpoints=DEFAULT_GRID_PINPOINTS,
        )
        pair = test_pairs[0]

    lev_config = _load_levanter_config(model_name, enable_flash_attention=True, gradient_checkpointing=True)
    trainer_config = TrainerConfig(per_device_parallelism=1)

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
            loss = compute_vlm_loss(model, batch_example, key=None)
            return loss.scalar()

        loss_value, grads = eqx.filter_value_and_grad(compute_loss)(lev_model)
        loss_value = float(loss_value)

        # Verify loss is reasonable
        assert not np.isnan(loss_value), "Loss is NaN"
        assert 0.0 < loss_value < 100.0, f"Loss out of range: {loss_value}"

        # Verify gradients flow to all components
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


@pytest.mark.entry
@skip_if_no_torch
def test_vlm_training_reproducibility():
    """Verify training steps are reproducible with same seed."""
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
            max_num_patches=9,
            grid_pinpoints=DEFAULT_GRID_PINPOINTS,
        )
        pair = test_pairs[0]

    lev_config = _load_levanter_config(model_name, enable_flash_attention=True, gradient_checkpointing=True)
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
    """Verify loss masking correctly excludes user prompts."""
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
            max_num_patches=9,
            grid_pinpoints=DEFAULT_GRID_PINPOINTS,
        )
        pair = test_pairs[0]

    loss_mask_np = np.array(pair.lev.loss_mask)
    total_positions = len(loss_mask_np)
    unmasked_positions = np.sum(loss_mask_np == 1.0)

    # Verify unmasked positions exist (assistant response)
    assert unmasked_positions > 0, "No unmasked positions - training would have no signal"
    # Verify most positions are masked (user prompt + image tokens + padding)
    assert unmasked_positions < total_positions * 0.5, "Too many unmasked positions"


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
