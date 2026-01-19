#!/usr/bin/env python
"""Debug script to investigate loss mismatch for multi-image samples."""

import sys
sys.path.insert(0, "/home/ruili/marin_private3/lib/levanter/tests")

import numpy as np
import tempfile
import torch
import jax
import jax.numpy as jnp
import haliax as hax
import equinox as eqx
import transformers.models.llava_onevision.modeling_llava_onevision as llava_modeling
from transformers import AutoModelForVision2Seq, AutoConfig

from test_image_utils import (
    prepare_test_data,
    get_interleaved_data,
    SINGLE_PATCH_GRID_PINPOINTS,
)

MODEL_NAME = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
MAX_LENGTH = 8192

jax.config.update("jax_default_matmul_precision", "float32")


def main():
    # Load data
    tmpdir = tempfile.mkdtemp()
    hf_dataset = get_interleaved_data(num_samples=16)
    parquet_path = f"{tmpdir}/test_data.parquet"
    hf_dataset.to_parquet(parquet_path)

    # Find sample with 2 images (Sample 3)
    sample_idx = 3

    test_pairs = prepare_test_data(
        parquet_path=parquet_path,
        sample_indices=[sample_idx],
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
        max_num_patches=2,  # For 2 images
        grid_pinpoints=SINGLE_PATCH_GRID_PINPOINTS,
        disable_anyres=True,
    )
    pair = test_pairs[0]

    print(f"=== Sample {sample_idx} Analysis ===")
    print(f"HF input_ids shape: {pair.hf.input_ids.shape}")
    print(f"HF pixel_values shape: {pair.hf.pixel_values.shape}")
    print(f"HF image_sizes: {pair.hf.image_sizes}")
    print(f"Lev input_ids shape: {pair.lev.input_ids.shape}")
    print(f"Lev pixel_values shape: {pair.lev.pixel_values.shape}")
    print(f"Lev grid_mask: {pair.lev.grid_mask}")

    # Check if input_ids match
    hf_ids = pair.hf.input_ids
    lev_ids = pair.lev.input_ids[:len(hf_ids)]
    ids_match = np.all(hf_ids == lev_ids)
    print(f"\nInput IDs match: {ids_match}")
    if not ids_match:
        mismatch_pos = np.where(hf_ids != lev_ids)[0]
        print(f"Mismatch positions: {mismatch_pos[:20]}")
        print(f"HF at mismatch: {hf_ids[mismatch_pos[:10]]}")
        print(f"Lev at mismatch: {lev_ids[mismatch_pos[:10]]}")

    # Get number of images
    if pair.hf.image_sizes.ndim == 1:
        num_images = 1
    else:
        num_images = pair.hf.image_sizes.shape[0]
    print(f"\nNumber of images: {num_images}")

    # Load HF model
    hf_model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    hf_model.model.config.image_grid_pinpoints = SINGLE_PATCH_GRID_PINPOINTS
    hf_model.model.config.vision_aspect_ratio = "single"
    hf_model.model.image_newline = None
    hf_model.eval()

    # Prepare HF inputs
    hf_input_ids = torch.from_numpy(pair.hf.input_ids).unsqueeze(0)
    hf_pixel_values = torch.from_numpy(pair.hf.pixel_values)
    hf_image_sizes = torch.from_numpy(pair.hf.image_sizes)
    if hf_image_sizes.dim() == 1:
        hf_image_sizes = hf_image_sizes.unsqueeze(0)

    # For multi-image: keep 4D format
    if hf_pixel_values.dim() == 4 and num_images > 1:
        hf_pixel_values = hf_pixel_values[:num_images]
    else:
        hf_pixel_values = hf_pixel_values.unsqueeze(0)
        if hf_pixel_values.dim() == 5:
            hf_pixel_values = hf_pixel_values[:, 0:1, :, :, :]

    print(f"\nHF model inputs:")
    print(f"  input_ids: {hf_input_ids.shape}")
    print(f"  pixel_values: {hf_pixel_values.shape}")
    print(f"  image_sizes: {hf_image_sizes.shape}")

    # Monkey-patch for disable_anyres
    original_fn = llava_modeling.image_size_to_num_patches
    llava_modeling.image_size_to_num_patches = lambda *args, **kwargs: 1

    try:
        # Get HF image features
        with torch.no_grad():
            hf_image_features = hf_model.get_image_features(
                pixel_values=hf_pixel_values,
                image_sizes=hf_image_sizes,
            )
            print(f"\n=== HF Image Features ===")
            for i, feat in enumerate(hf_image_features):
                print(f"  Image {i}: shape={feat.shape}, mean={feat.mean():.6f}, std={feat.std():.6f}")

            # Get per-token logits
            hf_output = hf_model(
                input_ids=hf_input_ids,
                pixel_values=hf_pixel_values,
                image_sizes=hf_image_sizes,
                batch_num_images=torch.tensor([num_images]),
            )
            hf_logits = hf_output.logits[0]  # (seq_len, vocab)
            print(f"\nHF logits shape: {hf_logits.shape}")

        # Create HF labels and compute loss
        hf_labels = hf_input_ids.clone()
        seq_len = hf_input_ids.shape[1]
        loss_mask_np = np.array(pair.lev.loss_mask)[:seq_len]
        mask_tensor = torch.from_numpy(loss_mask_np).unsqueeze(0)
        hf_labels[mask_tensor == 0] = -100

        with torch.no_grad():
            hf_output_with_loss = hf_model(
                input_ids=hf_input_ids,
                pixel_values=hf_pixel_values,
                image_sizes=hf_image_sizes,
                labels=hf_labels,
                batch_num_images=torch.tensor([num_images]),
            )
            hf_loss = hf_output_with_loss.loss.item()
            print(f"\nHF Loss: {hf_loss:.6f}")

        # Compute per-token CE loss for HF
        hf_shift_logits = hf_logits[:-1]
        hf_shift_labels = hf_input_ids[0, 1:]
        hf_ce = torch.nn.functional.cross_entropy(hf_shift_logits, hf_shift_labels, reduction='none')

        # Get valid positions from loss_mask
        shifted_mask = np.roll(loss_mask_np, -1)
        shifted_mask[-1] = 0
        valid_positions = np.where(shifted_mask[:-1] > 0)[0]

        print(f"\n=== Per-Token Loss Analysis ===")
        print(f"Valid loss positions: {len(valid_positions)}")
        print(f"First 10 valid positions: {valid_positions[:10]}")

        hf_ce_valid = hf_ce.numpy()[valid_positions]
        print(f"HF CE at valid positions: mean={hf_ce_valid.mean():.6f}, std={hf_ce_valid.std():.6f}")
        print(f"HF CE first 10: {hf_ce_valid[:10]}")

    finally:
        llava_modeling.image_size_to_num_patches = original_fn

    # Now load Levanter model and compute loss
    from levanter.models.llava_onevision import LlavaOnevisionConfig, LlavaOnevisionModel
    from levanter.trainer import TrainerConfig
    from levanter.main.train_vlm import compute_vlm_loss
    from test_image_utils import create_lev_jax_tensors
    from levanter.data.image import ImageTextExample as ImgTextEx
    import dataclasses
    from levanter.layers.attention import AttentionBackend

    hf_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    lev_config = LlavaOnevisionConfig.from_hf_config(hf_config)

    vision_config_updated = dataclasses.replace(
        lev_config.vision_config,
        use_flash_attention=False,
        gradient_checkpointing=False,
    )
    text_config_updated = dataclasses.replace(
        lev_config.text_config,
        attn_backend=AttentionBackend.VANILLA,
        gradient_checkpointing=False,
    )
    lev_config = dataclasses.replace(
        lev_config,
        vision_config=vision_config_updated,
        text_config=text_config_updated,
    )

    # Use tensor parallelism instead of data parallelism for batch_size=1
    # This avoids the "batch size not divisible by num devices" error
    from levanter.utils.mesh import MeshConfig
    mesh_config = MeshConfig(
        axes={"model": -1},
        shared_mapping={"mlp": "model", "heads": "replica"},
    )
    trainer_config = TrainerConfig(mesh=mesh_config)

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

        # Create Levanter tensors
        jax_tensors = create_lev_jax_tensors(pair.lev, batch_size=1)

        print(f"\n=== Levanter Tensors ===")
        print(f"input_ids shape: {jax_tensors.input_ids.axes}")
        print(f"pixel_values shape: {jax_tensors.pixel_values.axes}")
        print(f"grid_mask: {np.array(jax_tensors.grid_mask.array)}")

        # Get Levanter image features
        lev_features, _ = lev_model.get_image_features(
            jax_tensors.pixel_values,
            jax_tensors.grid_mask,
            key=None,
        )
        lev_features_np = np.array(lev_features.array)
        print(f"\n=== Levanter Image Features ===")
        print(f"Shape: {lev_features_np.shape}")  # (batch, num_patches, features_per_patch, embed)

        # Get valid features based on grid_mask
        grid_mask_np = np.array(jax_tensors.grid_mask.array)
        batch_size_np, num_patches, features_per_patch, embed_dim = lev_features_np.shape

        for img_idx in range(num_images):
            if grid_mask_np[0, img_idx]:
                img_features = lev_features_np[0, img_idx]  # (features_per_patch, embed)
                print(f"  Image {img_idx}: shape={img_features.shape}, mean={img_features.mean():.6f}, std={img_features.std():.6f}")

        # Compare image features with HF
        print(f"\n=== Image Feature Comparison ===")
        hf_all_features = torch.cat(hf_image_features, dim=0).numpy()  # (total_tokens, embed)

        # Levanter features: reshape to match HF
        lev_valid_features = []
        for img_idx in range(num_images):
            if grid_mask_np[0, img_idx]:
                lev_valid_features.append(lev_features_np[0, img_idx].reshape(-1, embed_dim))
        lev_all_features = np.concatenate(lev_valid_features, axis=0)  # (total_tokens, embed)

        print(f"HF features shape: {hf_all_features.shape}")
        print(f"Lev features shape: {lev_all_features.shape}")

        if hf_all_features.shape == lev_all_features.shape:
            abs_diff = np.abs(hf_all_features - lev_all_features)
            print(f"Max abs diff: {abs_diff.max():.6f}")
            print(f"Mean abs diff: {abs_diff.mean():.6f}")

            # Per-image comparison
            tokens_per_image = 729  # 27x27
            for img_idx in range(num_images):
                start = img_idx * tokens_per_image
                end = (img_idx + 1) * tokens_per_image
                img_diff = abs_diff[start:end]
                print(f"  Image {img_idx}: max_diff={img_diff.max():.6f}, mean_diff={img_diff.mean():.6f}")

        # Compute Levanter loss
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

        lev_loss = compute_loss(lev_model)
        print(f"\n=== Loss Comparison ===")
        print(f"HF Loss: {hf_loss:.6f}")
        print(f"Lev Loss: {float(lev_loss):.6f}")
        print(f"Abs diff: {abs(hf_loss - float(lev_loss)):.6f}")
        print(f"Rel diff: {abs(hf_loss - float(lev_loss)) / hf_loss * 100:.4f}%")

        # Get Levanter logits for detailed comparison
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
        print(f"\nLev logits shape: {lev_logits_np.shape}")

        # Compare logits
        print(f"\n=== Logits Comparison ===")
        hf_logits_np = hf_logits.numpy()
        logits_diff = np.abs(hf_logits_np - lev_logits_np)
        print(f"Max logits diff: {logits_diff.max():.6f}")
        print(f"Mean logits diff: {logits_diff.mean():.6f}")

        # Find positions with largest logits diff
        max_diff_per_pos = logits_diff.max(axis=1)
        top_diff_positions = np.argsort(max_diff_per_pos)[-10:][::-1]
        print(f"\nTop 10 positions with largest logits diff:")
        for pos in top_diff_positions:
            print(f"  Position {pos}: max_diff={max_diff_per_pos[pos]:.6f}")

        # Check if these positions are near image tokens
        image_token_id = hf_model.config.image_token_index
        image_positions = np.where(pair.hf.input_ids == image_token_id)[0]
        print(f"\nImage token positions: {image_positions[:20]}... (total: {len(image_positions)})")

        # Compute per-token Levanter CE
        lev_shift_logits = lev_logits_np[:-1]
        lev_shift_labels = np.array(batch_example.input_ids.array)[0, 1:seq_len]

        # Manual CE computation
        lev_log_softmax = lev_shift_logits - np.log(np.exp(lev_shift_logits).sum(axis=-1, keepdims=True) + 1e-10)
        lev_ce = -lev_log_softmax[np.arange(len(lev_shift_labels)), lev_shift_labels]

        lev_ce_valid = lev_ce[valid_positions]
        print(f"\n=== Per-Token CE Comparison at Valid Positions ===")
        print(f"HF CE: mean={hf_ce_valid.mean():.6f}, std={hf_ce_valid.std():.6f}")
        print(f"Lev CE: mean={lev_ce_valid.mean():.6f}, std={lev_ce_valid.std():.6f}")

        ce_diff = np.abs(hf_ce_valid - lev_ce_valid)
        print(f"CE diff: max={ce_diff.max():.6f}, mean={ce_diff.mean():.6f}")

        # Find positions with largest CE diff
        top_ce_diff_idx = np.argsort(ce_diff)[-10:][::-1]
        print(f"\nTop 10 positions with largest CE diff:")
        for idx in top_ce_diff_idx:
            pos = valid_positions[idx]
            print(f"  Position {pos}: HF_CE={hf_ce_valid[idx]:.6f}, Lev_CE={lev_ce_valid[idx]:.6f}, diff={ce_diff[idx]:.6f}")


if __name__ == "__main__":
    main()
