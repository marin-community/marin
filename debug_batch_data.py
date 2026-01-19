#!/usr/bin/env python
"""Script to inspect batch data from ImageDataLoader using real interleaved data."""

import sys
sys.path.insert(0, "/home/ruili/marin_private3/lib/levanter/tests")

import numpy as np
import tempfile
from test_image_utils import prepare_batched_test_data, get_interleaved_data, SINGLE_PATCH_GRID_PINPOINTS
from transformers import AutoTokenizer

QWEN3_MODEL = "Qwen/Qwen3-0.6B"


def print_array_info(name: str, arr, max_print: int = 20):
    """Print array info with sample values."""
    print(f"\n{'='*60}")
    print(f"Field: {name}")
    print(f"{'='*60}")

    if hasattr(arr, 'array'):
        # NamedArray
        np_arr = np.array(arr.array)
        print(f"  Shape: {arr.array.shape}")
        print(f"  Axes: {[ax.name for ax in arr.axes]}")
    else:
        np_arr = np.array(arr)
        print(f"  Shape: {np_arr.shape}")

    print(f"  Dtype: {np_arr.dtype}")
    print(f"  Min: {np_arr.min()}, Max: {np_arr.max()}")

    # Print values per batch item
    if np_arr.ndim >= 2:
        batch_size = np_arr.shape[0]
        for b in range(batch_size):
            item = np_arr[b]
            flat = item.flatten()
            if len(flat) > max_print:
                print(f"  [Batch {b}] First {max_print}: {flat[:max_print]}")
                print(f"  [Batch {b}] Last {max_print}: {flat[-max_print:]}")
            else:
                print(f"  [Batch {b}] Values: {flat}")

            # For sequence data, show where non-zero ends
            if np_arr.ndim == 2:
                nonzero_mask = item != 0
                if nonzero_mask.any():
                    last_nonzero = np.where(nonzero_mask)[0][-1]
                    print(f"  [Batch {b}] Last non-zero index: {last_nonzero}, actual_seq_len: {last_nonzero + 1}")
    else:
        print(f"  Values: {np_arr}")


def main():
    print("="*60)
    print("LOADING INTERLEAVED DATA")
    print("="*60)

    # Load interleaved data and save to temp parquet
    sample_indices = [0, 1, 2, 3]
    hf_dataset = get_interleaved_data(num_samples=len(sample_indices))

    # Print image count per example from original dataset
    print("\n" + "="*60)
    print("IMAGE COUNT PER EXAMPLE (from original dataset)")
    print("="*60)
    for i, idx in enumerate(sample_indices):
        images = hf_dataset[idx]["images"]
        num_images = len(images) if images is not None else 0
        print(f"  Sample {idx}: {num_images} image(s)")

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = f"{tmpdir}/test_data.parquet"
        hf_dataset.to_parquet(parquet_path)

        print(f"Parquet path: {parquet_path}")
        print(f"Sample indices: {sample_indices}")
        print(f"Grid pinpoints: {SINGLE_PATCH_GRID_PINPOINTS} (disable_anyres mode)")

        test_pairs, batch = prepare_batched_test_data(
            parquet_path=parquet_path,
            sample_indices=sample_indices,
            max_length=4096,
            grid_pinpoints=SINGLE_PATCH_GRID_PINPOINTS,
            max_num_patches=1,
            disable_anyres=True,
        )

    print(f"\nLoaded {len(test_pairs)} test pairs")

    # Print original sequence lengths from test_pairs
    print("\n" + "="*60)
    print("ORIGINAL SEQUENCE LENGTHS (before batching)")
    print("="*60)
    for i, pair in enumerate(test_pairs):
        orig_len = len(pair.lev_dict["input_ids"])
        print(f"  Sample {i}: original seq_len = {orig_len}")

    # Print batch info
    print("\n" + "="*60)
    print("BATCH OUTPUT FROM ImageDataLoader")
    print("="*60)

    print_array_info("input_ids", batch.input_ids)
    print_array_info("loss_mask", batch.loss_mask)
    print_array_info("combined_mask", batch.combined_mask)
    print_array_info("position_ids", batch.position_ids)
    print_array_info("grid_mask", batch.grid_mask)

    # Pixel values - just print shape and sample
    print(f"\n{'='*60}")
    print("Field: pixel_values")
    print(f"{'='*60}")
    pv = np.array(batch.pixel_values.array)
    print(f"  Shape: {pv.shape}")
    print(f"  Axes: {[ax.name for ax in batch.pixel_values.axes]}")
    for b in range(min(pv.shape[0], len(test_pairs))):
        print(f"  [Batch {b}] pixel_values mean: {pv[b].mean():.4f}, std: {pv[b].std():.4f}")

    # Verify padding correctness
    print("\n" + "="*60)
    print("PADDING VERIFICATION")
    print("="*60)

    input_ids_np = np.array(batch.input_ids.array)
    loss_mask_np = np.array(batch.loss_mask.array)
    combined_mask_np = np.array(batch.combined_mask.array)
    position_ids_np = np.array(batch.position_ids.array)

    for i, pair in enumerate(test_pairs):
        orig_input_ids = pair.lev_dict["input_ids"]
        orig_loss_mask = pair.lev_dict["loss_mask"]
        orig_len = len(orig_input_ids)
        target_len = input_ids_np.shape[1]

        print(f"\nBatch {i} (original seq_len={orig_len}, target_len={target_len}):")

        # Check input_ids
        original_preserved = np.all(input_ids_np[i, :orig_len] == orig_input_ids)
        padding_zero = np.all(input_ids_np[i, orig_len:] == 0) if orig_len < target_len else True
        print(f"  input_ids: original preserved={original_preserved}, padding zero={padding_zero}")

        # Check loss_mask
        loss_original = np.all(loss_mask_np[i, :orig_len] == orig_loss_mask)
        loss_padding = np.all(loss_mask_np[i, orig_len:] == 0.0) if orig_len < target_len else True
        print(f"  loss_mask: original preserved={loss_original}, padding zero={loss_padding}")

        # Show some actual token values
        print(f"  First 10 input_ids: {input_ids_np[i, :10]}")
        print(f"  Last 10 input_ids (before padding): {input_ids_np[i, max(0, orig_len-10):orig_len]}")
        if orig_len < target_len:
            print(f"  First 10 padding positions: {input_ids_np[i, orig_len:min(orig_len+10, target_len)]}")

    # Text visualization using Qwen3 tokenizer
    print("\n" + "="*60)
    print("TEXT VISUALIZATION (decoded with Qwen3 tokenizer)")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL, trust_remote_code=True)

    def compress_repeated_tokens(ids: list, tokenizer) -> str:
        """Compress repeated tokens into 'N × token' format."""
        if not ids:
            return ""

        result_parts = []
        i = 0
        while i < len(ids):
            current_id = ids[i]
            count = 1
            # Count consecutive identical tokens
            while i + count < len(ids) and ids[i + count] == current_id:
                count += 1

            token_str = tokenizer.decode([current_id], skip_special_tokens=False)
            token_str_escaped = token_str.replace("\n", "\\n").replace("\t", "\\t")

            if count >= 3:
                # Compress repeated tokens
                result_parts.append(f"{count} × {token_str_escaped}")
            else:
                # Show individual tokens
                for _ in range(count):
                    result_parts.append(token_str_escaped)

            i += count

        return "".join(result_parts)

    # Visualize tokens where masks == 1
    for i in range(len(test_pairs)):
        orig_len = len(test_pairs[i].lev_dict["input_ids"])
        ids = input_ids_np[i, :orig_len]
        loss_mask = loss_mask_np[i, :orig_len]
        combined_mask = combined_mask_np[i, :orig_len]

        print(f"\n--- Sample {i} ---")

        # Tokens where loss_mask == 1
        loss_masked_ids = ids[loss_mask == 1].tolist()
        print(f"loss_mask=1 ({len(loss_masked_ids)} tokens):")
        print(compress_repeated_tokens(loss_masked_ids, tokenizer))

        # Tokens where combined_mask == 1
        combined_masked_ids = ids[combined_mask == 1].tolist()
        print(f"combined_mask=1 ({len(combined_masked_ids)} tokens):")
        print(compress_repeated_tokens(combined_masked_ids, tokenizer))

    print("\nDone!")


if __name__ == "__main__":
    main()
