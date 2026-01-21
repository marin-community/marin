"""
Debug script to trace image token count through the data pipeline.
Run this to find where 2044 tokens come from.
"""

import numpy as np
from PIL import Image
import io
import logging
import sys

# Add lib/levanter/src to path
sys.path.insert(0, "/home/ruili/marin_private3/lib/levanter/src")

# Enable DEBUG logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_image(width: int = 600, height: int = 400):
    """Create a test image with specified dimensions."""
    img = Image.new('RGB', (width, height), color='red')
    return img


def test_processor_token_count():
    """Test CustomVLMProcessor token expansion."""
    print("\n" + "="*80)
    print("STEP 1: Testing CustomVLMProcessor with ANYRES enabled")
    print("="*80)

    from levanter.data.image import CustomVLMProcessor
    from levanter.compat.hf_checkpoints import load_processor
    from transformers import AutoTokenizer, AutoProcessor

    # Load HF processor - use LlavaOnevision specifically
    model_name = "lmms-lab/llava-onevision-qwen2-7b-si"
    print(f"Loading HF processor from: {model_name}")

    # Use AutoProcessor to get the proper LlavaOnevision processor
    hf_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Configure anyres mode
    if hasattr(hf_processor, 'image_processor'):
        # Set anyres mode
        hf_processor.image_processor.vision_aspect_ratio = "anyres_max_9"
        # Set grid pinpoints for anyres
        hf_processor.image_processor.image_grid_pinpoints = [
            [384, 384], [384, 768], [384, 1152], [384, 1536],
            [768, 384], [768, 768], [768, 1152],
            [1152, 384], [1152, 768], [1536, 384]
        ]
        print(f"Configured image_processor for anyres_max_9")

    print(f"HF processor type: {type(hf_processor)}")
    print(f"vision_aspect_ratio: {getattr(hf_processor, 'vision_aspect_ratio', 'N/A')}")
    print(f"num_image_tokens: {getattr(hf_processor, 'num_image_tokens', 'N/A')}")

    # Check image_processor settings
    if hasattr(hf_processor, 'image_processor'):
        ip = hf_processor.image_processor
        print(f"\nImage processor:")
        print(f"  type: {type(ip)}")
        print(f"  vision_aspect_ratio: {getattr(ip, 'vision_aspect_ratio', 'N/A')}")
        print(f"  image_grid_pinpoints: {getattr(ip, 'image_grid_pinpoints', 'N/A')}")

    # Create CustomVLMProcessor with Qwen tokenizer
    print("\nLoading Qwen3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

    print("Creating CustomVLMProcessor...")
    # model_name_or_path auto-detected from hf_processor.tokenizer.name_or_path
    custom_processor = CustomVLMProcessor.from_processor_and_tokenizer(
        hf_processor,
        tokenizer,
        use_full_padded_tokens=True,  # Levanter mode
    )

    print(f"\nCustomVLMProcessor created:")
    print(f"  use_full_padded_tokens: {custom_processor.use_full_padded_tokens}")
    print(f"  num_image_tokens: {custom_processor.num_image_tokens}")
    print(f"  vision_aspect_ratio: {custom_processor.vision_aspect_ratio}")

    # Find image token ID
    image_token_id = None
    for token_name in ["<|image_pad|>", "<image>", "<|vision_start|>", "<image_pad>"]:
        tid = tokenizer.convert_tokens_to_ids(token_name)
        if tid is not None and tid != tokenizer.unk_token_id:
            image_token_id = tid
            print(f"  image_token: '{token_name}' -> id={tid}")
            break

    # Test with different image sizes
    test_sizes = [
        (384, 384),   # Base resolution
        (600, 400),   # Landscape
        (400, 600),   # Portrait
        (800, 600),   # Larger
        (1200, 800),  # Even larger
    ]

    for width, height in test_sizes:
        print(f"\n--- Testing image size: {width}x{height} ---")
        test_image = create_test_image(width, height)

        # Process through CustomVLMProcessor
        text = "<image>\nDescribe this image."

        result = custom_processor(
            images=test_image,
            text=text,
            return_tensors="np",
        )

        # Check pixel_values - analyze shape carefully
        pixel_values = result.get("pixel_values")
        if pixel_values is not None:
            if isinstance(pixel_values, (list, tuple)):
                print(f"pixel_values: list of {len(pixel_values)} items")
                for i, pv in enumerate(pixel_values):
                    shape = pv.shape
                    print(f"  [{i}] shape: {shape}")
                    # Analyze the shape
                    if len(shape) == 3:
                        # Could be (C, H, W) for single patch or (num_patches, H, W) unlikely
                        if shape[0] == 3:
                            print(f"      -> Likely (C, H, W) format - single patch, NOT anyres!")
                        else:
                            print(f"      -> Ambiguous: could be {shape[0]} patches or channels")
                    elif len(shape) == 4:
                        # (num_patches, C, H, W)
                        print(f"      -> (num_patches, C, H, W): {shape[0]} patches")
                    else:
                        print(f"      -> Unexpected shape!")
            else:
                print(f"pixel_values shape: {pixel_values.shape}")
                shape = pixel_values.shape
                if len(shape) == 3 and shape[0] == 3:
                    print(f"  -> (C, H, W) format - single patch, NOT anyres!")
                elif len(shape) == 4:
                    print(f"  -> (num_patches, C, H, W): {shape[0]} patches")

        # Check input_ids for image tokens
        input_ids = result.get("input_ids")
        if input_ids is not None and image_token_id is not None:
            if isinstance(input_ids, np.ndarray):
                if input_ids.ndim == 1:
                    count = int(np.sum(input_ids == image_token_id))
                else:
                    count = int(np.sum(input_ids[0] == image_token_id))
            else:
                count = sum(1 for t in input_ids if t == image_token_id)

            print(f"Image token count in input_ids: {count}")
            print(f"Is multiple of 576? {count % 576 == 0}")
            if count > 0:
                print(f"  = {count // 576} patches × 576 + {count % 576}")
            if count % 576 != 0:
                print(f"  WARNING: {count} is NOT a multiple of 576!")
                print(f"  This suggests HF mode formula: 576 + {count - 576} = {count}")


def test_batch_processor():
    """Test BatchImageProcessor processing."""
    print("\n" + "="*80)
    print("STEP 2: Testing BatchImageProcessor")
    print("="*80)

    from levanter.data.image import CustomVLMProcessor, BatchImageProcessor, _extract_anyres_params
    from levanter.compat.hf_checkpoints import load_processor
    from transformers import AutoTokenizer

    # Load processor
    model_name = "lmms-lab/llava-onevision-qwen2-7b-si"
    hf_processor = load_processor(model_name)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

    custom_processor = CustomVLMProcessor.from_processor_and_tokenizer(
        hf_processor,
        tokenizer,
        use_full_padded_tokens=True,
    )

    # Extract anyres params
    grid_pinpoints, patch_size, vision_feature_height, max_num_patches = _extract_anyres_params(custom_processor)

    print(f"Anyres params:")
    print(f"  grid_pinpoints: {grid_pinpoints}")
    print(f"  patch_size: {patch_size}")
    print(f"  vision_feature_height: {vision_feature_height}")
    print(f"  max_num_patches: {max_num_patches}")

    # Create BatchImageProcessor
    batch_processor = BatchImageProcessor(
        custom_processor,
        max_length=2048,
        padding=True,
        messages_key="messages",
        images_key="images",
        grid_pinpoints=grid_pinpoints,
        patch_size=patch_size,
        vision_feature_height=vision_feature_height,
        max_num_patches=max_num_patches,
    )

    print(f"\nBatchImageProcessor created")
    print(f"  processor type: {type(batch_processor.processor)}")
    print(f"  use_full_padded_tokens: {getattr(batch_processor.processor, 'use_full_padded_tokens', 'N/A')}")

    # Find image token ID
    image_token_id = None
    for token_name in ["<|image_pad|>", "<image>", "<|vision_start|>", "<image_pad>"]:
        tid = tokenizer.convert_tokens_to_ids(token_name)
        if tid is not None and tid != tokenizer.unk_token_id:
            image_token_id = tid
            break

    # Test with a batch item
    test_sizes = [(600, 400), (800, 600), (1200, 800)]

    for width, height in test_sizes:
        print(f"\n--- Testing BatchImageProcessor with {width}x{height} ---")

        test_image = create_test_image(width, height)

        # Save to bytes for loading
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        batch_item = {
            "messages": [
                {"role": "user", "content": "<image>\nDescribe this image."},
                {"role": "assistant", "content": "This is a test image."},
            ],
            "images": [img_bytes],
        }

        result = batch_processor([batch_item])

        if result:
            item = result[0]
            print(f"input_ids shape: {item['input_ids'].shape}")
            print(f"pixel_values shape: {item['pixel_values'].shape}")

            if 'grid_mask' in item and item['grid_mask'] is not None:
                grid_mask = item['grid_mask']
                valid_patches = int(np.sum(grid_mask))
                print(f"grid_mask shape: {grid_mask.shape}")
                print(f"valid patches in grid_mask: {valid_patches}")
                expected_features = valid_patches * 576
                print(f"expected features: {expected_features}")

            # Count image tokens
            input_ids = item['input_ids']
            if image_token_id is not None:
                n_image_tokens = int(np.sum(input_ids == image_token_id))
                print(f"Image tokens in input_ids: {n_image_tokens}")

                # Check mismatch
                if 'grid_mask' in item and item['grid_mask'] is not None:
                    if n_image_tokens != expected_features:
                        print(f"  *** MISMATCH! {n_image_tokens} tokens vs {expected_features} expected ***")
                        print(f"  Difference: {n_image_tokens - expected_features}")
                    else:
                        print(f"  ✓ Match!")


def test_actual_patch_counts_extraction():
    """Test actual_patch_counts extraction from pixel_values."""
    print("\n" + "="*80)
    print("STEP 3: Testing actual_patch_counts extraction")
    print("="*80)

    from levanter.data.image import CustomVLMProcessor
    from levanter.compat.hf_checkpoints import load_processor
    from transformers import AutoTokenizer

    model_name = "lmms-lab/llava-onevision-qwen2-7b-si"
    hf_processor = load_processor(model_name)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

    custom_processor = CustomVLMProcessor.from_processor_and_tokenizer(
        hf_processor,
        tokenizer,
        use_full_padded_tokens=True,
    )

    test_sizes = [(600, 400), (800, 600), (1200, 800)]

    for width, height in test_sizes:
        print(f"\n--- Testing {width}x{height} ---")
        test_image = create_test_image(width, height)

        # Process just the image through image_processor
        image_inputs = custom_processor.image_processor([test_image])

        pixel_values = image_inputs.get("pixel_values")
        print(f"pixel_values type: {type(pixel_values)}")

        actual_patch_counts = None
        if pixel_values is not None:
            if isinstance(pixel_values, (list, tuple)):
                actual_patch_counts = [pv.shape[0] for pv in pixel_values]
                print(f"pixel_values: list of {len(pixel_values)} arrays")
                for i, pv in enumerate(pixel_values):
                    print(f"  [{i}] shape: {pv.shape}")
            elif hasattr(pixel_values, 'shape'):
                actual_patch_counts = [pixel_values.shape[0]]
                print(f"pixel_values shape: {pixel_values.shape}")

        print(f"actual_patch_counts: {actual_patch_counts}")

        if actual_patch_counts:
            expected_tokens = actual_patch_counts[0] * 576
            print(f"Expected tokens (Levanter mode): {expected_tokens}")
            print(f"  = {actual_patch_counts[0]} patches × 576")


def test_different_num_image_tokens():
    """Test with different num_image_tokens values (576 vs 729)."""
    print("\n" + "="*80)
    print("STEP 4: Testing num_image_tokens override")
    print("="*80)

    from levanter.data.image import CustomVLMProcessor
    from levanter.compat.hf_checkpoints import load_processor
    from transformers import AutoTokenizer

    model_name = "lmms-lab/llava-onevision-qwen2-7b-si"
    hf_processor = load_processor(model_name)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

    # Check HF processor's default num_image_tokens
    print(f"HF processor num_image_tokens: {getattr(hf_processor, 'num_image_tokens', 'N/A')}")

    # Override to 576 (for SigLIP with patch_size=16)
    if hasattr(hf_processor, 'num_image_tokens'):
        hf_processor.num_image_tokens = 576
        print(f"After override: {hf_processor.num_image_tokens}")

    custom_processor = CustomVLMProcessor.from_processor_and_tokenizer(
        hf_processor,
        tokenizer,
        use_full_padded_tokens=True,
    )

    print(f"CustomVLMProcessor num_image_tokens: {custom_processor.num_image_tokens}")

    # Test
    test_image = create_test_image(600, 400)
    text = "<image>\nDescribe this image."

    result = custom_processor(
        images=test_image,
        text=text,
        return_tensors="np",
    )

    # Find image token ID
    image_token_id = None
    for token_name in ["<|image_pad|>", "<image>", "<|vision_start|>", "<image_pad>"]:
        tid = tokenizer.convert_tokens_to_ids(token_name)
        if tid is not None and tid != tokenizer.unk_token_id:
            image_token_id = tid
            break

    input_ids = result.get("input_ids")
    if input_ids is not None and image_token_id is not None:
        if isinstance(input_ids, np.ndarray):
            count = int(np.sum(input_ids == image_token_id))
        else:
            count = len([t for t in input_ids if t == image_token_id])

        print(f"Image token count: {count}")
        print(f"Is multiple of 576? {count % 576 == 0}")
        if count % 576 == 0:
            print(f"  = {count // 576} patches × 576")
        print(f"Is multiple of 729? {count % 729 == 0}")


if __name__ == "__main__":
    print("="*80)
    print("Debug script for image token count mismatch")
    print("This will help identify where 2044 tokens come from")
    print("="*80)

    try:
        test_processor_token_count()
    except Exception as e:
        print(f"STEP 1 FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_batch_processor()
    except Exception as e:
        print(f"STEP 2 FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_actual_patch_counts_extraction()
    except Exception as e:
        print(f"STEP 3 FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_different_num_image_tokens()
    except Exception as e:
        print(f"STEP 4 FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("Debug complete. Check output above for mismatches or warnings.")
    print("="*80)
