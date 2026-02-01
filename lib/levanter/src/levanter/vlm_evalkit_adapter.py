# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
VLMEvalKit Adapter for Levanter VLM Models.

This module provides an adapter to use Levanter VLM models with the VLMEvalKit
evaluation framework (https://github.com/open-compass/VLMEvalKit).

VLMEvalKit requires implementing the `generate_inner()` method. All other
functionality (data loading, prompt formatting, metric calculation) is handled
by VLMEvalKit.

Example usage:
    from levanter.vlm_evalkit_adapter import LevanterVLM
    from vlmeval.evaluate import Evaluator

    model = LevanterVLM(
        levanter_model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        EvalBatch=Batch,
        EvalPos=Pos,
        axis_resources=axis_mapping,
        mp=mp,
    )

    # Use VLMEvalKit's Evaluator
    results = Evaluator(model=model, datasets=["MME", "GQA"]).run()
"""

import logging
from io import BytesIO
from typing import List, Optional, Union

import jax.numpy as jnp
import jax.random as jrandom
import jmp
from PIL import Image

import haliax as hax
from haliax import Axis, NamedArray
from haliax.partitioning import ResourceMapping

from levanter.data.image import BatchImageProcessor
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.models.llava_onevision import LlavaInferenceEngine, LlavaOnevisionModel, VLMRequest
from levanter.utils.hf_utils import HfTokenizer

try:
    from vlmeval.vlm.base import BaseModel
except ImportError:
    # VLMEvalKit not installed, create a placeholder
    class BaseModel:
        """Placeholder when VLMEvalKit is not installed."""
        INTERLEAVE = True

        def generate(self, msgs, dataset=None):
            raise NotImplementedError("VLMEvalKit is not installed")


logger = logging.getLogger(__name__)


class LevanterVLM(BaseModel):
    """VLMEvalKit adapter for Levanter VLM models.

    This adapter wraps a Levanter LlavaOnevisionModel to be compatible with
    VLMEvalKit's evaluation framework.

    Attributes:
        INTERLEAVE: Whether the model supports interleaved image-text input.
            Set to True for LLaVA-style models.
    """

    INTERLEAVE = True

    def __init__(
        self,
        levanter_model: LlavaOnevisionModel,
        tokenizer: HfTokenizer,
        image_processor: BatchImageProcessor,
        EvalBatch: Axis,
        EvalPos: Axis,
        axis_resources: ResourceMapping,
        mp: jmp.Policy,
        max_gen_toks: int = 512,
        temperature: float = 0.0,
        vlm_batch_size: int = 1,
        **kwargs,
    ):
        """Initialize the LevanterVLM adapter.

        Args:
            levanter_model: The Levanter LlavaOnevisionModel to wrap.
            tokenizer: The tokenizer for the model.
            image_processor: BatchImageProcessor for processing images.
            EvalBatch: Batch axis for evaluation.
            EvalPos: Position axis for evaluation.
            axis_resources: Resource mapping for sharding.
            mp: Mixed precision policy.
            max_gen_toks: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 for greedy).
            vlm_batch_size: Number of requests to process in parallel.
        """
        self.model = levanter_model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.EvalBatch = EvalBatch
        self.EvalPos = EvalPos
        self.axis_resources = axis_resources
        self.mp = mp
        self.max_gen_toks = max_gen_toks
        self.temperature = temperature
        self.vlm_batch_size = vlm_batch_size

        # Create inference engine
        self._engine = None
        self._init_engine()

    def _init_engine(self):
        """Initialize the LlavaInferenceEngine."""
        from levanter.inference.engine import InferenceEngineConfig

        max_seq_len = getattr(self.EvalPos, 'size', 4096)

        # Match vlm_eval_harness.py settings to avoid OOM
        # See vlm_eval_harness.py:398-409 for reference
        max_seqs = max(1, self.vlm_batch_size)
        engine_config = InferenceEngineConfig(
            max_seq_len=max_seq_len,
            max_seqs=max_seqs,  # Use vlm_batch_size for batched inference
            max_seqs_in_prefill=max_seqs,
            page_size=8,  # Smaller page size
            compute_dtype=jnp.bfloat16,
            hbm_utilization=0.5,  # Only use 50% HBM for KV cache
        )

        self._engine = LlavaInferenceEngine.from_model_with_config(
            model=self.model,
            tokenizer=self.tokenizer,
            config=engine_config,
            Vocab=self.model.Vocab,  # Use model's Vocab axis, not tokenizer length
            mesh=hax.partitioning._get_mesh(),
        )

    def _load_image(self, image_source: Union[str, bytes, Image.Image]) -> Image.Image:
        """Load an image from various sources.

        Args:
            image_source: Can be a file path, URL, bytes, or PIL Image.

        Returns:
            PIL Image in RGB format.
        """
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        elif isinstance(image_source, str):
            # File path or URL
            if image_source.startswith(("http://", "https://")):
                import requests
                response = requests.get(image_source, timeout=10)
                return Image.open(BytesIO(response.content)).convert("RGB")
            else:
                return Image.open(image_source).convert("RGB")
        elif isinstance(image_source, bytes):
            return Image.open(BytesIO(image_source)).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_source)}")

    def _parse_messages(self, msgs: List[dict]) -> tuple[str, List[Image.Image]]:
        """Parse VLMEvalKit message format into prompt and images.

        VLMEvalKit message format:
            [
                {"type": "image", "value": "path/to/image.jpg"},
                {"type": "text", "value": "What is in this image?"},
            ]

        Args:
            msgs: List of message dictionaries with 'type' and 'value' keys.

        Returns:
            Tuple of (prompt_text, list_of_images).
        """
        images = []
        text_parts = []

        for msg in msgs:
            msg_type = msg.get("type", "")
            value = msg.get("value", "")

            if msg_type == "image":
                images.append(self._load_image(value))
                # Add image placeholder to text
                text_parts.append("<image>")
            elif msg_type == "text":
                text_parts.append(value)

        prompt = "\n".join(text_parts)

        # Debug: log prompt and image count
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"_parse_messages: {len(images)} images, prompt preview: {prompt[:200]}...")

        return prompt, images

    def _create_vlm_request(
        self,
        prompt: str,
        images: List[Image.Image],
        request_id: int = 0,
    ) -> VLMRequest:
        """Create a VLMRequest from prompt and images.

        Args:
            prompt: The text prompt with <image> placeholders.
            images: List of PIL images.
            request_id: Request identifier.

        Returns:
            VLMRequest object for inference.
        """
        # Create messages in the format expected by image_processor
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]

        # Process with image processor
        example = {"messages": messages, "images": images}
        processed = self.image_processor([example])[0]

        # Extract processed data
        input_ids = processed["input_ids"]
        pixel_values = processed.get("pixel_values")
        grid_mask = processed.get("grid_mask")
        unpad_indices = processed.get("unpad_indices")
        num_unpadded_features = processed.get("num_unpadded_features")

        # Debug: log processed data info
        import logging
        logger = logging.getLogger(__name__)

        # Check for image tokens in input_ids
        # Image token ID is typically 151655 for Qwen2-VL/LLaVA-OneVision
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        num_image_tokens = sum(1 for t in input_ids if t == image_token_id)

        # Also decode first 100 tokens to see the prompt structure
        first_tokens = self.tokenizer.decode(input_ids[:100], skip_special_tokens=False)

        logger.info(f"_create_vlm_request: input_ids len={len(input_ids)}, "
                    f"pixel_values shape={pixel_values.shape if pixel_values is not None else None}, "
                    f"num_images={len(images)}, "
                    f"image_token_id={image_token_id}, num_image_tokens={num_image_tokens}")
        logger.info(f"First 100 tokens decoded: {first_tokens[:200]}...")

        # Convert input_ids to list
        input_ids_list = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)

        # Create sequence decoding parameters
        base_key = jrandom.PRNGKey(42)

        # Build stop tokens
        stop_token_ids = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
            try:
                im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
                if im_end_id is not None and im_end_id != self.tokenizer.unk_token_id:
                    stop_token_ids.append(im_end_id)
            except Exception:
                pass

        stop_tokens_array = jnp.array([[token_id] for token_id in stop_token_ids], dtype=jnp.int32)
        stop_tokens_na = hax.named(stop_tokens_array, axis=("stop_seq", "position"))

        seq_params = SeqDecodingParams(
            max_num_tokens=jnp.array(len(input_ids_list) + self.max_gen_toks, dtype=jnp.int32),
            stop_tokens=stop_tokens_na,
            temperature=jnp.array(self.temperature, dtype=jnp.float32),
            key=base_key,
        )

        # Convert to NamedArray format
        Batch = hax.Axis("batch", 1)
        TotalPatches = hax.Axis("TotalPatches", pixel_values.shape[0]) if pixel_values is not None else None

        pixel_values_na = None
        grid_mask_na = None
        unpad_indices_na = None
        input_ids_na = None

        if pixel_values is not None:
            pixel_values_na = hax.named(
                jnp.array(pixel_values)[None, ...],
                (Batch, TotalPatches, hax.Axis("channels", pixel_values.shape[1]),
                 hax.Axis("height", pixel_values.shape[2]), hax.Axis("width", pixel_values.shape[3]))
            )

        if grid_mask is not None:
            grid_mask_na = hax.named(jnp.array(grid_mask)[None, ...], (Batch, TotalPatches,))

        if unpad_indices is not None:
            UnpadFeatures = hax.Axis("UnpadFeatures", len(unpad_indices))
            unpad_indices_na = hax.named(jnp.array(unpad_indices), (UnpadFeatures,))

        Position = hax.Axis("position", len(input_ids_list))
        input_ids_array = jnp.array(input_ids_list, dtype=jnp.int32).reshape(1, -1)
        input_ids_na = hax.named(input_ids_array, (Batch, Position))

        return VLMRequest(
            prompt_tokens=input_ids_list,
            request_id=request_id,
            decode_params=seq_params,
            n_generations=1,
            pixel_values=pixel_values_na,
            grid_mask=grid_mask_na,
            input_ids=input_ids_na,
            unpad_indices=unpad_indices_na,
            num_unpadded_features=num_unpadded_features,
        )

    def generate_inner(self, msgs: List[dict], dataset: Optional[str] = None) -> str:
        """Generate a response for the given multimodal messages.

        This is the main method required by VLMEvalKit. It takes multimodal
        messages and returns the model's generated response.

        Args:
            msgs: List of message dictionaries. Each dict has:
                - "type": "image" or "text"
                - "value": image path/URL/bytes or text string
            dataset: Optional dataset name for dataset-specific handling.

        Returns:
            The generated text response.
        """
        # Parse messages into prompt and images
        prompt, images = self._parse_messages(msgs)

        if not images:
            logger.warning("No images found in messages, using text-only prompt")

        # Create VLM request
        vlm_request = self._create_vlm_request(prompt, images)

        # Generate
        result = self._engine.generate([vlm_request])

        # Decode generated tokens
        if result.tokens and len(result.tokens) > 0:
            generated_tokens = result.tokens[0]
            # Remove prompt tokens
            prompt_len = len(vlm_request.prompt_tokens)
            new_tokens = generated_tokens[prompt_len:]
            # Decode
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return text.strip()
        else:
            logger.warning("No tokens generated")
            return ""

    def generate(self, message: Union[List[dict], List[str]], dataset: Optional[str] = None) -> str:
        """Generate a response (VLMEvalKit's main entry point).

        This method handles both the dict format and the simplified list format.

        Args:
            message: Either a list of dicts with type/value, or a list of strings
                where image paths are auto-detected.
            dataset: Optional dataset name.

        Returns:
            The generated text response.
        """
        # Convert simplified format to dict format if needed
        if message and isinstance(message[0], str):
            converted_msgs = []
            for item in message:
                if self._is_image_path(item):
                    converted_msgs.append({"type": "image", "value": item})
                else:
                    converted_msgs.append({"type": "text", "value": item})
            message = converted_msgs

        return self.generate_inner(message, dataset=dataset)

    def _is_image_path(self, s: str) -> bool:
        """Check if a string looks like an image path or URL."""
        if s.startswith(("http://", "https://")):
            return True
        lower = s.lower()
        return any(lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"])
