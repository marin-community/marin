# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from jaxtyping import PRNGKeyArray

import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call

from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmConfig
from levanter.models.qwen import QwenConfig, QwenLMHeadModel, Qwen3Config, Qwen3LMHeadModel
from levanter.models.siglip import SiglipVisionConfig, SiglipVisionModel
from levanter.models.siglip2 import Siglip2VisionConfig, Siglip2VisionModel
from levanter.inference.engine import InferenceEngine, Request
from levanter.layers.kv_cache import KvPageCache, ListCache
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.logging import silence_transformer_nag

import logging

logger = logging.getLogger(__name__)

silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig  # noqa: E402
from transformers import LlavaOnevisionConfig as HfLlavaOnevisionConfig  # noqa: E402


@LmConfig.register_subclass("llava_onevision")
@dataclass(frozen=True)
class LlavaOnevisionConfig:
    """
    Configuration class for LLaVA OneVision multimodal model.

    LLaVA OneVision combines a vision encoder (SigLIP or Siglip2) with a Qwen2/Qwen3 language model
    through a multimodal projector.

    Implements the VlmConfig interface (vision_config + text_config) via duck typing.

    Args:
        vision_config: Configuration for the vision encoder (SigLIP or Siglip2)
        text_config: Configuration for the Qwen2/Qwen3 language model
        vision_encoder_type: Type of vision encoder to use ("siglip" or "siglip2")
        image_token_index: Token ID used to represent image patches in text
        video_token_index: Token ID used to represent video frames in text
        projector_hidden_act: Activation function for the multimodal projector
        vision_feature_select_strategy: How to select vision features ("default" or "full")
        vision_feature_layer: Which vision layer(s) to use for features (-1 for last layer)
        vision_aspect_ratio: Aspect ratio strategy for image processing
        image_grid_pinpoints: List of (height, width) resolutions for multi-scale processing
        multimodal_projector_bias: Whether to use bias in the projector
        gradient_checkpointing: Whether to use gradient checkpointing
    """

    vision_config: Union[SiglipVisionConfig, Siglip2VisionConfig]
    text_config: QwenConfig  # QwenConfig base, use text_model_type to specify qwen3
    vision_encoder_type: str = "siglip"  # "siglip" or "siglip2"
    text_model_type: str = "qwen"  # "qwen" for Qwen2, "qwen3" for Qwen3

    image_token_index: int = 151646
    video_token_index: int = 151647
    pad_token_id: int = 151643  # Qwen's default pad token (<|endoftext|>)
    projector_hidden_act: ActivationFunctionEnum = ActivationFunctionEnum.gelu
    vision_feature_select_strategy: str = "full"
    vision_feature_layer: Union[int, List[int]] = -1
    vision_aspect_ratio: str = "anyres_max_9"
    image_grid_pinpoints: Optional[List[List[int]]] = None
    multimodal_projector_bias: bool = True
    gradient_checkpointing: bool = True
    disable_anyres: bool = False  # When True, use single resolution (base patch only)

    # Reference checkpoint for loading pretrained models
    reference_checkpoint: Optional[str] = None
    tokenizer: Optional[str] = None

    def __post_init__(self):
        if self.vision_encoder_type not in ["siglip", "siglip2"]:
            raise ValueError(f"vision_encoder_type must be 'siglip' or 'siglip2', got {self.vision_encoder_type}")

        if self.text_model_type not in ["qwen", "qwen3"]:
            raise ValueError(f"text_model_type must be 'qwen' or 'qwen3', got {self.text_model_type}")

        # Convert QwenConfig to Qwen3Config if text_model_type is "qwen3"
        if self.text_model_type == "qwen3" and not isinstance(self.text_config, Qwen3Config):
            # Create Qwen3Config from QwenConfig fields
            qwen3_config = Qwen3Config(
                max_seq_len=self.text_config.max_seq_len,
                hidden_dim=self.text_config.hidden_dim,
                intermediate_dim=self.text_config.intermediate_dim,
                num_layers=self.text_config.num_layers,
                num_heads=self.text_config.num_heads,
                num_kv_heads=self.text_config.num_kv_heads,
                activation_function=self.text_config.activation_function,
                initializer_range=self.text_config.initializer_range,
                layer_norm_epsilon=self.text_config.layer_norm_epsilon,
                tie_word_embeddings=self.text_config.tie_word_embeddings,
                rope=self.text_config.rope,
                use_bias=self.text_config.use_bias,  # Qwen3 uses this for attention_bias
                gradient_checkpointing=self.text_config.gradient_checkpointing,
                scan_layers=self.text_config.scan_layers,
                attn_backend=self.text_config.attn_backend,
                flash_attention_block_size=self.text_config.flash_attention_block_size,
            )
            object.__setattr__(self, "text_config", qwen3_config)

        if self.vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                f"vision_feature_select_strategy must be 'default' or 'full', got {self.vision_feature_select_strategy}"
            )

        # Handle disable_anyres: use single resolution (base patch only)
        if self.disable_anyres:
            image_size = self.vision_config.image_size
            object.__setattr__(self, "vision_aspect_ratio", "single")
            # Use minimal grid pinpoints (HF processor requires at least one entry)
            # Our code will only use the base patch when vision_aspect_ratio="single"
            object.__setattr__(self, "image_grid_pinpoints", [[image_size, image_size]])
        # Set default image_grid_pinpoints if not provided
        elif self.image_grid_pinpoints is None:
            # Default pinpoints for anyres_max_9 strategy
            object.__setattr__(
                self,
                "image_grid_pinpoints",
                [
                    [384, 384],
                    [384, 768],
                    [384, 1152],
                    [768, 384],
                    [768, 768],
                    [768, 1152],
                    [1152, 384],
                    [1152, 768],
                    [1152, 1152],
                ],
            )

    @property
    def vocab_size(self) -> int:
        """Return vocab_size from text_config for compatibility with HFCheckpointConverter.load_pretrained()."""
        return self.text_config.vocab_size

    @property
    def vision_feature_height(self) -> int:
        """Return the vision encoder's output feature height (tokens per spatial dimension).

        This is computed as image_size // patch_size. For example:
        - SigLIP with 384x384 images and patch_size=16: 384 // 16 = 24
        - LLaVA-OneVision default with patch_size=14: 384 // 14 = 27

        Use this to validate that data preprocessing uses the correct vision_feature_height.
        """
        image_size = self.vision_config.image_size
        patch_size = self.vision_config.patch_size
        return image_size // patch_size

    @property
    def features_per_patch(self) -> int:
        """Return the total number of features per image patch (vision_feature_height^2)."""
        return self.vision_feature_height ** 2

    @property
    def model_type(self) -> Type["LlavaOnevisionModel"]:
        """Return the model class type."""
        return LlavaOnevisionModel

    def build(self, Vocab: hax.Axis, *, key: PRNGKeyArray) -> "LlavaOnevisionModel":
        """Build the model from this config.

        This method matches the LmConfig.build() interface for consistency.
        """
        return self.model_type.init(Vocab, self, key=key)

    def hf_checkpoint_converter(
        self, ref_checkpoint: Optional[str] = None
    ) -> HFCheckpointConverter["LlavaOnevisionConfig"]:  # type: ignore
        """Create HuggingFace checkpoint converter for this config."""
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=True,
            tokenizer=ref_checkpoint if self.tokenizer is None else self.tokenizer,
            HfConfigClass=HfLlavaOnevisionConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig) -> "LlavaOnevisionConfig":
        """Convert from HuggingFace config to Levanter config."""
        # Detect vision encoder type from HF config
        # Check if it's Siglip2 by looking for num_patches attribute
        is_siglip2 = hasattr(hf_config.vision_config, "num_patches")
        vision_encoder_type = "siglip2" if is_siglip2 else "siglip"

        # Extract vision and text configs based on type
        if vision_encoder_type == "siglip2":
            vision_config = Siglip2VisionConfig.from_hf_config(hf_config.vision_config)
        else:
            vision_config = SiglipVisionConfig.from_hf_config(hf_config.vision_config)

        # Detect text model type (Qwen2 vs Qwen3) and use appropriate config class
        text_model_type = getattr(hf_config.text_config, "model_type", "qwen2")
        if text_model_type == "qwen3":
            text_config = Qwen3Config.from_hf_config(hf_config.text_config)
        else:
            # Ensure no_bias attribute exists (Qwen2 default is True, meaning use_bias=False)
            if not hasattr(hf_config.text_config, "no_bias"):
                hf_config.text_config.no_bias = True
            text_config = QwenConfig.from_hf_config(hf_config.text_config)

        # Parse activation function
        act_map = {
            "gelu": ActivationFunctionEnum.gelu,
            "gelu_new": ActivationFunctionEnum.gelu_new,
            "relu": ActivationFunctionEnum.relu,
            "silu": ActivationFunctionEnum.silu,
        }
        activation_fn = act_map.get(hf_config.projector_hidden_act, ActivationFunctionEnum.gelu)

        return cls(
            vision_config=vision_config,
            text_config=text_config,
            vision_encoder_type=vision_encoder_type,
            image_token_index=hf_config.image_token_index,
            video_token_index=hf_config.video_token_index,
            projector_hidden_act=activation_fn,
            vision_feature_select_strategy=hf_config.vision_feature_select_strategy,
            vision_feature_layer=hf_config.vision_feature_layer,
            vision_aspect_ratio=hf_config.vision_aspect_ratio,
            image_grid_pinpoints=hf_config.image_grid_pinpoints,
            multimodal_projector_bias=hf_config.multimodal_projector_bias,
        )

    def with_token_ids(
        self,
        image_token_id: Optional[int] = None,
        video_token_id: Optional[int] = None,
    ) -> "LlavaOnevisionConfig":
        """Create a new config with updated token IDs.

        Use this method to update the config when replacing the tokenizer with a different one
        (e.g., Qwen3 tokenizer), which may assign different IDs to <image> and <video> tokens.

        Args:
            image_token_id: New token ID for <image> placeholder. If None, keeps current value.
            video_token_id: New token ID for <video> placeholder. If None, keeps current value.

        Returns:
            New LlavaOnevisionConfig with updated token IDs.

        Example:
            >>> from levanter.data.image import BatchImageProcessor
            >>> bp = BatchImageProcessor(processor, tokenizer=qwen3_tokenizer)
            >>> token_ids = bp.get_token_ids()
            >>> model_config = model_config.with_token_ids(
            ...     image_token_id=token_ids["image_token_id"],
            ...     video_token_id=token_ids["video_token_id"],
            ... )
        """
        updates = {}
        if image_token_id is not None:
            updates["image_token_index"] = image_token_id
        if video_token_id is not None:
            updates["video_token_index"] = video_token_id

        if updates:
            return replace(self, **updates)
        return self

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfLlavaOnevisionConfig:
        """Convert from Levanter config to HuggingFace config.

        Args:
            vocab_size: Vocabulary size for the text model
            config_overrides: Optional config overrides
        """
        if config_overrides is None:
            config_overrides = {}

        # Convert vision and text configs
        vision_hf_config = self.vision_config.to_hf_config()
        text_hf_config = self.text_config.to_hf_config(vocab_size=vocab_size)

        # Map activation function
        if isinstance(self.projector_hidden_act, ActivationFunctionEnum):
            projector_act = self.projector_hidden_act.value
        else:
            projector_act = self.projector_hidden_act

        return HfLlavaOnevisionConfig(
            vision_config=vision_hf_config.to_dict(),
            text_config=text_hf_config.to_dict(),
            image_token_index=self.image_token_index,
            video_token_index=self.video_token_index,
            projector_hidden_act=projector_act,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layer=self.vision_feature_layer,
            vision_aspect_ratio=self.vision_aspect_ratio,
            image_grid_pinpoints=self.image_grid_pinpoints,
            multimodal_projector_bias=self.multimodal_projector_bias,
            **config_overrides,
        )

    # Axis definitions (implementing VlmConfig interface via duck typing)

    @property
    def Embed(self) -> Axis:
        """Text embedding dimension."""
        return self.text_config.Embed

    @property
    def max_Pos(self) -> Axis:
        """Maximum position axis."""
        return self.text_config.max_Pos

    @property
    def KeyPos(self) -> Axis:
        """Key position axis."""
        return self.text_config.KeyPos

    @property
    def VisionEmbed(self) -> Axis:
        """Vision embedding dimension (renamed to avoid collision with text embed)."""
        return Axis(name="vision_embed", size=self.vision_config.hidden_size)

    @property
    def TextEmbed(self) -> Axis:
        """Text embedding dimension (same as Embed for compatibility)."""
        return self.text_config.Embed

    @property
    def Pos(self) -> Axis:
        """Maximum position axis (alias for max_Pos)."""
        return self.text_config.max_Pos


class LlavaOnevisionMultimodalProjector(eqx.Module):
    """
    Multimodal projector that maps vision features to text embedding space.

    This is a simple MLP with one hidden layer:
    vision_embed -> hidden -> text_embed
    """

    config: LlavaOnevisionConfig = eqx.field(static=True)
    linear_1: hnn.Linear
    act: Callable = eqx.field(static=True)
    linear_2: hnn.Linear

    @staticmethod
    def init(config: LlavaOnevisionConfig, *, key) -> "LlavaOnevisionMultimodalProjector":
        """Initialize the multimodal projector."""
        k1, k2 = jrandom.split(key, 2)

        VisionEmbed = config.VisionEmbed
        TextEmbed = config.TextEmbed
        # Create intermediate hidden axis for projector (same size as TextEmbed but different name)
        # This avoids axis collision in linear_2 where In and Out would be the same
        ProjectorHidden = Axis(name="projector_hidden", size=config.text_config.hidden_dim)
        use_bias = config.multimodal_projector_bias

        # First linear layer: vision_embed -> projector_hidden
        linear_1 = hnn.Linear.init(In=VisionEmbed, Out=ProjectorHidden, key=k1, use_bias=use_bias, out_first=True)

        # Activation function
        if isinstance(config.projector_hidden_act, ActivationFunctionEnum):
            act_fn = config.projector_hidden_act.to_fn()
        else:
            act_fn = config.projector_hidden_act

        # Second linear layer: projector_hidden -> text_embed
        linear_2 = hnn.Linear.init(In=ProjectorHidden, Out=TextEmbed, key=k2, use_bias=use_bias, out_first=True)

        return LlavaOnevisionMultimodalProjector(config, linear_1, act_fn, linear_2)

    @named_call
    def __call__(self, image_features: NamedArray, *, key=None) -> NamedArray:
        """
        Project vision features to text embedding space.

        Args:
            image_features: Vision features with shape (..., vision_embed)
            key: Optional PRNGKey for dropout (not used currently)

        Returns:
            Projected features with shape (..., text_embed)
        """
        k1, k2 = maybe_rng_split(key, 2)

        # Vision features come with "vision_embed" axis from Siglip/Siglip2
        # (renamed at source to avoid collision with LLM's "embed" axis for FSDP sharding)

        # First linear: vision_embed -> projector_hidden + activation
        hidden = self.linear_1(image_features, key=k1)
        hidden = self.act(hidden)

        # Second linear: projector_hidden -> text_embed
        # The output will have "embed" axis (from TextEmbed)
        output = self.linear_2(hidden, key=k2)

        return output


class LlavaOnevisionModel(eqx.Module):
    """
    LLaVA OneVision model combining vision and language.

    Architecture:
    1. Vision encoder (Siglip2): Processes images
    2. Multimodal projector: Maps vision features to text space
    3. Language model (Qwen2/3): Generates text with vision context
    """

    config: LlavaOnevisionConfig = eqx.field(static=True)
    vision_tower: Union[SiglipVisionModel, Siglip2VisionModel]
    multi_modal_projector: LlavaOnevisionMultimodalProjector
    language_model: QwenLMHeadModel

    @staticmethod
    def init(Vocab: Axis, config: LlavaOnevisionConfig, *, key) -> "LlavaOnevisionModel":
        """Initialize LLaVA OneVision model."""
        k_vision, k_proj, k_lm = jrandom.split(key, 3)

        # Initialize vision tower based on encoder type
        if config.vision_encoder_type == "siglip2":
            vision_tower = Siglip2VisionModel.init(Vocab=Vocab, config=config.vision_config, key=k_vision)
        elif config.vision_encoder_type == "siglip":
            vision_tower = SiglipVisionModel.init(Vocab=Vocab, config=config.vision_config, key=k_vision)
        else:
            raise ValueError(f"Unsupported vision_encoder_type: {config.vision_encoder_type}")

        # Initialize multimodal projector
        multi_modal_projector = LlavaOnevisionMultimodalProjector.init(config=config, key=k_proj)

        # Initialize language model (Qwen)
        language_model = QwenLMHeadModel.init(Vocab=Vocab, config=config.text_config, key=k_lm)

        return LlavaOnevisionModel(
            config=config,
            vision_tower=vision_tower,
            multi_modal_projector=multi_modal_projector,
            language_model=language_model,
        )

    @staticmethod
    def _compute_position_ids(validity_mask: jnp.ndarray) -> jnp.ndarray:
        """Compute compact position IDs from a validity mask using cumsum.

        Args:
            validity_mask: Boolean or int array where True/1 indicates valid positions.

        Returns:
            Position IDs where valid positions get incrementing IDs, invalid positions get 0.
        """
        position_ids = jnp.cumsum(validity_mask.astype(jnp.int32), axis=-1) - 1
        return jnp.maximum(position_ids, 0)

    @staticmethod
    def _batch_gather(arrays: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
        """Gather elements from arrays using indices, batched over first dimension.

        Args:
            arrays: Array of shape (batch, seq, ...) to gather from.
            indices: Array of shape (batch, num_indices) specifying indices to gather.

        Returns:
            Gathered array of shape (batch, num_indices, ...).
        """
        # Use take_along_axis instead of vmap for better performance
        # Expand indices to broadcast along trailing dimensions
        indices_expanded = indices
        for _ in range(len(arrays.shape) - 2):
            indices_expanded = jnp.expand_dims(indices_expanded, axis=-1)
        # Broadcast to match trailing dimensions of arrays
        indices_expanded = jnp.broadcast_to(indices_expanded, (*indices.shape, *arrays.shape[2:]))
        return jnp.take_along_axis(arrays, indices_expanded, axis=1)

    @staticmethod
    def _compute_per_segment_image_indices(
        special_image_mask: jnp.ndarray,
        segment_ids: jnp.ndarray,
        image_segment_ids: jnp.ndarray,
        features_per_patch: int,
        total_image_tokens: int,
    ) -> jnp.ndarray:
        """Compute image feature indices using per-segment cumsum for packing support.

        For packed sequences, each segment has its own set of image features.
        This function computes indices that correctly map each image placeholder
        to its corresponding image feature within the same segment.

        Example:
            segment_ids:       [0, 0, 0, 1, 1, 1, -1]  # 2 samples packed
            special_image_mask:[1, 1, 0, 1, 0, 0,  0]  # 2 images in seg0, 1 in seg1
            image_segment_ids: [0, 0, 1, -1]           # 2 patches for seg0, 1 for seg1

            For placeholder at position 0 (seg 0, 1st placeholder): index = 0
            For placeholder at position 1 (seg 0, 2nd placeholder): index = features_per_patch
            For placeholder at position 3 (seg 1, 1st placeholder): index = 2*features_per_patch

        Args:
            special_image_mask: Boolean mask for image placeholders (batch, seq_len)
            segment_ids: Segment ID per token (batch, seq_len), -1 for padding
            image_segment_ids: Segment ID per patch (batch, max_patches), -1 for padding
            features_per_patch: Number of features per image patch
            total_image_tokens: Total number of image tokens in the flattened features

        Returns:
            Indices array (batch, seq_len) mapping each position to an image feature index.
            For non-placeholder positions, returns 0 (will be masked out anyway).
        """
        batch_size, seq_len = special_image_mask.shape
        _, max_patches = image_segment_ids.shape

        # Compute starting feature index for each segment
        # image_segment_ids: (batch, max_patches) with segment IDs or -1
        # For each patch, compute its cumulative index in the flattened features
        # patch_feature_starts[b, p] = starting index of features for patch p

        # Compute starting feature index for each patch
        # IMPORTANT: The feature array includes ALL patches (including padding),
        # so patch p's features start at p * features_per_patch, regardless of validity
        # patch_feature_starts[p] = p * features_per_patch
        patch_indices = jnp.arange(max_patches)
        patch_feature_starts = jnp.broadcast_to(
            patch_indices * features_per_patch,
            (batch_size, max_patches)
        )

        # For each segment, find the starting feature index
        # We need to map: segment_id -> first patch index for that segment
        # Then: first patch index -> feature start index

        # Create segment to first feature index mapping
        # For each unique segment, find the minimum patch index with that segment ID
        max_segments = jnp.max(segment_ids) + 1
        max_segments = jnp.maximum(max_segments, 1)  # Ensure at least 1

        def compute_segment_starts_for_batch(img_seg_ids, patch_starts):
            """Compute starting feature index for each segment in one batch element."""
            # For each segment s, find the first patch with segment_id == s
            # segment_feature_starts[s] = patch_starts[first_patch_with_segment_s]

            # Initialize with large values (will take min)
            segment_starts = jnp.full((64,), total_image_tokens, dtype=jnp.int32)  # Max 64 segments

            # For each patch, if it's the first patch for its segment, record the start
            for p in range(max_patches):
                seg_id = img_seg_ids[p]
                # Only process valid patches (segment_id >= 0)
                valid = seg_id >= 0
                # Update segment_starts[seg_id] if this is earlier than current value
                current = segment_starts[seg_id]
                new_value = jnp.where(valid & (patch_starts[p] < current), patch_starts[p], current)
                segment_starts = segment_starts.at[seg_id].set(new_value)

            return segment_starts

        # Use lax.scan or vmap for efficiency across batch
        def compute_for_batch(img_seg_ids, patch_starts):
            # Create segment -> first feature index mapping
            segment_starts = jnp.full((64,), 0, dtype=jnp.int32)  # Default to 0 for invalid

            # Vectorized approach: find first occurrence of each segment
            # For segment s: find min patch index p where img_seg_ids[p] == s
            # Simpler approach: for each segment, find its first patch and get the feature start
            # We can do this by computing the minimum patch index for each segment
            # Then look up the feature start for that patch index

            # For each segment s (0 to max_seg), find first patch p where img_seg_ids[p] == s
            # patch_starts[p] gives the starting feature index for that segment

            # Approach: scatter-reduce to find minimum patch index per segment
            # Initialize with a large value, then take min for each segment
            large_val = max_patches + 1
            first_patch_per_seg = jnp.full((64,), large_val, dtype=jnp.int32)

            def update_first_patch(first_patches, patch_data):
                patch_idx, seg_id = patch_data
                valid = seg_id >= 0
                safe_seg_id = jnp.where(valid, seg_id, 0)
                # Update if this patch index is smaller than current (first occurrence wins)
                current = first_patches[safe_seg_id]
                should_update = valid & (patch_idx < current)
                new_val = jnp.where(should_update, patch_idx, current)
                # Use scatter to update (works with JIT)
                first_patches = first_patches.at[safe_seg_id].min(jnp.where(valid, patch_idx, large_val))
                return first_patches, None

            scan_inputs = (jnp.arange(max_patches), img_seg_ids)
            first_patch_per_seg, _ = jax.lax.scan(update_first_patch, first_patch_per_seg, scan_inputs)

            # Now look up feature starts for each segment's first patch
            # Clamp to valid range for lookup
            first_patch_clamped = jnp.clip(first_patch_per_seg, 0, max_patches - 1)
            segment_starts = jnp.where(
                first_patch_per_seg < large_val,
                patch_starts[first_patch_clamped],
                0  # Default for segments with no patches
            )
            return segment_starts

        # Compute segment starts for each batch element
        segment_feature_starts = jax.vmap(compute_for_batch)(image_segment_ids, patch_feature_starts)
        # segment_feature_starts: (batch, 64) - starting feature index for each segment

        # Now compute indices for each position in the sequence
        def compute_indices_for_batch(mask, seg_ids, img_seg_ids, seg_starts):
            """Compute image indices for one batch element using per-segment cumsum."""
            # For each position, if it's a placeholder:
            # 1. Get its segment ID
            # 2. Count how many placeholders with same segment ID came before (per-segment cumsum)
            # 3. Add to segment's starting feature index

            # Per-segment cumsum: for each segment, count placeholders
            # indices[i] = seg_starts[seg_ids[i]] + cumsum_within_segment[i] * features_per_patch

            # Compute per-segment cumsum
            # For each position, count previous placeholders with same segment
            def per_segment_cumsum(carry, x):
                """Accumulate count per segment."""
                seg_counts, prev_seg = carry
                pos_idx, is_placeholder, seg_id = x

                # Use safe indexing to avoid negative index issues (seg_id=-1 for padding)
                safe_seg_id = jnp.clip(seg_id, 0, 63)

                # Get current count for this segment
                current_count = seg_counts[safe_seg_id]

                # Update count if this is a placeholder with valid segment
                valid = is_placeholder & (seg_id >= 0)
                new_count = jnp.where(valid, current_count + 1, current_count)
                # Only update if valid to avoid corrupting other segments
                seg_counts = jnp.where(valid, seg_counts.at[safe_seg_id].set(new_count), seg_counts)

                # The index for this position is current_count (before increment)
                local_idx = jnp.where(valid, current_count, 0)

                return (seg_counts, seg_id), local_idx

            # Initialize per-segment counts
            init_counts = jnp.zeros((64,), dtype=jnp.int32)

            # Pack inputs
            inputs = (jnp.arange(seq_len), mask.astype(jnp.bool_), seg_ids)

            # Run scan
            (final_counts, _), local_indices = jax.lax.scan(
                per_segment_cumsum,
                (init_counts, -1),
                inputs,
            )

            # Compute global indices: seg_start + local_idx
            # local_indices counts placeholder tokens (not patches) before this position in the same segment
            # seg_start_per_pos is already in feature/token units (patch_idx * features_per_patch)
            # So the formula is simply: seg_start + local_idx

            # Get segment start for each position
            seg_start_per_pos = seg_starts[jnp.clip(seg_ids, 0, 63)]

            # Compute final index: segment_start + local_index
            # Each placeholder token maps to exactly one image feature
            global_indices = seg_start_per_pos + local_indices

            # Ensure indices are within bounds
            global_indices = jnp.clip(global_indices, 0, total_image_tokens - 1)

            return global_indices

        # Apply to each batch element
        all_indices = jax.vmap(compute_indices_for_batch)(
            special_image_mask,
            segment_ids,
            image_segment_ids,
            segment_feature_starts,
        )

        return all_indices

    @property
    def Vocab(self) -> Axis:
        """Get the vocabulary axis from the language model."""
        return self.language_model.Vocab

    def get_input_embeddings(self) -> hnn.Embedding:
        """Get the input embeddings from the language model."""
        return self.language_model.embeddings.token_embeddings

    def get_image_features(
        self,
        pixel_values: NamedArray,
        grid_mask: NamedArray,
        *,
        key=None,
    ) -> Tuple[NamedArray, NamedArray]:
        """
        Extract image features with fixed-shape processing for JIT compatibility.

        This implementation processes all patches (including padding) through the vision tower,
        then applies feature scrubbing to zero out invalid patches based on grid_mask.

        Args:
            pixel_values: Fixed-shape patches (batch, TOTAL_PATCHES, C, H, W) - padded to max patches
            grid_mask: Boolean mask indicating valid patches (batch, TOTAL_PATCHES)
            key: Optional PRNGKey

        Returns:
            image_features: (batch, TOTAL_PATCHES, features_per_patch, embed) - with padding zeroed out
            grid_mask: (batch, TOTAL_PATCHES) - passed through for later use
        """
        k_vision, k_proj = maybe_rng_split(key, 2)

        vision_feature_layer = self.config.vision_feature_layer
        vision_feature_select_strategy = self.config.vision_feature_select_strategy

        # Only 5D input supported: (batch, num_patches, channels, height, width)
        if len(pixel_values.axes) != 5:
            raise ValueError(f"Expected 5D pixel_values (batch, num_patches, C, H, W), got {pixel_values.axes}")

        batch_ax, num_patches_ax, *_ = pixel_values.axes

        # Flatten batch and patches for vision tower: (batch * TOTAL_PATCHES, C, H, W)
        total_images = batch_ax.size * num_patches_ax.size
        VisionBatch = Axis("vision_batch", total_images)
        pixel_values_flat = hax.flatten_axes(pixel_values, (batch_ax, num_patches_ax), VisionBatch)

        # Run vision tower on all patches (including padding patches)
        image_outputs = self.vision_tower(pixel_values_flat, output_hidden_states=True, key=k_vision)
        if image_outputs.hidden_states is None:
            raise ValueError("Vision tower must return hidden states when output_hidden_states=True")

        # Select features from specified layer(s)
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        else:
            # Concatenate features from multiple layers
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            selected_image_feature = hax.concatenate(self.config.VisionEmbed, hs_pool)

        # Apply feature selection strategy: "default" skips first token (CLS), "full" keeps all
        if vision_feature_select_strategy == "default":
            # Skip first token (CLS token) - slice from num_patches 1 onwards
            patch_axis = selected_image_feature.resolve_axis("num_patches")
            selected_image_feature = hax.slice(
                selected_image_feature, {"num_patches": 1}, {"num_patches": patch_axis.size - 1}
            )

        # Project to text embedding space
        image_features = self.multi_modal_projector(selected_image_feature, key=k_proj)

        # Rename "num_patches" axis to "features_per_patch" to avoid collision after unflatten
        # siglip outputs (vision_batch, num_patches, embed) where num_patches = patches per image
        # After unflatten, we'll have (batch, num_patches, features_per_patch, embed)
        image_features = image_features.rename({"num_patches": "features_per_patch"})

        # Unflatten vision_batch back to (batch, num_patches)
        image_features = image_features.unflatten_axis("vision_batch", (batch_ax, num_patches_ax))
        # Now shape: (batch, num_patches, features_per_patch, embed)

        # === FEATURE SCRUBBING ===
        # Zero out padding patches: features = features * mask
        # Broadcast mask to match feature dimensions
        # Create a mask with shape (batch, num_patches) and broadcast to (batch, num_patches, features_per_patch, embed)
        mask_expanded = grid_mask.astype(jnp.float32)

        # Apply mask - this zeros out all features for padding patches
        image_features_array = image_features.array * mask_expanded.array[:, :, None, None]
        image_features = hax.named(image_features_array, image_features.axes)

        return image_features, grid_mask

    def get_placeholder_mask(self, input_ids: NamedArray, image_features: Optional[NamedArray] = None):
        """
        Get mask for placeholder tokens (image/video tokens) in the input.

        Args:
            input_ids: Input token IDs with shape (batch, seq_len)
            image_features: Image features with shape (total_patches, embed) or
                           (total_patches, features_per_patch, embed)

        Returns:
            special_image_mask: Boolean mask with shape (batch, seq_len)
        """
        # Find positions where input_ids == image_token_index
        special_image_mask = input_ids == self.config.image_token_index

        # Note: Token count validation is done outside JIT context when needed
        # During JIT compilation, we cannot use concrete values from traced arrays
        # The validation is performed in non-JIT contexts (e.g., tests)

        return special_image_mask

    def validate_placeholder_mask(self, input_ids: NamedArray, image_features: Optional[NamedArray] = None):
        """
        Validate that image token count matches feature count. Call outside JIT context.

        Args:
            input_ids: Input token IDs with shape (batch, seq_len)
            image_features: Image features with shape (total_patches, embed) or
                           (total_patches, features_per_patch, embed)

        Raises:
            ValueError: If token count doesn't match feature count
        """
        if image_features is None:
            return

        special_image_mask = input_ids == self.config.image_token_index
        n_image_tokens = int(jnp.sum(special_image_mask.array))

        # Get total feature count from image_features
        if len(image_features.axes) == 2:
            n_features = image_features.axes[0].size
        else:
            # For (total_patches, features_per_patch, embed), total features = patches * features_per_patch
            n_features = image_features.axes[0].size * image_features.axes[1].size

        if n_image_tokens != n_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_features}"
            )

        return special_image_mask

    def forward_with_activations(
        self,
        input_ids: NamedArray,
        pixel_values: Optional[NamedArray] = None,
        grid_mask: Optional[NamedArray] = None,
        unpad_indices: Optional[NamedArray] = None,
        num_unpadded_features: Optional[int] = None,
        inputs_embeds: Optional[NamedArray] = None,
        combined_mask: Optional[NamedArray] = None,
        position_ids: Optional[NamedArray] = None,
        segment_ids: Optional[NamedArray] = None,
        image_segment_ids: Optional[NamedArray] = None,
        *,
        key=None,
    ) -> Tuple[NamedArray, NamedArray]:
        """
        Forward pass returning activations and lm_head for blockwise loss computation.

        This avoids materializing the full logits tensor (batch * seq * vocab),
        which can cause OOM for large vocab sizes.

        Args:
            input_ids: Text token IDs with shape (batch, seq_len)
            pixel_values: Fixed-shape image patches (batch, TOTAL_PATCHES, C, H, W)
                         Padded to max_patches + 1 (base patch + highres patches)
            grid_mask: Boolean mask indicating valid patches (batch, TOTAL_PATCHES)
                      True for actual image patches, False for padding
            unpad_indices: Pre-computed indices to reorder features to HF's unpadded order
                          (batch, num_image_tokens) - maps HF position to Levanter index
            num_unpadded_features: Actual number of unpadded features (before padding)
            inputs_embeds: Optional pre-computed embeddings (batch, seq_len, embed)
            combined_mask: Optional precomputed validity mask from CPU data pipeline
                          (batch, seq_len) int32 - if provided, skips GPU computation
            position_ids: Optional precomputed position IDs from CPU data pipeline
                          (batch, seq_len) int32 - if provided, skips GPU computation
            segment_ids: Optional segment IDs for packing support (batch, seq_len)
                        Each sample in a pack gets a unique segment ID (0, 1, 2, ...)
                        Padding positions get segment_id = -1
                        When provided, enables per-segment attention masking
            image_segment_ids: Optional segment IDs for image patches (batch, max_patches)
                              Maps each patch to its segment ID for per-segment cumsum
            key: Optional PRNGKey

        Returns:
            Tuple of (activations, lm_head) for use with fused_cross_entropy_loss.
        """
        k_vision, k_lm = maybe_rng_split(key, 2)

        # Merge text embeddings with image features and compute position IDs
        inputs_embeds, position_ids, validity_mask = self._merge_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            grid_mask=grid_mask,
            unpad_indices=unpad_indices,
            num_unpadded_features=num_unpadded_features,
            precomputed_combined_mask=combined_mask,
            precomputed_position_ids=position_ids,
            segment_ids=segment_ids,
            image_segment_ids=image_segment_ids,
            key=k_vision,
        )

        # Forward through language model with merged embeddings
        # Create attention mask: causal + segment-based padding mask
        if segment_ids is not None:
            # Packing mode: use provided segment_ids directly for attention mask
            # segment_ids contains 0, 1, 2, ... for different samples, -1 for padding
            # Splash attention prevents attention between different segments
            attn_mask = AttentionMask.causal().with_segment_ids(segment_ids)
        else:
            # Non-packing mode: validity_mask is (batch, seq) with True for valid, False for invalid
            # Use segment_ids instead of explicit_mask for splash attention compatibility
            # Valid tokens get segment_id=1, padding tokens get segment_id=0
            segment_ids_from_validity = validity_mask.astype(jnp.int32)
            attn_mask = AttentionMask.causal().with_segment_ids(segment_ids_from_validity)

        activations = self.language_model.transformer(
            inputs_embeds, attn_mask=attn_mask, pos_ids=position_ids, key=k_lm
        )

        # Return activations and lm_head for blockwise loss computation
        lm_head = self.language_model.get_lm_head()
        return activations, lm_head

    @named_call
    def __call__(
        self,
        input_ids: NamedArray,
        pixel_values: Optional[NamedArray] = None,
        grid_mask: Optional[NamedArray] = None,
        unpad_indices: Optional[NamedArray] = None,
        num_unpadded_features: Optional[int] = None,
        inputs_embeds: Optional[NamedArray] = None,
        combined_mask: Optional[NamedArray] = None,
        position_ids: Optional[NamedArray] = None,
        segment_ids: Optional[NamedArray] = None,
        image_segment_ids: Optional[NamedArray] = None,
        *,
        key=None,
    ) -> NamedArray:
        """
        Forward pass through LLaVA OneVision with fixed-shape processing.

        Args:
            input_ids: Text token IDs with shape (batch, seq_len)
            pixel_values: Fixed-shape image patches (batch, TOTAL_PATCHES, C, H, W)
                         Padded to max_patches + 1 (base patch + highres patches)
            grid_mask: Boolean mask indicating valid patches (batch, TOTAL_PATCHES)
                      True for actual image patches, False for padding
            unpad_indices: Pre-computed indices to reorder features to HF's unpadded order
                          (batch, num_image_tokens) - maps HF position to Levanter index
            num_unpadded_features: Actual number of unpadded features (before padding)
            inputs_embeds: Optional pre-computed embeddings (batch, seq_len, embed)
            combined_mask: Optional precomputed validity mask from CPU data pipeline
                          (batch, seq_len) int32 - if provided, skips GPU computation
            position_ids: Optional precomputed position IDs from CPU data pipeline
                          (batch, seq_len) int32 - if provided, skips GPU computation
            segment_ids: Optional segment IDs for packing support (batch, seq_len)
                        Each sample in a pack gets a unique segment ID (0, 1, 2, ...)
                        Padding positions get segment_id = -1
            image_segment_ids: Optional segment IDs for image patches (batch, max_patches)
                              Maps each patch to its segment ID for per-segment cumsum
            key: Optional PRNGKey

        Returns:
            Logits with shape (batch, seq_len, vocab)
        """
        activations, lm_head = self.forward_with_activations(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_mask=grid_mask,
            unpad_indices=unpad_indices,
            num_unpadded_features=num_unpadded_features,
            inputs_embeds=inputs_embeds,
            combined_mask=combined_mask,
            position_ids=position_ids,
            segment_ids=segment_ids,
            image_segment_ids=image_segment_ids,
            key=key,
        )
        return hax.dot(activations, lm_head, axis=self.config.TextEmbed)

    def _merge_embeddings(
        self,
        input_ids: NamedArray,
        inputs_embeds: Optional[NamedArray],
        pixel_values: Optional[NamedArray],
        grid_mask: Optional[NamedArray],
        unpad_indices: Optional[NamedArray] = None,
        num_unpadded_features: Optional[int] = None,
        *,
        precomputed_combined_mask: Optional[NamedArray] = None,
        precomputed_position_ids: Optional[NamedArray] = None,
        precomputed_image_features: Optional[NamedArray] = None,
        segment_ids: Optional[NamedArray] = None,
        image_segment_ids: Optional[NamedArray] = None,
        key=None,
    ) -> Tuple[NamedArray, NamedArray, NamedArray]:
        """
        Merge text embeddings with projected image features and compute position IDs.

        This function:
        1. Gets image features via get_image_features() (fixed-shape with mask)
        2. Flattens image features to (batch, TOTAL_PATCHES * features_per_patch, embed)
        3. If unpad_indices provided, reorders features to HF's unpadded spatial order
        4. Merges image features into text embeddings at placeholder positions
        5. Computes compact position IDs that skip padding using cumsum (or uses precomputed values)

        Packing Support:
        When segment_ids and image_segment_ids are provided, this function uses per-segment
        cumsum to correctly map image placeholders to their corresponding image features
        within each segment. This is essential for VLM packing where multiple samples
        are combined into a single training example.

        Args:
            input_ids: Text token IDs (batch, seq_len) - used to derive text validity mask
            inputs_embeds: Optional pre-computed text embeddings
            pixel_values: Fixed-shape patches (batch, TOTAL_PATCHES, C, H, W)
            grid_mask: Boolean mask for valid patches (batch, TOTAL_PATCHES)
            unpad_indices: Pre-computed indices to reorder features to HF's unpadded order
                          (batch, num_image_tokens) - if None, uses sequential ordering
            precomputed_combined_mask: Optional precomputed validity mask from CPU data pipeline
                          (batch, seq_len) int32 - if provided, skips GPU computation
            precomputed_position_ids: Optional precomputed position IDs from CPU data pipeline
                          (batch, seq_len) int32 - if provided, skips GPU computation
            precomputed_image_features: Optional precomputed image features from batched vision encoder
                          (batch, TOTAL_PATCHES, features_per_patch, embed) - if provided, skips vision encoder
            segment_ids: Optional segment IDs for packing (batch, seq_len)
                        Each token gets segment_id 0, 1, 2, ... or -1 for padding
            image_segment_ids: Optional segment IDs for patches (batch, max_patches)
                              Maps each patch to its segment for per-segment cumsum
            key: Optional PRNGKey

        Returns:
            Tuple of:
            - merged_embeds: (batch, seq_len, embed) with image features at placeholders
            - position_ids: (batch, seq_len) compact position IDs skipping padding
            - validity_mask: (batch, seq_len) boolean mask for valid tokens (for attention masking)
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids are required when inputs_embeds is None.")
            inputs_embeds = self.get_input_embeddings().embed(input_ids)

        batch_ax = inputs_embeds.axes[0]
        seq_ax = inputs_embeds.axes[1]
        embed_ax = inputs_embeds.axes[2]

        # Text validity mask: valid text tokens (not padding)
        text_mask = (input_ids != self.config.pad_token_id).astype(jnp.int32)

        if pixel_values is None:
            # No images - just return text embeddings with text-only position IDs
            position_ids_array = self._compute_position_ids(text_mask.array)
            Pos = Axis("position", seq_ax.size)
            position_ids = hax.named(position_ids_array, (batch_ax, Pos))
            # Return text_mask as validity mask (already a NamedArray)
            validity_mask = text_mask.astype(jnp.bool_)
            return inputs_embeds, position_ids, validity_mask

        # Get image features: (batch, TOTAL_PATCHES, features_per_patch, embed)
        # Use precomputed features if provided (for batched inference), otherwise compute
        if precomputed_image_features is not None:
            image_features = precomputed_image_features
        else:
            image_features, grid_mask = self.get_image_features(
                pixel_values=pixel_values,
                grid_mask=grid_mask,
                key=key,
            )

        # Get dimensions
        num_patches_ax = image_features.axes[1]
        features_per_patch_ax = image_features.axes[2]
        features_per_patch = features_per_patch_ax.size
        total_patches = num_patches_ax.size

        # Flatten image features to (batch, total_image_tokens, embed)
        # where total_image_tokens = TOTAL_PATCHES * features_per_patch
        total_image_tokens = total_patches * features_per_patch
        ImageTokens = Axis("image_tokens", total_image_tokens)
        image_features_flat = hax.flatten_axes(image_features, (num_patches_ax, features_per_patch_ax), ImageTokens)

        # If unpad_indices provided, reorder features to HF's unpadded spatial order
        # unpad_indices: (batch, num_image_tokens) - padded array with valid indices in first num_unpadded_features positions
        # unpad_indices[i] = Levanter index for HF position i (for i < num_unpadded_features)
        if unpad_indices is not None and num_unpadded_features is not None:
            # Use the actual unpadded count (not the padded axis size)
            num_unpadded_tokens = num_unpadded_features

            # Slice unpad_indices to only the valid indices
            unpad_indices_valid = unpad_indices.array[:, :num_unpadded_tokens]

            # Gather features in HF's unpadded order
            image_features_reordered = self._batch_gather(image_features_flat.array, unpad_indices_valid)
            # Now image_features_reordered: (batch, num_unpadded_tokens, embed) in HF order
            UnpaddedTokens = Axis("image_tokens", num_unpadded_tokens)
            image_features_flat = hax.named(image_features_reordered, (batch_ax, UnpaddedTokens, embed_ax))
            # Update total_image_tokens to reflect the unpadded count
            total_image_tokens = num_unpadded_tokens

        # Get placeholder mask: where image tokens should be inserted
        special_image_mask = self.get_placeholder_mask(input_ids)

        # === VALIDATION: Check image token count matches vision features ===
        # This runs inside JIT via debug.callback to detect mismatched vision_feature_height
        def _validate_token_feature_match(mask_array, total_features, grid_mask_array, features_per_patch_val):
            """Validate that image token count in input_ids matches vision encoder output.

            When grid_mask is provided (disable_anyres mode), validates against the number
            of valid patches indicated by grid_mask, not the total padded patches.

            Note: Uses np.asarray() and int() to convert all values to CPU/Python types
            because this runs in jax.debug.callback which executes on CPU host,
            while arrays may be on TPU.
            """
            # Convert all inputs to CPU/Python types (avoids device mismatch in callback)
            mask_np = np.asarray(mask_array)
            total_features_int = int(np.asarray(total_features))
            features_per_patch_int = int(np.asarray(features_per_patch_val))

            n_image_tokens_per_batch = np.sum(mask_np, axis=-1)  # (batch,)

            # Calculate expected features per batch element based on grid_mask
            if grid_mask_array is not None:
                grid_mask_np = np.asarray(grid_mask_array)
                valid_patches_per_batch = np.sum(grid_mask_np, axis=-1)  # (batch,)
                expected_per_batch = valid_patches_per_batch * features_per_patch_int
            else:
                expected_per_batch = np.full(len(n_image_tokens_per_batch), total_features_int)

            # Check ALL batch elements for mismatches
            mismatched = []
            for i in range(len(n_image_tokens_per_batch)):
                n_tok = int(n_image_tokens_per_batch[i])
                n_exp = int(expected_per_batch[i])
                if n_tok > 0 and n_tok != n_exp:
                    mismatched.append(i)

            if mismatched:
                batch_size = len(n_image_tokens_per_batch)
                # Build detailed diagnostic info
                diag_lines = []
                diag_lines.append(f"=== IMAGE TOKEN MISMATCH DIAGNOSTIC (batch_size={batch_size}) ===")
                diag_lines.append(f"features_per_patch={features_per_patch_int}, total_features={total_features_int}")
                diag_lines.append(f"Mismatched batch elements: {mismatched} ({len(mismatched)}/{batch_size})")
                diag_lines.append("")

                # Log details for mismatched elements
                diag_lines.append("--- Mismatched elements ---")
                for i in mismatched[:50]:  # Cap at 50 to avoid excessive output
                    n_tok = int(n_image_tokens_per_batch[i])
                    n_exp = int(expected_per_batch[i])
                    gm_info = ""
                    if grid_mask_array is not None:
                        gm_row = grid_mask_np[i]
                        gm_info = f", grid_mask={gm_row.tolist()}, valid_patches={int(valid_patches_per_batch[i])}"
                    diag_lines.append(f"  batch[{i}]: n_image_tokens={n_tok}, expected={n_exp}{gm_info}")

                # Also log a summary of ALL elements for context
                diag_lines.append("")
                diag_lines.append("--- All batch elements summary ---")
                for i in range(batch_size):
                    n_tok = int(n_image_tokens_per_batch[i])
                    n_exp = int(expected_per_batch[i])
                    status = "MISMATCH" if i in mismatched else ("img" if n_tok > 0 else "text")
                    gm_info = ""
                    if grid_mask_array is not None:
                        gm_info = f", valid_patches={int(valid_patches_per_batch[i])}"
                    diag_lines.append(f"  batch[{i}]: tokens={n_tok}, expected={n_exp}, status={status}{gm_info}")

                diag_msg = "\n".join(diag_lines)
                logger.error(diag_msg)

                # Raise with summary (detailed info already logged above)
                first_bad = mismatched[0]
                raise ValueError(
                    f"Image token count mismatch in {len(mismatched)}/{batch_size} batch elements! "
                    f"First mismatch at batch[{first_bad}]: "
                    f"input_ids has {int(n_image_tokens_per_batch[first_bad])} image placeholder tokens, "
                    f"but expected {int(expected_per_batch[first_bad])} features "
                    f"(valid_patches={int(valid_patches_per_batch[first_bad]) if grid_mask_array is not None else 'N/A'} * "
                    f"features_per_patch={features_per_patch_int}). "
                    f"See ERROR log above for full batch diagnostic. "
                    f"Model config: image_size={self.config.vision_config.image_size}, "
                    f"patch_size={self.config.vision_config.patch_size}."
                )

        grid_mask_array = grid_mask.array if grid_mask is not None else None
        # Temporarily disable validation during inference with unpadded input_ids
        # The validation can fail when using unpadded input_ids because the JIT trace
        # may capture an older value of total_image_tokens
        if num_unpadded_features is None:  # Only validate when not using unpadded mode
            jax.debug.callback(_validate_token_feature_match, special_image_mask.array, total_image_tokens, grid_mask_array, features_per_patch)

        # Compute gather indices: for each placeholder, which image token to gather
        if segment_ids is not None and image_segment_ids is not None:
            # Packing mode: use per-segment cumsum
            # Each placeholder should map to the next image feature within its segment
            all_indices = self._compute_per_segment_image_indices(
                special_image_mask.array,
                segment_ids.array if isinstance(segment_ids, NamedArray) else segment_ids,
                image_segment_ids.array if isinstance(image_segment_ids, NamedArray) else image_segment_ids,
                features_per_patch,
                total_image_tokens,
            )
        else:
            # Non-packing mode: global cumsum
            def compute_indices(mask):
                return (jnp.cumsum(mask.astype(jnp.int32)) - 1) % total_image_tokens

            all_indices = jax.vmap(compute_indices)(special_image_mask.array)

        # Gather image features and merge with text embeddings
        gathered = self._batch_gather(image_features_flat.array, all_indices)
        merged = jnp.where(special_image_mask.array[:, :, None], gathered, inputs_embeds.array)

        merged_embeds = hax.named(merged, inputs_embeds.axes)

        # === POSITION ID COMPUTATION ===
        # Use precomputed values if available, otherwise compute on device
        if precomputed_combined_mask is not None and precomputed_position_ids is not None:
            # Use precomputed values from CPU data pipeline
            combined_mask = precomputed_combined_mask.array
            position_ids_array = precomputed_position_ids.array
        else:
            # Compute on device (fallback for inference or when not precomputed)
            # Combined validity mask: valid text OR valid image at placeholder positions
            if unpad_indices is not None and num_unpadded_features is not None:
                # Only the first num_unpadded_features image placeholders are valid
                # The remaining are padding and should not get incrementing position IDs

                # Map each image placeholder to its index (0, 1, 2, ...)
                image_token_indices = jnp.cumsum(special_image_mask.array.astype(jnp.int32), axis=-1) - 1

                # Mark as valid only if the index is within the unpadded range
                image_validity = (image_token_indices < num_unpadded_features).astype(jnp.int32)

                # Combine: valid text OR valid image placeholder
                combined_mask = jnp.where(special_image_mask.array, image_validity, text_mask.array).astype(jnp.int32)
            else:
                # Need to check grid_mask validity for each placeholder position
                # Use ACTUAL valid patches from grid_mask (more robust)
                num_valid_patches = jnp.sum(grid_mask.array, axis=-1, keepdims=True)  # (batch, 1)
                num_valid_tokens = num_valid_patches * features_per_patch  # (batch, 1)

                grid_mask_expanded = jnp.repeat(grid_mask.array, features_per_patch, axis=1)
                image_token_indices = jnp.cumsum(special_image_mask.array.astype(jnp.int32), axis=-1) - 1
                image_token_indices = jnp.clip(image_token_indices, 0, num_valid_tokens - 1)
                image_validity = self._batch_gather(grid_mask_expanded, image_token_indices)
                combined_mask = jnp.where(special_image_mask.array, image_validity, text_mask.array).astype(jnp.int32)

            # Compute compact position IDs
            position_ids_array = self._compute_position_ids(combined_mask)

        Pos = Axis("position", seq_ax.size)
        position_ids = hax.named(position_ids_array, (batch_ax, Pos))

        # Return validity mask for attention masking
        validity_mask = hax.named(combined_mask.astype(jnp.bool_), (batch_ax, seq_ax))

        return merged_embeds, position_ids, validity_mask

    def initial_cache(self, spec, *, dtype):
        """Creates an initial paged KV cache for the language model."""
        tc = self.config.text_config
        kv_heads = Axis("kv_head", tc.num_kv_heads)
        # Use explicit head_dim if set (e.g., Qwen3-4B uses 128), otherwise compute from hidden_dim/num_heads
        head_dim = getattr(tc, 'head_dim', None) or (tc.hidden_dim // tc.num_heads)
        head_size = Axis("head_size", head_dim)
        caches = [KvPageCache.init(spec, kv_heads, head_size, dtype=dtype) for _ in range(tc.num_layers)]
        return ListCache(caches)

    def decode(
        self,
        embeds: NamedArray | None,
        kv_cache,
        batch_info,
        pos_ids: NamedArray,
        *,
        input_ids: NamedArray | None = None,
        pixel_values: Optional[NamedArray] = None,
        grid_mask: Optional[NamedArray] = None,
        unpad_indices: Optional[NamedArray] = None,
        key=None,
    ):
        """Paged decode/prefill using paged KV cache.

        Args:
            embeds: Pre-computed embeddings, or None to compute from input_ids
            kv_cache: KV cache for paged attention
            batch_info: Batch information for paged attention
            pos_ids: Position IDs (will be overwritten if embeds is None and images present)
            input_ids: Input token IDs (required if embeds is None)
            pixel_values: Fixed-shape image patches (batch, TOTAL_PATCHES, C, H, W)
            grid_mask: Boolean mask indicating valid patches (batch, TOTAL_PATCHES)
            unpad_indices: Pre-computed indices to reorder features to HF's unpadded order
            key: Optional PRNGKey

        Returns:
            Tuple of (logits, updated_kv_cache)
        """
        k_vision, key = maybe_rng_split(key, 2) if key is not None and pixel_values is not None else (None, key)

        if embeds is None:
            if input_ids is None:
                raise ValueError("When embeds is None, input_ids is required.")
            embeds, pos_ids, _validity_mask = self._merge_embeddings(
                input_ids=input_ids,
                inputs_embeds=None,
                pixel_values=pixel_values,
                grid_mask=grid_mask,
                unpad_indices=unpad_indices,
                key=k_vision,
            )
            # Note: For paged attention, validity masking is handled through batch_info
            # which manages the KV cache pages and slot positions

        transformer = self.language_model.transformer
        num_layers = self.config.text_config.num_layers
        keys = maybe_rng_split(key, num_layers) if key is not None else [None] * num_layers

        # Ensure batch axis exists for paged_decode
        need_batch = "batch" not in [ax.name for ax in embeds.axes]
        if need_batch:
            embeds = embeds.broadcast_axis(Axis("batch", 1))
            pos_ids = pos_ids.broadcast_axis(Axis("batch", 1))

        # Flatten batch+position when batch_size=1
        batch_axis, pos_axis = embeds.resolve_axis("batch"), embeds.resolve_axis("position")
        need_flatten = batch_axis.size == 1
        if need_flatten:
            embeds = embeds.flatten_axes(("batch", "position"), "position")
            pos_ids = pos_ids.flatten_axes(("batch", "position"), "position")

        x = embeds
        updated_caches = []

        for i in range(num_layers):
            layer = hax.tree_util.tree_map(lambda l: l["layer", i], transformer.layers.stacked)

            # Attention block
            attn_out, cache = layer.self_attn.paged_decode(
                layer.input_layernorm(x), list(kv_cache)[i], batch_info, pos_ids=pos_ids, key=keys[i]
            )
            x = x + attn_out

            # MLP block
            x = x + layer.mlp(layer.post_attention_layernorm(x), key=None)
            updated_caches.append(cache)

        x = transformer.norm(x)

        # Restore shape if flattened
        if need_flatten:
            x = x.unflatten_axis("position", (batch_axis, pos_axis))

        logits = hax.dot(x, self.language_model.get_lm_head(), axis=self.config.TextEmbed)
        return logits, ListCache(updated_caches)


@dataclass(frozen=True)
class VLMRequest:
    """A request for VLM generation that includes image data.

    This extends the concept of Request to include vision-language model data
    with fixed-shape processing for JIT compatibility.

    Uses fixed-shape tensors with masks:
    - pixel_values: (TOTAL_PATCHES, C, H, W) - padded to fixed size
    - grid_mask: (TOTAL_PATCHES,) - True for valid patches, False for padding
    - unpad_indices: (num_image_tokens,) - indices to reorder features to HF's unpadded order
    """

    prompt_tokens: list[int]
    request_id: int
    decode_params: "SeqDecodingParams"  # From levanter.inference.jit_scheduler
    n_generations: int

    # VLM-specific fields (fixed-shape for JIT compatibility)
    pixel_values: NamedArray  # (TOTAL_PATCHES, C, H, W) - FIXED shape, padded
    grid_mask: NamedArray  # (TOTAL_PATCHES,) - boolean mask for valid patches
    input_ids: Optional[NamedArray] = None  # Full input_ids with image tokens
    unpad_indices: Optional[NamedArray] = None  # Indices for HF-style feature ordering
    num_unpadded_features: Optional[int] = None  # Actual number of unpadded features (for combined_mask)


class _LlavaInferenceWrapper(eqx.Module):
    """Adapter to run LlavaOnevisionModel through InferenceEngine.

    This wrapper keeps the full LlavaOnevisionModel and dynamically computes
    embeddings during prefill. It stores the current request's image data
    (pixel_values, grid_mask, input_ids) and uses them during the prefill phase.

    Supports both single-request and batched processing:
    - Single request: pixel_values (TOTAL_PATCHES, C, H, W), grid_mask (TOTAL_PATCHES,)
    - Batched: pixel_values (batch, TOTAL_PATCHES, C, H, W), grid_mask (batch, TOTAL_PATCHES)

    Usage:
        # Create wrapper with the model
        wrapper = _LlavaInferenceWrapper.create(
            model=model,
            Vocab=Vocab,
            mesh=mesh,  # Optional: for sharding
        )

        # Option 1: Set image data for current request before generation
        wrapper = wrapper.set_request_data(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_mask=grid_mask,
        )

        # Option 2: Use VLMRequest with LlavaInferenceEngine (recommended)
        engine = LlavaInferenceEngine.from_model_with_config(model=wrapper, ...)
        vlm_request = VLMRequest(
            prompt_tokens=...,
            pixel_values=...,
            grid_mask=...,
            ...
        )
        result = engine.generate([vlm_request])
    """

    model: "LlavaOnevisionModel"
    Vocab: Axis = eqx.field(static=True)
    _text_config: "QwenConfig" = eqx.field(static=True)

    # Request-specific data (set before each generation)
    # Supports both single (no batch axis) and batched (with batch axis) inputs
    _input_ids: NamedArray | None = None
    _pixel_values: NamedArray | None = None
    _grid_mask: NamedArray | None = None
    _unpad_indices: NamedArray | None = None
    _num_unpadded_features: int | None = eqx.field(static=True, default=None)
    _precomputed_image_features: NamedArray | None = None  # For batched inference

    # Batch size for tracking (1 for single request, >1 for batched)
    _batch_size: int = eqx.field(static=True, default=1)

    # Cached embeddings and position IDs (computed lazily during prefill)
    _cached_embeds: NamedArray | None = None
    _cached_pos_ids: NamedArray | None = None

    # Position offset for decode phase (static since it's computed from static fields)
    # This is the difference between the padded sequence length and the actual valid length
    _position_offset: int = eqx.field(static=True, default=0)

    # Batched inference mode fields
    # When _batched_mode is True, multiple requests are processed together
    _batched_mode: bool = eqx.field(static=True, default=False)
    # Stacked image features from all requests: (total_patches, features_per_patch, embed)
    _batched_image_features: NamedArray | None = None
    # Number of patches per segment (request), used to build image_segment_ids
    _batched_num_patches_per_segment: tuple[int, ...] | None = eqx.field(static=True, default=None)
    # Mapping from slot_id to segment index
    _slot_to_segment_map: dict[int, int] | None = eqx.field(static=True, default=None)
    # Input IDs per segment (slot_id → input_ids), for building segment_ids from token patterns
    _batched_input_ids_per_slot: dict[int, NamedArray] | None = None

    @classmethod
    def create(
        cls,
        model: "LlavaOnevisionModel",
        Vocab: Axis,
        mesh=None,
    ) -> "_LlavaInferenceWrapper":
        """Create a wrapper from a LlavaOnevisionModel.

        Args:
            model: The full LlavaOnevisionModel
            Vocab: Vocabulary axis
            mesh: Optional JAX mesh for sharding

        Returns:
            A wrapper that holds the full model
        """

        # Note: We no longer force replication here.
        # The model should already be properly sharded (FSDP-style with embed on data axis).
        # Forcing replication would cause OOM for large models.
        # If the model is already on the correct mesh with proper sharding, we keep it as-is.
        # The InferenceEngine's named_jit will handle the compute-time sharding appropriately.

        return cls(
            model=model,
            Vocab=Vocab,
            _text_config=model.config.text_config,
        )

    def set_request_data(
        self,
        input_ids: NamedArray,
        pixel_values: NamedArray,
        grid_mask: NamedArray,
        unpad_indices: Optional[NamedArray] = None,
        num_unpadded_features: Optional[int] = None,
        batch_size: int = 1,
        precomputed_image_features: Optional[NamedArray] = None,
    ) -> "_LlavaInferenceWrapper":
        """Set the image data for the current request(s).

        This must be called before generating with InferenceEngine.

        Args:
            input_ids: Input token IDs with shape (batch, position)
            pixel_values: Fixed-shape pixel values
                - Single request: (TOTAL_PATCHES, C, H, W)
                - Batched: (batch, TOTAL_PATCHES, C, H, W)
            grid_mask: Boolean mask indicating valid patches
                - Single request: (TOTAL_PATCHES,)
                - Batched: (batch, TOTAL_PATCHES)
            unpad_indices: Pre-computed indices to reorder features to HF's unpadded order
            num_unpadded_features: Actual number of unpadded features (before padding)
            batch_size: Number of requests in the batch (1 for single request)
            precomputed_image_features: Optional pre-computed image features for batched inference
                (batch, TOTAL_PATCHES, features_per_patch, embed) - skips vision encoder if provided

        Returns:
            A new wrapper with the request data set
        """
        # Compute position offset for decode phase
        # This is the difference between image placeholders in input_ids and actual unpadded features
        # When using unpadded input_ids (like HF), this will be 0
        # When using padded input_ids (for training), this will be padding_count
        position_offset = 0
        if num_unpadded_features is not None:
            # Count image placeholders in input_ids
            image_token_id = self.model.config.image_token_index
            num_image_placeholders = int(jnp.sum(input_ids.array == image_token_id))
            position_offset = num_image_placeholders - num_unpadded_features

        # Create a new instance with updated request data
        return _LlavaInferenceWrapper(
            model=self.model,
            Vocab=self.Vocab,
            _text_config=self._text_config,
            _input_ids=input_ids,
            _pixel_values=pixel_values,
            _grid_mask=grid_mask,
            _unpad_indices=unpad_indices,
            _num_unpadded_features=num_unpadded_features,
            _precomputed_image_features=precomputed_image_features,
            _batch_size=batch_size,
            _position_offset=position_offset,
            _cached_embeds=None,
            _cached_pos_ids=None,
        )

    def set_batched_request_data(
        self,
        requests: list,  # list of VLMRequest
        slot_ids: list[int],
    ) -> "_LlavaInferenceWrapper":
        """Set data for multiple requests for batched inference.

        This enables true data parallelism where multiple requests are processed
        together in a single prefill and decode loop.

        Args:
            requests: List of VLMRequest objects
            slot_ids: List of slot IDs assigned by the engine to each request

        Returns:
            A new wrapper configured for batched inference
        """
        import logging
        logger = logging.getLogger(__name__)

        if len(requests) != len(slot_ids):
            raise ValueError("Number of requests must match number of slot_ids")

        if len(requests) == 0:
            raise ValueError("At least one request is required")

        # Pre-compute image features for all requests in one batched call
        # Stack pixel_values and grid_masks from all requests
        # First, squeeze out any existing batch dimension (size 1) from each request
        def squeeze_batch(arr):
            """Remove batch dimension if it exists and has size 1."""
            if arr is None:
                return None
            if "batch" in [ax.name for ax in arr.axes]:
                return arr["batch", 0]
            return arr

        pixel_values_list = [squeeze_batch(req.pixel_values) for req in requests]
        grid_mask_list = [squeeze_batch(req.grid_mask) for req in requests]

        # Find the maximum TOTAL_PATCHES across all requests
        def get_total_patches(pv):
            for ax in pv.axes:
                if ax.name == "TotalPatches":
                    return ax.size
            raise ValueError("TotalPatches axis not found in pixel_values")

        total_patches_list = [get_total_patches(pv) for pv in pixel_values_list]
        max_total_patches = max(total_patches_list)
        logger.info(f"Padding {len(requests)} requests to max TOTAL_PATCHES={max_total_patches}")

        # Pad each request's pixel_values and grid_mask to max_total_patches
        padded_pixel_values = []
        padded_grid_masks = []
        for pv, gm, current_patches in zip(pixel_values_list, grid_mask_list, total_patches_list):
            if current_patches < max_total_patches:
                # Need to pad
                pad_size = max_total_patches - current_patches

                # Pad pixel_values with zeros (hax.pad auto-resizes the axis)
                pv_padded = hax.pad(pv, {"TotalPatches": (0, pad_size)}, constant_values=0.0)

                # Pad grid_mask with False (indicates padding is invalid)
                gm_padded = hax.pad(gm, {"TotalPatches": (0, pad_size)}, constant_values=False)

                padded_pixel_values.append(pv_padded)
                padded_grid_masks.append(gm_padded)
            else:
                # No padding needed
                padded_pixel_values.append(pv)
                padded_grid_masks.append(gm)

        # Stack along a new batch dimension (now all have the same shape)
        Batch = Axis("batch", len(requests))
        stacked_pixel_values = hax.stack(Batch, padded_pixel_values)
        stacked_grid_mask = hax.stack(Batch, padded_grid_masks)

        # Compute image features for all requests at once
        with hax.axis_mapping({}):
            batched_features, _ = self.model.get_image_features(
                pixel_values=stacked_pixel_values,
                grid_mask=stacked_grid_mask,
            )

        # Flatten the batched features: (batch, patches, features_per_patch, embed) -> (total_patches, features_per_patch, embed)
        # First get the shape info
        num_requests = len(requests)
        patches_per_request = batched_features.axis_size("TotalPatches")
        features_per_patch = batched_features.axis_size("features_per_patch")
        embed_size = batched_features.axis_size("embed")

        # Create axes for the flattened features
        TotalPatches = Axis("TotalPatches", num_requests * patches_per_request)
        FeaturesPerPatch = Axis("features_per_patch", features_per_patch)
        Embed = Axis("embed", embed_size)

        # Reshape: (batch, patches, features_per_patch, embed) -> (total_patches, features_per_patch, embed)
        flattened_features = batched_features.array.reshape(
            num_requests * patches_per_request, features_per_patch, embed_size
        )
        flattened_features = hax.named(flattened_features, (TotalPatches, FeaturesPerPatch, Embed))

        # Build slot_id to segment mapping (segment 0, 1, 2, ... based on order in requests)
        slot_to_segment = {slot: i for i, slot in enumerate(slot_ids)}

        # Count valid patches per segment (from original grid_masks before padding)
        # This is the number of TRUE values in each grid_mask
        num_patches_per_segment = tuple(
            int(jnp.sum(gm.array).item()) for gm in grid_mask_list
        )

        # Store input_ids per slot for later use in decode()
        input_ids_per_slot = {slot: req.input_ids for slot, req in zip(slot_ids, requests)}

        return _LlavaInferenceWrapper(
            model=self.model,
            Vocab=self.Vocab,
            _text_config=self._text_config,
            _batched_mode=True,
            _batched_image_features=flattened_features,
            _batched_num_patches_per_segment=num_patches_per_segment,
            _slot_to_segment_map=slot_to_segment,
            _batched_input_ids_per_slot=input_ids_per_slot,
        )

    def _compute_embeddings(self) -> Tuple[NamedArray, NamedArray]:
        """Compute merged embeddings and position IDs for the current request.

        Returns:
            Tuple of (merged_embeds, position_ids)
        """
        if self._input_ids is None or self._pixel_values is None:
            raise ValueError("Request data not set. Call set_request_data() before generation.")

        # Use empty axis_mapping to avoid auto_sharding issues with
        # vision encoder's intermediate tensors (e.g., 31 patches not divisible by 4)
        with hax.axis_mapping({}):
            merged_embeds, position_ids, _ = self.model._merge_embeddings(
                input_ids=self._input_ids,
                inputs_embeds=None,
                pixel_values=self._pixel_values,
                grid_mask=self._grid_mask,
                unpad_indices=self._unpad_indices,
                num_unpadded_features=self._num_unpadded_features,
                precomputed_image_features=self._precomputed_image_features,
                key=None,
            )

        # Squeeze batch axis since InferenceEngine expects no batch axis
        if "batch" in [ax.name for ax in merged_embeds.axes]:
            merged_embeds = merged_embeds["batch", 0]
            position_ids = position_ids["batch", 0]

        return merged_embeds, position_ids

    def _compute_batched_embeddings(
        self,
        tokens: NamedArray,
        batch_info,
    ) -> NamedArray:
        """Compute embeddings for batched mode with flat packed tokens.

        This method handles multiple requests packed into a single flat array.
        It computes per-token segment_ids from batch_info and merges image
        features using segment-aware indexing.

        Args:
            tokens: Flat packed tokens with shape (position,)
            batch_info: PageBatchInfo containing slot_ids, cu_q_lens, etc.

        Returns:
            merged_embeds: embeddings with image features merged at placeholder positions
        """
        if self._batched_image_features is None or self._slot_to_segment_map is None:
            raise ValueError("Batched mode data not set. Call set_batched_request_data() first.")

        lm = self.model.language_model
        num_tokens = tokens.axis_size("position")

        # 1. Get text embeddings for all packed tokens
        text_embeds = lm.embeddings.embed(tokens)  # (position, embed)

        # 2. Compute per-token segment_ids from batch_info
        # batch_info.cu_q_lens: cumulative token counts [0, n0, n0+n1, ...]
        # IMPORTANT: cu_q_lens has max_seqs+1 elements, but only the first num_seqs+1 are valid!
        # The rest are zeros, which breaks searchsorted (it assumes sorted array).
        cu_q_lens_full = batch_info.cu_q_lens.array  # (max_seqs+1,)
        num_seqs = batch_info.num_seqs

        # Fix: Replace invalid (zero) entries beyond num_seqs+1 with a large value
        # so searchsorted works correctly. This is JIT-compatible because we use
        # jnp.where with a mask rather than Python slicing.
        max_seqs_static = cu_q_lens_full.shape[0] - 1  # max_seqs from shape
        valid_mask = jnp.arange(max_seqs_static + 1) <= num_seqs
        # Replace invalid entries with a large value (larger than any token position)
        large_value = num_tokens + 1000
        cu_q_lens = jnp.where(valid_mask, cu_q_lens_full, large_value)

        # For each position, find which sequence it belongs to in the current batch
        # batch_seq_indices[i] = j means token i belongs to sequence j in this batch
        positions = jnp.arange(num_tokens)
        # searchsorted gives index where position would be inserted to keep sorted order
        # We use 'right' and subtract 1 to get the batch sequence index containing each position
        batch_seq_indices = jnp.searchsorted(cu_q_lens, positions, side='right') - 1
        batch_seq_indices = jnp.clip(batch_seq_indices, 0, num_seqs - 1)

        # Get number of segments from batched data
        num_batched_segments = len(self._batched_num_patches_per_segment)

        # Map batch sequence indices to segment indices in stored image features
        # The slot_ids in _slot_to_segment_map match the engine's allocation order
        # (engine uses free_slots.pop() which pops from the END, so [7,6,5,...,0])
        #
        # Create a lookup table: slot_id -> segment_index
        max_slots = batch_info.slot_ids.axis_size("seq")
        slot_to_segment_array = jnp.zeros(max_slots, dtype=jnp.int32)
        for slot_id, segment_idx in self._slot_to_segment_map.items():
            slot_to_segment_array = slot_to_segment_array.at[slot_id].set(segment_idx)

        # Get slot_ids for each batch sequence position
        batch_slot_ids = batch_info.slot_ids.array  # (max_seqs,)

        # Map: token position -> batch_seq_idx -> slot_id -> segment_idx
        slot_ids_per_token = batch_slot_ids[batch_seq_indices]
        segment_ids = slot_to_segment_array[slot_ids_per_token]

        # Ensure segment_ids are valid (within num_batched_segments)
        segment_ids = jnp.clip(segment_ids, 0, num_batched_segments - 1)

        # 3. Identify image placeholder tokens
        image_token_id = self.model.config.image_token_index
        is_image_token = tokens.array == image_token_id  # (position,)

        # 4. Compute gather indices for image features
        # For each image placeholder, compute which image feature to gather
        # Using per-segment cumsum
        features_per_patch = self._batched_image_features.axis_size("features_per_patch")
        total_patches_axis = self._batched_image_features.axis_size("TotalPatches")

        # Compute starting feature index for each segment
        # After padding, all segments have the same number of patches (max_total_patches)
        # So we use uniform patch count per segment, not the original (unpadded) counts
        #
        # Layout after padding (with max_patches = patches_per_segment = total_patches / num_segments):
        # segment 0: features [0, patches_per_segment * fpp)
        # segment 1: features [patches_per_segment * fpp, 2 * patches_per_segment * fpp)
        # etc.
        patches_per_segment = total_patches_axis // num_batched_segments
        features_per_segment_padded = patches_per_segment * features_per_patch

        # segment_feature_starts[s] = s * features_per_segment_padded
        segment_feature_starts = jnp.arange(num_batched_segments) * features_per_segment_padded

        # Compute per-segment cumsum of image placeholders using scan
        def segment_cumsum_fn(carry, x):
            mask_val, seg_id = x
            # carry is (count_per_segment,)
            count = carry[seg_id]
            new_carry = carry.at[seg_id].add(mask_val)
            return new_carry, count

        init_counts = jnp.zeros(num_batched_segments, dtype=jnp.int32)
        _, within_seg_counts = jax.lax.scan(
            segment_cumsum_fn,
            init_counts,
            (is_image_token.astype(jnp.int32), segment_ids)
        )

        # DEBUG: Check for mismatch between image placeholders and features
        image_token_count = jnp.sum(is_image_token)
        # Only count within_seg for image tokens to get accurate max
        max_within_seg_image_only = jnp.max(within_seg_counts * is_image_token.astype(jnp.int32))
        total_features_computed = total_patches_axis * features_per_patch
        jax.debug.print(
            "Batched embeddings debug: "
            "num_seqs={ns}, num_batched_segments={nbs}, "
            "image_token_count={itc}, total_features={tf}, "
            "max_within_seg_count={mwsc}, features_per_segment={fps}, "
            "slot_to_segment_map_size={stsm}",
            ns=num_seqs, nbs=num_batched_segments,
            itc=image_token_count, tf=total_features_computed,
            mwsc=max_within_seg_image_only, fps=features_per_segment_padded,
            stsm=len(self._slot_to_segment_map)
        )
        # Check for out-of-bounds (only considering image tokens)
        potential_max_idx = jnp.max(segment_feature_starts) + max_within_seg_image_only
        # Print whether there's potential overflow
        jax.debug.print(
            "Feature bounds check: potential_max_idx={mpi}, total_features={tf}, overflow={of}",
            mpi=potential_max_idx, tf=total_features_computed,
            of=potential_max_idx >= total_features_computed
        )

        # Compute final feature indices
        # For each image placeholder at position i with segment s:
        # feature_index = segment_feature_starts[s] + within_seg_counts[i]
        segment_starts_per_token = segment_feature_starts[segment_ids]
        feature_indices = segment_starts_per_token + within_seg_counts

        # DEBUG: Print more details about the mismatch
        jax.debug.print(
            "Feature index debug: max_feature_idx={mfi}, segment_starts={ss}, patches_per_seg={pps}",
            mfi=jnp.max(feature_indices * is_image_token.astype(jnp.int32)),
            ss=segment_feature_starts,
            pps=patches_per_segment
        )

        # 5. Flatten image features to (total_features, embed)
        # _batched_image_features: (total_patches, features_per_patch, embed)
        total_patches = self._batched_image_features.axis_size("TotalPatches")
        embed_size = self._batched_image_features.axis_size("embed")
        flat_features = self._batched_image_features.array.reshape(total_patches * features_per_patch, embed_size)

        # 6. Gather image features at placeholder positions
        gathered_features = flat_features[feature_indices]  # (position, embed)

        # 7. Merge: replace image placeholders with image features
        merged = jnp.where(is_image_token[:, None], gathered_features, text_embeds.array)
        merged_embeds = hax.named(merged, text_embeds.axes)

        return merged_embeds

    def initial_cache(self, spec, *, dtype):
        """Creates an initial paged KV cache for the language model."""
        return self.model.initial_cache(spec, dtype=dtype)

    @property
    def Pos(self):
        """Return the position axis based on input_ids."""
        if self._input_ids is not None:
            return self._input_ids.resolve_axis("position")
        raise ValueError("Request data not set. Call set_request_data() first.")

    @property
    def language_model(self):
        """Access the underlying language model."""
        return self.model.language_model

    def decode(self, tokens, kv_cache, batch_info, pos_ids):
        """Decode using dynamically computed embeddings for prefill, language model for decode.

        For VLM with masked tokens (e.g., anyres padding):
        - During prefill: Uses position IDs from _compute_embeddings() which skip masked tokens
        - During decode: Adjusts position IDs to account for skipped padding tokens

        For batched inference (_batched_mode=True):
        - During prefill: Uses _compute_batched_embeddings() which handles flat packed tokens
        - During decode: Uses standard text embeddings
        """
        is_prefill = tokens.axis_size("position") > 1
        lm = self.model.language_model

        if is_prefill:
            if self._batched_mode:
                # Batched mode: compute embeddings for flat packed tokens
                # Use the pos_ids passed by the engine (already correct per-sequence positions)
                embeds = self._compute_batched_embeddings(tokens, batch_info)
                # pos_ids stays as passed by the engine
            else:
                # Single request mode: use position IDs from _compute_embeddings for proper RoPE with padding
                embeds, computed_pos_ids = self._compute_embeddings()
                pos_ids = computed_pos_ids
        else:
            embeds = lm.embeddings.embed(tokens)
            # Adjust position IDs for decode phase to account for skipped padding
            # During prefill, we skip _position_offset padding tokens
            # So decode positions should be adjusted by the same amount
            # Note: In batched mode, _position_offset is 0 so no adjustment needed
            if self._position_offset > 0:
                pos_ids = hax.named(pos_ids.array - self._position_offset, pos_ids.axes)

        x, new_cache = lm.transformer.decode(kv_cache, embeds, batch_info, pos_ids, key=None)
        logits = lm.lm_head(x, key=None) if lm.lm_head is not None else lm.embeddings.unembed(x)
        return logits, new_cache


class LlavaInferenceEngine:
    """InferenceEngine for LlavaOnevision that handles VLMRequest.

    This engine wraps a standard InferenceEngine and extracts VLM-specific
    data (pixel_values, image_sizes, etc.) from VLMRequest objects before
    generation.

    Usage:
        # Create engine
        engine = LlavaInferenceEngine.from_model_with_config(
            model=lev_model,
            tokenizer=processor.tokenizer,
            config=engine_config,
            Vocab=Vocab,
            mesh=mesh,
        )

        # Create VLM request (with fixed-shape tensors)
        vlm_request = VLMRequest(
            prompt_tokens=prompt_tokens,
            request_id=0,
            decode_params=decode_params,
            n_generations=1,
            pixel_values=pixel_values,  # (TOTAL_PATCHES, C, H, W) - padded
            grid_mask=grid_mask,  # (TOTAL_PATCHES,) - boolean
            input_ids=input_ids,
        )

        # Generate
        result = engine.generate([vlm_request])
    """

    def __init__(
        self,
        wrapper: _LlavaInferenceWrapper,
        base_engine,  # InferenceEngine
    ):
        """Initialize with a wrapper and base engine.

        Args:
            wrapper: The _LlavaInferenceWrapper (without request data set)
            base_engine: The underlying InferenceEngine
        """
        self._wrapper = wrapper
        self._base_engine = base_engine

    @classmethod
    def from_model_with_config(
        cls,
        model: "LlavaOnevisionModel",
        tokenizer,
        config,  # InferenceEngineConfig
        Vocab: Axis,
        mesh=None,
    ) -> "LlavaInferenceEngine":
        """Build a LlavaInferenceEngine from a model and config.

        Args:
            model: The LlavaOnevisionModel
            tokenizer: Tokenizer with encode/decode methods
            config: InferenceEngineConfig for sizing
            Vocab: Vocabulary axis
            mesh: Optional JAX mesh for sharding

        Returns:
            A LlavaInferenceEngine ready for generation
        """
        # Create the wrapper
        wrapper = _LlavaInferenceWrapper.create(
            model=model,
            Vocab=Vocab,
            mesh=mesh,
        )

        # Create the base engine with the wrapper
        base_engine = InferenceEngine.from_model_with_config(
            model=wrapper,
            tokenizer=tokenizer,
            config=config,
        )

        return cls(wrapper=wrapper, base_engine=base_engine)

    def generate(self, requests: list[VLMRequest], step_callback=None, use_batched_inference: bool = True):
        """Generate tokens for a batch of VLMRequests.

        This method supports two modes:
        1. True batched inference (use_batched_inference=True): All requests are processed
           together in a single prefill and decode loop. This provides maximum parallelism.
        2. Sequential inference (use_batched_inference=False): Each request is processed
           separately, with vision encoder batching for image features.

        Args:
            requests: List of VLMRequest objects
            step_callback: Optional callback for each decode iteration
            use_batched_inference: Whether to use true batched LLM inference (default True)

        Returns:
            GenerationResult with tokens, logprobs, and total_generated
        """
        if not requests:
            raise ValueError("At least one request is required")

        # For single request, use the simpler sequential path
        if len(requests) == 1:
            return self._generate_sequential(requests, step_callback)

        # Try true batched inference if enabled
        if use_batched_inference:
            result = self._try_batched_inference(requests, step_callback)
            if result is not None:
                return result
            # Fall back to sequential if batched inference fails

        # Sequential inference with vision encoder batching
        return self._generate_sequential(requests, step_callback)

    def _try_batched_inference(self, requests: list[VLMRequest], step_callback=None):
        """Try to use true batched LLM inference for all requests.

        Padding is handled in set_batched_request_data() so different TOTAL_PATCHES sizes
        are supported.

        Returns:
            GenerationResult if successful, None if batching not possible
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            # Reset the engine first to ensure clean state
            self._base_engine.reset()

            # Get the actual free_slots order from the engine to match slot_ids
            # Engine uses free_slots.pop() which pops from the END of the list
            # After reset(), free_slots = [0, 1, 2, ..., max_seqs-1]
            # So pop() returns: max_seqs-1, max_seqs-2, ..., 1, 0
            engine_free_slots = list(self._base_engine.free_slots)  # Copy the list
            num_requests = len(requests)

            if len(engine_free_slots) < num_requests:
                logger.warning(f"Not enough free slots: {len(engine_free_slots)} < {num_requests}")
                return None

            # Predict which slot_ids the engine will assign (pop from end)
            # First request gets engine_free_slots[-1], second gets engine_free_slots[-2], etc.
            predicted_slot_ids = [engine_free_slots[-(i+1)] for i in range(num_requests)]

            logger.info(f"Engine free_slots (first 16): {engine_free_slots[:16]}, predicted slot_ids: {predicted_slot_ids}")

            # Use the predicted slot_ids for our mapping
            slot_ids = predicted_slot_ids

            # Set up batched mode on the wrapper
            self._base_engine.model = self._wrapper.set_batched_request_data(requests, slot_ids)

            # Convert VLMRequests to standard Requests for the base engine
            standard_requests = [
                Request(
                    prompt_tokens=vlm_request.prompt_tokens,
                    request_id=i,  # Use index as request_id for slot mapping
                    decode_params=vlm_request.decode_params,
                    n_generations=vlm_request.n_generations,
                )
                for i, vlm_request in enumerate(requests)
            ]

            # Generate using the base engine with all requests at once
            result = self._base_engine.generate(standard_requests, step_callback=step_callback)

            # Reset the engine state
            self._base_engine.reset()

            logger.info(f"Batched inference completed for {len(requests)} requests")
            return result

        except Exception as e:
            logger.warning(f"Batched inference failed: {e}. Falling back to sequential.")
            # Reset engine state in case of failure
            self._base_engine.reset()
            return None

    def _generate_sequential(self, requests: list[VLMRequest], step_callback=None):
        """Generate using sequential processing with vision encoder batching.

        This is the fallback mode when true batched inference is not possible.
        """
        # Pre-compute image features for all requests in one batched call
        precomputed_features = self._batch_compute_image_features(requests)

        # Process each request sequentially (LLM generation)
        all_results = []
        for i, vlm_request in enumerate(requests):
            # Set the VLM data on the wrapper for this request
            self._base_engine.model = self._wrapper.set_request_data(
                input_ids=vlm_request.input_ids,
                pixel_values=vlm_request.pixel_values,
                grid_mask=vlm_request.grid_mask,
                unpad_indices=vlm_request.unpad_indices,
                num_unpadded_features=vlm_request.num_unpadded_features,
                precomputed_image_features=precomputed_features[i] if precomputed_features else None,
            )

            # Convert VLMRequest to standard Request for the base engine
            standard_request = Request(
                prompt_tokens=vlm_request.prompt_tokens,
                request_id=vlm_request.request_id,
                decode_params=vlm_request.decode_params,
                n_generations=vlm_request.n_generations,
            )

            # Generate using the base engine
            result = self._base_engine.generate([standard_request], step_callback=step_callback)
            all_results.append(result)

            # Reset the engine state for the next request
            self._base_engine.reset()

        # Merge results if multiple requests
        if len(all_results) == 1:
            return all_results[0]

        return self._merge_generation_results(all_results)

    def _batch_compute_image_features(self, requests: list[VLMRequest]) -> list[NamedArray] | None:
        """Pre-compute image features for all requests in a single batched forward pass.

        This batches all requests' pixel_values together and runs them through the
        vision encoder once, which is much faster than processing them one by one.

        Args:
            requests: List of VLMRequest objects

        Returns:
            List of image features, one per request (each with shape TOTAL_PATCHES, features_per_patch, embed)
            Returns None if batching is not possible (e.g., mismatched shapes)
        """
        if len(requests) <= 1:
            # No benefit from batching single request
            return None

        try:
            # Stack pixel_values from all requests: (num_requests, TOTAL_PATCHES, C, H, W)
            # First, squeeze out any existing batch dimension (size 1) from each request
            def squeeze_batch(arr):
                """Remove batch dimension if it exists and has size 1."""
                if arr is None:
                    return None
                if "batch" in [ax.name for ax in arr.axes]:
                    return arr["batch", 0]
                return arr

            pixel_values_list = [squeeze_batch(req.pixel_values) for req in requests]
            grid_mask_list = [squeeze_batch(req.grid_mask) for req in requests]

            # Find the maximum TOTAL_PATCHES across all requests
            def get_total_patches(pv):
                for ax in pv.axes:
                    if ax.name == "TotalPatches":
                        return ax.size
                raise ValueError("TotalPatches axis not found in pixel_values")

            total_patches_list = [get_total_patches(pv) for pv in pixel_values_list]
            max_total_patches = max(total_patches_list)

            # Pad each request's pixel_values and grid_mask to max_total_patches
            padded_pixel_values = []
            padded_grid_masks = []
            for pv, gm, current_patches in zip(pixel_values_list, grid_mask_list, total_patches_list):
                if current_patches < max_total_patches:
                    # Need to pad
                    pad_size = max_total_patches - current_patches

                    # Pad pixel_values with zeros (hax.pad auto-resizes the axis)
                    pv_padded = hax.pad(pv, {"TotalPatches": (0, pad_size)}, constant_values=0.0)

                    # Pad grid_mask with False (indicates padding is invalid)
                    gm_padded = hax.pad(gm, {"TotalPatches": (0, pad_size)}, constant_values=False)

                    padded_pixel_values.append(pv_padded)
                    padded_grid_masks.append(gm_padded)
                else:
                    # No padding needed
                    padded_pixel_values.append(pv)
                    padded_grid_masks.append(gm)

            # Stack along a new batch dimension (now all have the same shape)
            Batch = Axis("batch", len(requests))
            stacked_pixel_values = hax.stack(Batch, padded_pixel_values)
            stacked_grid_mask = hax.stack(Batch, padded_grid_masks)

            # Compute image features for all requests at once
            # Use empty axis_mapping to avoid sharding issues
            with hax.axis_mapping({}):
                batched_features, _ = self._wrapper.model.get_image_features(
                    pixel_values=stacked_pixel_values,
                    grid_mask=stacked_grid_mask,
                )

            # Split back into individual features
            # batched_features: (batch, TOTAL_PATCHES, features_per_patch, embed)
            # Keep batch dimension (size 1) for each feature to match _merge_embeddings expectations
            # Also slice back to original TOTAL_PATCHES size for each request
            Batch1 = Axis("batch", 1)
            features_list = []
            for i, orig_patches in enumerate(total_patches_list):
                single_feature = batched_features["batch", i]  # (max_TOTAL_PATCHES, features_per_patch, embed)

                # Slice back to original TotalPatches if needed
                if orig_patches < max_total_patches:
                    # Create axis with original size and slice
                    single_feature = single_feature["TotalPatches", :orig_patches]

                # Add batch dimension back with size 1 using broadcast_axis
                single_feature_batched = hax.broadcast_axis(single_feature, Batch1)
                features_list.append(single_feature_batched)
            return features_list

        except Exception as e:
            # If batching fails for any reason, fall back to sequential processing
            import logging
            logging.getLogger(__name__).warning(f"Failed to batch image features: {e}. Falling back to sequential.")
            return None

    def _merge_generation_results(self, results: list):
        """Merge multiple GenerationResults into one.

        Args:
            results: List of GenerationResult objects

        Returns:
            Merged GenerationResult
        """
        from levanter.inference.engine import GenerationResult

        # Concatenate tokens from all results
        all_tokens = []
        all_logprobs = []
        total_generated = 0

        for result in results:
            all_tokens.extend(result.tokens)
            if result.logprobs is not None:
                all_logprobs.extend(result.logprobs)
            total_generated += result.total_generated

        return GenerationResult(
            tokens=all_tokens,
            logprobs=all_logprobs if all_logprobs else None,
            total_generated=total_generated,
        )

    def reset(self):
        """Reset the engine state."""
        self._base_engine.reset()

    @property
    def config(self):
        """Return the engine config."""
        return self._base_engine.config


__all__ = [
    "LlavaOnevisionConfig",
    "LlavaOnevisionMultimodalProjector",
    "LlavaOnevisionModel",
    "_LlavaInferenceWrapper",
    "VLMRequest",
    "LlavaInferenceEngine",
]
