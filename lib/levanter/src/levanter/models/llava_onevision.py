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

        # Rename vision embed axis to avoid collision with text embed axis
        # Vision features come with "embed" axis from Siglip2, but we need to map it to text "embed" axis
        # First, rename to a temporary unique name to avoid axis collision during projection
        image_features = image_features.rename({"embed": "vision_embed"})

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

        # First, create a mask for valid patches (segment_id >= 0)
        valid_patch_mask = (image_segment_ids >= 0).astype(jnp.int32)

        # Compute cumulative feature count up to each patch
        # Each valid patch contributes features_per_patch features
        cumulative_features = jnp.cumsum(valid_patch_mask * features_per_patch, axis=-1)
        # patch_feature_starts[p] = cumulative_features[p-1] or 0 for p=0
        patch_feature_starts = jnp.concatenate([
            jnp.zeros((batch_size, 1), dtype=jnp.int32),
            cumulative_features[:, :-1]
        ], axis=-1)

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
            def update_segment_start(segment_starts, patch_idx):
                seg_id = img_seg_ids[patch_idx]
                valid = seg_id >= 0
                # If this segment hasn't been seen yet (current value is 0 and we're not at start)
                # or if this patch comes earlier, update
                current = segment_starts[seg_id]
                feature_start = patch_starts[patch_idx]
                # Only update if valid and this is the first occurrence
                # We detect first occurrence by checking if all previous patches have different seg_id
                is_first = jnp.sum((img_seg_ids[:patch_idx] == seg_id).astype(jnp.int32)) == 0
                should_update = valid & is_first
                new_value = jnp.where(should_update, feature_start, current)
                return segment_starts.at[seg_id].set(new_value), None

            segment_starts, _ = jax.lax.scan(update_segment_start, segment_starts, jnp.arange(max_patches))
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

                # Get current count for this segment
                current_count = seg_counts[seg_id]

                # Update count if this is a placeholder with valid segment
                valid = is_placeholder & (seg_id >= 0)
                new_count = jnp.where(valid, current_count + 1, current_count)
                seg_counts = seg_counts.at[seg_id].set(new_count)

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

            # Compute global indices: seg_start + local_idx * features_per_patch
            # But we need to handle the case where each placeholder expands to features_per_patch
            # Actually, local_indices counts placeholders, each gets features_per_patch tokens
            # The i-th placeholder in a segment gets features [i*features_per_patch : (i+1)*features_per_patch]
            # We gather the first feature of each patch, so index = i * features_per_patch

            # Get segment start for each position
            seg_start_per_pos = seg_starts[jnp.clip(seg_ids, 0, 63)]

            # Compute final index: segment_start + local_index * features_per_patch
            # local_indices is the count of placeholders before this one in the same segment
            global_indices = seg_start_per_pos + local_indices * features_per_patch

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
            # Check first batch element (assuming all batches have same structure)
            n_image_tokens = int(n_image_tokens_per_batch[0])

            # Calculate expected features based on grid_mask if provided
            if grid_mask_array is not None:
                # Use number of valid patches (True values in grid_mask)
                grid_mask_np = np.asarray(grid_mask_array)
                num_valid_patches = int(np.sum(grid_mask_np[0]))  # First batch element
                expected_features = num_valid_patches * features_per_patch_int
            else:
                expected_features = total_features_int

            if n_image_tokens > 0 and n_image_tokens != expected_features:
                raise ValueError(
                    f"Image token count mismatch! "
                    f"input_ids has {n_image_tokens} image placeholder tokens, "
                    f"but vision encoder produces {expected_features} features. "
                    f"This usually means data config's vision_feature_height doesn't match the model. "
                    f"Expected vision_feature_height={int(expected_features**0.5)} based on model config "
                    f"(image_size={self.config.vision_config.image_size}, "
                    f"patch_size={self.config.vision_config.patch_size})."
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
        head_size = Axis("head_size", tc.hidden_dim // tc.num_heads)
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

    Uses fixed-shape processing for JIT compatibility:
    - pixel_values: (TOTAL_PATCHES, C, H, W) - padded to fixed size
    - grid_mask: (TOTAL_PATCHES,) - True for valid patches, False for padding

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
    _input_ids: NamedArray | None = None
    _pixel_values: NamedArray | None = None
    _grid_mask: NamedArray | None = None
    _unpad_indices: NamedArray | None = None
    _num_unpadded_features: int | None = eqx.field(static=True, default=None)

    # Cached embeddings and position IDs (computed lazily during prefill)
    _cached_embeds: NamedArray | None = None
    _cached_pos_ids: NamedArray | None = None

    # Position offset for decode phase (static since it's computed from static fields)
    # This is the difference between the padded sequence length and the actual valid length
    _position_offset: int = eqx.field(static=True, default=0)

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
    ) -> "_LlavaInferenceWrapper":
        """Set the image data for the current request.

        This must be called before generating with InferenceEngine.

        Args:
            input_ids: Input token IDs with shape (batch, position)
            pixel_values: Fixed-shape pixel values (TOTAL_PATCHES, C, H, W)
            grid_mask: Boolean mask indicating valid patches (TOTAL_PATCHES,)
            unpad_indices: Pre-computed indices to reorder features to HF's unpadded order
            num_unpadded_features: Actual number of unpadded features (before padding)

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
            _position_offset=position_offset,
            _cached_embeds=None,
            _cached_pos_ids=None,
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
                key=None,
            )

        # Squeeze batch axis since InferenceEngine expects no batch axis
        if "batch" in [ax.name for ax in merged_embeds.axes]:
            merged_embeds = merged_embeds["batch", 0]
            position_ids = position_ids["batch", 0]

        return merged_embeds, position_ids

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
        """
        is_prefill = tokens.axis_size("position") > 1
        lm = self.model.language_model

        if is_prefill:
            # Use position IDs from _compute_embeddings for proper RoPE with padding
            embeds, computed_pos_ids = self._compute_embeddings()
            pos_ids = computed_pos_ids
        else:
            embeds = lm.embeddings.embed(tokens)
            # Adjust position IDs for decode phase to account for skipped padding
            # During prefill, we skip _position_offset padding tokens
            # So decode positions should be adjusted by the same amount
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

    def generate(self, requests: list[VLMRequest], step_callback=None):
        """Generate tokens for a batch of VLMRequests.

        This method:
        1. Extracts VLM data from the first request
        2. Sets the request data on the wrapper
        3. Calls the base engine's generate method

        Args:
            requests: List of VLMRequest objects
            step_callback: Optional callback for each decode iteration

        Returns:
            GenerationResult with tokens, logprobs, and total_generated
        """
        if not requests:
            raise ValueError("At least one request is required")

        # For now, we only support single-request generation for VLM
        # (because the wrapper stores a single set of pixel_values)
        if len(requests) > 1:
            raise NotImplementedError(
                "LlavaInferenceEngine currently only supports single-request generation. "
                "Multi-request batching for VLM is not yet implemented."
            )

        vlm_request = requests[0]

        # Set the VLM data on the wrapper
        # We need to update the model in the base engine
        self._base_engine.model = self._wrapper.set_request_data(
            input_ids=vlm_request.input_ids,
            pixel_values=vlm_request.pixel_values,
            grid_mask=vlm_request.grid_mask,
            unpad_indices=vlm_request.unpad_indices,
            num_unpadded_features=vlm_request.num_unpadded_features,
        )

        # Convert VLMRequest to standard Request for the base engine
        standard_requests = [
            Request(
                prompt_tokens=r.prompt_tokens,
                request_id=r.request_id,
                decode_params=r.decode_params,
                n_generations=r.n_generations,
            )
            for r in requests
        ]

        # Generate using the base engine
        return self._base_engine.generate(standard_requests, step_callback=step_callback)

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
