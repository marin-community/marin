# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call

from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmConfig
from levanter.models.qwen import QwenConfig, QwenLMHeadModel
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
    text_config: QwenConfig
    vision_encoder_type: str = "siglip"  # "siglip" or "siglip2"

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

    # Reference checkpoint for loading pretrained models
    reference_checkpoint: Optional[str] = None
    tokenizer: Optional[str] = None

    def __post_init__(self):
        if self.vision_encoder_type not in ["siglip", "siglip2"]:
            raise ValueError(f"vision_encoder_type must be 'siglip' or 'siglip2', got {self.vision_encoder_type}")

        if self.vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                f"vision_feature_select_strategy must be 'default' or 'full', got {self.vision_feature_select_strategy}"
            )

        # Set default image_grid_pinpoints if not provided
        if self.image_grid_pinpoints is None:
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
    def model_type(self) -> Type["LlavaOnevisionModel"]:
        """Return the model class type."""
        return LlavaOnevisionModel

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

        # Ensure no_bias attribute exists (Qwen2 default is True, meaning use_bias=False)
        if not hasattr(hf_config.text_config, 'no_bias'):
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

    # Axis definitions
    @property
    def VisionEmbed(self) -> Axis:
        """Vision embedding dimension (renamed to avoid collision with text embed)."""
        return Axis(name="vision_embed", size=self.vision_config.hidden_size)

    @property
    def TextEmbed(self) -> Axis:
        """Text embedding dimension (same as Embed for compatibility)."""
        return self.text_config.Embed

    @property
    def Embed(self) -> Axis:
        """Text embedding dimension."""
        return self.text_config.Embed

    @property
    def Pos(self) -> Axis:
        """Maximum position axis."""
        return self.text_config.max_Pos

    @property
    def max_Pos(self) -> Axis:
        """Maximum position axis."""
        return self.text_config.max_Pos

    @property
    def KeyPos(self) -> Axis:
        """Key position axis."""
        return self.text_config.KeyPos


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

        # Create axis for vision embeddings with unique name to avoid collision
        VisionEmbed = Axis(name="vision_embed", size=config.vision_config.hidden_size)
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
            vision_tower = Siglip2VisionModel.init(
                Vocab=Vocab, config=config.vision_config, key=k_vision
            )
        elif config.vision_encoder_type == "siglip":
            vision_tower = SiglipVisionModel.init(
                Vocab=Vocab, config=config.vision_config, key=k_vision
            )
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
        return jax.vmap(lambda arr, idx: arr[idx])(arrays, indices)

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
        # selected_image_feature shape: (vision_batch, num_patches, embed)
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
        inputs_embeds: Optional[NamedArray] = None,
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
            inputs_embeds: Optional pre-computed embeddings (batch, seq_len, embed)
            key: Optional PRNGKey

        Returns:
            Tuple of (activations, lm_head) for use with fused_cross_entropy_loss.
        """
        k_vision, k_lm = maybe_rng_split(key, 2)

        # Merge text embeddings with image features and compute position IDs
        inputs_embeds, position_ids = self._merge_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            grid_mask=grid_mask,
            unpad_indices=unpad_indices,
            key=k_vision,
        )

        # Forward through language model with merged embeddings
        causal_mask = AttentionMask.causal()

        activations = self.language_model.transformer(
            inputs_embeds, attn_mask=causal_mask, pos_ids=position_ids, key=k_lm
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
        inputs_embeds: Optional[NamedArray] = None,
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
            inputs_embeds: Optional pre-computed embeddings (batch, seq_len, embed)
            key: Optional PRNGKey

        Returns:
            Logits with shape (batch, seq_len, vocab)
        """
        activations, lm_head = self.forward_with_activations(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_mask=grid_mask,
            unpad_indices=unpad_indices,
            inputs_embeds=inputs_embeds,
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
        *,
        key=None,
    ) -> Tuple[NamedArray, NamedArray]:
        """
        Merge text embeddings with projected image features and compute position IDs.

        This function:
        1. Gets image features via get_image_features() (fixed-shape with mask)
        2. Flattens image features to (batch, TOTAL_PATCHES * features_per_patch, embed)
        3. If unpad_indices provided, reorders features to HF's unpadded spatial order
        4. Merges image features into text embeddings at placeholder positions
        5. Computes compact position IDs that skip padding using cumsum

        Args:
            input_ids: Text token IDs (batch, seq_len) - used to derive text validity mask
            inputs_embeds: Optional pre-computed text embeddings
            pixel_values: Fixed-shape patches (batch, TOTAL_PATCHES, C, H, W)
            grid_mask: Boolean mask for valid patches (batch, TOTAL_PATCHES)
            unpad_indices: Pre-computed indices to reorder features to HF's unpadded order
                          (batch, num_image_tokens) - if None, uses sequential ordering
            key: Optional PRNGKey

        Returns:
            Tuple of:
            - merged_embeds: (batch, seq_len, embed) with image features at placeholders
            - position_ids: (batch, seq_len) compact position IDs skipping padding
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
            return inputs_embeds, position_ids

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
        # unpad_indices: (batch, num_image_tokens) where num_image_tokens is the unpadded count
        # unpad_indices[i] = Levanter index for HF position i
        if unpad_indices is not None:
            # Get the number of unpadded image tokens from unpad_indices shape
            num_unpadded_tokens = unpad_indices.axis_size("num_image_tokens")

            # Gather features in HF's unpadded order
            image_features_reordered = self._batch_gather(image_features_flat.array, unpad_indices.array)
            # Now image_features_reordered: (batch, num_unpadded_tokens, embed) in HF order
            UnpaddedTokens = Axis("image_tokens", num_unpadded_tokens)
            image_features_flat = hax.named(image_features_reordered, (batch_ax, UnpaddedTokens, embed_ax))
            # Update total_image_tokens to reflect the unpadded count
            total_image_tokens = num_unpadded_tokens

        # Get placeholder mask: where image tokens should be inserted
        special_image_mask = self.get_placeholder_mask(input_ids)

        # Compute gather indices: for each placeholder, which image token to gather
        def compute_indices(mask):
            return (jnp.cumsum(mask.astype(jnp.int32)) - 1) % total_image_tokens

        all_indices = jax.vmap(compute_indices)(special_image_mask.array)

        # Gather image features and merge with text embeddings
        gathered = self._batch_gather(image_features_flat.array, all_indices)
        merged = jnp.where(special_image_mask.array[:, :, None], gathered, inputs_embeds.array)
        merged_embeds = hax.named(merged, inputs_embeds.axes)

        # === POSITION ID COMPUTATION ===
        # Combined validity mask: valid text OR valid image at placeholder positions
        if unpad_indices is not None:
            # When unpad_indices is provided, all image tokens are valid (they're the unpadded ones)
            combined_mask = jnp.where(special_image_mask.array, 1, text_mask.array).astype(jnp.int32)
        else:
            # Need to check grid_mask validity for each placeholder position
            grid_mask_expanded = jnp.repeat(grid_mask.array, features_per_patch, axis=1)
            image_token_indices = jnp.cumsum(special_image_mask.array.astype(jnp.int32), axis=-1) - 1
            image_token_indices = jnp.clip(image_token_indices, 0, total_image_tokens - 1)
            image_validity = self._batch_gather(grid_mask_expanded, image_token_indices)
            combined_mask = jnp.where(special_image_mask.array, image_validity, text_mask.array).astype(jnp.int32)

        # Compute compact position IDs
        position_ids_array = self._compute_position_ids(combined_mask)

        Pos = Axis("position", seq_ax.size)
        position_ids = hax.named(position_ids_array, (batch_ax, Pos))

        return merged_embeds, position_ids

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
            embeds, pos_ids = self._merge_embeddings(
                input_ids=input_ids,
                inputs_embeds=None,
                pixel_values=pixel_values,
                grid_mask=grid_mask,
                unpad_indices=unpad_indices,
                key=k_vision,
            )

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

    # Cached embeddings and position IDs (computed lazily during prefill)
    _cached_embeds: NamedArray | None = None
    _cached_pos_ids: NamedArray | None = None

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
    ) -> "_LlavaInferenceWrapper":
        """Set the image data for the current request.

        This must be called before generating with InferenceEngine.

        Args:
            input_ids: Input token IDs with shape (batch, position)
            pixel_values: Fixed-shape pixel values (TOTAL_PATCHES, C, H, W)
            grid_mask: Boolean mask indicating valid patches (TOTAL_PATCHES,)
            unpad_indices: Pre-computed indices to reorder features to HF's unpadded order

        Returns:
            A new wrapper with the request data set
        """
        # Create a new instance with updated request data
        return _LlavaInferenceWrapper(
            model=self.model,
            Vocab=self.Vocab,
            _text_config=self._text_config,
            _input_ids=input_ids,
            _pixel_values=pixel_values,
            _grid_mask=grid_mask,
            _unpad_indices=unpad_indices,
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
            merged_embeds, position_ids = self.model._merge_embeddings(
                input_ids=self._input_ids,
                inputs_embeds=None,
                pixel_values=self._pixel_values,
                grid_mask=self._grid_mask,
                unpad_indices=self._unpad_indices,
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
        """Decode using dynamically computed embeddings for prefill, language model for decode."""
        is_prefill = tokens.axis_size("position") > 1
        lm = self.model.language_model

        if is_prefill:
            # Use position IDs from _compute_embeddings for proper RoPE with padding
            embeds, computed_pos_ids = self._compute_embeddings()
            pos_ids = computed_pos_ids
        else:
            embeds = lm.embeddings.embed(tokens)

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
