# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass, replace
from typing import Dict, Optional, Self, Tuple, Type

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis, NamedArray
from haliax._src.state_dict import default_eqx_module_from_state_dict
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.state_dict import ModuleWithStateDictSerialization, from_state_dict, with_prefix
from jaxtyping import PRNGKeyArray

from levanter.compat.hf_checkpoints import HfVisionCheckpointConverter, VisionModelWithHfSerializationMixin
from levanter.layers.attention import AttentionBackend, AttentionConfig, AttentionMask
from levanter.models.siglip import (
    DEFAULT_SIGLIP_INITIALIZER_RANGE,
    HF_SIGLIP_GELU_ACTIVATION,
    SiglipEncoderLayer,
    SiglipMultiheadAttentionPoolingHead,
    siglip_activation,
    siglip_hidden_states,
)
from levanter.models.vision_model import VisionEncoderConfig
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.logging import silence_transformer_nag

silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig  # noqa: E402
from transformers import Siglip2Config as HfSiglip2Config  # noqa: E402
from transformers import Siglip2VisionConfig as HfSiglip2VisionConfig  # noqa: E402


@dataclass
class Siglip2VisionModelOutput:
    """
    Output class for Siglip2 Vision Model, similar to HuggingFace's output format.

    Args:
        last_hidden_state: Final hidden states after post-layer normalization.
            Shape: (batch, num_patches, embed)
        hidden_states: Tuple of hidden states from each layer (including embeddings).
            Each element has shape: (batch, num_patches, embed)
        pooler_output: Attention-pooled image representation.
    """

    last_hidden_state: NamedArray
    hidden_states: Tuple[NamedArray, ...]
    pooler_output: Optional[NamedArray] = None


@VisionEncoderConfig.register_subclass("siglip2")
@dataclass(frozen=True)
class Siglip2VisionConfig(VisionEncoderConfig["Siglip2VisionModel"]):
    """
    Configuration class for Siglip2 Vision Encoder (marin version).

    This configuration follows the Levanter/marin patterns for model configs,
    supporting HuggingFace checkpoint conversion and serialization.

    Args:
        hidden_size: Dimensionality of the encoder layers and the pooler layer.
        intermediate_size: Dimensionality of the "intermediate" (i.e., feed-forward) layer.
        num_hidden_layers: Number of hidden layers in the Transformer encoder.
        num_attention_heads: Number of attention heads for each attention layer.
        num_channels: Number of channels in the input images.
        num_patches: Maximum number of patches in the image (with aspect ratio preservation).
        patch_size: The size (resolution) of each patch.
        hidden_act: The non-linear activation function.
        layer_norm_eps: The epsilon used by the layer normalization layers.
        attention_dropout: The dropout ratio for the attention probabilities.
        initializer_range: Standard deviation of the normal initializer for the pooling probe.
        gradient_checkpointing: Whether to use gradient checkpointing to save memory.
        use_flash_attention: Whether to use flash attention.
        attn_backend: Attention backend to use (if None, uses default).
        flash_attention_block_size: Block size for flash attention.
        inference: Whether to run in inference mode (disables dropout).
    """

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    num_patches: int = 256
    patch_size: int = 16
    hidden_act: ActivationFunctionEnum = ActivationFunctionEnum.gelu_new
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    initializer_range: float = DEFAULT_SIGLIP_INITIALIZER_RANGE
    gradient_checkpointing: bool = True
    vision_use_head: bool = True

    use_flash_attention: bool = False
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None
    inference: bool = True  # Whether to run in inference mode (disables dropout)

    reference_checkpoint: Optional[str] = None

    @property
    def model_type(self) -> Type:
        """Return the model class type."""
        return Siglip2VisionModel

    def hf_checkpoint_converter(
        self, ref_checkpoint: Optional[str] = None
    ) -> HfVisionCheckpointConverter["Siglip2VisionConfig"]:
        """Create HuggingFace checkpoint converter for this config."""
        return HfVisionCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=False,
            HfConfigClass=HfSiglip2VisionConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig) -> "Siglip2VisionConfig":
        """Convert from HuggingFace config to Levanter config."""
        if isinstance(hf_config, HfSiglip2Config):
            vision_config = hf_config.vision_config
            assert isinstance(vision_config, HfSiglip2VisionConfig)
            hf_config = vision_config
        assert isinstance(hf_config, HfSiglip2VisionConfig)
        activation_fn = siglip_activation(hf_config.hidden_act)

        return cls(
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_channels=hf_config.num_channels,
            num_patches=hf_config.num_patches,
            patch_size=hf_config.patch_size,
            hidden_act=activation_fn,
            layer_norm_eps=hf_config.layer_norm_eps,
            attention_dropout=hf_config.attention_dropout,
            initializer_range=getattr(hf_config, "initializer_range", DEFAULT_SIGLIP_INITIALIZER_RANGE),
            vision_use_head=getattr(hf_config, "vision_use_head", True),
        )

    def to_hf_config(self, config_overrides: Optional[Dict] = None) -> HfSiglip2VisionConfig:
        """Convert from Levanter config to HuggingFace config.

        Args:
            config_overrides: Optional config overrides
        """
        if config_overrides is None:
            config_overrides = {}

        # gelu_new in Levanter maps back to gelu_pytorch_tanh in HF (for Siglip2 compatibility)
        if isinstance(self.hidden_act, ActivationFunctionEnum):
            if self.hidden_act == ActivationFunctionEnum.gelu_new:
                hf_hidden_act = HF_SIGLIP_GELU_ACTIVATION
            else:
                hf_hidden_act = self.hidden_act.value
        else:
            hf_hidden_act = self.hidden_act

        # Calculate image_size from num_patches and patch_size
        # This is needed for compatibility with LlavaOnevision which expects image_size
        grid_size = int(self.num_patches**0.5)
        image_size = grid_size * self.patch_size

        config_dict = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_channels": self.num_channels,
            "num_patches": self.num_patches,
            "patch_size": self.patch_size,
            "hidden_act": hf_hidden_act,
            "layer_norm_eps": self.layer_norm_eps,
            "attention_dropout": self.attention_dropout,
            "initializer_range": self.initializer_range,
            "vision_use_head": self.vision_use_head,
            **config_overrides,
        }
        hf_config = HfSiglip2VisionConfig.from_dict(config_dict)

        hf_config.image_size = image_size

        return hf_config

    @property
    def Embed(self) -> Axis:
        """Embedding dimension axis."""
        return Axis(name="embed", size=self.hidden_size)

    @property
    def Mlp(self) -> Axis:
        """MLP intermediate dimension axis."""
        return Axis(name="mlp", size=self.intermediate_size)

    @property
    def Heads(self) -> Axis:
        """Number of attention heads axis."""
        return Axis(name="heads", size=self.num_attention_heads)

    @property
    def HeadSize(self) -> Axis:
        """Size of each attention head axis."""
        return Axis(name="head_size", size=self.hidden_size // self.num_attention_heads)

    @property
    def Layers(self) -> Axis:
        """Number of transformer layers axis."""
        return Axis(name="layers", size=self.num_hidden_layers)

    @property
    def Channels(self) -> Axis:
        """Number of image channels axis."""
        return Axis(name="channels", size=self.num_channels)

    @property
    def PatchSize(self) -> Axis:
        """Patch size axis."""
        return Axis(name="patch_size", size=self.patch_size)

    @property
    def NumPatches(self) -> Axis:
        """Maximum number of patches axis."""
        return Axis(name="num_patches", size=self.num_patches)

    def attention_config(self) -> AttentionConfig:
        """Build the standard Levanter attention configuration for SigLIP2."""
        backend = self.attn_backend if self.use_flash_attention else AttentionBackend.VANILLA
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_attention_heads,
            use_bias=True,
            use_output_bias=True,
            attn_backend=backend,
            flash_attention_block_size=self.flash_attention_block_size,
        )


class Siglip2VisionEmbeddings(eqx.Module):
    """
    Vision embeddings for Siglip2.

    Converts patchified images to embeddings and adds position embeddings.
    Unlike traditional ViT, Siglip2 uses flexible aspect ratio handling.
    """

    config: Siglip2VisionConfig = eqx.field(static=True)
    patch_embedding: hnn.Linear
    position_embedding: hnn.Embedding

    @staticmethod
    def init(config: Siglip2VisionConfig, *, key: PRNGKeyArray) -> "Siglip2VisionEmbeddings":
        """
        Initialize Siglip2VisionEmbeddings.

        Args:
            config: Siglip2VisionConfig
            key: PRNGKey for initialization

        Returns:
            Initialized Siglip2VisionEmbeddings module
        """
        k_patch, k_pos = maybe_rng_split(key, 2)

        patch_input_dim = config.num_channels * config.patch_size * config.patch_size
        PatchInput = Axis(name="patch_input", size=patch_input_dim)

        patch_embedding = hnn.Linear.init(
            In=PatchInput,
            Out=config.Embed,
            key=k_patch,
            use_bias=True,
            out_first=True,
        )

        position_embedding = hnn.Embedding.init(
            config.NumPatches,
            config.Embed,
            key=k_pos,
        )

        return Siglip2VisionEmbeddings(config, patch_embedding, position_embedding)

    @named_call
    def __call__(
        self, pixel_values: NamedArray, spatial_shapes: np.ndarray | None = None, *, key: PRNGKeyArray | None = None
    ) -> NamedArray:
        """
        Forward pass through vision embeddings.

        Args:
            pixel_values: Patchified pixel values with shape (..., num_patches, patch_input_dim)
                where patch_input_dim = num_channels * patch_size * patch_size
            spatial_shapes: Optional array of shape (batch, 2) containing [height, width] in patches
                for each image. If provided, position embeddings will be interpolated to match.
            key: Optional PRNGKey (unused, kept for API compatibility)

        Returns:
            Embeddings with position information added
        """
        patch_embeds = self.patch_embedding(pixel_values)

        num_patches_axis = pixel_values.resolve_axis("num_patches")

        if spatial_shapes is not None:
            grid_size = int(self.config.NumPatches.size**0.5)
            positions = self.position_embedding(hax.arange(self.config.NumPatches)).array
            grid = positions.reshape(grid_size, grid_size, -1)
            batch_axes = tuple(ax for ax in patch_embeds.axes if ax not in (num_patches_axis, self.config.Embed))
            shapes = jnp.asarray(spatial_shapes).reshape(-1, 2)

            def resize_one(shape):
                height, width = shape
                indices = jnp.arange(num_patches_axis.size)
                # HF fills padding with the first resized positional embedding.
                indices = jnp.where(indices < height * width, indices, 0)
                source = jnp.arange(grid_size, dtype=jnp.float32)

                def weights(coordinates, target_size):
                    # Half-pixel centers with a widened triangle kernel match HF antialiased bilinear resize.
                    scale = grid_size / target_size
                    centers = (coordinates + 0.5) * scale - 0.5
                    support = jnp.maximum(scale, 1.0)
                    kernel = jnp.maximum(1.0 - jnp.abs(source[None, :] - centers[:, None]) / support, 0.0)
                    return kernel / jnp.sum(kernel, axis=-1, keepdims=True)

                y_weights = weights(indices // width, height)
                x_weights = weights(indices % width, width)
                return jnp.einsum("ph,hwe,pw->pe", y_weights, grid.astype(jnp.float32), x_weights).astype(grid.dtype)

            resized = jax.vmap(resize_one)(shapes)
            resized = resized.reshape(*(ax.size for ax in batch_axes), num_patches_axis.size, self.config.Embed.size)
            pos_embeds = hax.named(resized, (*batch_axes, num_patches_axis, self.config.Embed))
        else:
            position_ids = hax.arange(num_patches_axis)
            pos_embeds = self.position_embedding(position_ids)

        embeddings = patch_embeds + pos_embeds

        return embeddings


class Siglip2VisionTransformer(ModuleWithStateDictSerialization):
    """
    Siglip2 Vision Transformer.

    Complete vision encoder consisting of:
    - Vision embeddings (patch + position)
    - Stack of encoder layers
    - Post-layer normalization
    """

    config: Siglip2VisionConfig = eqx.field(static=True)
    embeddings: Siglip2VisionEmbeddings
    layers: Stacked[SiglipEncoderLayer]
    post_layernorm: hnn.LayerNorm

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        """Map Levanter field names to HuggingFace state dict keys."""
        return {"layers": "encoder.layers"}  # HF uses encoder.layers instead of layers

    @staticmethod
    def init(config: Siglip2VisionConfig, *, key: PRNGKeyArray) -> "Siglip2VisionTransformer":
        """
        Initialize Siglip2VisionTransformer.

        Args:
            config: Siglip2VisionConfig
            key: PRNGKey for initialization

        Returns:
            Initialized Siglip2VisionTransformer module
        """
        k_embed, k_layers = maybe_rng_split(key, 2)

        embeddings = Siglip2VisionEmbeddings.init(config, key=k_embed)

        layers = Stacked.init(
            config.Layers,
            SiglipEncoderLayer,
            gradient_checkpointing=config.gradient_checkpointing,
        )(config, key=shaped_rng_split(k_layers, config.num_hidden_layers))

        post_layernorm = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_eps, use_bias=True)

        return Siglip2VisionTransformer(config, embeddings, layers, post_layernorm)

    @named_call
    def __call__(
        self,
        pixel_values: NamedArray,
        mask: Optional[AttentionMask] = None,
        spatial_shapes: np.ndarray | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Siglip2VisionModelOutput:
        """
        Forward pass through vision transformer.

        Args:
            pixel_values: Patchified pixel values with shape (..., num_patches, patch_input_dim)
            mask: Optional attention mask
            spatial_shapes: Optional array of shape (batch, 2) containing [height, width] in patches
            key: PRNGKey for dropout

        Returns:
            Siglip2VisionModelOutput containing:
                - last_hidden_state: Final encoded representations after post_layernorm
                - hidden_states: Embeddings followed by every encoder layer output.
        """
        k_embed, k_layers = maybe_rng_split(key, 2)

        hidden_states = self.embeddings(pixel_values, spatial_shapes=spatial_shapes, key=k_embed)

        hidden_states, result_hidden_states = siglip_hidden_states(
            self.layers,
            hidden_states,
            mask,
            key=k_layers,
        )

        last_hidden_state = self.post_layernorm(hidden_states)

        return Siglip2VisionModelOutput(last_hidden_state=last_hidden_state, hidden_states=result_hidden_states)


class Siglip2VisionModel(ModuleWithStateDictSerialization, VisionModelWithHfSerializationMixin[Siglip2VisionConfig]):
    """
    Siglip2 Vision Model with HuggingFace compatibility.

    This is a wrapper around Siglip2VisionTransformer that implements
    vision checkpoint conversion.
    """

    vision_model: Siglip2VisionTransformer
    head: Optional[SiglipMultiheadAttentionPoolingHead]

    @property
    def config(self) -> Siglip2VisionConfig:
        return self.vision_model.config

    def get_hf_config(self) -> HfConfig:
        """Return the Hugging Face vision configuration."""
        return self.config.to_hf_config()

    @classmethod
    def init(cls, config: Siglip2VisionConfig, *, key: PRNGKeyArray) -> "Siglip2VisionModel":
        """
        Initialize Siglip2VisionModel.

        Args:
            config: Siglip2VisionConfig
            key: PRNGKey for initialization

        Returns:
            Initialized Siglip2VisionModel
        """
        key_vision, key_head = maybe_rng_split(key, 2)
        vision_model = Siglip2VisionTransformer.init(config, key=key_vision)
        head = SiglipMultiheadAttentionPoolingHead.init(config, key=key_head) if config.vision_use_head else None
        return cls(vision_model=vision_model, head=head)

    @named_call
    def __call__(
        self,
        pixel_values: NamedArray,
        mask: Optional[AttentionMask] = None,
        spatial_shapes: np.ndarray | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Siglip2VisionModelOutput:
        """
        Forward pass through vision model.

        Args:
            pixel_values: Patchified pixel values with shape (..., num_patches, patch_input_dim)
            mask: Optional attention mask
            spatial_shapes: Optional array of shape (batch, 2) containing [height, width] in patches
            key: PRNGKey for dropout

        Returns:
            Siglip2VisionModelOutput containing encoder and pooled features.
        """
        key_vision, key_head = maybe_rng_split(key, 2)
        output = self.vision_model(pixel_values, mask=mask, spatial_shapes=spatial_shapes, key=key_vision)
        pooler_output = self.head(output.last_hidden_state, mask=mask, key=key_head) if self.head is not None else None
        return replace(output, pooler_output=pooler_output)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        """Map Levanter field names to HuggingFace state dict keys."""
        return {}  # Keep vision_model prefix as-is (matches HF structure)

    def from_state_dict(self, state_dict: Dict[str, jnp.ndarray], prefix: Optional[str] = None) -> Self:
        """Load from state dict."""

        # Standalone HF vision checkpoints omit the composite model's vision_model prefix.
        if with_prefix(prefix, "embeddings.patch_embedding.weight") in state_dict:
            vision_model = default_eqx_module_from_state_dict(self.vision_model, state_dict, prefix)
            head = (
                from_state_dict(self.head, state_dict, with_prefix(prefix, "head")) if self.head is not None else None
            )
            return replace(self, vision_model=vision_model, head=head)
        return default_eqx_module_from_state_dict(self, state_dict, prefix)
