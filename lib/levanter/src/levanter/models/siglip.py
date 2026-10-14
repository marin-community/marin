# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0


from haliax._src.state_dict import default_eqx_module_from_state_dict


from jaxtyping import PRNGKeyArray
from dataclasses import dataclass, replace
from typing import Self, Dict, Optional, Tuple, Type
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.logging import silence_transformer_nag
from levanter.models.vision_model import VisionEncoderConfig
from levanter.models.vision_transformer import vision_activation, vision_hidden_states, VisionEncoderLayer


silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig  # noqa: E402
from transformers import SiglipConfig as HfSiglipConfig  # noqa: E402
from transformers import SiglipVisionConfig as HfSiglipVisionConfig  # noqa: E402

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import haliax as hax  # noqa: E402
import haliax.nn as hnn  # noqa: E402
from haliax import Axis, NamedArray  # noqa: E402
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split  # noqa: E402
from haliax.nn.scan import Stacked  # noqa: E402
from haliax.state_dict import ModuleWithStateDictSerialization, with_prefix  # noqa: E402

from levanter.compat.hf_checkpoints import HFCheckpointConverter, ModelWithHfSerializationMixin  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402


@dataclass
class SiglipVisionModelOutput:
    """
    Output class for SigLIP Vision Model, similar to HuggingFace's output format.

    Args:
        last_hidden_state: Final hidden states after post-layer normalization.
            Shape: (batch, num_patches, embed)
        hidden_states: Tuple of hidden states from each layer (including embeddings).
            Each element has shape: (batch, num_patches, embed)
            Contains all layers when requested, otherwise only the final encoder output.
    """

    last_hidden_state: NamedArray
    hidden_states: Optional[Tuple[NamedArray, ...]] = None


@VisionEncoderConfig.register_subclass("siglip")
@dataclass(frozen=True)
class SiglipVisionConfig(VisionEncoderConfig["SiglipVisionModel"]):
    """
    Configuration class for SigLIP Vision Encoder (standard version, not Siglip2).

    This configuration follows the Levanter patterns for model configs,
    supporting HuggingFace checkpoint conversion and serialization.

    Based on google/siglip-base-patch16-224 architecture.

    Args:
        hidden_size: Dimensionality of the encoder layers and the pooler layer.
        intermediate_size: Dimensionality of the "intermediate" (i.e., feed-forward) layer.
        num_hidden_layers: Number of hidden layers in the Transformer encoder.
        num_attention_heads: Number of attention heads for each attention layer.
        num_channels: Number of channels in the input images.
        image_size: The size (resolution) of each image.
        patch_size: The size (resolution) of each patch.
        hidden_act: The non-linear activation function.
        layer_norm_eps: The epsilon used by the layer normalization layers.
        attention_dropout: The dropout ratio for the attention probabilities.
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
    image_size: int = 224
    patch_size: int = 16
    hidden_act: ActivationFunctionEnum = ActivationFunctionEnum.gelu_new
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    gradient_checkpointing: bool = True

    # Attention-related config
    use_flash_attention: bool = True
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None
    inference: bool = True  # Whether to run in inference mode (disables dropout)

    # Reference checkpoint for loading pretrained models
    reference_checkpoint: Optional[str] = None

    def hf_checkpoint_converter(
        self, ref_checkpoint: Optional[str] = None
    ) -> HFCheckpointConverter["SiglipVisionConfig"]:  # type: ignore
        """Create HuggingFace checkpoint converter for this config."""
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=False,
            HfConfigClass=HfSiglipVisionConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig) -> "SiglipVisionConfig":
        """Convert from HuggingFace config to Levanter config.

        Handles both:
        - SiglipVisionConfig directly (has hidden_act attribute)
        - SiglipConfig (has vision_config nested inside)
        """
        # If this is a full SiglipConfig, extract the vision_config
        if isinstance(hf_config, HfSiglipConfig):
            vision_config = hf_config.vision_config
            assert isinstance(vision_config, HfSiglipVisionConfig)
            hf_config = vision_config
        assert isinstance(hf_config, HfSiglipVisionConfig)

        activation_fn = vision_activation(hf_config.hidden_act)

        return cls(
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_channels=hf_config.num_channels,
            image_size=hf_config.image_size,
            patch_size=hf_config.patch_size,
            hidden_act=activation_fn,
            layer_norm_eps=hf_config.layer_norm_eps,
            attention_dropout=hf_config.attention_dropout,
        )

    def to_hf_config(self, vocab_size: int = 1, config_overrides: Optional[Dict] = None) -> HfSiglipVisionConfig:
        """Convert from Levanter config to HuggingFace config.

        Args:
            vocab_size: Vocabulary size (unused for vision-only models, but required by interface)
            config_overrides: Optional config overrides
        """
        # vocab_size is not used for vision-only models, but required by the interface
        if config_overrides is None:
            config_overrides = {}

        # Map activation function back to HF format
        # gelu_new in Levanter maps back to gelu_pytorch_tanh in HF (for SigLIP compatibility)
        if isinstance(self.hidden_act, ActivationFunctionEnum):
            if self.hidden_act == ActivationFunctionEnum.gelu_new:
                hf_hidden_act = "gelu_pytorch_tanh"
            else:
                hf_hidden_act = self.hidden_act.value
        else:
            hf_hidden_act = self.hidden_act

        # Build config dict with defaults from self
        config_dict = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_channels": self.num_channels,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "hidden_act": hf_hidden_act,
            "layer_norm_eps": self.layer_norm_eps,
            "attention_dropout": self.attention_dropout,
        }

        config_dict.update(config_overrides)

        hf_config = HfSiglipVisionConfig(**config_dict)

        return hf_config

    @property
    def model_type(self) -> Type["SiglipVisionModel"]:
        """Return the corresponding model class."""
        return SiglipVisionModel

    # Axis definitions following Levanter patterns
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
    def ImageSize(self) -> Axis:
        """Image size axis."""
        return Axis(name="image_size", size=self.image_size)

    @property
    def PatchSize(self) -> Axis:
        """Patch size axis."""
        return Axis(name="patch_size", size=self.patch_size)

    @property
    def NumPatches(self) -> Axis:
        """Number of patches axis (calculated from image_size and patch_size)."""
        num_patches = (self.image_size // self.patch_size) ** 2
        return Axis(name="num_patches", size=num_patches)


class SiglipVisionEmbeddings(eqx.Module):
    """
    Vision embeddings for SigLIP.

    Converts images to patches using Conv2d and adds learnable position embeddings.
    Unlike Siglip2 which uses patchified input, this module expects full images.
    """

    config: SiglipVisionConfig = eqx.field(static=True)
    patch_embedding: hnn.Conv  # Conv2d for patch embedding
    position_embedding: hnn.Embedding

    @staticmethod
    def init(config: SiglipVisionConfig, *, key: PRNGKeyArray) -> "SiglipVisionEmbeddings":
        """
        Initialize SiglipVisionEmbeddings.

        Args:
            config: SiglipVisionConfig
            key: PRNGKey for initialization

        Returns:
            Initialized SiglipVisionEmbeddings module
        """
        k_patch, k_pos = maybe_rng_split(key, 2)

        # Patch embedding using Conv2d
        # Input: (batch, channels, height, width)
        # Output: (batch, embed_dim, num_patches_h, num_patches_w)
        In_Channels = config.Channels
        Out_Features = config.Embed
        patch_size = config.patch_size

        # Define spatial dimensions for the input image
        Height = Axis("height", config.image_size)
        Width = Axis("width", config.image_size)

        patch_embedding = hnn.Conv.init(
            Spatial=(Height, Width),
            In=In_Channels,
            Out=Out_Features,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            key=k_patch,
            use_bias=True,
        )

        # Position embedding: learnable embeddings for each patch position
        # For standard SigLIP, this is (num_patches,) where num_patches = (image_size // patch_size)^2
        position_embedding = hnn.Embedding.init(
            config.NumPatches,
            config.Embed,
            key=k_pos,
        )

        return SiglipVisionEmbeddings(config, patch_embedding, position_embedding)

    @named_call
    def __call__(self, pixel_values: NamedArray, *, key: PRNGKeyArray | None = None) -> NamedArray:
        """
        Forward pass through vision embeddings.

        Args:
            pixel_values: Input images with shape (batch, channels, height, width)
            key: Optional PRNGKey

        Returns:
            Embeddings with position information added, shape (batch, num_patches, embed)
        """
        # Apply patch embeddings using Conv2d
        # Input: (batch, channels, height, width)
        # Output: (batch, embed, h_patches, w_patches)
        patch_embeds = self.patch_embedding(pixel_values, key=key)

        # Flatten spatial dimensions (h_patches, w_patches) into num_patches
        # Conv output has spatial axes named after input spatial axes (height, width)
        # with sizes divided by stride (patch_size)
        embed_axis = self.config.Embed
        spatial_axes = [ax for ax in patch_embeds.axes if ax.name in ("height", "width")]

        # Flatten spatial dims into num_patches
        num_patches_total = 1
        for ax in spatial_axes:
            num_patches_total *= ax.size
        NumPatchesActual = Axis("num_patches", num_patches_total)

        patch_embeds = hax.flatten_axes(patch_embeds, spatial_axes, NumPatchesActual)

        # Rearrange to put num_patches before embed: (batch, embed, num_patches) -> (batch, num_patches, embed)
        # Get all axes except embed and num_patches (i.e., batch if present), then num_patches, then embed
        other_axes = [ax for ax in patch_embeds.axes if ax not in (embed_axis, NumPatchesActual)]
        patch_embeds = patch_embeds.rearrange((*other_axes, NumPatchesActual, embed_axis))

        # Add position embeddings
        position_ids = hax.arange(NumPatchesActual)
        pos_embeds = self.position_embedding(position_ids)

        return patch_embeds + pos_embeds


class SiglipVisionTransformer(ModuleWithStateDictSerialization):
    """
    SigLIP Vision Transformer.

    Complete vision encoder consisting of:
    - Vision embeddings (patch + position)
    - Stack of encoder layers
    - Post-layer normalization
    """

    config: SiglipVisionConfig = eqx.field(static=True)
    embeddings: SiglipVisionEmbeddings
    layers: Stacked[VisionEncoderLayer]
    post_layernorm: hnn.LayerNorm

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        """Map Levanter field names to HuggingFace state dict keys."""
        return {"layers": "encoder.layers"}  # HF uses encoder.layers instead of layers

    @staticmethod
    def init(config: SiglipVisionConfig, *, key: PRNGKeyArray) -> "SiglipVisionTransformer":
        """
        Initialize SiglipVisionTransformer.

        Args:
            config: SiglipVisionConfig
            key: PRNGKey for initialization

        Returns:
            Initialized SiglipVisionTransformer module
        """
        k_embed, k_layers = maybe_rng_split(key, 2)

        embeddings = SiglipVisionEmbeddings.init(config, key=k_embed)

        # Initialize stacked encoder layers
        layers = Stacked.init(
            config.Layers,
            VisionEncoderLayer,
            gradient_checkpointing=config.gradient_checkpointing,
        )(config, key=shaped_rng_split(k_layers, config.num_hidden_layers))

        # Post-encoder layer norm
        post_layernorm = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_eps, use_bias=True)

        return SiglipVisionTransformer(config, embeddings, layers, post_layernorm)

    @named_call
    def __call__(
        self,
        pixel_values: NamedArray,
        mask: Optional[AttentionMask] = None,
        output_hidden_states: bool = False,
        *,
        key: PRNGKeyArray | None = None,
    ) -> SiglipVisionModelOutput:
        """
        Forward pass through vision transformer.

        Args:
            pixel_values: Input images with shape (batch, channels, height, width)
            mask: Optional attention mask
            output_hidden_states: Whether to return hidden states from all layers.
                If True, returns all layer hidden states.
                If False, only returns the last hidden state (more efficient).
            key: PRNGKey for dropout

        Returns:
            SiglipVisionModelOutput containing:
                - last_hidden_state: Final encoded representations after post_layernorm (batch, num_patches, embed)
                - hidden_states: Tuple of hidden states. When output_hidden_states=True, contains all layers.
                    When output_hidden_states=False, contains only the last layer output (before post_layernorm)
                    to support vision_feature_layer=-1 without collecting all intermediate states.
        """
        k_embed, k_layers = maybe_rng_split(key, 2)

        # Get embeddings
        hidden_states = self.embeddings(pixel_values, key=k_embed)

        hidden_states, result_hidden_states = vision_hidden_states(
            self.layers,
            hidden_states,
            mask,
            output_hidden_states,
            key=k_layers,
        )

        last_hidden_state = self.post_layernorm(hidden_states)

        return SiglipVisionModelOutput(last_hidden_state=last_hidden_state, hidden_states=result_hidden_states)


class SiglipVisionModel(ModuleWithStateDictSerialization, ModelWithHfSerializationMixin[SiglipVisionConfig]):
    """
    SigLIP Vision Model with HuggingFace compatibility.

    This is a wrapper around SiglipVisionTransformer that implements
    the ModelWithHfSerializationMixin interface for checkpoint conversion.
    """

    vision_model: SiglipVisionTransformer

    @property
    def config(self) -> SiglipVisionConfig:
        return self.vision_model.config

    @property
    def Vocab(self) -> Axis:
        # Vision models don't have a vocab, but ModelWithHfSerializationMixin requires it
        # We use a dummy axis for compatibility
        return Axis(name="vocab", size=1)

    def get_hf_config(self) -> HfConfig:
        """Return the Hugging Face vision configuration."""
        return self.config.to_hf_config()

    @classmethod
    def init(cls, Vocab: Axis, config: SiglipVisionConfig, *, key: PRNGKeyArray) -> "SiglipVisionModel":
        """
        Initialize SiglipVisionModel.

        Args:
            Vocab: Dummy vocab axis (not used for vision models, but required by interface)
            config: SiglipVisionConfig
            key: PRNGKey for initialization

        Returns:
            Initialized SiglipVisionModel
        """
        vision_model = SiglipVisionTransformer.init(config, key=key)
        return cls(vision_model=vision_model)

    @named_call
    def __call__(
        self,
        pixel_values: NamedArray,
        mask: Optional[AttentionMask] = None,
        output_hidden_states: bool = False,
        *,
        key: PRNGKeyArray | None = None,
    ) -> SiglipVisionModelOutput:
        """
        Forward pass through vision model.

        Args:
            pixel_values: Input images with shape (batch, channels, height, width)
            mask: Optional attention mask
            output_hidden_states: Whether to return hidden states from all layers
            key: PRNGKey for dropout

        Returns:
            SiglipVisionModelOutput containing last_hidden_state and optionally hidden_states
        """
        return self.vision_model(pixel_values, mask=mask, output_hidden_states=output_hidden_states, key=key)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        """Map Levanter field names to HuggingFace state dict keys."""
        return {}  # Keep vision_model prefix as-is (matches HF structure)

    def from_state_dict(self, state_dict: Dict[str, jnp.ndarray], prefix: Optional[str] = None) -> Self:
        """Load from state dict."""

        # Standalone HF vision checkpoints omit the composite model's vision_model prefix.
        if with_prefix(prefix, "embeddings.patch_embedding.weight") in state_dict:
            vision_model = default_eqx_module_from_state_dict(self.vision_model, state_dict, prefix)
            return replace(self, vision_model=vision_model)
        return default_eqx_module_from_state_dict(self, state_dict, prefix)
