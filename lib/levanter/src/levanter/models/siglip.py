# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.logging import silence_transformer_nag


silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig  # noqa: E402
from transformers import SiglipVisionConfig as HfSiglipVisionConfig  # noqa: E402

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import haliax as hax  # noqa: E402
import haliax.nn as hnn  # noqa: E402
from haliax import Axis, NamedArray  # noqa: E402
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split  # noqa: E402
from haliax.nn.scan import Stacked  # noqa: E402
from haliax.state_dict import ModuleWithStateDictSerialization  # noqa: E402

from levanter.compat.hf_checkpoints import HFCheckpointConverter, ModelWithHfSerializationMixin  # noqa: E402
from levanter.layers.attention import AttentionMask, dot_product_attention  # noqa: E402


@dataclass(frozen=True)
class SiglipVisionConfig:
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
        initializer_range: The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        gradient_checkpointing: Whether to use gradient checkpointing to save memory.
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
    initializer_range: float = 0.02
    gradient_checkpointing: bool = True

    # Reference checkpoint for loading pretrained models
    reference_checkpoint: Optional[str] = None

    def hf_checkpoint_converter(
        self, ref_checkpoint: Optional[str] = None
    ) -> HFCheckpointConverter["SiglipVisionConfig"]:  # type: ignore
        """Create HuggingFace checkpoint converter for this config."""
        # Vision-only models don't have a tokenizer, but HFCheckpointConverter requires one
        # Use gpt2 tokenizer as a placeholder since it's always available
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=False,
            tokenizer="gpt2",  # Dummy tokenizer for vision-only model
            HfConfigClass=HfSiglipVisionConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig) -> "SiglipVisionConfig":
        """Convert from HuggingFace config to Levanter config."""
        # Extract activation function, handle both string and enum
        hidden_act = hf_config.hidden_act
        if isinstance(hidden_act, str):
            # Map HF activation names to our enum
            # Note: gelu_pytorch_tanh in HF maps to gelu_new in Levanter (approximate GELU)
            if hidden_act == "gelu_pytorch_tanh":
                activation_fn = ActivationFunctionEnum.gelu_new
            elif hidden_act == "gelu":
                activation_fn = ActivationFunctionEnum.gelu
            elif hidden_act == "gelu_new":
                activation_fn = ActivationFunctionEnum.gelu_new
            elif hidden_act == "relu":
                activation_fn = ActivationFunctionEnum.relu
            elif hidden_act == "silu" or hidden_act == "swish":
                activation_fn = ActivationFunctionEnum.silu
            elif hidden_act == "quick_gelu":
                activation_fn = ActivationFunctionEnum.quick_gelu
            else:
                # Default to gelu_new for unknown activations
                activation_fn = ActivationFunctionEnum.gelu_new
        else:
            activation_fn = ActivationFunctionEnum.gelu_new

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

        # Apply overrides
        config_dict.update(config_overrides)

        hf_config = HfSiglipVisionConfig(**config_dict)

        return hf_config

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


# =====================
# SigLIP MLP
# =====================


class SiglipMLP(eqx.Module):
    """
    MLP module for SigLIP Vision Transformer.

    Implements a two-layer feedforward network with activation function in between.
    """

    fc1: hnn.Linear  # projection from Embed to Mlp (intermediate)
    fc2: hnn.Linear  # projection from Mlp to Embed
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(Embed: Axis, Mlp: Axis, activation_fn: ActivationFunctionEnum, *, key) -> "SiglipMLP":
        """
        Initialize SiglipMLP.

        Args:
            Embed: Embedding dimension axis
            Mlp: MLP intermediate dimension axis
            activation_fn: Activation function enum
            key: PRNGKey for initialization

        Returns:
            Initialized SiglipMLP module
        """
        k_fc1, k_fc2 = maybe_rng_split(key, 2)

        # In SigLIP, fc1 goes from hidden_size to intermediate_size
        fc1 = hnn.Linear.init(In=Embed, Out=Mlp, key=k_fc1, use_bias=True, out_first=True)
        # fc2 goes from intermediate_size back to hidden_size
        fc2 = hnn.Linear.init(In=Mlp, Out=Embed, key=k_fc2, use_bias=True, out_first=True)

        # Convert activation function enum to callable
        activation_fn_callable = (
            activation_fn.to_fn() if isinstance(activation_fn, ActivationFunctionEnum) else activation_fn
        )

        return SiglipMLP(fc1, fc2, activation_fn_callable)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        """
        Forward pass through MLP.

        Args:
            x: Input tensor with Embed axis
            key: Optional PRNGKey for dropout (not used in SigLIP)

        Returns:
            Output tensor with Embed axis
        """
        k1, k2 = maybe_rng_split(key, 2)
        x = self.fc1(x, key=k1)
        x = self.act(x)
        x = self.fc2(x, key=k2)
        return x


# =====================
# SigLIP Attention
# =====================


class SiglipAttention(eqx.Module):
    """
    Multi-headed attention module for SigLIP.

    Implements standard multi-head self-attention with separate Q, K, V projections
    and an output projection.
    """

    config: SiglipVisionConfig = eqx.field(static=True)
    q_proj: hnn.Linear  # Query projection from Embed to (Heads, HeadSize)
    k_proj: hnn.Linear  # Key projection from Embed to (Heads, HeadSize)
    v_proj: hnn.Linear  # Value projection from Embed to (Heads, HeadSize)
    out_proj: hnn.Linear  # Output projection from (Heads, HeadSize) to Embed

    @staticmethod
    def init(config: SiglipVisionConfig, *, key) -> "SiglipAttention":
        """
        Initialize SiglipAttention.

        Args:
            config: SiglipVisionConfig
            key: PRNGKey for initialization

        Returns:
            Initialized SiglipAttention module
        """
        k_q, k_k, k_v, k_out = maybe_rng_split(key, 4)

        Embed = config.Embed
        Heads = config.Heads
        HeadSize = config.HeadSize

        # Initialize projection layers
        # All projections use bias in SigLIP
        q_proj = hnn.Linear.init(In=Embed, Out=(Heads, HeadSize), key=k_q, use_bias=True, out_first=True)
        k_proj = hnn.Linear.init(In=Embed, Out=(Heads, HeadSize), key=k_k, use_bias=True, out_first=True)
        v_proj = hnn.Linear.init(In=Embed, Out=(Heads, HeadSize), key=k_v, use_bias=True, out_first=True)
        out_proj = hnn.Linear.init(In=(Heads, HeadSize), Out=Embed, key=k_out, use_bias=True, out_first=True)

        return SiglipAttention(config, q_proj, k_proj, v_proj, out_proj)

    @named_call
    def __call__(
        self,
        x: NamedArray,
        mask: Optional[AttentionMask] = None,
        *,
        key=None,
    ) -> NamedArray:
        """
        Forward pass through attention.

        Args:
            x: Input tensor with shape (..., position, embed)
            mask: Optional attention mask
            key: PRNGKey for dropout

        Returns:
            Output tensor with shape (..., position, embed)
        """
        k_q, k_k, k_v, k_out, k_drop = maybe_rng_split(key, 5)

        # Find the sequence axis (position or num_patches)
        embed_axis = self.config.Embed
        common_batch_axes = {"batch", "Batch"}
        sequence_axis = None

        # First, check if "position" axis already exists
        for axis in x.axes:
            if axis.name == "position":
                sequence_axis = axis
                break

        # If not, look for num_patches
        if sequence_axis is None:
            for axis in x.axes:
                if axis.name == "num_patches":
                    sequence_axis = axis
                    break

        # If still not found, find the first non-Embed, non-batch axis
        if sequence_axis is None:
            for axis in x.axes:
                if axis != embed_axis and axis.name not in common_batch_axes:
                    sequence_axis = axis
                    break

        if sequence_axis is None:
            raise ValueError(f"Could not find sequence axis in input {x.axes}")

        # Rename sequence axis to "position" for consistent processing
        original_seq_name = sequence_axis.name
        if original_seq_name != "position":
            x = x.rename({original_seq_name: "position"})

        # Project to Q, K, V
        # Shape: (..., position, embed) -> (..., position, heads, head_size)
        q = self.q_proj(x, key=k_q).rearrange((..., "heads", "position", "head_size"))
        k = self.k_proj(x, key=k_k).rearrange((..., "heads", "position", "head_size"))
        v = self.v_proj(x, key=k_v).rearrange((..., "heads", "position", "head_size"))

        # Rename k and v's position axis to avoid conflicts
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        # Compute attention
        # SigLIP uses standard scaled dot-product attention
        attn_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask=mask,
            inference=False,
            use_flash=self.config.gradient_checkpointing,
            dropout=self.config.attention_dropout,
            prng=k_drop,
        )

        # Project back to embedding dimension
        # Shape: (..., position, heads, head_size) -> (..., position, embed)
        attn_output = attn_output.astype(x.dtype)
        output = self.out_proj(attn_output, key=k_out)

        # Rename position axis back to original name if needed
        if original_seq_name != "position":
            output = output.rename({"position": original_seq_name})

        return output


# =====================
# SigLIP Encoder Layer
# =====================


class SiglipEncoderLayer(eqx.Module):
    """
    SigLIP Encoder Layer.

    Implements a transformer encoder layer with:
    - Pre-LayerNorm architecture
    - Self-attention with residual connection
    - MLP with residual connection
    """

    config: SiglipVisionConfig = eqx.field(static=True)
    layer_norm1: hnn.LayerNorm  # Pre-attention layer norm
    self_attn: SiglipAttention  # Self-attention module
    layer_norm2: hnn.LayerNorm  # Pre-MLP layer norm
    mlp: SiglipMLP  # MLP module

    @staticmethod
    def init(config: SiglipVisionConfig, *, key) -> "SiglipEncoderLayer":
        """
        Initialize SiglipEncoderLayer.

        Args:
            config: SiglipVisionConfig
            key: PRNGKey for initialization

        Returns:
            Initialized SiglipEncoderLayer module
        """
        k_attn, k_mlp = maybe_rng_split(key, 2)

        # Initialize layer norms (with bias in SigLIP)
        layer_norm1 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_eps, use_bias=True)
        layer_norm2 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_eps, use_bias=True)

        # Initialize attention and MLP
        self_attn = SiglipAttention.init(config, key=k_attn)
        mlp = SiglipMLP.init(config.Embed, config.Mlp, config.hidden_act, key=k_mlp)

        return SiglipEncoderLayer(config, layer_norm1, self_attn, layer_norm2, mlp)

    @named_call
    def __call__(
        self,
        x: NamedArray,
        mask: Optional[AttentionMask] = None,
        *,
        key=None,
    ) -> NamedArray:
        """
        Forward pass through encoder layer.

        Args:
            x: Input tensor with shape (..., position, embed)
            mask: Optional attention mask
            key: PRNGKey for dropout

        Returns:
            Output tensor with shape (..., position, embed)
        """
        k_attn, k_mlp = maybe_rng_split(key, 2)

        # Self-attention block with pre-norm and residual
        residual = x
        x_norm = self.layer_norm1(x)
        attn_output = self.self_attn(x_norm, mask=mask, key=k_attn)
        x = residual + attn_output

        # MLP block with pre-norm and residual
        residual = x
        x_norm = self.layer_norm2(x)
        mlp_output = self.mlp(x_norm, key=k_mlp)
        x = residual + mlp_output

        return x


# =====================
# SigLIP Vision Embeddings
# =====================


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
    def init(config: SiglipVisionConfig, *, key) -> "SiglipVisionEmbeddings":
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
    def __call__(self, pixel_values: NamedArray, *, key=None) -> NamedArray:
        """
        Forward pass through vision embeddings.

        Args:
            pixel_values: Input images with shape (batch, channels, height, width)
            key: Optional PRNGKey

        Returns:
            Embeddings with position information added, shape (batch, num_patches, embed)
        """
        k_patch, k_pos = maybe_rng_split(key, 2)

        # Apply patch embeddings using Conv2d
        # Input: (batch, channels, height, width)
        # Output: (batch, embed, num_patches_h, num_patches_w)
        patch_embeds = self.patch_embedding(pixel_values, key=k_patch)

        # Flatten spatial dimensions to get (batch, embed, num_patches)
        # Then transpose to (batch, num_patches, embed)
        # Note: We need to handle named axes properly
        # patch_embeds has axes like (batch, embed, height, width) after conv
        # We need to flatten height and width into num_patches

        # Flatten the spatial dimensions
        # Assuming patch_embeds has shape (batch, embed, h_patches, w_patches)
        batch_axes = [ax for ax in patch_embeds.axes if ax.name == "batch"]
        embed_axis = self.config.Embed
        spatial_axes = [ax for ax in patch_embeds.axes if ax not in batch_axes and ax != embed_axis]

        # Calculate total number of patches
        num_patches_total = 1
        for ax in spatial_axes:
            num_patches_total *= ax.size

        # Create the num_patches axis with actual size from flattened spatial dims
        NumPatchesActual = Axis("num_patches", num_patches_total)

        # Rearrange: flatten spatial dimensions and move to sequence position
        # We'll use array manipulation since haliax doesn't have a direct flatten for multiple axes
        arr = patch_embeds.array

        # Get the batch size if present
        if batch_axes:
            batch_size = batch_axes[0].size
            # Reshape to (batch, embed, num_patches)
            arr = arr.reshape(batch_size, embed_axis.size, -1)
            # Transpose to (batch, num_patches, embed)
            arr = jnp.transpose(arr, (0, 2, 1))
            patch_embeds = hax.named(arr, (batch_axes[0], NumPatchesActual, embed_axis))
        else:
            # No batch dimension
            arr = arr.reshape(embed_axis.size, -1)
            arr = jnp.transpose(arr, (1, 0))
            patch_embeds = hax.named(arr, (NumPatchesActual, embed_axis))

        # Add position embeddings
        # Standard position IDs: 0, 1, 2, ..., num_patches-1
        position_ids = hax.arange(NumPatchesActual)
        pos_embeds = self.position_embedding(position_ids)

        # Add position embeddings to patch embeddings
        # Broadcasting will handle batch dimensions
        embeddings = patch_embeds + pos_embeds

        return embeddings


# =====================
# SigLIP Vision Transformer
# =====================


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
    layers: Stacked[SiglipEncoderLayer]
    post_layernorm: hnn.LayerNorm

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        """Map Levanter field names to HuggingFace state dict keys."""
        return {"layers": "encoder.layers"}  # HF uses encoder.layers instead of layers

    @staticmethod
    def init(config: SiglipVisionConfig, *, key) -> "SiglipVisionTransformer":
        """
        Initialize SiglipVisionTransformer.

        Args:
            config: SiglipVisionConfig
            key: PRNGKey for initialization

        Returns:
            Initialized SiglipVisionTransformer module
        """
        k_embed, k_layers = maybe_rng_split(key, 2)

        # Initialize embeddings
        embeddings = SiglipVisionEmbeddings.init(config, key=k_embed)

        # Initialize stacked encoder layers
        layers = Stacked.init(
            config.Layers,
            SiglipEncoderLayer,
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
        *,
        key=None,
    ) -> NamedArray:
        """
        Forward pass through vision transformer.

        Args:
            pixel_values: Input images with shape (batch, channels, height, width)
            mask: Optional attention mask
            key: PRNGKey for dropout

        Returns:
            Encoded representations with shape (batch, num_patches, embed)
        """
        k_embed, k_layers = maybe_rng_split(key, 2)

        # Get embeddings
        hidden_states = self.embeddings(pixel_values, key=k_embed)

        # Pass through encoder layers
        keys = maybe_rng_split(k_layers, self.config.num_hidden_layers) if k_layers is not None else None
        hidden_states = self.layers.fold(hidden_states, mask, key=keys)

        # Apply post-layer normalization
        hidden_states = self.post_layernorm(hidden_states)

        return hidden_states


# =====================
# SigLIP Vision Model (HF-compatible wrapper)
# =====================


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

    def get_hf_config(self):
        """Override to avoid requiring vocab_size for vision models."""
        return self.config.to_hf_config()

    @classmethod
    def init(cls, Vocab: Axis, config: SiglipVisionConfig, *, key) -> "SiglipVisionModel":
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
        *,
        key=None,
    ) -> NamedArray:
        """
        Forward pass through vision model.

        Args:
            pixel_values: Input images with shape (batch, channels, height, width)
            mask: Optional attention mask
            key: PRNGKey for dropout

        Returns:
            Encoded representations with shape (batch, num_patches, embed)
        """
        return self.vision_model(pixel_values, mask=mask, key=key)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        """Map Levanter field names to HuggingFace state dict keys."""
        return {}  # Keep vision_model prefix as-is (matches HF structure)

    def from_state_dict(self, state_dict: Dict[str, jnp.ndarray], prefix: Optional[str] = None):
        """Load from state dict."""
        from haliax._src.state_dict import default_eqx_module_from_state_dict

        # Use default loading
        return default_eqx_module_from_state_dict(self, state_dict, prefix)


__all__ = [
    "SiglipVisionConfig",
    "SiglipMLP",
    "SiglipAttention",
    "SiglipEncoderLayer",
    "SiglipVisionEmbeddings",
    "SiglipVisionTransformer",
    "SiglipVisionModel",
]
