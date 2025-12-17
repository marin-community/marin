# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

import equinox as eqx
import jax.numpy as jnp

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.compat.hf_checkpoints import HFCheckpointConverter, ModelWithHfSerializationMixin
from levanter.layers.attention import AttentionMask, dot_product_attention
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.logging import silence_transformer_nag


silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig  # noqa: E402
from transformers import Siglip2VisionConfig as HfSiglip2VisionConfig  # noqa: E402


@dataclass(frozen=True)
class Siglip2VisionConfig:
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
        initializer_range: The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        gradient_checkpointing: Whether to use gradient checkpointing to save memory.
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
    initializer_range: float = 0.02
    gradient_checkpointing: bool = True

    # Reference checkpoint for loading pretrained models
    reference_checkpoint: Optional[str] = None

    @property
    def model_type(self) -> Type:
        """Return the model class type."""
        return Siglip2VisionModel

    def hf_checkpoint_converter(
        self, ref_checkpoint: Optional[str] = None
    ) -> HFCheckpointConverter["Siglip2VisionConfig"]:  # type: ignore
        """Create HuggingFace checkpoint converter for this config."""
        # Vision-only models don't have a tokenizer, but HFCheckpointConverter requires one
        # Use gpt2 tokenizer as a placeholder since it's always available
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=False,
            tokenizer="gpt2",  # Dummy tokenizer for vision-only model
            HfConfigClass=HfSiglip2VisionConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig) -> "Siglip2VisionConfig":
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

        # Calculate num_patches if not provided
        # num_patches = (image_size / patch_size) ^ 2
        if hasattr(hf_config, "num_patches"):
            num_patches = hf_config.num_patches
        else:
            # Calculate from image_size and patch_size
            grid_size = hf_config.image_size // hf_config.patch_size
            num_patches = grid_size * grid_size

        return cls(
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_channels=hf_config.num_channels,
            num_patches=num_patches,
            patch_size=hf_config.patch_size,
            hidden_act=activation_fn,
            layer_norm_eps=hf_config.layer_norm_eps,
            attention_dropout=hf_config.attention_dropout,
        )

    def to_hf_config(
        self, vocab_size: Optional[int] = None, config_overrides: Optional[Dict] = None
    ) -> HfSiglip2VisionConfig:
        """Convert from Levanter config to HuggingFace config.

        Args:
            vocab_size: Ignored for vision models (present for interface compatibility)
            config_overrides: Optional config overrides
        """
        # vocab_size is ignored for vision models
        if config_overrides is None:
            config_overrides = {}

        # Map activation function back to HF format
        # gelu_new in Levanter maps back to gelu_pytorch_tanh in HF (for Siglip2 compatibility)
        if isinstance(self.hidden_act, ActivationFunctionEnum):
            if self.hidden_act == ActivationFunctionEnum.gelu_new:
                hf_hidden_act = "gelu_pytorch_tanh"
            else:
                hf_hidden_act = self.hidden_act.value
        else:
            hf_hidden_act = self.hidden_act

        # Calculate image_size from num_patches and patch_size
        # This is needed for compatibility with LlavaOnevision which expects image_size
        grid_size = int(self.num_patches**0.5)
        image_size = grid_size * self.patch_size

        hf_config = HfSiglip2VisionConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_channels=self.num_channels,
            num_patches=self.num_patches,
            patch_size=self.patch_size,
            hidden_act=hf_hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            attention_dropout=self.attention_dropout,
            **config_overrides,
        )

        # Add image_size as a manual attribute for LlavaOnevision compatibility
        # HfSiglip2VisionConfig doesn't have image_size in __init__, but we can set it manually
        hf_config.image_size = image_size

        return hf_config

    # Axis definitions following marin/Levanter patterns
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


# =====================
# Siglip2 MLP
# =====================


class Siglip2MLP(eqx.Module):
    """
    MLP module for Siglip2 Vision Transformer.

    Implements a two-layer feedforward network with activation function in between.
    """

    fc1: hnn.Linear  # projection from Embed to Mlp (intermediate)
    fc2: hnn.Linear  # projection from Mlp to Embed
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(Embed: Axis, Mlp: Axis, activation_fn: ActivationFunctionEnum, *, key) -> "Siglip2MLP":
        """
        Initialize Siglip2MLP.

        Args:
            Embed: Embedding dimension axis
            Mlp: MLP intermediate dimension axis
            activation_fn: Activation function enum
            key: PRNGKey for initialization

        Returns:
            Initialized Siglip2MLP module
        """
        k_fc1, k_fc2 = maybe_rng_split(key, 2)

        # In Siglip2, fc1 goes from hidden_size to intermediate_size
        fc1 = hnn.Linear.init(In=Embed, Out=Mlp, key=k_fc1, use_bias=True, out_first=True)
        # fc2 goes from intermediate_size back to hidden_size
        fc2 = hnn.Linear.init(In=Mlp, Out=Embed, key=k_fc2, use_bias=True, out_first=True)

        # Convert activation function enum to callable
        activation_fn_callable = (
            activation_fn.to_fn() if isinstance(activation_fn, ActivationFunctionEnum) else activation_fn
        )

        return Siglip2MLP(fc1, fc2, activation_fn_callable)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        """
        Forward pass through MLP.

        Args:
            x: Input tensor with Embed axis
            key: Optional PRNGKey for dropout (not used in Siglip2)

        Returns:
            Output tensor with Embed axis
        """
        k1, k2 = maybe_rng_split(key, 2)
        x = self.fc1(x, key=k1)
        x = self.act(x)
        x = self.fc2(x, key=k2)
        return x


# =====================
# Siglip2 Attention
# =====================


class Siglip2Attention(eqx.Module):
    """
    Multi-headed attention module for Siglip2.

    Implements standard multi-head self-attention with separate Q, K, V projections
    and an output projection.
    """

    config: Siglip2VisionConfig = eqx.field(static=True)
    q_proj: hnn.Linear  # Query projection from Embed to (Heads, HeadSize)
    k_proj: hnn.Linear  # Key projection from Embed to (Heads, HeadSize)
    v_proj: hnn.Linear  # Value projection from Embed to (Heads, HeadSize)
    out_proj: hnn.Linear  # Output projection from (Heads, HeadSize) to Embed

    @staticmethod
    def init(config: Siglip2VisionConfig, *, key) -> "Siglip2Attention":
        """
        Initialize Siglip2Attention.

        Args:
            config: Siglip2VisionConfig
            key: PRNGKey for initialization

        Returns:
            Initialized Siglip2Attention module
        """
        k_q, k_k, k_v, k_out = maybe_rng_split(key, 4)

        Embed = config.Embed
        Heads = config.Heads
        HeadSize = config.HeadSize

        # Initialize projection layers
        # All projections use bias in Siglip2
        q_proj = hnn.Linear.init(In=Embed, Out=(Heads, HeadSize), key=k_q, use_bias=True, out_first=True)
        k_proj = hnn.Linear.init(In=Embed, Out=(Heads, HeadSize), key=k_k, use_bias=True, out_first=True)
        v_proj = hnn.Linear.init(In=Embed, Out=(Heads, HeadSize), key=k_v, use_bias=True, out_first=True)
        out_proj = hnn.Linear.init(In=(Heads, HeadSize), Out=Embed, key=k_out, use_bias=True, out_first=True)

        return Siglip2Attention(config, q_proj, k_proj, v_proj, out_proj)

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

        # Find the sequence axis (the one that's not Embed and not a common batch axis)
        # This handles cases where the axis might be named "num_patches" or "position"
        embed_axis = self.config.Embed
        common_batch_axes = {"batch", "Batch"}
        sequence_axis = None

        # First, check if "position" axis already exists
        for axis in x.axes:
            if axis.name == "position":
                sequence_axis = axis
                break

        # If not, look for sequence-like axes (num_patches, seq_len, etc.)
        if sequence_axis is None:
            sequence_like_names = {"num_patches", "seq_len", "seq", "length"}
            for axis in x.axes:
                if axis != embed_axis and axis.name not in common_batch_axes:
                    if axis.name in sequence_like_names:
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
        # We'll rename it back at the end
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
        # Siglip2 uses standard scaled dot-product attention
        attn_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask=mask,
            inference=False,  # Siglip2VisionConfig doesn't have inference mode
            use_flash=self.config.gradient_checkpointing,  # Use flash attention if gradient checkpointing enabled
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
# Siglip2 Encoder Layer
# =====================


class Siglip2EncoderLayer(eqx.Module):
    """
    Siglip2 Encoder Layer.

    Implements a transformer encoder layer with:
    - Pre-LayerNorm architecture
    - Self-attention with residual connection
    - MLP with residual connection
    """

    config: Siglip2VisionConfig = eqx.field(static=True)
    layer_norm1: hnn.LayerNorm  # Pre-attention layer norm
    self_attn: Siglip2Attention  # Self-attention module
    layer_norm2: hnn.LayerNorm  # Pre-MLP layer norm
    mlp: Siglip2MLP  # MLP module

    @staticmethod
    def init(config: Siglip2VisionConfig, *, key) -> "Siglip2EncoderLayer":
        """
        Initialize Siglip2EncoderLayer.

        Args:
            config: Siglip2VisionConfig
            key: PRNGKey for initialization

        Returns:
            Initialized Siglip2EncoderLayer module
        """
        k_attn, k_mlp = maybe_rng_split(key, 2)

        # Initialize layer norms (no bias in Siglip2)
        layer_norm1 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_eps, use_bias=True)
        layer_norm2 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_eps, use_bias=True)

        # Initialize attention and MLP
        self_attn = Siglip2Attention.init(config, key=k_attn)
        mlp = Siglip2MLP.init(config.Embed, config.Mlp, config.hidden_act, key=k_mlp)

        return Siglip2EncoderLayer(config, layer_norm1, self_attn, layer_norm2, mlp)

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
# Siglip2 Vision Embeddings
# =====================


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
    def init(config: Siglip2VisionConfig, *, key) -> "Siglip2VisionEmbeddings":
        """
        Initialize Siglip2VisionEmbeddings.

        Args:
            config: Siglip2VisionConfig
            key: PRNGKey for initialization

        Returns:
            Initialized Siglip2VisionEmbeddings module
        """
        k_patch, k_pos = maybe_rng_split(key, 2)

        # Patch embedding: linear projection from flattened patches to embed_dim
        # Input: num_channels * patch_size * patch_size
        # Output: hidden_size
        patch_input_dim = config.num_channels * config.patch_size * config.patch_size
        PatchInput = Axis(name="patch_input", size=patch_input_dim)

        patch_embedding = hnn.Linear.init(
            In=PatchInput,
            Out=config.Embed,
            key=k_patch,
            use_bias=True,
            out_first=True,
        )

        # Position embedding: learnable embeddings for each patch position
        position_embedding = hnn.Embedding.init(
            config.NumPatches,
            config.Embed,
            key=k_pos,
        )

        return Siglip2VisionEmbeddings(config, patch_embedding, position_embedding)

    @named_call
    def __call__(self, pixel_values: NamedArray, spatial_shapes=None, *, key=None) -> NamedArray:
        """
        Forward pass through vision embeddings.

        Args:
            pixel_values: Patchified pixel values with shape (..., num_patches, patch_input_dim)
                where patch_input_dim = num_channels * patch_size * patch_size
            spatial_shapes: Optional array of shape (batch, 2) containing [height, width] in patches
                for each image. If provided, position embeddings will be interpolated to match.
            key: Optional PRNGKey

        Returns:
            Embeddings with position information added
        """
        import jax.numpy as jnp
        import jax.image

        k_patch, k_pos = maybe_rng_split(key, 2)

        # Apply patch embeddings to patchified pixels
        # Shape: (..., num_patches, patch_input_dim) -> (..., num_patches, hidden_size)
        patch_embeds = self.patch_embedding(pixel_values, key=k_patch)

        # Get position embeddings
        num_patches_axis = pixel_values.resolve_axis("num_patches")

        if spatial_shapes is not None:
            # Interpolate position embeddings to match spatial_shapes
            # This is needed for flexible aspect ratio support

            # Get the pretrained position embeddings (assuming square grid)
            num_positions = self.config.NumPatches.size
            grid_size = int(num_positions**0.5)

            # Get all position embeddings and reshape to 2D grid
            # Shape: (num_positions, embed_dim) -> (grid_size, grid_size, embed_dim)
            all_pos_ids = hax.arange(self.config.NumPatches)
            all_pos_embeds = self.position_embedding(all_pos_ids)  # (num_patches, embed)
            pos_embeds_2d = all_pos_embeds.array.reshape(grid_size, grid_size, -1)

            # Get target height and width from pixel_values shape (JIT-safe)
            # num_patches_axis.size is static at trace time
            # For square grids: target_h = target_w = sqrt(num_patches)
            # For non-square: use spatial_shapes if it contains Python ints, otherwise infer from num_patches
            expected_num_patches = num_patches_axis.size

            # Check if spatial_shapes contains concrete Python values or is traced
            # If spatial_shapes is a numpy array or contains Python ints, use it directly
            # Otherwise, infer from pixel_values shape (assumes square grid)
            try:
                # Try to get concrete values - works for numpy arrays and Python values
                target_h = int(spatial_shapes[0, 0])
                target_w = int(spatial_shapes[0, 1])
            except (TypeError, jax.errors.ConcretizationTypeError):
                # spatial_shapes is traced, infer from pixel_values (assumes square)
                target_h = target_w = int(expected_num_patches**0.5)

            # Use JAX's resize function to interpolate
            # Need to permute to (embed, height, width) for resize, then back
            pos_embeds_2d = jnp.transpose(pos_embeds_2d, (2, 0, 1))  # (embed, h, w)
            pos_embeds_resized = jax.image.resize(
                pos_embeds_2d,
                shape=(pos_embeds_2d.shape[0], target_h, target_w),
                method="linear",  # 'linear' (bilinear for 2D) is the closest to PyTorch's bilinear
            )
            # Reshape back to (num_patches, embed)
            pos_embeds_resized = jnp.transpose(pos_embeds_resized, (1, 2, 0))  # (h, w, embed)
            pos_embeds_flat = pos_embeds_resized.reshape(-1, pos_embeds_resized.shape[-1])

            # The interpolated position embeddings may have different number of patches than pixel_values
            # (e.g., 14*18=252 vs 256 if pixel_values is padded)
            # We need to broadcast/pad the position embeddings to match
            actual_num_patches_interp = target_h * target_w

            if actual_num_patches_interp < expected_num_patches:
                # Pad by repeating the first embedding value (matching HF behavior)
                # HF does: resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]
                padding = expected_num_patches - actual_num_patches_interp
                first_embedding = pos_embeds_flat[0:1]  # Shape: (1, embed_dim)
                repeated_padding = jnp.repeat(first_embedding, padding, axis=0)  # Shape: (padding, embed_dim)
                pos_embeds_flat = jnp.concatenate([pos_embeds_flat, repeated_padding], axis=0)
            elif actual_num_patches_interp > expected_num_patches:
                # Truncate to match expected size (shouldn't happen normally)
                # pos_embeds_flat = pos_embeds_flat[:expected_num_patches]
                raise ValueError(
                    f"Actual number of patches {actual_num_patches_interp} does not match expected number of patches {expected_num_patches}"
                )
            # assert actual_num_patches_interp == expected_num_patches, f"Actual number of patches {actual_num_patches_interp} does not match expected number of patches {expected_num_patches}"

            # Create NamedArray with correct axis
            pos_embeds = hax.named(pos_embeds_flat, (num_patches_axis, self.config.Embed))
        else:
            # Standard position embeddings (square grid)
            position_ids = hax.arange(num_patches_axis)
            pos_embeds = self.position_embedding(position_ids)

        # Add position embeddings to patch embeddings
        # Broadcasting will handle batch dimensions
        embeddings = patch_embeds + pos_embeds

        return embeddings


# =====================
# Siglip2 Vision Transformer
# =====================


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
    layers: Stacked[Siglip2EncoderLayer]
    post_layernorm: hnn.LayerNorm

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        """Map Levanter field names to HuggingFace state dict keys."""
        return {"layers": "encoder.layers"}  # HF uses encoder.layers instead of layers

    @staticmethod
    def init(config: Siglip2VisionConfig, *, key) -> "Siglip2VisionTransformer":
        """
        Initialize Siglip2VisionTransformer.

        Args:
            config: Siglip2VisionConfig
            key: PRNGKey for initialization

        Returns:
            Initialized Siglip2VisionTransformer module
        """
        k_embed, k_layers = maybe_rng_split(key, 2)

        # Initialize embeddings
        embeddings = Siglip2VisionEmbeddings.init(config, key=k_embed)

        # Initialize stacked encoder layers
        layers = Stacked.init(
            config.Layers,
            Siglip2EncoderLayer,
            gradient_checkpointing=config.gradient_checkpointing,
        )(config, key=shaped_rng_split(k_layers, config.num_hidden_layers))

        # Post-encoder layer norm
        post_layernorm = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_eps, use_bias=True)

        return Siglip2VisionTransformer(config, embeddings, layers, post_layernorm)

    @named_call
    def __call__(
        self,
        pixel_values: NamedArray,
        mask: Optional[AttentionMask] = None,
        spatial_shapes=None,
        *,
        key=None,
    ) -> NamedArray:
        """
        Forward pass through vision transformer.

        Args:
            pixel_values: Patchified pixel values with shape (..., num_patches, patch_input_dim)
            mask: Optional attention mask
            spatial_shapes: Optional array of shape (batch, 2) containing [height, width] in patches
            key: PRNGKey for dropout

        Returns:
            Encoded representations with shape (..., num_patches, embed)
        """
        k_embed, k_layers = maybe_rng_split(key, 2)

        # Get embeddings with spatial_shapes support
        hidden_states = self.embeddings(pixel_values, spatial_shapes=spatial_shapes, key=k_embed)

        # Pass through encoder layers
        keys = maybe_rng_split(k_layers, self.config.num_hidden_layers) if k_layers is not None else None
        hidden_states = self.layers.fold(hidden_states, mask, key=keys)

        # Apply post-layer normalization
        hidden_states = self.post_layernorm(hidden_states)

        return hidden_states


# =====================
# Siglip2 Multihead Attention Pooling Head
# =====================


class Siglip2MultiheadAttentionPoolingHead(ModuleWithStateDictSerialization):
    """
    Multihead attention pooling head for Siglip2.

    Uses a learnable probe to attend to encoder outputs and produce a pooled representation.
    The output is a single vector per batch element (not a sequence).
    """

    config: Siglip2VisionConfig = eqx.field(static=True)
    probe: NamedArray  # Learnable query: (1, embed)
    q_proj: hnn.Linear  # Query projection for probe
    k_proj: hnn.Linear  # Key projection for hidden states
    v_proj: hnn.Linear  # Value projection for hidden states
    out_proj: hnn.Linear  # Output projection
    layernorm: hnn.LayerNorm
    mlp: Siglip2MLP

    @staticmethod
    def init(config: Siglip2VisionConfig, *, key) -> "Siglip2MultiheadAttentionPoolingHead":
        """
        Initialize Siglip2MultiheadAttentionPoolingHead.

        Args:
            config: Siglip2VisionConfig
            key: PRNGKey for initialization

        Returns:
            Initialized head module
        """
        k_probe, k_q, k_k, k_v, k_out, k_mlp = maybe_rng_split(key, 6)

        ProbeSeq = Axis("probe_seq", 1)

        # Learnable probe: (1, hidden_size)
        probe = hax.random.normal(k_probe, (ProbeSeq, config.Embed)) * config.initializer_range

        # Attention projections (Q, K, V, out)
        # Q projection for probe
        q_proj = hnn.Linear.init(
            In=config.Embed,
            Out=(config.Heads, config.HeadSize),
            key=k_q,
            use_bias=True,
            out_first=True,
        )
        # K projection for hidden states
        k_proj = hnn.Linear.init(
            In=config.Embed,
            Out=(config.Heads, config.HeadSize),
            key=k_k,
            use_bias=True,
            out_first=True,
        )
        # V projection for hidden states
        v_proj = hnn.Linear.init(
            In=config.Embed,
            Out=(config.Heads, config.HeadSize),
            key=k_v,
            use_bias=True,
            out_first=True,
        )
        # Output projection
        out_proj = hnn.Linear.init(
            In=(config.Heads, config.HeadSize),
            Out=config.Embed,
            key=k_out,
            use_bias=True,
            out_first=True,
        )

        # Layer norm
        layernorm = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_eps, use_bias=True)

        # MLP
        mlp = Siglip2MLP.init(config.Embed, config.Mlp, config.hidden_act, key=k_mlp)

        return Siglip2MultiheadAttentionPoolingHead(
            config=config,
            probe=probe,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            out_proj=out_proj,
            layernorm=layernorm,
            mlp=mlp,
        )

    @named_call
    def __call__(
        self,
        hidden_states: NamedArray,
        mask: Optional[AttentionMask] = None,
        *,
        key=None,
    ) -> NamedArray:
        """
        Forward pass through attention pooling head.

        Args:
            hidden_states: Encoder output with shape (..., num_patches, embed)
            mask: Optional attention mask
            key: PRNGKey for dropout

        Returns:
            Pooled representation with shape (..., embed)
        """
        k_q, k_k, k_v, k_out, k_mlp = maybe_rng_split(key, 5)

        # Expand probe for batch dimensions
        # probe: (probe_seq=1, embed) -> broadcast with hidden_states batch dims
        probe = self.probe

        # Project probe to Q
        q = self.q_proj(probe, key=k_q)  # (probe_seq, heads, head_size)

        # Project hidden states to K, V
        k = self.k_proj(hidden_states, key=k_k)  # (..., num_patches, heads, head_size)
        v = self.v_proj(hidden_states, key=k_v)  # (..., num_patches, heads, head_size)

        # Broadcast q to match batch dimensions of k and v
        # q needs to have the same batch dims as k/v for attention
        # Extract batch axes from k (all axes except num_patches, heads, head_size)
        batch_axes = [ax for ax in k.axes if ax.name not in ["num_patches", "heads", "head_size"]]
        for ax in batch_axes:
            q = hax.broadcast_to(q, (ax,) + q.axes)

        # Rearrange for attention: put heads first
        q = q.rearrange((..., "heads", "probe_seq", "head_size"))
        k = k.rearrange((..., "heads", "num_patches", "head_size"))
        v = v.rearrange((..., "heads", "num_patches", "head_size"))

        # Rename for attention
        k = k.rename({"num_patches": "key_position"})
        v = v.rename({"num_patches": "key_position"})

        # Cross-attention: probe attends to hidden states
        attn_output = dot_product_attention(
            "probe_seq",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask=mask,
            inference=False,
            dropout=self.config.attention_dropout,
            prng=key,
        )

        # Project back to embed dimension
        attn_output = attn_output.astype(hidden_states.dtype)
        attn_output = self.out_proj(attn_output, key=k_out)  # (..., probe_seq, embed)

        # Residual connection with probe (broadcast probe to batch dims)
        hidden_states = probe + attn_output

        # Squeeze probe_seq dimension to get (..., embed)
        ProbeSeq = hidden_states.resolve_axis("probe_seq")
        hidden_states = hidden_states[ProbeSeq, 0]  # Remove probe_seq dim

        # Layer norm + MLP with residual
        residual = hidden_states
        hidden_states = self.layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states, key=k_mlp)

        return hidden_states

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        """Map Levanter field names to HuggingFace state dict keys."""
        return {
            "out_proj": "attention.out_proj",
            "layernorm": "layernorm",
            "mlp": "mlp",
        }

    def to_state_dict(self, prefix: Optional[str] = None) -> Dict[str, jnp.ndarray]:
        """Convert to HuggingFace state dict format with combined in_proj."""
        from haliax.state_dict import to_state_dict as eqx_to_state_dict, with_prefix

        state_dict: Dict[str, jnp.ndarray] = {}

        # Probe
        state_dict[with_prefix(prefix, "probe")] = self.probe.array

        # Combine Q, K, V projections into in_proj
        # HF shape: (3 * hidden_size, hidden_size)
        q_weight = self.q_proj.weight.array  # (heads, head_size, embed)
        k_weight = self.k_proj.weight.array
        v_weight = self.v_proj.weight.array

        # Reshape to (hidden_size, embed) and stack
        hidden_size = q_weight.shape[0] * q_weight.shape[1]
        embed_size = q_weight.shape[2]

        q_flat = q_weight.reshape(hidden_size, embed_size)
        k_flat = k_weight.reshape(hidden_size, embed_size)
        v_flat = v_weight.reshape(hidden_size, embed_size)

        in_proj_weight = jnp.concatenate([q_flat, k_flat, v_flat], axis=0)
        state_dict[with_prefix(prefix, "attention.in_proj_weight")] = in_proj_weight

        # Combine biases
        if self.q_proj.bias is not None:
            q_bias = self.q_proj.bias.array.reshape(-1)
            k_bias = self.k_proj.bias.array.reshape(-1)
            v_bias = self.v_proj.bias.array.reshape(-1)
            in_proj_bias = jnp.concatenate([q_bias, k_bias, v_bias], axis=0)
            state_dict[with_prefix(prefix, "attention.in_proj_bias")] = in_proj_bias

        # Output projection
        out_dict = eqx_to_state_dict(self.out_proj, with_prefix(prefix, "attention.out_proj"))
        state_dict.update(out_dict)

        # Layer norm
        ln_dict = eqx_to_state_dict(self.layernorm, with_prefix(prefix, "layernorm"))
        state_dict.update(ln_dict)

        # MLP
        mlp_dict = eqx_to_state_dict(self.mlp, with_prefix(prefix, "mlp"))
        state_dict.update(mlp_dict)

        return state_dict

    def from_state_dict(self, state_dict: Dict[str, jnp.ndarray], prefix: Optional[str] = None):
        """Load from HuggingFace state dict format with combined in_proj."""
        from haliax.state_dict import with_prefix, from_state_dict
        import dataclasses

        # Load probe
        probe_key = with_prefix(prefix, "probe")
        if probe_key in state_dict:
            probe_array = state_dict[probe_key]
            # HF shape: (1, 1, hidden_size) -> we want (probe_seq=1, embed)
            if probe_array.ndim == 3:
                probe_array = probe_array.squeeze(0)  # Remove batch dim
            probe = hax.named(probe_array, self.probe.axes)
        else:
            probe = self.probe

        # Split in_proj into Q, K, V
        in_proj_weight_key = with_prefix(prefix, "attention.in_proj_weight")
        in_proj_bias_key = with_prefix(prefix, "attention.in_proj_bias")

        if in_proj_weight_key in state_dict:
            in_proj_weight = state_dict[in_proj_weight_key]  # (3 * hidden_size, hidden_size)

            # Split into Q, K, V
            q_weight, k_weight, v_weight = jnp.split(in_proj_weight, 3, axis=0)

            # The weights are already in the flattened format (hidden_size, embed_size)
            # which matches our expected axes (__OUT__, __IN__) after flattening
            # No need to reshape since the template is already flattened at this point

            q_proj_weight = hax.named(q_weight, self.q_proj.weight.axes)
            k_proj_weight = hax.named(k_weight, self.k_proj.weight.axes)
            v_proj_weight = hax.named(v_weight, self.v_proj.weight.axes)
        else:
            q_proj_weight = self.q_proj.weight
            k_proj_weight = self.k_proj.weight
            v_proj_weight = self.v_proj.weight

        # Handle biases
        if in_proj_bias_key in state_dict:
            in_proj_bias = state_dict[in_proj_bias_key]  # (3 * hidden_size,)
            q_bias, k_bias, v_bias = jnp.split(in_proj_bias, 3, axis=0)

            # The biases are already in the flattened format (hidden_size,)
            # which matches our expected axes (__OUT__,) after flattening
            # No need to reshape since the template is already flattened at this point

            q_proj_bias = hax.named(q_bias, self.q_proj.bias.axes)
            k_proj_bias = hax.named(k_bias, self.k_proj.bias.axes)
            v_proj_bias = hax.named(v_bias, self.v_proj.bias.axes)
        else:
            q_proj_bias = self.q_proj.bias
            k_proj_bias = self.k_proj.bias
            v_proj_bias = self.v_proj.bias

        # Create updated projections
        q_proj = dataclasses.replace(self.q_proj, weight=q_proj_weight, bias=q_proj_bias)
        k_proj = dataclasses.replace(self.k_proj, weight=k_proj_weight, bias=k_proj_bias)
        v_proj = dataclasses.replace(self.v_proj, weight=v_proj_weight, bias=v_proj_bias)

        # Load out_proj using default mechanism
        out_proj = from_state_dict(self.out_proj, state_dict, with_prefix(prefix, "attention.out_proj"))

        # Load layernorm
        layernorm = from_state_dict(self.layernorm, state_dict, with_prefix(prefix, "layernorm"))

        # Load MLP
        mlp = from_state_dict(self.mlp, state_dict, with_prefix(prefix, "mlp"))

        return Siglip2MultiheadAttentionPoolingHead(
            config=self.config,
            probe=probe,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            out_proj=out_proj,
            layernorm=layernorm,
            mlp=mlp,
        )


# =====================
# Siglip2 Vision Model (HF-compatible wrapper)
# =====================


class Siglip2VisionModel(ModuleWithStateDictSerialization, ModelWithHfSerializationMixin[Siglip2VisionConfig]):
    """
    Siglip2 Vision Model with HuggingFace compatibility.

    This is a wrapper around Siglip2VisionTransformer that implements
    the ModelWithHfSerializationMixin interface for checkpoint conversion.
    """

    vision_model: Siglip2VisionTransformer

    @property
    def config(self) -> Siglip2VisionConfig:
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
    def init(cls, Vocab: Axis, config: Siglip2VisionConfig, *, key) -> "Siglip2VisionModel":
        """
        Initialize Siglip2VisionModel.

        Args:
            Vocab: Dummy vocab axis (not used for vision models, but required by interface)
            config: Siglip2VisionConfig
            key: PRNGKey for initialization

        Returns:
            Initialized Siglip2VisionModel
        """
        vision_model = Siglip2VisionTransformer.init(config, key=key)
        return cls(vision_model=vision_model)

    @named_call
    def __call__(
        self,
        pixel_values: NamedArray,
        mask: Optional[AttentionMask] = None,
        spatial_shapes=None,
        *,
        key=None,
    ) -> NamedArray:
        """
        Forward pass through vision model.

        Args:
            pixel_values: Patchified pixel values with shape (..., num_patches, patch_input_dim)
            mask: Optional attention mask
            spatial_shapes: Optional array of shape (batch, 2) containing [height, width] in patches
            key: PRNGKey for dropout

        Returns:
            Encoded representations with shape (..., num_patches, embed)
        """
        return self.vision_model(pixel_values, mask=mask, spatial_shapes=spatial_shapes, key=key)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        """Map Levanter field names to HuggingFace state dict keys."""
        return {}  # Keep vision_model prefix as-is (matches HF structure)

    def from_state_dict(self, state_dict: Dict[str, jnp.ndarray], prefix: Optional[str] = None):
        """Load from state dict."""
        from haliax._src.state_dict import default_eqx_module_from_state_dict

        # Use default loading
        return default_eqx_module_from_state_dict(self, state_dict, prefix)
