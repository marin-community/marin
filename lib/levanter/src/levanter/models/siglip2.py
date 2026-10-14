# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from haliax.state_dict import to_state_dict as eqx_to_state_dict, with_prefix
from haliax.state_dict import from_state_dict
import dataclasses
from haliax._src.state_dict import default_eqx_module_from_state_dict


from jaxtyping import PRNGKeyArray
from dataclasses import dataclass
from typing import Self, Dict, Optional, Tuple, Type

import jax
from levanter.models.vision_model import VisionEncoderConfig
from levanter.models.vision_transformer import vision_activation, vision_hidden_states, VisionEncoderLayer, VisionMLP
import equinox as eqx
import jax.numpy as jnp

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.compat.hf_checkpoints import HFCheckpointConverter, ModelWithHfSerializationMixin
from levanter.layers.attention import AttentionBackend, AttentionMask, dot_product_attention
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
            Contains all layers when requested, otherwise only the final encoder output.
    """

    last_hidden_state: NamedArray
    hidden_states: Optional[Tuple[NamedArray, ...]] = None


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
    initializer_range: float = 0.02
    gradient_checkpointing: bool = True

    # Attention-related config
    use_flash_attention: bool = False
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None
    inference: bool = True  # Whether to run in inference mode (disables dropout)

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
        return HFCheckpointConverter(
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
        activation_fn = vision_activation(hf_config.hidden_act)

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
        # Apply patch embeddings to patchified pixels
        # Shape: (..., num_patches, patch_input_dim) -> (..., num_patches, hidden_size)
        patch_embeds = self.patch_embedding(pixel_values)

        # Get position embeddings
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
            # Standard position embeddings (square grid)
            position_ids = hax.arange(num_patches_axis)
            pos_embeds = self.position_embedding(position_ids)

        # Add position embeddings to patch embeddings
        # Broadcasting will handle batch dimensions
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
    layers: Stacked[VisionEncoderLayer]
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

        # Initialize stacked encoder layers
        layers = Stacked.init(
            config.Layers,
            VisionEncoderLayer,
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
        spatial_shapes: np.ndarray | None = None,
        output_hidden_states: bool = False,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Siglip2VisionModelOutput:
        """
        Forward pass through vision transformer.

        Args:
            pixel_values: Patchified pixel values with shape (..., num_patches, patch_input_dim)
            mask: Optional attention mask
            spatial_shapes: Optional array of shape (batch, 2) containing [height, width] in patches
            output_hidden_states: Whether to return hidden states from all layers.
                If True, returns all layer hidden states.
                If False, only returns the last hidden state (more efficient).
            key: PRNGKey for dropout

        Returns:
            Siglip2VisionModelOutput containing:
                - last_hidden_state: Final encoded representations after post_layernorm
                - hidden_states: Tuple of hidden states. When output_hidden_states=True, contains all layers.
                    When output_hidden_states=False, contains only the last layer output (before post_layernorm)
                    to support vision_feature_layer=-1 without collecting all intermediate states.
        """
        k_embed, k_layers = maybe_rng_split(key, 2)

        # Get embeddings with spatial_shapes support
        hidden_states = self.embeddings(pixel_values, spatial_shapes=spatial_shapes, key=k_embed)

        hidden_states, result_hidden_states = vision_hidden_states(
            self.layers,
            hidden_states,
            mask,
            output_hidden_states,
            key=k_layers,
        )

        last_hidden_state = self.post_layernorm(hidden_states)

        return Siglip2VisionModelOutput(last_hidden_state=last_hidden_state, hidden_states=result_hidden_states)


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
    mlp: VisionMLP
    inference: bool  # Whether to run in inference mode (disables dropout)

    @staticmethod
    def init(config: Siglip2VisionConfig, *, key: PRNGKeyArray) -> "Siglip2MultiheadAttentionPoolingHead":
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

        layernorm = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_eps, use_bias=True)

        mlp = VisionMLP.init(config.Embed, config.Mlp, config.hidden_act, key=k_mlp)

        return Siglip2MultiheadAttentionPoolingHead(
            config=config,
            probe=probe,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            out_proj=out_proj,
            layernorm=layernorm,
            mlp=mlp,
            inference=config.inference,
        )

    @named_call
    def __call__(
        self,
        hidden_states: NamedArray,
        mask: Optional[AttentionMask] = None,
        *,
        key: PRNGKeyArray | None = None,
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
        k_drop = maybe_rng_split(key, 1)[0] if key is not None else None

        # Project probe to Q
        q = self.q_proj(self.probe)  # (probe_seq, heads, head_size)

        # Project hidden states to K, V
        k = self.k_proj(hidden_states)  # (..., num_patches, heads, head_size)
        v = self.v_proj(hidden_states)  # (..., num_patches, heads, head_size)

        # Broadcast q to match batch dimensions of k and v
        batch_axes = [ax for ax in k.axes if ax.name not in ["num_patches", "heads", "head_size"]]
        for ax in batch_axes:
            q = hax.broadcast_to(q, (ax,) + q.axes)

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
            inference=self.inference,
            use_flash=self.config.use_flash_attention,
            attn_backend=self.config.attn_backend,
            flash_block_size=self.config.flash_attention_block_size,
            dropout=self.config.attention_dropout,
            prng=k_drop,
        )

        # Project back to embed dimension
        attn_output = self.out_proj(attn_output.astype(hidden_states.dtype))  # (..., probe_seq, embed)

        # Residual connection with probe (broadcast probe to batch dims)
        hidden_states = self.probe + attn_output

        # Squeeze probe_seq dimension to get (..., embed)
        ProbeSeq = hidden_states.resolve_axis("probe_seq")
        hidden_states = hidden_states[ProbeSeq, 0]  # Remove probe_seq dim

        # Layer norm + MLP with residual
        residual = hidden_states
        hidden_states = self.layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

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

        ln_dict = eqx_to_state_dict(self.layernorm, with_prefix(prefix, "layernorm"))
        state_dict.update(ln_dict)

        mlp_dict = eqx_to_state_dict(self.mlp, with_prefix(prefix, "mlp"))
        state_dict.update(mlp_dict)

        return state_dict

    def from_state_dict(self, state_dict: Dict[str, jnp.ndarray], prefix: Optional[str] = None) -> Self:
        """Load from HuggingFace state dict format with combined in_proj."""

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
            inference=self.inference,
        )


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

    def get_hf_config(self) -> HfConfig:
        """Return the Hugging Face vision configuration."""
        return self.config.to_hf_config()

    @classmethod
    def init(cls, Vocab: Axis, config: Siglip2VisionConfig, *, key: PRNGKeyArray) -> "Siglip2VisionModel":
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
        spatial_shapes: np.ndarray | None = None,
        output_hidden_states: bool = False,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Siglip2VisionModelOutput:
        """
        Forward pass through vision model.

        Args:
            pixel_values: Patchified pixel values with shape (..., num_patches, patch_input_dim)
            mask: Optional attention mask
            spatial_shapes: Optional array of shape (batch, 2) containing [height, width] in patches
            output_hidden_states: Whether to return hidden states from all layers
            key: PRNGKey for dropout

        Returns:
            Siglip2VisionModelOutput containing last_hidden_state and optionally hidden_states
        """
        return self.vision_model(
            pixel_values, mask=mask, spatial_shapes=spatial_shapes, output_hidden_states=output_hidden_states, key=key
        )

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        """Map Levanter field names to HuggingFace state dict keys."""
        return {}  # Keep vision_model prefix as-is (matches HF structure)

    def from_state_dict(self, state_dict: Dict[str, jnp.ndarray], prefix: Optional[str] = None) -> Self:
        """Load from state dict."""

        # Standalone HF vision checkpoints omit the composite model's vision_model prefix.
        if with_prefix(prefix, "embeddings.patch_embedding.weight") in state_dict:
            vision_model = default_eqx_module_from_state_dict(self.vision_model, state_dict, prefix)
            return dataclasses.replace(self, vision_model=vision_model)
        return default_eqx_module_from_state_dict(self, state_dict, prefix)
