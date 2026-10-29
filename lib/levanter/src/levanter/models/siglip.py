# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0


from haliax._src.state_dict import default_eqx_module_from_state_dict


from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional, Protocol, Self, Tuple, Type

from jaxtyping import PRNGKeyArray

from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.logging import silence_transformer_nag
from levanter.models.vision_model import VisionEncoderConfig


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
from haliax.state_dict import (  # noqa: E402
    ModuleWithStateDictSerialization,
    from_state_dict,
    to_state_dict as eqx_to_state_dict,
    with_prefix,
)

from levanter.compat.hf_checkpoints import (  # noqa: E402
    HfVisionCheckpointConverter,
    VisionModelWithHfSerializationMixin,
)
from levanter.layers.attention import (  # noqa: E402
    Attention,
    AttentionBackend,
    AttentionConfig,
    AttentionMask,
    dot_product_attention,
)

DEFAULT_SIGLIP_INITIALIZER_RANGE = 0.02
HF_SIGLIP_GELU_ACTIVATION = "gelu_pytorch_tanh"
ATTENTION_IN_PROJ_WEIGHT_KEY = "attention.in_proj_weight"
ATTENTION_IN_PROJ_BIAS_KEY = "attention.in_proj_bias"


@dataclass
class SiglipVisionModelOutput:
    """
    Output class for SigLIP Vision Model, similar to HuggingFace's output format.

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
    initializer_range: float = DEFAULT_SIGLIP_INITIALIZER_RANGE
    gradient_checkpointing: bool = True
    vision_use_head: bool = True

    use_flash_attention: bool = True
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None
    inference: bool = True  # Whether to run in inference mode (disables dropout)

    reference_checkpoint: Optional[str] = None

    def hf_checkpoint_converter(
        self, ref_checkpoint: Optional[str] = None
    ) -> HfVisionCheckpointConverter["SiglipVisionConfig"]:
        """Create HuggingFace checkpoint converter for this config."""
        return HfVisionCheckpointConverter(
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
        if isinstance(hf_config, HfSiglipConfig):
            vision_config = hf_config.vision_config
            assert isinstance(vision_config, HfSiglipVisionConfig)
            hf_config = vision_config
        assert isinstance(hf_config, HfSiglipVisionConfig)

        activation_fn = siglip_activation(hf_config.hidden_act)

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
            initializer_range=getattr(hf_config, "initializer_range", DEFAULT_SIGLIP_INITIALIZER_RANGE),
            vision_use_head=getattr(hf_config, "vision_use_head", True),
        )

    def to_hf_config(self, config_overrides: Optional[Dict] = None) -> HfSiglipVisionConfig:
        """Convert from Levanter config to HuggingFace config.

        Args:
            config_overrides: Optional config overrides
        """
        if config_overrides is None:
            config_overrides = {}

        # gelu_new in Levanter maps back to gelu_pytorch_tanh in HF (for SigLIP compatibility)
        if isinstance(self.hidden_act, ActivationFunctionEnum):
            if self.hidden_act == ActivationFunctionEnum.gelu_new:
                hf_hidden_act = HF_SIGLIP_GELU_ACTIVATION
            else:
                hf_hidden_act = self.hidden_act.value
        else:
            hf_hidden_act = self.hidden_act

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
            "initializer_range": self.initializer_range,
            "vision_use_head": self.vision_use_head,
        }

        config_dict.update(config_overrides)

        hf_config = HfSiglipVisionConfig.from_dict(config_dict)

        return hf_config

    @property
    def model_type(self) -> Type["SiglipVisionModel"]:
        """Return the corresponding model class."""
        return SiglipVisionModel

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

    def attention_config(self) -> AttentionConfig:
        """Build the standard Levanter attention configuration for SigLIP."""
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


class SiglipBlockConfig(Protocol):
    @property
    def Embed(self) -> Axis: ...

    @property
    def Mlp(self) -> Axis: ...

    @property
    def Layers(self) -> Axis: ...

    @property
    def gradient_checkpointing(self) -> bool: ...

    @property
    def hidden_act(self) -> ActivationFunctionEnum: ...

    @property
    def layer_norm_eps(self) -> float: ...

    @property
    def attention_dropout(self) -> float: ...

    @property
    def initializer_range(self) -> float: ...

    @property
    def inference(self) -> bool: ...

    def attention_config(self) -> AttentionConfig: ...


def siglip_activation(name: str) -> ActivationFunctionEnum:
    aliases = {HF_SIGLIP_GELU_ACTIVATION: "gelu_new", "swish": "silu"}
    return ActivationFunctionEnum(aliases.get(name, name))


class SiglipMLP(eqx.Module):
    """Two-layer feedforward block shared by SigLIP and SigLIP2."""

    fc1: hnn.Linear
    fc2: hnn.Linear
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(Embed: Axis, Mlp: Axis, activation_fn: ActivationFunctionEnum, *, key: PRNGKeyArray) -> "SiglipMLP":
        k_fc1, k_fc2 = maybe_rng_split(key, 2)
        fc1 = hnn.Linear.init(In=Embed, Out=Mlp, key=k_fc1, use_bias=True, out_first=True)
        fc2 = hnn.Linear.init(In=Mlp, Out=Embed, key=k_fc2, use_bias=True, out_first=True)
        return SiglipMLP(fc1, fc2, activation_fn.to_fn())

    @named_call
    def __call__(self, x: NamedArray) -> NamedArray:
        return self.fc2(self.act(self.fc1(x)))


class SiglipAttention(ModuleWithStateDictSerialization, Attention):
    """SigLIP axis and dropout semantics on Levanter's standard attention layer."""

    attention_dropout: float = eqx.field(static=True, default=0.0)
    inference: bool = eqx.field(static=True, default=True)

    @staticmethod
    def init(config: SiglipBlockConfig, *, key: PRNGKeyArray) -> "SiglipAttention":  # type: ignore[override]
        attention = Attention.init(config.attention_config(), key=key)
        return SiglipAttention(
            config=attention.config,
            q_proj=attention.q_proj,
            k_proj=attention.k_proj,
            v_proj=attention.v_proj,
            o_proj=attention.o_proj,
            q_norm=attention.q_norm,
            k_norm=attention.k_norm,
            rot_embs=attention.rot_embs,
            attention_dropout=config.attention_dropout,
            inference=config.inference,
        )

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"o_proj": "out_proj"}

    @named_call
    def __call__(
        self,
        x: NamedArray,
        mask: Optional[AttentionMask] = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> NamedArray:
        patch_axis = x.resolve_axis("num_patches")
        position_axis = Axis("position", patch_axis.size)
        key_position_axis = Axis("key_position", patch_axis.size)

        if mask is not None:
            explicit_mask = mask.materialize(patch_axis, key_position_axis)
            mask = (
                None
                if explicit_mask is None
                else AttentionMask.explicit(explicit_mask.rename({patch_axis: position_axis}))
            )

        x = x.rename({patch_axis: position_axis})
        key_proj, key_out = maybe_rng_split(key, 2)
        query, key_states, value = self._compute_qkv(x, key=key_proj)
        query = query.rearrange((..., "kv_head", "q_heads_per_group", "position", "head_size"))
        key_states = key_states.rearrange((..., "kv_head", "position", "head_size")).rename(
            {"position": "key_position"}
        )
        value = value.rearrange((..., "kv_head", "position", "head_size")).rename({"position": "key_position"})
        attention_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            query,
            key_states,
            value,
            mask,
            attention_dtype=x.dtype,
            attn_backend=self.config.attn_backend,
            flash_block_size=self.config.flash_attention_block_size,
            dropout=self.attention_dropout,
            inference=self.inference,
            prng=key,
        )
        attention_output = attention_output.flatten_axes(("kv_head", "q_heads_per_group"), "heads")
        attention_output = self.o_proj(attention_output.astype(x.dtype), key=key_out)
        return attention_output.rename({position_axis: patch_axis})


class SiglipEncoderLayer(eqx.Module):
    """Pre-normalized SigLIP attention and feedforward residual blocks."""

    config: SiglipBlockConfig = eqx.field(static=True)
    layer_norm1: hnn.LayerNorm
    self_attn: SiglipAttention
    layer_norm2: hnn.LayerNorm
    mlp: SiglipMLP

    @staticmethod
    def init(config: SiglipBlockConfig, *, key: PRNGKeyArray) -> "SiglipEncoderLayer":
        k_attn, k_mlp = maybe_rng_split(key, 2)
        return SiglipEncoderLayer(
            config=config,
            layer_norm1=hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_eps, use_bias=True),
            self_attn=SiglipAttention.init(config, key=k_attn),
            layer_norm2=hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_eps, use_bias=True),
            mlp=SiglipMLP.init(config.Embed, config.Mlp, config.hidden_act, key=k_mlp),
        )

    @named_call
    def __call__(
        self,
        x: NamedArray,
        mask: Optional[AttentionMask] = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> NamedArray:
        key_attention, _ = maybe_rng_split(key, 2)
        x = x + self.self_attn(self.layer_norm1(x), mask=mask, key=key_attention)
        return x + self.mlp(self.layer_norm2(x))


def siglip_encoder_layers(config: SiglipBlockConfig, *, key: PRNGKeyArray) -> Stacked[SiglipEncoderLayer]:
    return Stacked.init(
        config.Layers,
        SiglipEncoderLayer,
        gradient_checkpointing=config.gradient_checkpointing,
    )(config, key=shaped_rng_split(key, config.Layers.size))


def siglip_hidden_states(
    layers: Stacked[SiglipEncoderLayer],
    hidden_states: NamedArray,
    mask: Optional[AttentionMask],
    *,
    key: Optional[PRNGKeyArray],
) -> tuple[NamedArray, tuple[NamedArray, ...]]:
    """Return final encoder features and every pre-normalization layer state."""
    keys = maybe_rng_split(key, layers.Block.size) if key is not None else None

    def apply_layer(
        block: SiglipEncoderLayer,
        carry: NamedArray,
        attention_mask: Optional[AttentionMask],
        layer_key: Optional[PRNGKeyArray],
    ) -> tuple[NamedArray, NamedArray]:
        output = block(carry, attention_mask, key=layer_key)
        return output, output

    final, stacked = layers.scan_via(apply_layer)(hidden_states, mask, keys)
    return final, (hidden_states, *hax.unbind(stacked, layers.Block))


class SiglipMultiheadAttentionPoolingHead(ModuleWithStateDictSerialization):
    """Multi-head attention pooling shared by SigLIP and SigLIP2."""

    config: SiglipBlockConfig = eqx.field(static=True)
    probe: NamedArray
    q_proj: hnn.Linear
    k_proj: hnn.Linear
    v_proj: hnn.Linear
    out_proj: hnn.Linear
    layernorm: hnn.LayerNorm
    mlp: SiglipMLP
    inference: bool

    @staticmethod
    def init(config: SiglipBlockConfig, *, key: PRNGKeyArray) -> "SiglipMultiheadAttentionPoolingHead":
        key_probe, key_q, key_k, key_v, key_out, key_mlp = maybe_rng_split(key, 6)
        probe_axis = Axis("probe_seq", 1)
        probe = hax.random.normal(key_probe, (probe_axis, config.Embed)) * config.initializer_range
        return SiglipMultiheadAttentionPoolingHead(
            config=config,
            probe=probe,
            q_proj=hnn.Linear.init(
                In=config.Embed,
                Out=(config.attention_config().Heads, config.attention_config().HeadSize),
                key=key_q,
                use_bias=True,
                out_first=True,
            ),
            k_proj=hnn.Linear.init(
                In=config.Embed,
                Out=(config.attention_config().Heads, config.attention_config().HeadSize),
                key=key_k,
                use_bias=True,
                out_first=True,
            ),
            v_proj=hnn.Linear.init(
                In=config.Embed,
                Out=(config.attention_config().Heads, config.attention_config().HeadSize),
                key=key_v,
                use_bias=True,
                out_first=True,
            ),
            out_proj=hnn.Linear.init(
                In=(config.attention_config().Heads, config.attention_config().HeadSize),
                Out=config.Embed,
                key=key_out,
                use_bias=True,
                out_first=True,
            ),
            layernorm=hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_eps, use_bias=True),
            mlp=SiglipMLP.init(config.Embed, config.Mlp, config.hidden_act, key=key_mlp),
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
        key_dropout = maybe_rng_split(key, 1)[0] if key is not None else None
        query = self.q_proj(self.probe)
        key_states = self.k_proj(hidden_states).rename({"num_patches": "key_position"})
        value = self.v_proj(hidden_states).rename({"num_patches": "key_position"})
        for axis in (axis for axis in key_states.axes if axis.name not in ("key_position", "heads", "head_size")):
            query = hax.broadcast_to(query, (axis,) + query.axes)
        attention_output = dot_product_attention(
            "probe_seq",
            "key_position",
            "head_size",
            query,
            key_states,
            value,
            mask=mask,
            inference=self.inference,
            attn_backend=self.config.attention_config().attn_backend,
            flash_block_size=self.config.attention_config().flash_attention_block_size,
            dropout=self.config.attention_dropout,
            prng=key_dropout,
        )
        hidden_states = self.out_proj(attention_output.astype(hidden_states.dtype))
        probe_axis = hidden_states.resolve_axis("probe_seq")
        hidden_states = hidden_states[probe_axis, 0]
        return hidden_states + self.mlp(self.layernorm(hidden_states))

    def to_state_dict(self, prefix: Optional[str] = None) -> Dict[str, jnp.ndarray]:
        state_dict = {with_prefix(prefix, "probe"): self.probe.array[None, ...]}
        weights = [
            projection.weight.array.reshape(self.config.Embed.size, self.config.Embed.size)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        ]
        state_dict[with_prefix(prefix, ATTENTION_IN_PROJ_WEIGHT_KEY)] = jnp.concatenate(weights, axis=0)
        if self.q_proj.bias is not None:
            biases = [projection.bias.array.reshape(-1) for projection in (self.q_proj, self.k_proj, self.v_proj)]
            state_dict[with_prefix(prefix, ATTENTION_IN_PROJ_BIAS_KEY)] = jnp.concatenate(biases, axis=0)
        state_dict.update(eqx_to_state_dict(self.out_proj, with_prefix(prefix, "attention.out_proj")))
        state_dict.update(eqx_to_state_dict(self.layernorm, with_prefix(prefix, "layernorm")))
        state_dict.update(eqx_to_state_dict(self.mlp, with_prefix(prefix, "mlp")))
        return state_dict

    def from_state_dict(self, state_dict: Dict[str, jnp.ndarray], prefix: Optional[str] = None) -> Self:
        probe_array = state_dict.get(with_prefix(prefix, "probe"), self.probe.array)
        if probe_array.ndim == 3:
            probe_array = probe_array.squeeze(0)
        probe = hax.named(probe_array, self.probe.axes)

        in_proj_weight = state_dict.get(with_prefix(prefix, ATTENTION_IN_PROJ_WEIGHT_KEY))
        if in_proj_weight is None:
            q_weight, k_weight, v_weight = self.q_proj.weight, self.k_proj.weight, self.v_proj.weight
        else:
            q_array, k_array, v_array = jnp.split(in_proj_weight, 3, axis=0)
            q_weight = hax.named(q_array, self.q_proj.weight.axes)
            k_weight = hax.named(k_array, self.k_proj.weight.axes)
            v_weight = hax.named(v_array, self.v_proj.weight.axes)

        in_proj_bias = state_dict.get(with_prefix(prefix, ATTENTION_IN_PROJ_BIAS_KEY))
        if in_proj_bias is None:
            q_bias, k_bias, v_bias = self.q_proj.bias, self.k_proj.bias, self.v_proj.bias
        else:
            q_array, k_array, v_array = jnp.split(in_proj_bias, 3, axis=0)
            assert self.q_proj.bias is not None and self.k_proj.bias is not None and self.v_proj.bias is not None
            q_bias = hax.named(q_array, self.q_proj.bias.axes)
            k_bias = hax.named(k_array, self.k_proj.bias.axes)
            v_bias = hax.named(v_array, self.v_proj.bias.axes)

        return replace(
            self,
            probe=probe,
            q_proj=replace(self.q_proj, weight=q_weight, bias=q_bias),
            k_proj=replace(self.k_proj, weight=k_weight, bias=k_bias),
            v_proj=replace(self.v_proj, weight=v_weight, bias=v_bias),
            out_proj=from_state_dict(self.out_proj, state_dict, with_prefix(prefix, "attention.out_proj")),
            layernorm=from_state_dict(self.layernorm, state_dict, with_prefix(prefix, "layernorm")),
            mlp=from_state_dict(self.mlp, state_dict, with_prefix(prefix, "mlp")),
        )


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
        patch_embeds = self.patch_embedding(pixel_values, key=key)

        # Flatten spatial dimensions (h_patches, w_patches) into num_patches
        # Conv output has spatial axes named after input spatial axes (height, width)
        # with sizes divided by stride (patch_size)
        embed_axis = self.config.Embed
        spatial_axes = [ax for ax in patch_embeds.axes if ax.name in ("height", "width")]

        num_patches_total = 1
        for ax in spatial_axes:
            num_patches_total *= ax.size
        NumPatchesActual = Axis("num_patches", num_patches_total)

        patch_embeds = hax.flatten_axes(patch_embeds, spatial_axes, NumPatchesActual)

        other_axes = [ax for ax in patch_embeds.axes if ax not in (embed_axis, NumPatchesActual)]
        patch_embeds = patch_embeds.rearrange((*other_axes, NumPatchesActual, embed_axis))

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
    layers: Stacked[SiglipEncoderLayer]
    post_layernorm: hnn.LayerNorm

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        """Map Levanter field names to HuggingFace state dict keys."""
        return {"layers": "encoder.layers"}

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

        layers = siglip_encoder_layers(config, key=k_layers)

        post_layernorm = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_eps, use_bias=True)

        return SiglipVisionTransformer(config, embeddings, layers, post_layernorm)

    @named_call
    def __call__(
        self,
        pixel_values: NamedArray,
        mask: Optional[AttentionMask] = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> SiglipVisionModelOutput:
        """
        Forward pass through vision transformer.

        Args:
            pixel_values: Input images with shape (batch, channels, height, width)
            mask: Optional attention mask
            key: PRNGKey for dropout

        Returns:
            SiglipVisionModelOutput containing:
                - last_hidden_state: Final encoded representations after post_layernorm (batch, num_patches, embed)
                - hidden_states: Embeddings followed by every encoder layer output.
        """
        k_embed, k_layers = maybe_rng_split(key, 2)

        hidden_states = self.embeddings(pixel_values, key=k_embed)

        hidden_states, result_hidden_states = siglip_hidden_states(
            self.layers,
            hidden_states,
            mask,
            key=k_layers,
        )

        last_hidden_state = self.post_layernorm(hidden_states)

        return SiglipVisionModelOutput(last_hidden_state=last_hidden_state, hidden_states=result_hidden_states)


class SiglipVisionModel(ModuleWithStateDictSerialization, VisionModelWithHfSerializationMixin[SiglipVisionConfig]):
    """
    SigLIP Vision Model with HuggingFace compatibility.

    This is a wrapper around SiglipVisionTransformer that implements
    vision checkpoint conversion.
    """

    vision_model: SiglipVisionTransformer
    head: Optional[SiglipMultiheadAttentionPoolingHead]

    @property
    def config(self) -> SiglipVisionConfig:
        return self.vision_model.config

    def get_hf_config(self) -> HfConfig:
        """Return the Hugging Face vision configuration."""
        return self.config.to_hf_config()

    @classmethod
    def init(cls, config: SiglipVisionConfig, *, key: PRNGKeyArray) -> "SiglipVisionModel":
        """
        Initialize SiglipVisionModel.

        Args:
            config: SiglipVisionConfig
            key: PRNGKey for initialization

        Returns:
            Initialized SiglipVisionModel
        """
        key_vision, key_head = maybe_rng_split(key, 2)
        vision_model = SiglipVisionTransformer.init(config, key=key_vision)
        head = SiglipMultiheadAttentionPoolingHead.init(config, key=key_head) if config.vision_use_head else None
        return cls(vision_model=vision_model, head=head)

    @named_call
    def __call__(
        self,
        pixel_values: NamedArray,
        mask: Optional[AttentionMask] = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> SiglipVisionModelOutput:
        """
        Forward pass through vision model.

        Args:
            pixel_values: Input images with shape (batch, channels, height, width)
            mask: Optional attention mask
            key: PRNGKey for dropout

        Returns:
            SiglipVisionModelOutput containing encoder and pooled features.
        """
        key_vision, key_head = maybe_rng_split(key, 2)
        output = self.vision_model(pixel_values, mask=mask, key=key_vision)
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
