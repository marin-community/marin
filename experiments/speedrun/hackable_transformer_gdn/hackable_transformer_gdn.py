# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Hackable transformer training speedrun sweep

This file is intentionally self-contained:
- Defines a compact, Llama-ish transformer that implements Levanter's LmHeadModel
- Provides a ready-to-run speedrun sweep across multiple model sizes

(this example allows comparing using / not using gpt-oss style attention sink)

How to run (GPU or TPU):
  1) Set env vars (WANDB_API_KEY, HF_TOKEN, etc.) as in the tutorial:
     https://marin.readthedocs.io/en/latest/tutorials/submitting-speedrun/
  2) From repo root:
       python marin/run/ray_run.py -- \
         python -m experiments.speedrun.hackable_transformer_gdn.hackable_transformer_gdn
  3) Optional: SR_USE_GPU=1 to use GPU resource presets.

The transformer is a pared-down version of levanter.models.llama; you can refer to it if you wish to
add back functionality (like inference, HF exports)

To edit this file for your speedrun:
  1) Copy and rename the file in your location under experiments.speedrun
  2) Make changes to the architecture or configurations
  3) Add your author information
  4) Submit (see "How to run" above)
"""

# nodryrun
import sys
import os

os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_scoped_vmem_limit_kib=26624"

import dataclasses
import logging
from dataclasses import dataclass
from collections.abc import Callable

import equinox as eqx
import jax
import jax.random as jrandom
from jax import debug as jdbg
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import BlockSeq, ScanCheckpointPolicy, Stacked
from haliax.state_dict import ModuleWithStateDictSerialization
from levanter.utils.types import BlockFoldable

from levanter.layers import RmsNormConfig, LayerNormConfigBase
from levanter.layers.attention import Attention, AttentionWithSink, AttentionConfig, AttentionMask, AttentionBackend
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.layers.gated_deltanet import GatedDeltaNet, GatedDeltaNetConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.logging import silence_transformer_nag

from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun
from marin.execution.executor import executor_main
from fray.cluster import ResourceConfig
from experiments.simple_train_config import SimpleTrainConfig

# Optional: Muon optimizer configs
from levanter.optim import MuonConfig
from experiments.llama import llama3_tokenizer_vocab_size

logger = logging.getLogger("ray")

_IMPORT_PATH = getattr(__spec__, "name", __name__)

silence_transformer_nag()

# =========================
# Hackable config & modules
# =========================


_GDN_DEBUG = bool(int(os.environ.get("GDN_DEBUG_SHARDING", "0")))


def _dbg_sharding(tag: str, arr):
    """Compile-time sharding and runtime shape/dtype logger (hackable wrapper)."""
    try:
        jax.debug.inspect_array_sharding(arr, callback=lambda s: print(f"[GDN][hackable][sharding] {tag}: {s}"))
    except Exception:
        pass
    jdbg.print("[GDN][hackable][array] {tag} shape={} dtype={}", arr.shape, arr.dtype, tag=tag)


@LmConfig.register_subclass("hackable_transformer")
@dataclass(frozen=True)
class HackableTransformerConfig(LmConfig["HackableLMHeadModel"]):
    # Core dims
    max_seq_len: int = 2048
    hidden_dim: int = 4096
    intermediate_dim: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    head_dim: int | None = None

    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.silu
    use_bias: bool = False
    use_layer_norm_weight: bool = True
    layer_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = False
    input_embedding_norm: bool = False

    # Attention
    use_attention_sink: bool = False
    upcast_attn: bool = False
    attn_backend: AttentionBackend | None = None
    flash_attention_block_size: int | None = None
    rope: RotaryEmbeddingsConfig = dataclasses.field(default_factory=DefaultRotaryEmbeddingsConfig)
    qk_norm: LayerNormConfigBase | None = None  # set to RmsNormConfig(...) to enable

    # Gated DeltaNet mixing
    use_gated_deltanet: bool = True
    gdn_layers_per_block: int = 3
    gdn_block_size: int = 4
    gdn_conv_kernel_size: int = 4
    gdn_chunk_size: int = 128
    gdn_segment_size: int = 16

    gradient_checkpointing: bool | ScanCheckpointPolicy | str = True
    initializer_range: float = 0.02
    reference_checkpoint: str = "NousResearch/Llama-2-7b-hf"
    tokenizer: str | None = None

    def __post_init__(self):
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        if self.head_dim is None:
            assert self.hidden_dim % self.num_heads == 0, "hidden_dim % num_heads must be 0 when head_dim=None"

    # ---- LmConfig API ----
    @property
    def model_type(self) -> type["HackableLMHeadModel"]:
        return HackableLMHeadModel

    Pos = property(lambda self: Axis("position", self.max_seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis("embed", self.hidden_dim))
    Layers = property(lambda self: Axis("layers", self.num_layers))
    Mlp = property(lambda self: Axis("mlp", self.intermediate_dim))

    @property
    def norm_config(self) -> LayerNormConfigBase:
        return RmsNormConfig(use_weight=self.use_layer_norm_weight, use_bias=self.use_bias, eps=self.layer_norm_epsilon)

    def mk_LayerNorm(self, axis: AxisSpec):
        return self.norm_config.build(axis)

    def attention_config(self) -> AttentionConfig:
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            use_bias=self.use_bias,
            upcast_attn=self.upcast_attn,
            attn_backend=self.attn_backend,
            flash_attention_block_size=self.flash_attention_block_size,
            rope=self.rope,
            qk_norm=self.qk_norm,
        )

    @property
    def actual_head_size(self) -> int:
        return self.head_dim or (self.hidden_dim // self.num_heads)

    def layer_uses_gdn(self, layer_index: int) -> bool:
        if not self.use_gated_deltanet:
            return False
        if self.gdn_block_size <= 0:
            return False
        layers_per_block = max(0, min(self.gdn_layers_per_block, self.gdn_block_size))
        return (layer_index % self.gdn_block_size) < layers_per_block

    @property
    def num_gdn_layers(self) -> int:
        return sum(1 for i in range(self.num_layers) if self.layer_uses_gdn(i))

    @property
    def num_attention_layers(self) -> int:
        return self.num_layers - self.num_gdn_layers

    def gated_deltanet_config(self) -> GatedDeltaNetConfig:
        head_dim = self.actual_head_size
        return GatedDeltaNetConfig(
            Embed=self.Embed,
            num_k_heads=self.num_kv_heads,
            num_v_heads=self.num_heads,
            head_k_dim=head_dim,
            head_v_dim=head_dim,
            conv_kernel_size=self.gdn_conv_kernel_size,
            rms_norm_eps=self.layer_norm_epsilon,
        )

    def flops_per_token(self, vocab_size: int, context_length: int) -> float | None:
        head_dim = self.actual_head_size
        mlp = 2 * 3 * self.hidden_dim * self.intermediate_dim
        # Standard attention per-layer flops
        qkv_proj = 2 * self.hidden_dim * (self.num_heads * head_dim + 2 * self.num_kv_heads * head_dim)
        dense_proj = 2 * self.hidden_dim * self.hidden_dim
        seq_flops = 2 * (context_length**2) * self.num_heads * head_dim
        seq_flops += 3 * (context_length**2) * self.num_heads
        seq_flops += 2 * (context_length**2) * head_dim * self.num_heads
        attn_per_layer = qkv_proj + dense_proj + seq_flops / context_length

        # Approximate GDN flops per layer
        gdn_cfg = self.gated_deltanet_config()
        key_dim = gdn_cfg.key_dim
        value_dim = gdn_cfg.value_dim
        qkvz_proj = 2 * self.hidden_dim * (2 * key_dim + 2 * value_dim)
        ba_proj = 2 * self.hidden_dim * (2 * self.num_heads)
        conv_channels = 2 * key_dim + value_dim
        conv = 2 * conv_channels * gdn_cfg.conv_kernel_size
        # Approximate kernel cost: per head matrix-vector updates
        kernel = 6 * self.num_heads * gdn_cfg.head_k_dim * gdn_cfg.head_v_dim
        gdn_out = 2 * self.hidden_dim * value_dim
        gdn_per_layer = qkvz_proj + ba_proj + conv + kernel + gdn_out

        total = self.num_layers * mlp
        total += self.num_attention_layers * attn_per_layer
        total += self.num_gdn_layers * gdn_per_layer
        total += 2 * self.hidden_dim * vocab_size
        return float(total)

    def total_trainable_params(self, vocab_size: int) -> int:
        token_embedding = vocab_size * self.hidden_dim
        hs = self.actual_head_size
        attn = (
            self.hidden_dim * hs * self.num_heads
            + 2 * self.hidden_dim * hs * self.num_kv_heads
            + hs * self.num_heads * self.hidden_dim
        )
        gdn_cfg = self.gated_deltanet_config()
        key_dim = gdn_cfg.key_dim
        value_dim = gdn_cfg.value_dim
        gdn = (
            self.hidden_dim * (2 * key_dim + 2 * value_dim)
            + self.hidden_dim * (2 * self.num_heads)
            + (2 * key_dim + value_dim) * gdn_cfg.conv_kernel_size
            + 2 * self.num_heads  # A_log and dt_bias
            + gdn_cfg.head_v_dim
            + self.hidden_dim * value_dim
        )
        mlp = 3 * self.hidden_dim * self.intermediate_dim
        transformer = (
            self.num_attention_layers * (attn + mlp + 2 * self.hidden_dim)
            + self.num_gdn_layers * (gdn + mlp + 2 * self.hidden_dim)
            + self.hidden_dim
        )
        if self.input_embedding_norm:
            transformer += self.hidden_dim
        head = 0 if self.tie_word_embeddings else token_embedding
        return int(transformer + token_embedding + head)


def _prepare_gdn_mask(
    mask: NamedArray | AttentionMask | None,
    q_axis: Axis,
    k_axis: Axis,
) -> NamedArray | None:
    """Extract a simple padding mask for the GDN layer.

    The current implementation supports NamedArray masks directly and attempts to reuse
    explicit masks from AttentionMask objects. More structured masks (e.g. causal-only)
    return ``None`` because GDN is inherently causal.
    """

    if mask is None:
        return None

    if isinstance(mask, NamedArray):
        return mask

    if isinstance(mask, AttentionMask):
        explicit = mask.explicit_mask
        if explicit is None:
            return None
        axis_names = {ax.name for ax in explicit.axes}
        if q_axis.name in axis_names:
            return explicit
        if k_axis.name in axis_names:
            return explicit.rename({k_axis.name: q_axis.name})
    return None


class HackableMlp(eqx.Module):
    """GLU MLP"""

    gate_proj: hnn.Linear
    up_proj: hnn.Linear
    down_proj: hnn.Linear
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(Embed: AxisSpec, Mlp: AxisSpec, activation_fn: ActivationFunctionEnum | Callable, *, key, use_bias=False):
        k_fc, k_up_proj, k_down_proj = jrandom.split(key, 3)
        gate_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_fc, use_bias=use_bias, out_first=True)
        up_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_up_proj, use_bias=use_bias, out_first=True)
        down_proj = hnn.Linear.init(Out=Embed, In=Mlp, key=k_down_proj, use_bias=use_bias, out_first=True)
        if isinstance(activation_fn, ActivationFunctionEnum):
            activation_fn = activation_fn.to_fn()
        elif isinstance(activation_fn, str):
            activation_fn = ActivationFunctionEnum(activation_fn).to_fn()
        return HackableMlp(gate_proj, up_proj, down_proj, activation_fn)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k_gate, k_up, k_down = maybe_rng_split(key, 3)
        h = self.act(self.gate_proj(x, key=k_gate)) * self.up_proj(x, key=k_up)
        return self.down_proj(h, key=k_down)


class HackableDecoderLayer(eqx.Module):
    """One transformer block."""

    config: HackableTransformerConfig = eqx.field(static=True)
    self_attn: Attention | AttentionWithSink | None
    gdn: GatedDeltaNet | None
    mlp: HackableMlp
    input_layernorm: hnn.RmsNorm
    post_attention_layernorm: hnn.RmsNorm
    post_attn_layernorm: hnn.RmsNorm | None = None
    post_mlp_layernorm: hnn.RmsNorm | None = None
    use_gdn: bool = eqx.field(static=True, default=False)
    gdn_chunk_size: int = eqx.field(static=True, default=64)
    gdn_segment_size: int = eqx.field(static=True, default=8)

    @staticmethod
    def init(config: HackableTransformerConfig, *, key, layer_index: int) -> "HackableDecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)
        use_gdn = config.layer_uses_gdn(layer_index)
        attn: Attention | AttentionWithSink | None
        gdn: GatedDeltaNet | None
        if use_gdn:
            gdn_cfg = config.gated_deltanet_config()
            gdn = GatedDeltaNet.init(gdn_cfg, key=k_attn)
            attn = None
        else:
            attn_cfg = config.attention_config()
            attn = (
                AttentionWithSink.init(attn_cfg, key=k_attn)
                if config.use_attention_sink
                else Attention.init(attn_cfg, key=k_attn)
            )
            gdn = None
        mlp = HackableMlp.init(config.Embed, config.Mlp, config.activation_function, key=k_mlp, use_bias=config.use_bias)
        ln1 = config.mk_LayerNorm(config.Embed)
        ln2 = config.mk_LayerNorm(config.Embed)
        return HackableDecoderLayer(
            config,
            attn,
            gdn,
            mlp,
            ln1,
            ln2,
            use_gdn=use_gdn,
            gdn_chunk_size=config.gdn_chunk_size,
            gdn_segment_size=config.gdn_segment_size,
        )

    @named_call
    def __call__(
        self, x: NamedArray, mask: NamedArray | AttentionMask | None, *, key=None, pos_ids: NamedArray | None = None
    ):
        k_attn, k_mlp = maybe_rng_split(key, 2)
        # self attention and skip connection
        residual = x
        x = self.input_layernorm(x)
        if self.use_gdn:
            attn_output = self._apply_gdn(x, mask)
        else:
            assert self.self_attn is not None
            attn_output = self.self_attn(x=x, mask=mask, key=k_attn, pos_ids=pos_ids)
        if self.post_attn_layernorm is not None:
            attn_output = self.post_attn_layernorm(attn_output)
        x = residual + attn_output

        # MLP and skip connection
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x, key=k_mlp)
        if self.post_mlp_layernorm is not None:
            mlp_output = self.post_mlp_layernorm(mlp_output)
        output = residual + mlp_output
        return output

    def _apply_gdn(self, x: NamedArray, mask: NamedArray | AttentionMask | None) -> NamedArray:
        assert self.gdn is not None
        attn_mask = _prepare_gdn_mask(mask, self.config.Pos, self.config.KeyPos)
        if _GDN_DEBUG:
            try:
                _dbg_sharding("layer_in/x", x.array if hasattr(x, "array") else x)
            except Exception:
                pass
        y, _ = self.gdn(
            x,
            inference=False,
            chunk_size=self.gdn_chunk_size,
            attention_mask=attn_mask,
            decode_state=None,
        )
        if _GDN_DEBUG:
            try:
                _dbg_sharding("layer_out/y", y.array if hasattr(y, "array") else y)
            except Exception:
                pass
        return y


class HackableTransformer(eqx.Module):
    config: HackableTransformerConfig = eqx.field(static=True)
    layers: BlockFoldable[HackableDecoderLayer]
    norm: hnn.RmsNorm

    @staticmethod
    def init(config: HackableTransformerConfig, *, key):
        checkpoint_policy = ScanCheckpointPolicy._mk(config.gradient_checkpointing)
        if config.use_gated_deltanet and config.num_gdn_layers > 0:
            layer_keys = tuple(jrandom.split(key, config.num_layers))
            blocks = tuple(
                HackableDecoderLayer.init(config, key=subkey, layer_index=i) for i, subkey in enumerate(layer_keys)
            )
            layers = BlockSeq(blocks, config.Layers, checkpoint_policy)
        else:
            S = Stacked
            layers = S.init(config.Layers, HackableDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
                config, key=shaped_rng_split(key, config.num_layers)
            )
        return HackableTransformer(config, layers, config.mk_LayerNorm(config.Embed))

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: NamedArray | AttentionMask | None, *, key=None, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids)
        return self.norm(x)


class HackableEmbedding(ModuleWithStateDictSerialization, eqx.Module):
    token_embeddings: hnn.Embedding
    norm: hnn.RmsNorm | None = None

    @staticmethod
    def init(Vocab: Axis, config: HackableTransformerConfig, *, key):
        emb = hnn.Embedding.init(Vocab, config.Embed, key=key)
        ln = config.mk_LayerNorm(config.Embed) if config.input_embedding_norm else None
        return HackableEmbedding(emb, ln)

    @property
    def Vocab(self) -> Axis:
        return self.token_embeddings.Vocab

    @named_call
    def embed(self, input_ids: NamedArray):
        x = self.token_embeddings(input_ids)
        return self.norm(x) if self.norm is not None else x


class HackableLMHeadModel(
    ModuleWithStateDictSerialization,
    LmHeadModel[HackableTransformerConfig],
):
    """Minimal Llama-like implementation of LmHeadModel"""

    transformer: HackableTransformer
    embeddings: HackableEmbedding
    lm_head: hnn.Linear | None

    @property
    def config(self) -> HackableTransformerConfig:
        return self.transformer.config

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @classmethod
    def init(cls, Vocab: Axis, config: HackableTransformerConfig, *, key) -> "HackableLMHeadModel":
        k_t, k_e = jrandom.split(key, 2)
        transformer = HackableTransformer.init(config, key=k_t)
        embeddings = HackableEmbedding.init(Vocab, config, key=k_e)
        lm_head = (
            None
            if config.tie_word_embeddings
            else hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_e, use_bias=False, out_first=True)
        )
        return HackableLMHeadModel(transformer, embeddings, lm_head)

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | NamedArray | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        return self.transformer(self.embeddings.embed(input_ids), attn_mask=attn_mask, key=key, pos_ids=pos_ids)

    def get_lm_head(self) -> hax.NamedArray:
        return self.embeddings.token_embeddings.weight if self.lm_head is None else self.lm_head.weight

    def resize_vocab(self, new_size: int, key: PRNGKeyArray | None = None) -> "HackableLMHeadModel":
        pass


# =========================
# Speedrun sweep definition
# =========================

AUTHOR = Author(name="Calvin Xu", affiliation="Stanford University", url="https://pinlinxu.com")  # TODO: update me


def _get_num_train_steps(param_count: int, batch_size: int, seq_len: int, tpp: int = 20) -> int:
    total_tokens = param_count * tpp
    return max(1, total_tokens // (batch_size * seq_len))


def _size_presets() -> dict[str, HackableTransformerConfig]:
    base = dict(
        max_seq_len=2048,
        rope=DefaultRotaryEmbeddingsConfig(),  # e.g., Llama3RotaryEmbeddingsConfig()
        attn_backend=None,
        qk_norm=None,  # e.g. RmsNormConfig(use_weight=True, eps=1e-5)
        tie_word_embeddings=False,
    )
    return {
        "130m": HackableTransformerConfig(
            hidden_dim=512, intermediate_dim=1792, num_layers=6, num_heads=8, num_kv_heads=8, **base
        ),
        "300m": HackableTransformerConfig(
            hidden_dim=768, intermediate_dim=2688, num_layers=12, num_heads=12, num_kv_heads=12, **base
        ),
        "520m": HackableTransformerConfig(
            hidden_dim=1024, intermediate_dim=3584, num_layers=24, num_heads=16, num_kv_heads=8, **base
        ),
        "1_2b": HackableTransformerConfig(
            hidden_dim=2048, intermediate_dim=7168, num_layers=16, num_heads=16, num_kv_heads=8, **base
        ),
    }


def _muon_presets() -> dict[str, MuonConfig]:
    # LRs sqrt-scaled from batch 128 baseline to current batch sizes
    return {
        "130m": MuonConfig(  # batch 132, sqrt(132/128) = 1.015
            learning_rate=0.0162,
            adam_lr=0.0032,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.95,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            lr_schedule="linear",
            decay=0.8,
        ),
        "300m": MuonConfig(  # batch 92, sqrt(92/128) = 0.848
            learning_rate=0.0068,
            adam_lr=0.0020,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.98,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            lr_schedule="linear",
            decay=0.8,
        ),
        "520m": MuonConfig(  # batch 52, sqrt(52/128) = 0.637
            learning_rate=0.0051,
            adam_lr=0.0015,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.98,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-25,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            lr_schedule="linear",
            decay=1,
        ),
        "1_2b": MuonConfig(  # batch 44, sqrt(44/128) = 0.586
            learning_rate=0.0023,
            adam_lr=0.0007,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.98,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=2,
            lr_schedule="linear",
            decay=1,
        ),
    }


def _resource_presets(use_gpu: bool = False):
    if use_gpu:
        return {
            "130m": ResourceConfig.with_gpu("A100-80G", count=1),
            "300m": ResourceConfig.with_gpu("A100-80G", count=1),
            "520m": ResourceConfig.with_gpu("A100-80G", count=2),
            "1_2b": ResourceConfig.with_gpu("A100-80G", count=4),
        }

    return {
        "130m": ResourceConfig.with_tpu("v5p-8"),
        "300m": ResourceConfig.with_tpu("v5p-8"),
        "520m": ResourceConfig.with_tpu("v5p-8"),
        "1_2b": ResourceConfig.with_tpu("v5p-8"),
    }


def _batch_sizes() -> dict[str, int]:
    return {"130m": 132, "300m": 92, "520m": 52, "1_2b": 44}


def build_run(size: str, *, use_gpu: bool = False) -> tuple[str, SpeedrunConfig]:
    sizes = _size_presets()
    if size not in sizes:
        raise ValueError(f"Unknown size: {size}")
    model_cfg = sizes[size]

    batch = _batch_sizes()[size]
    seq_len = model_cfg.max_seq_len
    params = int(model_cfg.total_trainable_params(llama3_tokenizer_vocab_size))
    print(params)
    steps = _get_num_train_steps(params, batch, seq_len, tpp=20)

    muon = _muon_presets()[size]
    resources = _resource_presets(use_gpu=use_gpu)[size]

    train = SimpleTrainConfig(
        resources,
        train_batch_size=batch,
        num_train_steps=steps,
        learning_rate=muon.learning_rate,
        optimizer_config=muon,
        steps_per_hf_export=-1,  # disable checkpointing
        profiler=True,
    )

    run_name = f"hacktx_{size}_gdn_{seq_len}_flash_ch_{model_cfg.gdn_chunk_size}_seg_{model_cfg.gdn_segment_size}_fp32_b_{batch}_lr_scaled"
    desc = f"Hackable Transformer ({size}) w/ hybrid Gated DeltaNet and standard attention layers (Muon)"
    cfg = SpeedrunConfig(author=AUTHOR, description=desc, model_config=model_cfg, train_config=train)
    return run_name, cfg


if __name__ == "__main__":
    ###
    # make the current __main__ module importable under its canonical name
    sys.modules[_IMPORT_PATH] = sys.modules[__name__]
    # allow the workers to import the classes
    for _cls in (
        HackableTransformerConfig,
        HackableMlp,
        HackableDecoderLayer,
        HackableTransformer,
        HackableEmbedding,
        HackableLMHeadModel,
    ):
        _cls.__module__ = _IMPORT_PATH
    ###

    # sizes = ["130m"]
    # sizes = ["1_2b"]
    sizes = ["300m", "520m", "1_2b"]
    # sizes = ["130m", "300m", "520m", "1_2b"]
    use_gpu = bool(int(os.environ.get("SR_USE_GPU", "0")))
    sink = False
    steps = []
    for s in sizes:
        name, cfg = build_run(s, use_gpu=use_gpu)
        steps.extend(default_speedrun(name, cfg))
    executor_main(steps=steps, description="Hackable transformer GDN sweep")
