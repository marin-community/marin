# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Type, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import shard_map

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import BlockFoldable, BlockSeq, Stacked
from haliax.state_dict import ModuleWithStateDictSerialization, StateDict

from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.compat.hf_config import hf_config_from_kwargs, hf_rope_config
from levanter.layers.attention import Attention, AttentionBackend, AttentionMask
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.models.llama import LlamaConfig, LlamaEmbedding
from levanter.models.lm_model import LmConfig, LmHeadModel, resize_embeddings_and_lm_head
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag


silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig  # noqa: E402
from transformers import Qwen3MoeConfig as HfQwen3MoeConfig  # noqa: E402


_SHARD_MAP_CHECK_KWARG = "check_vma" if "check_vma" in inspect.signature(shard_map).parameters else "check_rep"


def _expert_state_dict_key(prefix: str | None, expert_index: int, projection_name: str) -> str:
    key = f"{expert_index}.{projection_name}.weight"
    if prefix is None:
        return key
    return f"{prefix}.{key}"


@LmConfig.register_subclass("qwen3_moe")
@dataclass(frozen=True)
class Qwen3MoeConfig(LlamaConfig):
    """Qwen3 MoE config for HF checkpoint loading and loss evaluation."""

    max_seq_len: int = 40960
    hidden_dim: int = 2048
    intermediate_dim: int = 6144
    moe_intermediate_dim: int = 768
    num_layers: int = 48
    num_heads: int = 32
    head_dim: int | None = 128
    num_kv_heads: int = 4
    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.silu
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-6
    tie_word_embeddings: bool = False
    use_qk_norm: bool = True

    num_experts: int = 128
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    router_aux_loss_coef: float | None = 0.001
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()

    sliding_window: int | None = None
    max_window_layers: int = 48
    use_sliding_window: bool = False

    upcast_attn: bool = False
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None

    gradient_checkpointing: bool = True

    use_bias: bool = False
    use_layer_norm_weight: bool = True
    rope: RotaryEmbeddingsConfig = dataclasses.field(
        default_factory=lambda: DefaultRotaryEmbeddingsConfig(theta=1_000_000.0)
    )

    reference_checkpoint: str = "Qwen/Qwen3-30B-A3B"
    tokenizer: Optional[str] = None

    @property
    def Experts(self) -> Axis:
        return Axis(name="experts", size=self.num_experts)

    @property
    def TopExperts(self) -> Axis:
        return Axis(name="top_experts", size=self.num_experts_per_tok)

    @property
    def MoeMlp(self) -> Axis:
        return Axis(name="mlp", size=self.moe_intermediate_dim)

    def __post_init__(self):
        super().__post_init__()
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError(
                f"num_experts_per_tok={self.num_experts_per_tok} greater than num_experts={self.num_experts}."
            )
        if self.mlp_only_layers:
            raise NotImplementedError("Qwen3 MoE dense-only layers are not supported in Levanter yet.")
        if self.decoder_sparse_step != 1:
            raise NotImplementedError("Qwen3 MoE decoder_sparse_step values other than 1 are not supported yet.")

    # config-reuse subclass narrows to its own HF config/model type (LSP narrowing; pyrefly flags the same)
    def hf_checkpoint_converter(  # pyrefly: ignore[bad-override]
        self, ref_checkpoint: Optional[str] = None
    ) -> HFCheckpointConverter["Qwen3MoeConfig"]:  # type: ignore
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=True,
            tokenizer=ref_checkpoint if self.tokenizer is None else self.tokenizer,
            HfConfigClass=HfQwen3MoeConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        rope_theta, rope_scaling = hf_rope_config(hf_config)
        rope_config = RotaryEmbeddingsConfig.from_hf_config(rope_theta, rope_scaling)
        return Qwen3MoeConfig(
            max_seq_len=hf_config.max_position_embeddings,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            moe_intermediate_dim=hf_config.moe_intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            head_dim=getattr(hf_config, "head_dim", None),
            num_kv_heads=hf_config.num_key_value_heads,
            activation_function=ActivationFunctionEnum(hf_config.hidden_act),
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            use_qk_norm=True,
            num_experts=hf_config.num_experts,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            norm_topk_prob=hf_config.norm_topk_prob,
            router_aux_loss_coef=hf_config.router_aux_loss_coef,
            decoder_sparse_step=hf_config.decoder_sparse_step,
            mlp_only_layers=tuple(hf_config.mlp_only_layers),
            sliding_window=getattr(hf_config, "sliding_window", None),
            max_window_layers=getattr(hf_config, "max_window_layers", hf_config.num_hidden_layers),
            use_sliding_window=getattr(hf_config, "use_sliding_window", False),
            use_bias=getattr(hf_config, "attention_bias", False),
            rope=rope_config,
        )

    def to_hf_config(  # pyrefly: ignore[bad-override]
        self, vocab_size: int, config_overrides: Optional[Dict] = None
    ) -> HfQwen3MoeConfig:
        if config_overrides is None:
            config_overrides = {}

        rope_theta, rope_scaling = self.rope.to_hf_config()

        return hf_config_from_kwargs(
            HfQwen3MoeConfig,
            vocab_size=vocab_size,
            max_position_embeddings=self.max_seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            moe_intermediate_size=self.moe_intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            head_dim=self.head_dim,
            num_key_value_heads=self.num_kv_heads,
            hidden_act=self.activation_function.name,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_epsilon,
            tie_word_embeddings=self.tie_word_embeddings,
            attention_bias=self.use_bias,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            norm_topk_prob=self.norm_topk_prob,
            router_aux_loss_coef=self.router_aux_loss_coef,
            decoder_sparse_step=self.decoder_sparse_step,
            mlp_only_layers=list(self.mlp_only_layers),
            sliding_window=self.sliding_window,
            max_window_layers=self.max_window_layers,
            use_sliding_window=self.use_sliding_window,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            **config_overrides,
        )

    @property
    def model_type(self) -> Type["Qwen3MoeLMHeadModel"]:  # pyrefly: ignore[bad-override]
        return Qwen3MoeLMHeadModel

    def flops_per_token(self, vocab_size: int, context_length: int):
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.moe_intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=context_length,
            vocab_size=vocab_size,
            glu=True,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
        )

    def total_trainable_params(self, vocab_size):
        token_embedding = vocab_size * self.hidden_dim
        head_size = self.actual_head_size
        q_proj = self.hidden_dim * head_size * self.num_heads
        kv_proj = 2 * self.hidden_dim * head_size * self.num_kv_heads
        o_proj = head_size * self.num_heads * self.hidden_dim
        qk_norm = 2 * head_size
        attn = q_proj + kv_proj + o_proj + qk_norm
        router = self.hidden_dim * self.num_experts
        mlps = 3 * self.num_experts * self.hidden_dim * self.moe_intermediate_dim
        transformer_layer = attn + router + mlps + 2 * self.hidden_dim
        transformer = self.num_layers * transformer_layer + self.hidden_dim
        lm_head = 0 if self.tie_word_embeddings else token_embedding
        return transformer + token_embedding + lm_head


class Qwen3MoeExperts(ModuleWithStateDictSerialization):
    gate_proj: hnn.MoELinear
    up_proj: hnn.MoELinear
    down_proj: hnn.MoELinear
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(
        Experts: Axis,
        Embed: Axis,
        Mlp: Axis,
        activation_fn: Callable,
        *,
        key,
        use_bias: bool = False,
    ) -> "Qwen3MoeExperts":
        k_gate, k_up, k_down = jrandom.split(key, 3)
        gate_proj = hnn.MoELinear.init(Experts=Experts, Out=Mlp, In=Embed, key=k_gate, use_bias=use_bias)
        up_proj = hnn.MoELinear.init(Experts=Experts, Out=Mlp, In=Embed, key=k_up, use_bias=use_bias)
        down_proj = hnn.MoELinear.init(Experts=Experts, Out=Embed, In=Mlp, key=k_down, use_bias=use_bias)
        return Qwen3MoeExperts(gate_proj, up_proj, down_proj, activation_fn)

    @named_call
    def __call__(self, x: NamedArray, group_sizes: NamedArray, *, key=None) -> NamedArray:
        k_gate, k_up, k_down = maybe_rng_split(key, 3)
        hidden_states = self.act(self.gate_proj(x, group_sizes, key=k_gate))
        hidden_states = hidden_states * self.up_proj(x, group_sizes, key=k_up)
        return self.down_proj(hidden_states, group_sizes, key=k_down)

    def to_state_dict(self, prefix: Optional[str] = None) -> StateDict:
        projections = {
            "gate_proj": self.gate_proj.weight,
            "up_proj": self.up_proj.weight,
            "down_proj": self.down_proj.weight,
        }
        out = {}
        for i in range(self.gate_proj.Experts.size):
            for name, weight in projections.items():
                out[_expert_state_dict_key(prefix, i, name)] = jnp.swapaxes(weight["experts", i].array, -1, -2)
        return out

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> "Qwen3MoeExperts":
        expert_axis_index = self.gate_proj.weight.axes.index(self.gate_proj.Experts)
        values = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            weights = []
            for i in range(self.gate_proj.Experts.size):
                key = _expert_state_dict_key(prefix, i, name)
                weights.append(jnp.swapaxes(state_dict[key], -1, -2))
            values[name] = jnp.stack(weights, axis=expert_axis_index)

        return eqx.tree_at(
            lambda m: [m.gate_proj.weight.array, m.up_proj.weight.array, m.down_proj.weight.array],
            self,
            [values["gate_proj"], values["up_proj"], values["down_proj"]],
        )


class Qwen3MoeSparseMoeBlock(eqx.Module):
    config: Qwen3MoeConfig = eqx.field(static=True)
    gate: hnn.Linear
    experts: Qwen3MoeExperts

    @staticmethod
    def init(config: Qwen3MoeConfig, *, key) -> "Qwen3MoeSparseMoeBlock":
        k_gate, k_experts = jrandom.split(key, 2)
        gate = hnn.Linear.init(config.Embed, config.Experts, key=k_gate, use_bias=False)
        experts = Qwen3MoeExperts.init(
            Experts=config.Experts,
            Embed=config.Embed,
            Mlp=config.MoeMlp,
            activation_fn=config.activation_function.to_fn(),
            key=k_experts,
            use_bias=False,
        )
        return Qwen3MoeSparseMoeBlock(config, gate, experts)

    def _route(self, router_logits: NamedArray, Token: Axis, TopExperts: Axis):
        Experts = self.config.Experts

        @partial(
            shard_map,
            mesh=hax.partitioning._get_mesh(),
            in_specs=hax.partitioning.pspec_for_axis(router_logits.axes),
            out_specs=(
                hax.partitioning.pspec_for_axis((Token, TopExperts)),
                hax.partitioning.pspec_for_axis((Token, TopExperts)),
                hax.partitioning.pspec_for_axis((Experts,)),
            ),
            **{_SHARD_MAP_CHECK_KWARG: False},
        )
        def sharded_route(router_logits_):
            router_probs_ = jax.nn.softmax(router_logits_, axis=-1)
            selected_weights_, selected_experts_ = jax.lax.top_k(router_probs_, TopExperts.size)
            if self.config.norm_topk_prob:
                selected_weights_ = selected_weights_ / selected_weights_.sum(-1, keepdims=True)
            expert_loads_ = jnp.bincount(selected_experts_.reshape(-1), length=self.config.num_experts)
            return selected_weights_, selected_experts_, expert_loads_

        # jax.shard_map erases the wrapped callable's signature, so pyrefly mis-binds its
        # decorator TypeVar to the Array argument; cast back to a plain callable.
        sharded_route = cast(Callable[..., Any], sharded_route)

        with jax.named_scope("route"):
            selected_weights, selected_experts, expert_loads = sharded_route(router_logits.array)
            return (
                hax.named(selected_weights, (Token, TopExperts)),
                hax.named(selected_experts, (Token, TopExperts)),
                hax.named(expert_loads, (Experts,)),
            )

    def _permute(self, x_flat: NamedArray, topk_idx_flat: NamedArray, TokenRepeat: Axis):
        Experts = self.config.Experts

        @partial(
            shard_map,
            mesh=hax.partitioning._get_mesh(),
            in_specs=(
                hax.partitioning.pspec_for_axis(x_flat.axes),
                hax.partitioning.pspec_for_axis(topk_idx_flat.axes),
            ),
            out_specs=(
                hax.partitioning.pspec_for_axis((TokenRepeat, self.config.Embed)),
                hax.partitioning.pspec_for_axis((Experts,)),
                hax.partitioning.pspec_for_axis((TokenRepeat,)),
            ),
            **{_SHARD_MAP_CHECK_KWARG: False},
        )
        def permute_sharded(x_flat_, topk_idx_flat_):
            sort_idx_ = jnp.argsort(topk_idx_flat_, axis=-1)
            x_repeat_sort_ = jnp.take(x_flat_, sort_idx_ // self.config.num_experts_per_tok, axis=0)
            group_sizes_ = jnp.bincount(topk_idx_flat_, length=self.config.num_experts)
            return x_repeat_sort_, group_sizes_, sort_idx_

        # See sharded_route above: cast around jax.shard_map's signature erasure.
        permute_sharded = cast(Callable[..., Any], permute_sharded)

        with jax.named_scope("permute"):
            x_repeat_sort, group_sizes, sort_idx = permute_sharded(x_flat.array, topk_idx_flat.array)
            return (
                hax.named(x_repeat_sort, (TokenRepeat, self.config.Embed)),
                hax.named(group_sizes, (Experts,)),
                hax.named(sort_idx, (TokenRepeat,)),
            )

    def _unpermute(
        self,
        out_repeat_sort: NamedArray,
        sort_idx: NamedArray,
        Token: Axis,
        TopExperts: Axis,
    ):
        @partial(
            shard_map,
            mesh=hax.partitioning._get_mesh(),
            in_specs=(
                hax.partitioning.pspec_for_axis(out_repeat_sort.axes),
                hax.partitioning.pspec_for_axis(sort_idx.axes),
            ),
            out_specs=hax.partitioning.pspec_for_axis((Token, TopExperts, self.config.Embed)),
            **{_SHARD_MAP_CHECK_KWARG: False},
        )
        def unpermute_sharded(out_repeat_sort_, sort_idx_):
            inv_sort_idx_ = jnp.argsort(sort_idx_)
            out_repeat_ = jnp.take(out_repeat_sort_, inv_sort_idx_, axis=0)
            return jnp.reshape(out_repeat_, (-1, self.config.num_experts_per_tok, self.config.hidden_dim))

        # See sharded_route above: cast around jax.shard_map's signature erasure.
        unpermute_sharded = cast(Callable[..., Any], unpermute_sharded)

        with jax.named_scope("unpermute"):
            out_repeat_unflat = unpermute_sharded(out_repeat_sort.array, sort_idx.array)
            return hax.named(out_repeat_unflat, (Token, TopExperts, self.config.Embed))

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        if x.has_axis("batch"):
            squash_axes = [x.resolve_axis("batch"), x.resolve_axis(self.config.max_Pos.name)]
        else:
            squash_axes = [x.resolve_axis(self.config.max_Pos.name)]

        TopExperts = self.config.TopExperts
        k_gate, k_experts = maybe_rng_split(key, 2)

        x_flat = hax.flatten_axes(x, old_axes=squash_axes, new_axis="token")
        Token = x_flat.resolve_axis("token")
        router_logits = self.gate(x_flat, key=k_gate)
        topk_weights, topk_idx, _ = self._route(router_logits.astype(jnp.float32), Token, TopExperts)
        topk_weights = topk_weights.astype(x.dtype)

        topk_idx_flat = hax.flatten_axes(topk_idx, old_axes=[Token, TopExperts], new_axis="token_repeat")
        TokenRepeat = topk_idx_flat.resolve_axis("token_repeat")
        x_repeat_sort, group_sizes, sort_idx = self._permute(x_flat, topk_idx_flat, TokenRepeat)
        out_repeat_sort = self.experts(x_repeat_sort, group_sizes, key=k_experts)
        out_repeat_unflat = self._unpermute(out_repeat_sort, sort_idx, Token, TopExperts)
        out = out_repeat_unflat.dot(topk_weights, axis=TopExperts)
        return hax.unflatten_axis(out, axis=Token, new_axes=squash_axes)


class Qwen3MoeDecoderLayer(eqx.Module):
    config: Qwen3MoeConfig = eqx.field(static=True)
    self_attn: Attention
    mlp: Qwen3MoeSparseMoeBlock
    input_layernorm: hnn.RmsNorm
    post_attention_layernorm: hnn.RmsNorm

    @staticmethod
    def init(config: Qwen3MoeConfig, *, key) -> "Qwen3MoeDecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)
        attn = Attention.init(config.attention_config(), key=k_attn)
        mlp = Qwen3MoeSparseMoeBlock.init(config, key=k_mlp)
        ln_1 = config.mk_LayerNorm(config.Embed)
        ln_2 = config.mk_LayerNorm(config.Embed)
        return Qwen3MoeDecoderLayer(config, attn, mlp, ln_1, ln_2)

    @named_call
    def __call__(
        self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)
        residual = x
        x = self.input_layernorm(x)
        x = residual + self.self_attn(x=x, mask=mask, key=k_attn, pos_ids=pos_ids)

        residual = x
        x = self.post_attention_layernorm(x)
        return residual + self.mlp(x, key=k_mlp)


class Qwen3MoeTransformer(eqx.Module):
    config: Qwen3MoeConfig = eqx.field(static=True)
    layers: BlockFoldable[Qwen3MoeDecoderLayer]
    norm: hnn.RmsNorm

    @staticmethod
    def init(config: Qwen3MoeConfig, *, key) -> "Qwen3MoeTransformer":
        S = Stacked
        if not config.scan_layers:
            S = BlockSeq
        layers = S.init(config.Layers, Qwen3MoeDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        return Qwen3MoeTransformer(config, layers, config.mk_LayerNorm(config.Embed))

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: AttentionMask | None, *, key, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids)
        return self.norm(x)


class Qwen3MoeLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[Qwen3MoeConfig]):
    transformer: Qwen3MoeTransformer
    embeddings: LlamaEmbedding
    lm_head: Optional[hnn.Linear]

    @property
    def config(self):
        return self.transformer.config

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @classmethod
    def init(cls, Vocab: Axis, config: Qwen3MoeConfig, *, key) -> "Qwen3MoeLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = Qwen3MoeTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        lm_head = None
        if not config.tie_word_embeddings:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)
        return Qwen3MoeLMHeadModel(transformer, embeddings, lm_head)

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | None = None,
        pos_ids: NamedArray | None = None,
        *,
        key=None,
    ) -> NamedArray:
        k_t, k_head = maybe_rng_split(key, 2)
        x = self.activations(input_ids, attn_mask=attn_mask, key=k_t, pos_ids=pos_ids)
        if self.lm_head is not None:
            return self.lm_head(x, key=k_head)
        return self.embeddings.unembed(x)

    def activations(  # pyrefly: ignore[bad-override]  # narrows activations return type for this MoE head
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        x = self.embeddings.embed(input_ids)
        return self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)

    def get_lm_head(self) -> hax.NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        return self.lm_head.weight

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[Qwen3MoeConfig]":
        new_embeddings, new_lm_head = resize_embeddings_and_lm_head(
            self.Vocab, self.embeddings, self.lm_head, new_size, key
        )
        if new_lm_head is None:
            return eqx.tree_at(lambda m: m.embeddings, self, new_embeddings)
        return eqx.tree_at(lambda m: (m.embeddings, m.lm_head), self, (new_embeddings, new_lm_head))

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": "model", "embeddings": None}
