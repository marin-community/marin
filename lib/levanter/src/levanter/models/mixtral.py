# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array
from jax.experimental.shard_map import shard_map

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax._src.scan import ScanCheckpointSpec
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.state_dict import ModuleWithStateDictSerialization, StateDict

import levanter.tracker
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.layers.attention import Attention, AttentionBackend, AttentionConfig, AttentionMask
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.models.llama import LlamaEmbedding, LlamaMlp
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.models.mistral import MistralConfig
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag
from levanter.utils.types import BlockFoldable


silence_transformer_nag()
from transformers import MixtralConfig as HfMixtralConfig  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402


@LmConfig.register_subclass("mixtral")
@dataclass(frozen=True)
class MixtralConfig(MistralConfig):
    """Config for MistralModel

    Args:
        seq_len (int, optional): maximum length of the input sequence. Defaults to 8192.
        hidden_dim (int, optional): dimension of the hidden state. Defaults to 4096.
        intermediate_dim (int, optional): dimension of the intermediate state. Defaults to 14336.
        num_layers (int, optional): number of hidden layers in the Transformer encoder. Defaults to 32.
        num_heads (int, optional): number of attention heads for each attention layer. Defaults to 32.
        num_kv_heads (int, optional): number of attention heads for keys and values in each attention layer.
            Setting to 1 means MQA. Setting to num_heads means MHA. Otherwise GQA.
            Note that num_heads must be divisible by this number. Defaults to 8.
        activation_function (str, optional): activation function for the hidden layer. Defaults to "silu".
        sliding_window (int, optional): window size of sliding window attention. Defaults to 4096.
        num_experts_per_tok (int, optional): number of experts to route per-token.
        n_routed_experts (int, optional): number of experts per Sparse MLP layer.
        lbl_coef (`float`, optional): aux loss factor for load balancing loss. Defaults to 0.01
        rzl_coef (`float`, optional): aux loss factor for router z-loss. Defaults to 0.001
    """

    seq_len: int = 8192
    hidden_dim: int = 4096
    intermediate_dim: int = 14336
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.silu
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-6
    tie_word_embeddings: bool = False

    num_experts_per_tok: int = 2
    n_routed_experts: int = 8
    n_shared_experts: int = 0

    lbl_coef: Optional[float] = 0.01
    rzl_coef: Optional[float] = 0.001

    # Attention-related config
    upcast_attn: bool = False
    use_flash_attention: Optional[bool] = True
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = 2048

    gradient_checkpointing: ScanCheckpointSpec = True
    scan_layers: bool = True

    use_bias: bool = False
    use_layer_norm_weight: bool = True
    # Note that HF mixtral defaults rope theta to 1e6. Here we default to 1e4.
    rope: RotaryEmbeddingsConfig = dataclasses.field(default_factory=DefaultRotaryEmbeddingsConfig)

    reference_checkpoint: str = "mistralai/Mixtral-8x7B-v0.1"
    tokenizer: Optional[str] = None

    # Axis
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Heads = property(lambda self: Axis(name="heads", size=self.num_heads))
    KVHeads = property(lambda self: Axis(name="kv_heads", size=self.num_kv_heads))
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))
    Experts = property(lambda self: Axis(name="experts", size=self.n_routed_experts))
    SharedExperts = property(lambda self: Axis(name="shared_experts", size=self.n_shared_experts))
    TopExperts = property(lambda self: Axis(name="top_experts", size=self.num_experts_per_tok))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.hidden_dim // self.num_heads))

    def __post_init__(self):
        super().__post_init__()
        assert (
            self.num_experts_per_tok <= self.n_routed_experts
        ), f"num_experts_per_tok={self.num_experts_per_tok} greater than by n_routed_experts={self.n_routed_experts}."

    def hf_checkpoint_converter(
        self, ref_checkpoint: Optional[str] = None
    ) -> HFCheckpointConverter["MixtralConfig"]:  # type: ignore
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=True,
            tokenizer=ref_checkpoint if self.tokenizer is None else self.tokenizer,
            HfConfigClass=HfMixtralConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        rope_theta = hf_config.rope_theta
        rope_config = RotaryEmbeddingsConfig.from_hf_config(rope_theta, None)
        return MixtralConfig(
            seq_len=hf_config.max_position_embeddings,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            activation_function=ActivationFunctionEnum(hf_config.hidden_act),
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            rope=rope_config,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            n_routed_experts=hf_config.num_local_experts,
            lbl_coef=hf_config.router_aux_loss_coef,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfMixtralConfig:
        """Convert to HuggingFace's MistralConfig

        Args:
            vocab_size (int, optional): Vocabulary size of the tokenizer. Defaults to 32000.
            config_overrides (dict, optional): Overrides for the config. Defaults to None.

        Returns:
            HfMistralConfig: HuggingFace's MistralConfig
        """
        if config_overrides is None:
            config_overrides = {}

        rope_theta, rope_scaling = self.rope.to_hf_config()

        return HfMixtralConfig(
            max_position_embeddings=self.seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            hidden_act=self.activation_function.name,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_epsilon,
            tie_word_embeddings=self.tie_word_embeddings,
            num_experts_per_tok=self.num_experts_per_tok,
            num_local_experts=self.n_routed_experts,
            router_aux_loss_coef=self.lbl_coef,
            vocab_size=vocab_size,
            rope_theta=rope_theta,
            sliding_window=None,
            **config_overrides,
        )

    @property
    def model_type(cls) -> Type["MixtralLMHeadModel"]:
        return MixtralLMHeadModel

    def mk_LayerNorm(self, axis: Axis) -> hnn.RmsNorm:
        return hnn.RmsNorm.init(
            axis, eps=self.layer_norm_epsilon, use_weight=self.use_layer_norm_weight, use_bias=self.use_bias
        )

    def flops_per_token(self, vocab_size: int):
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            vocab_size=vocab_size,
            glu=True,
            num_experts=self.n_routed_experts,
            num_shared_experts=self.n_shared_experts,
            num_experts_per_tok=self.num_experts_per_tok,
        )

    def total_trainable_params(self, vocab_size):
        token_embedding = vocab_size * self.hidden_dim

        head_size = self.hidden_dim // self.num_heads
        q_proj = self.hidden_dim * head_size * self.num_heads
        kv_proj = 2 * self.hidden_dim * head_size * self.num_kv_heads
        o_proj = head_size * self.num_heads * self.hidden_dim
        attn = q_proj + kv_proj + o_proj

        router = self.hidden_dim * self.n_routed_experts
        mlps = 3 * (self.n_routed_experts + self.n_shared_experts) * self.hidden_dim * self.intermediate_dim
        moe = router + mlps

        transformer_layer = attn + moe + 2 * self.hidden_dim  # plus 2 rmsnorm

        transformer = self.num_layers * transformer_layer + self.hidden_dim  # plus final rmsnorm

        return transformer + token_embedding * 2  # plus embedding and lm head

    def attention_config(self) -> AttentionConfig:
        """Convert this MixtralConfig to an AttentionConfig for use with Attention."""
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_bias=self.use_bias,
            upcast_attn=self.upcast_attn,
            attn_backend=self.attn_backend,
            flash_attention_block_size=self.flash_attention_block_size,
            rope=self.rope,
        )


class MixtralMoEMlp(ModuleWithStateDictSerialization):
    """Multi-layer Perceptron with Mixture of Experts"""

    w1: hnn.MoELinear  # (gate_proj) projection from Embed to Mlp
    w2: hnn.MoELinear  # (down_proj) projection from Mlp to Embed
    w3: hnn.MoELinear  # (up_proj) projection from Embed to Mlp
    Embed: hax.Axis = eqx.field(static=True)
    Mlp: hax.Axis = eqx.field(static=True)
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(
        Experts: Axis,
        Embed: Axis,
        Mlp: Axis,
        activation_fn: Union[ActivationFunctionEnum, Callable],
        *,
        key,
        use_bias: bool = False,
    ) -> "MixtralMoEMlp":
        k1, k2, k3 = jrandom.split(key, 3)
        w1 = hnn.MoELinear.init(Experts=Experts, Out=Mlp, In=Embed, key=k1, use_bias=use_bias)
        w2 = hnn.MoELinear.init(Experts=Experts, Out=Embed, In=Mlp, key=k2, use_bias=use_bias)
        w3 = hnn.MoELinear.init(Experts=Experts, Out=Mlp, In=Embed, key=k3, use_bias=use_bias)
        if isinstance(activation_fn, ActivationFunctionEnum):
            activation_fn = activation_fn.to_fn()
        return MixtralMoEMlp(w1, w2, w3, Embed, Mlp, activation_fn)

    @named_call
    def __call__(self, x: NamedArray, group_sizes: NamedArray, *, key=None) -> NamedArray:
        k1, k2, k3 = maybe_rng_split(key, 3)
        hidden_states = self.w1(x, group_sizes, key=k1)
        hidden_states = self.act(hidden_states)
        hidden_states = hidden_states * self.w3(x, group_sizes, key=k3)
        outputs = self.w2(hidden_states, group_sizes, key=k2)
        return outputs

    def to_state_dict(self, prefix: Optional[str] = None) -> StateDict:
        w = [self.w1.weight, self.w2.weight, self.w3.weight]
        out = {}

        num_experts = self.w1.Experts.size
        for i in range(num_experts):
            for j in range(3):
                key = f"{prefix}.{i}.w{j + 1}.weight"
                val = w[j]["experts", i].array
                # out[key] = val
                out[key] = jnp.swapaxes(val, -1, -2)

        return out

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> "MixtralMoEMlp":
        w: List[List[Array]] = [[], [], []]
        num_experts = self.w1.Experts.size
        for i in range(num_experts):
            for j in range(3):
                key = f"{prefix}.{i}.w{j + 1}.weight"
                val = jnp.swapaxes(state_dict[key], -1, -2)[..., None, :, :]
                # val = state_dict[key][..., None, :, :]
                w[j].append(val)

        for j in range(3):
            w[j] = jnp.concat(w[j], axis=1)

        return eqx.tree_at(lambda m: [m.w1.weight.array, m.w2.weight.array, m.w3.weight.array], self, w)


class MixtralSparseMoeBlock(eqx.Module):
    """Mixture-of-Experts"""

    config: MistralConfig = eqx.field(static=True)
    gate: hnn.Linear  # projection from Embed to Experts
    experts: MixtralMoEMlp

    @staticmethod
    def init(config: MistralConfig, *, key) -> "MixtralSparseMoeBlock":
        k_gate, k_experts, key = maybe_rng_split(key, 3)

        gate = hnn.Linear.init(config.Embed, config.Experts, key=k_gate, use_bias=config.use_bias)
        experts = MixtralMoEMlp.init(
            Experts=config.Experts,
            Embed=config.Embed,
            Mlp=config.Mlp,
            activation_fn=config.activation_function,
            key=k_experts,
            use_bias=config.use_bias,
        )

        return MixtralSparseMoeBlock(config, gate, experts)

    def _route(self, router_probs: NamedArray, Token: Axis, TopExperts: Axis):
        @partial(
            shard_map,
            mesh=hax.partitioning._get_mesh(),
            in_specs=hax.partitioning.pspec_for_axis(router_probs.axes),
            out_specs=(
                hax.partitioning.pspec_for_axis((Token, TopExperts)),
                hax.partitioning.pspec_for_axis((Token, TopExperts)),
            ),
            check_rep=False,
        )
        def sharded_route(router_probs_):
            selected_weights_, selected_experts_ = jax.lax.top_k(router_probs_, TopExperts.size)
            selected_weights_ = selected_weights_ / selected_weights_.sum(-1, keepdims=True)

            return selected_weights_, selected_experts_

        with jax.named_scope("route"):
            selected_weights_, selected_experts_ = sharded_route(router_probs.array)

            selected_weights = NamedArray(selected_weights_, axes=(Token, TopExperts))
            selected_experts = NamedArray(selected_experts_, axes=(Token, TopExperts))

        return selected_weights, selected_experts

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
            check_rep=False,
        )
        def permute_sharded(x_flat_: Array, topk_idx_flat_: Array):
            sort_idx_ = jnp.argsort(topk_idx_flat_, axis=-1)
            x_repeat_sort_ = jnp.take(x_flat_, sort_idx_ // self.config.num_experts_per_tok, axis=0)
            group_sizes_ = jnp.bincount(topk_idx_flat_, length=self.config.n_routed_experts)

            return x_repeat_sort_, group_sizes_, sort_idx_

        with jax.named_scope("permute"):
            x_repeat_sort_, group_sizes_, sort_idx_ = permute_sharded(x_flat.array, topk_idx_flat.array)
            x_repeat_sort = NamedArray(x_repeat_sort_, axes=(TokenRepeat, self.config.Embed))
            group_sizes = NamedArray(group_sizes_, axes=(Experts,))
            sort_idx = NamedArray(sort_idx_, axes=(TokenRepeat,))

        return x_repeat_sort, group_sizes, sort_idx

    def _unpermute(
        self,
        out_repeat_sort: NamedArray,
        sort_idx: NamedArray,
        topk_weights: NamedArray,
        Token: Axis,
        TokenRepeat: Axis,
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
            check_rep=False,
        )
        def unpermute_sharded(out_repeat_sort_: Array, sort_idx_: Array):
            inv_sort_idx_ = jnp.argsort(sort_idx_)
            out_repeat_ = jnp.take(out_repeat_sort_, inv_sort_idx_, axis=0)
            out_repeat_unflat_ = jnp.reshape(
                out_repeat_, (-1, self.config.num_experts_per_tok, self.config.hidden_dim)
            )

            return out_repeat_unflat_

        with jax.named_scope("unpermute"):
            out_repeat_unflat_ = unpermute_sharded(out_repeat_sort.array, sort_idx.array)
            out_repeat_unflat = NamedArray(out_repeat_unflat_, axes=(Token, TopExperts, self.config.Embed))

        return out_repeat_unflat

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        if x.has_axis("batch"):
            squash_axes = [x.resolve_axis("batch"), x.resolve_axis(self.config.Pos.name)]
        else:
            squash_axes = [x.resolve_axis(self.config.Pos.name)]
        Experts = self.config.Experts
        TopExperts = self.config.TopExperts

        k_gate, k_experts, key = maybe_rng_split(key, 3)

        x_flat = hax.flatten_axes(x, old_axes=squash_axes, new_axis="token")  # [Batch, Pos, Embed] -> [Token, Embed]
        Token = x_flat.resolve_axis("token")

        router_logits = self.gate(x_flat, key=k_gate)
        router_probs = hnn.softmax(router_logits, axis=Experts)
        topk_weights, topk_idx = self._route(router_probs, Token, TopExperts)

        topk_idx_flat = hax.flatten_axes(topk_idx, old_axes=[Token, TopExperts], new_axis="token_repeat")
        TokenRepeat = topk_idx_flat.resolve_axis("token_repeat")
        x_repeat_sort, group_sizes, sort_idx = self._permute(x_flat, topk_idx_flat, TokenRepeat)

        out_repeat_sort = self.experts(x_repeat_sort, group_sizes, key=k_experts)

        out_repeat_unflat = self._unpermute(
            out_repeat_sort, sort_idx, topk_weights, Token, TokenRepeat, TopExperts
        )  # [TokenRepeat, Embed]

        out = out_repeat_unflat.dot(topk_weights, axis=TopExperts)  # [Token, Embed]

        # aux loss
        extras = {}
        expert_loads = group_sizes / hax.sum(group_sizes, axis=Experts)
        extras = {
            "expert_loads": expert_loads,
        }
        if self.config.lbl_coef is not None:
            f = expert_loads * self.config.n_routed_experts / self.config.num_experts_per_tok
            p = hax.mean(router_probs, axis=Token)
            extras["load_balancing_loss"] = self.config.lbl_coef * hax.sum(f * p, axis=Experts)
        if self.config.rzl_coef is not None:
            extras["router_z_loss"] = self.config.rzl_coef * hax.mean(
                hnn.logsumexp(router_logits, axis=Experts) ** 2, axis=Token
            )

        return hax.unflatten_axis(out, axis=Token, new_axes=squash_axes), extras  # [Batch, Pos, Embed]


class MixtralDecoderLayer(eqx.Module):
    config: MixtralConfig = eqx.field(static=True)
    self_attn: Attention
    block_sparse_moe: MixtralSparseMoeBlock
    input_layernorm: hnn.RmsNorm
    post_attention_layernorm: hnn.RmsNorm
    shared_mlp: Optional[LlamaMlp]

    @staticmethod
    def init(config: MistralConfig, *, key) -> "MixtralDecoderLayer":
        k_attn, k_moe, k_mlp = jrandom.split(key, 3)

        attn_config = config.attention_config()
        attn = Attention.init(attn_config, key=k_attn)
        block_sparse_moe = MixtralSparseMoeBlock.init(config, key=k_moe)
        ln_1 = hnn.RmsNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
        ln_2 = hnn.RmsNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
        shared_mlp = None
        if config.n_shared_experts > 0:
            shared_mlp = LlamaMlp.init(
                config.Embed,
                config.Mlp,
                config.activation_function,
                key=k_mlp,
                use_bias=config.use_bias,
            )
        return MixtralDecoderLayer(config, attn, block_sparse_moe, ln_1, ln_2, shared_mlp)

    @named_call
    def __call__(self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)
        # self attention and skip connection
        residual = x
        x = self.input_layernorm(x)
        attn_output = self.self_attn(x=x, mask=mask, key=k_attn)
        x = residual + attn_output

        # MLP and skip connection
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = self.shared_mlp(x, key=k_mlp) if self.shared_mlp is not None else 0
        moe_output, extras = self.block_sparse_moe(x, key=k_mlp)
        output = residual + mlp_output + moe_output
        return output, extras


class MixtralTransformer(eqx.Module):
    config: MistralConfig = eqx.field(static=True)
    layers: BlockFoldable[MixtralDecoderLayer]
    norm: hnn.RmsNorm

    @staticmethod
    def init(config: MistralConfig, *, key) -> "MixtralTransformer":
        S = Stacked
        if not config.scan_layers:
            from haliax.nn.scan import BlockSeq

            S = BlockSeq

        layers = S.init(config.Layers, MixtralDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = config.mk_LayerNorm(config.Embed)

        return MixtralTransformer(config, layers, ln_f)

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: Optional[NamedArray], *, key, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x, extras = self.layers.scan(x, mask=attn_mask, key=keys)
        x = self.norm(x)

        # moe logging
        expert_loads = extras["expert_loads"]
        entropy = -hax.sum(expert_loads * hax.log(expert_loads + 1e-6), axis=self.config.Experts)

        stats = {}
        for i in range(self.config.num_layers):
            stats[f"moe/layer{i}/routing_entropy"] = entropy.array[i]
            for j in range(self.config.n_routed_experts):
                stats[f"moe/layer{i}/expert{j}_load"] = expert_loads.array[i, j]

        if self.config.lbl_coef is not None:
            extras["load_balancing_loss"] = hax.sum(extras["load_balancing_loss"], axis=self.config.Layers)
            stats["train/load_balancing_loss"] = extras["load_balancing_loss"].array
        if self.config.rzl_coef is not None:
            extras["router_z_loss"] = hax.sum(extras["router_z_loss"], axis=self.config.Layers)
            stats["train/router_z_loss"] = extras["router_z_loss"].array

        levanter.tracker.jit_log(stats)

        return x, extras


class MixtralLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[MixtralConfig]):
    transformer: MixtralTransformer
    embeddings: LlamaEmbedding
    lm_head: Optional[hnn.Linear]

    @property
    def config(self):
        return self.transformer.config

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @classmethod
    def init(cls, Vocab: Axis, config: MistralConfig, *, key) -> "MixtralLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = MixtralTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)

        return MixtralLMHeadModel(transformer, embeddings, lm_head)

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[Union[NamedArray, AttentionMask]] = None,
        pos_ids: NamedArray | None = None,
        *,
        key=None,
    ) -> NamedArray:
        """
        Args:
            input_ids (NamedArray): [batch, position]
                Indices of input sequence tokens in the vocabulary.
            attn_mask (Union[NamedArray, AttentionMask], optional): [batch, position]
                Mask to avoid performing attention on the padding token indices of the encoder input.
                The attn_mask from training pipeline may be an AttentionMask object instead of NamedArray
        """
        k_t, k_head = maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids)
        x, _ = self.transformer(x, attn_mask=attn_mask, key=k_t, pos_ids=pos_ids)
        if self.lm_head:
            lm_logits = self.lm_head(x, key=k_head)
        else:
            lm_logits = self.embeddings.unembed(x)
        return lm_logits

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        pos_ids: NamedArray | None = None,
        *,
        key=None,
    ) -> NamedArray:
        """
        Compute the activations for the next token in a sequence.
        Args:
            input_ids: token IDs with shape {Pos}
            attn_mask: attention mask with shape {Pos, KeyPos}
            key: PRNGKey for random number generation

        Returns:
            NamedArray: activations with shape {Pos, Embed}

        """
        x = self.embeddings.embed(input_ids)
        x, extras = self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)

        aux_loss = 0
        if self.config.lbl_coef is not None:
            aux_loss += extras["load_balancing_loss"]
        if self.config.rzl_coef is not None:
            aux_loss += extras["router_z_loss"]
        return x, aux_loss

    def get_lm_head(self) -> hax.NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        else:
            return self.lm_head.weight

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[MixtralConfig]":
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        if self.lm_head is not None:
            new_lm_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
            new_lm_head = dataclasses.replace(self.lm_head, Out=new_Vocab, weight=new_lm_matrix)
            return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)
        else:
            return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": "model", "embeddings": None}
