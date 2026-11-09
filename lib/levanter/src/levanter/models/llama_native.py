# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Llama over JAX arrays with explicit storage and compute sharding.

This implementation remains separate from the named-array model while their
numerics, placement and training performance are compared on identical weights.
The state-dict boundary uses canonical output-first Hugging Face parameter shapes.
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Literal, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray

from levanter.grug.attention import AttentionMask, GrugAttentionImplementation, attention, reference_attention
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.layers.linear import INITIALIZER_TRUNCATION_BOUND, Linear
from levanter.layers.rms_norm import RmsNorm
from levanter.layers.rotary_native import NativeRotaryConfig, RotaryConfig, apply_rotary
from levanter.layers.scan_checkpoint import ScanCheckpointSpec
from levanter.layers.stacked import ArrayStacked
from levanter.utils.activation import ActivationFunctionEnum, JaxActivationFunction


_HF_TOKEN_EMBEDDING_WEIGHT_KEY = "model.embed_tokens.weight"
_HF_FINAL_NORM_PREFIX = "model.norm"
_HF_LAYER_PREFIX = "model.layers."
_LAYER_NORM_FIELDS = ("input_layernorm", "post_attention_layernorm", "post_attn_layernorm", "post_mlp_layernorm")
_ATTENTION_FIELDS = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
_MLP_FIELDS = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class NativeLlamaSharding:
    """Concrete Llama layouts for batch, FSDP storage and tensor parallelism.

    FSDP shards the input dimension of column projections and the output
    dimension of row projections in storage. Forward computation gathers only
    that FSDP placement; tensor-parallel heads/MLP channels remain partitioned.
    Layer dimensions are replicated.
    """

    batch_axes: tuple[str, ...] = ()
    fsdp_axis: str | None = None
    model_axis: str | None = None

    @property
    def batch(self) -> tuple[str, ...] | None:
        return self.batch_axes or None

    @property
    def tokens(self) -> P:
        return P(self.batch, None)

    @property
    def residual(self) -> P:
        return P(self.batch, None, None)

    @property
    def intermediate(self) -> P:
        return P(self.batch, None, self.model_axis)

    @property
    def query(self) -> P:
        return P(self.batch, None, self.model_axis, None)

    @property
    def column_storage(self) -> P:
        return P(self.model_axis, self.fsdp_axis)

    @property
    def column_compute(self) -> P:
        return P(self.model_axis, None)

    @property
    def row_storage(self) -> P:
        return P(self.fsdp_axis, self.model_axis)

    @property
    def row_compute(self) -> P:
        return P(None, self.model_axis)

    @property
    def norm_storage(self) -> P:
        return P(self.fsdp_axis)

    @property
    def norm_compute(self) -> P:
        return P(None)

    @property
    def logits(self) -> P:
        return P(self.batch, None, self.model_axis)


@dataclass(frozen=True)
class NativeLlamaConfig:
    """Dense Llama architecture with native dimensions and kernel selection."""

    vocab_size: int
    hidden_dim: int = 4096
    intermediate_dim: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.silu
    head_dim: int | None = None
    max_seq_len: int = 2048
    layer_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = False
    use_bias: bool = False
    use_layer_norm_weight: bool = True
    input_embedding_norm: bool = False
    hybrid_norm: bool = False
    use_qk_norm: bool = False
    upcast_attn: bool = False
    rope: NativeRotaryConfig = RotaryConfig()
    attention_implementation: GrugAttentionImplementation = "reference"
    gradient_checkpointing: ScanCheckpointSpec = True

    def __post_init__(self):
        self.activation_function.to_jax_fn()
        if self.attention_implementation == "xla_flash":
            raise ValueError("xla_flash still uses named-array operations; select a native attention implementation")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.head_dim is None and self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads when head_dim is not set")
        if self.actual_head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")

    @property
    def actual_head_dim(self) -> int:
        return self.head_dim if self.head_dim is not None else self.hidden_dim // self.num_heads

    def build(self, *, key: PRNGKeyArray, sharding: NativeLlamaSharding = NativeLlamaSharding()) -> "NativeLlamaModel":
        return NativeLlamaModel.init(self, key=key, sharding=sharding)


def _norm(config: NativeLlamaConfig, dim: int, storage_spec: P) -> RmsNorm:
    return RmsNorm.init(
        dim,
        eps=config.layer_norm_epsilon,
        use_weight=config.use_layer_norm_weight,
        use_bias=config.use_bias,
        storage_spec=storage_spec,
        compute_spec=P(None),
    )


class NativeLlamaAttention(eqx.Module):
    """Grouped-query attention with explicitly placed head and residual arrays."""

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    q_norm: RmsNorm | None
    k_norm: RmsNorm | None
    config: NativeLlamaConfig = eqx.field(static=True)
    sharding: NativeLlamaSharding = eqx.field(static=True)

    @staticmethod
    def init(config: NativeLlamaConfig, sharding: NativeLlamaSharding, *, key: PRNGKeyArray):
        q_key, k_key, v_key, o_key = jax.random.split(key, 4)
        head_dim = config.actual_head_dim

        def column(out_features: int, key: PRNGKeyArray) -> Linear:
            return Linear.init(
                config.hidden_dim,
                out_features,
                key=key,
                use_bias=config.use_bias,
                initializer_std=1 / math.sqrt(config.hidden_dim),
                storage_spec=sharding.column_storage,
                compute_spec=sharding.column_compute,
            )

        return NativeLlamaAttention(
            column(config.num_heads * head_dim, q_key),
            column(config.num_kv_heads * head_dim, k_key),
            column(config.num_kv_heads * head_dim, v_key),
            Linear.init(
                config.num_heads * head_dim,
                config.hidden_dim,
                key=o_key,
                use_bias=config.use_bias,
                initializer_std=1 / math.sqrt(config.num_heads * head_dim),
                storage_spec=sharding.row_storage,
                compute_spec=sharding.row_compute,
            ),
            _norm(config, head_dim, P(None)) if config.use_qk_norm else None,
            _norm(config, head_dim, P(None)) if config.use_qk_norm else None,
            config,
            sharding,
        )

    @jax.named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | Array,
        pos_ids: Int[Array, "B S"],
    ) -> Float[Array, "B S D"]:
        q, k, v = self.project_qkv(x, pos_ids, sharding=self.sharding)
        if self.config.attention_implementation == "reference":
            output = reference_attention(
                q,
                k,
                v,
                mask,
                logits_dtype=None,
                attention_dtype=jnp.float32 if self.config.upcast_attn else None,
            )
        else:
            if self.config.upcast_attn:
                q = q.astype(jnp.float32)
                k = k.astype(jnp.float32)
            output = attention(q, k, v, mask, implementation=self.config.attention_implementation)
        return self.project_output(output, dtype=x.dtype, sharding=self.sharding)

    def project_qkv(
        self,
        x: Float[Array, "B S D"],
        pos_ids: Int[Array, "B S"],
        *,
        sharding: NativeLlamaSharding,
    ) -> tuple[Array, Array, Array]:
        """Project and rotate queries/keys using the caller's token layout."""
        batch, sequence = x.shape[:2]
        dim = self.config.actual_head_dim
        q = self.q_proj(x, out_sharding=sharding.intermediate).reshape(batch, sequence, self.config.num_heads, dim)
        k = self.k_proj(x, out_sharding=sharding.intermediate).reshape(batch, sequence, self.config.num_kv_heads, dim)
        v = self.v_proj(x, out_sharding=sharding.intermediate).reshape(batch, sequence, self.config.num_kv_heads, dim)
        q = reshard(q, sharding.query)
        k = reshard(k, sharding.query)
        v = reshard(v, sharding.query)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rotary(q, pos_ids, self.config.rope)
        k = apply_rotary(k, pos_ids, self.config.rope)
        return q, k, v

    def project_output(
        self,
        output: Float[Array, "B S H D"],
        *,
        dtype: jnp.dtype,
        sharding: NativeLlamaSharding,
    ) -> Float[Array, "B S D"]:
        """Restore the residual layout after either dense or paged attention."""
        output = output.astype(dtype).reshape(*output.shape[:2], self.config.num_heads * self.config.actual_head_dim)
        return self.o_proj(output, out_sharding=sharding.residual)


class NativeLlamaMlp(eqx.Module):
    """Gated MLP with column-parallel inputs and a row-parallel output."""

    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear
    act: JaxActivationFunction = eqx.field(static=True)
    sharding: NativeLlamaSharding = eqx.field(static=True)

    @staticmethod
    def init(config: NativeLlamaConfig, sharding: NativeLlamaSharding, *, key: PRNGKeyArray):
        gate_key, up_key, down_key = jax.random.split(key, 3)

        def column(key):
            return Linear.init(
                config.hidden_dim,
                config.intermediate_dim,
                key=key,
                use_bias=config.use_bias,
                initializer_std=1 / math.sqrt(config.hidden_dim),
                storage_spec=sharding.column_storage,
                compute_spec=sharding.column_compute,
            )

        return NativeLlamaMlp(
            column(gate_key),
            column(up_key),
            Linear.init(
                config.intermediate_dim,
                config.hidden_dim,
                key=down_key,
                use_bias=config.use_bias,
                initializer_std=1 / math.sqrt(config.intermediate_dim),
                storage_spec=sharding.row_storage,
                compute_spec=sharding.row_compute,
            ),
            config.activation_function.to_jax_fn(),
            sharding,
        )

    @jax.named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        return self.forward(x, sharding=self.sharding)

    def forward(
        self,
        x: Float[Array, "B S D"],
        *,
        sharding: NativeLlamaSharding,
    ) -> Float[Array, "B S D"]:
        gate = self.act(self.gate_proj(x, out_sharding=sharding.intermediate))
        hidden = gate * self.up_proj(x, out_sharding=sharding.intermediate)
        return self.down_proj(hidden, out_sharding=sharding.residual)


class NativeLlamaBlock(eqx.Module):
    """Pre-normalized attention and MLP residual block."""

    self_attn: NativeLlamaAttention
    mlp: NativeLlamaMlp
    input_layernorm: RmsNorm
    post_attention_layernorm: RmsNorm
    post_attn_layernorm: RmsNorm | None
    post_mlp_layernorm: RmsNorm | None

    @staticmethod
    def init(config: NativeLlamaConfig, sharding: NativeLlamaSharding, *, key: PRNGKeyArray):
        attn_key, mlp_key = jax.random.split(key)
        return NativeLlamaBlock(
            NativeLlamaAttention.init(config, sharding, key=attn_key),
            NativeLlamaMlp.init(config, sharding, key=mlp_key),
            _norm(config, config.hidden_dim, sharding.norm_storage),
            _norm(config, config.hidden_dim, sharding.norm_storage),
            _norm(config, config.hidden_dim, sharding.norm_storage) if config.hybrid_norm else None,
            _norm(config, config.hidden_dim, sharding.norm_storage) if config.hybrid_norm else None,
        )

    @jax.named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | Array,
        pos_ids: Int[Array, "B S"],
    ) -> Float[Array, "B S D"]:
        attn = self.self_attn(self.input_layernorm(x), mask, pos_ids)
        return self.finish_attention(x, attn, sharding=self.self_attn.sharding)

    def finish_attention(
        self,
        x: Float[Array, "B S D"],
        attn: Float[Array, "B S D"],
        *,
        sharding: NativeLlamaSharding,
    ) -> Float[Array, "B S D"]:
        """Apply the shared residual/MLP computation after attention."""
        x = x + (self.post_attn_layernorm(attn) if self.post_attn_layernorm is not None else attn)
        mlp = self.mlp.forward(
            self.post_attention_layernorm(x),
            sharding=sharding,
        )
        return x + (self.post_mlp_layernorm(mlp) if self.post_mlp_layernorm is not None else mlp)


class NativeLlamaModel(eqx.Module):
    """Dense Llama with array-only forward, loss and parameter leaves."""

    token_embeddings: Float[Array, "V D"]
    layers: ArrayStacked[NativeLlamaBlock]
    norm: RmsNorm
    lm_head: Linear | None
    embedding_norm: RmsNorm | None
    config: NativeLlamaConfig = eqx.field(static=True)
    sharding: NativeLlamaSharding = eqx.field(static=True)

    @staticmethod
    def init(
        config: NativeLlamaConfig,
        *,
        key: PRNGKeyArray,
        sharding: NativeLlamaSharding = NativeLlamaSharding(),
    ) -> "NativeLlamaModel":
        if get_abstract_mesh().empty:
            raise ValueError("NativeLlamaModel requires an explicit mesh context via jax.set_mesh")
        # Embeddings and an untied head share their key, preserving the existing initialization.
        layer_key, embed_key = jax.random.split(key, 2)
        embeddings = (1 / config.hidden_dim) * jax.random.truncated_normal(
            embed_key,
            -INITIALIZER_TRUNCATION_BOUND,
            INITIALIZER_TRUNCATION_BOUND,
            (config.vocab_size, config.hidden_dim),
            out_sharding=sharding.column_storage,
        )
        # The one-layer initializer consumes its parent key without splitting.
        layer_keys = layer_key[None] if config.num_layers == 1 else jax.random.split(layer_key, config.num_layers)
        layers = ArrayStacked.init(
            config.num_layers, NativeLlamaBlock, gradient_checkpointing=config.gradient_checkpointing
        )(config, sharding, key=layer_keys)
        head = (
            None
            if config.tie_word_embeddings
            else Linear.init(
                config.hidden_dim,
                config.vocab_size,
                key=embed_key,
                use_bias=False,
                initializer_std=1 / math.sqrt(config.hidden_dim),
                storage_spec=sharding.column_storage,
                compute_spec=sharding.column_compute,
            )
        )
        return NativeLlamaModel(
            embeddings,
            layers,
            _norm(config, config.hidden_dim, sharding.norm_storage),
            head,
            _norm(config, config.hidden_dim, sharding.norm_storage) if config.input_embedding_norm else None,
            config,
            sharding,
        )

    @jax.named_call
    def activations(
        self,
        tokens: Int[Array, "B S"],
        mask: AttentionMask | Array | None = None,
        *,
        pos_ids: Int[Array, "B S"] | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B S D"]:
        """Return final normalized hidden states for a batch of token sequences."""
        # The public model interface supplies a key; this unquantized route has no stochastic forward operation.
        del key
        if tokens.shape[1] > self.config.max_seq_len:
            raise ValueError("Input sequence exceeds max_seq_len")
        tokens = reshard(tokens, self.sharding.tokens)
        if pos_ids is None:
            pos_ids = jnp.broadcast_to(jnp.arange(tokens.shape[1]), tokens.shape)
        pos_ids = reshard(pos_ids, self.sharding.tokens)
        mask = AttentionMask.causal() if mask is None else mask
        hidden = self.embed_tokens(tokens, sharding=self.sharding)
        hidden = self.layers.fold(hidden, mask, pos_ids)
        return self.norm(hidden)

    def embed_tokens(self, tokens: Int[Array, "B S"], *, sharding: NativeLlamaSharding) -> Float[Array, "B S D"]:
        # Gather from the stored shards so the pullback can accumulate repeated token IDs
        # without first replicating the weight over FSDP.
        hidden = self.token_embeddings.at[tokens].get(out_sharding=sharding.residual)
        return self.embedding_norm(hidden) if self.embedding_norm is not None else hidden

    def get_lm_head(self) -> Float[Array, "V D"]:
        """Return output-first head weights, sharing embeddings when tied."""
        return self.token_embeddings if self.lm_head is None else self.lm_head.weight

    def logits(
        self,
        tokens: Int[Array, "B S"],
        mask: AttentionMask | Array | None = None,
        *,
        pos_ids: Int[Array, "B S"] | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B S V"]:
        hidden = self.activations(tokens, mask, pos_ids=pos_ids, key=key)
        return self.unembed(hidden, sharding=self.sharding)

    def unembed(self, hidden: Float[Array, "B S D"], *, sharding: NativeLlamaSharding) -> Float[Array, "B S V"]:
        head = reshard(self.get_lm_head(), sharding.column_compute)
        return jnp.einsum("bsd,vd->bsv", hidden, head, out_sharding=sharding.logits)

    def __call__(
        self,
        tokens: Int[Array, "B S"],
        mask: AttentionMask | Array | None = None,
        *,
        pos_ids: Int[Array, "B S"] | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B S V"]:
        return self.logits(tokens, mask, pos_ids=pos_ids, key=key)

    def next_token_loss(
        self,
        tokens: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        mask: AttentionMask | Array | None = None,
        pos_ids: Int[Array, "B S"] | None = None,
        key: PRNGKeyArray | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        loss_dtype: jnp.dtype = jnp.float32,
        loss_implementation: str | tuple[str, ...] | None = None,
        logsumexp_weight: float | None = None,
    ) -> jax.Array:
        """Predict next tokens, excluding the final position of each sequence."""
        hidden = self.activations(tokens, mask, pos_ids=pos_ids, key=key)
        labels = jnp.roll(tokens, -1, axis=-1)
        weights = loss_weight.astype(loss_dtype).at[:, -1].set(0)
        return fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.get_lm_head().T,
            labels,
            weight=weights,
            reduction=reduction,
            dtype=loss_dtype,
            implementation=loss_implementation,
            logsumexp_weight=logsumexp_weight,
        )

    def to_state_dict(self, prefix: str | None = None) -> dict[str, jax.Array]:
        """Export canonical HF keys and output-first arrays, also for gradients."""
        state: dict[str, jax.Array] = {}
        _put(state, _HF_TOKEN_EMBEDDING_WEIGHT_KEY, self.token_embeddings)
        _put_module(state, _HF_FINAL_NORM_PREFIX, self.norm)
        if self.embedding_norm is not None:
            _put_module(state, "norm", self.embedding_norm)
        if self.lm_head is not None:
            _put_module(state, "lm_head", self.lm_head)
        for index, layer in enumerate(self.layers.unstacked()):
            path = f"{_HF_LAYER_PREFIX}{index}"
            for name in _LAYER_NORM_FIELDS:
                module = getattr(layer, name)
                if module is not None:
                    _put_module(state, f"{path}.{name}", module)
            for name in _ATTENTION_FIELDS:
                module = getattr(layer.self_attn, name)
                if module is not None:
                    _put_module(state, f"{path}.self_attn.{name}", module)
            for name in _MLP_FIELDS:
                _put_module(state, f"{path}.mlp.{name}", getattr(layer.mlp, name))
        return {f"{prefix}.{name}" if prefix else name: value for name, value in state.items()}

    def from_state_dict(self, state: Mapping[str, jax.Array], prefix: str | None = None) -> "NativeLlamaModel":
        """Load canonical or articulated legacy Llama arrays into native storage."""
        source = {name: state[f"{prefix}.{name}" if prefix else name] for name in self.to_state_dict()}
        model = eqx.tree_at(
            lambda m: m.token_embeddings,
            self,
            _load(source[_HF_TOKEN_EMBEDDING_WEIGHT_KEY], self.token_embeddings, self.sharding.column_storage),
        )
        model = eqx.tree_at(
            lambda m: m.norm,
            model,
            _load_module(model.norm, source, _HF_FINAL_NORM_PREFIX, self.sharding.norm_storage),
        )
        if model.embedding_norm is not None:
            model = eqx.tree_at(
                lambda m: m.embedding_norm,
                model,
                _load_module(model.embedding_norm, source, "norm", self.sharding.norm_storage),
            )
        if model.lm_head is not None:
            model = eqx.tree_at(
                lambda m: m.lm_head,
                model,
                _load_module(model.lm_head, source, "lm_head", self.sharding.column_storage),
            )
        layers = []
        for index, layer in enumerate(model.layers.unstacked()):
            path = f"{_HF_LAYER_PREFIX}{index}"
            for name in _LAYER_NORM_FIELDS:
                module = getattr(layer, name)
                if module is not None:
                    layer = dataclasses.replace(
                        layer, **{name: _load_module(module, source, f"{path}.{name}", self.sharding.norm_storage)}
                    )
            attn = layer.self_attn
            for name in _ATTENTION_FIELDS:
                module = getattr(attn, name)
                if module is None:
                    continue
                spec = self.sharding.row_storage if name == "o_proj" else self.sharding.column_storage
                if name in ("q_norm", "k_norm"):
                    spec = P(None)
                attn = dataclasses.replace(
                    attn, **{name: _load_module(module, source, f"{path}.self_attn.{name}", spec)}
                )
            mlp = layer.mlp
            for name in _MLP_FIELDS:
                spec = self.sharding.row_storage if name == "down_proj" else self.sharding.column_storage
                mlp = dataclasses.replace(
                    mlp, **{name: _load_module(getattr(mlp, name), source, f"{path}.mlp.{name}", spec)}
                )
            layers.append(dataclasses.replace(layer, self_attn=attn, mlp=mlp))
        stacked = jax.tree.map(lambda *arrays: jnp.stack(arrays), *layers)
        return eqx.tree_at(lambda m: m.layers.stacked, model, stacked)


def _put(state: dict[str, jax.Array], key: str, value: jax.Array | None) -> None:
    if value is not None:
        state[key] = value


def _put_module(state: dict[str, jax.Array], prefix: str, module: Linear | RmsNorm) -> None:
    assert isinstance(module, (Linear, RmsNorm))
    _put(state, f"{prefix}.weight", module.weight)
    _put(state, f"{prefix}.bias", module.bias)


def _load(value: jax.Array, template: jax.Array, spec: P) -> jax.Array:
    return reshard(jnp.asarray(value, dtype=template.dtype).reshape(template.shape), spec)


def _load_module(module: Linear | RmsNorm, state: Mapping[str, jax.Array], prefix: str, spec: P):
    if not isinstance(module, (Linear, RmsNorm)):
        raise ValueError("Load native base weights before applying adapters")
    weight = None if module.weight is None else _load(state[f"{prefix}.weight"], module.weight, spec)
    bias = None if module.bias is None else _load(state[f"{prefix}.bias"], module.bias, P(spec[0]))
    return dataclasses.replace(module, weight=weight, bias=bias)
