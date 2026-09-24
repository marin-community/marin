# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dense replication of the arXiv 2609.19107 base-size operator study.

Architecture (paper A.1): pre-norm decoder-only transformer with full RoPE,
SwiGLU MLPs, non-parametric QK-norm before RoPE, no biases, no learned norm
gains, an RMSNorm after the token embedding and before the LM head. Attention
output projections and MLP down-projections are zero-initialized, the token
embedding is normal-initialized, and attention/MLP input matrices are
uniform-initialized (paper Table 5: WTE / UIS). The residual multiplier (RM)
scales attention output and MLP down projections; the output multiplier (OM)
scales the LM head; both follow the paper's F(alpha*Theta) semantics as
forward-time scalings of the stored weights.

The boundary operator (paper Eq. 2/3, K=1) is the Operator-1 variant: the
prelude output e is re-injected at the core and coda entries.
"""

import dataclasses
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from haliax.jax_utils import named_call
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.grug.attention import AttentionMask, RotaryConfig, apply_rotary_embedding, attention
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pbatch, Pembed_vocab, Plm_head, Plogits

PAPER_VOCAB_SIZE = 50_304


@dataclass(frozen=True)
class GrugModelConfig:
    """Hyperparameters for the paper-replication dense transformer.

    Shape knobs default to the paper's base size (d8: width 1024 = 128 * depth,
    d_ff = 3 * width). Init/multiplier knobs default to the paper's Table 3
    initial values; launch-time recipes override them with the Table 5 values.
    """

    vocab_size: int = PAPER_VOCAB_SIZE
    hidden_dim: int = 1024
    intermediate_dim: int = 3072
    num_layers: int = 8
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int | None = None
    max_seq_len: int = 2048
    layer_norm_eps: float = 1e-5
    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)

    # Paper Table 5 model-side recipe knobs.
    embed_init_std: float = 0.08
    """WTE: token-embedding init std (normal)."""
    input_init_scale: float = 0.25
    """UIS: input-matrix init bound (uniform in [-UIS, UIS])."""
    residual_multiplier: float = 0.5
    """RM: forward-time multiplier on attention output and MLP down projections."""
    output_multiplier: float = 1.0
    """OM: forward-time multiplier on the LM head."""

    # Boundary operator (paper Eq. 2/3, K=1).
    boundary_operator: bool = False
    """Operator-1 wiring: the prelude output e is re-injected at the core entry
    (state h_0 = 0, so BO(0, e) = injection_scale * e) and at the coda entry
    (BO(h, e) = rms_norm(h) + injection_scale * e). False reproduces the
    vanilla forward exactly."""
    prelude_len: int | None = None
    """Blocks in the prelude P. Required when boundary_operator is set."""
    coda_len: int | None = None
    """Blocks in the coda D. Required when boundary_operator is set."""
    injection_scale: float = 1.0
    """Boundary-operator injection weight alpha (paper Table 5: 1 for Operator-1)."""

    def __post_init__(self) -> None:
        _ = self.inferred_head_dim
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for grouped-query attention")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.intermediate_dim <= 0:
            raise ValueError("intermediate_dim must be positive")
        if self.boundary_operator:
            if self.prelude_len is None or self.coda_len is None:
                raise ValueError("boundary_operator requires prelude_len and coda_len")
            if not 0 <= self.prelude_len <= self.num_layers:
                raise ValueError("prelude_len must be within [0, num_layers]")
            if not 0 <= self.coda_len <= self.num_layers - self.prelude_len:
                raise ValueError("prelude_len + coda_len must be within [0, num_layers]")
            if self.injection_scale < 0:
                raise ValueError("injection_scale must be non-negative")
        elif self.prelude_len is not None or self.coda_len is not None or self.injection_scale != 1.0:
            raise ValueError("prelude_len, coda_len, and injection_scale require boundary_operator=True")

    @property
    def inferred_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} is not divisible by num_heads={self.num_heads}; set head_dim explicitly"
            )
        return self.hidden_dim // self.num_heads


def rms_norm(x: jax.Array, eps: float) -> jax.Array:
    """Non-parametric RMSNorm (paper A.1: no learned norm gains)."""
    dtype = x.dtype
    x = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(variance + eps)).astype(dtype)


class BlockSplit(eqx.Module):
    """Layer counts for the prelude / core / coda partition of a boundary-operator model."""

    prelude: int
    core: int
    coda: int


def split_prelude_core_coda(num_layers: int) -> BlockSplit:
    """Paper Table 2 allocation: even split, remainders to the core first then the coda.

    d6 -> 2/2/2, d8 -> 2/3/3, d11 -> 3/4/4, d13 -> 4/5/4, d26 -> 8/9/9.
    """
    if num_layers < 0:
        raise ValueError(f"num_layers must be non-negative, got {num_layers}")
    base = num_layers // 3
    remainder = num_layers % 3
    core_extra = min(remainder, 1)
    coda_extra = remainder - core_extra
    return BlockSplit(prelude=base, core=base + core_extra, coda=base + coda_extra)


def _init_uniform(key: PRNGKeyArray, shape: tuple[int, ...], bound: float) -> Float[Array, "..."]:
    return random.uniform(key, shape, minval=-bound, maxval=bound)


def _init_normal(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.normal(key, shape)


def _init_zero(shape: tuple[int, ...]) -> Float[Array, "..."]:
    return jnp.zeros(shape, dtype=jnp.float32)


class CausalSelfAttention(eqx.Module):
    w_q: jax.Array
    w_k: jax.Array
    w_v: jax.Array
    w_o: jax.Array
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        k_q, k_k, k_v = random.split(key, 3)
        d_model, n_heads, n_kv_heads, head_dim = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.inferred_head_dim
        uis = cfg.input_init_scale
        return CausalSelfAttention(
            w_q=reshard(_init_uniform(k_q, (d_model, n_heads * head_dim), uis), P("data", "model")),
            w_k=reshard(_init_uniform(k_k, (d_model, n_kv_heads * head_dim), uis), P("data", "model")),
            w_v=reshard(_init_uniform(k_v, (d_model, n_kv_heads * head_dim), uis), P("data", "model")),
            # Zero-init output projection (paper A.1); the residual multiplier is
            # applied to the stored weights at forward time (F(alpha*Theta)).
            w_o=reshard(_init_zero((n_heads * head_dim, d_model)), P("model", "data")),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]

        q = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_v), "... (m d) -> ... m d", d=head_dim)
        # QK-norm before RoPE (paper A.1; non-parametric over the head dim).
        q = rms_norm(q, self.cfg.layer_norm_eps)
        k = rms_norm(k, self.cfg.layer_norm_eps)
        q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope=self.cfg.rope)
        attn_out = attention(q, k, v, mask)
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o * self.cfg.residual_multiplier, out_sharding=Pbatch)


class MLP(eqx.Module):
    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "MLP":
        k_gate, k_up = random.split(key, 2)
        d_model, d_ff = cfg.hidden_dim, cfg.intermediate_dim
        uis = cfg.input_init_scale
        return MLP(
            w_gate=reshard(_init_uniform(k_gate, (d_model, d_ff), uis), P("data", "model")),
            w_up=reshard(_init_uniform(k_up, (d_model, d_ff), uis), P("data", "model")),
            w_down=reshard(_init_zero((d_ff, d_model)), P("model", "data")),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        # SwiGLU (paper A.1). The down projection is zero-initialized; the
        # residual multiplier is applied to the stored weights at forward time.
        gate = jax.nn.silu(jnp.einsum("bsh,hm->bsm", x, self.w_gate))
        up = jnp.einsum("bsh,hm->bsm", x, self.w_up)
        return jnp.einsum("bsm,mh->bsh", gate * up, self.w_down * self.cfg.residual_multiplier, out_sharding=Pbatch)


class Block(eqx.Module):
    attn: CausalSelfAttention
    mlp: MLP
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Block":
        attn_key, mlp_key = random.split(key, 2)
        return Block(
            attn=CausalSelfAttention.init(cfg, key=attn_key),
            mlp=MLP.init(cfg, key=mlp_key),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        eps = self.cfg.layer_norm_eps
        x = x + self.attn(rms_norm(x, eps), mask)
        x = x + self.mlp(rms_norm(x, eps))
        return x


class Transformer(eqx.Module):
    token_embed: jax.Array
    output_proj: jax.Array
    blocks: tuple[Block, ...]
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Transformer":
        embed_key, *block_keys = random.split(key, cfg.num_layers + 1)
        token_embed = reshard(
            _init_normal(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.embed_init_std), Pembed_vocab
        )
        output_proj = reshard(_init_zero((cfg.hidden_dim, cfg.vocab_size)), Plm_head)
        blocks = tuple(Block.init(cfg, key=layer_key) for layer_key in block_keys)
        return Transformer(
            token_embed=token_embed,
            output_proj=output_proj,
            blocks=blocks,
            config=cfg,
        )

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S D"]:
        if mask is None:
            mask = AttentionMask.causal()

        cfg = self.config
        eps = cfg.layer_norm_eps

        with jax.named_scope("token_embed"):
            hidden = self.token_embed.at[token_ids].get(out_sharding=Pbatch)
            # Paper A.1: an RMSNorm follows the token embedding.
            hidden = rms_norm(hidden, eps)

        num_blocks = len(self.blocks)
        boundary = cfg.boundary_operator
        # Paper Eq. 2/3 (K = 1): e is the prelude output P(s); the core state
        # starts at h_0 = 0, so the state entering the core is
        # BO(0, e) = injection_scale * e, and the state entering the coda is
        # BO(h, e) = rms_norm(h) + injection_scale * e. With prelude_len == 0
        # the prelude is empty and e falls back to the embedded input.
        prelude_len = cfg.prelude_len if boundary else 0
        coda_len = cfg.coda_len if boundary else 0
        core_len = num_blocks - prelude_len - coda_len
        e = hidden
        for i, block in enumerate(self.blocks):
            if boundary and i == prelude_len:
                hidden = cfg.injection_scale * e
            if boundary and i == prelude_len + core_len:
                hidden = rms_norm(hidden, eps) + cfg.injection_scale * e
            with jax.named_scope(f"block_{i}"):
                hidden = eqx.filter_checkpoint(block)(hidden, mask)
            if boundary and i == prelude_len - 1:
                e = hidden

        # Paper A.1: an RMSNorm precedes the LM head.
        with jax.named_scope("final_norm"):
            return rms_norm(hidden, eps)

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        hidden = self(token_ids, mask=mask)
        return jnp.einsum("bsh,hd->bsd", hidden, self.output_proj * self.config.output_multiplier, out_sharding=Plogits)

    def next_token_loss(
        self,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        mask: AttentionMask | jax.Array | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
        loss_implementation: str | tuple[str, ...] | None = None,
    ) -> jax.Array:
        """Compute next-token cross-entropy loss for a batch."""
        hidden = self(token_ids, mask=mask)
        labels = jnp.pad(token_ids[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)

        return fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.output_proj * self.config.output_multiplier,
            labels,
            weight=loss_weight,
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
            implementation=loss_implementation,
        )


def debug_mesh_and_token_pspec(num_devices: int, model_axis_size: int = 1) -> tuple[jax.sharding.AbstractMesh, P]:
    """Return a small abstract mesh and token sharding for lowering contract tests."""
    if num_devices <= 0:
        raise ValueError(f"num_devices must be positive, got {num_devices}")
    if model_axis_size <= 0:
        raise ValueError(f"model_axis_size must be positive, got {model_axis_size}")
    if num_devices % model_axis_size != 0:
        raise ValueError(f"num_devices ({num_devices}) must be divisible by model_axis_size ({model_axis_size})")
    data_axis_size = num_devices // model_axis_size
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(1, data_axis_size, model_axis_size),
        axis_names=("replica_dcn", "data", "model"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    return mesh, P(("replica_dcn", "data"), None)


__all__ = [
    "MLP",
    "PAPER_VOCAB_SIZE",
    "Block",
    "BlockSplit",
    "CausalSelfAttention",
    "GrugModelConfig",
    "Transformer",
    "debug_mesh_and_token_pspec",
    "rms_norm",
    "split_prelude_core_coda",
]
