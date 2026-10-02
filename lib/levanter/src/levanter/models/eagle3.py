# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Array-first inference for the pinned Snowball EAGLE3 draft architecture."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, NamedTuple

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis, NamedArray
from jax.sharding import PartitionSpec as P, reshard

from levanter.compat.hf_checkpoints import load_safetensors_state_dict
from levanter.grug.attention import (
    AttentionMask,
    PagedAttentionImplementation,
    RotaryConfig,
    apply_rotary_embedding,
    attention,
    ragged_paged_attention,
)
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.layers.kv_cache import KvPageCache
from levanter.models.snowball import DenseMLP, RMSNorm


@dataclass(frozen=True)
class Eagle3Config:
    """The one-layer, normalized-residual Snowball draft checkpoint contract."""

    hidden_dim: int
    intermediate_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    draft_vocab_size: int
    auxiliary_layers: tuple[int, ...]
    sliding_window: int
    max_seq_len: int
    norm_eps: float
    rope_theta: float
    inference_attention_implementation: PagedAttentionImplementation | None = None

    @classmethod
    def from_hf_config(cls, config: Mapping[str, Any]) -> "Eagle3Config":
        """Read the pinned Speculators nested schema, rejecting other architectures."""
        layer = config["transformer_layer_config"]
        required = {
            "architectures": ["Eagle3DraftModel"],
            "speculators_model_type": "eagle3",
            "norm_before_fc": True,
            "norm_before_residual": True,
            "norm_output": True,
            "fc_norm": False,
            "tie_word_embeddings": False,
        }
        for key, value in required.items():
            if config.get(key) != value:
                raise ValueError(f"Snowball EAGLE3 requires {key}={value!r}")
        required_layer = {
            "model_type": "llama",
            "num_hidden_layers": 1,
            "hidden_act": "silu",
            "attention_bias": False,
            "mlp_bias": False,
            "layer_types": ["sliding_attention"],
            "use_sliding_window": True,
        }
        for key, value in required_layer.items():
            if layer.get(key) != value:
                raise ValueError(f"Snowball EAGLE3 requires transformer_layer_config.{key}={value!r}")
        if config.get("target_hidden_size") not in (None, layer["hidden_size"]):
            raise ValueError("Snowball EAGLE3 requires matching target and draft hidden dimensions")
        rope = layer["rope_parameters"]
        if rope["rope_type"] != "default":
            raise ValueError("Snowball EAGLE3 supports default RoPE only")
        auxiliary = tuple(config["eagle_aux_hidden_state_layer_ids"])
        if not auxiliary or auxiliary != tuple(sorted(set(auxiliary))) or auxiliary[0] < 0:
            raise ValueError("EAGLE auxiliary boundaries must be nonempty, unique and increasing")
        result = cls(
            hidden_dim=layer["hidden_size"],
            intermediate_dim=layer["intermediate_size"],
            num_heads=layer["num_attention_heads"],
            num_kv_heads=layer["num_key_value_heads"],
            head_dim=layer["head_dim"],
            vocab_size=layer["vocab_size"],
            draft_vocab_size=config["draft_vocab_size"],
            auxiliary_layers=auxiliary,
            sliding_window=layer["sliding_window"],
            max_seq_len=layer["max_position_embeddings"],
            norm_eps=layer["rms_norm_eps"],
            rope_theta=rope["rope_theta"],
        )
        if result.num_heads % result.num_kv_heads or result.head_dim % 2:
            raise ValueError("EAGLE requires grouped query heads and even rotary head dimensions")
        return result


class _AttentionInputs(NamedTuple):
    residual: jax.Array
    query: jax.Array
    key: jax.Array
    value: jax.Array


class Eagle3Output(NamedTuple):
    logits: jax.Array  # Draft vocabulary rows, not expanded to the target vocabulary.
    hidden_states: jax.Array  # Normalized recurrent state for the next draft step.


class Eagle3DecodeOutput(NamedTuple):
    output: Eagle3Output
    cache: KvPageCache


class Eagle3Draft(eqx.Module):
    """Learned greedy draft with target-owned embeddings and an independent KV cache."""

    config: Eagle3Config = eqx.field(static=True)
    embedding: jax.Array
    input_norm: RMSNorm
    fc: jax.Array
    embed_norm: RMSNorm
    hidden_norm: RMSNorm
    post_attention_norm: RMSNorm
    final_norm: RMSNorm
    q_proj: jax.Array
    k_proj: jax.Array
    v_proj: jax.Array
    o_proj: jax.Array
    mlp: DenseMLP
    lm_head: jax.Array
    draft_to_target: jax.Array

    @classmethod
    def from_checkpoint(cls, directory: Path, *, target_embedding: jax.Array) -> "Eagle3Draft":
        """Load config.json and model.safetensors from an embedding-free local export."""
        config = Eagle3Config.from_hf_config(json.loads((directory / "config.json").read_text()))
        state = load_safetensors_state_dict(str(directory / "model.safetensors"))
        return cls.from_state_dict(config, state, target_embedding=target_embedding)

    @classmethod
    def from_state_dict(
        cls,
        config: Eagle3Config,
        state: Mapping[str, np.ndarray | jax.Array],
        *,
        target_embedding: jax.Array,
    ) -> "Eagle3Draft":
        """Load exact HF keys, including offset d2t and its inverse boolean mask.

        The public checkpoint omits embeddings. They are supplied by the installed
        target without reconstructing another verifier or overwriting the draft head.
        """
        h, d, n, m = config.hidden_dim, config.head_dim, config.num_heads, config.num_kv_heads
        if target_embedding.shape != (config.vocab_size, h):
            raise ValueError("Target embedding does not match the EAGLE checkpoint architecture")
        expected = set()
        dtype = target_embedding.dtype

        def weight(name, shape):
            expected.add(name)
            value = state[name]
            if value.shape != shape:
                raise ValueError(f"{name}: expected shape {shape}, got {value.shape}")
            return reshard(jnp.asarray(value, dtype=dtype), P(*([None] * len(shape))))

        def norm(name, size):
            return RMSNorm(weight(name + ".weight", (size,)), config.norm_eps)

        def linear(name, rows, columns):
            return weight(name + ".weight", (rows, columns)).T

        offsets = np.asarray(state["d2t"])
        target_mask = np.asarray(state["t2d"])
        expected.update(("d2t", "t2d"))
        if offsets.shape != (config.draft_vocab_size,) or not np.issubdtype(offsets.dtype, np.integer):
            raise ValueError("d2t must contain one integer offset per draft vocabulary row")
        target_ids = offsets.astype(np.int64) + np.arange(config.draft_vocab_size)
        if (target_ids < 0).any() or (target_ids >= config.vocab_size).any() or (np.diff(target_ids) <= 0).any():
            raise ValueError("d2t must map to increasing unique target vocabulary IDs")
        if target_mask.shape != (config.vocab_size,) or target_mask.dtype != np.bool_:
            raise ValueError("t2d must be the target vocabulary boolean mask")
        if not np.array_equal(np.flatnonzero(target_mask), target_ids):
            raise ValueError("d2t and t2d describe different vocabulary mappings")
        prefix = "layers.0"
        result = cls(
            config=config,
            embedding=reshard(target_embedding, P(None, None)),
            input_norm=norm("input_norm", h * len(config.auxiliary_layers)),
            fc=linear("fc", h, h * len(config.auxiliary_layers)),
            embed_norm=norm(prefix + ".input_layernorm", h),
            hidden_norm=norm(prefix + ".hidden_norm", h),
            post_attention_norm=norm(prefix + ".post_attention_layernorm", h),
            final_norm=norm("norm", h),
            q_proj=linear(prefix + ".self_attn.q_proj", n * d, 2 * h),
            k_proj=linear(prefix + ".self_attn.k_proj", m * d, 2 * h),
            v_proj=linear(prefix + ".self_attn.v_proj", m * d, 2 * h),
            o_proj=linear(prefix + ".self_attn.o_proj", h, n * d),
            mlp=DenseMLP(
                linear(prefix + ".mlp.gate_proj", config.intermediate_dim, h),
                linear(prefix + ".mlp.up_proj", config.intermediate_dim, h),
                linear(prefix + ".mlp.down_proj", h, config.intermediate_dim),
            ),
            lm_head=linear("lm_head", config.draft_vocab_size, h),
            draft_to_target=reshard(jnp.asarray(target_ids, dtype=jnp.int32), P(None)),
        )
        if set(state) != expected:
            raise ValueError(f"Unexpected EAGLE checkpoint tensors: {sorted(set(state) - expected)}")
        return result

    def trainable_state_dict(self) -> dict[str, jax.Array]:
        """Export the trainable-only tensor contract used by the online draft trainer."""
        return {
            "fc.weight": self.fc.T,
            "input_norm.weight": self.input_norm.weight,
            "layers.0.input_layernorm.weight": self.embed_norm.weight,
            "layers.0.hidden_norm.weight": self.hidden_norm.weight,
            "layers.0.post_attention_layernorm.weight": self.post_attention_norm.weight,
            "norm.weight": self.final_norm.weight,
            "layers.0.self_attn.q_proj.weight": self.q_proj.T,
            "layers.0.self_attn.k_proj.weight": self.k_proj.T,
            "layers.0.self_attn.v_proj.weight": self.v_proj.T,
            "layers.0.self_attn.o_proj.weight": self.o_proj.T,
            "layers.0.mlp.gate_proj.weight": self.mlp.w_gate.T,
            "layers.0.mlp.up_proj.weight": self.mlp.w_up.T,
            "layers.0.mlp.down_proj.weight": self.mlp.w_down.T,
        }

    def with_trainable_state_dict(self, state: Mapping[str, np.ndarray | jax.Array]) -> "Eagle3Draft":
        """Stage a complete trainable overlay, preserving target-owned tensors and maps."""
        current = self.trainable_state_dict()
        if set(state) != set(current):
            raise ValueError(
                f"Draft update tensor mismatch: missing {sorted(set(current) - set(state))}, "
                f"unexpected {sorted(set(state) - set(current))}"
            )
        weights = {}
        finite = []
        for name, old in current.items():
            new = state[name]
            if new.shape != old.shape or new.dtype != old.dtype:
                raise ValueError(f"Draft update {name} must preserve shape and dtype")
            weights[name] = jax.device_put(new, old.sharding)
            finite.append(jnp.all(jnp.isfinite(weights[name])))
        if not bool(jnp.all(jnp.stack(finite))):
            raise ValueError("Draft update contains nonfinite weights")
        weights["lm_head.weight"] = self.lm_head.T
        weights["d2t"] = self.draft_to_target - jnp.arange(self.config.draft_vocab_size)
        weights["t2d"] = jnp.zeros((self.config.vocab_size,), jnp.bool_).at[self.draft_to_target].set(True)
        return type(self).from_state_dict(self.config, weights, target_embedding=self.embedding)

    def with_trainable_checkpoint(self, path: Path) -> "Eagle3Draft":
        """Stage a local single-file checkpoint emitted by the online draft trainer."""
        return self.with_trainable_state_dict(load_safetensors_state_dict(str(path)))

    def with_target_weights(self, embedding: jax.Array, output_projection: jax.Array) -> "Eagle3Draft":
        """Refresh target embeddings and mapped target-head rows after policy publication."""
        if embedding.shape != self.embedding.shape or output_projection.shape != (
            self.config.hidden_dim,
            self.config.vocab_size,
        ):
            raise ValueError("Published target weights do not match the draft architecture")
        head = output_projection.at[:, self.draft_to_target].get(out_sharding=P(None, None))
        return eqx.tree_at(
            lambda model: (model.embedding, model.lm_head),
            self,
            (reshard(embedding, P(None, None)), head),
        )

    def project_target_states(self, auxiliary: jax.Array) -> jax.Array:
        """Project concatenated target boundaries once; recurrent draft states bypass FC."""
        if auxiliary.shape[-1] != self.fc.shape[0]:
            raise ValueError("EAGLE target auxiliary width does not match configured boundaries")
        return jnp.einsum("...d,dh->...h", self.input_norm(auxiliary), self.fc)

    def greedy_tokens(self, output: Eagle3Output) -> jax.Array:
        return self.draft_to_target.at[jnp.argmax(output.logits, axis=-1)].get(
            out_sharding=P(*([None] * (output.logits.ndim - 1)))
        )

    def _inputs(self, token_ids, hidden_states, positions):
        cfg = self.config
        embeddings = self.embedding.at[token_ids].get(out_sharding=P(None, None, None))
        normalized_hidden = self.hidden_norm(hidden_states)
        x = jnp.concatenate([self.embed_norm(embeddings), normalized_hidden], axis=-1)
        q = jnp.einsum("...h,hd->...d", x, self.q_proj).reshape(*x.shape[:-1], cfg.num_heads, cfg.head_dim)
        k = jnp.einsum("...h,hd->...d", x, self.k_proj).reshape(*x.shape[:-1], cfg.num_kv_heads, cfg.head_dim)
        v = jnp.einsum("...h,hd->...d", x, self.v_proj).reshape(*x.shape[:-1], cfg.num_kv_heads, cfg.head_dim)
        q, k = apply_rotary_embedding(
            q,
            k,
            seq_len=x.shape[1],
            head_dim=cfg.head_dim,
            rope=RotaryConfig(theta=cfg.rope_theta),
            position_ids=positions,
        )
        return _AttentionInputs(normalized_hidden, q, k, v)

    def _output(self, residual, attended):
        attended = attended.reshape(*attended.shape[:-2], -1)
        x = residual + jnp.einsum("...h,hd->...d", attended, self.o_proj)
        x = x + self.mlp(self.post_attention_norm(x))
        hidden = self.final_norm(x)
        return Eagle3Output(jnp.einsum("...h,hv->...v", hidden, self.lm_head), hidden)

    def __call__(self, token_ids: jax.Array, auxiliary_states: jax.Array) -> Eagle3Output:
        """Score full causal sequences from target residual states [batch, position, features]."""
        hidden = self.project_target_states(auxiliary_states)
        residual, q, k, v = self._inputs(token_ids, hidden, None)
        attended = attention(
            q,
            k,
            v,
            AttentionMask(is_causal=True, sliding_window=self.config.sliding_window),
            implementation="reference",
        )
        return self._output(residual, attended)

    def initial_cache(self, spec: PageTableSpec, *, dtype) -> KvPageCache:
        cache = KvPageCache.init(
            spec, Axis("kv_head", self.config.num_kv_heads), Axis("head_size", self.config.head_dim), dtype
        )
        return KvPageCache(hax.named(reshard(cache.kv_pages.array, P(None, None, None, None)), cache.kv_pages.axes))

    def decode(
        self,
        token_ids: NamedArray,
        hidden_states: jax.Array,
        cache: KvPageCache,
        batch_info: PageBatchInfo,
        positions: NamedArray,
    ) -> Eagle3DecodeOutput:
        """Consume projected target or recurrent draft states with independent paged KV."""
        cfg = self.config
        residual, q, k, v = self._inputs(token_ids.array[:, None], hidden_states[:, None], positions.array[:, None])
        axes = (token_ids.axes[0], Axis("kv_head", cfg.num_kv_heads), Axis("head_size", cfg.head_dim))
        cache = cache.update(batch_info, hax.named(k[:, 0], axes), hax.named(v[:, 0], axes))
        q = q.reshape(token_ids.size, cfg.num_kv_heads, cfg.num_heads // cfg.num_kv_heads, cfg.head_dim)
        attended = ragged_paged_attention(
            q,
            cache.kv_pages.array,
            batch_info.seq_lens.array,
            batch_info.page_indices.array,
            batch_info.cu_q_lens.array,
            batch_info.num_seqs,
            sm_scale=cfg.head_dim**-0.5,
            sliding_window=cfg.sliding_window,
            implementation=cfg.inference_attention_implementation,
        ).reshape(token_ids.size, 1, cfg.num_heads, cfg.head_dim)
        output = self._output(residual, attended)
        return Eagle3DecodeOutput(Eagle3Output(output.logits[:, 0], output.hidden_states[:, 0]), cache)
