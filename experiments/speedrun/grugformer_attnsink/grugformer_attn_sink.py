# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Grugformer speedrun with attention sinks (TPU splash attention `sinks`).

Analogue of:
  experiments/speedrun/hackable_transformer_starter/hackable_transformer_attn_sink.py

How to run:
  python marin/run/ray_run.py -- \
    python -m experiments.speedrun.grugformer_attnsink.grugformer_attn_sink
"""

# nodryrun

import dataclasses
import functools
import logging
import math
import os

import jax
import jax.numpy as jnp
from einops import rearrange
from fray.cluster import ResourceConfig
from haliax import Axis
from jax.sharding import PartitionSpec
from jaxtyping import PRNGKeyArray

from levanter.grug.attention import AttentionMask, apply_rotary_embedding
from levanter.grug.model import GrugModelConfig
from levanter.grug.model import rms_norm, mlp
from levanter.models.grug_wrapper import GrugWrapper
from levanter.models.lm_model import LmConfig
from levanter.utils.flop_utils import lm_flops_per_token
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

from experiments.llama import llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger("ray")


def _get_num_train_steps(param_count: int, batch_size: int, max_seq_len: int, tpp: int = 20) -> int:
    total_tokens = param_count * tpp
    return max(1, total_tokens // (batch_size * max_seq_len))


def _resource_presets(use_tpu: bool = False):
    if use_tpu:
        return {
            "130m": ResourceConfig.with_tpu("v5p-8"),
            "300m": ResourceConfig.with_tpu("v5p-8"),
            "520m": ResourceConfig.with_tpu("v5p-8"),
            "1_2b": ResourceConfig.with_tpu("v5p-8"),
        }
    return {
        "130m": ResourceConfig.with_gpu("A100-80G", count=1),
        "300m": ResourceConfig.with_gpu("A100-80G", count=1),
        "520m": ResourceConfig.with_gpu("A100-80G", count=2),
        "1_2b": ResourceConfig.with_gpu("A100-80G", count=4),
    }


def _batch_sizes() -> dict[str, int]:
    return {"130m": 128, "300m": 128, "520m": 128, "1_2b": 256}


def _size_presets() -> dict[str, "GrugformerAttnSinkLmConfig"]:
    base = dict(max_seq_len=2048, head_dim=None)
    return {
        "130m": GrugformerAttnSinkLmConfig(
            hidden_dim=512,
            intermediate_dim=1792,
            num_layers=6,
            num_heads=8,
            num_kv_heads=8,
            **base,
        ),
        "300m": GrugformerAttnSinkLmConfig(
            hidden_dim=768,
            intermediate_dim=2688,
            num_layers=12,
            num_heads=12,
            num_kv_heads=12,
            **base,
        ),
        "520m": GrugformerAttnSinkLmConfig(
            hidden_dim=1024,
            intermediate_dim=3584,
            num_layers=24,
            num_heads=16,
            num_kv_heads=16,
            **base,
        ),
        "1_2b": GrugformerAttnSinkLmConfig(
            hidden_dim=2048,
            intermediate_dim=7168,
            num_layers=16,
            num_heads=16,
            num_kv_heads=16,
            **base,
        ),
    }


@dataclasses.dataclass(frozen=True)
class GrugformerAttnSinkConfig:
    """Extra knobs for the sink speedrun (kept out of grug core config)."""

    num_sinks: int = 1
    init_logit: float = 0.0


@dataclasses.dataclass(frozen=True)
class GrugformerAttnSinkModelConfig:
    """Config object carried by GrugWrapper for this speedrun.

    This is separate from `GrugModelConfig` to keep the sink knobs out of grug core.
    """

    core: GrugModelConfig
    sink: GrugformerAttnSinkConfig

    @property
    def vocab_size(self) -> int:
        return self.core.vocab_size

    @property
    def max_seq_len(self) -> int:
        return self.core.max_seq_len

    @property
    def hidden_dim(self) -> int:
        return self.core.hidden_dim


def _init_grugformer_with_sinks(cfg: GrugformerAttnSinkModelConfig, *, key: PRNGKeyArray) -> dict:
    core = _init_core(cfg.core, key)

    # Initialize sink logits per layer.
    # Use a small constant init by default; users can set init_logit to control initial sink mass.
    #
    # NOTE: splash attention takes one sink logit per query head.
    if cfg.sink.num_sinks != 1:
        raise NotImplementedError("This speedrun currently supports only num_sinks=1.")
    sink_init = jnp.full((cfg.core.num_heads,), cfg.sink.init_logit, dtype=jnp.float32)
    sink_logits = tuple(sink_init for _ in range(cfg.core.num_layers))

    return {"core": core, "sink_logits": sink_logits}


def _init_core(core_cfg: GrugModelConfig, key: PRNGKeyArray):
    from levanter.grug.model import init_parameters as grug_init

    return grug_init(core_cfg, key=key)


def _splash_attention_with_sink(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: AttentionMask | jax.Array | None,
    *,
    softmax_aux: jax.Array | None,
) -> jax.Array:
    if not isinstance(mask, AttentionMask) and mask is not None:
        raise NotImplementedError("Only Grug AttentionMask is supported in this speedrun.")

    if mask is None:
        mask = AttentionMask.causal()

    if softmax_aux is None:
        raise ValueError("softmax_aux (sink logits) must be provided for the attn sink speedrun.")

    if jax.default_backend() != "tpu":
        raise NotImplementedError("This speedrun currently uses TPU splash attention.")

    from jax.experimental.pallas.ops.tpu.splash_attention import SegmentIds as SplashSegmentIds
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask

    mesh = jax.sharding.get_abstract_mesh()
    if mesh is None:
        raise RuntimeError("Splash attention requires a JAX mesh to be set.")

    _batch, q_len, num_q_heads, head_dim = q.shape
    kv_len = k.shape[1]
    num_kv_heads = k.shape[2]

    if q_len != kv_len:
        raise NotImplementedError("Splash attention speedrun currently assumes Sq == Sk.")

    if num_kv_heads != num_q_heads:
        raise NotImplementedError("This speedrun currently requires num_kv_heads == num_heads for splash attention.")

    if kv_len % 128 != 0:
        raise NotImplementedError("Splash attention requires the KV sequence length to be a multiple of 128.")

    scaling_factor = 1.0 / math.sqrt(head_dim)
    q = q * scaling_factor

    q_bhsd = q.transpose(0, 2, 1, 3)
    k_bhsd = k.transpose(0, 2, 1, 3)
    v_bhsd = v.transpose(0, 2, 1, 3)

    if mask is None:
        base_mask = splash_attention_mask.FullMask(_shape=(q_len, kv_len))
    else:
        base_mask = splash_attention_mask.FullMask(_shape=(q_len, kv_len))
        if mask.is_causal:
            base_mask = splash_attention_mask.CausalMask((q_len, kv_len), offset=0, shard_count=1)
        if mask.sliding_window is not None:
            local_mask = splash_attention_mask.LocalMask(
                shape=(q_len, kv_len),
                window_size=(mask.sliding_window - 1, None),
                offset=0,
                shard_count=1,
            )
            base_mask = splash_attention_mask.LogicalAnd(base_mask, local_mask)

    kernel_mask = splash_attention_mask.MultiHeadMask(masks=[base_mask for _ in range(num_q_heads)])
    kernel = splash_attention_kernel.make_splash_mha(
        mask=kernel_mask,
        head_shards=1,
        q_seq_shards=1,
        block_sizes=splash_attention_kernel.BlockSizes.get_default(),
        attn_logits_soft_cap=None,
    )

    sinks = softmax_aux.astype(jnp.float32)

    segment_ids = None
    if mask is not None and mask.segment_ids is not None:
        q_seg, kv_seg = mask.segment_ids
        segment_ids = SplashSegmentIds(q_seg.astype(jnp.int32), kv_seg.astype(jnp.int32))

    q_spec = PartitionSpec(("data",), None, None, None)
    k_spec = PartitionSpec(("data",), None, None, None)
    v_spec = PartitionSpec(("data",), None, None, None)
    sinks_spec = PartitionSpec(None)

    if segment_ids is None:

        @functools.partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(q_spec, k_spec, v_spec, sinks_spec),
            out_specs=q_spec,
            check_vma=False,
        )
        def _call_splash_attention(q_, k_, v_, sinks_):
            return jax.vmap(lambda q_b, k_b, v_b: kernel(q_b, k_b, v_b, sinks=sinks_), in_axes=(0, 0, 0))(q_, k_, v_)

        out_bhsd = _call_splash_attention(q_bhsd, k_bhsd, v_bhsd, sinks)
    else:
        segment_ids_spec = SplashSegmentIds(
            PartitionSpec(("data",), None),
            PartitionSpec(("data",), None),
        )

        @functools.partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(q_spec, k_spec, v_spec, segment_ids_spec, sinks_spec),
            out_specs=q_spec,
            check_vma=False,
        )
        def _call_splash_attention(q_, k_, v_, seg_ids, sinks_):
            return jax.vmap(
                lambda q_b, k_b, v_b, si: kernel(q_b, k_b, v_b, segment_ids=si, sinks=sinks_),
                in_axes=(0, 0, 0, 0),
            )(q_, k_, v_, seg_ids)

        out_bhsd = _call_splash_attention(q_bhsd, k_bhsd, v_bhsd, segment_ids, sinks)
    return out_bhsd.transpose(0, 2, 1, 3)


def _grug_activations_with_sinks(
    params: dict,
    token_ids: jax.Array,
    cfg: GrugformerAttnSinkModelConfig,
    *,
    mask: AttentionMask | jax.Array | None = None,
) -> jax.Array:
    cfg = cfg.core
    head_dim = cfg.inferred_head_dim
    seq_len = token_ids.shape[1]

    if mask is None:
        mask = AttentionMask.causal()
    elif isinstance(mask, AttentionMask) and not mask.is_causal:
        mask = dataclasses.replace(mask, is_causal=True)

    core = params["core"]
    hidden = core.token_embed.at[token_ids].get(out_sharding=_PBATCH)

    for block, sink_logits in zip(core.blocks, params["sink_logits"], strict=True):
        attn_in = rms_norm(hidden, block.rms_attn, cfg.layer_norm_eps)

        q = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_v), "... (m d) -> ... m d", d=head_dim)
        q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope=cfg.rope)

        attn_out = _splash_attention_with_sink(q, k, v, mask, softmax_aux=sink_logits)
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        attn_out = jnp.einsum("bsh,hd->bsd", attn_out, block.attn.w_o, out_sharding=_PBATCH)
        hidden = hidden + attn_out

        mlp_in = rms_norm(hidden, block.rms_mlp, cfg.layer_norm_eps)
        mlp_out = mlp(block, mlp_in)
        hidden = hidden + mlp_out

    # final rms norm
    hidden = rms_norm(hidden, core.final_norm, cfg.layer_norm_eps)
    return hidden


def _lm_head_from_sink_params(params: dict) -> jax.Array:
    return params["core"].output_proj


_PBATCH = jax.sharding.PartitionSpec(("data",), None)


@LmConfig.register_subclass("grugformer_attn_sink")
@dataclasses.dataclass(frozen=True)
class GrugformerAttnSinkLmConfig(LmConfig[GrugWrapper]):
    max_seq_len: int = 2048

    hidden_dim: int = 1024
    intermediate_dim: int = 2752
    num_layers: int = 12
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int | None = None
    rope_theta: float = 10000.0

    sink: GrugformerAttnSinkConfig = dataclasses.field(default_factory=GrugformerAttnSinkConfig)

    @property
    def model_type(self) -> type[GrugWrapper]:
        return GrugWrapper

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    def to_grug_model_config(self) -> GrugModelConfig:
        return GrugModelConfig(
            vocab_size=llama3_tokenizer_vocab_size,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
        )

    def build(self, Vocab: Axis, *, key: PRNGKeyArray) -> GrugWrapper:
        core_cfg = self.to_grug_model_config()
        cfg = GrugformerAttnSinkModelConfig(core=core_cfg, sink=self.sink)
        params = _init_grugformer_with_sinks(cfg, key=key)
        return GrugWrapper(
            params=params,
            grug_config=cfg,
            init_fn=_init_grugformer_with_sinks,
            forward_fn=_grug_activations_with_sinks,
            lm_head_fn=_lm_head_from_sink_params,
        )

    def flops_per_token(self, vocab_size: int, context_length: int) -> float | None:
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=context_length,
            vocab_size=vocab_size,
            glu=True,
        )

    def total_trainable_params(self, vocab_size: int) -> int:
        head_dim = self.head_dim or (self.hidden_dim // self.num_heads)
        token_embedding = vocab_size * self.hidden_dim
        attn = (
            self.hidden_dim * head_dim * self.num_heads
            + 2 * self.hidden_dim * head_dim * self.num_kv_heads
            + head_dim * self.num_heads * self.hidden_dim
        )
        mlp = 3 * self.hidden_dim * self.intermediate_dim
        transformer = self.num_layers * (attn + mlp + 2 * self.hidden_dim) + self.hidden_dim
        sinks = self.num_layers * self.num_heads
        return int(transformer + 2 * token_embedding + sinks)


speedrun_config = SpeedrunConfig(
    author=Author(
        name="David Hall",
        affiliation="OpenAthena",
        url="TODO",
    ),
    description="Grugformer with attention sinks via TPU splash attention.",
    model_config=GrugformerAttnSinkLmConfig(
        max_seq_len=2048,
        hidden_dim=1024,
        intermediate_dim=2752,
        num_layers=12,
        num_heads=16,
        num_kv_heads=16,
        sink=GrugformerAttnSinkConfig(num_sinks=1, init_logit=0.0),
    ),
    train_config=SimpleTrainConfig(
        ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=32,
        num_train_steps=100,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
        steps_per_hf_export=-1,
        explicit_mesh_axes=True,
    ),
)


def build_run(size: str, *, use_tpu: bool = False) -> tuple[str, SpeedrunConfig]:
    sizes = _size_presets()
    if size not in sizes:
        raise ValueError(f"Unknown size: {size}")
    model_cfg = sizes[size]

    if not use_tpu:
        raise ValueError("grugformer_attn_sink requires SR_USE_TPU=1 (it uses TPU splash attention sinks).")

    batch = _batch_sizes()[size]
    max_seq_len = model_cfg.max_seq_len
    params = int(model_cfg.total_trainable_params(llama3_tokenizer_vocab_size))
    steps = _get_num_train_steps(params, batch, max_seq_len, tpp=20)
    resources = _resource_presets(use_tpu=use_tpu)[size]

    train = SimpleTrainConfig(
        resources,
        train_seq_len=max_seq_len,
        train_batch_size=batch,
        num_train_steps=steps,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
        steps_per_hf_export=-1,
        explicit_mesh_axes=True,
    )

    run_name = f"grugformer_attn_sink_{size}"
    desc = f"Grugformer with attention sinks via TPU splash attention ({size})."
    cfg = SpeedrunConfig(
        author=speedrun_config.author,
        description=desc,
        model_config=model_cfg,
        train_config=train,
    )
    return run_name, cfg


def main() -> None:
    sizes = [
        "130m",
        "300m",
        "520m",
        "1_2b",
    ]
    # sizes = ["130m", "300m", "520m", "1_2b"]
    use_tpu = bool(int(os.environ.get("SR_USE_TPU", "0")))

    steps = []
    for s in sizes:
        name, cfg = build_run(s, use_tpu=use_tpu)
        if cfg.vocab_size != llama3_tokenizer_vocab_size:
            raise AssertionError("Speedrun vocab_size mismatch; expected llama3_tokenizer_vocab_size")
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg))
    executor_main(steps=steps, description="Grugformer with attention sinks via TPU splash attention.")


if __name__ == "__main__":
    main()
