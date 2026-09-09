# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Document scoring directly from native Snowball training checkpoints."""

import dataclasses
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import load_checkpoint
from levanter.data.text.examples import GrugLmExample
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.main.perplexity_gap import ModelLossRunner
from levanter.tokenizers import load_tokenizer
from marin.testing.inference.snowball_checkpoint import apply_pending_qb_betas, decode_vendored_config
from rigging.filesystem.storage_path import StoragePath

from experiments.june_tpu_67b_a2b.moe.model import Transformer


def native_token_losses(model: Transformer, batch: GrugLmExample) -> jax.Array:
    """Return predictive NLL only; reduction='none' excludes router regularizers."""
    hidden, _ = model(batch.tokens, mask=batch.attn_mask)
    tail = jax.sharding.reshard(batch.tokens[:, :1], jax.typeof(batch.tokens).sharding)
    labels = jnp.concatenate([batch.tokens[:, 1:], tail], axis=1)
    return fused_linear_softmax_cross_entropy_loss(
        hidden,
        model.output_proj,
        labels,
        weight=batch.loss_weight,
        reduction="none",
        logsumexp_weight=None,
        dtype=jnp.float32,
        implementation=model.config.ce_implementation,
    )


@eqx.filter_jit(donate="all")
def prepare_native_model(model: Transformer, betas: jax.Array, mp: jmp.Policy) -> Transformer:
    """Apply pending router biases before the export-equivalent precision cast."""
    # TensorStore restores source dtypes. Donate the FP32 weights while casting
    # so the router update does not allocate a second full checkpoint in HBM.
    return mp.cast_to_param(apply_pending_qb_betas(model, betas))


def load_native_runner(
    *,
    checkpoint_path: str,
    executor_info_path: str,
    tokenizer_path: str,
    eval_batch_size: int,
    max_eval_length: int,
    mp: jmp.Policy,
    mesh: jax.sharding.Mesh,
) -> ModelLossRunner[Transformer]:
    """Load model weights and pending router biases, without reading optimizer state.

    Call inside the supplied explicit mesh and keep that mesh active for scoring.
    The executor metadata supplies the architecture, including long-context QK
    gains and checkpoint tree layout. Only execution backends and window length
    are overridden for evaluation.
    """
    executor_info = json.loads(StoragePath(executor_info_path).read_text())
    # This newer training metric changes FLOPs reporting, never forward math.
    executor_info["config"]["model"].pop("hybrid_attention_flops_accounting", None)
    source_config = decode_vendored_config(executor_info)
    config = dataclasses.replace(
        source_config,
        max_seq_len=max_eval_length,
        attention_implementation=None,
        moe_implementation="ring",
        ce_implementation=None,
    )
    template = eqx.filter_eval_shape(lambda: mp.cast_to_param(Transformer.init(config, key=jax.random.PRNGKey(0))))
    state = load_checkpoint(
        {
            "params": template,
            "pending_qb_betas": jax.ShapeDtypeStruct((config.num_layers, config.num_experts), jnp.float32),
        },
        checkpoint_path,
        mesh=mesh,
    )
    model = prepare_native_model(state["params"], state["pending_qb_betas"], mp)
    jax.block_until_ready(model)
    tokenizer = load_tokenizer(tokenizer_path)
    if len(tokenizer) != config.vocab_size:
        raise ValueError(f"Tokenizer vocabulary {len(tokenizer)} does not match model vocabulary {config.vocab_size}")

    @eqx.filter_jit
    def compute_losses(model: Transformer, batch: GrugLmExample) -> jax.Array:
        return native_token_losses(mp.cast_to_compute(model), batch)

    return ModelLossRunner(
        label=checkpoint_path,
        model=model,
        tokenizer=tokenizer,
        hf_tokenizer=tokenizer.as_hf_tokenizer(),
        eval_batch_size=eval_batch_size,
        eval_length=max_eval_length,
        compute_losses=compute_losses,
        batch_sharding=NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None)),
    )
