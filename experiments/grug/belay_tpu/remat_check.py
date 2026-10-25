# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare attention rematerialization with saved residuals on a reduced TPU fixture."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask

from experiments.grug.belay_tpu import train
from experiments.grug.belay_tpu.model import Transformer


def check_attention_remat(config, mesh):
    # Keep all six KDA/MLA/AttnRes/MoE blocks and recipe switches. Reduce widths,
    # expert count, vocabulary and sequence length so both reference gradients fit.
    cfg = dataclasses.replace(
        config.model,
        hidden_dim=64,
        intermediate_dim=32,
        shared_expert_intermediate_dim=32,
        num_experts=16,
        latent_dim=32,
        vocab_size=128,
        max_seq_len=128,
        head_dim=128,
        mla_kv_latent_dim=64,
        rel_extent=128,
        attn_res_remat_attention=False,
    )
    rng = np.random.RandomState(43)
    tokens = rng.randint(0, 128, (128, 128)).astype(np.int32)
    weights = np.ones(tokens.shape, np.float32)
    weights[:, -1] = 0
    segments = np.broadcast_to(np.arange(128) // 37, tokens.shape).astype(np.int32)

    def place(value):
        return jax.make_array_from_callback(
            value.shape, NamedSharding(mesh, P(("data", "expert"), None)), lambda index: value[index]
        )

    ids = place(segments)
    batch = GrugLmExample(place(tokens), place(weights), AttentionMask(is_causal=True, segment_ids=(ids, ids)))
    initialize = eqx.filter_jit(Transformer.init)
    reference = initialize(cfg, key=jax.random.PRNGKey(47))
    candidate = initialize(dataclasses.replace(cfg, attn_res_remat_attention=True), key=jax.random.PRNGKey(47))

    @jax.jit
    def compare(left, right):
        relative = [jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-12) for a, b in zip(left, right, strict=True)]
        return jnp.max(jnp.stack(relative))

    initial_error = float(compare(jax.tree.leaves(candidate), jax.tree.leaves(reference)))
    if initial_error != 0:
        raise AssertionError(f"Rematerialization changed initialization: {initial_error}")
    run = eqx.filter_jit(lambda model, data: train._loss_and_grads(model, data, jmp.get_policy("f32"), 1e-4))
    with jax.default_matmul_precision("highest"):
        (reference_loss, _), reference_grad = run(reference, batch)
        (candidate_loss, _), candidate_grad = run(candidate, batch)
        error = float(compare(jax.tree.leaves(candidate_grad), jax.tree.leaves(reference_grad)))
    np.testing.assert_allclose(float(candidate_loss), float(reference_loss), rtol=1e-5, atol=1e-6)
    # Source AttnRes test bounds each gradient leaf by 1e-4 of its maximum magnitude.
    if not np.isfinite(error) or error >= 1e-4:
        raise AssertionError(f"Rematerialized gradient mismatch: {error}")
    return {
        "rank": jax.process_index(),
        "fixture": "L6 d64/head128, 16 experts/top10, B128 S128, FP32; no optimizer update",
        "reference_loss": float(reference_loss),
        "candidate_loss": float(candidate_loss),
        "max_relative_gradient_error": error,
        "gradient_tolerance": 1e-4,
    }
