# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the pipeline primitive's gradients against a non-pipelined reference.

Run on a forced 8-device CPU mesh:

    XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        uv run python -m experiments.grug.moe_zb.check_pipeline_parity
"""

from __future__ import annotations

import os

if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_zb.pipeline import (
    PipelineModel,
    PipelineParams,
    pipeline_value_and_grad,
    zero_bubble_value_and_grad,
)

H = 16
NUM_STAGES = 8
LAYERS_PER_STAGE = 2
NUM_MICROBATCHES = 6
MICROBATCH = 4


def embed_fn(embed_params, tokens):
    # `tokens` are float feature vectors [mb, H] for this dense test.
    return jnp.tanh(tokens @ embed_params)


def stage_fn(stage_params, h):
    # stage_params: [layers_per_stage, H, H]. Apply each layer in order and return
    # a stage-local aux scalar (stands in for the MoE router z-loss).
    def one_layer(h, w):
        return jnp.tanh(h @ w), None

    h, _ = jax.lax.scan(one_layer, h, stage_params)
    aux = 0.05 * jnp.mean(h**2)
    return h, aux


def head_loss_fn(head_params, h, target):
    logits = h @ head_params
    return 0.5 * jnp.mean((logits - target) ** 2)


MODEL = PipelineModel(embed_fn=embed_fn, stage_fn=stage_fn, head_loss_fn=head_loss_fn)


def make_params(key):
    k_embed, k_stage, k_head = jax.random.split(key, 3)
    scale = 1.0 / np.sqrt(H)
    embed = jax.random.normal(k_embed, (H, H)) * scale
    stage = jax.random.normal(k_stage, (NUM_STAGES, LAYERS_PER_STAGE, H, H)) * scale
    head = jax.random.normal(k_head, (H, H)) * scale
    return PipelineParams(embed=embed, stage=stage, head=head)


def reference_loss(params, tokens, targets):
    """Non-pipelined ground truth: every microbatch through every stage/layer."""

    def loss_one(m):
        h = embed_fn(params.embed, tokens[m])
        aux_sum = jnp.zeros(())
        for s in range(NUM_STAGES):
            h, aux = stage_fn(params.stage[s], h)
            aux_sum = aux_sum + aux
        return head_loss_fn(params.head, h, targets[m]) + aux_sum

    losses = jnp.stack([loss_one(m) for m in range(NUM_MICROBATCHES)])
    return jnp.mean(losses)


def main():
    devices = np.array(jax.devices())
    assert devices.size == NUM_STAGES, f"need {NUM_STAGES} devices, got {devices.size}"
    mesh = Mesh(devices.reshape(NUM_STAGES), (("stage",)), axis_types=(AxisType.Auto,))

    key = jax.random.PRNGKey(0)
    k_p, k_x, k_t = jax.random.split(key, 3)
    params = make_params(k_p)
    tokens = jax.random.normal(k_x, (NUM_MICROBATCHES, MICROBATCH, H))
    targets = jax.random.normal(k_t, (NUM_MICROBATCHES, MICROBATCH, H))

    ref_loss, ref_grads = jax.value_and_grad(reference_loss)(params, tokens, targets)

    with jax.set_mesh(mesh):
        stage_sharded = jax.device_put(params.stage, NamedSharding(mesh, P("stage")))
        params_sharded = PipelineParams(
            embed=jax.device_put(params.embed, NamedSharding(mesh, P())),
            stage=stage_sharded,
            head=jax.device_put(params.head, NamedSharding(mesh, P())),
        )
        tok_r = jax.device_put(tokens, NamedSharding(mesh, P()))
        tgt_r = jax.device_put(targets, NamedSharding(mesh, P()))
        gpipe_loss, gpipe_grads = pipeline_value_and_grad(
            params_sharded,
            tok_r,
            tgt_r,
            model=MODEL,
            mesh=mesh,
            num_microbatches=NUM_MICROBATCHES,
            hidden_shape=(MICROBATCH, H),
        )
        zb_loss, zb_grads = zero_bubble_value_and_grad(
            params_sharded,
            tok_r,
            tgt_r,
            model=MODEL,
            mesh=mesh,
            num_microbatches=NUM_MICROBATCHES,
            hidden_shape=(MICROBATCH, H),
        )

    ref_loss = np.asarray(ref_loss)

    def report(label, loss, grads):
        loss = np.asarray(loss)
        print(f"\n[{label}] loss={loss:.8f} diff_vs_ref={abs(ref_loss - loss):.2e}")
        rels = []
        for name, a, b in (
            ("embed", ref_grads.embed, grads.embed),
            ("stage", ref_grads.stage, grads.stage),
            ("head", ref_grads.head, grads.head),
        ):
            a, b = np.asarray(a), np.asarray(b)
            max_abs = np.max(np.abs(a - b))
            rel = max_abs / (np.max(np.abs(a)) + 1e-12)
            print(f"  grad {name:6s} max_abs_diff={max_abs:.2e}  rel={rel:.2e}")
            rels.append(rel)
        return abs(ref_loss - loss) < 1e-5 and max(rels) < 1e-5

    ok_gpipe = report("GPipe autodiff oracle", gpipe_loss, gpipe_grads)
    ok_zb = report("Zero-bubble B/W split", zb_loss, zb_grads)
    print("\nRESULT:", "PASS" if (ok_gpipe and ok_zb) else "FAIL")
    return 0 if (ok_gpipe and ok_zb) else 1


if __name__ == "__main__":
    raise SystemExit(main())
