# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal on-hardware smoke for the sonic_cute (QuACK SM100) MoE backend on B200/GB200.

Builds a tiny d1280-shaped Grug MoE with ``moe_implementation="sonic_cute"`` and a
pure-JAX ``reference`` attention (so this exercises only the QuACK grouped GEMM, not the
SM90 fa4 kernels), then runs one fwd+bwd. Run on an allocated GB200 node with quack
installed. Prints ``SONIC_CUTE_SMOKE_OK`` on success.
"""

import traceback

import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.sharding import AxisType, Mesh

from experiments.grug.moe.model import GrugModelConfig, Transformer


def _check_shared_expert_quack() -> None:
    """Assert the QuACK single-group shared-expert path matches the einsum reference (fwd + grads).

    Runs replicated (no mesh sharding) so it isolates kernel numerics from FSDP resharding. QuACK's
    GemmGatedSm100 is bf16-compute, so compare against a bf16 einsum reference at a bf16-appropriate
    tolerance, not fp32.
    """
    from levanter.grug._moe.sonic_cute import shared_expert_sonic_cute  # noqa: PLC0415

    h, i, t = 512, 256, 384
    key = jax.random.PRNGKey(1)
    kx, kg, ku, kd = jax.random.split(key, 4)
    x = (jax.random.normal(kx, (t, h)) * 0.1).astype(jnp.bfloat16)
    w_gate = (jax.random.normal(kg, (h, i)) * 0.05).astype(jnp.bfloat16)
    w_up = (jax.random.normal(ku, (h, i)) * 0.05).astype(jnp.bfloat16)
    w_down = (jax.random.normal(kd, (i, h)) * 0.05).astype(jnp.bfloat16)

    def ref(x_, wg, wu, wd):
        gate = jnp.einsum("td,dm->tm", x_, wg)
        up = jnp.einsum("td,dm->tm", x_, wu)
        return jnp.einsum("tm,md->td", jax.nn.silu(gate) * up, wd)

    def quack(x_, wg, wu, wd):
        return shared_expert_sonic_cute(x_, wg, wu, wd)

    y_ref, vjp_ref = jax.vjp(ref, x, w_gate, w_up, w_down)
    y_q, vjp_q = jax.vjp(quack, x, w_gate, w_up, w_down)
    cot = jax.random.normal(jax.random.PRNGKey(2), y_ref.shape).astype(jnp.bfloat16)
    g_ref = vjp_ref(cot)
    g_q = vjp_q(cot)

    def _rel(a, b):
        a, b = a.astype(jnp.float32), b.astype(jnp.float32)
        return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(a)) + 1e-6))

    rtol = 3e-2  # bf16 matmul accumulation across K
    fwd_err = _rel(y_ref, y_q)
    grad_errs = [_rel(a, b) for a, b in zip(g_ref, g_q, strict=True)]
    print(f"shared-quack fwd_rel={fwd_err:.4f} grad_rel={[f'{e:.4f}' for e in grad_errs]}", flush=True)
    worst = max([fwd_err, *grad_errs])
    if worst > rtol:
        raise AssertionError(f"shared_expert_sonic_cute mismatch vs einsum ref: worst rel {worst:.4f} > {rtol}")
    print("SHARED_QUACK_MATCH_OK", flush=True)


def main() -> None:
    devs = jax.devices()
    print(f"jax {jax.__version__} devices={devs}", flush=True)
    n = len(devs)
    # sonic_cute is a local/no-EP backend: expert=1, shard the batch over data.
    mesh = Mesh(
        np.array(devs).reshape(1, n, 1, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    cfg = GrugModelConfig(
        vocab_size=512,
        hidden_dim=1280,
        num_layers=2,
        num_heads=10,
        num_kv_heads=2,
        head_dim=128,
        intermediate_dim=640,
        shared_expert_intermediate_dim=1280,
        num_experts=256,
        num_experts_per_token=4,
        max_seq_len=256,
        moe_implementation="sonic_cute",
        attention_implementation="reference",
    )
    with jax.set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        tokens = jnp.ones((4, 128), dtype=jnp.int32)
        loss_weight = jnp.ones((4, 128), dtype=jnp.float32)

        # QuACK's GemmGatedSm100 requires 16-bit compute (bf16 postact); mirror training's
        # mixed-precision policy (params=float32, compute=bfloat16) by casting inside loss_fn.
        policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")

        def loss_fn(m):
            return policy.cast_to_compute(m).next_token_loss(tokens, loss_weight, reduction="mean")[0]

        loss, grads = jax.value_and_grad(loss_fn)(model)
        jax.block_until_ready(grads)
        print(f"SONIC_CUTE_SMOKE_OK loss={float(loss):.4f}", flush=True)

    # Kernel-numerics check for the QuACK shared-expert path (replicated, outside the mesh).
    _check_shared_expert_quack()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("SONIC_CUTE_SMOKE_FAIL", flush=True)
        traceback.print_exc()
        raise
