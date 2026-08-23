"""Count compiled ragged-all-to-all ops in the ragged MoE grad under the current XLA_FLAGS.

The hero-shape traces show the device-kernel configuration compiles ~2x the ragged-all-to-all
ops per step of the one-shot configuration (backward recomputes the forward transport). This
probe reproduces the compile at 4-GPU scale and prints the op count so flag configurations can
be bisected in separate processes (XLA flags bind at backend init).
"""

import os
import re

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug.grug_moe import moe_mlp

TOKENS_PER_DEVICE = 128
HIDDEN = 64
INTERMEDIATE = 96
NUM_EXPERTS = 8
TOPK = 2


def main() -> None:
    devices = jax.devices()
    assert len(devices) >= 4, f"need 4 devices, got {len(devices)}"
    mesh = Mesh(
        np.array(devices[:4]).reshape(1, 4, 1),
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    tokens = 4 * TOKENS_PER_DEVICE

    k_x, k_sel, k_logits, k_w13, k_w2 = jax.random.split(jax.random.key(23), 5)
    x = jax.random.normal(k_x, (tokens, HIDDEN), dtype=jnp.bfloat16)
    selected = jax.random.randint(k_sel, (tokens, TOPK), 0, NUM_EXPERTS, dtype=jnp.int32)
    combine = jax.nn.softmax(jax.random.normal(k_logits, (tokens, TOPK), dtype=jnp.float32), axis=-1).astype(
        jnp.bfloat16
    )
    w13 = jax.random.normal(k_w13, (NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE), dtype=jnp.bfloat16)
    w2 = jax.random.normal(k_w2, (NUM_EXPERTS, INTERMEDIATE, HIDDEN), dtype=jnp.bfloat16)

    with jax.set_mesh(mesh):
        batch = NamedSharding(mesh, P(("data", "expert"), None))
        expert = NamedSharding(mesh, P("expert", None, None))
        x = jax.sharding.reshard(x, batch)
        selected = jax.sharding.reshard(selected, batch)
        combine = jax.sharding.reshard(combine, batch)
        w13 = jax.sharding.reshard(w13, expert)
        w2 = jax.sharding.reshard(w2, expert)

        def loss(x, combine, w13, w2):
            out, _ = moe_mlp(
                x,
                selected,
                combine,
                w13,
                w2,
                implementation="ragged_all_to_all",
                mesh=None,
                report_capacity_overflow=True,
                capacity_factor=2.0,
            )
            return jnp.sum(out.astype(jnp.float32) ** 2)

        grad = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))
        compiled = grad.lower(x, combine, w13, w2).compile()
        text = compiled.as_text()

    defs = re.findall(r"= \S+ ragged-all-to-all\(", text)
    starts = re.findall(r"= \S+ ragged-all-to-all-start\(", text)
    print(f"PROBE_FLAGS={os.environ.get('XLA_FLAGS', '')}")
    print(f"PROBE_RA2A_SYNC_DEFS={len(defs)}")
    print(f"PROBE_RA2A_START_DEFS={len(starts)}")
    print(f"PROBE_RA2A_TOTAL={len(defs) + len(starts)}")


if __name__ == "__main__":
    main()
