"""Confirm the XSA reshard fix lowers on a multi-device production-like mesh (model axis > 1).

8 fake CPU devices: replica_dcn=1, data=2, expert=2, model=2. EP ring MoE backend. If this lowers,
the attn_out reshard is a valid no-op-or-consistent op on the production path, not just single-device.
"""
import jax
import jax.numpy as jnp
from jax.sharding import AxisType

from experiments.grug.moe.model import GrugModelConfig, Transformer

cfg = GrugModelConfig(
    vocab_size=256,
    hidden_dim=256,
    intermediate_dim=128,
    shared_expert_intermediate_dim=128,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=1,
    num_heads=4,
    num_kv_heads=2,
    head_dim=64,
    max_seq_len=64,
    sliding_window=64,
    moe_implementation="ring",
)

print("devices:", jax.device_count())
mesh = jax.make_mesh(
    (1, 2, 2, 2),
    ("replica_dcn", "data", "expert", "model"),
    axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
)
print("mesh:", mesh.shape)

with jax.set_mesh(mesh):
    model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
    tokens = jnp.zeros((8, 64), dtype=jnp.int32)

    def fwd(m, t):
        hidden, _ = m(t)
        return hidden

    try:
        out = jax.eval_shape(fwd, model, tokens)
        print("PROD-LIKE LOWERED OK ->", out.shape, out.dtype)
    except Exception as ex:
        print("FAILED:", type(ex).__name__)
        print(str(ex)[:500])
