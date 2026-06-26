"""Reproduce the GFP8-035 XSA single-device sharding failure on a concrete 1-device CPU mesh.

The 1-H100 run (count=1) used a single-device explicit mesh; the same mesh on CPU reproduces the
ShardingTypeError during the forward. Used to verify the model.py XSA fix before relaunching on GPU.
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
    num_experts=2,
    num_experts_per_token=1,
    num_layers=1,
    num_heads=2,
    num_kv_heads=1,
    head_dim=128,
    max_seq_len=64,
    sliding_window=64,
    moe_implementation="scatter",
)

mesh = jax.make_mesh(
    (1, 1, 1, 1),
    ("replica_dcn", "data", "expert", "model"),
    axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
)
print("mesh:", mesh.shape)

with jax.set_mesh(mesh):
    model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
    tokens = jnp.zeros((4, 64), dtype=jnp.int32)

    def fwd(m, t):
        hidden, _ = m(t)
        return hidden

    try:
        out = jax.eval_shape(fwd, model, tokens)
        print("LOWERED OK ->", out.shape, out.dtype)
    except Exception as ex:
        print("FAILED:", type(ex).__name__)
        print(str(ex)[:500])
