"""Test Grug model with real multi-device sharding.

This tests that the model works correctly with actual sharding across devices.
The XLA_FLAGS must be set before JAX is imported, so we do it at module load time.
"""

import os
import sys

# Must set before importing JAX - this creates fake CPU devices for testing
_DEVICE_COUNT = 4
_current_flags = os.environ.get("XLA_FLAGS", "")
_device_flag = f"--xla_force_host_platform_device_count={_DEVICE_COUNT}"
if _device_flag not in _current_flags:
    os.environ["XLA_FLAGS"] = f"{_current_flags} {_device_flag}".strip()

# Prevent grugfuzz from being imported before we set up JAX
# (it sets up a 1x1 mesh at import time)
if "grugfuzz" in sys.modules:
    raise RuntimeError("grugfuzz was imported before test_sharding - JAX mesh already configured")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src import mesh as mesh_lib


def setup_mesh(data_parallel: int = 2, model_parallel: int = 2):
    """Create a mesh with the specified parallelism."""
    devices = jax.devices()
    n_devices = len(devices)
    expected = data_parallel * model_parallel
    if n_devices < expected:
        pytest.skip(f"Need {expected} devices, only have {n_devices}. "
                   f"Run with XLA_FLAGS=--xla_force_host_platform_device_count={expected}")

    device_array = np.array(devices[:expected]).reshape(data_parallel, model_parallel)
    mesh = Mesh(
        device_array,
        axis_names=("data", "model"),
        axis_types=(mesh_lib.AxisType.Explicit, mesh_lib.AxisType.Explicit),
    )
    return mesh


class TestShardedForward:
    """Test forward pass with actual multi-device sharding."""

    def test_dense_mlp_sharded(self):
        """Test dense MLP with sharded weights."""
        from levanter.grug.model import (
            GrugModelConfig, GrugBlockParams, GrugAttentionParams, mlp
        )
        from levanter.grug.sharding import Pbatch

        mesh = setup_mesh(2, 2)
        jax.set_mesh(mesh)

        with mesh:
            cfg = GrugModelConfig(
                vocab_size=1024,
                hidden_dim=256,
                intermediate_dim=512,
                num_heads=8,
                num_kv_heads=8,
                num_layers=1,
                use_clipped_gated_activation=True,
            )

            D, I = cfg.hidden_dim, cfg.intermediate_dim

            # Create sharded weights
            key = jax.random.PRNGKey(42)
            k1, k2, k3 = jax.random.split(key, 3)

            gate = jax.device_put(
                jax.random.normal(k1, (D, I)) * 0.02,
                NamedSharding(mesh, P("data", "model"))
            )
            up = jax.device_put(
                jax.random.normal(k2, (D, I)) * 0.02,
                NamedSharding(mesh, P("data", "model"))
            )
            down = jax.device_put(
                jax.random.normal(k3, (I, D)) * 0.02,
                NamedSharding(mesh, P("model", "data"))
            )

            block = GrugBlockParams(
                attn=GrugAttentionParams(
                    w_q=jnp.zeros((D, D)),
                    w_k=jnp.zeros((D, D)),
                    w_v=jnp.zeros((D, D)),
                    w_o=jnp.zeros((D, D)),
                ),
                rms_attn=jnp.ones(D),
                rms_mlp=jnp.ones(D),
                mlp_gate=gate,
                mlp_up=up,
                mlp_down=down,
            )

            # Sharded input
            x = jax.device_put(
                jax.random.normal(jax.random.PRNGKey(123), (4, 16, D)) * 0.1,
                NamedSharding(mesh, Pbatch)
            )

            # Run MLP
            out = mlp(block, x, cfg)

            assert out.shape == (4, 16, D)
            assert jnp.isfinite(out).all()
            print(f"\nDense MLP sharded test passed!")
            print(f"  Input sharding: {x.sharding}")
            print(f"  Output sharding: {out.sharding}")

    def test_moe_sharded(self):
        """Test MoE with sharded weights."""
        from levanter.grug.model import GrugModelConfig, GrugMoEParams, moe_forward
        from levanter.grug.sharding import Pbatch

        mesh = setup_mesh(2, 2)
        jax.set_mesh(mesh)

        with mesh:
            cfg = GrugModelConfig(
                vocab_size=1024,
                hidden_dim=256,
                intermediate_dim=512,
                num_heads=8,
                num_kv_heads=8,
                num_layers=1,
                use_moe=True,
                num_experts=4,
                num_experts_per_tok=2,
                use_clipped_gated_activation=True,
            )

            D, I, E = cfg.hidden_dim, cfg.intermediate_dim, cfg.num_experts

            key = jax.random.PRNGKey(42)

            # Router: (D, E) sharded on data, replicated on E
            k_router, key = jax.random.split(key)
            router = jax.device_put(
                jax.random.normal(k_router, (D, E)) * 0.02,
                NamedSharding(mesh, P("data", None))
            )

            # Experts (stacked)
            k_gate, k_up, k_down, key = jax.random.split(key, 4)
            gate = jax.device_put(
                jax.random.normal(k_gate, (E, D, I)) * 0.02,
                NamedSharding(mesh, P(None, "data", "model")),
            )
            up = jax.device_put(
                jax.random.normal(k_up, (E, D, I)) * 0.02,
                NamedSharding(mesh, P(None, "data", "model")),
            )
            down = jax.device_put(
                jax.random.normal(k_down, (E, I, D)) * 0.02,
                NamedSharding(mesh, P(None, "model", "data")),
            )

            moe = GrugMoEParams(router=router, gate=gate, up=up, down=down)

            # Sharded input
            x = jax.device_put(
                jax.random.normal(jax.random.PRNGKey(123), (4, 16, D)) * 0.1,
                NamedSharding(mesh, Pbatch)
            )

            # Run MoE
            out, extras = moe_forward(moe, x, cfg)

            assert out.shape == (4, 16, D)
            assert jnp.isfinite(out).all()
            assert "load_balancing_loss" in extras
            assert "router_z_loss" in extras

            print(f"\nMoE sharded test passed!")
            print(f"  Input sharding: {x.sharding}")
            print(f"  Output sharding: {out.sharding}")
            print(f"  Aux losses: {list(extras.keys())}")

    def test_full_model_sharded(self):
        """Test full model forward with sharding."""
        from levanter.grug.model import GrugModelConfig, init_parameters, forward

        mesh = setup_mesh(2, 2)

        with mesh:
            jax.set_mesh(mesh)

            cfg = GrugModelConfig(
                vocab_size=1024,
                hidden_dim=256,
                intermediate_dim=512,
                num_heads=8,
                num_kv_heads=8,
                num_layers=2,
                max_seq_len=64,
                use_clipped_gated_activation=True,
                use_attention_sinks=True,
                use_moe=True,
                num_experts=4,
                num_experts_per_tok=2,
            )

            key = jax.random.PRNGKey(42)
            params = init_parameters(cfg, key=key)

            # Check shardings
            print(f"\nParameter shardings:")
            print(f"  token_embed: {params.token_embed.sharding}")
            print(f"  block[0].moe.router: {params.blocks[0].moe.router.sharding}")
            print(f"  block[0].moe.experts[0].gate: {params.blocks[0].moe.experts[0].gate.sharding}")

            # Forward pass
            tokens = jax.random.randint(jax.random.PRNGKey(123), (4, 32), 0, cfg.vocab_size)
            logits = forward(params, tokens, cfg)

            assert logits.shape == (4, 32, cfg.vocab_size)
            assert jnp.isfinite(logits).all()

            print(f"\nFull model sharded test passed!")
            print(f"  Output shape: {logits.shape}")
            print(f"  Output sharding: {logits.sharding}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
