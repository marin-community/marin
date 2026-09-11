# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check hero routing, gradients, and parameter placement across context layouts.

Each case uses a subprocess to configure four CPU devices before JAX initializes.
The reference and context meshes own the same token blocks but group experts
differently, so routing comparisons account for each independent context group.
"""

import textwrap

import pytest
from levanter.testing.cpu_devices import run_on_cpu_devices

_PRELUDE = """
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import P
from jax.sharding import AxisType, Mesh, NamedSharding, reshard, set_mesh

from experiments.grug.moe_hero_ep import model as hero

_MESH_AXES = ("replica_dcn", "data", "context", "expert", "model")


def mesh_of(shape):
    devices = np.asarray(jax.devices()[: math.prod(shape)]).reshape(shape)
    return Mesh(devices, _MESH_AXES, axis_types=(AxisType.Explicit,) * len(_MESH_AXES))


def moe_config(**overrides):
    config = dict(
        vocab_size=64,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        global_every=2,
        capacity_factor=4.0,
        initializer_std=0.5 / math.sqrt(16),
        qk_mult=1.3,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
        report_capacity_overflow=True,
    )
    config.update(overrides)
    return hero.GrugModelConfig(**config)


def activation_spec(seq_sharded):
    return P(hero._BATCH_AXES, "context" if seq_sharded else None, None)

"""


def _run(body: str) -> None:
    run_on_cpu_devices(_PRELUDE + textwrap.dedent(body), device_count=4)


@pytest.mark.timeout(300)
def test_routed_moe_matches_independent_routing_per_context_shard():
    # Each context group routes independently; total drops must sum across groups.
    _run(
        """
        config = moe_config(capacity_factor=0.5)
        x = jax.random.normal(jax.random.key(3), (4, 4, config.hidden_dim), dtype=jnp.float32)

        def routed(mesh, tokens, seq_sharded):
            call = lambda l, t: l(reshard(t, activation_spec(seq_sharded)), jnp.ones(t.shape[:2], dtype=jnp.bool_))
            with set_mesh(mesh):
                layer = hero.MoEMLP.init(config, key=jax.random.key(0))
                out, stats = jax.jit(call)(layer, tokens)
            drops = {k: int(v) for k, v in stats.items() if k.endswith("capacity_overflow")}
            return np.asarray(out), drops

        context_mesh = mesh_of((1, 1, 2, 2, 1))
        reference_mesh = mesh_of((1, 1, 1, 2, 1))
        # Expert-parallel group `c` on the context mesh holds token blocks {c, c + 2}, i.e. rows
        # {c, c + 2}; the reference mesh runs each of those pairs on its own.
        out, drops = routed(context_mesh, x, seq_sharded=True)
        out_even, drops_even = routed(reference_mesh, x[0::2], seq_sharded=False)
        out_odd, drops_odd = routed(reference_mesh, x[1::2], seq_sharded=False)

        np.testing.assert_array_equal(out[0::2], out_even)
        np.testing.assert_array_equal(out[1::2], out_odd)
        assert drops["capacity_overflow"] > 0, "capacity_factor=0.5 must actually drop assignments"
        assert drops_even != drops_odd, "the halves must differ, or summing them proves nothing"
        assert drops == {k: drops_even[k] + drops_odd[k] for k in drops}, (drops, drops_even, drops_odd)

        """
    )


@pytest.mark.timeout(300)
def test_qb_threshold_matches_across_token_layouts():
    # Keep four token shards while moving partitioning from data to context.
    _run(
        """
        x = jax.random.normal(jax.random.key(3), (4, 4, 16), dtype=jnp.float32)

        def thresholds(mesh, config, seq_sharded):
            with set_mesh(mesh):
                layer = hero.MoEMLP.init(config, key=jax.random.key(0))
                call = jax.jit(
                    lambda l, t: l(reshard(t, activation_spec(seq_sharded)), jnp.ones(t.shape[:2], dtype=jnp.bool_))[1]
                )
                stats = call(layer, x)
            return {k: np.asarray(v) for k, v in stats.items() if k.startswith("qb_beta")}

        context_mesh = mesh_of((1, 1, 2, 2, 1))
        reference_mesh = mesh_of((1, 2, 1, 2, 1))
        for estimator in (hero.QbEstimator.TOPK, hero.QbEstimator.HIST):
            config = moe_config(qb_estimator=estimator, qb_hist_bins=64)
            reference = thresholds(reference_mesh, config, seq_sharded=False)
            context = thresholds(context_mesh, config, seq_sharded=True)
            assert reference.keys() == context.keys() and reference
            for name, expected in reference.items():
                np.testing.assert_allclose(context[name], expected, rtol=1e-6, atol=1e-6)
        """
    )


@pytest.mark.timeout(300)
def test_moe_preserves_sequence_sharding():
    # Both MLP branches must restore the residual layout after flattening tokens.
    _run(
        """
        config = moe_config()
        with set_mesh(mesh_of((1, 1, 2, 2, 1))):
            routed = hero.MoEMLP.init(config, key=jax.random.key(0))
            shared = hero.DenseMLP.init(
                config.hidden_dim, config.shared_expert_intermediate_dim,
                config.initializer_std, key=jax.random.key(1),
            )
            for seq_sharded in (True, False):
                spec = activation_spec(seq_sharded)
                x = jnp.zeros((4, 4, config.hidden_dim), dtype=jnp.float32)
                routed_out = jax.eval_shape(
                    lambda t: routed(reshard(t, spec), jnp.ones(t.shape[:2], dtype=jnp.bool_))[0], x
                )
                shared_out = jax.eval_shape(lambda t: shared(reshard(t, spec)), x)
                assert routed_out.sharding.spec == spec, (seq_sharded, routed_out.sharding.spec)
                assert shared_out.sharding.spec == spec, (seq_sharded, shared_out.sharding.spec)
        """
    )


@pytest.mark.timeout(300)
def test_shared_and_routed_gradients_match_across_context_degree():
    # Force capacity drops so the gradients exercise shard-local routing decisions.
    _run(
        """
        config = moe_config(capacity_factor=0.5)
        x = jax.random.normal(jax.random.key(3), (4, 4, config.hidden_dim), dtype=jnp.float32)
        cotangent = jax.random.normal(jax.random.key(4), (4, 4, config.hidden_dim), dtype=jnp.float32)

        def gradients(mesh, seq_sharded):
            with set_mesh(mesh):
                routed = hero.MoEMLP.init(config, key=jax.random.key(0))
                shared = hero.DenseMLP.init(
                    config.hidden_dim, config.shared_expert_intermediate_dim,
                    config.initializer_std, key=jax.random.key(1),
                )

                def objective(routed, shared, tokens):
                    tokens = reshard(tokens, activation_spec(seq_sharded))
                    out, _ = routed(tokens, jnp.ones(tokens.shape[:2], dtype=jnp.bool_))
                    return jnp.sum((out + shared(tokens)) * cotangent)

                grads = jax.jit(jax.grad(objective, argnums=(0, 1, 2)))(routed, shared, x)
            return [np.asarray(leaf) for leaf in jax.tree.leaves(grads)]

        reference = gradients(mesh_of((1, 2, 1, 2, 1)), seq_sharded=False)
        context = gradients(mesh_of((1, 1, 2, 2, 1)), seq_sharded=True)
        assert reference and len(reference) == len(context)
        for actual, expected in zip(context, reference, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        """
    )


@pytest.mark.timeout(300)
def test_parameter_context_sharding_preserves_placement_and_numerics():
    # Context storage sharding reduces resident parameters without changing layer numerics.
    _run(
        """
        config = moe_config()
        x = jax.random.normal(jax.random.key(3), (4, 4, config.hidden_dim), dtype=jnp.float32)
        cotangent = jax.random.normal(jax.random.key(4), (4, 4, config.hidden_dim), dtype=jnp.float32)

        def run(mesh, seq_sharded):
            with set_mesh(mesh):
                routed = hero.MoEMLP.init(config, key=jax.random.key(0))
                shared = hero.DenseMLP.init(
                    config.hidden_dim, config.shared_expert_intermediate_dim,
                    config.initializer_std, key=jax.random.key(1),
                )

                def forward(routed, shared, tokens):
                    tokens = reshard(tokens, activation_spec(seq_sharded))
                    out, _ = routed(tokens, jnp.ones(tokens.shape[:2], dtype=jnp.bool_))
                    return out + shared(tokens)

                out = jax.jit(forward)(routed, shared, x)
                grads = jax.jit(jax.grad(lambda r, s, t: jnp.sum(forward(r, s, t) * cotangent), argnums=(0, 1)))(
                    routed, shared, x
                )
            if mesh.shape["context"] == 1:
                # Check initialized parameters against their pre-CP device ownership.
                for param, legacy in (
                    (routed.expert_mlp.w_gate, P("expert", "data", "model")),
                    (shared.w_gate, P(("data", "expert"), "model")),
                    (shared.w_down, P("model", ("data", "expert"))),
                ):
                    assert param.sharding.devices_indices_map(param.shape) == NamedSharding(
                        mesh, legacy
                    ).devices_indices_map(param.shape), (param.shape, param.sharding, legacy)
            expert_weight = routed.expert_mlp.w_gate
            return {
                "experts_per_shard": expert_weight.addressable_shards[0].data.shape[0],
                "out": np.asarray(out),
                "grads": [np.asarray(leaf) for leaf in jax.tree.leaves(grads)],
            }

        reference = run(mesh_of((1, 2, 1, 2, 1)), seq_sharded=False)
        context = run(mesh_of((1, 1, 2, 2, 1)), seq_sharded=True)

        # Four experts over expert=2 alone, then over expert=2 x context=2.
        assert (reference["experts_per_shard"], context["experts_per_shard"]) == (2, 1), (
            reference["experts_per_shard"], context["experts_per_shard"]
        )
        np.testing.assert_allclose(context["out"], reference["out"], rtol=1e-5, atol=1e-5)
        for actual, expected in zip(context["grads"], reference["grads"], strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        """
    )
