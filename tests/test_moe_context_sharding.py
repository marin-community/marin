# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check hero routing, gradients, and parameter placement across context layouts.

Each case uses a subprocess to configure four CPU devices before JAX initializes.
The reference and context meshes own the same token blocks but group experts
differently, so routing comparisons account for each independent context group.
"""

import os
import subprocess
import sys
import textwrap

import pytest

_PRELUDE = """
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import P
from jax.extend import core as jax_core
from jax.sharding import AxisType, Mesh, reshard, set_mesh

from experiments.grug.moe_hero_ep import model as hero

_MESH_AXES = ("replica_dcn", "data", "context", "expert", "model")
TOKEN_AXES = (*hero._BATCH_AXES, "context")


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


def equation_params(jaxpr, primitive):
    \"\"\"Params of every equation named `primitive`, recursing into nested jaxprs.\"\"\"
    found = []
    for equation in jaxpr.eqns:
        if equation.primitive.name == primitive:
            found.append(equation.params)
        for value in equation.params.values():
            inner = getattr(value, "jaxpr", value)
            if isinstance(inner, jax_core.Jaxpr):
                found.extend(equation_params(inner, primitive))
    return found


"""


def _run(body: str) -> None:
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_NUM_CPU_DEVICES"] = "4"
    result = subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(body)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.timeout(300)
def test_routed_moe_matches_independent_routing_per_context_shard():
    # The policy is that a context shard routes its own tokens exactly as a data shard would, so a
    # context-parallel run must reproduce, block for block, two smaller runs over the same token
    # blocks -- including the global drop total, which only comes out right if the token dim enters
    # the expert-parallel shard_map split over "context" and the capacity psum reduces over it.
    _run(
        """
        config = moe_config(capacity_factor=0.5)
        x = jax.random.normal(jax.random.key(3), (4, 4, config.hidden_dim), dtype=jnp.float32)

        def routed(mesh, tokens, seq_sharded):
            call = lambda l, t: l(reshard(t, activation_spec(seq_sharded)))
            with set_mesh(mesh):
                layer = hero.MoEMLP.init(config, key=jax.random.key(0))
                out, stats = jax.jit(call)(layer, tokens)
                jaxpr = jax.make_jaxpr(call)(layer, tokens).jaxpr
            drops = {k: int(v) for k, v in stats.items() if k.endswith("capacity_overflow")}
            return np.asarray(out), drops, jaxpr

        context_mesh = mesh_of((1, 1, 2, 2, 1))
        reference_mesh = mesh_of((1, 1, 1, 2, 1))
        # Expert-parallel group `c` on the context mesh holds token blocks {c, c + 2}, i.e. rows
        # {c, c + 2}; the reference mesh runs each of those pairs on its own.
        out, drops, jaxpr = routed(context_mesh, x, seq_sharded=True)
        out_even, drops_even, _ = routed(reference_mesh, x[0::2], seq_sharded=False)
        out_odd, drops_odd, _ = routed(reference_mesh, x[1::2], seq_sharded=False)

        np.testing.assert_array_equal(out[0::2], out_even)
        np.testing.assert_array_equal(out[1::2], out_odd)
        assert drops["capacity_overflow"] > 0, "capacity_factor=0.5 must actually drop assignments"
        assert drops_even != drops_odd, "the halves must differ, or summing them proves nothing"
        assert drops == {k: drops_even[k] + drops_odd[k] for k in drops}, (drops, drops_even, drops_odd)

        # The token dim reaches the expert-parallel shard_map split over the whole tuple, and the
        # capacity counters are summed over exactly that tuple.
        specs = [params["in_specs"][0] for params in equation_params(jaxpr, "shard_map")]
        assert P(TOKEN_AXES) in specs, specs
        psum_axes = [tuple(params["axes"]) for params in equation_params(jaxpr, "psum")]
        assert TOKEN_AXES in psum_axes, psum_axes
        """
    )


@pytest.mark.timeout(300)
def test_qb_threshold_survives_moving_token_shards_from_data_to_context():
    # Both meshes below carry four token shards, so this pins the axis split rather than the shard
    # count: the top-k estimator's per-shard population is a function of how many token shards there
    # are (by design), and the histogram estimator's quantile is global. What must not matter is
    # whether those shards come from "data" or from "context".
    _run(
        """
        x = jax.random.normal(jax.random.key(3), (4, 4, 16), dtype=jnp.float32)

        def thresholds(mesh, config, seq_sharded):
            with set_mesh(mesh):
                layer = hero.MoEMLP.init(config, key=jax.random.key(0))
                call = jax.jit(lambda l, t: l(reshard(t, activation_spec(seq_sharded)))[1])
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
def test_moe_layer_returns_the_sequence_sharding_it_was_given():
    # The routed and shared branches both round-trip through a flat token axis, and unflattening
    # leaves the whole fused tuple on the batch dim. Snapping back to the caller's own layout is
    # what keeps the residual add and the layer-scan carry on a single sharding; pinning the
    # batch-only spec instead would drop the sequence sharding on the floor.
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
                routed_out = jax.eval_shape(lambda t: routed(reshard(t, spec))[0], x)
                shared_out = jax.eval_shape(lambda t: shared(reshard(t, spec)), x)
                assert routed_out.sharding.spec == spec, (seq_sharded, routed_out.sharding.spec)
                assert shared_out.sharding.spec == spec, (seq_sharded, shared_out.sharding.spec)
        """
    )


@pytest.mark.timeout(300)
def test_shared_and_routed_gradients_match_across_context_degree():
    # Gradients are where a wrong token partition hides: the keep mask that capacity computes on
    # each shard rides into the backward, so run this over capacity to keep the routed gradient
    # partition-sensitive rather than trivially invariant.
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
                    out, _ = routed(tokens)
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
def test_a_context_axis_of_size_one_places_parameters_where_it_did_before():
    # The parameter specs name "context" unconditionally. That is only safe because a length-1 axis
    # partitions nothing: this pins the placement against the specs that predate the axis, which is
    # what "no behavior change at context_axis_size == 1" has to mean for weights, and for the
    # master and optimizer state that inherit their placement.
    _run(
        """
        from jax.sharding import NamedSharding

        mesh = mesh_of((1, 2, 1, 2, 1))
        assert int(mesh.shape["context"]) == 1, mesh.shape
        cases = (
            ((4, 16, 8), P(hero._EXPERT_WEIGHT_AXES, None, None), P("expert", None, None)),
            ((16, 8), P(hero._FSDP_AXES, "model"), P(("data", "expert"), "model")),
            ((8, 16), P("model", hero._FSDP_AXES), P("model", ("data", "expert"))),
        )
        for shape, composite, legacy in cases:
            assert NamedSharding(mesh, composite).devices_indices_map(shape) == NamedSharding(
                mesh, legacy
            ).devices_indices_map(shape), (shape, composite, legacy)
        """
    )


@pytest.mark.timeout(300)
def test_parameters_shard_over_context_without_changing_the_layer():
    # EP x CP would otherwise hold a full expert bank per context shard, and the fp32 master plus
    # Muon momentum that inherit its placement do not fit in node RAM at the hero shape. Sharding
    # the bank over the composite has to be invisible to the layer: `moe_mlp` all-gathers the
    # context group before its shard_map, so the same weights meet the same tokens.
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
                    out, _ = routed(tokens)
                    return out + shared(tokens)

                out = jax.jit(forward)(routed, shared, x)
                grads = jax.jit(jax.grad(lambda r, s, t: jnp.sum(forward(r, s, t) * cotangent), argnums=(0, 1)))(
                    routed, shared, x
                )
            expert_weight = routed.expert_mlp.w_gate
            return {
                "expert_spec": expert_weight.sharding.spec,
                "dense_spec": shared.w_gate.sharding.spec,
                "experts_per_shard": expert_weight.addressable_shards[0].data.shape[0],
                "out": np.asarray(out),
                "grads": [np.asarray(leaf) for leaf in jax.tree.leaves(grads)],
            }

        reference = run(mesh_of((1, 2, 1, 2, 1)), seq_sharded=False)
        context = run(mesh_of((1, 1, 2, 2, 1)), seq_sharded=True)

        assert reference["expert_spec"][0] == hero._EXPERT_WEIGHT_AXES, reference["expert_spec"]
        assert reference["dense_spec"][0] == hero._FSDP_AXES, reference["dense_spec"]
        # Four experts over expert=2 alone, then over expert=2 x context=2.
        assert (reference["experts_per_shard"], context["experts_per_shard"]) == (2, 1), (
            reference["experts_per_shard"], context["experts_per_shard"]
        )
        np.testing.assert_allclose(context["out"], reference["out"], rtol=1e-5, atol=1e-5)
        for actual, expected in zip(context["grads"], reference["grads"], strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        """
    )
