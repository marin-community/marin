# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The MoE and loss token-axis policy under a context-sharded sequence.

Every case runs in a subprocess: the behavior under test only appears on a multi-device mesh, and
the CPU device count has to be fixed before JAX brings up its backend.

Two kinds of assertion appear here, because the policy has two kinds of consequence. Where context
sharding changes a number -- drop totals, the QB threshold, gradients through a capacity-limited
layer -- the tests compare values. Where it only changes a layout, values cannot regress: a
too-narrow PartitionSpec makes JAX gather the missing axis, and the psum that follows then covers
every token anyway, so the answer stays right and only the collectives change. Those cases assert
on the sharding the arrays carry into `shard_map` and on the collectives XLA emits.

The shared shape is `[B=4, S=4]`, chosen so the flat token dim splits into four blocks of one batch
row on both meshes compared here:

- reference mesh `data=2, expert=2, context=1`, block index `data * 2 + expert`
- context mesh `data=1, expert=2, context=2`, block index `expert * 2 + context`

Both partition the tokens identically, so any quantity the policy says is global has to come out the
same. Only the expert-parallel grouping differs, which is what the block-wise comparisons account
for.
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


def count_hlo_collectives(lowered):
    text = lowered.compile().as_text()
    return {op: text.count("= " + op) + text.count(" " + op + "(") for op in ("all-gather", "all-reduce")}
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
def test_drop_totals_count_each_assignment_once_when_tokens_skip_the_context_axis():
    # `moe_mlp` sums its capacity counters over the axes the caller's token dim is split over, read
    # off that tensor rather than off the mesh. A caller that leaves the token dim on the batch axes
    # while the mesh carries a context axis -- any variant not ported to context sharding -- has the
    # same tokens on every context shard, so reducing over the mesh's full tuple would report each
    # dropped assignment once per context shard.
    _run(
        """
        from levanter.grug.grug_moe import MoEExpertMlp

        num_experts, hidden_dim, top_k = 4, 16, 2
        x = jax.random.normal(jax.random.key(3), (16, hidden_dim), dtype=jnp.float32)
        selected = jax.random.randint(jax.random.key(6), (16, top_k), 0, num_experts, dtype=jnp.int32)
        weights = jnp.ones((16, top_k), dtype=jnp.float32)

        def dropped(mesh):
            with set_mesh(mesh):
                experts = MoEExpertMlp.init(
                    num_experts=num_experts, hidden_dim=hidden_dim, intermediate_dim=8,
                    initializer_std=0.1, key=jax.random.key(0),
                    implementation="fixed_all_to_all", capacity_factor=0.5,
                )

                def call(experts, tokens, selected, weights):
                    # Batch axes only: the token dim never touches "context".
                    tokens = reshard(tokens, P(hero._BATCH_AXES, None))
                    return experts(tokens, selected, weights, mesh=mesh, report_capacity_overflow=True)[1]

                overflow = jax.jit(call)(experts, x, selected, weights)
            return int(overflow.sender) + int(overflow.receiver)

        context_drops = dropped(mesh_of((1, 1, 2, 2, 1)))
        reference_drops = dropped(mesh_of((1, 1, 1, 2, 1)))
        assert reference_drops > 0, "capacity_factor=0.5 must actually drop assignments"
        assert context_drops == reference_drops, (context_drops, reference_drops)
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
def test_fused_cross_entropy_reduces_over_the_sequence_instead_of_gathering_it():
    # Naming only the batch dim still returns the right loss -- the reshard gathers the sequence
    # first -- so the regression to catch is the layout: the activation must reach the kernel still
    # split on "context", with one all-reduce and no all-gather, and the psum must then name every
    # axis the token dims are split over. The value and gradient checks against a dense reference
    # guard the other direction, a psum naming an axis the layout did not split over.
    #
    # The batch-replicated layout is the case a batch-dim-only probe misses entirely: a long-context
    # run can put its whole batch on one shard and split only the sequence, and reading `spec[0]`
    # sees `None` there and falls back to the replicated default.
    _run(
        """
        k_hidden, k_head, k_labels, k_weight = jax.random.split(jax.random.key(5), 4)
        hidden = jax.random.normal(k_hidden, (4, 4, 16), dtype=jnp.float32)
        lm_head = jax.random.normal(k_head, (16, 64), dtype=jnp.float32) * 0.1
        labels = jax.random.randint(k_labels, (4, 4), 0, 64, dtype=jnp.int32)
        weight = jax.random.uniform(k_weight, (4, 4), dtype=jnp.float32, minval=0.1, maxval=2.0)

        def reference(hidden, lm_head, reduction):
            per_token = -jnp.take_along_axis(
                jax.nn.log_softmax(jnp.einsum("bsd,dv->bsv", hidden, lm_head), axis=-1),
                labels[..., None],
                axis=-1,
            )[..., 0]
            total = jnp.sum(per_token * weight)
            return total if reduction == "sum" else total / jnp.sum(weight)

        from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss

        mesh = mesh_of((1, 1, 2, 2, 1))
        layouts = (
            (activation_spec(True), TOKEN_AXES),
            (P(None, "context", None), ("context",)),  # batch replicated, sequence split
        )
        for hidden_spec, expected_psum_axes in layouts:
            for reduction in ("sum", "mean"):
                def sharded(hidden, lm_head, hidden_spec=hidden_spec, reduction=reduction):
                    return fused_linear_softmax_cross_entropy_loss(
                        reshard(hidden, hidden_spec), lm_head, labels, weight=weight, reduction=reduction,
                    )

                with set_mesh(mesh):
                    loss = jax.jit(sharded)(hidden, lm_head)
                    grads = jax.jit(jax.grad(sharded, argnums=(0, 1)))(hidden, lm_head)
                    jaxpr = jax.make_jaxpr(sharded)(hidden, lm_head).jaxpr
                    collectives = count_hlo_collectives(jax.jit(sharded).lower(hidden, lm_head))
                expected_loss = reference(hidden, lm_head, reduction)
                expected_grads = jax.grad(reference, argnums=(0, 1))(hidden, lm_head, reduction)

                np.testing.assert_allclose(np.asarray(loss), np.asarray(expected_loss), rtol=1e-5, atol=1e-5)
                for actual, expected in zip(grads, expected_grads, strict=True):
                    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)

                context = (hidden_spec, reduction)
                (shard_map_params,) = equation_params(jaxpr, "shard_map")
                assert shard_map_params["in_specs"][0] == hidden_spec, (context, shard_map_params["in_specs"])
                psum_axes = [tuple(params["axes"]) for params in equation_params(jaxpr, "psum")]
                assert psum_axes and set(psum_axes) == {expected_psum_axes}, (context, psum_axes)
                assert collectives == {"all-gather": 0, "all-reduce": 1}, (context, collectives)
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
