# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-device MoE drop accounting and loss reductions over context shards."""

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

BATCH_AXES = ("replica_dcn", "data", "expert")

_MESH_AXES = ("replica_dcn", "data", "context", "expert", "model")
TOKEN_AXES = (*BATCH_AXES, "context")


def mesh_of(shape):
    devices = np.asarray(jax.devices()[: math.prod(shape)]).reshape(shape)
    return Mesh(devices, _MESH_AXES, axis_types=(AxisType.Explicit,) * len(_MESH_AXES))


def activation_spec(seq_sharded):
    return P(BATCH_AXES, "context" if seq_sharded else None, None)


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
def test_drop_totals_count_each_assignment_once_when_tokens_skip_the_context_axis():
    # A context replica must not multiply drop counts for batch-sharded tokens.
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
                    tokens = reshard(tokens, P(BATCH_AXES, None))
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
def test_fused_cross_entropy_reduces_over_the_sequence_instead_of_gathering_it():
    # Loss values alone miss a sequence all-gather; check compiled collectives too.
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
