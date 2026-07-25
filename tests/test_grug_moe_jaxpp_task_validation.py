# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
from jax.extend import core as jax_core

from experiments.grug.moe import train as grug_train


def test_jaxpp_task_call_jaxpr_validation_accepts_closed_task():
    call_jaxpr = jax.make_jaxpr(lambda value: value + 1)(jnp.array(0, dtype=jnp.int32))

    grug_train._check_jaxpp_task_call_jaxpr("fwd_0", call_jaxpr)


def test_jaxpp_task_call_jaxpr_validation_rejects_missing_operand():
    call_jaxpr = jax.make_jaxpr(lambda value: value + 1)(jnp.array(0, dtype=jnp.int32))
    malformed_jaxpr = call_jaxpr.jaxpr.replace(invars=())
    malformed_call_jaxpr = call_jaxpr.replace(jaxpr=malformed_jaxpr)

    with pytest.raises(ValueError, match="JaxPP generated invalid task JAXPR 'fwd_0'"):
        grug_train._check_jaxpp_task_call_jaxpr("fwd_0", malformed_call_jaxpr)


def test_bind_jaxpp_meshes_replaces_only_each_jaxpr_top_level():
    call_jaxpr = jax.make_jaxpr(lambda value: value + 1)(jnp.array(0, dtype=jnp.int32))
    task_primitive = jax_core.Primitive("test_jaxpp_task")
    loop_primitive = jax_core.Primitive("test_jaxpp_loop")
    transfer_primitive = jax_core.Primitive("test_jaxpp_transfer")
    loop_input = jax_core.Var(call_jaxpr.in_avals[0])
    loop_output = jax_core.Var(call_jaxpr.out_avals[0])
    task_equation = jax_core.new_jaxpr_eqn(
        [loop_input],
        [loop_output],
        task_primitive,
        {"call_jaxpr": call_jaxpr, "mpmd_idx": 0},
        call_jaxpr.effects,
    )
    loop_jaxpr = jax_core.Jaxpr((), (loop_input,), (loop_output,), (task_equation,), call_jaxpr.effects)
    loop_cjaxpr = jax_core.ClosedJaxpr(loop_jaxpr, ())
    outer_input = jax_core.Var(call_jaxpr.in_avals[0])
    outer_output = jax_core.Var(call_jaxpr.out_avals[0])
    loop_equation = jax_core.new_jaxpr_eqn(
        [outer_input],
        [outer_output],
        loop_primitive,
        {"jaxpr": loop_cjaxpr},
        call_jaxpr.effects,
    )
    outer_jaxpr = jax_core.Jaxpr((), (outer_input,), (outer_output,), (loop_equation,), call_jaxpr.effects)
    outer_cjaxpr = jax_core.ClosedJaxpr(outer_jaxpr, ())
    fake_core = SimpleNamespace(
        task_p=task_primitive,
        dax_pscan_p=loop_primitive,
        transfer_p=transfer_primitive,
        _resolve_placement=lambda *_args, **_kwargs: ((0,), object()),
        _bind_task_eqn_to_mesh=lambda equation, _mesh: equation,
    )

    bound = grug_train._bind_jaxpp_meshes_shallow(outer_cjaxpr, object(), fake_core)

    bound_loop = bound.jaxpr.eqns[0].params["jaxpr"]
    bound_task_call = bound_loop.jaxpr.eqns[0].params["call_jaxpr"]
    assert bound_task_call.jaxpr.eqns == call_jaxpr.jaxpr.eqns
