# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core as jax_core
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe import train as grug_train


@dataclass(frozen=True)
class _AutomaticJaxPPInfo:
    in_shardings: tuple[NamedSharding, ...]
    out_shardings: tuple[NamedSharding, ...]
    out_avals: tuple[jax.ShapeDtypeStruct, ...]


@dataclass(frozen=True)
class _AutomaticJaxPPCompiled:
    in_info: _AutomaticJaxPPInfo


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


def test_replace_jaxpp_captured_meshes_preserves_abstract_shard_map():
    call_jaxpr = jax.make_jaxpr(lambda value: value + 1)(jnp.array(0, dtype=jnp.int32))
    task_primitive = jax_core.Primitive("test_jaxpp_task")
    jit_primitive = jax_core.Primitive("test_jaxpp_jit")
    shard_map_primitive = jax_core.Primitive("test_jaxpp_shard_map")
    shard_map_primitive.def_abstract_eval(lambda value, **_params: value)
    input_variable = jax_core.Var(call_jaxpr.in_avals[0])
    output_variable = jax_core.Var(call_jaxpr.out_avals[0])
    shard_map_equation = jax_core.new_jaxpr_eqn(
        [input_variable],
        [output_variable],
        shard_map_primitive,
        {"mesh": jax.sharding.AbstractMesh((1,), ("data",))},
        call_jaxpr.effects,
    )
    outer_jaxpr = jax_core.Jaxpr(
        (),
        (input_variable,),
        (output_variable,),
        (shard_map_equation,),
        call_jaxpr.effects,
    )
    outer_cjaxpr = jax_core.ClosedJaxpr(outer_jaxpr, ())
    fake_core = SimpleNamespace(
        jcore=jax_core,
        jc=SimpleNamespace(jit_p=jit_primitive, shard_map_p=shard_map_primitive),
        task_p=task_primitive,
    )

    replaced = grug_train._replace_jaxpp_captured_meshes(outer_cjaxpr, object(), fake_core)

    assert len(replaced.jaxpr.eqns) == 1
    assert replaced.jaxpr.eqns[0].primitive is shard_map_primitive
    assert replaced.jaxpr.eqns[0].invars == shard_map_equation.invars
    assert replaced.jaxpr.eqns[0].outvars == shard_map_equation.outvars
    assert replaced.jaxpr.eqns[0].params["mesh"] == shard_map_equation.params["mesh"]
    jax_core.check_jaxpr(replaced.jaxpr)


def test_localize_automatic_jaxpp_shardings_expands_replicated_output_rank():
    mesh = Mesh(np.asarray(jax.devices()), ("device",))
    compiled = _AutomaticJaxPPCompiled(
        in_info=_AutomaticJaxPPInfo(
            in_shardings=(NamedSharding(mesh, P()),),
            out_shardings=(
                NamedSharding(mesh, P()),
                NamedSharding(mesh, P("device")),
            ),
            out_avals=(
                jax.ShapeDtypeStruct((2, 3), jnp.float32),
                jax.ShapeDtypeStruct((2, 3), jnp.float32),
            ),
        )
    )
    mpmd_mesh = SimpleNamespace(
        jax_mesh=SimpleNamespace(is_multi_process=True),
        lowering_mesh=lambda: mesh,
    )

    localized = grug_train._localize_automatic_jaxpp_shardings(compiled, mpmd_mesh)

    assert localized.in_info.out_shardings[0].spec == P(None, None)
    assert localized.in_info.out_shardings[1].spec == P("device", None)
