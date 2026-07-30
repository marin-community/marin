# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import AbstractMesh, AxisType, NamedSharding, PartitionSpec as P, use_abstract_mesh

from levanter.optim.grugmuon import (
    GrugMuonConfig,
    STACK_BATCH_SHARDED,
    VMAP_REPLICATED,
    _grug_scale_with_muon,
    _newtonschulz_4d_distributed,
    _newtonschulz_padded_stack_sharded,
    _zeropower_via_newtonschulz_batched_stack_sharded,
    _zeropower_via_newtonschulz_replicated,
)


def _run_cpu_mesh_script(script: str):
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_grug_scale_with_muon_orthogonalizes_matrix_trailing_dims():
    updates = {
        "matrix": jnp.ones((2, 3), dtype=jnp.float32),
        "moe_tensor": jnp.ones((2, 2, 2), dtype=jnp.float32),
        "vector": jnp.ones((3,), dtype=jnp.float32),
    }
    transform = _grug_scale_with_muon(
        momentum=0.0,
        nesterov=False,
        use_kimi_scaling=False,
        orthogonalization_layout=VMAP_REPLICATED,
    )

    new_updates, _ = transform.update(updates, transform.init(updates))

    assert new_updates["matrix"].shape == updates["matrix"].shape
    assert new_updates["moe_tensor"].shape == updates["moe_tensor"].shape
    assert not jnp.array_equal(new_updates["matrix"], updates["matrix"])
    assert not jnp.array_equal(new_updates["moe_tensor"], updates["moe_tensor"])
    assert jnp.array_equal(new_updates["vector"], updates["vector"])


def test_grug_scale_with_muon_warns_when_newton_schulz_is_disabled(monkeypatch, caplog):
    monkeypatch.setenv("SCALE_MUON_NO_NS", "1")

    with caplog.at_level(logging.WARNING):
        _grug_scale_with_muon()

    assert "SCALE_MUON_NO_NS=1 disables Newton-Schulz" in caplog.text


def test_grug_scale_with_muon_warns_when_env_overrides_explicit_layout(monkeypatch, caplog):
    monkeypatch.setenv("SCALE_MUON_DIST_NONEXPERT", "1")

    with caplog.at_level(logging.WARNING):
        _grug_scale_with_muon(orthogonalization_layout=VMAP_REPLICATED)

    assert "SCALE_MUON_DIST_NONEXPERT=1 overrides orthogonalization_layout" in caplog.text


def test_grug_muon_mask_routes_matrix_stacks_to_muon():
    params = {
        "embed": jnp.ones((16, 8), dtype=jnp.float32),
        "router": jnp.ones((8, 4), dtype=jnp.float32),
        "norm": jnp.ones((2, 8), dtype=jnp.float32),
        "attention": {"w_q": jnp.ones((2, 8, 16), dtype=jnp.float32)},
        "moe": {
            "w_up_gate": jnp.ones((2, 4, 8, 16), dtype=jnp.float32),
            "w_gate_up": jnp.ones((2, 4, 8, 16), dtype=jnp.float32),
            "w_down": jnp.ones((2, 4, 16, 8), dtype=jnp.float32),
        },
        "vector": jnp.ones((8,), dtype=jnp.float32),
    }

    mask = GrugMuonConfig().create_mask(params)

    assert mask["embed"] == "adamw"
    assert mask["router"] == "muon"
    assert mask["norm"] == "adamw"
    assert mask["attention"]["w_q"] == "muon"
    assert mask["moe"]["w_up_gate"] == "muon"
    assert mask["moe"]["w_gate_up"] == "muon"
    assert mask["moe"]["w_down"] == "muon"
    assert mask["vector"] == "adamw"


def test_batched_stack_sharded_matches_vmap_replicated_without_mesh():
    x = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)

    expected = jax.vmap(
        lambda matrix: _zeropower_via_newtonschulz_replicated(matrix, steps=2, eps=1e-7, coefficient_type="quintic")
    )(x)
    actual = _zeropower_via_newtonschulz_batched_stack_sharded(
        x,
        steps=2,
        eps=1e-7,
        coefficient_type="quintic",
    )

    assert jnp.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_grug_scale_with_muon_stack_batch_sharded_handles_stacked_expert_tensor():
    updates = {"moe_tensor": jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)}
    transform = _grug_scale_with_muon(
        momentum=0.0,
        nesterov=False,
        use_kimi_scaling=False,
        orthogonalization_layout=STACK_BATCH_SHARDED,
    )

    new_updates, _ = transform.update(updates, transform.init(updates))

    assert new_updates["moe_tensor"].shape == updates["moe_tensor"].shape
    assert not jnp.array_equal(new_updates["moe_tensor"], updates["moe_tensor"])


@pytest.mark.parametrize(
    ("weight_name", "input_spec", "output_spec"),
    [
        ("w_gate", P(None, "expert", None, None), P(None, "expert", "data", "model")),
        ("w_down", P(None, "expert", None, None), P(None, "expert", "model", "data")),
    ],
)
def test_distributed_4d_ns_keeps_expert_axis_unmerged(weight_name, input_spec, output_spec):
    mesh = AbstractMesh(
        axis_sizes=(1, 1, 64, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    input_sharding = NamedSharding(mesh, input_spec)
    x = jax.ShapeDtypeStruct((48, 256, 8, 4), jnp.float32, sharding=input_sharding)

    def apply_ns(y):
        path = (jax.tree_util.GetAttrKey(weight_name),)
        return _newtonschulz_4d_distributed(path, y, steps=0, eps=1e-8, coefficient_type="quintic")

    with use_abstract_mesh(mesh):
        closed_jaxpr = jax.make_jaxpr(apply_ns)(x)
        output = jax.eval_shape(apply_ns, x)

    merged_shape = (48 * 256, 8, 4)
    merged_reshapes = [
        eqn
        for eqn in closed_jaxpr.jaxpr.eqns
        if eqn.primitive.name == "reshape" and eqn.outvars[0].aval.shape == merged_shape
    ]
    assert not merged_reshapes
    assert output.sharding == NamedSharding(mesh, output_spec)


def test_distributed_4d_ns_matches_replicated_path_on_cpu_mesh():
    _run_cpu_mesh_script(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

        from levanter.optim.grugmuon import (
            _newtonschulz_4d_distributed,
            _zeropower_via_newtonschulz_replicated,
        )

        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 1, 4, 1),
            ("replica_dcn", "data", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 4,
        )
        x = jax.random.normal(jax.random.key(0), (2, 4, 8, 4), dtype=jnp.float32)
        input_sharding = NamedSharding(mesh, P(None, "expert", "data", "model"))
        x_sharded = jax.device_put(x, input_sharding)
        path = (jax.tree_util.GetAttrKey("w_gate"),)
        expected = jax.vmap(
            jax.vmap(
                lambda matrix: _zeropower_via_newtonschulz_replicated(
                    matrix, steps=2, eps=1e-7, coefficient_type="quintic"
                )
            )
        )(x)

        apply_ns = jax.jit(
            lambda y: _newtonschulz_4d_distributed(
                path, y, steps=2, eps=1e-7, coefficient_type="quintic"
            )
        )
        with jax.set_mesh(mesh):
            actual = apply_ns(x_sharded)

        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5, rtol=1e-5)
        """
    )


def test_padded_stack_ns_uses_two_hop_inbound_reshard():
    mesh = AbstractMesh(
        axis_sizes=(2, 1, 4, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    input_sharding = NamedSharding(mesh, P(None, "expert", None))
    x = jax.ShapeDtypeStruct((6, 8, 4), jnp.float32, sharding=input_sharding)

    def apply_ns(y):
        return _newtonschulz_padded_stack_sharded(
            y,
            steps=0,
            eps=1e-8,
            coefficient_type="quintic",
        )

    with use_abstract_mesh(mesh):
        closed_jaxpr = jax.make_jaxpr(apply_ns)(x)

    padded_shape = (8, 8, 4)
    padded_reshard_specs = [
        eqn.params["dst_sharding"].spec
        for eqn in closed_jaxpr.jaxpr.eqns
        if eqn.primitive.name == "reshard" and eqn.invars[0].aval.shape == padded_shape
    ]
    assert padded_reshard_specs[:2] == [
        P("replica_dcn", None, None),
        P(("replica_dcn", "expert"), None, None),
    ]


def test_padded_stack_ns_returns_directly_to_parameter_sharding():
    mesh = AbstractMesh(
        axis_sizes=(1, 1, 64, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    input_sharding = NamedSharding(mesh, P(None, "expert", None))
    x = jax.ShapeDtypeStruct((48, 64, 4), jnp.float32, sharding=input_sharding)

    def apply_ns(y):
        return _newtonschulz_padded_stack_sharded(
            y,
            steps=0,
            eps=1e-8,
            coefficient_type="quintic",
            target_sharding=input_sharding,
        )

    with use_abstract_mesh(mesh):
        closed_jaxpr = jax.make_jaxpr(apply_ns)(x)
        output = jax.eval_shape(apply_ns, x)

    padded_shape = (64, 64, 4)
    replicated_padded_reshards = [
        eqn
        for eqn in closed_jaxpr.jaxpr.eqns
        if eqn.primitive.name == "reshard"
        and eqn.invars[0].aval.shape == padded_shape
        and eqn.params["dst_sharding"].spec == P(None, None, None)
    ]
    assert not replicated_padded_reshards
    assert output.shape == x.shape
    assert output.sharding == input_sharding


def test_padded_stack_ns_matches_unsharded_path_on_cpu_mesh():
    _run_cpu_mesh_script(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

        from levanter.optim.grugmuon import (
            _newtonschulz_padded_stack_sharded,
            _zeropower_via_newtonschulz_replicated,
        )

        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 1, 4, 1),
            ("replica_dcn", "data", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 4,
        )
        x = jax.random.normal(jax.random.key(1), (3, 8, 4), dtype=jnp.float32)
        parameter_sharding = NamedSharding(mesh, P(None, "expert", None))
        x_sharded = jax.device_put(x, parameter_sharding)
        expected = jax.vmap(
            lambda matrix: _zeropower_via_newtonschulz_replicated(
                matrix, steps=2, eps=1e-7, coefficient_type="quintic"
            )
        )(x)

        apply_ns = jax.jit(
            lambda y: _newtonschulz_padded_stack_sharded(
                y,
                steps=2,
                eps=1e-7,
                coefficient_type="quintic",
                target_sharding=parameter_sharding,
            )
        )
        with jax.set_mesh(mesh):
            actual = apply_ns(x_sharded)

        assert actual.sharding == parameter_sharding
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5, rtol=1e-5)
        """
    )
