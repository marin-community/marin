# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess
import sys
import textwrap
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from levanter.grug._moe.ep_ring import (
    _moe_mlp_ep_ring_fp8_wire_approx_local,
    _moe_mlp_ep_ring_local,
    _validate_fp8_wire_contract,
)


def _single_device_mesh() -> Mesh:
    return Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1),
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


@pytest.mark.parametrize("overflow", (False, True))
def test_fp8_wire_preserves_ring_routing_and_produces_finite_output(overflow: bool) -> None:
    mesh = _single_device_mesh()
    tokens = 8
    topk = 2
    hidden_dim = 8
    intermediate_dim = 4
    num_experts = 2
    capacity_factor = 0.5 if overflow else 1.0
    keys = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(keys[0], (tokens, hidden_dim), dtype=jnp.bfloat16)
    if overflow:
        selected_experts = jnp.zeros((tokens, topk), dtype=jnp.int32)
    else:
        selected_experts = jnp.arange(tokens * topk, dtype=jnp.int32).reshape(tokens, topk) % num_experts
    combine_weights = jax.nn.softmax(
        jax.random.normal(keys[1], selected_experts.shape, dtype=jnp.float32),
        axis=-1,
    )
    w13 = 0.02 * jax.random.normal(
        keys[2],
        (num_experts, hidden_dim, 2 * intermediate_dim),
        dtype=jnp.bfloat16,
    )
    w2 = 0.02 * jax.random.normal(
        keys[3],
        (num_experts, intermediate_dim, hidden_dim),
        dtype=jnp.bfloat16,
    )
    batch_spec = P(("data", "expert"), None)
    expert_spec = P("expert", None, None)
    batch_sharding = NamedSharding(mesh, batch_spec)
    expert_sharding = NamedSharding(mesh, expert_spec)

    def runner(local_fn):
        return jax.jit(
            jax.shard_map(
                partial(
                    local_fn,
                    activation_fn=jax.nn.silu,
                    num_experts=num_experts,
                    capacity_factor=capacity_factor,
                ),
                mesh=mesh,
                in_specs=(batch_spec, batch_spec, batch_spec, expert_spec, expert_spec),
                out_specs=(batch_spec, P()),
                check_vma=False,
            )
        )

    with jax.set_mesh(mesh):
        inputs = (
            jax.device_put(x, batch_sharding),
            jax.device_put(selected_experts, batch_sharding),
            jax.device_put(combine_weights, batch_sharding),
            jax.device_put(w13, expert_sharding),
            jax.device_put(w2, expert_sharding),
        )
        ring_output, ring_dropped = runner(_moe_mlp_ep_ring_local)(*inputs)
        fp8_output, fp8_dropped = runner(_moe_mlp_ep_ring_fp8_wire_approx_local)(*inputs)

    assert int(fp8_dropped) == int(ring_dropped)
    assert (int(fp8_dropped) > 0) == overflow
    assert fp8_output.dtype == jnp.bfloat16
    assert np.isfinite(np.asarray(fp8_output)).all()


@pytest.mark.parametrize(
    ("field", "value", "error", "match"),
    (
        ("x", jax.ShapeDtypeStruct((8, 8), jnp.float32), TypeError, "bfloat16"),
        ("selected", jax.ShapeDtypeStruct((8, 2), jnp.float32), TypeError, "integer"),
        ("weights", jax.ShapeDtypeStruct((8, 2), jnp.bfloat16), TypeError, "float32"),
        ("selected", jax.ShapeDtypeStruct((8, 3), jnp.int32), ValueError, "same shape"),
        ("w13", jax.ShapeDtypeStruct((2, 8, 10), jnp.bfloat16), ValueError, "twice"),
    ),
)
def test_fp8_wire_rejects_unsupported_contract(field, value, error, match) -> None:
    arguments = {
        "x": jax.ShapeDtypeStruct((8, 8), jnp.bfloat16),
        "selected": jax.ShapeDtypeStruct((8, 2), jnp.int32),
        "weights": jax.ShapeDtypeStruct((8, 2), jnp.float32),
        "w13": jax.ShapeDtypeStruct((2, 8, 8), jnp.bfloat16),
        "w2": jax.ShapeDtypeStruct((2, 4, 8), jnp.bfloat16),
    }
    arguments[field] = value

    with pytest.raises(error, match=match):
        _validate_fp8_wire_contract(
            arguments["x"],
            arguments["selected"],
            arguments["weights"],
            arguments["w13"],
            arguments["w2"],
        )


def test_fp8_wire_forced_ep8_hlo_uses_fp8_payload_collectives() -> None:
    script = textwrap.dedent(
        """
        from functools import partial

        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

        from levanter.grug._moe.ep_ring import _moe_mlp_ep_ring_fp8_wire_approx_local

        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 8, 1),
            ("data", "expert", "model"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )
        batch_spec = P(("data", "expert"), None)
        expert_spec = P("expert", None, None)
        mapped = jax.jit(
            jax.shard_map(
                partial(
                    _moe_mlp_ep_ring_fp8_wire_approx_local,
                    activation_fn=jax.nn.silu,
                    num_experts=16,
                    capacity_factor=1.0,
                ),
                mesh=mesh,
                in_specs=(batch_spec, batch_spec, batch_spec, expert_spec, expert_spec),
                out_specs=(batch_spec, P()),
                check_vma=False,
            )
        )
        with jax.set_mesh(mesh):
            batch_sharding = NamedSharding(mesh, batch_spec)
            expert_sharding = NamedSharding(mesh, expert_spec)
            inputs = (
                jax.device_put(jnp.ones((16, 8), dtype=jnp.bfloat16), batch_sharding),
                jax.device_put(jnp.arange(32, dtype=jnp.int32).reshape(16, 2) % 16, batch_sharding),
                jax.device_put(jnp.full((16, 2), 0.5, dtype=jnp.float32), batch_sharding),
                jax.device_put(jnp.full((16, 8, 8), 0.01, dtype=jnp.bfloat16), expert_sharding),
                jax.device_put(jnp.full((16, 4, 8), 0.01, dtype=jnp.bfloat16), expert_sharding),
            )
            def loss(*arguments):
                output, _ = mapped(*arguments)
                return jnp.mean(jnp.square(output.astype(jnp.float32)))

            value_and_grad = jax.jit(jax.value_and_grad(loss, argnums=(0, 2, 3, 4)))
            print("FORWARD_HLO")
            print(mapped.lower(*inputs).as_text())
            print("VALUE_AND_GRAD_HLO")
            print(value_and_grad.lower(*inputs).as_text())
        """
    )
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    hlo = result.stdout
    assert "FORWARD_HLO" in hlo
    assert "VALUE_AND_GRAD_HLO" in hlo
    all_gathers = [line for line in hlo.splitlines() if '"stablehlo.all_gather"' in line]
    reduce_scatters = re.findall(
        r'"stablehlo\.reduce_scatter".*?\) : \(tensor<[^>]*xf8E4M3FN>\) -> tensor<[^>]*xf8E4M3FN>',
        hlo,
        flags=re.DOTALL,
    )

    assert any("xf8E4M3FN>" in line for line in all_gathers)
    assert not any("xbf16>" in line for line in all_gathers)
    assert len(reduce_scatters) >= 2
