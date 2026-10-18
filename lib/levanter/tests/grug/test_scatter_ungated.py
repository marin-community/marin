# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Dropless scatter outputs and VJPs against direct per-route arithmetic."""
import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug._moe.local import _moe_mlp_local


def test_scatter_gated_and_ungated_routes():
    rng = np.random.default_rng(42)
    cases = []
    for gated in (False, True):
        for dtype in (jnp.float32, jnp.bfloat16):
            for routing in ("mixed", "concentrated", "all-masked"):
                t, h, i, e, k = 13, 8, 6, 5, 3
                selected = jnp.asarray(rng.integers(e, size=(t, k)), jnp.int32)
                if routing == "concentrated":
                    selected = jnp.zeros((t, k), jnp.int32)
                valid = jnp.asarray(np.arange(t) % 3 != 0)
                if routing == "all-masked":
                    valid = jnp.zeros(t, bool)
                arrays = tuple(
                    jnp.asarray(rng.normal(0, 0.2, size=s), dtype)
                    for s in ((t, h), (t, k), (e, h, i * (2 if gated else 1)), (e, i, h))
                )
                cotangent = jnp.asarray(rng.normal(size=(t, h)), dtype)
                activation = jax.nn.silu if gated else lambda x: jax.nn.relu(x) ** 2

                def reference(x, w, up, down):
                    hidden = jnp.einsum("th,tkhi->tki", x, up[selected])
                    if gated:
                        a, b = jnp.split(hidden, 2, axis=-1)
                        hidden = activation(a) * b
                    else:
                        hidden = activation(hidden)
                    routed = jnp.einsum("tki,tkih->tkh", hidden, down[selected])
                    return jnp.sum(jnp.where(valid[:, None, None], routed * w[:, :, None], 0), axis=1)

                def actual(x, w, up, down):
                    result, drops = _moe_mlp_local(
                        x,
                        selected,
                        w,
                        valid,
                        up,
                        down,
                        activation_fn=activation,
                        num_experts=e,
                        implementation="ragged_all_to_all",
                    )
                    return result, drops

                got, drops = jax.jit(actual)(*arrays)
                vg = jax.jit(jax.grad(lambda *a: jnp.sum(actual(*a)[0] * cotangent), argnums=(0, 1, 2, 3)))(*arrays)
                ref = jax.jit(reference)(*arrays)
                vr = jax.jit(jax.grad(lambda *a: jnp.sum(reference(*a) * cotangent), argnums=(0, 1, 2, 3)))(*arrays)
                errors = []
                for expected, observed in zip((ref, *vr), (got, *vg), strict=True):
                    a, b = map(lambda z: np.asarray(z, dtype=np.float32), (expected, observed))
                    np.testing.assert_allclose(
                        b,
                        a,
                        rtol=0.03 if dtype == jnp.bfloat16 else 2e-4,
                        atol=2e-4 if dtype == jnp.bfloat16 else 2e-6,
                    )
                    errors.append(float(np.max(np.abs(a - b))))
                assert int(drops) == 0
                cases.append(
                    dict(
                        gated=gated,
                        dtype=str(dtype),
                        routing=routing,
                        output_and_vjp_max_abs=errors,
                        drops=int(drops),
                        passed=True,
                    )
                )
