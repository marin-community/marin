# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track stacked Muon: orthogonalizing same-shaped matrices of different leaves in one batched
Newton-Schulz call gives the per-leaf directions."""

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.fast_track.grugmuon_stacked import _grug_scale_with_muon


def _updates_and_params(mesh: Mesh):
    """Mixed leaves: wide and tall 3D stacks sharing a wide shape, a 2D matrix of that shape, and two
    leaves of other shapes."""
    shapes_and_specs = {
        "wide": ((3, 8, 16), P(None, "data", None)),
        "tall": ((2, 16, 8), P(None, None, "data")),
        "matrix": ((8, 16), P(None, None)),
        "square": ((1, 12, 12), P(None, "data", None)),
        "small": ((4, 16), P(None, None)),
    }
    keys = jax.random.split(jax.random.key(0), 2 * len(shapes_and_specs))
    updates, params = {}, {}
    for (name, (shape, spec)), k_u, k_p in zip(shapes_and_specs.items(), keys[::2], keys[1::2], strict=True):
        sharding = NamedSharding(mesh, spec)
        updates[name] = jax.device_put(jax.random.normal(k_u, shape), sharding)
        params[name] = jax.device_put(jax.random.normal(k_p, shape), sharding)
    return updates, params


def check_bucketed_newton_schulz_matches_per_leaf() -> None:
    mesh = Mesh(
        np.asarray(jax.devices()[:2]).reshape((1, 2, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    transform = _grug_scale_with_muon(momentum=0.95, nesterov=True, steps=5)
    updates, params = _updates_and_params(mesh)
    with jax.set_mesh(mesh):
        state = transform.init(params)
        bucketed, _ = jax.jit(transform.update)(updates, state, params)
    host = jax.tree.map(lambda x: np.asarray(x), (updates, params))
    # No mesh: every leaf takes its own (vmapped) Newton-Schulz.
    per_leaf, _ = jax.jit(transform.update)(*(jax.tree.map(jnp.asarray, host[0]),), transform.init(host[1]), host[1])
    for name in updates:
        got, want = np.asarray(bucketed[name], np.float32), np.asarray(per_leaf[name], np.float32)
        assert got.shape == want.shape, name
        assert bucketed[name].sharding.spec == params[name].sharding.spec, name
        # bf16 Newton-Schulz: agree to well within one bf16 ulp of the largest entry.
        np.testing.assert_allclose(got, want, rtol=0, atol=2e-3 * np.abs(want).max(), err_msg=name)


def test_bucketed_newton_schulz_matches_per_leaf():
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
    script = (
        f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); "
        "import test_fast_track_muon_bucketing as t; t.check_bucketed_newton_schulz_matches_per_leaf()"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=Path(__file__).parents[2], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-4000:]
