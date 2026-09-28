# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track stacked Muon: orthogonalizing same-shaped matrices of different leaves in one batched
Newton-Schulz call gives the per-leaf directions."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.testing.cpu_devices import run_on_cpu_devices

from experiments.grug.fast_track.grugmuon_stacked import _grug_scale_with_muon, _zeropower_via_newtonschulz_local


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


def check_bucketed_newton_schulz_matches_per_leaf(coefficient_type: str, steps: int) -> None:
    mesh = Mesh(
        np.asarray(jax.devices()[:2]).reshape((1, 2, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    transform = _grug_scale_with_muon(momentum=0.95, nesterov=True, steps=steps, coefficient_type=coefficient_type)
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


@pytest.mark.parametrize(("coefficient_type", "steps"), [("quintic", 5), ("aol", 4)])
def test_bucketed_newton_schulz_matches_per_leaf(coefficient_type, steps):
    # bf16 Newton-Schulz amplifies single roundings; without this flag XLA:CPU may skip the per-leaf
    # path's bf16 cast when fusing it into the first Gram matmul, which AOL feeds the raw input.
    run_on_cpu_devices(
        "import os; os.environ['XLA_FLAGS'] += ' --xla_allow_excess_precision=false'; "
        f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); "
        "import test_fast_track_muon_bucketing as t; "
        f"t.check_bucketed_newton_schulz_matches_per_leaf({coefficient_type!r}, {steps})",
        device_count=2,
    )


@pytest.mark.parametrize("shape", [(256, 384), (384, 256)])
def test_aol_newton_schulz_approximates_polar_factor(shape):
    """Turbo-Muon's four AOL-preconditioned steps land every singular value near 1, like five quintic steps."""
    x = jax.random.normal(jax.random.key(0), shape)
    u, _, vt = np.linalg.svd(np.asarray(x), full_matrices=False)
    polar = u @ vt
    result = np.asarray(_zeropower_via_newtonschulz_local(x, 4, 1e-8, "aol"), np.float32)
    singular_values = np.linalg.svd(result, compute_uv=False)
    assert singular_values.min() > 0.95 and singular_values.max() < 1.1
    assert np.linalg.norm(result - polar) / np.linalg.norm(polar) < 0.05


def test_aol_rejects_other_step_counts():
    with pytest.raises(ValueError, match="exactly 4"):
        _grug_scale_with_muon(steps=5, coefficient_type="aol")
