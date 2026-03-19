import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


_BENCH_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "bench" / "bench_moe_hillclimb.py"
)
_BENCH_SPEC = importlib.util.spec_from_file_location("bench_moe_hillclimb", _BENCH_PATH)
assert _BENCH_SPEC is not None and _BENCH_SPEC.loader is not None
bench_moe_hillclimb = importlib.util.module_from_spec(_BENCH_SPEC)
_BENCH_SPEC.loader.exec_module(bench_moe_hillclimb)


@pytest.mark.parametrize("collapse_impl", ["sorted_segment_sum", "scatter_add", "lax_scatter"])
def test_collapse_deepep_local_assignments_impls_match_segment_sum(collapse_impl: str):
    out_dispatch = jnp.array(
        [
            [1.0, -1.0, 0.5],
            [0.5, 0.0, 2.0],
            [3.0, 1.0, -2.0],
            [0.25, -0.5, 1.5],
            [7.0, 7.0, 7.0],
        ],
        dtype=jnp.float32,
    )
    assignment_weights = jnp.array([0.2, 0.8, 1.5, 0.5, 0.0], dtype=jnp.float32)
    recv_token_indices = jnp.array([2, 0, 2, 1, 3], dtype=jnp.int32)
    recv_capacity = 4
    num_recv_tokens = jnp.array(3, dtype=jnp.int32)

    ref = bench_moe_hillclimb._collapse_deepep_local_assignments(
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        recv_capacity=recv_capacity,
        num_recv_tokens=num_recv_tokens,
        collapse_impl="segment_sum",
    )
    got = bench_moe_hillclimb._collapse_deepep_local_assignments(
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        recv_capacity=recv_capacity,
        num_recv_tokens=num_recv_tokens,
        collapse_impl=collapse_impl,
    )
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-6, atol=1e-6)

    def loss_fn(out_dispatch_in, assignment_weights_in, *, impl: str):
        collapsed = bench_moe_hillclimb._collapse_deepep_local_assignments(
            out_dispatch_in,
            assignment_weights_in,
            recv_token_indices,
            recv_capacity=recv_capacity,
            num_recv_tokens=num_recv_tokens,
            collapse_impl=impl,
        )
        return jnp.sum(jnp.square(collapsed))

    ref_grads = jax.grad(loss_fn, argnums=(0, 1))(out_dispatch, assignment_weights, impl="segment_sum")
    got_grads = jax.grad(loss_fn, argnums=(0, 1))(out_dispatch, assignment_weights, impl=collapse_impl)
    np.testing.assert_allclose(np.asarray(got_grads[0]), np.asarray(ref_grads[0]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(got_grads[1]), np.asarray(ref_grads[1]), rtol=1e-6, atol=1e-6)
