# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance tests: simulator vs dense oracle (SPEC.md Section 6 invariants).

The oracle (`oracle.py`) computes the keep rule from the global argsort and
values densely with JAX autodiff gradients; the simulator (`simcore.py`)
computes the keep rule from distributed counts+offsets and values via
message-passing device programs with an explicit backward. Parity between
the two exercises both independent derivations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.marin_ep.oracle import expert_capacity, moe_oracle, pooled_keep_mask
from experiments.marin_ep.simcore import simulate_backward, simulate_forward

jax.config.update("jax_enable_x64", False)

ACT = jax.nn.silu


def _random_instance(rng, *, devices, tokens, topk, hidden, intermediate, num_experts, skew=None):
    """Random inputs; `skew` concentrates routing (dirichlet alpha < 1 = hot experts)."""
    x = rng.standard_normal((devices, tokens, hidden), dtype=np.float32)
    weights = rng.random((devices, tokens, topk), dtype=np.float32) + 0.05
    if skew is None:
        experts = rng.integers(0, num_experts, size=(devices, tokens, topk))
    else:
        probs = rng.dirichlet(np.full(num_experts, skew))
        experts = rng.choice(num_experts, size=(devices, tokens, topk), p=probs)
    w13 = (0.3 * rng.standard_normal((num_experts, hidden, 2 * intermediate))).astype(np.float32)
    w2 = (0.3 * rng.standard_normal((num_experts, intermediate, hidden))).astype(np.float32)
    return x, experts.astype(np.int32), weights, w13, w2


def _shard_weights(w, devices):
    return jnp.asarray(w).reshape(devices, w.shape[0] // devices, *w.shape[1:])


@pytest.mark.parametrize(
    "devices,tokens,topk,num_experts,skew,group_size",
    [
        (1, 32, 2, 8, None, 1),
        (4, 16, 2, 8, None, 1),
        (4, 16, 3, 8, 0.3, 1),
        (4, 16, 3, 8, 0.3, 2),
        (8, 8, 4, 16, 0.1, 1),
        (8, 8, 4, 16, 0.1, 2),
    ],
)
@pytest.mark.parametrize("capacity_factor", [1.0, 1.25, 1e9])
def test_forward_matches_oracle_across_shapes_and_capacity(
    devices, tokens, topk, num_experts, skew, group_size, capacity_factor
):
    rng = np.random.default_rng(seed=hash((devices, tokens, topk, num_experts, capacity_factor)) % 2**32)
    x, experts, weights, w13, w2 = _random_instance(
        rng, devices=devices, tokens=tokens, topk=topk, hidden=8, intermediate=12, num_experts=num_experts, skew=skew
    )
    capacity = expert_capacity(devices * tokens * topk, num_experts, capacity_factor)
    keep, dropped_oracle = pooled_keep_mask(experts, num_experts=num_experts, capacity=capacity, group_size=group_size)

    result = simulate_forward(
        jnp.asarray(x),
        experts,
        jnp.asarray(weights),
        _shard_weights(w13, devices),
        _shard_weights(w2, devices),
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        activation_fn=ACT,
        pool_group_size=group_size,
    )
    y_oracle = moe_oracle(
        jnp.asarray(x),
        jnp.asarray(experts),
        jnp.asarray(weights),
        jnp.asarray(w13),
        jnp.asarray(w2),
        jnp.asarray(keep),
        activation_fn=ACT,
    )

    assert result.dropped_total == dropped_oracle
    np.testing.assert_array_equal(np.asarray(result.saved.routing.keep), keep)
    np.testing.assert_allclose(np.asarray(result.y), np.asarray(y_oracle), rtol=1e-5, atol=1e-5)
    if capacity_factor >= 1e9:
        assert result.dropped_total == 0


def test_drop_accounting_under_total_hot_expert_skew():
    """All assignments to one expert: drops equal the exact overflow, output still matches."""
    rng = np.random.default_rng(seed=0)
    devices, tokens, topk, num_experts = 4, 8, 2, 8
    x, _, weights, w13, w2 = _random_instance(
        rng, devices=devices, tokens=tokens, topk=topk, hidden=8, intermediate=12, num_experts=num_experts
    )
    experts = np.full((devices, tokens, topk), 5, dtype=np.int32)
    capacity_factor = 1.0
    assignments_global = devices * tokens * topk
    capacity = expert_capacity(assignments_global, num_experts, capacity_factor)

    result = simulate_forward(
        jnp.asarray(x),
        experts,
        jnp.asarray(weights),
        _shard_weights(w13, devices),
        _shard_weights(w2, devices),
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        activation_fn=ACT,
    )
    assert result.dropped_total == assignments_global - capacity

    keep, _ = pooled_keep_mask(experts, num_experts=num_experts, capacity=capacity)
    y_oracle = moe_oracle(
        jnp.asarray(x),
        jnp.asarray(experts),
        jnp.asarray(weights),
        jnp.asarray(w13),
        jnp.asarray(w2),
        jnp.asarray(keep),
        activation_fn=ACT,
    )
    np.testing.assert_allclose(np.asarray(result.y), np.asarray(y_oracle), rtol=1e-5, atol=1e-5)


def test_group_pool_floor_protects_cold_expert_and_grants_surplus():
    """SPEC S2 waterfilling: a hot expert cannot push a cold groupmate below
    its floor C, and unused floor capacity is granted to the hot expert."""
    num_experts, capacity, group_size = 2, 4, 2
    # Expert 0: 10 assignments (hot). Expert 1: 2 assignments (cold, under C=4).
    experts = np.array([[0] * 10 + [1] * 2]).reshape(1, 12, 1)
    keep, dropped = pooled_keep_mask(experts, num_experts=num_experts, capacity=capacity, group_size=group_size)
    keep = keep.reshape(-1)
    # Cold expert keeps everything; hot expert gets C plus the groupmate's slack of 2.
    assert keep[10:].all()
    assert keep[:10].sum() == capacity + 2
    assert keep[:6].all() and not keep[6:10].any()  # earliest arrivals kept
    assert dropped == 10 - (capacity + 2)


@pytest.mark.parametrize(
    "group_size,ep_degrees",
    [(1, (1, 2, 4, 8)), (2, (1, 2, 4))],  # G must divide El = E / S
)
def test_output_independent_of_ep_partitioning(group_size, ep_degrees):
    """SPEC I3: for fixed pool group size, the same global batch gives
    bitwise-identical results at every EP degree."""
    rng = np.random.default_rng(seed=1)
    total_tokens, topk, num_experts, hidden, intermediate = 32, 3, 8, 8, 12
    x, experts, weights, w13, w2 = _random_instance(
        rng,
        devices=1,
        tokens=total_tokens,
        topk=topk,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        skew=0.3,
    )

    def resharded(a, devices):
        return a.reshape(devices, total_tokens // devices, *a.shape[2:])

    outputs, dropped = [], []
    for devices in ep_degrees:
        result = simulate_forward(
            jnp.asarray(resharded(x, devices)),
            resharded(experts, devices),
            jnp.asarray(resharded(weights, devices)),
            _shard_weights(w13, devices),
            _shard_weights(w2, devices),
            num_experts=num_experts,
            capacity_factor=1.1,
            activation_fn=ACT,
            pool_group_size=group_size,
        )
        outputs.append(np.asarray(result.y).reshape(total_tokens, hidden))
        dropped.append(result.dropped_total)

    assert len(set(dropped)) == 1
    for other in outputs[1:]:
        np.testing.assert_array_equal(outputs[0], other)


@pytest.mark.parametrize("capacity_factor", [1.1, 1e9])
@pytest.mark.parametrize("group_size", [1, 2])
def test_backward_matches_oracle_autodiff(capacity_factor, group_size):
    """SPEC I5: explicit backward parity with JAX autodiff of the keep-masked oracle."""
    rng = np.random.default_rng(seed=2)
    devices, tokens, topk, num_experts, hidden, intermediate = 4, 12, 3, 8, 8, 12
    x, experts, weights, w13, w2 = _random_instance(
        rng,
        devices=devices,
        tokens=tokens,
        topk=topk,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        skew=0.5,
    )
    capacity = expert_capacity(devices * tokens * topk, num_experts, capacity_factor)
    keep, _ = pooled_keep_mask(experts, num_experts=num_experts, capacity=capacity, group_size=group_size)
    dy = rng.standard_normal((devices, tokens, hidden), dtype=np.float32)

    result = simulate_forward(
        jnp.asarray(x),
        experts,
        jnp.asarray(weights),
        _shard_weights(w13, devices),
        _shard_weights(w2, devices),
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        activation_fn=ACT,
        pool_group_size=group_size,
    )
    bwd = simulate_backward(
        jnp.asarray(dy),
        jnp.asarray(weights),
        _shard_weights(w13, devices),
        _shard_weights(w2, devices),
        result.saved,
        activation_fn=ACT,
    )

    def oracle_fn(x_, weights_, w13_, w2_):
        return moe_oracle(x_, jnp.asarray(experts), weights_, w13_, w2_, jnp.asarray(keep), activation_fn=ACT)

    _, vjp = jax.vjp(oracle_fn, jnp.asarray(x), jnp.asarray(weights), jnp.asarray(w13), jnp.asarray(w2))
    dx_ref, dweights_ref, dw13_ref, dw2_ref = vjp(jnp.asarray(dy))

    np.testing.assert_allclose(np.asarray(bwd.dx), np.asarray(dx_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(bwd.dcombine_weights), np.asarray(dweights_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(bwd.dw13).reshape(num_experts, hidden, 2 * intermediate),
        np.asarray(dw13_ref),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(bwd.dw2).reshape(num_experts, intermediate, hidden),
        np.asarray(dw2_ref),
        rtol=1e-5,
        atol=1e-5,
    )


def test_forward_and_backward_are_deterministic():
    """SPEC I6: repeated simulation is bitwise identical."""
    rng = np.random.default_rng(seed=3)
    devices, tokens, topk, num_experts = 4, 8, 2, 8
    x, experts, weights, w13, w2 = _random_instance(
        rng,
        devices=devices,
        tokens=tokens,
        topk=topk,
        hidden=8,
        intermediate=12,
        num_experts=num_experts,
        skew=0.2,
    )
    dy = rng.standard_normal((devices, tokens, 8), dtype=np.float32)

    def run():
        fwd = simulate_forward(
            jnp.asarray(x),
            experts,
            jnp.asarray(weights),
            _shard_weights(w13, devices),
            _shard_weights(w2, devices),
            num_experts=num_experts,
            capacity_factor=1.05,
            activation_fn=ACT,
        )
        bwd = simulate_backward(
            jnp.asarray(dy),
            jnp.asarray(weights),
            _shard_weights(w13, devices),
            _shard_weights(w2, devices),
            fwd.saved,
            activation_fn=ACT,
        )
        return np.asarray(fwd.y), np.asarray(bwd.dx), np.asarray(bwd.dw13)

    first, second = run(), run()
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_trace_dispatch_bytes_account_for_kept_assignments_only():
    """The comm trace is the perf-model input: dispatch bytes must equal kept rows x H x itemsize."""
    rng = np.random.default_rng(seed=4)
    devices, tokens, topk, num_experts, hidden = 4, 8, 2, 8, 8
    x, experts, weights, w13, w2 = _random_instance(
        rng,
        devices=devices,
        tokens=tokens,
        topk=topk,
        hidden=hidden,
        intermediate=12,
        num_experts=num_experts,
        skew=0.2,
    )
    result = simulate_forward(
        jnp.asarray(x),
        experts,
        jnp.asarray(weights),
        _shard_weights(w13, devices),
        _shard_weights(w2, devices),
        num_experts=num_experts,
        capacity_factor=1.0,
        activation_fn=ACT,
    )
    keep = np.asarray(result.saved.routing.keep)
    local_experts = num_experts // devices
    owner = np.asarray(experts) // local_experts
    remote_kept = int((keep & (owner != np.arange(devices)[:, None, None])).sum())
    itemsize = 4
    assert result.trace.comm_bytes(phase="fwd_dispatch") == remote_kept * hidden * itemsize
    assert result.trace.comm_bytes(phase="fwd_combine") == remote_kept * hidden * itemsize
