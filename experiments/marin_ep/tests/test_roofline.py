# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""L0 roofline behavior: scaling laws and agreement with simulated traces."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.marin_ep.perfmodel.roofline import (
    GB200,
    HERO,
    MoeShape,
    estimate_from_trace,
    estimate_layer,
    hero_report,
)
from experiments.marin_ep.simcore import simulate_forward

SMALL = MoeShape(
    tokens_per_device=16, topk=2, hidden=8, intermediate=12, num_experts=8, ep=4, capacity_factor=1e9, wire_bytes=4
)


def test_ep1_has_no_transport_time():
    single = dataclasses.replace(SMALL, ep=1)
    est = estimate_layer(single, GB200, backward=False)
    assert est.transport_time == 0.0
    assert est.total_serial == est.gemm_time


def test_gemm_time_scales_linearly_with_tokens_and_transport_with_wire_bytes():
    est = estimate_layer(SMALL, GB200, backward=False)
    doubled_tokens = estimate_layer(dataclasses.replace(SMALL, tokens_per_device=32), GB200, backward=False)
    assert doubled_tokens.gemm_time == pytest.approx(2 * est.gemm_time)
    halved_wire = estimate_layer(dataclasses.replace(SMALL, wire_bytes=2), GB200, backward=False)
    assert halved_wire.transport_time == pytest.approx(est.transport_time / 2)


def test_overlap_interpolates_between_serial_and_max():
    est = estimate_layer(SMALL, GB200, backward=False)
    assert est.total(0.0) == est.total_serial
    assert est.total(1.0) == est.total_overlapped
    assert est.total_overlapped <= est.total(0.5) <= est.total_serial


def test_trace_estimate_matches_closed_form_on_balanced_routing():
    """Two independent accountings (event trace vs closed form) must agree when
    routing is perfectly balanced and dropless."""
    devices, tokens, topk, num_experts, hidden, intermediate = 4, 16, 2, 8, 8, 12
    rng = np.random.default_rng(seed=7)
    # Round-robin routing: exactly balanced across experts, so per-device
    # egress equals the expected remote fraction exactly.
    flat = np.arange(devices * tokens * topk) % num_experts
    experts = flat.reshape(devices, tokens, topk).astype(np.int32)
    x = rng.standard_normal((devices, tokens, hidden), dtype=np.float32)
    weights = rng.random((devices, tokens, topk), dtype=np.float32)
    w13 = rng.standard_normal((num_experts, hidden, 2 * intermediate)).astype(np.float32)
    w2 = rng.standard_normal((num_experts, intermediate, hidden)).astype(np.float32)

    result = simulate_forward(
        jnp.asarray(x),
        experts,
        jnp.asarray(weights),
        jnp.asarray(w13).reshape(devices, num_experts // devices, hidden, 2 * intermediate),
        jnp.asarray(w2).reshape(devices, num_experts // devices, intermediate, hidden),
        num_experts=num_experts,
        capacity_factor=1e9,
        activation_fn=jax.nn.silu,
    )
    shape = MoeShape(
        tokens_per_device=tokens,
        topk=topk,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        ep=devices,
        capacity_factor=1e9,
        wire_bytes=4,
    )
    traced = estimate_from_trace(result.trace, GB200)
    closed = estimate_layer(shape, GB200, backward=False)
    # Balanced routing: each device's egress carries its dispatch bytes (as
    # source) plus its combine bytes (as owner) = the closed form's 2x one-way,
    # plus the F2 count exchange (E int32 counts to each peer), which the
    # closed form deliberately ignores.
    count_exchange_time = (devices - 1) * num_experts * 4 / (GB200.link_bytes * GB200.link_efficiency)
    assert traced.gemm_time == pytest.approx(closed.gemm_time, rel=1e-9)
    assert traced.transport_time == pytest.approx(closed.transport_time + count_exchange_time, rel=1e-9)


def test_hero_report_orders_gemm_above_transport():
    """At the hero shape the layer must be GEMM-bound, not transport-bound —
    the premise of the overlap design. Guard it with the model itself."""
    est = estimate_layer(HERO, GB200, backward=False)
    assert est.gemm_time > est.transport_time
    report = hero_report()
    assert "fwd" in report and "bwd" in report
