# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""L0 performance model: analytic roofline for one Marin EP MoE layer.

Coarsest rung of the simulator ladder (plan Section "Simulator"): closed-form
per-phase bytes/FLOPs with a single overlap knob. Companion
`estimate_from_trace` computes the same quantities from a `simcore.Trace`,
giving two independent accountings that must agree on balanced routing.

All times are seconds for one device's slice of one MoE layer. The model is
deliberately ignorant of tiles, SMs, and latency; those belong to L1+ once a
design decision needs them. Calibration against measured hero segments is a
gate (plan G1b) before any simulated number is trusted.
"""

from collections import defaultdict
from dataclasses import dataclass

from experiments.marin_ep.oracle import expert_capacity
from experiments.marin_ep.simcore import CommEvent, Trace

# Peak dense bf16 per GB200 GPU mirrors lib/fray/src/fray/device_flops.py;
# NVLink5 is 900 GB/s per direction per GPU.
GB200_BF16_FLOPS = 2.5e15
GB200_NVLINK_BYTES = 900e9
GB200_HBM_BYTES = 8e12


@dataclass(frozen=True)
class MoeShape:
    """Per-device routed-MoE workload shape (defaults irrelevant; hero below)."""

    tokens_per_device: int
    topk: int
    hidden: int  # routed (latent) width
    intermediate: int
    num_experts: int
    ep: int
    capacity_factor: float
    wire_bytes: int = 2  # bf16

    @property
    def assignments_per_device(self) -> int:
        return self.tokens_per_device * self.topk


HERO = MoeShape(
    tokens_per_device=65536,
    topk=4,
    hidden=3072,
    intermediate=6272,
    num_experts=192,
    ep=64,
    capacity_factor=1.33,
)


@dataclass(frozen=True)
class Hardware:
    matmul_flops: float  # peak dense FLOP/s per device for the compute dtype
    link_bytes: float  # bytes/s per direction per device (NVLink5: 900e9)
    gemm_efficiency: float = 0.70
    link_efficiency: float = 0.80


GB200 = Hardware(matmul_flops=GB200_BF16_FLOPS, link_bytes=GB200_NVLINK_BYTES)


@dataclass(frozen=True)
class LayerEstimate:
    """Per-device time decomposition for one MoE layer direction (fwd or bwd)."""

    transport_time: float  # one direction-pair (dispatch + combine), serialized on the link
    gemm_time: float
    total_serial: float  # no overlap: transport + gemm
    total_overlapped: float  # perfect overlap: max(transport, gemm)

    def total(self, overlap: float) -> float:
        """Interpolate between serial (overlap=0) and perfectly hidden (overlap=1)."""
        if not 0.0 <= overlap <= 1.0:
            raise ValueError(f"overlap must be in [0, 1], got {overlap}")
        return self.total_serial + overlap * (self.total_overlapped - self.total_serial)


def _remote_fraction(shape: MoeShape) -> float:
    """Expected fraction of assignments crossing NVLink under balanced routing."""
    return (shape.ep - 1) / shape.ep


def transport_bytes_one_way(shape: MoeShape) -> float:
    """Expected remote bytes per device for dispatch (== combine), balanced routing, pre-drop."""
    return shape.assignments_per_device * shape.hidden * shape.wire_bytes * _remote_fraction(shape)


def gemm_flops(shape: MoeShape, *, backward: bool) -> float:
    """Expert GEMM FLOPs per device per layer, balanced routing (12x/6x m*H*I with m = A)."""
    per_row = (12 if backward else 6) * shape.hidden * shape.intermediate
    return float(shape.assignments_per_device) * per_row


def estimate_layer(shape: MoeShape, hw: Hardware, *, backward: bool) -> LayerEstimate:
    link = hw.link_bytes * hw.link_efficiency
    transport = 2 * transport_bytes_one_way(shape) / link  # dispatch + combine on the same egress link
    gemm = gemm_flops(shape, backward=backward) / (hw.matmul_flops * hw.gemm_efficiency)
    return LayerEstimate(
        transport_time=transport,
        gemm_time=gemm,
        total_serial=transport + gemm,
        total_overlapped=max(transport, gemm),
    )


def estimate_from_trace(trace: Trace, hw: Hardware) -> LayerEstimate:
    """Same decomposition, but from a simulated event trace (worst device / worst link).

    Uses the bottleneck device for compute and the bottleneck egress link for
    transport, so skewed routing shows up as a longer estimate than the
    balanced closed form.
    """
    egress: dict[int, int] = defaultdict(int)
    flops: dict[int, int] = defaultdict(int)
    for event in trace.events:
        if isinstance(event, CommEvent):
            if event.src != event.dst:
                egress[event.src] += event.num_bytes
        else:
            flops[event.device] += event.flops
    transport = max(egress.values(), default=0) / (hw.link_bytes * hw.link_efficiency)
    gemm = max(flops.values(), default=0) / (hw.matmul_flops * hw.gemm_efficiency)
    return LayerEstimate(
        transport_time=transport,
        gemm_time=gemm,
        total_serial=transport + gemm,
        total_overlapped=max(transport, gemm),
    )


def hero_report(hw: Hardware = GB200) -> str:
    """Human-readable L0 decomposition at the hero shape."""
    lines = [f"Marin EP L0 roofline at hero shape (EP{HERO.ep}, per device, per layer):"]
    capacity = expert_capacity(HERO.assignments_per_device * HERO.ep, HERO.num_experts, HERO.capacity_factor)
    lines.append(
        f"  per-expert capacity C = {capacity} rows; one-way remote {transport_bytes_one_way(HERO)/2**30:.2f} GiB"
    )
    for backward in (False, True):
        est = estimate_layer(HERO, hw, backward=backward)
        name = "bwd" if backward else "fwd"
        lines.append(
            f"  {name}: transport {est.transport_time*1e3:.2f} ms, gemm {est.gemm_time*1e3:.2f} ms, "
            f"serial {est.total_serial*1e3:.2f} ms, overlapped {est.total_overlapped*1e3:.2f} ms"
        )
    return "\n".join(lines)
