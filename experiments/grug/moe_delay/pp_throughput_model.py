# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parametric v6e/DCN throughput model for pipeline parallelism vs synchronous DP.

The staleness experiments answer "what does PP cost in *tokens* (the token
overhead to reach a target loss)". This answers the other half: "what does PP buy
in *throughput*", so the two can be combined into a viability verdict:

    PP is worth it  <=>  throughput_speedup(PP over sync)  >  token_overhead(PP)

Core mechanism. PP does not change the arithmetic (same FLOPs/token); it changes
*communication* and *memory*. The decisive quantity for large models trained over
multiple slices is cross-slice (DCN) traffic:

    synchronous data/replica parallelism  -> all-reduce the GRADIENTS over DCN:
        volume ~ N_params  (huge for a large MoE; every expert param is reduced)
    pipeline parallelism over DCN         -> pass only stage-boundary ACTIVATIONS:
        volume ~ batch * seq * hidden     (independent of depth / param count)

So PP's DCN advantage is ~ N_params / (batch*seq*hidden), which *grows with model
size*. That is the "PP for large MoEs" argument, quantified here.

Synchronous PP schedules (Zero-Bubble, 1F1B/PipeDream-Flush) are bit-exact and pay
only a pipeline *bubble*; asynchronous PP removes the bubble but pays the measured
token overhead. Both are modeled.

All hardware numbers are explicit, conservative, and easy to override -- the goal
is the *regime* and *break-even*, not false precision.

    .venv/bin/python -m experiments.grug.moe_delay.pp_throughput_model
"""

from dataclasses import dataclass

GIGA = 1e9
TERA = 1e12


@dataclass(frozen=True)
class Hardware:
    """Per-chip v6e (Trillium) numbers; defaults are deliberately conservative."""

    peak_flops: float = 900 * TERA  # bf16 peak per chip (~0.9 PFLOP/s)
    mfu: float = 0.40  # realized fraction of peak in a real training step
    dcn_bw_per_chip: float = 25 * GIGA  # inter-slice (DCN) bytes/s per chip (the cross-slice bottleneck)
    ici_bw_per_chip: float = 800 * GIGA  # intra-slice (ICI) bytes/s per chip (fast)
    # Fraction of the sync-DP cross-slice all-reduce that overlaps compute. The
    # boundary all-reduce after backward is only partly hideable in practice; the
    # exposed remainder is what PP removes. 0=fully exposed, 1=fully hidden.
    dcn_overlap: float = 0.5


@dataclass(frozen=True)
class ModelSpec:
    """A decoder MoE. ``n_active`` drives compute; ``n_total`` drives sync DCN traffic."""

    name: str
    hidden: int
    layers: int
    seq: int
    n_total: float  # all params (incl. experts) -- reduced in sync DP
    n_active: float  # params touched per token -- sets the FLOPs

    def flops_per_token(self) -> float:
        # fwd+bwd training FLOPs ~ 6 * active-params per token.
        return 6.0 * self.n_active


@dataclass(frozen=True)
class Topology:
    """How the job is laid out. ``num_slices`` slices are the DCN-connected units."""

    num_slices: int  # cross-slice (DCN) replicas in sync DP == pipeline stages in PP-over-DCN
    chips_per_slice: int
    global_batch_tokens: float  # tokens per optimizer step (global)
    microbatches: int  # pipeline microbatches (sets the bubble)

    @property
    def num_chips(self) -> int:
        return self.num_slices * self.chips_per_slice


def compute_time(model: ModelSpec, hw: Hardware, topo: Topology) -> float:
    """Per-step compute time (identical for sync and PP -- same arithmetic)."""
    flops = model.flops_per_token() * topo.global_batch_tokens
    return flops / (topo.num_chips * hw.peak_flops * hw.mfu)


def sync_dcn_allreduce_time(model: ModelSpec, hw: Hardware, topo: Topology, grad_bytes: int = 4) -> float:
    """Cross-slice gradient all-reduce time for synchronous DP (volume ~ N_total).

    Ring all-reduce moves ~2 * volume * (G-1)/G; the per-chip DCN egress is the
    bottleneck, so divide the per-slice share by per-chip DCN bandwidth.
    """
    g = topo.num_slices
    if g <= 1:
        return 0.0
    volume_bytes = model.n_total * grad_bytes
    # Per chip, the slice shares the reduction; effective per-chip volume ~ volume/chips_per_slice.
    per_chip = 2.0 * (g - 1) / g * volume_bytes / topo.chips_per_slice
    return per_chip / hw.dcn_bw_per_chip


def pp_activation_comm_time(model: ModelSpec, hw: Hardware, topo: Topology, act_bytes: int = 2) -> float:
    """Stage-boundary activation passing for PP over DCN (volume ~ batch*seq*hidden).

    Each microbatch crosses each of the (P-1) boundaries on fwd and bwd; summed over
    microbatches the per-boundary traffic is the full global batch's activations,
    twice. Boundaries are distinct DCN links, so the per-step critical path is the
    per-boundary time (the (P-1) boundaries run concurrently and overlap compute).
    """
    p = topo.num_slices
    if p <= 1:
        return 0.0
    # tokens per step * hidden * bytes, fwd+bwd, per boundary link.
    per_boundary = 2.0 * topo.global_batch_tokens * model.hidden * act_bytes
    return per_boundary / (topo.chips_per_slice * hw.dcn_bw_per_chip)


def bubble_fraction(topo: Topology, zero_bubble: bool) -> float:
    """1F1B bubble ~ (P-1)/M; Zero-Bubble drives the steady-state bubble ~0."""
    if zero_bubble:
        return 0.02  # residual scheduling overhead
    return (topo.num_slices - 1) / topo.microbatches


def step_times(model: ModelSpec, hw: Hardware, topo: Topology):
    """Per-step wall times (seconds) for sync DP, sync PP (ZB), and async PP."""
    c = compute_time(model, hw, topo)
    sync_dcn = sync_dcn_allreduce_time(model, hw, topo)
    pp_act = pp_activation_comm_time(model, hw, topo)
    # Sync DP: a (1 - dcn_overlap) fraction of the gradient all-reduce is exposed
    # past compute. This is the cost PP's thin activation passing removes.
    sync_dp = c + (1.0 - hw.dcn_overlap) * sync_dcn
    # Sync PP (Zero-Bubble): compute + small bubble + (overlapped) activation comm.
    sync_pp = c * (1.0 + bubble_fraction(topo, zero_bubble=True)) + max(0.0, pp_act - c)
    # Async PP: no bubble, just exposed activation comm (token overhead applied separately).
    async_pp = c + max(0.0, pp_act - c)
    return {
        "compute": c,
        "sync_dp": sync_dp,
        "sync_pp": sync_pp,
        "async_pp": async_pp,
        "sync_dcn_comm": sync_dcn,
        "pp_act_comm": pp_act,
        "comm_compute_ratio": sync_dcn / c if c else float("inf"),
    }


def _fmt_params(n: float) -> str:
    return f"{n/1e9:.1f}B" if n >= 1e9 else f"{n/1e6:.0f}M"


def report(model: ModelSpec, hw: Hardware, topo: Topology, token_overhead_async: float):
    t = step_times(model, hw, topo)
    speedup_sync_pp = t["sync_dp"] / t["sync_pp"]
    raw_async = t["sync_dp"] / t["async_pp"]
    # Async PP progresses token_overhead x slower in tokens; useful-progress speedup
    # = (step-time speedup) / token_overhead.
    net_async = raw_async / token_overhead_async
    print(
        f"== {model.name}  ({topo.num_slices} slices x {topo.chips_per_slice} chips, batch "
        f"{topo.global_batch_tokens/1e6:.1f}M tok, {_fmt_params(model.n_total)} total / "
        f"{_fmt_params(model.n_active)} active) =="
    )
    print(
        f"  compute/step {t['compute']*1e3:.1f} ms | sync DCN all-reduce {t['sync_dcn_comm']*1e3:.1f} ms "
        f"(~N_total) | PP act comm {t['pp_act_comm']*1e3:.1f} ms (~batch*hidden)"
    )
    print(
        f"  DCN comm/compute ratio = {t['comm_compute_ratio']:.2f}  "
        f"({'DCN-bound: PP regime' if t['comm_compute_ratio'] > 1 else 'compute-bound: comm hides, PP moot'})"
    )
    print(
        f"  step: sync-DP {t['sync_dp']*1e3:.1f} | sync-PP(ZB) {t['sync_pp']*1e3:.1f} | "
        f"async-PP {t['async_pp']*1e3:.1f} ms"
    )
    print(f"  throughput vs sync-DP: sync-PP {speedup_sync_pp:.2f}x | async-PP {raw_async:.2f}x raw")
    print(
        f"  async-PP break-even token-overhead = {raw_async:.2f}x; "
        f"at measured {token_overhead_async:.2f}x -> NET {net_async:.2f}x useful "
        f"({'WIN' if net_async > 1.02 else 'wash/los'})"
    )
    print()


# Representative scenarios: the grug testbed scale vs a large MoE, across batch
# sizes (compute-bound at large batch where comm hides, DCN-bound at small batch).
TESTBED = ModelSpec("grug-d512 testbed", hidden=512, layers=6, seq=2048, n_total=290e6, n_active=14e6)
LARGE = ModelSpec("large MoE (~d6144, 48L)", hidden=6144, layers=48, seq=4096, n_total=300e9, n_active=20e9)


if __name__ == "__main__":
    hw = Hardware()
    # Placeholder token-overhead until the iso-loss runs land; override per measurement.
    measured_overhead = 1.3
    print("Pipeline-parallelism throughput vs synchronous DP (v6e, conservative defaults)")
    print(
        f"(dcn_overlap={hw.dcn_overlap}, mfu={hw.mfu}, dcn_bw={hw.dcn_bw_per_chip/1e9:.0f} GB/s/chip, "
        f"measured token-overhead placeholder={measured_overhead}x)\n"
    )
    print("Large MoE over 8 DCN slices, sweeping per-step batch (smaller batch -> comm exposed):")
    for batch in (8e6, 4e6, 2e6, 1e6):
        report(
            LARGE,
            hw,
            Topology(num_slices=8, chips_per_slice=256, global_batch_tokens=batch, microbatches=32),
            measured_overhead,
        )
    print("Single slice (no DCN traffic; PP has nothing to beat):")
    report(
        LARGE,
        hw,
        Topology(num_slices=1, chips_per_slice=256, global_batch_tokens=4e6, microbatches=1),
        measured_overhead,
    )
