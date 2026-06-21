# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-device op order for the zero-bubble pipeline backward.

The zero-bubble pipeline runs a single SPMD program on every stage (see
``pipeline.py``): a device with stage id ``sid`` processes microbatch
``m = t - sid`` at timestep ``t``, for ``t`` in ``range(T)`` with
``T = M + S - 1`` (``M`` microbatches, ``S`` stages). Every device issues the
same ``T`` forward (F), ``T`` input-gradient (B) and ``T`` weight-gradient (W)
ops; the warmup/cooldown diagonal is handled by masking, not by a per-stage op
count, so the schedule is one ordered list reused on every device.

Dependencies that the order must respect:

- **F** is a forward sweep: ``F(t)`` ships its activation to the next stage,
  consumed by ``F(t)`` there at the following wavefront step, so a device runs
  ``F(0), F(1), ..., F(T-1)`` in time order.
- **B** is the backward critical path: ``B(t)`` produces the input cotangent
  ``dx`` and ppermutes it upstream, where the neighbour consumes it as its
  ``B`` input. A device therefore runs ``B(T-1), B(T-2), ..., B(0)`` in reverse
  time, and each ``B`` blocks on the cross-stage ppermute from the stage below.
- **W** depends only on ``dy_by_t[t]`` (produced by ``B(t)``) and a forward vjp
  closure. It has no cross-stage and no cross-time dependency, so ``W(t)`` may
  be issued at any point after ``B(t)``, and any interleaving of W into the B
  stream is numerically identical (the grad accumulation is associative).

Schedule (ZB-H1 / Qi et al. "Zero Bubble Pipeline Parallelism", steady-state
1B1W): emit the forward sweep, then interleave one W after each B, offset by one
slot so ``W(t)`` rides in the device-idle window of the *next* ``B`` while that
``B`` waits on its upstream ppermute:

    F(0) ... F(T-1)                          # forward sweep, time order
    B(T-1)                                    # warmup: no W ready yet
    B(T-2) W(T-1)  B(T-3) W(T-2) ...          # steady 1B1W: W fills the B gap
    B(0)   W(1)
    W(0)                                      # cooldown drain

The single warmup B (``B(T-1)``) and single cooldown W (``W(0)``) are the only
slots without a partner, so on the per-device timeline the B-critical-path
ppermute stalls are covered by W everywhere except the very first B.

Bubble fraction (unit costs F = B = W = 1). Useful work per device is ``3 M``
slots (the ``S - 1`` invalid diagonal slots do no real work but, being masked,
cost the same; the cost model below counts only the wavefront makespan of the
critical-path device). The B critical path is a length-``T`` reverse wavefront
whose first ``S - 1`` steps drain the pipeline before any W can pack, so the
makespan of the schedule is ``F_sweep + B_path + 1`` cooldown ``= T + T + 1``
B/W slots after the shared forward, i.e. the bubble is the ``S - 1`` warmup
diagonal of each of the F and B sweeps that cannot be filled:

    bubble_fraction(S, M) = 2 * (S - 1) / (3 * M + 2 * (S - 1))

The W sweep contributes no bubble of its own — every W but the last is packed
into a B stall — so this halves the naive three-phase bubble
``3 * (S - 1) / (3 * M + 3 * (S - 1))`` (where the deferred W sweep adds its own
``S - 1`` drain). See ``bubble_fraction`` and the table in its docstring.
"""

from __future__ import annotations

import enum


class Op(enum.Enum):
    """A pipeline backend op keyed by timestep in the per-device schedule."""

    F = enum.auto()  # forward: stage_fn + vjp closures for one wavefront slot
    B = enum.auto()  # input-gradient: dx upstream (critical path) + head/embed grad
    W = enum.auto()  # weight-gradient: deferred replay through vjp_p


def forward_order(num_stages: int, num_microbatches: int) -> list[int]:
    """Timesteps for the forward sweep, in execution (time) order."""
    timesteps = num_microbatches + num_stages - 1
    return list(range(timesteps))


def backward_order(num_stages: int, num_microbatches: int) -> list[tuple[Op, int]]:
    """Interleaved (B, W) op order for one device's backward.

    Returns ``(op, t)`` pairs. B runs in reverse time (the ppermute critical
    path); each W is slotted one step after its B so it fills the next B's
    cross-stage stall. ``W(t)`` always follows ``B(t)``, so dy/vjp_p
    dependencies hold for every interleaving this produces.
    """
    timesteps = num_microbatches + num_stages - 1
    order: list[tuple[Op, int]] = []
    b_times = list(reversed(range(timesteps)))  # T-1, T-2, ..., 0
    for i, t in enumerate(b_times):
        order.append((Op.B, t))
        if i > 0:
            # Pair this B with the W of the previously-issued (one-larger) B,
            # so W rides in this B's ppermute stall.
            order.append((Op.W, b_times[i - 1]))
    order.append((Op.W, b_times[-1]))  # cooldown: drain the last W
    return order


def bubble_fraction(num_stages: int, num_microbatches: int) -> float:
    """Theoretical bubble fraction of the interleaved schedule (unit F=B=W=1).

    The 1B1W fill packs every W but one into a B-critical-path stall, so only
    the ``S - 1`` warmup diagonals of the F and B sweeps remain as bubble:

        bubble = 2 * (S - 1) / (3 * M + 2 * (S - 1))

    Reference (current three-phase backend, W deferred as its own sweep, so its
    ``S - 1`` drain is an extra unfilled bubble):

        bubble_three_phase = 3 * (S - 1) / (3 * M + 3 * (S - 1))

    Table (S in {2, 4, 8}; rows at M=2S and M=4S):

        S    M     new      three-phase
        2    4     0.143    0.200
        2    8     0.077    0.111
        4    8     0.200    0.273
        4    16    0.111    0.158
        8    16    0.226    0.304
        8    32    0.127    0.179
    """
    s, m = num_stages, num_microbatches
    return 2 * (s - 1) / (3 * m + 2 * (s - 1))
