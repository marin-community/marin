# Coupon-clipping pyramid arms

These entry points test whether redistributing a fixed shared-expert parameter
budget toward a few layers changes early knowledge acquisition. All arms use
the same 48-layer Grug MoE, four scan segments `(4, 18, 4, 22)`, Datamix token
order, 6.711B-token horizon, optimizer schedule, and 16-GB200 topology.

- `c0_p0.py`: uniform shared-expert width 1280 in every layer.
- `p1.py`: shared-expert width 4096 in layers 0–3 and 1024 elsewhere.
- `p2.py`: shared-expert width 4096 in layers 22–25 and 1024 elsewhere.
- `pilot_c0_p0.py`, `pilot_c0_p0_low.py`, and `pilot_c0_p0_high.py`: 128-update
  uniform controls at the 0.55676x, 0.5x, and 0.625x learning-rate candidates.
- `pilot_p1.py` and `pilot_p2.py`: 128-update pyramid throughput checks at the
  center learning rate.
- `pilot_l1.py`: the one-layer source bounded to 128 updates for the throughput gate.
- `pilot_growth.py`: 32 one-layer updates, an L1-to-L48 transform, and 16 grown updates.
- `pilot_growth_target.py`: target-only recovery from an explicit completed source checkpoint root.
- `d1.py`: the token-matched D1 pipeline, with L1 through update 4,480 and L48
  through update 6,400.
- `pilot_aggressive_source.py`: the 128-update `d1536/L1` throughput gate.
- `pilot_aggressive_growth.py`: 32 `d1536/L1` updates followed by 16 updates on
  the widened `d3072/L48` target.
- `wd1.py`: the 95/5 narrow-shallow to wide-deep arm, with all 320 wide/deep
  updates covering the terminal decay.
- `c_short.py`: the 3,200-update full-depth WSD control.
- `paloma_wd1.py`, `paloma_c_short.py`, and `paloma_c0.py`: checkpoint-only
  Paloma evaluations for the main treatment and controls. Each subset is capped
  at eight batches of 64 sequences and the result is written to `metrics.json`.

The production arms use the 0.625x candidate selected by the 128-update control
gate. The low, center, and high pilots remain explicit entry points so the gate
is reproducible.

The launcher sets `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` before dispatch.
The Grug dispatcher propagates `XLA_*` variables to every child task.

Paloma runs as a separate artifact after its checkpoint dependency completes.
It loads only model parameters and never enters the optimization loop, avoiding
the in-run evaluation state mutation tracked in issue #7712. Run these entry
points sequentially on the same four-node slice when comparing allocation time.
