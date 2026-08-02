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
- `pilot_extreme_source.py`: a `d768/L1` throughput gate that keeps 64 experts,
  top-4 routing, and 1536-wide selected experts. It tests the 10x systems target
  without depending on cold-start routing across additional experts.
- `pilot_extreme_growth.py`: 32 WD2 source updates followed by 16 updates on an
  exactly widened `d3072/L48` target.
- `pilot_aggressive_growth.py`: 32 `d1536/L1` updates followed by 16 updates on
  the widened `d3072/L48` target.
- `wd1.py`: the 95/5 narrow-shallow to wide-deep arm, with all 320 wide/deep
  updates covering the terminal decay.
- `wd2.py`: the 90/10 `d768/L1` to `d3072/L48` arm. Its target retains
  1536-wide routed/shared experts and a 24:8 query/KV-head ratio so every
  widened axis grows by exactly four and the transition preserves the source
  function.
- `pilot_l4_source.py` and `pilot_random_layer_dropout_source.py`: matched
  128-update throughput gates for a physical `d1536/L4` source and a
  `d1536/L48` source that stores all layers but executes four uniformly sampled
  layers per update. The latter freezes inactive parameter and optimizer-state
  slices.
- `pilot_l4_growth.py` and `pilot_random_layer_dropout_growth.py`: 32 source
  updates followed by 16 target updates, covering the corresponding
  width-and-depth or width-only transition.
- `l4.py` and `random_layer_dropout.py`: matched 80/20 arms. Both execute four
  narrow layers through update 5,120 and full `d3072/L48` through update 6,400;
  the former inserts 44 identity layers, while the latter trains four randomly
  sampled positions from an existing 48-layer stack.
- `c_short.py`: the 3,200-update full-depth WSD control.
- `paloma_wd1.py`, `paloma_wd2.py`, `paloma_l4.py`,
  `paloma_random_layer_dropout.py`, `paloma_c_short.py`, and `paloma_c0.py`:
  checkpoint-only Paloma evaluations for the treatments and controls. Each
  subset is capped at eight batches of 64 sequences and the result is written
  to `metrics.json`.

The production arms use the 0.625x candidate selected by the 128-update control
gate. The low, center, and high pilots remain explicit entry points so the gate
is reproducible.

The launcher sets `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` before dispatch.
The Grug dispatcher propagates `XLA_*` variables to every child task.

Paloma runs as a separate artifact after its checkpoint dependency completes.
It loads only model parameters and never enters the optimization loop, avoiding
the in-run evaluation state mutation tracked in issue #7712. Run these entry
points sequentially on the same four-node slice when comparing allocation time.
