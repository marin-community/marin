# Marin EP

Clean-room fused expert-parallel MoE kernel for the grug MoE hero
architecture, targeting GB200 NVL72 (EP64). Plan:
`.agents/projects/20260814_marin_ep_kernel.md`. Behavior spec: `SPEC.md`
(implementations conform to it; disagreements are spec bugs first).

Layout:

- `oracle.py` — dense fp32 reference (value + autodiff grads) and the
  pooled-capacity keep rule; the source of truth for conformance tests.
- `simcore.py` — correctness simulator: per-device programs over an
  abstract message-passing machine (`put` + arrival signaling), explicit
  backward, emits a comm/compute event trace.
- `perfmodel/` — L0 analytic roofline and (later) L1 discrete-event
  simulator consuming simcore traces.
- `kernels/` — real Mosaic/CuTe/FFI kernels as they materialize.
- `tests/` — collected by root pytest; CPU-only unless marked otherwise.
- `bench/` — microbenches and hardware calibration probes.

Run tests:

```bash
uv run pytest experiments/marin_ep/tests -q
```
