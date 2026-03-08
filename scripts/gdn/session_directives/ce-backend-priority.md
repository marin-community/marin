# Session Directive: CE Backward First

Current diagnosis:
- Iteration 67 was the real regime change: forcing TPU fused CE to `pallas_tpu` removed the giant CE/XLA false wall and improved MFU materially.
- Iteration 68 then proved that reducing GDN train-path budget is no longer sufficient by itself; the step can still regress.
- The remaining CE-attributed `while` is now much smaller than before, but still large enough to matter, and the most plausible remaining CE opportunity is the backward/custom-VJP shell.

Implications for this session:
- Hold CE implementation fixed at `pallas_tpu` unless the point of the run is explicit CE implementation A/B.
- Treat CE backward mode as the first experiment axis:
  - `pallas`
  - `xla_streaming`
- Do not spend mainline budget on new GDN-local work until the CE backward A/B has been run or refreshed under the current head.

Required behavior:
1. Record `CE backend selected: <impl>` in every profiled iteration writeup.
2. Record `CE bwd mode: pallas | xla_streaming` in every profiled iteration writeup.
3. If available, record `CE-attributed while: <before> ms -> <after> ms`.
4. If CE implementation or CE backward mode changed, treat the iteration as `Change class: CE backend`.
5. If CE is not the axis under test, keep CE fixed and explain why another direction is justified now.

Preferred experiment matrix:
- current deployable head + `pallas_tpu` CE + `pallas` CE backward
- current deployable head + `pallas_tpu` CE + `xla_streaming` CE backward
- optional sanity run: explicit `xla` CE when bottleneck attribution is unclear

Goal:
- Determine whether the residual CE control cost is best attacked by backward-mode selection before resuming major GDN structural work.
