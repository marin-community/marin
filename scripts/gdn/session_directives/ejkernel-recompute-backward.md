# Session Directive: ejkernel-Style Minimal-Tape Backward Control Arm

Goal:
- Validate the most promising takeaway from `ejkernel` / `EasyDeL`: save less tape, recompute more in backward, and keep the train path chunk-first and simple.

Reference files:
- `~/Projects/Work/Marin/ejkernel/ejkernel/modules/operations/gated_delta_rule.py`
- `~/Projects/Work/Marin/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_fwd.py`
- `~/Projects/Work/Marin/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule/_pallas_impl_bwd.py`
- `~/Projects/Work/Marin/EasyDeL/easydel/operations/kernels/gated_delta_rule.py`

What to borrow:
- the backward tradeoff, not the whole stack:
  - smaller saved residual contract,
  - recompute-heavy backward from raw inputs plus saved chunk-start state,
  - simpler chunk-level execution surface.

What not to over-interpret:
- multi-backend wrapper logic,
- quoted inference speedups as if they proved the TPU training path,
- packaging/autotune machinery unless it directly affects the train-step critical path.

Required design constraints for this arm:
1. Keep CE fixed unless the run is explicitly a CE comparison:
   - `LEVANTER_FUSED_CE_IMPLEMENTATION=pallas_tpu`
   - one explicit CE backward mode
2. Treat this as `Change class: outer control structure` unless the run is explicitly just a probe.
3. Reduce backward tape first:
   - prefer eliminating saved `v_pseudo`, `k_cumdecay`, `solve_transform`, or equivalent per-chunk tapes when recomputation is feasible,
   - keep only raw inputs plus the minimum chunk-start state needed to make backward tractable.
4. Recompute forward prepare intermediates in backward instead of hauling them through residuals.
5. Prefer a plain chunked experimental path before adding or preserving a `segment_size` hierarchy.
6. If the arm changes geometry, try chunk sizes `{32, 64}` before assuming `128` is still optimal.

Evaluation requirements:
- Compare against the current deployable head under the same CE settings.
- Record whether:
  - scanned residual state shrank,
  - `while_ms` changed,
  - `remainder_budget_ms` changed,
  - step time changed.
- If train-path budget drops but step time does not, classify it as `off-critical-path` / `overlap-loss`.

Success criterion:
- The arm is only promising if the smaller tape + backward recompute tradeoff improves end-to-end train-step time or reduces the measured remainder/control burden enough to justify more work.
