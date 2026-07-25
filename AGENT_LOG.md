# AGENT_LOG — ep25-d3 round 6 (R6-2: global-QB controller probes)

Append-only. All times UTC. Branch: agent/ep25-d3-qbprobes (from agent/ep25-d4-pipelined @1650246c5).
Prior-round logs live on their own branches (fa4lse, token-chunk, TE).

Mission: probe whether a global-controller variant (damped gain g<1 or DeepSeek-V3-style
integral accumulation) drives QB-on cf1.0 steady-state drops under 3% at <=0.5pp MFU cost
with loss parity. Baselines (d4, 350-step g=1): drop series 0.885(5)/0.271(60)/0.175(119)/
0.089(250)/0.064(349), tail-100 mean 7.3%, p50 MFU 22.002%, loss 3.335@350 (tail20 3.3434).
120-step draw: 22.595%, drops 0.083@119. g=2 diverges (limit cycle, drops 0.67+).

## Check-in 2026-07-25 20:55 UTC — setup; arm 1 (g=0.5) submitted

- Read ROUND6_BRIEF.md; branch created from agent/ep25-d4-pipelined (custom adjoint,
  SCALE_REPORT_DROPS emission fix, SCALE_CAPACITY_FACTOR, SCALE_QB_GAIN all present).
- QB code read: `_compute_qb_beta` (model.py:392) = per-step equalizing quantile, pmean
  over batch axes; `_apply_qb_betas` (train.py:261) sets bias = -center(beta), full
  replacement = implicit proportional controller at gain 1. SCALE_QB_GAIN blending
  (train.py:409) gives g<1 damping for free: pending <- g*beta + (1-g)*pending.
- Reference env recovered verbatim from iris job_config provenance of
  /mwittmann/ep25d4-qb-cf100-drops-350-v1-20260725 (incl. SCALE_A2A_CUSTOM_ADJOINT=1,
  SCALE_MOE_QB=1, cf default 1.0, 350 steps, checkpoints disabled).
- Fleet check 20:50 UTC: only peer SMOKES running (d1 batch-smoke-ep4, d4 sqb-smoke-ep4,
  rav hybridep smoke) — no EP25 rack job in flight; slot free.
- ARM 1 SUBMITTED: /mwittmann/ep25d3-qbg05-cf100-350-v1-20260725 — g=0.5 damped gain
  (SCALE_QB_GAIN=0.5, stock code path otherwise byte-identical), cf1.0, adjoint, drops,
  350 steps, DISABLE_CHECKPOINT, operating point. ETA ~22:30 UTC (setup+compile+350x~12.3s).
  Job mutations this session: submissions only.
- Falsifiable read: if the ~6% plateau is a mild proportional limit cycle, g=0.5 halves
  the correction rate and should settle lower; if it is sender-local bucket hotspots
  (invisible to any global bias), the series matches the g=1 baseline within draw variance.

Confidence: 4/10 that g=0.5 reaches <3% steady-state drops
Next: build arm 2 (SCALE_QB_INTEGRAL) + CPU test while arm 1 runs; babysit arm 1.
