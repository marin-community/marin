# EP25 round-6 brief (2026-07-25 ~18:00Z)

Goal: **>=25% p50 MFU at the operating point WITH full fidelity** — QB-on, dropped-assignment
fraction < 3% at steady state, loss parity. Honest starting points: QB-on cf1.0 + adjoint =
22.60% p50 (~6% steady drops); strict-3% config = QB + cf1.15 = 20.85%.

## Measurement protocol (NEW, supersedes round 1-5 habits)

- ALL new runs: `SCALE_MOE_QB=1` + `SCALE_REPORT_DROPS=1` + custom adjoint
  (`SCALE_A2A_GATHER_DISPATCH=1 SCALE_A2A_CUSTOM_ADJOINT=1`), cf1.0 unless the direction
  says otherwise. Report p50/p10/p90 MFU, the step-indexed drop series, and loss tail.
- MFU is comparable ONLY within matched drop regimes (heavy-drop runs read higher — dropped
  assignments gather the zero pad row). Never cite a cross-regime delta as a win.
- Reference numbers: QB-on cf1.0 adjoint 120-step = 22.595% (drops 0.083@119) and the
  350-step draw = 22.002% (drops 0.064@349, tail-100 mean 7.3%); QB-on cf1.15 = 20.848%
  (0.037@119). QB draw variance is large — matched back-to-back legs, and >=2 draws before
  claiming < 0.5pp effects.
- Fleet rules unchanged: one rack job in flight per agent; smoke first; no job mutations on
  jobs you didn't submit; no rack kills without coordinator approval; log every mutation;
  AGENT_LOG.md check-ins every ~15 min with Confidence n/10; commit locally; NEVER push or
  write to GitHub (reads fine). Setup-flake failures ([iris setup] dep-sync exit before any
  task): just resubmit with -vN suffix.

## Directions this round

- **R6-1 (d1): leg-batching x QB-on composition.** Expected ~+1.4pp (22.6 -> ~24.0).
- **R6-2 (d3): QB controller probes — damped gain g<1 and DeepSeek-style integral rule.**
  Prize = making cf1.0 compliant (<3%) so we keep 22.6 instead of paying cf1.15's -1.75pp.
- **R6-3 (d4): sender-local balancing.** The kernel-level fix aimed at the localized cause
  (sender-local bucket hotspots at 64x256 granularity).
- **R6-4 (d2): MXFP8 expert GEMMs at the operating point.** Speed + short-horizon fidelity;
  the 66B-token quality gate (+0.11-0.21% held-out loss, #7271) is flagged, not resolved here.

Milestone context: marin#7279 comment 5080435482 (full round 1-5 report) and marin#7201
comment 5080459722 (QB-off flag). Do not post or reference these publicly yourselves.

## R6-5 (added, user-suggested): manual-PGLE x prefetch-gate probe — UNASSIGNED, goes to
the first agent that closes its current direction. Auto-PGLE/LHS are sealed
null-to-negative on this stack (#7012), and the prefetch reorder alone was an exact null
(scheduler-gated). The re-probe case: post-adjoint the profile is comm-heavy (SendRecv
22.4% top op), and MANUAL PGLE (capture fbs profile from a representative QB-on run, feed
back via --xla_gpu_pgle_profile_file_or_directory_path + LHS on recompile) was never
tested — nor was it tested WITH the committed prefetch gate (agent/ep25-d4-pipelined,
SCALE_A2A_PREFETCH=1) supplying the legal dataflow freedom. Probe: one profile capture +
one 120-step QB-on cf1.0 adjoint+prefetch+PGLE leg vs same-draw control. Crisp
falsification; low-to-medium prior.

## Policy amendment (user, 2026-07-25): attempts vs closures
Only MEASURED results close a direction (a matched A/B negative, a falsified mechanism).
Operational difficulties — toolchain, env, build, infra flakes — never close a direction:
after ~3 focused attempts write a scoping CHECKPOINT (what blocks, what was tried, options)
and continue or escalate to the coordinator; do not convert operational friction into a
scoping negative. Escalation paths: replicate a known-green environment verbatim, hand the
env fight to a peer, or surface to the human.
