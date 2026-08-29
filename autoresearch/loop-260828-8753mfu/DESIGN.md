# Campaign: ragged EP hero MFU loop (loop-260828-8753mfu)

Goal: raise hero MFU beyond the #8753 stack's frontier without fidelity loss
(no added quantization; loss trajectory equivalent or better) and without
breaking hero-checkpoint compatibility (every keep must hot-swap into the live
hero run; checkpoint migration OK, restore from hero checkpoints must work).

## Base

- Branch `research/mcwitt/8753-mfu-loop` @ 04026e94fb = PR #8753 head
  (`mcwitt/8317-hero-step-optims`, based on xla-fork/#8684).
- Contents: offload_carry remat + latency-hiding scheduler (forced
  overlap-limit 1), bitcast pair-interleave, FA4 128x64 wide forward tile as
  `gpu_fa4_cute_wide` (landed by implementation name; dispatch has passed unit
  tests but has NOT yet run on a rack — the iteration-0 control validates it).
- PJRT wheel: `jax_cuda13_pjrt-0.11.1+marin.c9526e8c0272` pinned in
  pyproject/uv.lock (GitHub release, marin-xla-pjrt-20260828). No wheel
  overlay in arms.
- Reference numbers: 24k-restore same-night pair (2026-08-28): control (#8753
  stack sans tile) 23.06/23.05, +tile 23.32/23.37 MFU. 30k-restore cross-ref:
  plain-#8684 gate 22.31 median. Protocols are not comparable; this campaign
  re-baselines at 30k.

## Fixed protocol (every arm)

- One NVL72 rack, cw-us-east-08a, `experiments.grug.moe_hero_ep.launch_diagnostics`,
  one arm on the rack at a time.
- Restore from the latest hero permanent checkpoint
  `s3://marin-us-east-02a/marin/grug/hero-12d8b6f0-dee637/2026.08.19.2/checkpoints/step-30000`
  (verified complete: manifest.json present). `--num-steps 30030` (ABSOLUTE
  stop step: checkpoint step + 30; `--num-steps 30` exits vacuously right
  after restore).
- Mixture data (trained-router routing), cf 1.15, `ragged_all_to_all`,
  `--master-params disabled` (in-process migration on this base),
  schedule-steps 4470000, no profiler/checkpoints/evals on scored arms.
- `JAX_COMPILATION_CACHE_DIR` rotated per run id (clique-init deadlock dodge)
  → every arm cold-compiles ~9.5 min; pre-phase (env sync + restore +
  migration) ~19 min cold, less when the object cache is warm.
- Priority: production allowed (user 2026-08-28) with job runtime capped
  under 1 hour: iris `--timeout 3500`. Watchdog: COMPILE_CEILING=1800 from
  first W&B-run existence, STEP_BUDGET=1200 (20-min post-compile cap, user),
  early release at +19 steps, cancel on preemption/failure/coord death;
  QUEUE_CEILING bounds pending time (queueing doesn't count against the 1h).
- Rack etiquette: hold/queue arms back-to-back while work is pending; do not
  submit anything if the loop is blocked >1h on analysis.

## Metric and guards

- Metric: mean `throughput/mfu` over scored steps +5..+19 relative to the
  first logged step (`score.py --relative`). Higher is better.
- Guards (from the same run): `moe/drop_fraction` max/mean not worse than the
  paired control; `train/loss` at the window's last step equivalent to
  control (bf16-noise band; 24k arms showed 1.29964-65 across four arms);
  `window_complete` must be true.
- Noise model: placement variance across nights is ±2pp; KEEP decisions
  require same-night interleaved control/treatment pairs (minimum 2 arms per
  decision, prefer control-treatment-control-treatment when the delta is
  small). A single-draw delta <0.15 MFU is treated as noise.
- Fidelity gate for keeps: interleaved A/B window as above; any keep that
  changes numerics beyond scheduling/layout gets flagged and needs the loss
  guard plus explicit reasoning about why the trajectory is unchanged.

## Iteration accounting

Per user: only hypotheses that reach the rack count toward the 24-iteration
floor; review-killed ideas do not. Campaign runs 24 rack iterations or 24 h,
whichever comes LAST. Iteration 0 = 30k-protocol baseline (control draw(s)).

## Hypothesis ledger (priors from prior campaigns; refresh with a new profile)

Excluded by constraint: fp8 dispatch wire #7665, MXFP8 (quantization);
anything breaking hero restore.

- H0 baseline+validation: control draw at 30k; validates `gpu_fa4_cute_wide`
  dispatch on-rack and establishes the campaign metric. 2 draws if time
  allows to size same-night noise.
- H1 profile refresh: one UNscored profile arm (xprof, few steps) on the
  baseline stack at 30k restore to source new levers; prior profile
  (dk-hero, pre-offload) showed marshal 2.55 s as top lever and a dense-MLP
  leg that turned out to be architecture (2 shared experts + remat).
- H2 marshal-step levers (SO-H2): reduce the ~2.55 s marshal leg — kernel or
  layout changes in the dispatch/marshal path. Needs H1 to confirm the leg
  survived the offload/LHS stack.
- H3 shared-expert/dense leg scheduling: overlap the shared-expert compute
  with ragged transport more aggressively; profile-informed.
- H4 capacity-factor titration at the restored router: hero drops at cf 1.15
  are ~5e-5 — far under any budget; cf 1.10/1.05 shrinks transport+GEMM
  buffers. Caveat: a prior loop note (2026-08-12, dev-scale) barred CF arms
  as frontier-walking; at the hero operating point with measured drops this
  may be a real trade. Review explicitly.
- H5 overlap-limit re-titration under carry offload (overlap 4 + LHS was
  +0.07 and -10 GiB at the 8733 protocol — likely sub-noise; only if H1
  shows exposed collectives).
- H6 attn-offload re-add: 23.14 but 3x stdev and the host allocation lands as
  one contiguous 72.56 GiB block regardless of HOST_MEM_LIMIT_GB — ledger
  says dead unless pod memory grows.
- H7 marin_ep_mgpu_fused transport on this stack: mok-goal showed 22.83 vs
  22.30 control at 18k restore with 16x fewer drops, but needs the kmax128
  wheel (conflicts with the pinned c9526e wheel) and the backend isn't on
  this branch. Big integration; park unless transport dominates H1 profile
  and cheaper levers dry up.
- Dead-lever ledger (do not re-run without new evidence): PGLE (null),
  command buffers at hero shape (+7.5 GiB, negative), FA4 backward tile
  (hard-locked 64x64), 128x128 forward tile (2.5x regression), fused shared
  experts (~0), small-param FSDP (negative), unchaining per-chunk a2a
  buffers (-0.13), NVLS/SHARP env (null), fa4-lse primal output (moot: XLA
  already device-saves FA4 out/lse).

## Decision rules

- keep: same-night pair shows treatment > control by >0.15 MFU with guards
  clean → commit stays, becomes the new control for later iterations.
- discard: delta <= noise or guards fail → revert commit.
- crash/invalid: watchdog kill, preemption, incomplete window → arm is
  invalid, does not consume the hypothesis; resubmit once before discarding.
- Every keep is re-verified implicitly by later same-night controls (the
  control arm always runs the current kept stack).

## Review protocol

Before rack time each experiment's design gets adversarial review from (a) a
fable agent and (b) `codex exec` CLI, both against the pinned worktree
(`codex exec` needs stdin closed; `--base` incompatible with a custom
prompt). Findings addressed or refuted in writing in the logbook before
submission. Reviews of iteration N+1 overlap with rack time of iteration N.
