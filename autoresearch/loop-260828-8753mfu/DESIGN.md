# Campaign: ragged EP hero MFU loop (loop-260828-8753mfu)

Goal: raise hero MFU beyond the #8753 stack's frontier without fidelity loss
(no added quantization; loss trajectory equivalent or better) and without
breaking hero-checkpoint compatibility (every keep must hot-swap into the live
hero run; checkpoint migration OK, restore from hero checkpoints must work).

Cost model (user, 2026-08-29): rack capacity is use-it-or-lose-it — failed or
extra arms are cheap (stay courteous to other rack users). The expensive
failure is a WRONG CONCLUSION: being sent in a bad direction, or prematurely
closing a promising one, because a flawed experiment misled the loop. Review
cycles and arm design optimize for inferential validity first, rack frugality
second. Concretely: every treatment arm pre-declares an ENGAGEMENT CHECK
(evidence the lever actually engaged — a kernel name in a profile, an echoed
flag, an HLO property, or a metric that cannot move unless the lever did);
"no effect" results are only recorded as dead after engagement is confirmed;
near-threshold results get more draws rather than a coin-flip call.

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
  `--master-params device` (this branch renamed the master-less mode
  `disabled` -> `device`; in-process migration reads a master-bearing
  checkpoint's fp32 master directly into device params), schedule-steps
  4470000, no checkpoints/evals. Profiler off on scored arms EXCEPT an
  allowed profile tail: `PROFILE_STEPS` few, `PROFILE_START_STEP` ABSOLUTE
  (e.g. 30021) so it lands past the scoring window — profiler start_step is
  a global step; a relative value like 5 silently never fires on a restore.
- `JAX_COMPILATION_CACHE_DIR` rotated per run id (clique-init deadlock dodge)
  → every arm cold-compiles ~9.5 min; pre-phase (env sync + restore +
  migration) ~19 min cold, less when the object cache is warm.
- Priority: production allowed (user 2026-08-28) with job runtime capped
  under 1 hour: iris `--timeout 3500`. Watchdog: COMPILE_CEILING=1800 from
  first W&B-run existence, STEP_BUDGET=1200 (20-min post-compile cap, user),
  early release at +19 steps, cancel on preemption/failure/coord death;
  QUEUE_CEILING bounds pending time (queueing doesn't count against the 1h).
- SERIALIZATION: one arm on the rack at a time, and arm N+1 is submitted only
  after arm N's watchdog exits — the coordinator's iris `--timeout 3500`
  clock starts when the coordinator starts (it schedules instantly and waits
  on its child), so an arm queued behind another arm burns its 1h budget in
  the queue and gets killed just as its train job starts.
- RESUBMISSION: an invalid arm (watchdog kill, preemption, incomplete
  window) is resubmitted with a FRESH RID and VERSION, never reused — a
  reused RID merges W&B histories (silent window mixing), reuses the
  leader-populated compile cache (the documented clique-init deadlock
  recipe), and can vacuously reuse the artifact layer.
- Hero liveness: the live hero is NOT currently training (step-30000 is the
  newest permanent checkpoint; no hero job in iris). Before each arm, check
  the rack queue; if the hero ever reappears on this cluster, pause the
  campaign and reassess priorities rather than contending with it.
- Rack etiquette: queue arms back-to-back while work is pending; do not
  submit anything if the loop is blocked >1h on analysis.
- Timestamp bookkeeping (calibration): for each arm record W&B-run-created
  -> first-step wall time. The compile ceiling starts at run creation, which
  is BEFORE restore + master migration + compile; a ceiling kill on a cold
  arm is not evidence against the arm until these timings say the ceiling
  actually fits the work.

## Metric and guards

- Metric: mean `throughput/mfu` over scored steps +5..+19 relative to the
  first logged step (`score.py --relative`). Higher is better.
- Guards (from the same run): `moe/drop_fraction` max/mean not worse than the
  paired control; `train/loss` at the window's last step equivalent to
  control (bf16-noise band; 24k arms showed 1.29964-65 across four arms);
  `window_complete` must be true.
- Noise model: only ONE NVL72 rack exists, so there is no placement variance
  (user-supplied fact, 2026-08-29; overrides the ±2pp cross-night figure,
  which came from multi-rack clusters) — residual noise is neighbor jobs,
  clocks/thermals, and run-to-run scheduling jitter. The decision statistic
  is the DIFFERENCE OF RUN-MEAN MFU, with run-level sd estimated from the
  >=3 iteration-0 control draws and refreshed by later controls; per-step
  points within a run are autocorrelated and are never treated as
  independent replicates. KEEP requires >=2 treatment draws and >=2
  same-session control draws (bracketing), delta > max(0.15 MFU,
  3 x sd̂_run); a single C/T pair never keeps. Near-threshold deltas get
  additional draws instead of a call. Periodic re-control draws guard
  against slow drift across the campaign.
- Fidelity gate for keeps: paired-window comparison of the FULL 15-step loss
  series (data is deterministic at a fixed restore step so series pair
  pointwise). The tolerance is CALIBRATED, not assumed: the iteration-0
  control-vs-control pointwise deltas define the null band for this exact
  30k protocol (the 24k figures are a different protocol). Gate on the
  median pointwise delta plus the calibrated max (median-not-max
  discriminates bf16 reduction noise from a real bug — prior campaign
  lesson); drops get a tolerance from the same control pairs rather than a
  raw "not worse" coin flip. A small exceedance triggers another paired
  draw or a numerical investigation, NOT an automatic kill; non-finite
  loss, materially changed drops, or a deviation beyond the control
  envelope is a hard fail. Any keep that changes numerics beyond
  scheduling/layout gets flagged and needs explicit reasoning about why
  the trajectory is unchanged.

## Iteration accounting

Per user: only hypotheses that reach the rack count toward the 24-iteration
floor; review-killed ideas do not. Campaign runs 24 rack iterations or 24 h,
whichever comes LAST. Iteration 0 = 30k-protocol baseline (control draw(s)).

## Hypothesis ledger (priors from prior campaigns; refresh with a new profile)

Excluded by constraint: fp8 dispatch wire #7665, MXFP8 (quantization);
anything breaking hero restore.

- H0 baseline+validation: >=3 control draws at 30k to establish the metric
  and size noise. Draw 2 carries a profile tail (PROFILE_STEPS=3,
  PROFILE_START_STEP=30021, past the scoring window) that (a) positively
  confirms the `gpu_fa4_cute_wide` 128x64 kernel engaged on-rack — its
  first rack outing, and cross-night MFU deltas cannot confirm dispatch —
  and (b) refreshes the lever-sourcing profile (H1 folded in: no separate
  profile arm). Draw 2 runs SCORE_MAX_STEP=28: the default early release at
  +19 would cancel at 30019, before the 30021 trace (codex finding). If the
  tail perturbs draw 2's scored window relative to draws 1/3, drop draw 2
  from the noise estimate.
- H1 profile refresh: FOLDED INTO H0 draw 2 (see above); prior profile
  (dk-hero, pre-offload) showed marshal 2.55 s as top lever and a dense-MLP
  leg that turned out to be architecture (2 shared experts + remat) — both
  priors need re-confirmation on the offload/LHS/tile stack.
- H2 marshal-step levers (SO-H2): reduce the ~2.55 s marshal leg — kernel or
  layout changes in the dispatch/marshal path. Needs H1 to confirm the leg
  survived the offload/LHS stack.
- H3 shared-expert/dense leg scheduling: overlap the shared-expert compute
  with ragged transport more aggressively; profile-informed.
- H4 capacity-factor titration at the restored router: hero drops at cf 1.15
  are ~5e-5 — far under any budget; cf 1.10/1.05 shrinks transport+GEMM
  buffers. GATE (review finding): a cf keep changes training dynamics and
  its drop cost varies with data over the whole run — a 15-step window
  cannot bound it. Pre-declared acceptance: drop_fraction_max <= 1e-3 in
  the window AND loss series within the control band; even then a cf keep
  is flagged for explicit user sign-off before hero promotion, and the
  prior loop note (2026-08-12: CF moves are frontier-walking) is quoted in
  the decision. NOTE (codex): the 15-step window cannot establish whole-run
  loss equivalence for a dynamics-touching change; H4's in-window
  measurement establishes the THROUGHPUT effect only, and any promotion
  decision belongs to the user.
- H5 overlap-limit re-titration: DEAD AS A FLAG ARM on this base —
  `offload_carry` force-sets overlap-limit 1 non-overridably and
  `_apply_hero_ep_runtime_defaults` silently strips conflicting
  ARM_XLA_FLAGS (a flag-based treatment would run as an exact control).
  Testing overlap>1 means changing the remat mode, which is a different
  hypothesis (and the documented silent-corruption recipe with LHS +
  offload — see activation-offload-lhs-overlap-race).
- H6 slop-factor raise (user suggestion 2026-08-29, HIGH priority — cheap,
  code-supported mechanism): `--xla_gpu_memory_limit_slop_factor=85` in
  train.py flag_defaults was calibrated for the pre-offload posture (default
  95 sized the temp arena at 125.7 GiB > the 120.2 GiB the pool can serve;
  85 -> 113.6 GiB), and the comment itself says a smaller arena makes
  HloRematerialization recompute more of the step. The carry offload since
  dropped peak from ~138 to ~116.5 GiB, so the remat pressure the arena cap
  creates may now be pure waste. The flag IS overridable via inherited
  XLA_FLAGS (verified: only overlap-limit and ragged-required flags are
  forced), so treatment = ARM_XLA_FLAGS with slop 90 (arena ~120.2, at the
  documented bound; ladder to 88 on OOM, 92/95 on green+gain). Fidelity:
  remat changes recompute scheduling only. Engagement check: peak-HBM
  metric must RISE vs control (less remat holds more live); an unchanged
  peak means the flag did not engage regardless of the MFU reading. Failure
  mode is a named first-step NCCL allocation abort — cheap crash, ladder
  down.
- H7 attn-offload re-add: 23.14 but 3x stdev and the host allocation lands as
  one contiguous 72.56 GiB block regardless of HOST_MEM_LIMIT_GB — ledger
  says dead unless pod memory grows.
- H8 marin_ep_mgpu_fused transport on this stack: mok-goal showed 22.83 vs
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
