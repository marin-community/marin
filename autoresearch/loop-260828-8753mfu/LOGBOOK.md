# 8753-mfu loop logbook

## 2026-08-29 Setup + design review round 1

Campaign branch `research/mcwitt/8753-mfu-loop` from #8753 head 04026e94fb.
Harness copied from loop-260828-mfu24 and adapted (30k restore, <1h
production arms, 20-min step budget). Latest permanent hero checkpoint
verified: step-30000 (manifest.json present).

### Fable review dispositions (blockers/majors all addressed before rack)

- B1 `--restore-from` missing on branch: CONFIRMED — backported the
  launch_diagnostics.py hunk of 76a078bb3a (7db74df3cf); all four hunks
  verified in the right functions; full arm vector parse-tested without
  `--run` (plan builds, load_checkpoint_path + stop 30030 + profiler 30021
  all present in the StepSpec).
- (found by the parse test, missed by both reviewers): `--master-params
  disabled` no longer exists — renamed `device` on this branch. Fixed in
  arm.sh.
- B2 profile start step is absolute: CONFIRMED — arm.sh default now 30021;
  DESIGN documents the trap; H1 folded into H0 draw 2's profile tail.
- M1 coord timeout counts queue time: CONFIRMED from iris reads.py —
  serialization rule added (submit N+1 only after N's watchdog exits);
  QUEUE_CEILING is unreachable under back-to-back queueing, accepted since
  serialization makes it moot.
- M2 resubmit-same-RID hazards: CONFIRMED — fresh RID+VERSION rule in
  DESIGN and arm.sh header.
- M3 engagement verification: ACCEPTED as a campaign-wide rule (user note
  reinforces: wrong conclusions are the top cost). Every treatment
  pre-declares an engagement check; H0 draw 2 profile tail confirms the
  FA4 128x64 kernel name on-rack; score.py now emits peak_gib_max.
- M4 hero-liveness: hero is not currently training; rack-queue check
  before each arm + pause-if-hero-returns rule in DESIGN. No watchdog code.
- M5 noise model: superseded by user note (single rack, no placement
  variance). >=3 control draws mandatory, keep bar max(0.15, 3*sd̂),
  near-threshold results get more draws (use-it-or-lose-it capacity).
- m1 compile-ceiling calibration: timestamp bookkeeping rule added.
- m2 score.py guard holes: loss series + drop/loss point counts emitted;
  pairs compared pointwise.
- m3 submit output swallowed: stderr kept, submit log per RID.
- m4 dirty-tree arms: porcelain guard in arm.sh (loop dir exempt).
- m5 ledger: H5 marked dead-as-flag-arm (forced overlap-limit +
  flag-stripping verified in train.py:238-253); H4 given a pre-declared
  gate + user-sign-off flag; COORD_SKIP_JAX_FLOOR removed (no consumer on
  branch).

### User additions (2026-08-29)

- Production priority allowed, jobs <1h.
- Single rack: no placement variance; use-it-or-lose-it capacity; the
  binding cost is wrong conclusions — reviews focus on inferential
  validity. Folded into DESIGN cost model.
- H6 slop-factor hypothesis added (85 -> 90 ladder, overridable-flag arm,
  peak-HBM engagement check).

### Codex review dispositions (round 1, full transcript in review-codex-design.txt)

- C1 silent scratch fallback (STOP-SHIP): CONFIRMED at train.py:979 (template
  falls back to _init_state). Fixed three-deep: launch_diagnostics now sets
  load_checkpoint=True whenever --restore-from is given (restore raises
  instead of reinitializing; ladder launchers keep optional-resume None on
  purpose — verified before touching hero_recipe, which now just plumbs the
  kwarg); watchdog cancels any run whose first logged step <
  EXPECTED_MIN_FIRST_STEP=30000; score.py fails (exit 1, valid:false) on
  first_step below --expect-min-first-step. Plan dry-run shows
  load_checkpoint: true.
- C2 watchdog ownership/paths: arm.sh now exports IRIS_USER=mwittmann for
  the submission itself; supervisor contract is submit -> immediately start
  watchdog (this loop always does; documented). The "unreviewed harness"
  point is moot as of this round — watchdog was in scope.
- 3 soft deadline + re-arming bug: re-arming FIXED (compile deadline can no
  longer overwrite a started step budget). RPC-blocking overshoot ACCEPTED:
  worst case is bounded by the server-side iris --timeout 3500, which is
  the true hard cap; minute-granularity scanning noted (codex) leaves ~40s
  margin under 1h.
- 4 unverified cancels: FIXED — cancel_arm now polls the train job to a
  terminal state and exits 2 with a do-not-submit marker if it cannot
  confirm.
- 5 RID/VERSION reuse: ENFORCED in arm.sh against arms.tsv (the campaign's
  full submission record); mfl- prefix cannot collide with hero names.
- 6 statistics: ADDRESSED — decision statistic is difference of run means;
  keep needs >=2 treatment + >=2 bracketing control draws and delta >
  max(0.15, 3*sd̂_run); per-step points never treated as independent. The
  placement-variance objection is resolved by the user's statement (single
  rack; the ±2pp figure was multi-rack).
- 7 guard enforcement: FIXED — score.py emits valid/problems and exits
  nonzero on scratch fallback, short/incomplete series, loss_last not at
  window end, or non-finite values.
- 8 profile tail vs early release: CONFIRMED, FIXED — H0 draw 2 runs
  SCORE_MAX_STEP=28.
- 9 ledger tidiness: H4 annotated (in-window measurement = throughput only;
  promotion is the user's call); H2/H3 get concrete treatments + engagement
  checks at draw time, each behind its own review cycle; dead entries stay
  listed as the do-not-rerun record.
- 10 dirty-guard exemption: FIXED — only loop-dir data files
  (.log/.out/.txt/.tsv) are exempt; protocol scripts must be committed.

Both reviews addressed; campaign cleared for rack. Iteration 0 next:
mfl-ctrl-a/b/c serialized control draws, draw b with SCORE_MAX_STEP=28 +
PROFILE_STEPS=3 at 30021.

## H6 arm spec (iteration 1 candidate, pending its own review)

Treatment: `ARM_XLA_FLAGS="--xla_gpu_memory_limit_slop_factor=90"`, everything
else identical to control; env-only (no commit), recorded in arms.tsv extra
column. Mechanism: raises the scheduler/remat temp-arena cap from 113.6 GiB
(85) to ~120.2 GiB (90, at the documented pool-free-space bound); the 85
calibration predates the carry offload (peak now ~116.5 GiB, was ~138) and
train.py's own comment says the smaller arena makes HloRematerialization
recompute more of the step.

Draws: one T at slop90; if delta > bar candidate, a second T draw; controls
bracket from iteration-0 draws plus a fresh control after (the keep rule's
>=2C/>=2T). Ladder: OOM at 90 -> 88; green+gain at 90 -> consider 92 with the
explicit caveat that arena 122.9 GiB exceeds the documented 120.2 bound and
must survive a fresh-mapping against ~17 GiB physical outside the pool.

Engagement check: memory/peak_gib (scored as peak_gib_max) must RISE vs
control. MFU moved but peak did not -> not engaged, inspect task-log XLA_FLAGS
echo before concluding anything. Failure modes: named first-step NCCL alloc
abort or BFC OOM (cheap crash, ladder down).

Confounds pre-declared: (a) remat disengaging when peak crosses the allocator
boundary is the DESIRED mechanism here, not an artifact (#8054's trap inverted)
— but drops and the paired loss series must stay clean; (b) cuda_async
MEM_FRACTION is a RELEASE threshold (#8490): if the higher peak crosses it,
the pool unmaps every step and the arm gets slower with high step-time
variance — watch mfu_stdev, and treat a high-variance slowdown as
threshold-crossing, not as "slop raise is bad".

## H6 review dispositions (fable + codex, 2026-08-29; transcripts: review-codex-h6.txt)

Both reviewers verified the flag path end-to-end (arm.sh -> coordinator ->
XLA_ forwarding -> explicit-wins dedupe in _apply_hero_ep_runtime_defaults ->
exactly one slop token =90 reaches the task; last-occurrence-wins in XLA's
parser if ever duplicated). Both BLOCKED the spec as written. Revised spec:

- ENGAGEMENT REDEFINED (codex blocker 1 / fable F2): slop sets a scheduler
  and remat LIMIT, not an allocation; peak_bytes_in_use can stay flat under
  genuine engagement and can rise for unrelated reasons (rank-0 lifetime
  high-water incl. restore/migration transients). New scheme:
  (1) config engagement — the flag present exactly once in the task env
  (verified by local pre-flight of _apply_hero_ep_runtime_defaults + task
  log); (2) mechanism evidence — compile-time remat/scheduler logs:
  dispatch.py now forwards TF_CPP_MIN_LOG_LEVEL/TF_CPP_VMODULE and every arm
  runs TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=hlo_rematerialization=1, so
  control and treatment logs carry the LHS limit line and remat stats.
  peak_gib demoted to supporting telemetry.
- OUTCOME SPACE PRE-DECLARED (fable F2a): {flag confirmed, remat/LHS logs
  unchanged, peak flat, MFU flat} = DEAD-ON-THIS-STACK (a legitimate
  conclusion, not instrumentation failure). BINDING PRE-CHECK: iteration-0
  control logs are read FIRST — if control shows zero remat and LHS usage
  well under the 113.6 limit, H6 closes as not-binding WITHOUT a treatment
  arm.
- MECHANISM LANGUAGE CORRECTED (fable F4): conclusions say "limit raise"
  (feeds both LHS and HloRematerialization), not specifically "less remat";
  post-offload actual temps (~98 GiB) sit far under both 113.6 and 120.2,
  so an OOM at 90 is UNLIKELY and the 92-rung "exceeds the bound" caveat is
  probably vacuous; any OOM gets diagnosed from the failed allocation size.
- TREATMENT RECORDED (codex 2 / fable F1): arms.tsv gained an xla_flags
  column (backfilled for mfl-ctrl-a), the submit echo prints it, and arm.sh
  refuses a nonempty ARM_XLA_FLAGS without TREATMENT=1 — a leaked env var
  can no longer contaminate a control silently.
- FAILURE CLASSIFICATION (codex 5 / fable F3): any H6-arm failure requires a
  task-log read before classification — BFC/CUDA/NCCL alloc signature ->
  ladder (88); anything else -> invalid/resubmit. A compile-ceiling kill on
  a treatment draw is checked against the control's compile time before
  being read as anything; if the treatment legitimately compiles slower,
  resubmit at interactive priority with a raised ceiling (production stays
  under 1h).
- FIDELITY TOLERANCE (codex 3 / fable F5): DESIGN now calibrates the
  pointwise loss null band and a drops tolerance from iteration-0
  control-control pairs; gate on median + calibrated max; small exceedance
  -> extra draw/investigation, not auto-kill.
- Verified fine by reviewers: no duplicate-flag hazard; per-RID compile
  cache isolates the 85-compiled executable; max_retries_failure=0 means no
  silent in-job retry; overlap-limit stays force-pinned under the
  treatment; MEM_FRACTION release-threshold confound pre-declaration sound.

H6 cleared for rack CONDITIONED on the binding pre-check from iteration-0
control logs (which now carry remat/LHS evidence for free).

## 2026-08-29 ~07:00Z: mfl-ctrl-a crash diagnosis (iteration 0, attempt 1)

The arm burned its whole 3500s iris timeout with all 16 tasks
SchedulingGated: Kueue never admitted the gang. Occupant: muchanem's
mokcamp-08290314 16-node gang (the whole experiment rack, running since
03:14Z); zack's exp177 r87/r90 gangs (5+8 nodes) are also gated ahead of us.
Corrections that came out of this:

- `iris job list` is blind to the real occupancy: the HERO IS LIVE (177 pods,
  38h, its own ~11 racks of the 185-node cluster) and none of it shows in
  job list. Occupancy checks now go through the 08a kubeconfig
  (kueue workloads + pods). The pause-if-hero-returns rule is moot — the
  hero on its own racks is the normal condition (mfu24 ran the same way);
  the campaign contends only for the single leftover rack.
- iris --timeout on the coordinator counts the child's queue wait (OPS.md
  says size for wait+run; the watchdog's queue/compile split already
  handles the accounting). Occupancy after admission is intrinsically
  bounded by the absolute NUM_STEPS stop (~30 steps past restore), so a
  long timeout is safe: ARM_TIMEOUT now defaults 28800, QUEUE_CEILING
  25200, and the <1h-runtime rule is enforced by the watchdog from
  admission.
- The watchdog behaved correctly (compile clock never started, no false
  classification) but "train=running step=-2 for 58 min" should have been
  readable as "gated" — iris reports a gated gang as running. QUEUE_CEILING
  is the working bound; task-level gating detection noted as a possible
  refinement, not required now.

Resubmitted as mfl-ctrl-a2 (VERSION .2, fresh RID per protocol) at 07:0xZ;
expect hours of queue. Iteration count: attempt 1 was a harness crash, not
an experiment — iteration 0 is still in progress.

## 2026-08-29 ~07:50Z: iteration 0 draw 1 + H6 binding pre-check

mfl-ctrl-a2 VALID: mfu_mean 23.3362 (sd 0.115), tokens/s 249358, drops
3.5e-5 mean / 8.5e-5 max, peak_gib 116.57, loss_last 1.26232@30019, window
30005-30019 complete. Queue cleared fast (mokcamp ended); total occupancy
~30 min. Score matches the mfu24 tile-stack numbers (23.32-23.37 at 24k) —
implicit sign the gpu_fa4_cute_wide dispatch engaged; positive kernel-name
confirmation comes from draw b's profile tail.

H6 BINDING PRE-CHECK PASSED from a2's TF_CPP logs (the forwarding worked):
`Rematerialized 325 instructions in module jit_train_step; 1822 net
instructions added` on every rank, with long per-computation remat sweeps.
The slop-85 budget binds. Mechanism-engagement metric for the treatment:
the remat-instruction count at slop=90 vs 325 at 85, plus the LHS/remat
limit values in the logs. H6 is GO after iteration 0 completes.

## 2026-08-29 ~08:20Z: iteration 0 draw 2 (mfl-ctrl-b) + engagement confirmations

mfl-ctrl-b VALID but noisy: mfu_mean 22.872, sd 0.856 (7x draw 1), drops
3.3e-5, peak 116.57, ran to natural completion at 30030. Per-step series
shows 11/15 steps at draw-1 level (23.1-23.5) and 4 isolated stalls
(21.97/21.87/20.55/22.09 at 30006/30010/30012/30014). Only functional
config delta vs a2 is profiler enabled (trace at 30021-23, past the
window). Draw c (plain control) disambiguates profiler-hook vs
environment; per DESIGN, draw b is dropped from the noise estimate if c
matches a2.

FA4 wide-tile engagement CONFIRMED (M3 closed): heuristic.py:109 selects
gpu_fa4_cute_wide for HERO_MODEL; _segmented_kernel_config RAISES rather
than silently downgrading off sm100/head_dim!=128 (its comment: refusing
beats reporting a false null); fa4_cute forward/backward kernels appear in
the draw-b trace; MFU sits on the tile-stack reference. Config-level
fallback is impossible by construction.

Fidelity null band (first C-C pair, a2 vs b, steps 30005-30019):
pointwise |dloss| max 4.1e-5, median 1.6e-5. Draw c adds the second pair.
Trace analysis delegated (ranked lever table incl. remat-cost estimate for
H6, marshal leg check, ragged transport exposure).

## 2026-08-29 ~08:40Z: draw-b trace analysis (lever table refresh)

Merged-interval methodology (overlap artifact avoided); step wall 17.17s avg,
compute stream busy 14.76s (86%), exposed-under-collective 1807ms, pure
starvation 20ms. Ranked levers (ms/step, % of 17168):

1. Remat recompute ~1470 (8.6%) — all cheap elementwise/concat fusions
   (*_remat*), stable across steps, zero GEMM/attention remat. H6's target;
   headroom bound confirmed real.
2. Exposed ragged a2a 1133 (6.6%) — 2021 total, 56% exposed, 12 chunk ops x
   48 launches; transport is SM-bound (prior campaign) so HIDE, don't speed.
3. Loop-carry D2D copies on the compute stream 785 (4.6%) — nine copy.7xx
   ops at 1.85-3.22 GB x 48 launches; buffer donation/aliasing candidate.
   NEW -> H9.
4. Exposed layer-carry offload copies 576 (3.4%) — D2H 316 + H2D 260 while
   compute idles; prefetch-earlier/writeback-later scheduling. NEW -> H10.
5. Straggler all-gather.101.1: +1.2s in 1 of 3 profiled steps — cross-node
   skew at a sync point. Same phenomenon as draw b's 4 in-window stall
   steps (environment, not profiler). Informs noise model: stall steps are
   real campaign noise; runs differ mainly by how many stalls they catch.
6. H2 marshal leg: GONE (zero marshal events, 20ms starvation). H2 DEAD on
   this stack — the dk-hero-era prior no longer applies.
7. Dense GEMM mass 8.3s busy (48%) — only kernel-efficiency/quantization
   moves it; excluded by fidelity constraint.

Optimistic ceiling from this trace: ~28.5% MFU if levers 1-5 fully
reclaimed. Near-term stack: H6 (remat) + H10 (offload prefetch), both
compute-stream-freeing without touching transport.

## 2026-08-29 ~09:10Z: H9/H10 mechanism diagnosis + treatment specs (agent investigation)

H9 copies IDENTIFIED by exact size match: the ragged-transport latent-dim
working buffers, not the layer carry — [TK=524288, 3072] bf16 = 3.221 GB
(sorted_x/returned) and [C=301466, 3072] bf16 = 1.852 GB (per-chunk
dispatch), 48 launches = layers, 9 ops = 3 copy sites x 3 loop bodies.
Root cause: ragged-all-to-all is in-place (result aliases the output-init
operand); the loop-invariant-hoisted zeros inits and the returned
double-use force CopyInsertion to copy every iteration.

- T-H9-1 (arm next): ARM_XLA_FLAGS --xla_gpu_async_copy_min_bytes=1000000000
  -> GpuCopyAsyncWrapper turns the nine >=1GB D2D copies into
  copy-start/done on their own stream under the ASYNC-COMPUTE resource
  (limit 2), NOT the pinned collective overlap limit (verified separate
  branches in gpu_latency_hiding_scheduler.cc). Predicted +0.4-0.9 MFU.
  Engagement: profile tail — copy.7xx leaves the compute stream,
  compute-stream MemcpyD2D busy -> ~0. Watch peak_gib and the
  remat-count line (longer live ranges can eat H6's budget).
- T-H10-1 (after H9-T1, separate arm): --xla_gpu_enable_pipelined_host_offloading=true
  -> CollectivePipeliner rotates the carry writeback into iteration i+1
  and the reload one iteration ahead (verified pass wiring at
  gpu_compiler.cc:1213-1290; predicates match the jax scan-offload
  pattern). Predicted +0.5-0.8 MFU. CORRECTNESS-SENSITIVE: changes reload
  timing in the same machinery as the #8317 offload race (overlap limit
  stays force-pinned at 1 — arms cannot override it). Hard gate: scored
  loss series within the C-C null band (max ~1e-4); any excursion = the
  race, not noise. Engagement: exposed offload-copy time -> ~0 in a
  profile tail, peak +~1.6 GB.
- DO NOT STACK the two in one arm (shared async-compute limit 2; joint
  attribution lost). If both keep, back-ablate per the re-ablation rule;
  --xla_gpu_experimental_parallel_async_compute_limit=4 is a stacking
  refinement only.

## H9/H10 review dispositions (fable + codex, transcripts: review-codex-h9h10.txt)

Verdicts: H9 conditionally runnable; H10 STOP-SHIPPED pending stronger
correctness gates. Actions taken now:

- score.py emits mfu_median + stall_steps as the primary statistic (F3:
  protocol drift — the mean field was the only emitted statistic while
  DESIGN had switched to medians; draw-b gap was 0.37, >2x the keep bar).
- arm.sh TF_CPP_VMODULE extended (uniformly, all arms) with
  execution_stream_assignment=1 (H9 engagement: #async_start_instructions
  count, control vs treatment — also counts UNINTENDED wrapped copies,
  since GpuCopyAsyncWrapper is a global >=1GB transformation, codex-3) and
  collective_pipeliner=1 (H10 engagement: "Pipelinable while"/"Transforming
  DUS" lines). Profile-free engagement is primary; one tail draw per
  treatment remains for runtime stream placement.
- DESIGN stale decision rules rewritten (codex-2): no single-pair keeps,
  guard excursions within ~2x band trigger draws/investigation not kills,
  attribution freeze across kept-stack changes, task-log read before
  crash classification.
- arm.sh refuses submissions with JAX_OPTIMIZATION_LEVEL set (codex-6:
  this fork enables pipelined host offloading at O1+ regardless of flag).

H9 OUTCOME TREE (pre-declared, F4/F5/codex-4):
- Engagement first: #async_start_instructions delta = +9 expected; 0 = 
  predicate no-op (no dead call); >>9 = collateral wraps (attribute before
  scoring).
- Null + remat-count UP vs control's 325 = BUDGET-BLOCKED on slop-85, not
  dead; retest only if the slop stack ever changes.
- Null + remat flat: profile tail must confirm copies left the compute
  stream AND offload-copy exposure did not worsen (H9 shares the limit-2
  async-compute resource with the offload copies; starving them mimics a
  null). Only then engaged-null-dead.
- Crash/OOM = negative memory-budget interaction, not dead-lever;
  consider --xla_gpu_experimental_parallel_async_compute_limit=4 as an
  H9-alone refinement arm.
- Positive: >=2 T draws vs frozen slop-85 controls, median bar 0.16; if
  keep, COMMIT the flag into train.py flag_defaults before any further
  arm (F6: env-only keeps break the control machinery — the TREATMENT
  guard would reject a kept-flag control).

H10 REVISED GATES (F1/F2 + codex-1; still stop-shipped until implemented):
- The overlap-limit pin is NOT a safety argument (offload copies ride the
  UNPINNED async-compute resource; the #8317 race was never root-caused;
  pipelining shifts reload timing a whole iteration). Spec language fixed.
- Required before running: a small committed observability change logging
  per-rank nonfinite/grad-norm (or rank-local loss terms) — the global
  loss psum dilutes a one-rank error ~64x below the null band; gate on max
  rank delta. Loss comparison from step 30001 (the #8317 signature step is
  OUTSIDE the +5..+19 window), both draws to natural completion (30 
  points), plus a T-T pointwise comparison against the C-C band
  (timing races show as T-T excess; bitwise T-T equality is NOT expected).
- An H10 keep cannot be race-bounded by this protocol: pre-declared as
  needing user sign-off + a longer soak before any hero promotion.

## 2026-08-29 ~11:00Z: iteration 2 decision — H9 DISCARD (engaged-negative, attributed)

Treatment medians 23.215/23.128 vs controls 23.321/23.240/23.340: -0.13
consistent. Trace comparison (clean steps both sides; the only straggler
step was in the CONTROL trace and was excluded):

- The flag did exactly what it promised: all nine multi-GB copies left the
  compute stream (compute-stream D2D 785 -> 32 ms/step), landing on the
  offload memcpy streams 75-78.
- The time was re-lost three ways: (1) +461 ms of kernel SLOWDOWN from
  contention — nvjet GEMMs at identical launch counts run 1.6x slower
  under the now-co-running copy traffic, and GEMMs scheduled into the
  window the on-stream copy used to occupy co-run with the SM-bound
  ragged-a2a at 3.3x slower; (2) +201 ms exposed copy-done waits + +160-194
  ms displaced offload overlap (the async copies serialized behind the
  carry-offload copies on the same streams — exactly review finding F4);
  (3) ragged-a2a exposure +145 (compute finishes its copy-free work sooner
  and stalls at the transport). Net +150 ms/step wall.
- The remat 325->268 "improvement" was relabeling (-155 remat kernels,
  +177 other fusions) — same for H6's remat drop, retrospectively suspect.

STRATEGIC INSIGHT (now 2-for-2 with H6): the slop-85 LHS schedule is a
defended local optimum. The trace's "reclaimable" legs are guarded by
contention: the on-stream copies were free overlap-fillers occupying the
compute stream precisely where the SM-bound a2a would poison co-running
GEMMs. Any lever that frees compute-stream time adjacent to the a2a window
feeds GEMMs into 3.3x contention. Corollaries: H3-naive (overlap GEMMs
with a2a) is COUNTER-INDICATED, upgraded from "needs treatment" to
near-dead; levers must either reduce the a2a's SM footprint (explored,
dead per prior campaigns) or fill its window with copy-engine work (what
control already partially does). H10's temporal rotation is the remaining
distinct mechanism; its risk is now understood as "exposure moves around
under scheduling perturbation", not just the correctness race.

H10 rank-local gate disposition (codex "required", RESOLVED AS REFUTED in
part): a whole-carry corruption on one rank moves GLOBAL loss by >=2e-3
(20x the null band — caught by the existing series gate); a structural
boundary race fires deterministically every step, so 30 points of
systematic shift are detectable globally at ~2e-5; only intermittent-AND-
tiny corruption stays invisible, which no 30-step protocol bounds with or
without rank-local channels — that residual is covered by the adopted
rule that an H10 keep needs user sign-off + a soak before hero promotion.
A graph-changing instrumentation commit would invalidate the iteration-0
calibration for less coverage than that rule already provides.

## 2026-08-29 ~12:25Z: iteration 3 — H10 OOM-blocked; H10b rescue spec

Both H10 draws died in the first jit_train_step execution with the same
NCCL alltoall CUDA OOM ('Cuda failure 2'), after the pipeliner engaged
(Pipelinable while: while.251.clone on all ranks). Deterministic, not
fragmentation. Memory anatomy (train.py:76-83): fraction 0.75 = 138.2 GiB
pool threshold, ~46 GiB outside, ~28.5 GiB held by NCCL/cuBLAS/context —
the pipelined program's extra first-execution demand exceeds the ~17.5 GiB
slack.

H10b rescue spec (pending fable+codex review): XLA_PYTHON_CLIENT_MEM_FRACTION
0.75 -> 0.72 frees +5.5 GiB outside the pool (pool 132.7 still >> peak
116.6); arms = 2x T(pipelined @0.72) vs 2x C(plain @0.72) isolating the
pipelining, plus the C@0.72-vs-C@0.75 bridge telling the fraction's own
cost. MEM_FRACTION plumbed as an explicit arm.sh knob recorded in
arms.tsv. All H10 fidelity gates carry over (loss from +1, natural
completion, T-T comparison, keep needs user sign-off + soak).

## 2026-08-29 ~13:00Z: H10b CANCELLED pre-rack (review + free log evidence)

Fable review blocker F1 (mine the dead runs first) paid off at zero rack
cost: the OOM is inside ncclCuMemAlloc — NCCL's cuMem/window registration
path (MNNVL registers collective operand buffers; tensor-scale, GB per
instance) — with no allocation size logged; the pipelined program shows
+21 async-start instructions (296 vs control 275) and remat restructured
to 261/745-net (vs 325/1822). If peeling duplicated a2a operand buffers,
the extra registration demand is plausibly >> the 5.5 GiB a 0.75->0.72
fraction change frees. F2: the fraction also feeds the slop-derived
scheduler budget, so BOTH sides of an @0.72 comparison would run a
different program — the bridge is non-neutral by mechanism and H6's
calibration would not transfer. F3: cuda_async grows the pool past the
threshold for in-pool demand, and a T peak in (132.7, ~134] would sit on
the #8490 churn cliff and misscore as "pipelining is slow".

DECISION: H10/H10b closed as OOM-BLOCKED (engaged, unmeasurable at any
defensible memory posture on this stack) — explicitly NOT disproven.
Revisit conditions: a compile-only memory estimate for the pipelined
program, or an XLA-side fix avoiding peeled-instance re-registration.
Priority note from review: arms stay at production per the user's
explicit 2026-08-29 exception (<1h runtime), superseding the older
interactive-only instruction.

Iterations 1-3 pattern for the record: three engaged flag levers, all
negative or blocked — the slop-85/overlap-1/on-stream-copy schedule is a
defended local optimum. Campaign proceeds to the structural fork
(marin_ep transport port scoping in flight; XLA-patch lever as the
alternative).

## 2026-08-29 ~13:15Z: structural fork decided — marin_ep DECLINED, T-H9-2 next

marin_ep scoping (agent, full evidence in transcript): cannot run at EP64
on the pinned c9526e wheel (kMaxPeers=32, deterministic step-0
CUDA_ERROR_ILLEGAL_ADDRESS, openxla/xla#47283); the fix is a one-constant
fork rebuild (~3.5h) plus an 11-file merge with one semantic conflict
(train.py runtime defaults rewritten on both sides); and the arm is a
POSTURE PACKAGE — marin_ep's validated posture (LHS + overlap 4, no carry
offload) conflicts with the #8753 keeps, so it answers "mok posture vs
campaign posture". Projected numbers: plain-8684 at 30k was 22.31, mep105's
gap over its own era control was +0.53 -> ~22.8 vs today's 23.3 stack; and
its headline drops advantage (0.19% vs 3.06%) is void against today's
ragged transport (3e-5 at the restored router). DECLINED: negative
expected value for this campaign. Revisit only if the transport posture
itself becomes the target (own project).

Iteration 4 = T-H9-2: delete the nine init copies at the source instead of
moving them (H9 proved moving them re-loses the time to contention; true
deletion removes ~785 ms of copy work minus the ~300 ms contention
giveback the schedule may re-take). Implementation being drafted; dual
review before rack per protocol.

## 2026-08-29 ~14:45Z: T-H9-2 codex review — NO-GO parsed

Codex P1: the return-path transpose mask miscomputes when an expert group
is EMPTY (duplicate unclipped offsets -> .at[offsets].set(1) collapses ->
cotangents leak through overwritten rows; concrete 2-shard counterexample
given). CRITICALLY: bit-for-bit inherited from jax 0.11.1's own
_ragged_all_to_all_transpose (parallel.py:1705) — the wrapper replicated
it faithfully, so control and treatment share it and the arm introduces
ZERO divergence. At hero scale the trigger is latent (mean ~2048
tokens/group at the trained router; P(empty) ~ e^-2048), but it is a real
upstream JAX bug to report separately (empty groups are realistic at
small scale / early training / skewed routers). Disposition: does not
block the arm (bit-parity is the point); flagged for an upstream report +
a marin issue after the campaign beat. P2 (custom_vjp kills JVP): nothing
in the training path jvp's this code (verified train.py:761
value_and_grad); acknowledged limitation, documented in-code.
Codex found NO other divergence: operand/offset order matches, output_t
dead, every tie nonempty+nonnegative, offload_carry re-trace keeps the
wrappers. Awaiting the fable review before GO.

## 2026-08-29 ~16:30Z: iteration 4 (T-H9-2) — engagement complete, keep pending bracket

Draws: mfl-h92-a median 23.680 (0 stalls, sd 0.127 — cleanest run of the
campaign), mfl-h92-b median 23.779 (4 stalls). Controls 23.240-23.384.
Median delta +0.36/+0.44 — 2x the 0.18 bar, consistent.

Engagement trio all green (the anti-H9 signature):
- compute-stream MemcpyD2D 785 -> ~4 ms/step (draw-b trace);
- NO relocation: the only big memcpys anywhere are the pre-existing
  layer-carry offload pinned<->device staging (10.87 GB copy-start.4x/9x
  on streams 76/77, present in control) — zero multi-GB D2D on any
  stream;
- #async_start_instructions 275, UNCHANGED from control (H9's failure
  showed +10 here).
Peak 113.5 (-3.1 GiB: the hoisted const buffers are gone); remat
restructured 275/1379-net. Fidelity: pointwise loss vs ctrl-d over 24
steps max 9.1e-5 / median 1.2e-5 (C-C levels); drops identical 3.3e-5.

Bracketing control mfl-ctrl-e submitted from the pre-change tree
(REPO-override worktree at 5324007f19). KEEP decision on its return.

## 2026-08-29 ~16:50Z: ITERATION 4 — T-H9-2 KEPT (+0.41)

mfl-ctrl-e (pre-change tree) median 23.337 confirms the bracket. Treatment
23.680/23.779 vs controls 23.240/23.321/23.337/23.340/23.384: +0.41
median, ~8x sd̂. First keep of the campaign; stack now ~23.73. The
mechanism lesson generalizes: the defended-equilibrium levers (moving
work) all lost; DELETING work won. Hot-swap safe for the live hero
(code-only, checkpoint-compatible, loss series at C-C level, drops
identical) — hero adoption is the user's call; PR prep queued as a
campaign deliverable.

Follow-ups now live: (a) new-stack lever table from xprof-h92 (analysis
delegated); (b) H10 revisit at 0.75 ON THE NEW STACK — the deleted
hoisted consts freed ~5 GB resident, which may clear the ncclCuMemAlloc
OOM that blocked pipelined host offloading (cheap feasibility probe);
(c) upstream JAX report for the inherited empty-group transpose-mask bug.

## 2026-08-29 ~17:20Z: new-stack lever table (from xprof-h92)

Cross-validation: wall 16,837 -> 16,565 ms/step (clean pairs) = -272 ms =
+0.38 MFU predicted vs +0.41 scored. Mechanism refinement: the deletion
actually removed 564 of the 785 ms (two 1.85 GB sites survive: copy.746/
copy.748, 48/step each, ~137 ms — T-H9-2b diagnosis delegated); the wall
gain came via BETTER A2A HIDING (exposed 1,133 -> 820) paid partly by
+550 ms of GEMM SM-contention at matched kernel counts (nvjet 256x192
1,440 -> 1,906 us/kernel under the now-overlapping a2a). Remat "drop" was
renaming again (fusion pool flat at ~3,450 ms).

New top levers: (1) exposed a2a 820-858 ms — but each hidden ms costs
~0.6 ms GEMM slowdown at current dk CTA footprint; (2) exposed offload
copies ~645 ms — the 10.87 GB device->pinned D2H staging bursts are the
exposed half; H10's target, now ~+0.95 MFU potential, copy-engine-based
so NO contention tax (probe mfl-h10n-a on the rack now); (3) NEW: cap the
a2a dk SM footprint (CTA cap/priority) to reclaim the +550 ms contention
without re-exposing the a2a; (4) T-H9-2b residual copies ~137 ms.

## 2026-08-29 ~18:40Z: lever #3 investigation — a2a dk SM footprint (H11)

Mechanism fully identified (agent, all file:line at the pinned fork rev):
DeviceKernelCtaCount = 8 x core_count (ragged_all_to_all_thunk.h:198-201,
min 8, threads 128) -> 1184 CTAs on GB200 = 8 resident CTAs x 148 SMs,
64 regs/thread via __launch_bounds__(128,8) = the ENTIRE sm_100 register
file, held for the whole op INCLUDING barrier spins. This is the +550
ms/step GEMM contention. No env var, no debug option reaches the grid
(exhaustively enumerated); TF_CPP_VMODULE=ragged_all_to_all_thunk=3
prints the live cta_count.

H11 = one-header patch: env-gated multiplier XLA_RAGGED_A2A_DK_CTAS_PER_SM
(1..8, DEFAULT 8 = today's behavior) in DeviceKernelCtaCount; all
consumers (launch, barrier/signal registration) route through the one
function so consistency is automatic; kernel is grid-size-agnostic
(balanced ranges + strided GIN loop, regular launch, no cooperative
assumptions). Default-identity means the rebuilt wheel is bit-comparable
for every existing arm — only env-carrying arms diverge. BDP estimate:
4/SM marginal-OK for the copy phase; the barrier-spin share shrinks
linearly at zero bandwidth cost. Requires one jax-cuda13-pjrt rebuild
(~3.5h) on the marin-community/xla fork + release + pin swap.
No-rebuild alternatives all dominated: one-shot and NCCL-fallback paths
lose ~0.5 MFU on prior same-night data (host marshalling per chunk-op);
stream-priority flip risks re-exposing the a2a; MPS/green contexts
inapplicable.

GATE: pushing a branch/release to marin-community/xla is an
external-repo write -> per standing instruction, needs user approval.
Patch + review will be prepared so the build can start on a word.

## 2026-08-29 ~21:30Z: ideation slate (agent, full text in transcript)

Ranked candidates, deletion-weighted: C1 offload byte titration (offload
only a layer subset; deletes copy bytes on the contention-free channel;
+0.35-0.45 expected; small code knob, default=all); C2 collective byte
audit (fp32-AG-before-cast / loop-invariant AG hoists; free phase 1 from
existing dumps); C3 optimizer-step anatomy (never decomposed; replicated
Newton-Schulz would be a +0.3-class bitwise deletion); C4 verify the salt
actually consumed copy.746/748 (needs a tail on a salted arm); C5
targeted save-policy (slop90's mechanism decoupled from the budget knob;
highest reschedule risk); C6 dispatch-chain Pallas fusion (~390ms of
round trips); C7 short_conv fusion (~near-bar); C8 host-side/pure-gap
audit rider; C9 symmetric-memory NCCL as a headroom enabler only.
Sequencing: free diagnostics now (C2/C3 from disk, C4 tail on next arm),
C1 as the first new arm after H11's identity control, C5 after the
headroom split settles. All memory-budget candidates pre-declare the
#8490 churn signature.

## H11 arm plan (iteration 6, pending dual review)

Wheel: jax_cuda13_pjrt-0.11.1+marin.ce6db0d2c555 (adhoc branch
mcwitt/adhoc-ragged-dk-cta-cap = pinned c9526e8c0272 + the one-header
env-gated CTA cap; built by the fork CI as artifact only, staged at
s3://marin-us-east-02a/marin/research/mcwitt-mfuloop/pjrt-h11-cta-cap/).
Consumption: --pjrt-wheel sideload (reinstalls exactly jax-cuda13-pjrt in
the worker env; verify_ragged_pjrt accepts the +marin. version).

Sequence, all on the salted tree, one arm at a time:
1. mfl-h11-id — IDENTITY CONTROL: sideloaded wheel, env UNSET. The patch
   defaults to multiplier 8 = today's grid, so this must land in the
   reference family (23.645-23.779, median band). A miss means the wheel
   differs beyond the patch (build drift) — stop and diagnose, run no
   sweep.
2. mfl-h11-c4 — TREATMENT: XLA_RAGGED_A2A_DK_CTAS_PER_SM=4 (592 CTAs,
   4/SM). Predicted: a2a copy phase lengthens ~0-20%, barrier-spin
   footprint halves, GEMM contention tax shrinks; net positive if the
   contention recovery beats the exposure growth.
3. mfl-h11-c2 — TREATMENT: =2 (296 CTAs) if c4 is non-negative; expected
   past the bandwidth knee (BDP says ~1.2MB in flight, likely undershoots).
4. Confirmation draw at the best rung + bracketing salted control.

Env plumbing: XLA_ prefix forwards to tasks (verified path); the knob must
be identical on all ranks (uniform arm env guarantees it). Engagement:
TF_CPP_VMODULE gains ragged_all_to_all_thunk=3 on all arms — the thunk
VLOGs the live cta_count, so every arm's grid is directly observable
(1184 default / 592 / 296). Metric side: exposed-a2a and GEMM per-kernel
times from a profile tail on the best rung tell WHERE any delta came from.
Fidelity: the kernel moves identical bytes; grid size does not touch
numerics (same put targets, same barrier semantics); loss series gate as
always. Risks: skew sensitivity — fewer CTAs spin longer per barrier at
arrival skew (watch stall counts); a c4 regression with exposed-a2a
GROWTH means bandwidth-bound (close H11 honestly); #8490-style variance
rules pre-declared.

## 2026-08-29 ~22:40Z: free diagnostics — C2 closed, C3 promoted to top lever

C2 (collective bytes): NO-ARM at zero cost. Grad reduce-scatter already
bf16; only f32 collectives are per-layer SCALAR pmax/pmin/psum latency ops
(~48ms); no loop-invariant payloads. Closed.

C3 (optimizer anatomy): SMOKING GUN. The optimizer tail is ~784 ms/step
(4.8%): three NS chains x 5 iterations of FULL-STACK GEMMs (nvjet grid
110,592 CTAs = all 48 layers x 6144x9216 on EVERY device) + fp32 momentum
for the three dense groups (exactly 10,871,635,968 B each — these are the
"10.87 GB offload staging bursts", i.e. the exposed-offload leg is largely
OPTIMIZER STATE round-trips, ~65 GB/step, not the layer carry) + embedding
moments 3.15 GB x2. A sharded 3D path EXISTS in grugmuon_hero.py
(_newtonschulz_padded_stack_sharded over the intra-rack axes) but the
trace shows no reshard collectives in the tail — the silent mesh.empty /
no-axes fallback is the suspect (optimizer update possibly traced outside
the abstract-mesh context). If the LIVE HERO takes the same fallback,
fixing it is +0.6-0.8 MFU at hero scale AND deletes most of the exposed
offload leg (C1's target collapses into C3). Root-cause agent running.
C8 rider: empty (no hoistable fa4 precompute).

## 2026-08-30 ~00:20Z: codex H11 review — NO-GO parsed, all P0s fixed

- P0-1 sideload suppressed uv sync (worker would have NO venv — the
  mok-era semantics differed; on this iris a setup_scripts list replaces
  the default wholesale, and the bundle excludes .venv): FIXED by
  switching to the pip route — GrugRunConfig.worker_pip_packages ->
  dispatch pip_packages -> create_environment; the default setup keeps uv
  sync and installs the wheel AFTER it. pjrt_sideload.py deleted.
  --pjrt-wheel now takes a fetchable URL (presigned S3 works; 7-day link
  generated via rclone link).
- P0-2 loaded-binary proof: custody chain documented — uv's task-log
  line '+ jax-cuda13-pjrt==0.11.1+marin.ce6db0d2c555' proves the exact
  version installed into IRIS_VENV after sync; codex itself verified
  nothing installs after; scoring requires grepping that line per arm.
- P0-3 knob not plumbed: arm.sh now passes -e XLA_RAGGED_A2A_DK_CTAS_PER_SM
  (empty default = identity), guards it behind TREATMENT=1, and records a
  dk_ctas column in arms.tsv.
- P1-4 mismatch hangs -> compile-ceiling kill: accepted, classification
  noted in the plan.
- P1-5 identity weaker than claimed (2-file branch diff, unpinned
  toolchain, vacuous delta grep): accepted with the codex identity-miss
  protocol — a low identity draw freezes the sweep and triggers a second
  identity draw + a contemporaneous stock-wheel salted control; wheel
  sha256 recorded (from the CI output).
- P1-6 pooling: adopted — the keep rule compares WITHIN the new wheel:
  interleaved id-a, c4-a, id-b, c4-b; median(c4) - median(id) >
  max(0.17, 3 x contemporaneous sd); pre-salt draws inform noise only.
- P1-7 thunk=3 VMODULE too chatty (~12k lines/step across ranks): DROPPED
  from scored arms. Engagement: xplane kernel grid dims from the profile
  tail on ONE treatment draw (592 vs 1184), plus the custody line; the
  identity arm needs no cta line (identical by design).

## Fable H11 review dispositions (round complete; both reviewers addressed)

F1/F4 = codex P0-1/P0-3, already fixed (pip route; DK_CTAS plumb). F2:
run-phase custody added — verify_ragged_pjrt now logs the exact installed
pjrt version at startup; per-arm scoring greps BOTH the uv install line
('+ jax-cuda13-pjrt==0.11.1+marin.ce6db0d2c555', after 'syncing deps')
AND the runtime line. F3 identity gate recalibrated: hard-stop only below
mean-3sigma (~23.54); gray zone [23.54, 23.645) -> second identity draw
(the min-max band alone false-alarms 40% at n=1). F5 adopted (= codex
P1-6): c4 scored against wheel-matched identity draws (2 of each,
interleaved); pre-salt draws are a sanity band only; the step-4
confirmation prices the deployable delta vs a fresh no-wheel salted
control. F6 REVIEWER DISAGREEMENT RESOLVED: codex estimated ~12k
lines/step and said drop; fable counted the engaged-path VLOG sites
(~12/step/process; fallback sites early-return) and says keep — and the
cta_count line is the only guard against an invalid knob silently
running at 8 (F4b). Fable's count is the more specific evidence:
ragged_all_to_all_thunk=3 ADDED to the uniform VMODULE for all H11-era
arms; identity tests wheel+VMODULE jointly (both expected null; a
VMODULE-only salted draw splits them if identity misses). F7 noted:
ctrl-f/g remain valid non-wheel references (the plumbing commits are
inert with empty worker_pip_packages).

H11 GO. Sequence: id-a, c4-a, id-b, c4-b (interleaved, all new wheel,
salted tree, uniform VMODULE); c4 keep = median(c4 pair) - median(id
pair) > max(0.17, 3 x contemporaneous sd); then best-rung confirmation +
fresh no-wheel salted control; c2 only after a genuine c4 keep-class
result.

## 2026-08-30 ~01:40Z: C3 RETRACTED — the "replicated NS" was an analysis artifact

Root-cause agent (three independent verifications: static, CPU repro with
the exact optimizer build, re-analysis of BOTH the diagnostic and the
11-rack hero profiles): Newton-Schulz sharding is NOT broken anywhere.
The two misreads in the earlier diagnostic: (1) the 110,592-CTA GEMMs are
the SHARDED expert NS inside shard_map (local operand bf16[288,3072,3072]
= 48 layers x 6 LOCAL experts; 288x24x16 tiles = 110,592), scope
grugmuon_hero.py:204 — each device does 1/64 of the work as designed;
(2) 10,871,635,968 B matches BOTH 48x6144x9216x4 AND 48x6x3072x3072x4 —
a byte-size coincidence; the buffers are the per-device fp32 momentum
SHARDS of the three expert leaves under offload_opt_state=True, already
1/64-sharded. The dense 3D padded-stack sharded path is also engaged
(line-325 pads, line-326 reshard a2as visible in-trace). The live hero
profiles show identical anatomy. The earlier "smoking gun" logbook entry
is WRONG and superseded by this one. Cost of the error: zero arms (the
falsification ran before any treatment) — the review-first protocol
working as intended.

Genuine residue -> C3' (opt-state offload titration): the tail's real
copy cost is the offload feature itself (~65 GB/step momentum round-trips
= ~450-530 ms partially overlapped + embed Adam ~106 ms). Keeping ALL
three expert momentum shards on device (+32.6 GB) does NOT fit (peak
112.75 + 32.6 > 138.2 threshold); keeping ONE (+10.9 GB -> peak ~123.6)
fits comfortably and deletes ~1/3 of the traffic (~+0.2 MFU class); TWO
(+21.7 -> ~134.5) sits 3.7 GiB from the churn cliff. Treatment shape:
per-leaf offload selection in initial_state (code knob, default =
current all-offload); checkpoint-compat question (memory-kind change on
restore templates) must be answered in review. Latent footgun for the
eventual PR: an optimizer harness calling update() outside set_mesh
silently retraces onto the replicated fallback — hardening = derive the
mesh from the param sharding or assert non-empty.
