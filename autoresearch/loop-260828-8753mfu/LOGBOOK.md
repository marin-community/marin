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

## 2026-08-30 ~06:20Z: H11 attribution (CTAS=1 trace vs kept-stack trace)

Grid confirmed in kernel_details: 1216 -> 152 CTAs, same block/regs/
launch count. Ledger (ms/step, clean means): wall -485; nvjet contention
recovery -398 (256x192 under-a2a stretch collapses 3.0x -> 1.21x;
256x256 3.2x -> 1.2x); small compute-stream copies recovered (copy.731
2855us -> 9us, -137). Givebacks: a2a busy +131 (only +6.6% from 8x fewer
CTAs — link-bound at this message size, several legs FASTER), exposed
a2a +128, quack grouped GEMMs now under the longer a2a window +189,
AG/RS exposure +76, remat pool +45. Grid-credible core ~-300 ms/step =
the scored +0.40.

TRACE-PAIR CAVEAT RESOLVED: the agent flagged module differences
(copy.746/748 absent, +288 remat kernels, AG mix) as "the cap
participating in compile" — actually the baseline trace (xprof-h92 =
mfl-h92-b) predates the SALT commit while the treatment is salted; the
module delta is the salt fix, not the cap. The SCORED +0.40 is clean
(both sides salted, same wheel). The -55 copy.746/748 line in the trace
ledger belongs to the salt (consistent with its predicted sub-noise
+0.04).

47928 ANSWER (empirical): the fixed 8/SM grid buys 6.6% a2a latency for
~535 ms/step of critical-path compute under overlap — over-provisioned
~8x for overlapped MoE training; recommended deployment default here is
152 (1/SM); upstream-worthy fix is a tunable/occupancy-aware per-SM
factor (report only with user permission).

## 2026-08-30 ~08:20Z: C3' dual review GO; iteration 7 arms

Codex GO (P2 hardenings applied: fail-closed key-path assertion — which
also covers fable's attn-4D-leaf fragility — and CLI range cap 0..3).
Fable GO with independent re-derivation: real optimizer stack flattens
the only 4D leaves at indices 35-37 as muonh w_gate/w_up/w_down; k=0
HLO byte-identical (re-proven); donation/aliasing clean; guards loud
never clamping; checkpoint loss-free both ways. Watch items: slop-arena
arithmetic ages at k=1 (persistent ~29 GiB; first-step alloc failure
would be loud; slop 78 in pocket — NOT applied preemptively, it would
change both sides); loss-overlay expectation is exact-but-diff-HLOs-
before-declaring-bug; k=2 stays off (churn band). HERO-ADOPTION NOTE:
launch_scaling_ladder does not pass the knob — one-line wire needed if
this keeps. Arms: k1-a, fresh control i (post-commit tree, HLO-identical
to f/g/h by evidence), k1-b.

## 2026-08-30 ~09:20Z: iteration 7 — C3' k=1 DISCARD-UNSAFE (NaN), parked

Loss anatomy: 30000 bitwise-equal to control (1.280589 both — restore of
params clean); 30001 +7.3e-4; grows ~1.5e-3/step; NaN at 30008. The
corruption enters at the FIRST optimizer update -> the resident momentum
leaf is wrong at update time. Prime suspect: donate_argnums aliasing on
GPU — the resident leaf's donated input can alias the output while the
update still reads it; CPU (serial) bitwise proofs cannot expose a race.
Secondary suspect: hero-scale restore into the device-kind template
(small-probe restore was clean but not at scale/with master migration).
Root-cause delegated; k>0 banned until resolved; the committed default
k=0 is HLO-hash-identical and stays. Both review GOs missed this — the
lesson for the protocol: bitwise CPU evidence does not transfer to GPU
aliasing semantics; future placement-touching treatments need a
GPU-executable smoke (even 2 steps) before a scored arm.

## 2026-08-30 ~11:10Z: C3' NaN root-cause — narrowed to two GB200-only suspects

Local work (GPU A/B on stock XLA + real muonh: k=0 vs k=1 BITWISE
lockstep 8-12 steps under deterministic ops; production save->restore
roundtrip bitwise-exact): H-B (restore) DEAD; H-A-as-stated (JAX donation
semantics) contradicted at probe scale; CPU "bitwise proofs" were VACUOUS
(CPU drops pinned_host — the xla-cpu-hides pattern again). Trajectory
analysis: garbage momentum would NaN at 30001, zeros can never NaN;
observed +7.3e-4 -> compounding -> discontinuous NaN at 30008 = 
progressive per-step clobbering. Survivors: S1 donation-enabled buffer
reuse under the async offload schedule (the #8317 territory); S2 a stray
write from the patched ragged-a2a transport (T-H9-2 zero-copy inits + dk
wheel, symmetric-memory windows) into the address range the resident
leaf newly occupies — at k=0 it would land in unoccupied pool space,
invisible to every loss gate to date. S2 GATES THE KEEPS' HERO
PROMOTION: before recommending T-H9-2/H11 for the hero, discriminate.
Plan: disc-3 first (k=1 on fixed_pooled_wave_all_to_all — whole ragged
path inert; finite sane loss => S2 implicated, NaN => S1/generic), then
the canary checksum arm if needed. Note today's hero runs NEITHER keep,
so no live risk exists now.

## 2026-08-30 ~13:20Z: C5 dual review — smoke-gated GO; committed default-inert

Both reviewers: the router-deletion claim was OVERSTATED (the router GEMM
stays — combine-weights vjp pins it via the pre-renorm gathered logits;
the softmax reduces were never in the recompute region). Honest effect
~130-180 ms => +0.18-0.25 vs the 0.17 bar — marginal; the 2-step GPU
smoke's trace diff decides whether the scored pair is bought. Codex
verified default-off byte-identity independently (sha256 of 1.67MB CUDA
HLO) incl. identical salted-fill/ragged operands — the minted names are
provably inert unsaved. Comment corrected; committed 7228b5e066.
Sequencing: S1/S2 discriminator retry first (rack), then the C5 smoke
(NUM_STEPS=30002, finite-loss + overlay + peak ~114.3 + former-copy-
family absence), then decide the pair. S2-quarantine: C5 arms' memory
maps excluded from the S2 evidence set.

## 2026-08-30 ~22:10Z: ITERATION 7 CLOSED — S1 verdict, keeps cleared, C3' dead

Discriminator outcome: excluded-donation k=1 runs clean through 30030
(the donated variant NaN'd at 30008), zero canary mismatches, loss in
family. S1: the corruption is donation-dependent in-step clobbering of
the resident momentum leaf under the async offload schedule — the
#8317 hazard family, new member: donated device-resident opt-state RMW
leaves adjacent to async offload traffic. No positive S2 evidence
anywhere; the T-H9-2/H11 keeps' promotion dossier is CLEAR (caveat
recorded: exclusion relocates the leaf, so an address-fixed stray
write is masked rather than falsified — but nothing ever produced
S2-positive evidence, and the keeps have ~15 clean fidelity-gated arms).
C3' is dead both ways: donated corrupts; excluded is safe but -0.35
(exclusion copies + canary + relocation costs exceed the deleted host
round-trips). The instrument (ResidentDonation + canary) stays in the
tree, default-inert, as reusable corruption forensics.

## 2026-08-31 ~00:10Z: C6 DESK-KILLED by the honesty gate; two new deletion candidates

C6's ~390 ms premise failed kernel-level verification at zero rack cost:
"quantize/pack 146ms" is a PHANTOM (no such kernels; it was the
loop_convert family's 3-step total misread as per-step — actual ~62
ms/step of model-wide dtype casts); "router/topk/sort 111ms" is half
remat-recompute of the [T,E] top-k sort (C5 territory, already ruled
marginal) and the dispatch's own [TK] argsort costs ~10us; "sonic gather
135ms" IS the already-fused combine kernel. The dispatch side is one
HBM-speed gather feeding the a2a directly — the pass C6 would have fused
does not exist. Honest fusible ceiling ~58 ms.

NEW CANDIDATES from the verification (deletion class, both in the a2a
backward machinery -> dual review + GPU smoke mandatory):
- C10 dead-select deletion ~104 ms (+0.15): chunk-0's return-a2a
  backward emits a [TK,H] passthrough select with ZERO consumers
  (jaxpr-proven, scratchpad c6_dead_select.py); trace confirms both
  copies execute (loop_select_fusion_12_remat2, 208 ms family). Fix: a
  first-chunk return-a2a variant owning its zero init whose vjp omits
  the passthrough — deletes provably dead compute, bitwise-safe by
  construction.
- C11 barrier-pinned recompute ~84 ms (+0.12): the dispatch gather +
  fill re-run in backward remat though their only consumer (the a2a)
  never re-executes; pinned by optimization_barrier re-emitted under
  remat.
Plan: implement both behind separate knobs, review together, smoke,
run STACKED (+0.27 expected, above bar), back-ablate on keep per the
interaction rule.

## 2026-08-31 ~01:30Z: C10 + C11 BOTH DESK-KILLED (HLO proof); a bigger finding falls out

C10 dead-select: FALSIFIED. The chunk-0 passthrough select is never
emitted — JAX's custom_vjp transposition drops it at lowering (before
XLA). Positive control: [TK,H] select count == chunks-1 exactly, swept
over _EXPERT_CHUNKS 1/2/3/6 (0/1/2/5), and the single survivor
source-attributes to the chunk-1 call site. The earlier "survives DCE at
jaxpr level" was a pre-DCE traced jaxpr artifact. Corollary: the
profiled loop_select_fusion_12_remat2 family (2/layer, 208 ms) is NOT
two passthrough selects — it is either the live chunk-1 select plus an
XLA HloRematerialization clone (what the _remat2 suffix means) or the
ragged_dot group-mask selects; its fusion body must be read before any
candidate is minted against it.
C11 barrier-pinned recompute: FALSIFIED. The recomputed dispatch gather
feeds the recomputed collectives, which feed the w13 wgrad dot — load-
bearing, not dangling. The optimization_barrier is not the pin; the
wgrad is.

NEW FINDING (C12 candidate, unverified on GPU): collectives ARE
recomputed in the backward on the current policy — per MoE call at
chunks=2 the module holds 12 ragged-a2a ops in three groups of four:
primal (jvp), RECOMPUTE (transpose/checkpoint/rematted_computation),
and transpose. That is 1/3 of the transport instruction count spent
re-running forward transport, and it contradicts the in-code comment at
ep_ragged_all_to_all.py:410-413 ("XLA never recomputes collectives")
and the campaign memory note that collective outputs are un-rematable.
Caveat before anyone gets excited: the escape (saving the a2a output)
needs [C,H] x chunks x 48 layers ~ 178 GB — impossible; this may simply
be the forced side of the remat tradeoff, and much of the recomputed
transport may be hidden on the collective stream. Next step is
evidence, not an arm: confirm the three-group structure in a GPU HLO
dump and measure how much of the recompute group is EXPOSED before
treating it as a lever. Also worth re-reading the chunking rationale
against it.

## 2026-08-31 ~03:00Z: C13 reclassified and deprioritized (codex scope finding)

Codex P1b is decisive on classification: hero MFU is an analytic FLOPs
constant over wall time (train.py:592, _metrics.py:50), so a chunks=1
gain would be real wall-time -- but chunks=1 CHANGES THE CAPACITY GATE
(6 local experts share one buffer instead of two rigid halves), i.e. it
accepts assignments the current policy drops. That is a routing-policy
change, not a fidelity-preserving optimization, and belongs in the same
user-sign-off category the campaign already assigned to capacity-factor
moves (H4). It can answer "is this gate faster", never "is the
trajectory equivalent". P1a adds that no value/gradient oracle exists
for explicit chunks=1 (the accelerator correctness harness hardcodes the
default resolver), so a smoke loss delta could not distinguish the
intended gate change from a one-chunk VJP bug. Prior-stack data also had
chunks=1 LOSING 0.14; the re-pricing thesis needs to beat that.
DECISION: knob stays committed (default-inert, byte-identical, and the
falsified chunking comment is now corrected in-tree -- both worth
keeping on their own); the ARM is deprioritized behind higher-value work
and would need the oracle plus user sign-off. Not a keep candidate.

## 2026-08-31 ~03:05Z: long-window validation pair (best remaining rack value)

Both keeps are validated on 15-step windows. The user's binding
constraint is "loss trajectory equivalent or better", and the strongest
remaining evidence for the promotion dossier is a LONGER trajectory
comparison. A ~60-step window fits the 20-min post-compile cap at 17
s/step (~17 min). Running deployment stack vs campaign-zero over
steps +5..+58, same night, same protocol: firmer MFU estimate (54
scored points instead of 15) and a 60-step pointwise loss comparison
against the calibrated ~1e-4 band.

## 2026-08-31 ~04:00Z: C13 fable review — deprioritization REVERSED, smoke-gated

Fable's F4 is stronger than the scope objection I acted on: the capacity
gate is a LEXICOGRAPHIC GREEDY (_prefix_cap_counts, then
_clip_receiver_group_sizes, both index-order) and chunks x
chunk_capacity == local_capacity EXACTLY at this shape, so every
(sender,expert) pair's accepted count is weakly greater at chunks=1 --
the accepted set is a strict SUPERSET. Nothing accepted today is dropped
there; no existing traffic is re-routed. That is unambiguously better
fidelity, not a different policy. (Caveat recorded: the superset property
needs the exact-division condition, which holds here and not in general.)
Consequences adopted: the C-C null band is NOT the gate (newly-routed
tokens legitimately move the loss; ~2800 of 33.5M assignments change
status, so expect the delta inside the band anyway); the arm is scored as
NON-INFERIORITY (keep at >= -0.17) rather than needing +0.17, since F5
confirms MFU is exactly 1/step_time and cannot see the fidelity gain.

Other findings adopted: F1 memory estimate revised UP to +12..+16 GiB
(the backward is the binding case -- the cuDNN wgrad custom_vjp pins gu
[C,I2] and h [C,I] alongside x_dispatch), leaving ~9-13 GiB margin, not
16-20. F2 reframes the lever honestly: the chunking rationale is
re-mechanised (transient working set, both directions), not dead --
"testing whether the arena can hold an unchunked layer". F6/F7 name two
failure modes to classify by signature (remat absorbs it => engaged-
negative like slop90; NCCL symmetric-buffer OOM at first execution =>
the H10 signature, distinct from a BFC/arena OOM). F3 was a live harness
bug, FIXED (a control could not carry the kept CTAS=1 without claiming
TREATMENT; an empty knob atoi's to 0 -> default grid). F12 mints C14: a
pooled gate permits LOWERING capacity_factor at equal drops (~9% off C),
which would repay F1's memory cost -- the real prize behind C13.

Smoke gates pre-declared (all must pass): no NCCL/symmetric-buffer
failure at first execution; no dot/grouped-GEMM in the remat set and
remat count within ~2x of 325; peak_gib <= ~125; drop_fraction down ~10x
(this is the engagement check); cta_count=152 and the wheel install line
present. Failing 1-3 closes C13 at desk+smoke cost with the mechanism
named; failing 4 means the knob did not reach the backend.

## 2026-08-31 ~07:50Z: ITERATION 10 CLOSED — C13 discarded for throughput, kept as an option

Pair 23.614/23.730 (mean 23.67) vs same-night control 24.161 (and the
deployment record 24.04): -0.40 to -0.49, well outside the -0.17
non-inferiority bar, so chunks=1 is NOT a throughput keep. What it IS:
the only configuration this campaign measured that is EXACTLY DROPLESS
(0.0 both draws vs 3.3e-5 baseline), with loss in family, peak only
+5.0 GiB, remat pressure DOWN (325->218 instructions), and the accepted
set provably a superset of the chunked one. That is a validated trade,
recorded for whoever prices drop-freedom against ~1.9% throughput.
Mechanism for the loss: at one chunk each transport op moves twice the
rows, so ops run ~2x longer and are correspondingly harder to hide; the
transport is link-bound at that size, so the CTA cap cannot recover it.
C14 (pooled gate + lowered capacity factor) remains open in principle --
the memory headroom exists (+5 not +12-16) -- but it would have to beat
a -0.45 starting deficit, so it is not the near-term prize the review
hoped; recorded as a lead, not a queued arm.

## 2026-08-31 ~09:30Z: fusion pool decomposed — C15 minted (365 ms, 100% critical path)

The 3.3 s/step pool is now mapped, and the long-unidentified
loop_select_fusion_12_remat2 family is RESOLVED: it is pad_grouped_rows
from cudnn_wgrad_cute.py:74-101, the 8-row alignment pre-pass the cuDNN
grouped-wgrad wrapper applies to BOTH operands. Proof is exact: observed
grids 3,617,844 and 1,808,922 at block 128, 4 elem/thread =
(301,466+21) x 6144 and (301,466+21) x 3072, and +21 == (8-1) x 3
experts/chunk. No other buffer in the model has that row count. Both
kernels are the immediate predecessors of the cuDNN dW13 wgrad ffi_call
in every occurrence. The .remat2 suffix is a rescheduling sink, not a
duplicate (single use, ~3.7 GB, no non-remat twin) -- which also kills
the "live select + clone" hypothesis.

C15 = delete the pad. Mechanism: have _clip_receiver_group_sizes emit
8-ALIGNED group offsets so the a2a writes an already-aligned layout; the
inter-group slack rows are already zero (the output buffer is
zero-initialised and the a2a writes only valid prefixes), so the GEMM
sees zeros there and the pad copy becomes the identity. The slack is
<= 21 rows of 301,466 -- and the buffer ALREADY carries exactly that
+21 headroom today, which is what makes this a layout change rather
than a capacity change. Fidelity: zero rows contribute zero to the
grouped GEMM; no token is accepted or dropped differently. Expected
~365 ms/step = ~+0.55 MFU, the largest lever since H11, and it is the
deletion class that has won twice.
Runners-up recorded: fold _zero_inactive_grouped_rows into the QuACK
epilogue (228 ms, 173 unshadowed, medium confidence); parallelise the
ragged-a2a metadata kernels (293 ms but 72% already hidden -- the real
prize there is second-order, since the a2a cannot launch until its
offsets exist and 986 ms of a2a is exposed). Also corrected: the
grouped-GEMM family is 4,028 ms/step (QuACK 3,036 + cuDNN wgrad 992),
not ~1,628, and 886 ms of it is jax-checkpoint recompute of the forward
expert MLP. Total remat tax ~2.17 s/step = 13.5% of the step.

## 2026-08-31 ~11:00Z: LIVE CORRECTNESS EXPOSURE FOUND (outranks the MFU work)

While implementing C15 the agent surfaced, and I verified, that the hero
is very likely training with corrupted expert weight gradients RIGHT NOW.

Chain of evidence (each link checked, not relayed):
1. lib/levanter/src/levanter/grug/_moe/cudnn_wgrad_cute.py is ON
   ORIGIN/MAIN with _GROUP_ALIGNMENT = 8 (git show origin/main:...).
2. Issue #8339 (marin-community/marin, CLOSED) documents that the
   internal MoEGroupedGemmWgradBF16Kernel silently corrupts BF16
   grouped-Wgrad unless group offsets are 64-aligned: measured on GB200
   at production dims, 0.269 relative error at 8-aligned starts, 0.217
   at 32, 0 at 64 -- against a bf16 accumulation floor of ~0.003. Its
   closing comment reframes it as an integration-contract violation:
   cuDNN Frontend 1.27.0 (the exact pin in lib/levanter/pyproject.toml)
   requires every expert token count divisible by 256, and Marin
   bypassed the public facade to call the internal kernel with 8-row
   padding. CLOSED != FIXED: main still pads to 8.
3. The hero's path reaches it: silu activation + SM100 => 
   _select_expert_mlp returns _cute_expert_mlp => _expert_mlp_cudnn =>
   cudnn_grouped_wgrad at _GROUP_ALIGNMENT 8.
4. My own hero-shaped trace (xprof-c1) contains
   MoEGroupedGemmWgradBF16Kernel, with pad_grouped_rows (the 8-row
   padder) as its immediate predecessor -- i.e. the exact configuration
   #8339 measured.
5. The hero W&B run hero-12d8b6f0-dee637 is state=running at step
   37,750 as of this entry, on 177 pods.

What I did NOT do: change the hero, change the alignment on this branch,
or touch anything outside my research branch. Raising the alignment
CHANGES gradients (for the better, presumably) and is a training-
affecting decision that belongs to the user.

Honest caveats: I did not independently reproduce the numerical error
(that needs a GB200 plus a reference reduction); I am confirming a
documented, measured finding and proving the code path is live. Training
has not visibly diverged -- loss at the restore point is sane -- so this
is "expert weight gradients ~90x above the bf16 noise floor", not
"obviously broken".

SYNERGY WORTH NOTING: C15 makes the fix nearly free. Today, raising the
alignment raises the COST of pad_grouped_rows (more padded rows to copy,
365 ms/step already). With C15 the pad copy does not exist at all -- the
transport lays the receiver out aligned -- so alignment 64 or even 256
costs only slack ROWS (<=765 of 301,466 = 0.25% at 256), no copy. The
MFU lever and the correctness fix are the same mechanism.

### HOLD (user instruction, 2026-08-31): the entry above is a SUSPICION under
### independent audit, not a confirmed defect. No issue, comment, or external
### communication of any kind until the audit concludes and the user decides.
### Two clean-room audits (a fresh opus agent and codex, neither told the
### diagnosis) are running; an empirical GB200 check is being prepared as a
### third, independent line of evidence.

## 2026-08-31 ~12:30Z: SCOPE CORRECTION — the suspicion does NOT reach the live hero

The user corrected a premise I had wrong: the deployed hero predates the
ragged migration. Verified from the RUNNING job's own W&B config rather
than from the source tree:
  model.moe_implementation = fixed_pooled_wave_all_to_all
  model.remat_mode = recompute_all, expert_chunks = 1, cf 1.15
  (run created 2026-08-20, state=running, step ~38,093)
And the pooled-wave backend computes the expert MLP with plain
jnp.einsum over a compacted [E,R,H] layout
(ep_fixed_pooled_wave_all_to_all.py:457-460) -- no grouped GEMM, no
cuDNN kernel; its weight gradients come from XLA's own einsum
transpose. The only reachable callers of cudnn_grouped_wgrad /
_expert_mlp_cudnn anywhere in the tree are ep_ragged_all_to_all.py and
sonic_cute.py itself.

CONCLUSION: the live hero does NOT execute the suspect kernel. There is
no production emergency and nothing to raise. My earlier framing ("the
hero is very likely training with corrupted expert weight gradients")
was WRONG ON SCOPE -- I traced _select_expert_mlp inside the RAGGED
module and assumed the hero's config selected it, without checking the
deployed run's actual backend. That is exactly the wrong-conclusion
class this campaign is built to avoid, and I made it in the alarm
direction.

WHAT REMAINS TRUE AND STILL MATTERS: the suspect path is the RAGGED
backend -- the one #8753 and this whole campaign are preparing the hero
to migrate onto, and the one every arm in this campaign ran. So it is
not a production incident; it is a MIGRATION BLOCKER to settle before
the hero moves to ragged. Also unchanged: the 4-GPU dense-reference
gradient gate that would catch this is not run by anything (#8605
deleted the cluster-smoke workflow; #8704 tracks a replacement).
Implication for this campaign's own results: every arm ran the same
(possibly wrong) wgrad on both sides, so the A/B deltas stand; what
would be affected is the absolute fidelity claim for ragged training,
not the +0.70 measurement.

Note for the reconciliation: I gave both auditors a prompt describing
the hero configuration as ragged_all_to_all. That premise is wrong for
the DEPLOYED hero (right for the campaign branch). Their audit of the
ragged path is still valid and on-point; I must correct the framing when
we compare notes rather than let it stand.

## 2026-08-31 ~14:00Z: CONFIRMED on hardware, and FIXED (commit 2ca4c1e046)

Controlled single-variable experiment on GB200, the repo's own 4-GPU
dense-reference gate, same seeds and same exact fp32 oracle, with QuACK
forward, the sonic combine, the transport and the reference all held
constant. Only the wgrad group padding changed (8 -> 256).

  seed   ragged grad worst_slice_median      ring (reference path)
         before        after                 (unchanged)
  0      0.0259    ->  0.000807              0.000673
  1      0.0301    ->  0.000443              0.000604
  2      0.0229    ->  0.000539              0.000738
  3      0.0255    ->  0.000515              0.000532

30-70x reduction, landing AT the ring floor (ratio 0.7-1.2). The ragged
path is now numerically indistinguishable from an independent
implementation of the same math. That attributes the entire pre-fix 40x
ragged-vs-ring gap to the padding and nothing else -- QuACK reduction
order, the combine, and the transport are all exonerated.

Both audits (fresh opus agent + codex, neither told the diagnosis, no
contact with each other) independently returned "incorrect" beforehand:
codex from the vendored package's own validators (FIX_PAD_SIZE=256,
can_implement rejecting k%256, _validate_offset_sequence requiring
256-alignment and offsets[-1]==tokens_sum, Marin importing the private
class and padding to 8); the opus agent from a mechanism derivation off
vendor source (one TMA descriptor over the whole buffer while
k_tile_cnt rounds UP, so a group over-reads its successor) whose CPU
simulation reproduces #8339's GB200 numbers to three decimals and
UNIQUELY at cta_tile_k=64, the tile the installed CUTLASS pins for bf16.

Scope, final: NOT a live incident (the deployed hero is pooled-wave,
which computes the expert MLP with plain einsum). It is a ragged-
migration blocker, and it corrupted every ragged arm in this campaign.
Conditions required: SM100 + SiLU + the GPU extra; TPU/H100/other
activations fall back to ragged_dot and are unaffected.

Fix (2ca4c1e046, isolated and cherry-pickable, 2 files): pad to 256
sourced from and asserted against the installed kernel's FIX_PAD_SIZE
at call time and in a test; satisfy BOTH halves of the contract by
giving the last group every remaining row so the final offset equals
the buffer's row count (codex caught that my first attempt satisfied
only divisibility); test the padder against the contract rather than
against the transport's matching idea of the layout, which is what let
the old value pass. NOTE: the gate above ran the divisibility-only
version (a1e770c540) -- that alone restored the ring floor, so the
final-offset completion is contract hygiene, not additional numerical
correction.

CAMPAIGN CAVEAT: this campaign's ragged arms ran the corrupted wgrad on
BOTH sides of every A/B. Throughput results stand unchanged (identical
FLOPs either way). Any claim resting on loss quality needs re-checking
post-fix, and the fidelity evidence should be re-stated on the fixed
stack before the keeps are promoted.

## 2026-08-31 ~14:30Z: structural map of the block mints three new candidates

A structural mapping pass (read against ~/projects/marin, i.e. main --
numbers involving the transport are pooled-wave there and must be
re-derived on this branch) produced three fidelity-neutral,
deletion-class candidates. All three attack work nobody has looked at,
and none touch numerics.

C16 -- QB histogram allreduce deferral. _qb_beta_hist psums an int32
[384, 10000] histogram (15.36 MB) once per layer: 48 allreduces and
737 MB of payload per step, plus a pmin and a pmax each (144 scalar
collectives/step). The quantity is a CONTROL value not read until the
NEXT step -- and the codebase already does exactly this deferral for
qb_beta_local in the TOPK branch (model.py:1010-1013, "the pmean that
used to live here is deferred to _reduce_router_stats ... beta is not
read until the next step"). Deferring the histogram reduction the same
way deletes 48 sizeable allreduces from the critical path per step.
Bonus: the same path costs ~390 MiB/layer of HBM traffic (~18 GiB/step)
via an atomic bincount plus a 10,000-deep sequential scan per expert.

C17 -- Newton-Schulz 3D padding waste. _newtonschulz_padded_stack_sharded
pads 48 layers up to the 64-way expert shard count, so 16 of 64 shards
run NS on ZERO padding: 25% of all 3D NS work is computed and thrown
away, every step. The fix logic already exists in the sibling 4D path
(_newtonschulz_4d_distributed's subset-search over batch mesh axes);
the 3D path just pads to the full shard count instead. Also flagged:
the 3D path is a genuine all-to-all (~64 GB payload/step) and NS on the
expert leaves alone is ~5.5% of model FLOPs, sequenced after the entire
backward with nothing to overlap against.

C18 -- norm/gate fusion. Of ~64 full-width [T,6144] passes per block
forward (~48 GiB of HBM traffic), ~26 are pure elementwise/norm tax --
RMSNorm x2, GatedNorm x2 (five full-width passes each for a rank-128
gate), four residual adds -- and NONE of it is a fused kernel except
SConv. RMSNorm and GatedNorm are plain JAX (model.py:594, :612). A
fused norm+gate kernel is the same deletion class as the two keeps.

Also corroborated independently: _clip_receiver_group_sizes compiles to
~14,100 HLO ops with a 64-deep dependency chain per receiver -- the same
"pathologically serial metadata" the fusion-pool decomposition found at
293 ms/step on one SM of 148. Two agents reaching that from different
directions raises my confidence that the metadata path is real work
worth attacking, though 72% of it is currently hidden.

## 2026-08-31 ~15:00Z: literature survey REOPENS three dead levers

A systems/kernel survey (2025-26 MoE reports + vendor measurements)
overturned three ledger entries. Recording the reframes, because two of
my recorded negatives were measured in a configuration the reference
work identifies as the losing one.

REOPENED-1: H10 (pipelined host offloading) was measured WITHOUT
--xla_gpu_experimental_parallel_async_compute_limit=8. NVIDIA's GB200
NVL72 / DeepSeek-V3 measurement (908 TFLOP/s/device, 57% over remat)
names a three-flag set: LHS=true + pipelined_host_offloading=true +
parallel_async_compute_limit=8, and attributes 67.7% of the gain to LHS.
My stack has the first two (offload_carry forces LHS on -- note the
survey's claim that this branch runs LHS=false is true of MAIN, not of
the campaign stack) but ran the default async-compute limit of 2. So
H10's -0.25 was measured one flag short of the published recipe.
Bandwidth math supports the reopening: 805 MB carry against an ~83 ms
per-layer window is a 19x margin at the measured ~185 GB/s per die, so
0.6 s of EXPOSED offload is anomalous, not inherent. -> H19.

REOPENED-2: command buffers. The ledger entry covers the FULL disable
after COLLECTIVES capture broke (#5675, independently reproduced as a
30% degradation in openxla/xla#35360). The safe subset
--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL was NEVER tried, and
jax#27988 notes pallas_call is not captured by default at all, leaving
144 ShortConv Triton launches/forward exposed on a platform whose own
guidance warns GB200 exposes launch overhead earlier than H100. One-line
env change. -> H20.

REOPENED-3: transport SM cap. H11 capped CTAs PER SM (1/SM = 152 CTAs
spread across all 152 SMs). The literature bar is different in kind:
MoK reports TMA saturating bandwidth with <1/3 of SMs and tunes
comm SMs over 4-52; DeepEP V2 needs 4-6; HybridEP 8-16. Capping the
TOTAL grid to 8-16 CTAs would free ~136 SMs ENTIRELY rather than
leaving one resident CTA on each -- a materially different allocation
than what I measured. -> H21 (needs a small patch change: absolute CTA
count, not a per-SM multiplier).

NEW, strongest single code candidate -> H22: fold the combine-weight
multiply into the expert down-projection epilogue. Three independent
sources: Megatron-Core's memory-efficient permutation ("eliminates the
saved tensors for the router backward outright, at zero compute cost"),
ERNIE 4.5 (frees the second a2a output earlier), and Ling 2.0 (+7-10%
end-to-end). Deletes a full [TK,3072] elementwise pass AND the router
backward residual. Our combine_weights are applied after the return
a2a (model.py:978-983), so this is not done here.

CHEAP DIAGNOSTICS (hours, env-only) -> H23: jax_compiler_enable_remat_pass
defaults TRUE, so an implicit XLA remat pass fires ON TOP of our explicit
eqx.filter_checkpoint -- A/B it off to separate XLA-forced remat (slop
factor) from policy-forced remat, which decides whether H19 or a
microbatch change is the right lever. And xla_gpu_triton_gemm_any
defaults FALSE; turning it on lets the Triton emitter absorb elementwise
epilogues, the structural reason norm/router chains do not fuse into
GEMMs (openxla/xla#6407).

KILLED by the survey before costing anything: token rounding to the
M-tile (SonicMoE Algorithm 4, +15.9% at E=256) -- the gain scales as
M_tile/M_e and our M_e is ~87,000 rows, so the padding waste is 0.15%.
DeepEP dedup: 5% ceiling inside one NVLink domain. Also demoted: C18
norm fusion (Snider & Liang: fusion gives ~10% once outputs exceed
registers, because memory movement, not launch overhead, is the
bottleneck -- reduce bytes, do not chase fusion), and C17/NS work (four
independent measurements put Muon at 1-3% of step time; a 6x optimizer
speedup buys ~1-2.5% at best, so treat it as a profiling check).

Ceiling: best published fine-grained-MoE MFU without quantization is
28.8% (Megatron, 64 experts top-8, H100, full stack). We are at 24.0.
