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
