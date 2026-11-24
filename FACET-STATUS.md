# Difficulty: facet_v2 systematic environment-start failure — STATUS (2026-11-24)

Worker: OpenCode session ses_df1e36369ffeJibQG2Kd6P560U (benfeuer). Coordinator:
b610073d-d871-4fd6-a41f-c087d2723ec1; difficulty lane operator: ses_dfec1e64affesTRF6vLcYkvXgy.
Controller untouched throughout (per dispatch).

## RESULT

**Root cause found, fixed at the source, revision published and verified.**

facet_v2's "Machine is closed" deaths are **not** Daytona, not image, not resource-profile:
they are the **grading quiesce killing the LOCAL sandbox's machine keeper**.

### Cause (receipt-verified in region)

The shared native root-delivery grader builder
`taskcompendium.harbor.root_delivery.native_file_reward_grader()` emits a shell quiesce as
grading `collect[0]` that kills every process outside its own ancestry except PID 1. On the
LOCAL sandbox tier the namespace's PID 1 is **bwrap itself** and the machine keeper's payload
(`exec sleep 2147483647`) is **PID 2** — so the quiesce killed the keeper, bwrap exited 137,
the keeper watcher closed the machine, and every later grading step raised
`MachineTerminated("Machine is closed")`. The attempt typed as GRADER_INFRASTRUCTURE_FAILURE
(verifier started, no verdict) and the planner re-dispatched forever (~700/hour; 6,122+ rows,
zero scored ever — facet_v2's grader is exactly this builder).

Evidence: in-region A/B repro jobs `/benfeuer/benfeuer-facet-keeper-diag-v5..v11-20261124`
(cw-us-east-08a). Dry-run quiesce printed `WOULD-KILL: 2` with `victim-cmd: sleep 2147483647`;
quiesce-only scenario killed the keeper at age 0.34s every run; without the quiesce the machine
survived 12s+ with strays present. Daytona-tier runs pass because the init layout matches the
PID-1 guard there.

**Shared, not facet_v2-specific**: every root-delivery source grading on the local tier dies the
same way (consistent with the stem_mcqa and Cascade-2 signatures the coordinator forwarded). The
~20% ENVIRONMENT_START_FAILURE rows are a second, creation-time close mode on the same tier,
not yet killer-pinned; the dominant mode above explains 3,437+ of the failures.

### Fix (landed)

- **Generator**: both quiesce variants (shell + python) now spare exactly the keeper sleeper
  argv (`sleep 2147483647`). **LANDED `12694eb511` on atlas-env-audit via atlas-push** (worktree
  facet-keeper-fix, branch benfeuer/facet-keeper-fix); 16 tests pass, including a new Linux-root
  regression (`unshare --pid`) proving strays die and the keeper sleeper survives both variants.
- **Revision (RL data)**: `open-athena/fweindel-facet-v2-quiesce-safe-20261124-v1` (private,
  775 tasks). Only `grader.collect[0].argv[2]` changed (old text verified byte-identical to the
  pre-landing generator across all rows); task bytes otherwise unchanged; `task_sha256`
  recomputed over the patched task_spec JSON. Readback verified on a fresh download.
- **Evidence**: `s3://marin-us-east-02a/marin/rl-data/atlas/facet-quiesce-repair/20261124/`
  (repair-summary.json, repair-ledger.json per-task old/new hashes, ab-verification-receipt.json).

### Verification (in-region A/B)

Job `/benfeuer/benfeuer-facet-revision-verify-v2-20261124` (privileged CPU pod, same task image):
- old spec: stray spawned → collect[0] → `MachineTerminated: Machine is closed`; machine dead.
- quiesce-safe: stray spawned → collect[0] `exit=0 out='1'` (killed exactly the stray) →
  restore → extract → `bash /tests/test.sh` all exit 0 → **machine alive throughout**.
(The bare replay's grader `cp` step returns 1 because the trial's reward-file staging was not
present; it is not a collect step.)

### Hold

Typed `backend-repair-pending` escalation for facet_v2 applied through the ledger's own
EscalationLog:
`s3://marin-us-east-02a/marin/rl-data/atlas/difficulty-ledger/v1/escalations/backend-repair-pending-FWeindel-validated-tasks-facet-v2-20261124.json`
(also in the ledger ESCALATIONS.md, DIFFICULTY_ESCALATION emitted). **HOLD VERIFIED WORKING
(lane operator, 2026-11-24 ~16:40 UTC): after the controller consumed the escalation, the
16:34 planning round has NO facet_v2 entry and attempt dirs held steady at 20 for ~49 min
(pre-hold rate: a batch every few minutes). Dispatch stopped.** During diagnosis (before the
controller wired the hold) churn continued (~140 unscored rows after 15:30 UTC).

**Cutover-safe conversion, then PARKED (2026-11-24 ~17:05 UTC):** the incoming controller's
startup plan consumes repair-verification objects (`difficulty/ledger.py repair_verifications()`).
The controller owner first read that as a hold; the lane operator then verified the actual
consumption path — `repair_verifications()` **prioritizes** the linked AttemptBatch at the front
of its band (`repair_verifies()`), it does not hold, and GRADER_INFRASTRUCTURE_FAILURE always
plans an AttemptBatch. With the re-queue not landed, the record would have *prioritized* burning
the old broken revision. **The record was therefore parked**: removed from
`v1/repair-verifications/` (the consumption glob no longer sees it — verified) with the reason
and re-install precondition at
`v1/repair-verifications-parked/backend-repair-pending-FWeindel-validated-tasks-facet-v2-20261124.json`.
State after this: the OLD controller's escalation hold keeps facet_v2 quiet until cutover; on
the new controller, facet_v2 resumes at normal band position against the OLD revision until the
difficulty owner's re-queue lands (registry → `open-athena/fweindel-facet-v2-quiesce-safe-20261124-v1`
+ fresh frozen panel) — recommended to gate the cutover. The parked record is re-installed
verbatim (its `reinstall_uri`) as the re-queue's final step, at which point front-of-band
verification is the intended semantics.

## Resume (owner actions)

**Cutover-gate status (2026-11-24 ~17:15 UTC, answered to the controller owner via relay):**
the canonical re-queue is **NOT complete**; the repair side is fully ready. Remaining:
1. **Registry/declaration update** (owner/Atlas surface): the served facet_v2 declaration's
   `current_release` → `open-athena/fweindel-facet-v2-quiesce-safe-20261124-v1` — cells key on
   `declaration_population_revision`, so nothing dispatches against the new revision before this.
2. **Fresh FreezeSamplePanel** against that release (lane machinery: the controller's
   TASK_SPEC_FACTORY_PENDING remediation can build it once (1) lands, or the owner drives it).
3. **Panel/queue binding verifies** (the frozen queues 2026.11.13.8 / 2026.11.17.5 carry only
   the old revision).
4. I then **re-install the parked repair-verification verbatim** (final step) and the repair
   verifies front-of-band.

Exact binding: release `open-athena/fweindel-facet-v2-quiesce-safe-20261124-v1`, tasks.parquet
sha256 `f026ee23f5eee69a5d191f974a3bd72ed681d66c1f5d75afa9cc85a343d7c34d`, 775 rows, every
`grader.collect[0]` keeper-safe (readback verified). The controller owner retains persisted
action dedupe until the re-queue verifies.

Then the lane operator verifies the first scored rows land and the 6,122 counter stops growing.

Also carried: facet_v3 (728 tasks, same builder, same one-line defect) needs the same revision —
the landed generator covers it. Optional serving-owner alternative: make the local keeper
payload PID 1 in shellbox so the existing PID-1 guard holds for every source without relying on
the sentinel.

## Re-pin execution (assigned to this worker end to end, 2026-11-24 ~16:10 EST)

The canonical re-queue = a fresh `OracleAdmittedReleaseReceipt` bound to the quiesce-safe
release (the facet binding flows through it), which the validator
(`oracle_admitted_release.py _qualified_population`) pins hard: actor census, grading census,
lexical, output dedup and the SAMPLE QC review must ALL name the new release revision +
task_specs digest — no repair shortcut exists in the validator.

Progress:
1. **Actor + grading census: DONE.** Job `/benfeuer/benfeuer-facet-repin-census-v8-20261124`
   (cw-us-east-08a): 775/775 quiesce-safe specs measured, 775 machine slots, 405 distinct
   shapes, 0 failed tasks, grading environments assessed; no Daytona API calls. Measurement
   document uploaded: sha256 `3e53b6772073661bbd5d6e2828ddcf1226e6e11ba741cd154d17d0adc91a3ab2`
   at `s3://marin-us-east-02a/marin/rl-data/atlas/facet-quiesce-repair/20261124/census-measurement.json`.
2. Lexical re-measurement — bounded CPU job, next.
3. Within-source dedup re-measurement — bounded CPU job.
4. **SAMPLE QC re-review at the new pin — the gating item**: 100 tasks re-reviewed through the
   sample QC machinery (model judges + spend). The validator requires
   `sample.review.panel.{dataset_revision, task_specs_sha256}` == the new values.
5. Receipt assembly → `open-athena/marin-rl-data-atlas-registry` publish → repo shard reference
   → `declaration_audit` → the catalog load → the controller's panel remediation → re-install
   the parked repair-verification.

Decision raised to the coordinator: fold the re-pin into the r40 release pipeline's flow (it
owns the SAMPLE/QC stage machinery and is mid-flight at cut-v8 with facet_v2 in the plan), or
authorize the QC spend for this worker. Items 2-3 can run meanwhile.

## Execution handoff (2026-11-24 ~17:00 EST)

This session reached its context limit after: root cause, generator fix (12694eb511), revision
publication + A/B verification, both hold records (escalation verified working; cutover-safe
parked copy), and the fresh census at the new pin (sha 3e53b677…). The remaining stages are
fully specified in **REPIN-PLAN.md** (same directory) — a self-contained execution plan with the
validator's hard requirements, the receipt assembly shape, registry shard rules, reference
facts, and SHAs. Run it in a fresh session or fold it into the r40 release pipeline's flow
(stage list maps 1:1 onto the pipeline's stages).

## Iris jobs (watchdog registry)

| Job | Cluster | Purpose | State |
| --- | --- | --- | --- |
| /benfeuer/benfeuer-facet-keeper-diag-20261124 (v1) | cw-us-east-08a | diag bundle build failure (shellbox README) | failed (superseded) |
| ...-diag-v2/-v3/-v4/-v5-20261124 | cw-us-east-08a | keeper repro iterations | completed |
| ...-diag-v6/-v7/-v8/-v9/-v10/-v11-20261124 | cw-us-east-08a | kill-isolation matrix, /proc observation, concurrency | completed |
| /benfeuer/benfeuer-facet-revision-verify-20261124 (v1) | cw-us-east-08a | missing job deps | failed (superseded) |
| /benfeuer/benfeuer-facet-revision-verify-v2-20261124 | cw-us-east-08a | A/B verification of the revision | succeeded |

| /benfeuer/benfeuer-facet-repin-census-v1..v8-20261124 | cw-us-east-08a | fresh declared-demand census at the quiesce-safe pin (v8 succeeded; earlier iterations fixed imports/deps) | v8 succeeded |
| /benfeuer/benfeuer-facet-repin-census-upload-20261124 | cw-us-east-08a | upload attempt (landed node-local /cache; superseded by v8's in-job upload) | failed as designed |

Echo work-log: #6866. Rhizome reports to coordinator + lane operator throughout.
