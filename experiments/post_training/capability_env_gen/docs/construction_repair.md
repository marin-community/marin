# Repairing a constructed task

Completed builder sessions are reusable work, not a reason to ignore a later failed
gate. `capability_pipeline.repair.run_repair` gives a fresh GLM agent the existing
workspace, admitted contract and concrete controller/reviewer findings. It retains
a hash-bound before-state outside the build item and records all changed files,
the repair transcript, a structured receipt and measured check artifacts.

The controller supplies a fresh directory for each bounded repair round. It must
carry the previous status, raw failing outcomes and independent quality findings
into `feedback`, rather than only a generic request to improve the task. A failed
inference or provider operation is diagnosed as such; it is not evidence that the
task should be made easier. Completed construction sessions need not be repeated.

The repair agent may change construction assets, but must preserve capability,
difficulty, intended environment, critical gates and reward aggregation. A necessary
substantive redesign returns `needs_readmission`. Changes to the admitted contract
or historical runtime artifacts invalidate the repair receipt. The before-state
remains available even when the repair does not finish.

`ready_for_validation` means only that edits and measured checks were recorded.
It never certifies runtime or semantic quality. Run fresh lowering, the separate
authored oracle and blind solver, controls, independent attacks, applicable judge
calibration and an independent semantic review on the modified task. Old evidence
must remain under its original attempt identity. Source patches implementing
composed verification are versioned runtime changes, not task-local reward rewrites.

The synthesis controller applies bounded repairs in the same worker after a fresh
actionable task failure, missing build-acceptance evidence, or a completed
nonaccepting semantic review. It also resumes repairable prior results without
resetting the per-item budget. Fresh ungraded runtime/provider errors, incomplete
adversary trials, unresolved solver or attack adjudication, unfinished reviews,
and incomplete judge calibration do not trigger task edits. Every repair round reserves a numbered
budget receipt and immutable copy of the triggering `status.json` before invoking
GLM, so even an interrupted repair counts across resume. It embeds matching
capability audits, the current construction checklist and shared quality conditions
in feedback, then archives prior runtime/Harbor/calibration evidence before fresh
validation. Completed builder sessions remain reusable.

While a repair is running, `items/<item>/controller/active-operation.json` is the
authoritative activity receipt. It records the repair attempt, transcript directory,
prior terminal state and status digest, start time, and an `active` state without
rewriting `status.json`. On return it closes as `completed` with the repair outcome;
an exception closes it as `error` before the existing controller error handling.
The same closed receipt is retained under
`controller/operations/repair-attempt-<n>.json`. The prior `status.json` remains the
last completed gate result, while the job scheduler remains authoritative for the
overall worker/job lifecycle. A stale failure status plus an `active` operation is
therefore an in-progress repair, not a second failure or evidence that the worker
has stopped.

Repair scope includes every applicable measured audit, even when the immediate
controller failure is only a malformed control label. The prompt explicitly
requires the builder's `task/build-acceptance.json` and its evidence before
readiness, while leaving fresh controller-owned runtime and semantic checks to
the next gate. This distinction responds to c05 continuation003 attempt 1,
which only corrected a control category and left the audited grader unchanged.

Unit tests cover evidence preservation, contract tampering, unmeasured readiness,
resume ordering and fresh validation. The c05 and c17 histories each contain two
construction repair rounds. Both current versions are rejected after later
validation: c05 for a grader false negative, c17 for a rewarded conflicting-key
submission. c17's earlier pilot acceptance remains historical evidence. Count
repairs from the union of run-level `repairs/<item>/attempt-*` and
`repair-budget/<item>/attempt-*`, not under the item directory. A new continuation
name does not reset the budget. Ungraded runtime infrastructure failures do not
start a construction repair. A GLM repair invocation that starts and fails does
consume its reserved round.
# Rewarded-attack repair routing

A completed independent adjudication is run in the synthesis worker. A hash-bound
receipt that cites the raw grade, transcript, admitted contract, and task rubric
may identify a rewarded attack as an exploit. When every attack trial has a
graded outcome, all rewarded steps have completed adjudication, and at least one
is an exploit, the controller preserves that evidence and spends the next
global construction-repair round on a GLM task revision. Legitimate rewarded
solutions can coexist with an exploit. An uncertain decision, missing grade,
changed evidence, or incomplete reviewer remains a hold; the controller does
not repeatedly sample adjudicators for unchanged evidence.

Each final item status records `repair_budget` with `used`, `max`, and
`exhausted`. When the current failure is a measured, repairable construction
failure and the budget is exhausted, `terminal_disposition` is `rejected`.
The original failure state and retained attempt evidence remain in the status
and repair history. Operational and ungraded holds do not become rejections
merely because the construction budget is exhausted.

A failed native or composed judge calibration enters the same bounded repair
loop only when its frozen fixture, pinned specification, raw results, and
calibration report still match their recorded digests; every declared case and
repeat has a finite graded reward on its expected judge and machine-gate path;
and the reproduced calibration metrics show a reward, stability, or statistical
failure. The controller passes the exact report, raw-results, fixture hashes,
issues, and metrics to GLM. Missing fixtures, runner exceptions, ungraded
attempts, missing model-path coverage, and changed artifacts remain
`pending_judge_calibration` without spending a task-edit round. The controller
does not resample judge calls merely to clear a measured failure.

Eligibility also binds the current item/proposal identity and the controller's
three-repeat, 0.15-spread policy. During that repair, existing calibration cases
must retain their IDs and full contents, including candidate answers and expected
ranges. Added cases and specification-hash rebinding are allowed; changing or
removing a measured case forces `needs_readmission`. This prevents relabeling a
known failure to obtain a passing calibration result.
