# Hero checkpoint completions

## Accepted scope

Sample every retained permanent checkpoint in the production hero lineage. Update the
public report and one GitHub issue comment each day. Keep the prompt bank in Git and
keep requests, queue state, and results in object storage. Do not start or change the hero.

## Implementation

1. Rebase on `origin/main`. Completed at `e9a1f53d6`.
2. Add typed requests, stable sample identities, and a persistent queue. Use the existing
   conditional-object API. Record an attempt before submission. Reconcile results and
   deterministic Iris job names before a retry. Permit one active sample job.
3. Add a native, weights-only sampler. Restore the exact checkpoint, including pending
   router bias. Validate prompts and use independent random streams. Record token IDs,
   stop reasons, source revision, and all generation settings.
4. Add the initial ten-prompt bank and an explicit production lineage. Discover committed
   permanent checkpoints only. Give new checkpoints priority over retained history.
5. Add a history browser with checkpoint comparison. Store small result objects once.
   Publish a dated report and update a stable public link each day. Update one marked
   comment on issue 8827. Report publication must not require another GPU job.
6. Add an hourly GitHub Actions controller with the existing Iris and storage credentials.
   Keep the queue between invocations. Bound job duration and retries.
7. Test queue recovery, duplicate prevention, checkpoint selection, generation limits, and
   report publication. Run repository checks and independent peer review.

## Validation and handoff

Use local storage and a fake job service for failure tests. Do not allocate 64 GPUs or
publish to the public bucket during local tests. Report these live validation limits at
handoff. Document required existing credentials and the automatic schedule.

Implementation and independent review are complete. All 40 focused tests pass, including
eight checkpoint restore layouts. Focused type checks, documentation links, and report DOM
checks pass. Full repository checks are limited by missing dependencies and a C compiler.
The actual browser cannot start because this environment lacks its system libraries.

Activation remains blocked: the scheduled CI account is not in the target cluster's
submitter allowlist. Obtain approval for that access change before deployment. No live GPU
job, public report upload, issue update, access change, or controller restart has run.
