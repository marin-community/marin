---
name: research
description: Research an explicitly requested Marin question or coordinate a multi-session experiment; find prior work, assess evidence, and record results in the task's existing durable surface.
---

# Research

Use this skill for a requested prior-work brief or research program. Ordinary
implementation does not require a research workflow.

## Choose the record

Return a compact brief in the conversation for bounded work. For durable or
cross-session work, use the task's existing issue, PR, W&B run or report,
published report, or durable session channel. Do not create standalone
repository design, research, plan, logbook, or snapshot files. If the user
explicitly requests a particular artifact, honor that request.

Update `docs/` or the relevant `OPS.md` when the result changes reusable product
or operational guidance. Keep raw logs and dense data in their source systems.

## Investigate

1. State the question, decision, and stopping condition.
2. Search the current checkout for task-local context. Search Echo for Marin
   context unless the same logical session already completed a relevant search
   and the question, repository scope, and freshness requirements have not
   changed. Reuse and cite those results instead of repeating the search. Use
   primary external sources when the question benefits from outside evidence.
3. Test the leading explanation against contradictions, negative results, and
   materially different operating regimes.
4. Stop when additional sources or experiments no longer change the decision,
   or when the requested effort is exhausted.

For experiment results, record the exact command, source revision, material
configuration, hardware, baseline, result, and interpretation. Label estimates
and exploratory results. Record failures that rule out a hypothesis; omit
routine debugging history.

Use W&B for scalar series, plots, large comparison tables, or raw artifacts that
are too dense for the narrative record. Runs that require direct comparison
must share a project. Verify row counts, key uniqueness, aggregation, and the
numbers cited in the narrative before publishing a claim.

## Report

Scale the response to the decision. Include:

- the question and conclusion;
- evidence and source links;
- contradictions, limitations, and confidence;
- the next experiment only when it could change the decision.

Do not require a special branch, experiment ID, hypothesis queue, update
cadence, issue, tag, or W&B project unless the task itself needs one.
