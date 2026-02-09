# Recipe: Agent-Directed Research

## Overview
Use this recipe for long-running, exploratory research where an agent iterates on
benchmarks, experiments, and hypotheses over multiple sessions.

This workflow optimizes for:
- reproducibility,
- clear decision history,
- fast iteration loops,
- handoff quality for humans and future agents.

## Naming
Use **research logbook** consistently in prose and file naming.

Suggested path pattern:
- `.agents/logbooks/<topic>.md`

## Core Artifacts
For each research thread, maintain all of:

1. A long-lived branch (for example `research/<topic>`).
2. A GitHub experiment issue (using `.github/ISSUE_TEMPLATE/experiment.md`).
3. A local append-only research logbook in `.agents/logbooks/<topic>.md`.
4. Optional W&B runs/report for dense numeric output and charts.
5. One or more tags to seal meaningful snapshots.

## W&B Project Policy
When using W&B:
- Default to creating a new W&B project for each new research logbook series.
- Reuse an existing W&B project only when the work is an obvious follow-up to that series.
- If explicit run-to-run comparison is required, runs must be in the same W&B project.
- You cannot reliably move/copy runs across W&B projects later, so decide project scope early.

## When to Use Marin Executor vs Dev TPU
- Use **Marin executor framework** when:
  - you need production-like pipeline behavior,
  - cluster job metadata/repro tracking matters,
  - experiment setup is non-trivial.
- Use **dev TPU** when:
  - iteration is quick,
  - you are tuning kernels or benchmarks,
  - full pipeline apparatus is unnecessary.

### Dev TPU script (explicit)
For fast iteration, use:
- `scripts/ray/dev_tpu.py`

Typical patterns:

```bash
# Allocate (or reconnect to) a dev TPU
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/<cluster-config>.yaml \
  --tpu-name <name> allocate

# Execute a quick benchmark command on the dev TPU
RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py \
  --config infra/<cluster-config>.yaml \
  --tpu-name <name> execute -- \
  .venv/bin/python lib/levanter/scripts/bench/bench_moe_hillclimb.py ...
```

Notes:
- Prefer one long-lived `execute` shell/script for sweeps to avoid repeated sync overhead.
- Pass environment variables through `execute` when needed (for example TPU VMEM flags).

Rule of thumb:
- Start with dev TPU for fast hillclimbing.
- Confirm final claims on the execution path that matters most.

## Standard Workflow

### 1) Kickoff
1. Create/switch to a long-lived research branch.
2. Create an experiment issue from `.github/ISSUE_TEMPLATE/experiment.md`.
3. Start a research logbook file in `.agents/logbooks/<topic>.md`.
4. Add links both ways immediately:
   - logbook links to the issue URL,
   - issue body links to the logbook path.

At kickoff, write:
- problem statement,
- success metrics,
- initial hypotheses,
- first experiment matrix,
- links to relevant code paths.
- stop criteria (what evidence is enough to stop/ship/escalate).
- a fixed baseline case for repeated comparison.

Issue timing guideline:
- Prefer creating the experiment issue sooner rather than later.
- Confirm timing with the human collaborator if there is uncertainty about scope or visibility.

Experiment ID convention:
- Assign a short experiment ID prefix for the series (for example `MOE-HC`).
- Use IDs like `MOE-HC-001`, `MOE-HC-002`, ... in:
  - logbook entries,
  - W&B run names,
  - issue comments.

### 2) Day-to-Day Research Loop
For each non-trivial experiment:
1. Run benchmark/experiment.
2. Append to the research logbook:
   - date/time,
   - exact command,
   - shape/config,
   - key outputs (tables, deltas, failures),
   - interpretation,
   - next decision.
3. Push dense scalar series and plots to W&B when tables are too large.
4. Add a GitHub issue comment with:
   - concise delta since last update,
   - important findings only,
   - links to research logbook sections and W&B runs.

Update cadence:
- Post an issue update whenever a significant milestone is reached, or every 6 hours, whichever is sooner.
- Significant milestone means someone is likely to want to find that update later.
- If no significant milestone occurred by the 6-hour mark, post a brief heartbeat update with current status, blockers, and next ETA.

Issue comment style:
- Mostly append-only.
- Do not rewrite historical comments.
- Keep claims scoped and falsifiable.

### 3) Maintain the Issue Body
The issue body is the public summary layer:
- keep a short TL;DR up to date,
- track scope changes,
- keep links current,
- summarize takeaways that matter for non-specialists.
- maintain a short decision log (decision, evidence, date, owner).
- maintain a negative-results index with links.

Write the body for readers who:
- know Marin/LLM systems generally,
- do not know this specific thread.

Reader-context rule:
- Write issue updates/body so they stand on their own for readers who were not part of the conversation.
- Include enough framing (goal, assumptions, exact commands, and result interpretation) for someone else to reproduce or critique the claim.
- Logbook entries can be terse/context-local, but should still include exact commands and links to supporting artifacts when relevant.

Confidence labels:
- Label major claims as one of:
  - `exploratory` (single run / weak evidence),
  - `replicated` (repeated and consistent),
  - `stable` (held across shape/seed/hardware variants relevant to scope).

### 4) Snapshot and Seal
When you reach a meaningful milestone:
1. Commit only relevant files.
2. Tag the commit (annotated tag preferred).
3. Push the tag.
4. Add issue comment linking:
   - tag,
   - commit,
   - benchmark/report files.
5. Include a repro bundle for the snapshot:
   - exact command(s),
   - hardware/cluster and device count,
   - critical environment variables,
   - primary comparison table.

This creates a stable checkpoint even if the branch continues.

### 5) Finish
At completion:
1. Add a final issue summary:
   - what worked,
   - what did not,
   - confidence level and limitations.
2. Add **next steps** (small, concrete, ordered).
3. Close the issue.

## Research Logbook Template
Use this structure in `.agents/logbooks/<topic>.md`:

```md
# <Topic>: Research Logbook

## Scope
- Goal:
- Primary metric(s):
- Constraints:

## Baseline
- Date:
- Code refs:
- Baseline numbers:

## Experiment Log
### YYYY-MM-DD HH:MM - <short label>
- Hypothesis:
- Command:
- Config:
- Result:
- Interpretation:
- Next action:
```

## Issue Update Template
Use concise updates in issue comments:

```md
Update: <short label>

- Change:
- Result delta:
- Confidence:
- Links:
  - Tag:
  - Logbook section:
  - W&B:
- Next:
```

## Issue Body Template Add-ons
Keep these sections in the issue body:
- `TL;DR`
- `Scope`
- `Decision log` (append as decisions are made)
- `Negative results index` (links to comments/logbook entries)
- `Current baseline` (shape/config + reference numbers)

## Experiment Design Rules
- Run one-axis sweeps first (change one variable at a time), then run interaction sweeps.
- Keep comparisons apples-to-apples (same shape, dtype, pass mode, and backend unless that axis is the test).
- Keep a fixed baseline case and compare against it repeatedly.
- Only move the baseline after enough repeated evidence; note the baseline change explicitly in the logbook.

## Practical Rules
- Prefer short-lived code changes unless a persistent harness is clearly useful.
- Keep benchmark harnesses configurable and minimal.
- Record exact command lines for every headline number.
- Treat failures as first-class data (include why they failed).
- Treat negative results as first-class outcomes; they prevent repeated dead ends.
- Separate "measurement code" from "production path" whenever possible.
- Prefer persistent remote shells/scripts for long sweeps; avoid repeated sync/launch overhead when possible.
- Check accelerator contention (existing processes/locks) before attributing regressions to code.

Ops hygiene checklist (before claiming regression):
- confirm no stale benchmark process is still occupying accelerator,
- confirm lockfiles/state are clean,
- confirm comparison uses same device count,
- confirm command/config/env are identical except the tested axis.

## Validation Checklist
Before posting a result:
- Command is reproducible.
- Shapes/config are explicitly listed.
- Comparison is apples-to-apples.
- Version snapshot exists (commit or tag).
- Result is written in the research logbook and linked from issue.

Before closing the issue:
- Final TL;DR is current.
- Next steps are listed.
- Final snapshot tag is linked.

## See Also
- `.github/ISSUE_TEMPLATE/experiment.md`
- `docs/recipes/organize_experiments.md`
- `docs/recipes/add_pallas_kernel.md`
