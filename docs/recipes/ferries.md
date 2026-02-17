# Recipe: Daily Ferry (Integration Test First)

## Overview

Use this recipe to propose, launch, and monitor a **daily 125M ferry** as a high-signal integration test of Marin TPU training infrastructure.

A ferry is a regularly scheduled run at approximately the same scale. For daily ferries, the goal is reliability and regression detection first, with limited experimentation.

Current defaults:
- cadence: daily
- scale: ~125M parameters, ~1e19 model FLOPs target
- PR target branch: `main`
- launch cluster: `us-central1` (maps to zone `us-central1-a`)
- issue filter: label `experiment`
- change budget: guidance only (not script-enforced), usually 1-2 knobs

## Prerequisites

- Local checkout of this repository with write access.
- `uv` available.
- `gh` CLI authenticated (for issue/PR workflows).
- Access to TPU launch environment used by `ray_run.py`.
- Existing ferry template path: `experiments/ferries/daily.py`.

## Inputs Required Before Proposing A Ferry

Collect these first:
1. Last ferry reference:
- issue link
- PR/commit link
- run/job link
2. Human objective for today:
- normal daily integration check, or
- targeted regression confirmation
3. Interval boundary:
- since last ferry run (not fixed 24h)

If objective is unclear, ask before editing.

## Human Workflow

1. Review last ferry outcome and choose today's objective.
2. Let the agent propose a bounded update to `experiments/ferries/daily.py`.
3. Review PR to confirm change size/risk is acceptable.
4. Give explicit launch approval.
5. Track run updates in the ferry issue.

## Agent Workflow

### 1) Build context since last ferry

```bash
# Set these from the last ferry PR/issue record.
LAST_FERRY_SHA=<last_ferry_commit_sha>
LAST_FERRY_DATE=<YYYY-MM-DD>

git log --oneline "${LAST_FERRY_SHA}..HEAD" -- experiments/ lib/ scripts/

gh issue list \
  --label experiment \
  --search "updated:>=${LAST_FERRY_DATE}" \
  --limit 100
```

Notes:
- Use `experiment` label only for now.
- If `LAST_FERRY_SHA` is unavailable, use a date fallback and record that in the PR rationale.

### 2) Propose the next ferry

Edit `experiments/ferries/daily.py` by pattern-matching the last ferry.

Rules:
- Keep model scale stable unless human asks otherwise.
- Usually change 1-2 knobs total.
- Avoid "everything changed" proposals.
- Identical rerun is allowed only for explicit regression checks.
- Update run name with date (example: `daily-125m-2026-02-12`).

### 3) Open PR to `main`

PR must include:
- link to last ferry issue/PR/run
- what changed and why
- risk level (`low`/`medium`/`high`)
- explicit before/after config delta
- launch checklist:
  - [ ] human approval
  - [ ] issue created/updated
  - [ ] monitoring loop started
  - [ ] optional/manual Discord update posted

### 4) Launch only after human approval

Illustrative launch shape:

```bash
uv run lib/marin/src/marin/run/ray_run.py \
  --no_wait \
  --cluster us-central1 \
  -- python experiments/ferries/daily.py --run_name "daily-125m-$(date +%F)"
```

Capture and post:
- `job_id`
- cluster
- run URL/log URLs
- launch timestamp

### 5) Monitor to completion

Follow `.agents/docs/job-monitoring-loop.md` using:
- `job_id`
- `cluster=us-central1`
- `experiment=experiments/ferries/daily.py`

Escalation policy:
- small obvious bug: fix + relaunch
- repeated same failure after one fix: escalate
- infra-wide failures (cluster/quotas/OOM instability): escalate immediately

### 6) Close the loop in GitHub

Post a final issue update with:
- pass/fail status
- key metrics (or links)
- notable failures/fixes
- next-step recommendation for tomorrow's ferry

Discord automation is deferred to Phase 2. Manual Discord posting is optional.

## Change Budget Guidance

For daily integration ferries:
- Prefer infra/pipeline validation signal over high-variance model exploration.
- Keep changes small and reversible.
- Suggested buckets:
  - datamix minor change
  - optimizer/hyperparameter micro-tweak
  - model architecture micro-adjustment
  - training-system toggle (logging/checkpointing/loader behavior)
  - observability instrumentation

## Validation Checklist

- `experiments/ferries/daily.py` diff is small and intentional.
- PR targets `main`.
- Ferry issue is updated before launch.
- Launch command uses `us-central1`.
- Monitoring loop started and maintained until terminal state.

## See Also

- `.agents/docs/job-monitoring-loop.md`
- `.agents/projects/ferry_framework.md`
