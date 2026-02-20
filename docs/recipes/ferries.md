# Recipe: Daily Ferry (Integration-Test First)

## Overview
Use this recipe to propose, launch, and monitor a daily ferry run as a high-signal integration test on TPU infrastructure.

A ferry is a regularly scheduled run at roughly fixed scale. The daily ferry is intentionally conservative:
- stable scale and architecture,
- bounded day-over-day edits,
- explicit human approval before launch,
- strict monitoring and restart hygiene.

## Current Baseline
Canonical template:
- `experiments/ferries/daily.py`

Current baseline defaults in that template:
- data: shared `nemotron_mix` baseline (same source used by `experiments/ferries/canary_ferry.py`)
- model size: Llama ~125M (`llama_150m`)
- sequence length: 4096
- train batch size: 512
- FLOP target: ~1e19 (overrideable via env)
- default cluster: `us-central1` (zone `us-central1-a`)

Companion canary:
- `experiments/ferries/canary_ferry.py` is the smaller always-on canary lane; keep data-mix assumptions aligned between daily and canary ferries.

## Prerequisites
- Local checkout with write access.
- `uv` installed.
- `gh` authenticated for issue/PR workflows.
- Access to the Ray/TPU launch path used by `lib/marin/src/marin/run/ray_run.py`.
- A canonical ferry issue and prior run/PR history.

## Inputs Before Proposing
Collect:
1. Last ferry references:
- issue URL
- PR/commit URL
- W&B run URL and Ray job ID
2. Human objective for this interval:
- standard integration pass
- or explicit regression investigation
3. Interval boundary:
- use "since last ferry run", not fixed wall-clock day boundaries

If objective is ambiguous, ask before editing.

## Standard Workflow

### 1) Build context since last ferry
```bash
LAST_FERRY_SHA=<last_ferry_commit_sha>
LAST_FERRY_DATE=<YYYY-MM-DD>

git log --oneline "${LAST_FERRY_SHA}..HEAD" -- experiments/ lib/ scripts/

gh issue list \
  --label experiment \
  --search "updated:>=${LAST_FERRY_DATE}" \
  --limit 100
```

Notes:
- Treat GitHub-tagged ferry PRs/issues as source of truth.
- If `LAST_FERRY_SHA` is unavailable, use a date fallback and call that out in the PR text.

### 2) Propose bounded ferry edits
Edit `experiments/ferries/daily.py` directly.

Guidelines:
- usually change 1-2 knobs total,
- avoid high-churn edits,
- keep changes reversible,
- allow identical rerun only for explicit regression confirmation.

### 3) Open PR targeting `main`
Include:
- last ferry links,
- exact config delta,
- rationale and risk level (`low`/`medium`/`high`),
- launch checklist:
  - [ ] human approval
  - [ ] issue updated
  - [ ] monitoring loop started

### 4) Launch after explicit approval
Use the canonical launch path:

```bash
uv run lib/marin/src/marin/run/ray_run.py \
  --no_wait \
  --cluster us-central1 \
  -- python experiments/ferries/daily.py
```

Optional deterministic naming for reruns:

```bash
uv run lib/marin/src/marin/run/ray_run.py \
  --no_wait \
  --cluster us-central1 \
  -e FERRY_DATE="$(date +%Y%m%d-%H%M%S)-daily-ferry" \
  -- python experiments/ferries/daily.py
```

Record in the issue:
- Ray job ID,
- cluster,
- launch timestamp,
- W&B link(s) when available.

### 5) Monitor to terminal state
Follow `.agents/docs/job-monitoring-loop.md` with:
- `job_id`,
- `cluster`,
- `experiment=experiments/ferries/daily.py`.

Hard rule:
- do not restart/recreate/mutate cluster without explicit human consent in-thread.

### 6) Close loop in issue
Post:
- final status,
- key metrics and regressions,
- links to PR/job/W&B,
- recommendation for next ferry.

## Recipe Promotion Rule
If a ferry variant is clearly better holistically, promote it as the new baseline recipe/template.

Promotion examples:
- eval losses are broadly better across major validation slices,
- LM eval soft metrics improve in aggregate,
- no obvious reliability regressions (launch, checkpointing, monitoring behavior).

When promotion is warranted:
- open follow-up PR updating `experiments/ferries/daily.py` and this recipe,
- include a concise before/after metric table and tradeoffs.

## Escalation Guidance
- Small obvious launch/config bugs: fix and relaunch.
- Same failure repeats after one fix: escalate.
- Infra-wide instability (node churn, quota/resource contention, repeated TPU runtime failures): escalate immediately.

## Validation Checklist
- Diff in `experiments/ferries/daily.py` is intentional and bounded.
- PR targets `main`.
- Ferry issue has updated launch metadata.
- Monitoring loop is active until terminal state.
- Final issue comment includes links and recommendation.

## See Also
- `.agents/docs/job-monitoring-loop.md`
- `.agents/projects/ferry_framework.md`
- `docs/recipes/agent_research.md`
