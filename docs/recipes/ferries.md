# Recipe: Ferries (Canary + Daily)

## Overview
Use this recipe to run the two ferry lanes:
- `canary`: fast, low-cost always-on health check
- `daily`: higher-scale integration run with bounded changes

Both ferries should keep core data assumptions aligned and share the same monitoring/triage discipline.

## Ferry Lanes
Templates:
- `experiments/ferries/canary.py`
- `experiments/ferries/daily.py`

Current intent:
- canary: catch infra/pretraining regressions early with a stable, mostly fixed config
- daily: exercise a larger run envelope and test small, explicit changes

Shared baseline:
- data: shared `nemotron_mix` baseline
- default cluster: `us-central1` (zone `us-central1-a`)

Daily baseline defaults:
- model size: Llama ~125M (`llama_150m`)
- sequence length: 4096
- train batch size: 512
- FLOP target: ~1e19 (overrideable via env)

## Prerequisites
- Local checkout with write access.
- `uv` installed.
- `gh` authenticated for issue/PR workflows.
- Access to `lib/marin/src/marin/run/ray_run.py`.
- A canonical ferry issue and prior run/PR history.

## Inputs Before Proposing (Daily Only)
This section is for the daily lane. Canary runs normally do not require a proposal cycle or PR.

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

## Operating Policy
- Hard launch gate: the agent must get explicit requester approval before launching any ferry job.
- Only exception: the requester explicitly says to launch without asking (single run or standing instruction in-thread).
- Always run `.agents/docs/job-monitoring-loop.md` until the run reaches a terminal state (`SUCCEEDED`/`FAILED`/`STOPPED`); do not stop early.
- Expect full ferry monitoring to often take 4-5 hours.
- Keep canary stable; only change it for explicit reliability fixes.
- Do not proactively edit canary; only touch it when diagnosing/fixing a concrete failure mode.
- Canary launches usually do not require a PR if the script/config is unchanged.
- Use daily for bounded evolution, usually 1-2 knobs.
- If canary fails, treat as urgent infrastructure/training-health triage.
- If daily fails, debug with one bounded fix attempt, then escalate.
- Never restart/recreate/mutate cluster without explicit human consent in-thread.

## Workflows

### Daily lane (proposal + run)

#### 1) Build context since last ferry
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
- Use "since last ferry run" rather than fixed wall-clock boundaries.

#### 2) Edit `experiments/ferries/daily.py`
- Keep edits bounded (typically 1-2 knobs).

#### 3) Open PR targeting `main`
Include:
- last ferry links (issue + PR/commit + W&B/job link),
- exact config delta and rationale,
- risk level (`low`/`medium`/`high`),
- launch checklist (explicit requester approval or explicit waiver + monitoring started).

#### 4) Launch
Before launch, confirm requester approval in-thread unless they already gave explicit "launch without asking" permission.

```bash
uv run lib/marin/src/marin/run/ray_run.py \
  --no_wait \
  --cluster us-central1 \
  -- python experiments/ferries/daily.py
```

Optional deterministic daily rerun name:
```bash
uv run lib/marin/src/marin/run/ray_run.py \
  --no_wait \
  --cluster us-central1 \
  -e FERRY_DATE="$(date +%Y%m%d-%H%M%S)-daily-ferry" \
  -- python experiments/ferries/daily.py
```

#### 5) Monitor to terminal state
Follow `.agents/docs/job-monitoring-loop.md` with:
- `job_id`
- `cluster`
- `experiment=<ferry script path>`
- Keep the monitoring loop active until terminal status; ferry runs commonly take 4-5 hours.

#### 6) Close the loop
Post in the ferry issue:
- final status,
- key metrics/regressions,
- Ray job ID and W&B link(s),
- recommendation for next ferry.

### Canary lane (steady-state run)
Default mode: launch the existing canary script as-is and monitor. Do not run the daily proposal/PR loop unless you are intentionally changing canary.
Even for unchanged canary runs, ask the requester before launch unless they explicitly waived that requirement.

Launch:
```bash
uv run lib/marin/src/marin/run/ray_run.py \
  --no_wait \
  --cluster us-central1 \
  -- python experiments/ferries/canary.py
```

If canary fails:
- triage and identify root cause,
- only then open a focused PR if a canary script/config change is necessary,
- relaunch and monitor to terminal state.

## Promotion Rule (Daily)
If a daily variant is clearly better holistically, promote it as the new default daily recipe/template.

Promotion signals:
- eval losses are broadly better,
- LM eval soft metrics improve in aggregate,
- no reliability regressions.

When promoting:
- open a follow-up PR updating `experiments/ferries/daily.py` and this recipe,
- include a concise before/after metrics table.

## Validation Checklist
- Diff is intentional and bounded for the selected lane.
- If daily was edited, PR targets `main`.
- Ferry issue has updated launch metadata.
- Monitoring loop ran until terminal state.

## See Also
- `.agents/docs/job-monitoring-loop.md`
- `.agents/projects/ferry_framework.md`
- `docs/recipes/agent_research.md`
