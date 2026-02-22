# Recipe: Ferries (Canary + Daily)

## Overview
Use this recipe to run the two ferry lanes:
- `canary`: fast, low-cost always-on health check
- `daily`: higher-scale integration run with bounded changes

Both ferries should keep core data assumptions aligned and share the same monitoring/triage discipline.

## Ferry Lanes
Templates:
- `experiments/ferries/canary_ferry.py`
- `experiments/ferries/daily.py`

Current intent:
- canary: catch infra/pretraining regressions early with a stable, mostly fixed config
- daily: exercise a larger run envelope and test small, explicit changes

Shared baseline:
- data: shared `nemotron_mix` baseline
- default cluster: `us-central1` (zone `us-central1-a`)
- run log: `docs/experiments/daily-ferry-log.md`

Daily baseline defaults:
- model size: Llama ~150M (`llama_150m`)
- sequence length: 4096
- train batch size: 512
- FLOP target: ~1e19 (overrideable via env)

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
### General
- Hard launch gate: the agent must get explicit requester approval before launching any ferry job.
- Only exception: the requester explicitly says to launch without asking (single run or standing instruction in-thread).
- Always run `.agents/docs/job-monitoring-loop.md` until the run reaches a terminal state (`SUCCEEDED`/`FAILED`/`STOPPED`); do not stop early.
- Expect full ferry monitoring to often take 4-5 hours.
- Never restart/recreate/mutate cluster without explicit human consent in-thread.
- Keep cluster mutation guardrails exactly aligned with `.agents/docs/job-monitoring-loop.md`, including the TPU bad-node exception path.
- Use major-event updates (not spam): launch, first eval, major incident, terminal state.
- Seal each completed daily run with a pushed git tag that points to the exact launch commit.
- Canonical run-closure PR labels:
  - `ferry`
  - `ferry-daily`
  - `ferry-log-only`
  - `ferry-sealed`
- Canonical seal-tag format:
  - daily: `ferry/daily/YYYYMMDD/<run_slug>`

### Canary
- Keep canary stable; only change it for explicit reliability fixes.
- Do not proactively edit canary; only touch it when diagnosing/fixing a concrete failure mode.
- Canary launches usually do not require a PR if the script/config is unchanged.
- If canary fails, treat as urgent infrastructure/training-health triage.
- Canary is run-only by default (W&B + issue updates); no sealing tag or run-closure PR in the normal canary path.

### Daily
- Use daily for bounded evolution, usually 1-2 knobs.
- If daily fails, debug with one bounded fix attempt, then escalate.
- Run-closure PR scope is log-only: update `docs/experiments/daily-ferry-log.md` and keep detailed debug/run narrative in the issue.

## Workflows

### Daily lane (proposal + run)

#### 1) Build context since last ferry
Start by checking the latest entries in `docs/experiments/daily-ferry-log.md`.

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
- Propose at least one intentional modification each interval.
- If there is no obvious change from recent commits/issues/ferry history, use judgment to pick a low-risk tweak
  (for example data-mix adjustment or hyperparameter change) that may improve loss at the same FLOPs budget.
- Pattern-match from the previous daily ferry; avoid high-churn rewrites.
- Update run naming for this interval (for example via `FERRY_DATE` in launch env: `daily-125m-YYYY-MM-DD` style).

#### 3) Record proposal in issue and push launch commit
In the run issue, record:
- last ferry links (issue + commit + W&B/job link),
- exact config delta and rationale,
- risk level (`low`/`medium`/`high`),
- relaunch fallback note (what to try next if this run fails),
- why this run is not literally identical to the previous daily run,
- launch checklist (explicit requester approval or explicit waiver + monitoring started).

Then push the launch commit (no proposal PR by default).

#### 4) Launch
Before launch, confirm requester approval in-thread unless they already gave explicit "launch without asking" permission.

```bash
uv run lib/marin/src/marin/run/ray_run.py \
  --no_wait \
  --cluster us-central1 \
  -- python experiments/ferries/daily.py
```

After launch, capture and post to the issue:
- Ray job id
- cluster
- launch timestamp
- W&B link(s) when available

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
- Follow monitoring-loop restart policy for recoverable failures.
- Escalate non-trivial failures to humans.

#### 6) Close the loop
Post in the ferry issue:
- final status,
- key metrics/regressions,
- Ray job ID and W&B link(s),
- recommendation for next ferry.
- Optional: post a manual Discord update for major run state changes.

For daily-log metric fields, extract canonical final keys with:

```bash
uv run python scripts/ferries/daily_analysis.py \
  --run <wandb_run_url_or_path> \
  --format markdown
```

Required terminal issue comment template:

```markdown
Final status: <SUCCEEDED|FAILED|STOPPED>
Ray job id: <job_id>
W&B link: <url>
Final eval summary: <short summary + key metrics>
Experiment link: <experiment JSON/browser link>
Recommendation / victory decision: <next action>
```

#### 7) Seal and open log-only PR
- Create and push a sealing tag for the exact launch commit (the commit containing the `experiments/ferries/daily.py` used for the run).
- Open a PR that updates only `docs/experiments/daily-ferry-log.md`.
- Keep all detailed launch/retry/debug narrative in the run issue, not in the PR.
- Apply canonical labels on the run-closure PR: `ferry`, `ferry-daily`, `ferry-log-only`, `ferry-sealed`.

### Canary lane (steady-state run)
Default mode: launch the existing canary script as-is and monitor. Do not run the daily proposal/PR loop unless you are intentionally changing canary.
Even for unchanged canary runs, ask the requester before launch unless they explicitly waived that requirement.

Launch:
```bash
uv run lib/marin/src/marin/run/ray_run.py \
  --no_wait \
  --cluster us-central1 \
  -- python experiments/ferries/canary_ferry.py
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
- If daily was edited, launch commit is pushed and referenced in the run issue.
- Run-closure PR only updates `docs/experiments/daily-ferry-log.md`.
- Ferry issue has updated launch metadata.
- Monitoring loop ran until terminal state.

## See Also
- `docs/experiments/daily-ferry-log.md`
- `.agents/docs/job-monitoring-loop.md`
- `.agents/projects/ferry_framework.md`
- `docs/recipes/agent_research.md`
