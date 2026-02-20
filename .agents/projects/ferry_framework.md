# Ferry Framework Project Plan (Daily 125M Integration Ferry)

## Objective

Build a repeatable ferry workflow that launches a daily ~125M-parameter pretraining run (~1e19 model FLOPs target), primarily as a high-signal integration test for TPU training infrastructure.
This is the initial target; we plan to expand to broader ferry scales/cadences once the daily integration loop is stable.

The workflow should support:
- proposing the next ferry as a PR based on the previous ferry + recent repo changes/issues
- bounded config evolution (not identical every day, but not high-risk churn)
- explicit requester approval before every launch, unless requester explicitly waives "ask before launch"
- monitored execution to completion
- status propagation to GitHub (Discord automation in Phase 2)

This plan maps to `.agents/the_plan.yaml` items:
- `ferry_framework`
- `automated_daily_ferry_launch`
- `experiment_updates_bot`
- `experiment_updates_github_to_discord`

## Scope (Phase 1)

Focus only on **daily ferry as integration test**:
- single canonical ferry template (`experiments/ferries/daily.py`)
- keep daily and canary ferry data-mix assumptions aligned (`experiments/ferries/canary.py`)
- one PR-generation workflow (agent/manual-triggered)
- one launch flow on TPU infra
- one monitoring loop using `.agents/docs/job-monitoring-loop.md`
- optional manual Discord update (automation deferred)

Out of scope for this phase:
- weekly/monthly automation
- Discord automation
- autonomous Discord->GitHub summarization
- large strategy search over many simultaneous hypotheses

## Deliverables

1. Recipe doc in `docs/recipes/`:
- `docs/recipes/ferries.md`
  - end-to-end human+agent ferry procedure
  - required inputs, safety gates, commands, and escalation paths

2. Canonical ferry experiment:
- `experiments/ferries/daily.py`
  - stable baseline structure
  - date-stamped run naming
  - clearly marked "agent edit surface" (data mix, lr/schedule, optimizer, infra toggles)

3. Ferry metadata source of truth:
- GitHub tagged ferry PRs/issues are the canonical source of truth.
  - each ferry PR should carry a ferry tag/label and link to the run issue
  - ferry issue should contain launch metadata (job id, cluster, links, outcome)
  - local metadata files are optional and non-canonical

4. Proposal workflow (markdown-first):
- Start with recipe-driven agent workflow (no required script initially).
  - reads last ferry metadata + recent commits/issues
  - proposes a constrained edit to `experiments/ferries/daily.py`
  - emits PR notes (what changed + why + risk level)
- Optional follow-up: add `scripts/pm/ferries/propose_daily_ferry.py` if manual repetition becomes expensive.

5. Launch + monitor helpers:
- `scripts/pm/ferries/launch_daily_ferry.py` (or equivalent command wrapper)
- recipe-driven monitoring using `.agents/docs/job-monitoring-loop.md`

6. Notification bridge (minimum viable):
- `scripts/pm/ferries/notify_discord.py` (or GitHub Action webhook)
  - posts "ferry started/finished/failed" with links
  - can later be replaced by `experiment_updates_github_to_discord`
  - explicitly deferred to Phase 2

## Operating Model

### Daily loop (human-assisted, agent-executed)

1. Build context window:
- find last ferry tagged PR/issue/commit in GitHub (canonical record)
- gather since-last-ferry commits relevant to experiments/infra
- gather issues updated since last ferry with experiment-related labels

2. Generate next ferry proposal:
- pattern-match from previous `experiments/ferries/daily.py`
- apply bounded changes (see "Change Budget")
- update run name with date (`daily-125m-YYYY-MM-DD`)

3. Open PR:
- include rationale, risk estimate, and rollback plan
- include "why this is not identical to prior run"
- target `main`

4. Human gate:
- explicit requester yes/no approval to launch every run
- exception: requester explicitly authorizes launching without asking first
- likely communicated inside the active agent session for now (not necessarily via GitHub review state)

5. Launch on TPU cluster:
- verify the requester approval gate is satisfied
- run the canonical launch command
- capture job id + links into issue/PR

6. Monitor to completion:
- execute `.agents/docs/job-monitoring-loop.md`
- keep monitoring active until terminal job state (`SUCCEEDED`/`FAILED`/`STOPPED`)
- expect this loop to run for 4-5 hours for typical ferry runs
- auto-restart only per documented loop policy
- escalate non-trivial failures to humans

7. Publish updates:
- GitHub issue comment with status and key links
- optional manual Discord post for run state

### Change Budget Policy (Daily Integration Ferry)

Default policy (integration-test-first):
- change at least 1, at most 2 knobs per day
- keep model size + high-level topology stable unless explicitly requested
- prioritize infra-sensitive changes (launcher, data pipeline path, logging, checkpointing) over architecture churn

Allowed no-change exception:
- identical rerun is permitted if the goal is explicit regression confirmation after infra incidents

Suggested change buckets:
- data source/mix minor revision
- optimizer/hyperparameter micro-adjustment
- model architecture adjustments
- instrumentation additions

## Decision Procedure for "What Changes Today?"

For each interval:
- Inputs:
  - last ferry config + outcome
  - commits since last run
  - experiment-labeled issues updated since last run
  - explicit human guidance for today's objective
- Rank candidate changes by:
  - integration-test value
  - blast radius
  - reversibility
  - expected signal in one run
- Pick smallest set with positive signal and bounded risk.

Pseudo-policy:

```python
candidates = gather_candidates(commits, issues, human_guidance)
scored = score(candidates, value, risk, reversibility, observability)
selected = choose_with_budget(scored, min_changes=1, max_changes=2)
if regression_investigation:
    selected = []  # allow identical rerun with explicit reason
```

## Recipe Promotion Rule

When a ferry variant is clearly better in a holistic sense, we should promote it as the new default recipe baseline.

Promotion signals (examples):
- most (or nearly all) key eval losses improve versus the prior baseline ferry
- LM eval soft metrics improve in aggregate
- no obvious regression in integration reliability (launch stability, checkpointing, monitoring behavior)

Operationally:
- open a follow-up PR that updates the canonical recipe/template (`experiments/ferries/daily.py` and recipe docs) to match the better variant
- include a short comparison table in the PR showing before/after metrics and any tradeoffs

## Suggested Command Skeletons

Context collection:

```bash
# Last ferry commits in the relevant area
LAST_FERRY_SHA=<last_ferry_commit_sha>
LAST_FERRY_DATE=<YYYY-MM-DD>
git log --oneline "${LAST_FERRY_SHA}..HEAD" -- experiments/ lib/ scripts/

# Experiment-labeled issues updated recently (requires gh auth)
gh issue list \
  --label experiment \
  --search "updated:>=${LAST_FERRY_DATE}" \
  --limit 100
```

Launch shape (illustrative, to pin in recipe):

```bash
uv run lib/marin/src/marin/run/ray_run.py \
  --no_wait \
  --cluster us-central1 \
  -- python experiments/ferries/daily.py --run_name "daily-125m-$(date +%F)"
```

Monitoring handoff:
- follow `.agents/docs/job-monitoring-loop.md` with:
  - `job_id`
  - `cluster`
  - `experiment=experiments/ferries/daily.py`

## PR Template Requirements (for Ferry Proposals)

Every ferry PR should include:
- last ferry reference (issue + commit + run link)
- summary of interval changes considered
- exact config delta (before/after)
- risk level (`low`/`medium`/`high`)
- launch checklist:
  - [ ] explicit requester approval (or explicit waiver recorded in-thread)
  - [ ] issue created/updated
  - [ ] (optional/manual) Discord update posted
  - [ ] monitoring loop started

## Failure and Escalation Policy

- Simple launch/config bugs: agent may patch and relaunch.
- Repeated identical failure after one fix attempt: escalate to human.
- Infra-wide issues (cluster unhealthy, quota, persistent OOM): escalate immediately.
- Keep all failure notes in the canonical ferry issue.

## Discord Integration Unit of Work

Phase-1 expectation:
- deferred (no Discord automation in Phase 1)

Phase-2:
- migrate to `experiment_updates_github_to_discord` path in `.agents/the_plan.yaml`
- thread-per-issue routing and dedupe

## Implementation Phases

1. **Recipe + Template**
- add `docs/recipes/ferries.md`
- establish `experiments/ferries/daily.py`

2. **Proposal Workflow**
- run the markdown recipe in an agent session to generate deterministic, reviewable diffs and PR body text
- optionally add `scripts/pm/ferries/propose_daily_ferry.py` later if repetition cost justifies it

3. **Launch + Monitoring**
- add launch wrapper and enforce monitoring loop handoff
- standardize state capture (job id, links, timestamps)

4. **Discord Notifications (Phase 2)**
- add minimal notifier + documented webhook setup
- connect launch and completion events

5. **Scheduled Daily Trigger**
- add cron/automation entry to open daily proposal PR
- retain explicit requester approval before launch (unless explicitly waived in-thread)

## Success Criteria

- A new ferry proposal PR can be generated daily in under 10 minutes of agent time.
- Daily ferry run is launchable from one documented path.
- Each run has issue + PR + run link traceability.
- Monitoring loop is consistently used until terminal state.
- Monitoring ownership is maintained for the full run duration (often 4-5 hours), not handed off early.
- Discord automation is intentionally deferred to Phase 2.

## Resolved Decisions

1. Daily ferry proposal PRs target `main`.
2. Default cluster for now is `us-central1` (Ray CLI cluster key; maps to zone `us-central1-a`).
3. "Experiment-relevant issues" filter starts with label `experiment` only.
4. "Max 2 knobs changed" remains policy guidance, not script-enforced.
5. Discord automation is deferred to Phase 2.
