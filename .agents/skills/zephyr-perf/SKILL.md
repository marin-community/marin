---
name: zephyr-perf
description: Run an A/B perf gate on a PR that touches Zephyr internals — submit control + treatment ferries on Iris, compare metrics, post the verdict back to the PR. Use when a PR modifies `lib/zephyr/src/zephyr/**` and a reviewer (or label) asks for a perf gate.
---

# Skill: Zephyr Perf Gate

A/B perf gate for Zephyr-internals PRs. The agent picks a gate based on touched
paths, submits the same ferry on the PR's merge-base (control) and PR head
(treatment), compares metrics, and posts a single canonical comment to the PR.

This skill **only** triggers on changes to Zephyr internals
(`lib/zephyr/src/zephyr/**`). Datakit / dedup / normalize / tokenize live in
`lib/marin/...` and are explicitly out of scope — they consume Zephyr but are
not Zephyr core. If a PR touches both, run this skill on the Zephyr part and
let the datakit smoke / nemotron ferry workflows cover the rest.

## Autonomy

The agent may, without asking:

- Read the PR diff and decide gate / scope via `select_gate.py`.
- Create temporary git worktrees at the PR's merge-base and head SHAs.
- Submit Iris ferry jobs at production priority for both runs.
- Poll job state and pull coordinator logs.
- Post **one** canonical comment on the PR (sentinel-marked, idempotent).
- Open a regression issue when the verdict is `❌ fail` (label
  `zephyr-perf-regression` + `agent-generated`).

The agent must ask before:

- Promoting a PR from Gate 1 to Gate 2 against `select_gate.py`'s recommendation
  (Gate 2 is overnight and expensive).
- Re-running on a different cluster than `lib/iris/examples/marin.yaml`.
- Stopping a ferry that has not crossed its gate timeout.

## Trigger / Scope

Run this skill when:

1. A PR's diff has at least one file matching the Zephyr-internals globs in
   `select_gate.py` (canonical list: `lib/zephyr/src/zephyr/**/*.py`,
   `lib/zephyr/pyproject.toml`), AND
2. None of those touched files are test-only (`lib/zephyr/tests/**`,
   `lib/zephyr/src/zephyr/_test_helpers.py`).

Out of scope (do **not** trigger):

- `lib/marin/src/marin/processing/classification/deduplication/**` (dedup)
- `lib/marin/src/marin/datakit/normalize/**` (normalize)
- `lib/marin/src/marin/processing/tokenize/**` (tokenize)
- `lib/fray/**` (execution backend — flag it in the PR comment but don't
  auto-gate; ask the reviewer)
- Any docs-only diff (`*.md`, `lib/zephyr/AGENTS.md`, `lib/zephyr/OPS.md`)

When unsure, run `select_gate.py` and trust its `in_scope` field.

## Gate ladder

| Gate | Ferry | Wall-time | Default for | Notes |
|---|---|---|---|---|
| **1 — fineweb smoke** | `experiments.ferries.datakit_ferry` (FineWeb-Edu sample/10BT) | ~30–60 min | every in-scope PR | Cheap; runs scatter, dedup, consolidate, tokenize end-to-end at small scale. |
| **2 — full nemotron** | `experiments.ferries.datakit_nemotron_ferry` (Nemotron-CC medium, ~3.4 TiB) | overnight (≤24 h) | PRs touching the high-blast-radius set below | Expensive; requires reviewer approval. |

**Gate 2 trigger paths** (any one of these in the diff escalates to Gate 2):

- `lib/zephyr/src/zephyr/shuffle.py` (scatter pipeline)
- `lib/zephyr/src/zephyr/plan.py` (operation fusion)
- `lib/zephyr/src/zephyr/execution.py` (coord/worker loop)
- `lib/zephyr/src/zephyr/external_sort.py` (k-way merge)
- `lib/zephyr/src/zephyr/spill.py` (on-disk spill)

Reviewer can promote/demote with PR labels `perf-gate:1` / `perf-gate:2` /
`perf-gate:skip`.

A medium tier ("Nemotron 1-slice", ~few hours) is intentionally **not** in this
ladder yet — there is no script for it. Track as future work; for now, in-scope
PRs go to Gate 1 by default and escalate straight to Gate 2 when the path set
fires.

## Workflow

### 1. Decide scope and gate

```bash
uv run python scripts/zephyr/perf/select_gate.py --pr <PR_NUMBER>
```

Output is JSON:

```json
{"in_scope": true, "gate": "1", "reason": "...", "touched_zephyr_files": [...]}
```

If `in_scope` is `false`, stop here and post nothing. If `gate` is `"2"`,
confirm with the reviewer (or check for the `perf-gate:2` label) before
launching.

### 2. Resolve SHAs

```bash
gh pr view <PR_NUMBER> --json headRefOid,baseRefOid,baseRefName -q '...'
git fetch origin <baseRefName>
CONTROL_SHA=$(git merge-base origin/<baseRefName> <headRefOid>)
TREATMENT_SHA=<headRefOid>
```

Re-running control at the PR's merge-base is the default. Reusing a recent
weekly nemotron ferry as control is a future optimization — do not skip control
in v1.

### 3. Set up worktrees

Iris bundles the working directory; submit each run from a worktree at the
right SHA.

```bash
git worktree add ../zephyr-perf-control "$CONTROL_SHA"
git worktree add ../zephyr-perf-treatment "$TREATMENT_SHA"
```

### 4. Submit both ferries

```bash
uv run python scripts/zephyr/perf/submit_perf_run.py \
  --gate 1 \
  --label control \
  --cwd ../zephyr-perf-control \
  --pr <PR_NUMBER> \
  --status-out gs://marin-us-central1/tmp/ttl=7d/zephyr-perf/<PR>/control.json

uv run python scripts/zephyr/perf/submit_perf_run.py \
  --gate 1 \
  --label treatment \
  --cwd ../zephyr-perf-treatment \
  --pr <PR_NUMBER> \
  --status-out gs://marin-us-central1/tmp/ttl=7d/zephyr-perf/<PR>/treatment.json
```

The script prints the Iris job ID and writes the status JSON path to stdout.

### 5. Babysit

Both runs babysat with the **babysit-zephyr** skill (or **babysit-job** for the
outer Iris job). Do not poll in a tight loop — Gate 1 is ~30–60 min, Gate 2 is
overnight. Sleep at least 10 min between checks for Gate 1, 30 min for Gate 2.

If a run fails (worker pool wedged, coordinator zombie), escalate to
**debug-infra** for triage and re-submit the failed leg only after the
underlying issue is understood. Never silently retry — a flaky run masks a real
regression.

### 6. Collect metrics

```bash
uv run python scripts/zephyr/perf/collect_perf_metrics.py \
  --status gs://marin-us-central1/tmp/ttl=7d/zephyr-perf/<PR>/control.json \
  --out /tmp/zephyr-perf/<PR>/control_metrics.json

uv run python scripts/zephyr/perf/collect_perf_metrics.py \
  --status gs://marin-us-central1/tmp/ttl=7d/zephyr-perf/<PR>/treatment.json \
  --out /tmp/zephyr-perf/<PR>/treatment_metrics.json
```

The collector pulls per-stage wall-times from the coordinator's progress logs,
the final counter snapshot via `iris actor call`, and the worker-pool death
count from `iris rpc controller list-tasks`. Only post the comment after both
runs reach a terminal state (`SUCCEEDED` or `FAILED`).

### 7. Compare and verdict

```bash
uv run python scripts/zephyr/perf/compare_perf_runs.py \
  --control /tmp/zephyr-perf/<PR>/control_metrics.json \
  --treatment /tmp/zephyr-perf/<PR>/treatment_metrics.json \
  --gate 1 \
  --markdown-out /tmp/zephyr-perf/<PR>/comment.md \
  --verdict-out /tmp/zephyr-perf/<PR>/verdict.json
```

Default thresholds (overridable via `--thresholds <yaml>`):

| Signal | Warn | Hard fail |
|---|---|---|
| Per-stage wall-time delta vs control | > +5% | > +10% |
| New OOMs | any | — |
| New failed shards | — | any |
| Total wall-time delta | > +5% | > +10% |

Verdict precedence: any hard-fail → `❌ fail`; otherwise any warn → `⚠ warn`;
otherwise `✅ pass`.

### 8. Post one canonical comment

```bash
uv run python scripts/zephyr/perf/post_pr_comment.py \
  --pr <PR_NUMBER> \
  --body /tmp/zephyr-perf/<PR>/comment.md
```

The script upserts a comment marked with `<!-- zephyr-perf-gate -->` so re-runs
replace the prior comment instead of stacking. Open a regression issue when the
verdict is `❌ fail`:

```bash
gh issue create \
  --title "[zephyr-perf] regression on #<PR_NUMBER>" \
  --label zephyr-perf-regression --label agent-generated \
  --body-file /tmp/zephyr-perf/<PR>/comment.md
```

### 9. Clean up

```bash
git worktree remove ../zephyr-perf-control
git worktree remove ../zephyr-perf-treatment
```

## Comment format (canonical)

```markdown
<!-- zephyr-perf-gate -->
🤖 ## Zephyr perf gate — Gate 1 (fineweb)

**Verdict:** ✅ pass | ⚠ warn | ❌ fail

| | Control | Treatment |
|---|---|---|
| SHA | `abc1234` | `def5678` |
| Iris job | [`...`](...) | [`...`](...) |
| W&B | [link](...) | [link](...) |
| Total wall-time | 31m 12s | 32m 04s (+2.8%) |

### Stage timings

| Stage | Control (s) | Treatment (s) | Δ | Verdict |
|---|---|---|---|---|
| download | 12 | 12 | +0% | ✅ |
| normalize | 845 | 870 | +3.0% | ✅ |
| minhash | 410 | 421 | +2.7% | ✅ |
| fuzzy_dups | 188 | 192 | +2.1% | ✅ |
| consolidate | 220 | 224 | +1.8% | ✅ |
| tokenize | 197 | 205 | +4.1% | ✅ |

### Workers

| | Control | Treatment |
|---|---|---|
| OOMs | 0 | 0 |
| Failed shards | 0 | 0 |
| Peak worker memory (MB) | 14202 | 14180 |

<details><summary>Counters</summary>

(zephyr counter deltas, JSON)

</details>
```

The comment **must** begin with `🤖` per repo convention (see
`/AGENTS.md` → "Communication & Commits").

## Failure modes

- **Control flakes, treatment passes**: re-submit control; do not call the gate
  pass on a single run.
- **Both flake at the same stage**: not a regression — a real environmental
  problem. Open an `agent-generated` issue and ping the reviewer; do not post a
  pass/fail verdict.
- **Treatment OOMs at a stage that control survives**: hard fail. Always
  surface the worker-pool death log with the OOM line in the comment so the
  author can act without re-pulling logs.
- **`select_gate.py` says out-of-scope but the reviewer disagrees**: reviewer
  applies `perf-gate:1` or `perf-gate:2` label; the agent re-runs with the
  forced gate.

## Composes with

- `babysit-zephyr` — for monitoring each run while in flight.
- `babysit-job` — for the outer Iris job lifecycle.
- `debug-infra` — when a leg flakes and the cause is unclear.
- `file-issue` — for the regression issue path.

## Open questions

- Should weekly scheduled nemotron ferries auto-feed control results so PRs can
  skip the control re-run? (Currently no — re-run for safety.)
- Add a Gate 1.5 scale-stress lane (10k workers, minimal RAM) to surface
  scatter OOMs and controller back-pressure faster than Gate 2? Not built yet.
