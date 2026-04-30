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

The agent does **not** open follow-up issues — even on `❌ fail`. The PR
comment is the artifact; the author owns the response (revert, fix, or accept
with rationale).

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

| Gate | Ferry | Wall-time | Notes |
|---|---|---|---|
| **skip** | — | — | All-trivial diff (e.g. comments, docstrings, type hints, renames). Reviewer must concur. |
| **1 — fineweb smoke** | `experiments.ferries.datakit_ferry` (FineWeb-Edu sample/10BT) | ~30–60 min | Default. Cheap; runs scatter, dedup, consolidate, tokenize end-to-end at small scale. |
| **2 — full nemotron** | `experiments.ferries.datakit_nemotron_ferry` (Nemotron-CC medium, ~3.4 TiB) | overnight (≤24 h) | Reach for this when the diff materially affects shuffle, memory, CPU, or zephyr design. Expensive; reviewer approval required. |

The gate is **not** chosen mechanically from file paths. The agent reads the
diff and judges (see *Assess the diff* below). A one-character fix in
`shuffle.py` should not trigger an overnight run; a five-line tweak elsewhere
that flips a buffer size should.

Reviewer always overrides via PR labels `zephyr-perf-gate:skip` /
`zephyr-perf-gate:1` / `zephyr-perf-gate:2`.

A medium tier ("Nemotron 1-slice", ~few hours) is intentionally **not** in this
ladder yet — there is no script for it. Track as future work; for now, in-scope
PRs are Gate 1 by default and escalate to Gate 2 when the assessment flags any
of the impact dimensions.

## Workflow

### 1. Mechanical scope check

```bash
uv run python scripts/zephyr/perf/select_gate.py --pr <PR_NUMBER>
```

Output is JSON:

```json
{
  "in_scope": true,
  "touched_zephyr_files": ["lib/zephyr/src/zephyr/shuffle.py"],
  "touched_fray_files": [],
  "hot_files_touched": ["lib/zephyr/src/zephyr/shuffle.py"],
  "next_step": "Read the diff for the hot files first, ..."
}
```

If `in_scope` is `false`, stop here and post nothing. The script does **not**
choose the gate — go to the next step.

`hot_files_touched` is a *hint* (scatter / planner / executor / sort / spill)
about which files have higher prior likelihood of perf impact. Read those
first; do not treat their presence as an automatic Gate-2 trigger.

### 2. Assess the diff

Read the actual diff, not just the file list:

```bash
gh pr diff <PR_NUMBER>          # PRs
git diff <merge_base>...<head>  # local
```

For each touched zephyr file, answer five yes/no questions and write the
answers to a small JSON file (used later in the PR comment):

| # | Question | Yes if… |
|---|---|---|
| 1 | Trivial? | comment-only, docstring-only, whitespace, rename, pure type-hint, log-string text, dead-code removal with no callers. |
| 2 | Affects shuffle? | scatter pipeline (hashing, fanout, combiner, byte-range sidecar), partitioning, k-way merge, chunk routing. |
| 3 | Affects memory consumption? | buffer sizes, in-memory accumulation, chunk shapes, spill thresholds, retained references in coord/worker, RPC payload size. |
| 4 | Affects CPU utilization? | hot loops, serialization paths, sort/merge inner loops, polling intervals, lock contention, JSON/parquet read/write. |
| 5 | Changes zephyr design in an important way? | new public API, changed actor protocol, changed stage semantics, changed `.result()` ordering, changed retry/error classification, changed plan/fusion rules. |

**Decision:**

- All-trivial (q1 yes for every file, q2–q5 no everywhere) → propose
  `zephyr-perf-gate:skip` and ask the reviewer to confirm before posting.
- Any of q2 / q3 / q4 / q5 = yes anywhere → **Gate 2**.
- Otherwise → **Gate 1**.

Record the answers and the agent's one-line rationale per file:

```json
{
  "gate": "2",
  "rationale": "shuffle.py: changes scatter combiner from per-key to per-shard buffer (memory + CPU)",
  "per_file": {
    "lib/zephyr/src/zephyr/shuffle.py": {
      "trivial": false, "shuffle": true, "memory": true, "cpu": true, "design": false,
      "summary": "scatter combiner buffering changed"
    }
  }
}
```

This file is consumed by `compare_perf_runs.py` (via `--assessment`) so the
posted comment shows the agent's reasoning, not just the timings.

If a `zephyr-perf-gate:skip|1|2` label is set on the PR, that label wins —
record the override in `rationale`.

### 3. Resolve SHAs

```bash
gh pr view <PR_NUMBER> --json headRefOid,baseRefOid,baseRefName -q '...'
git fetch origin <baseRefName>
CONTROL_SHA=$(git merge-base origin/<baseRefName> <headRefOid>)
TREATMENT_SHA=<headRefOid>
```

Re-running control at the PR's merge-base is the default. Reusing a recent
weekly nemotron ferry as control is a future optimization — do not skip control
in v1.

### 4. Set up worktrees

Iris bundles the working directory; submit each run from a worktree at the
right SHA. Use the shared `../.zephyr_perf_worktrees/` directory and a
single timestamp per run so the two legs are clearly paired and easy to
clean up later:

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
WT_DIR="../.zephyr_perf_worktrees"
mkdir -p "$WT_DIR"
CONTROL_WT="$WT_DIR/${PR_NUMBER}-${TS}-control"
TREATMENT_WT="$WT_DIR/${PR_NUMBER}-${TS}-treatment"
git worktree add "$CONTROL_WT"   "$CONTROL_SHA"
git worktree add "$TREATMENT_WT" "$TREATMENT_SHA"
```

Resulting paths look like
`../.zephyr_perf_worktrees/5199-20260430T134522Z-control`. Stale runs from a
prior gate execution can be wiped with
`git worktree remove ../.zephyr_perf_worktrees/${PR_NUMBER}-*`.

### 5. Submit both ferries

```bash
uv run python scripts/zephyr/perf/submit_perf_run.py \
  --gate 1 \
  --label control \
  --cwd "$CONTROL_WT" \
  --pr <PR_NUMBER> \
  --status-out gs://marin-us-central1/tmp/ttl=7d/zephyr-perf/<PR>/control.json

uv run python scripts/zephyr/perf/submit_perf_run.py \
  --gate 1 \
  --label treatment \
  --cwd "$TREATMENT_WT" \
  --pr <PR_NUMBER> \
  --status-out gs://marin-us-central1/tmp/ttl=7d/zephyr-perf/<PR>/treatment.json
```

The script prints the Iris job ID and writes the status JSON path to stdout.

### 6. Babysit

Both runs babysat with the **babysit-zephyr** skill (or **babysit-job** for the
outer Iris job). Do not poll in a tight loop — Gate 1 is ~30–60 min, Gate 2 is
overnight. Sleep at least 10 min between checks for Gate 1, 30 min for Gate 2.

If a run fails (worker pool wedged, coordinator zombie), escalate to
**debug-infra** for triage and re-submit the failed leg only after the
underlying issue is understood. Never silently retry — a flaky run masks a real
regression.

### 7. Collect metrics

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

### 8. Compare and verdict

```bash
uv run python scripts/zephyr/perf/compare_perf_runs.py \
  --control /tmp/zephyr-perf/<PR>/control_metrics.json \
  --treatment /tmp/zephyr-perf/<PR>/treatment_metrics.json \
  --assessment /tmp/zephyr-perf/<PR>/assessment.json \
  --gate 1 \
  --markdown-out /tmp/zephyr-perf/<PR>/comment.md \
  --verdict-out /tmp/zephyr-perf/<PR>/verdict.json
```

The assessment JSON from step 2 is included in the rendered comment so the
reviewer sees both the verdict (timings + OOMs) and the agent's reasoning
(what the diff was judged to affect).

Default thresholds (overridable via `--thresholds <yaml>`):

| Signal | Warn | Hard fail |
|---|---|---|
| Per-stage wall-time delta vs control | > +5% | > +10% |
| New OOMs | any | — |
| New failed shards | — | any |
| Total wall-time delta | > +5% | > +10% |

Verdict precedence: any hard-fail → `❌ fail`; otherwise any warn → `⚠ warn`;
otherwise `✅ pass`.

### 9. Post one canonical comment

The comment is sentinel-marked so re-runs replace the prior comment instead of
stacking. Two `gh api` calls — find the existing comment, then patch or post:

```bash
PR=<PR_NUMBER>
REPO=marin-community/marin
BODY=/tmp/zephyr-perf/$PR/comment.md
EXISTING=$(gh api --paginate "repos/$REPO/issues/$PR/comments" \
  --jq '.[] | select(.body | startswith("<!-- zephyr-perf-gate -->")) | .id' | head -1)

if [ -n "$EXISTING" ]; then
  gh api --method PATCH "repos/$REPO/issues/comments/$EXISTING" -F "body=@$BODY"
else
  gh api --method POST  "repos/$REPO/issues/$PR/comments"      -F "body=@$BODY"
fi
```

The comment is the only output — no separate issue is filed on `❌ fail`. The
author decides next steps (revert, fix, or accept with rationale).

### 10. Clean up

```bash
git worktree remove "$CONTROL_WT"
git worktree remove "$TREATMENT_WT"
```

To wipe stale worktrees from earlier runs:

```bash
for wt in ../.zephyr_perf_worktrees/${PR_NUMBER}-*; do
  git worktree remove --force "$wt"
done
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
  problem. Post a comment on the PR explaining the flake and ping the reviewer;
  do not post a pass/fail verdict and do not file an issue.
- **Treatment OOMs at a stage that control survives**: hard fail. Always
  surface the worker-pool death log with the OOM line in the comment so the
  author can act without re-pulling logs.
- **`select_gate.py` says out-of-scope but the reviewer disagrees**: reviewer
  applies `zephyr-perf-gate:1` or `zephyr-perf-gate:2` label; the agent re-runs
  with the forced gate.

## Composes with

- `babysit-zephyr` — for monitoring each run while in flight.
- `babysit-job` — for the outer Iris job lifecycle.
- `debug-infra` — when a leg flakes and the cause is unclear.

## Open questions

- Should weekly scheduled nemotron ferries auto-feed control results so PRs can
  skip the control re-run? (Currently no — re-run for safety.)
- Add a Gate 1.5 scale-stress lane (10k workers, minimal RAM) to surface
  scatter OOMs and controller back-pressure faster than Gate 2? Not built yet.
