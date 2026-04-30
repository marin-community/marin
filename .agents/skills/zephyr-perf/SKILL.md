---
name: zephyr-perf
description: Run an A/B perf gate on a PR that touches Zephyr internals — submit control + treatment ferries on Iris, compare metrics, post the verdict back to the PR. Use when a PR modifies `lib/zephyr/src/zephyr/**` and a reviewer (or label) asks for a perf gate.
---

# Skill: Zephyr Perf Gate

A/B perf gate for Zephyr-internals PRs. The agent reads the diff, picks a
max-gate from the assessment, submits the treatment ferry, compares against
the latest scheduled run, and posts a single canonical comment to the PR.
Scheduled ferry runs (daily fineweb smoke, weekly nemotron) are the golden
baseline — the agent does not re-run a fresh control.

This skill **only** triggers on changes to Zephyr internals
(`lib/zephyr/src/zephyr/**`). Datakit / dedup / normalize / tokenize live in
`lib/marin/...` and are explicitly out of scope — they consume Zephyr but are
not Zephyr core. If a PR touches both, run this skill on the Zephyr part and
let the datakit smoke / nemotron ferry workflows cover the rest.

## Autonomy

The agent may, without asking:

- Read the PR diff and decide scope + max-gate (see *Assess the diff*).
- Create a temporary git worktree at the PR head SHA.
- Submit Iris ferry jobs at production priority.
- Poll job state and pull coordinator logs.
- Post **one** canonical comment on the PR (sentinel-marked, idempotent).

The agent does **not** open follow-up issues — even on `❌ fail`. The PR
comment is the artifact; the author owns the response (revert, fix, or accept
with rationale).

The agent must ask before:

- Escalating to Gate 3 (overnight nemotron). Reviewer label
  `zephyr-perf-gate:3` is the explicit consent.
- Re-running on a different cluster than `lib/iris/examples/marin.yaml`.
- Stopping a ferry that has not crossed its gate timeout.

## Trigger / Scope

Run this skill when:

1. A PR's diff has at least one non-test, non-docs file under
   `lib/zephyr/src/zephyr/**` (or `lib/zephyr/pyproject.toml`), AND
2. The reviewer asked for a perf gate (label, comment, or @-mention).

Out of scope (do **not** trigger):

- `lib/marin/src/marin/processing/classification/deduplication/**` (dedup)
- `lib/marin/src/marin/datakit/normalize/**` (normalize)
- `lib/marin/src/marin/processing/tokenize/**` (tokenize)
- `lib/fray/**` (execution backend — flag it in the PR comment but don't
  auto-gate; ask the reviewer)
- Any docs-only diff (`*.md`, `lib/zephyr/AGENTS.md`, `lib/zephyr/OPS.md`)
- Test-only changes (`lib/zephyr/tests/**`)

The agent makes this scope call by reading the diff. There is no path-glob
script — when in doubt, ask the reviewer.

## Gate ladder

| Gate | Ferry | Scheduled baseline | Wall-time | Notes |
|---|---|---|---|---|
| **skip** | — | — | — | All-trivial diff (e.g. comments, docstrings, type hints, renames). Reviewer must concur. |
| **1 — fineweb smoke** | `experiments.ferries.datakit_ferry` (FineWeb-Edu sample/10BT) | daily `marin-datakit-smoke` workflow | ~30–60 min | Cheap end-to-end pass at small scale. |
| **2 — nemotron 1-slice** | `experiments.ferries.datakit_nemotron_ferry --stride 5` (every 5th file of nemotron-medium) | TBD — needs a scheduled workflow at the same stride | ~2–3 h | Mid-tier signal between fineweb and full medium. Surfaces scale-only effects (memory pressure, OOMs, fanout shape) without paying for an overnight run. |
| **3 — full nemotron** | `experiments.ferries.datakit_nemotron_ferry` (full nemotron-medium, ~3.4 TiB) | weekly `marin-datakit-nemotron-ferry` workflow | overnight (≤24 h) | Reserved for diffs the reviewer wants tested at full scale. Expensive; reviewer approval (`zephyr-perf-gate:3`) required. |

**Gate 1 is always run first**, regardless of `max_gate`. If Gate 1 passes
and `max_gate` is `2` or `3`, escalate. If Gate 1 fails, post the verdict
and stop — no point burning bigger budget on a regression already proven at
small scale. The assessment in step 2 yields a `max_gate`, not a single
chosen gate.

The gate is **not** chosen mechanically from file paths. The agent reads the
diff and judges (see *Assess the diff* below).

Reviewer always overrides via PR labels `zephyr-perf-gate:{skip,1,2,3}`.
The label sets `max_gate`; Gate 1 still runs first.

Sizing note for Gate 2 (`--stride 5`): from one fineweb run (~46 min total)
and a partial full-medium run (~10–14 h extrapolated), 1/5 of medium lands
near 2.5 h. Recalibrate the stride after the first real Gate 2 run if it
falls outside [2 h, 3.5 h].

## Workflow

### 1. Assess the diff

Read the actual diff:

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

The agent should also use `lib/zephyr/AGENTS.md` and the diff context to
identify which files most likely matter (scatter / planner / executor / sort
/ spill historically have the highest prior probability of perf impact) —
but this is judgment, not a path-glob rule.

**Decision (`max_gate`, not a single chosen gate — Gate 1 always runs first):**

- All-trivial (q1 yes for every file, q2–q5 no everywhere) → propose
  `zephyr-perf-gate:skip` and ask the reviewer to confirm before posting.
- Any of q2 / q3 / q4 / q5 = yes anywhere → `max_gate = "2"` (1-slice
  nemotron). The reviewer can promote to `max_gate = "3"` (full overnight
  nemotron) by applying the `zephyr-perf-gate:3` label.
- Otherwise → `max_gate = "1"`.

Record the answers and the agent's one-line rationale per file:

```json
{
  "max_gate": "2",
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

If a `zephyr-perf-gate:{skip,1,2,3}` label is set on the PR, that label wins
— record the override in `rationale`. The label sets `max_gate`; Gate 1
still runs first when `max_gate >= 2`.

### 2. Locate the scheduled baselines

Each gate compares its treatment run against the **latest successful
scheduled ferry** on `main`. Scheduled ferries are the golden baseline; the
agent does not re-run a fresh control.

```bash
# Gate 1 baseline: latest successful daily fineweb smoke.
BASELINE_GATE1_RUN=$(gh run list \
  --repo marin-community/marin \
  --workflow=marin-datakit-smoke.yaml \
  --branch=main --status=success --limit=1 \
  --json databaseId,headSha,createdAt -q '.[0]')

# Gate 3 baseline: latest successful weekly full-nemotron ferry.
BASELINE_GATE3_RUN=$(gh run list \
  --repo marin-community/marin \
  --workflow=marin-datakit-nemotron-ferry.yaml \
  --branch=main --status=success --limit=1 \
  --json databaseId,headSha,createdAt -q '.[0]')

# Gate 2 baseline: TBD — needs a scheduled `--stride 5` workflow. Until
# that exists, fall back to either:
#   (a) the most recent successful Gate 2 run from a prior PR (search by
#       run-id pattern `zephyr-perf-pr*-g2-*`), or
#   (b) a one-off `--stride 5` run kicked off on `main` ahead of the gate.
# Track in the open questions section.
```

The Iris job id and W&B run for each scheduled run are in its workflow logs
(`gh run view --log <run_id>`). Pass them to `collect_perf_metrics.py` in
step 5.

### 3. Set up the treatment worktree

Iris bundles the working directory; submit the treatment from a worktree at
the PR head.

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
WT_DIR="../.zephyr_perf_worktrees"
mkdir -p "$WT_DIR"
TREATMENT_WT="$WT_DIR/${PR_NUMBER}-${TS}-treatment"

gh pr view <PR_NUMBER> --json headRefOid -q .headRefOid > /tmp/pr-head
TREATMENT_SHA=$(cat /tmp/pr-head)
git worktree add "$TREATMENT_WT" "$TREATMENT_SHA"
```

Stale runs from a prior gate execution can be wiped with
`git worktree remove ../.zephyr_perf_worktrees/${PR_NUMBER}-*`.

### 4. Run zephyr tests on the treatment worktree

Before paying for ferries, confirm the treatment compiles, type-checks, and
passes the zephyr unit/integration suite. A broken test is much cheaper to
catch here than after a 30-minute Gate 1 (or an overnight Gate 2).

```bash
( cd "$TREATMENT_WT" && \
  ./infra/pre-commit.py lib/zephyr/ && \
  uv run pyrefly && \
  uv run pytest lib/zephyr/tests/ )
```

Treatment-only — there is no control worktree. CI is assumed green on
`main`; if it isn't, the broken commit is upstream of the gate's concerns
and the agent should call that out separately.

If any of these fail, **stop here**. Do not submit ferries. Post a halt
comment using the same sentinel as the verdict (so re-runs upsert in place):

```bash
PR=<PR_NUMBER>
REPO=marin-community/marin
BODY=$(mktemp)
cat > "$BODY" <<EOF
<!-- zephyr-perf-gate -->
🤖 ## Zephyr perf gate — halted (local tests failed)

Treatment worktree (\`$TREATMENT_SHA\`) failed lint / pyrefly / zephyr tests.
Ferries were not submitted.

Fix the failing tests and the gate will re-run.
EOF
EXISTING=$(gh api --paginate "repos/$REPO/issues/$PR/comments" \
  --jq '.[] | select(.body | startswith("<!-- zephyr-perf-gate -->")) | .id' | head -1)
if [ -n "$EXISTING" ]; then
  gh api --method PATCH "repos/$REPO/issues/comments/$EXISTING" -F "body=@$BODY"
else
  gh api --method POST  "repos/$REPO/issues/$PR/comments"      -F "body=@$BODY"
fi
```

### 5. Run Gate 1 (always — even when `max_gate >= 2`)

Run treatment, compare against the latest scheduled fineweb smoke. The
scheduled run **is** the control; we never re-run a fresh control.

**a. Submit treatment.**

```bash
uv run python scripts/zephyr/perf/submit_perf_run.py \
  --gate 1 --label treatment --cwd "$TREATMENT_WT" \
  --pr <PR_NUMBER> \
  --status-out gs://marin-us-central1/tmp/ttl=7d/zephyr-perf/<PR>/treatment-g1.json
```

**b. Babysit until terminal.** Delegate to **babysit-zephyr** (or
**babysit-job** for the outer Iris job). Don't poll in a tight loop — sleep
≥ 10 min between checks. If the leg flakes (worker pool wedged, coord
zombie), escalate to **debug-infra**; do not silently retry — a flaky run
masks a real regression.

**c. Collect treatment + baseline metrics.**

```bash
uv run python scripts/zephyr/perf/collect_perf_metrics.py \
  --status gs://marin-us-central1/tmp/ttl=7d/zephyr-perf/<PR>/treatment-g1.json \
  --job-id <treatment_iris_job_id> \
  --out /tmp/zephyr-perf/<PR>/treatment-g1.json

SCHED_JOB_ID=$(gh run view --log <BASELINE_GATE1_RUN_ID> | grep -oE 'iris-run-[a-z0-9-]+' | head -1)
uv run python scripts/zephyr/perf/collect_perf_metrics.py \
  --job-id "$SCHED_JOB_ID" \
  --out /tmp/zephyr-perf/<PR>/baseline-g1.json
```

**d. Compare and produce verdict.**

```bash
uv run python scripts/zephyr/perf/compare_perf_runs.py \
  --control /tmp/zephyr-perf/<PR>/baseline-g1.json \
  --treatment /tmp/zephyr-perf/<PR>/treatment-g1.json \
  --assessment /tmp/zephyr-perf/<PR>/assessment.json \
  --gate 1 \
  --markdown-out /tmp/zephyr-perf/<PR>/comment.md \
  --verdict-out /tmp/zephyr-perf/<PR>/verdict.json
```

Thresholds (vs scheduled baseline — looser than an in-pair re-run, because
the baseline ran days ago on a different SHA on the same cluster):

| Signal | Warn | Hard fail |
|---|---|---|
| Total wall-time delta | > +10% | > +20% |
| Per-stage wall-time delta | > +10% | > +20% |
| New OOMs in treatment | any | — |
| New failed shards in treatment | — | any |

The rendered comment notes the baseline source
(`baseline = scheduled run #<id>, sha=<sha>, age=<days>`) so the reviewer
can sanity-check the comparison.

If Gate 1 returns `❌ fail`, **stop** — post the verdict and don't
escalate. The regression is proven; no point in burning Gate 2 or 3
budget.

### 6. Escalate to Gate 2 / Gate 3 (only if Gate 1 passed *and* `max_gate >= 2`)

Same protocol as step 5, with two changes:
- Submit with `--gate 2` (or `--gate 3` if the reviewer applied
  `zephyr-perf-gate:3`).
- Baseline = `$BASELINE_GATE2_RUN` for Gate 2, `$BASELINE_GATE3_RUN` for
  Gate 3.

Gate 2 sleeps ≥ 30 min between babysit checks; Gate 3 ≥ 60 min. The same
threshold table as step 5 applies.

If Gate 2 fails, do not escalate to Gate 3 even when the label is set —
Gate 2 already proved the regression. The reviewer can manually request
Gate 3 if they want full-scale data on a non-regressing change.

### 7. Post one canonical comment

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

### 8. Clean up

```bash
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

- **Treatment flakes**: re-submit treatment; do not call the gate based on
  a single failed run. If it flakes again, escalate to **debug-infra**.
- **Scheduled baseline is missing or unparseable** (Iris log retention,
  expired status JSON, parsing error): post a comment explaining the gap and
  ping the reviewer; do not invent numbers and do not run a fresh control
  unless explicitly asked.
- **Treatment OOMs at a stage the baseline didn't**: hard fail. Always
  surface the worker-pool death log with the OOM line in the comment so the
  author can act without re-pulling logs.
- **Agent says out of scope but the reviewer disagrees**: reviewer applies
  `zephyr-perf-gate:{1,2,3}`; the agent re-runs with the forced gate.

## Composes with

- `babysit-zephyr` — for monitoring each run while in flight.
- `babysit-job` — for the outer Iris job lifecycle.
- `debug-infra` — when a leg flakes and the cause is unclear.

## Open questions

- Gate 2 needs its own scheduled CI workflow (`marin-datakit-nemotron-ferry`
  with `--stride 5`, on a daily/2-day cadence) to provide a baseline. Until
  it exists, Gate 2 falls back to ad-hoc baselines (see step 2).
- Calibrate the Gate 2 stride after the first real run lands. Target wall
  ~2.5h; adjust between `--stride 4` and `--stride 6` if the actual run
  falls outside [2h, 3.5h].
- Should the agent compute median-of-N scheduled runs instead of "latest
  successful" to stabilize the baseline?
- Add a high-fanout / low-RAM stress lane (10k workers, minimal RAM) to
  surface scatter OOMs and controller back-pressure faster than Gate 2?
  Not built yet.
