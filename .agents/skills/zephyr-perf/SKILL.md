---
name: zephyr-perf
description: Run perf gates on a PR that touches Zephyr internals — submit a treatment ferry on Iris, compare against the latest scheduled baseline run, post the verdict back to the PR. Use when a PR modifies `lib/zephyr/src/zephyr/**` and a reviewer (or label) asks for a perf gate.
---

# Skill: Zephyr Perf Gate

Perf gate for Zephyr-internals PRs. The agent reads the diff, picks a
max-gate from the assessment, submits a treatment ferry, compares against
the latest scheduled run, and posts a single canonical comment to the PR.
The control side of the comparison is always the latest successful
scheduled ferry run on `main` (daily fineweb smoke, weekly nemotron
partial-slice, weekly nemotron full-slice) — the agent never submits a
control ferry of its own.

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
| **2 — nemotron partial-slice** | `experiments.ferries.datakit_nemotron_ferry --stride 5 --sft-general-path <gs://...>` (every 5th file of the medium quality slice + Nemotron-SFT-General as a parquet workload) | weekly `marin-datakit-nemotron-partial-slice-ferry` workflow (Tuesday 01:00 UTC) | ~2–3 h | Mid-tier signal between fineweb and the full medium slice. Exercises *both* read paths — fsspec (JSONL inputs in medium) and pyarrow-native (parquet inputs in SFT-General). The SFT path is caller-supplied so the ferry isn't pinned to a single region; it's verified at runtime and never downloaded. |
| **3 — nemotron full-slice** | `experiments.ferries.datakit_nemotron_ferry` (full medium quality slice, ~3.4 TiB) | weekly `marin-datakit-nemotron-ferry` workflow (Monday 01:00 UTC) | overnight (≤24 h) | Reserved for diffs the reviewer wants tested at the largest slice we run. Expensive; reviewer approval (`zephyr-perf-gate:3`) required. Note: this is the *full medium quality slice*, not all of Nemotron-CC. |

**Gate 1 is always run first**, regardless of `max_gate`. If Gate 1 passes
and `max_gate` is `2` or `3`, escalate. If Gate 1 fails, post the verdict
and stop — no point burning bigger budget on a regression already proven at
small scale. The assessment in step 2 yields a `max_gate`, not a single
chosen gate.

The gate is **not** chosen mechanically from file paths. The agent reads the
diff and judges (see *Assess the diff* below).

Reviewer always overrides via PR labels `zephyr-perf-gate:{skip,1,2,3}`.
The label sets `max_gate`; Gate 1 still runs first.

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
scheduled ferry** on `main`. The scheduled run is the control side of
the comparison — the agent does not submit a separate control ferry.

```bash
# Gate 1 baseline: latest successful daily fineweb smoke.
BASELINE_GATE1_RUN=$(gh run list \
  --repo marin-community/marin \
  --workflow=marin-datakit-smoke.yaml \
  --branch=main --status=success --limit=1 \
  --json databaseId,headSha,createdAt -q '.[0]')

# Gate 2 baseline: latest successful weekly nemotron partial-slice ferry.
BASELINE_GATE2_RUN=$(gh run list \
  --repo marin-community/marin \
  --workflow=marin-datakit-nemotron-partial-slice-ferry.yaml \
  --branch=main --status=success --limit=1 \
  --json databaseId,headSha,createdAt -q '.[0]')

# Gate 3 baseline: latest successful weekly nemotron full-slice ferry.
BASELINE_GATE3_RUN=$(gh run list \
  --repo marin-community/marin \
  --workflow=marin-datakit-nemotron-ferry.yaml \
  --branch=main --status=success --limit=1 \
  --json databaseId,headSha,createdAt -q '.[0]')
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
scheduled run **is** the control side of the comparison — the agent
never submits a control ferry of its own.

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

The `Control` column refers to the latest successful scheduled ferry on
`main` (see "Locate the scheduled baselines" above) — the agent does not
submit a separate control ferry.

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
  ping the reviewer; do not invent numbers. The agent does not submit a
  ferry of its own to fill the gap unless the reviewer explicitly asks.
- **Treatment OOMs at a stage the baseline didn't**: hard fail. Always
  surface the worker-pool death log with the OOM line in the comment so the
  author can act without re-pulling logs.
- **Agent says out of scope but the reviewer disagrees**: reviewer applies
  `zephyr-perf-gate:{1,2,3}`; the agent re-runs with the forced gate.
- **Iris worker preemptions during the run**: spot-VM preemptions inflate
  wall-time, retry counts, and worker-pool churn — the leg may complete but
  the timing is infra noise, not a code regression. Signals: elevated
  `KILLED` worker tasks in `iris rpc controller list-tasks`, frequent
  "worker disconnected"/re-registration lines in the coord log, sudden
  worker-count drops in the stage-progress lines. **Action**: if any stage
  saw >5% of its workers churn during the run, mark the verdict
  `⚠ inconclusive` (not pass/warn/fail), surface the churn count in the
  comment, and re-submit treatment. Do not call a regression on a single
  preempted run.
- **Cluster scheduling delay (queue wait, not pipeline wall-time)**: the
  job sits in `JOB_STATE_PENDING`/`JOB_STATE_BUILDING` for an unusual
  duration before any pipeline stage starts. This is not a perf signal —
  the gate measures stage wall-times from the coord's progress lines, not
  end-to-end submit-to-finish. Note the queue wait in the comment if
  notable (>30 min), but it does not affect the verdict.
- **Cluster contention / mixed worker generations**: another large job
  competing for europe-west4 capacity, or autoscaler bringing up workers
  on a different machine type/zone, can shift baseline timing by 10–30%
  even with no code change. Signals: control and treatment ran far apart
  in time, or the scheduled-baseline run noted unusual cluster state in
  its own logs. **Action**: if the wall-time delta is in the warn band
  (+5–20%) and the contention signal is plausible, mark `⚠ inconclusive`
  rather than `⚠ warn`. Hard-fail thresholds (>+20%, OOMs, failed shards)
  still apply — those are too large to be cluster noise.
- **TPU/CPU bad-node retries in coord**: shard retried on a different
  worker due to hardware fault. Inflates per-stage wall-time. Same handling
  as preemptions — count toward the churn metric and consider re-running
  if pervasive.

## Composes with

- `babysit-zephyr` — for monitoring each run while in flight.
- `babysit-job` — for the outer Iris job lifecycle.
- `debug-infra` — when a leg flakes and the cause is unclear.

