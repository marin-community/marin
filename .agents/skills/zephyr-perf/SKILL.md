---
name: zephyr-perf
description: Run perf gates on a PR that touches Zephyr internals — submit a treatment ferry on Iris, compare against the latest scheduled tier-N baseline run, post the verdict back to the PR. Use when a PR modifies `lib/zephyr/src/zephyr/**` and a reviewer asks for a perf gate.
---

# Skill: Zephyr Perf Gate

Perf gate for Zephyr-internals PRs. The agent reads the diff, picks a
max-gate from the assessment, submits a treatment ferry, fetches the
matching scheduled-baseline perf report from its workflow artifact, reads
both JSON reports, applies the threshold table below, and posts a single
canonical comment to the PR.

The control side of the comparison is always the latest successful scheduled
`marin-canary-datakit-tier<N>` workflow run on `main`. Each tier's
`Capture perf report` step writes a `datakit-tier<N>-perf-report` workflow
artifact (retained 90 days) and mirrors the same JSON to
`gs://marin-us-central1/infra/datakit/ferry_perf/`. The agent never submits
a baseline ferry of its own.

This skill **only** triggers on changes to Zephyr internals
(`lib/zephyr/src/zephyr/**`). Datakit / dedup / normalize / tokenize live in
`lib/marin/...` and are explicitly out of scope — they consume Zephyr but
are not Zephyr core. If a PR touches both, run this skill on the Zephyr
part and let the datakit canary workflows cover the rest.

## Autonomy

The agent may, without asking:

- Read the PR diff and decide scope + max-gate (see *Assess the diff*).
- Create a temporary git worktree at the PR head SHA.
- Submit Iris ferry jobs at the same priority used by the matching tier
  workflow.
- Poll job state and pull coordinator logs.
- Post **one** canonical comment on the PR (sentinel-marked, idempotent).
- Escalate up the gate ladder (1 → 2 → 3) when the prior gate passes and
  `max_gate` allows.

The agent does **not** open follow-up issues — even on `❌ fail`. The PR
comment is the artifact; the author owns the response (revert, fix, or
accept with rationale).

The agent must ask before:

- Re-running on a different cluster than `lib/iris/examples/marin.yaml`.
- Stopping a ferry that has not crossed its tier wall-time.

## Trigger / Scope

Run this skill when:

1. A PR's diff has at least one non-test, non-docs file under
   `lib/zephyr/src/zephyr/**` (or `lib/zephyr/pyproject.toml`), AND
2. The reviewer asked for a perf gate (comment or @-mention).

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

| Gate | Tier workflow | Schedule | Ferry / coverage | Wall-time |
|---|---|---|---|---|
| **skip** | — | — | — | All-trivial diff (e.g. comments, docstrings, type hints, renames). Reviewer must concur. |
| **1 — smoke** | `marin-canary-datakit-tier1.yaml` | daily 06:30 UTC | `experiments.ferries.datakit_ferry` (FineWeb-Edu sample/10BT). Typical web text, end-to-end pass at small scale. | ~30–60 min |
| **2 — long-tail stress** | `marin-canary-datakit-tier2.yaml` | daily 07:00 UTC | `experiments.ferries.datakit_tier2_skewed_ferry`. Synthetic skewed doc-length distribution (log-normal mean ~5 KB body + Pareto tail + ~100 mega-docs in [128 MB, 256 MB]) — exercises spill, scatter, and consolidate under buffer pressure. | ~2.5 h |
| **3 — nemotron** | `marin-canary-datakit-tier3.yaml` | weekly Mon 01:00 UTC | `experiments.ferries.datakit_nemotron_ferry` with `quality=high` and `max_files=1000`. Bulk filtered web at production-realistic scale within the GH 6h cap. Runs in europe-west4, non-preemptible. | ~3 h |

**Gate 1 is always run first**, regardless of `max_gate`. If Gate 1 passes
and `max_gate >= 2`, escalate to Gate 2; if Gate 2 passes and `max_gate >= 3`,
escalate to Gate 3. If any gate fails, post the verdict and stop — no
point burning bigger budget on a regression already proven at smaller
scale.

The gate is **not** chosen mechanically from file paths. The agent reads
the diff, judges (see *Assess the diff* below), and **confirms the
chosen `max_gate` with the reviewer before submitting any ferry**. The
reviewer can override with a different `max_gate` (or `skip`) in the
confirmation reply. There are no PR labels for this — the confirmation
is a chat exchange in the session that invoked the skill.

**Baseline freshness:** tier1/tier2 baselines are <24h old. The tier3
baseline can be up to a week old (weekly schedule). Surface the baseline
age in the comment so the reviewer can sanity-check the comparison.

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
identify which files most likely matter (scatter / planner / executor /
sort / spill historically have the highest prior probability of perf
impact) — but this is judgment, not a path-glob rule.

**Decision (`max_gate`, not a single chosen gate — Gate 1 always runs first):**

- All-trivial (q1 yes for every file, q2–q5 no everywhere) → propose
  `max_gate = "skip"`.
- Any of q2 / q3 / q4 / q5 = yes anywhere → `max_gate = "3"`.
- Otherwise → `max_gate = "1"`.

Record the answers and the agent's one-line rationale per file:

```json
{
  "max_gate": "3",
  "rationale": "shuffle.py: changes scatter combiner from per-key to per-shard buffer (memory + CPU)",
  "per_file": {
    "lib/zephyr/src/zephyr/shuffle.py": {
      "trivial": false, "shuffle": true, "memory": true, "cpu": true, "design": false,
      "summary": "scatter combiner buffering changed"
    }
  }
}
```

The agent renders this assessment as a small table in the final PR
comment so reviewers see the agent's reasoning, not just the timings.

### 1a. Confirm `max_gate` with the reviewer

Before submitting any ferry, the agent posts the assessment back to the
reviewer (in the chat session that invoked the skill — **not** as a PR
comment) and waits for confirmation. Format the message tightly:

```
🤖 Zephyr perf gate assessment

Proposed max_gate: <skip|1|2|3>
Rationale: <one-line summary>

Per-file:
- <path>: <one-line summary>
- ...

Reply "go" to run, or override with "max_gate=<skip|1|2|3>".
```

If the reviewer overrides, record the override in the assessment JSON's
`rationale` field (`"reviewer override: max_gate=2 — only minhash
sensitivity matters"`). Then proceed.

If the reviewer says `skip`, post a one-line PR comment explaining the
gate was waived and stop. Do **not** submit any ferry.

### 2. Locate the scheduled baseline for each gate you'll run

Each gate compares its treatment run against the **latest successful
scheduled tier-N run** on `main`. The scheduled run's `Capture perf
report` step uploads a `datakit-tier<N>-perf-report` workflow artifact
that is the file we read.

```bash
# Pick the tier you need; repeat per gate.
RUN_ID_TIER1=$(gh run list --repo marin-community/marin \
  --workflow=marin-canary-datakit-tier1.yaml \
  --branch=main --status=success --limit=1 \
  --json databaseId,headSha,createdAt -q '.[0]')

RUN_ID_TIER2=$(gh run list --repo marin-community/marin \
  --workflow=marin-canary-datakit-tier2.yaml \
  --branch=main --status=success --limit=1 \
  --json databaseId,headSha,createdAt -q '.[0]')

RUN_ID_TIER3=$(gh run list --repo marin-community/marin \
  --workflow=marin-canary-datakit-tier3.yaml \
  --branch=main --status=success --limit=1 \
  --json databaseId,headSha,createdAt -q '.[0]')
```

The actual artifact download happens inside each gate (step 5d). Capture
the run id, head SHA, and `createdAt` here so you can render baseline
provenance in the comment.

### 3. Set up the treatment worktree

Iris bundles the working directory; submit the treatment from a worktree
at the PR head.

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

Before paying for ferries, confirm the treatment compiles, type-checks,
and passes the zephyr unit/integration suite. A broken test is much
cheaper to catch here than after a Gate 1 ferry.

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
comment using the same sentinel as the verdict (so re-runs upsert in
place):

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

### 5. Run a gate (Gate 1 always; Gates 2 / 3 conditional on prior pass)

The same protocol applies to all three gates; substitute the tier-N
specifics where indicated.

**a. Submit the treatment ferry.**

```bash
mkdir -p /tmp/zephyr-perf/<PR>
uv run python scripts/datakit/submit_perf_run.py \
  --gate <N> --pr <PR_NUMBER> --cwd "$TREATMENT_WT" \
  --status-out gs://marin-us-central1/tmp/ttl=7d/zephyr-perf/<PR>/treatment-g<N>.json \
  > /tmp/zephyr-perf/<PR>/submit-g<N>.json
TREATMENT_JOB_ID=$(jq -r .job_id < /tmp/zephyr-perf/<PR>/submit-g<N>.json)
```

`submit_perf_run.py` mirrors the iris CLI shape used by the tier-N
workflow YAML (region, memory, disk, cpu, priority, preemptibility,
extra env vars). Drift between this script and the tier YAML breaks
parity — keep them in lockstep.

**b. Babysit until terminal.** Delegate to **babysit-zephyr** (or
**babysit-job** for the outer Iris job). Don't poll in a tight loop —
sleep ≥ 10 min between checks for Gate 1, ≥ 15 min for Gate 2,
≥ 20 min for Gate 3. If the leg flakes (worker pool wedged, coord
zombie), escalate to **debug-infra**; do not silently retry — a flaky
run masks a real regression.

**c. Collect the treatment perf report.**

Use the same script the scheduled workflows use, so the JSON is
structurally identical to the baseline:

```bash
uv run python scripts/datakit/collect_perf_metrics.py \
  --job-id "$TREATMENT_JOB_ID" \
  --status gs://marin-us-central1/tmp/ttl=7d/zephyr-perf/<PR>/treatment-g<N>.json \
  --out /tmp/zephyr-perf/<PR>/treatment-g<N>-perf-report.json
```

**d. Pull the baseline perf report.**

```bash
RUN_ID=$(gh run list --repo marin-community/marin \
  --workflow=marin-canary-datakit-tier<N>.yaml \
  --branch=main --status=success --limit=1 \
  --json databaseId -q '.[0].databaseId')
mkdir -p /tmp/zephyr-perf/<PR>/baseline-g<N>
gh run download "$RUN_ID" --name datakit-tier<N>-perf-report \
  --dir /tmp/zephyr-perf/<PR>/baseline-g<N>
# → /tmp/zephyr-perf/<PR>/baseline-g<N>/perf-report.json
```

If the artifact is missing or unreadable (rare — 90d retention covers
any plausible gate run), fall back to the GCS mirror at
`gs://marin-us-central1/infra/datakit/ferry_perf/report_*_tier<N>/perf_report.json`
and `gsutil cp` it locally.

**e. Compare and write the verdict.**

The agent reads both JSONs (baseline and treatment perf reports) and
writes the verdict comment **by hand** following the threshold table
below. There is no separate compare script — judgment about cached
steps, multi-attempt churn, and infra noise lives in the agent.

#### Threshold table (apply per gate)

`wall_seconds_total` is the **launcher-task** wall time (max
`duration_ms` across the launcher's own tasks, in seconds). It does
**not** include time the job spent in `JOB_STATE_PENDING` /
`JOB_STATE_BUILDING` waiting for cluster capacity — that queue wait
isn't a perf signal and doesn't count against the verdict (see *Failure
modes* below). `stage_wall_seconds` is the actual pipeline-step work,
derived from the iris job tree.

| Signal | ✅ Pass | ⚠ Warn | ❌ Hard fail |
|---|---|---|---|
| `wall_seconds_total` delta (treatment − baseline) / baseline | ≤ +5% | +5–10% | > +10% |
| Per-step `stage_wall_seconds` delta (any stage) | ≤ +5% | +5–10% | > +10% |
| Any new entry in `infra_failures` (treatment > baseline in any bucket: `oom`, `hardware_fault`, `scheduling_timeout`, `application_failure`, `other`) | — | — | any |
| `failed_shards` strictly higher in treatment | — | — | any |
| `peak_worker_memory_mb` delta | ≤ +5% | +5–15% | > +15% |

#### Inconclusive (infra noise, not a code regression)

Mark the verdict `⚠ inconclusive` (not pass / warn / fail) and re-submit
the treatment when **any** of the following holds:

- `treatment.preemption_count` is materially higher than
  `baseline.preemption_count` (e.g. > 3 over baseline, or > 0 when
  baseline is 0). Stage durations split across attempts are not
  comparable.
- `treatment.task_state_counts.preempted > 0`.
- `treatment.infra_failures.hardware_fault > 0` (TPU/CPU bad-node
  retry).
- The treatment ran on a visibly different cluster generation than the
  baseline (e.g. autoscaler bumped the worker machine type) — surface
  in the comment if known.

Do **not** call a regression on a single preempted or hardware-flaky run.

#### Cached steps

If a step appears in either report's `cached_steps` list, its
`stage_wall_seconds[step]` is `0.0` and the delta is meaningless.
Render "—" in the per-step table; do not count toward the verdict.
Note the cache hit in a footnote so the reviewer knows that stage
wasn't measured.

#### Agent self-check

Before writing the comment, the agent should walk both JSONs and
sanity-check:

- Treatment and baseline `ferry_module` match (otherwise something is
  mis-wired).
- `iris_job_id` differs (otherwise we're comparing a run to itself).
- `task_state_counts` totals roughly equal between the two (large
  divergence usually means one side did less work — flag it).

#### Comment shape (canonical)

The comment **must** begin with the sentinel and `🤖` per repo
convention.

```markdown
<!-- zephyr-perf-gate -->
🤖 ## Zephyr perf gate — Gate <N> (<tier name>)

**Verdict:** ✅ pass | ⚠ warn | ⚠ inconclusive | ❌ fail

**Baseline:** scheduled run [#<RUN_ID>](<url>), sha=`<sha>`, age=<N>d

**Hard fails:** … (omit if none)
**Warns:** … (omit if none)

### Diff assessment

(rendered from the assessment JSON in step 1: per-file table with the
five yes/no answers + one-line summary, plus the agent's overall
rationale)

### Run summary

| | Baseline | Treatment |
|---|---|---|
| Iris job | `<id>` | `<id>` |
| Status | succeeded | succeeded |
| Total wall-time | 31m 12s | 32m 04s (+2.8%) |
| Peak worker memory (MB) | 14202 | 14180 |

### Stage timings

| Stage | Baseline | Treatment | Δ | Verdict |
|---|---|---|---|---|
| download | 12s | 12s | +0% | ✅ |
| normalize | 14m 05s | 14m 30s | +3.0% | ✅ |
| minhash | 6m 50s | 7m 01s | +2.7% | ✅ |
| fuzzy_dups | 3m 08s | 3m 12s | +2.1% | ✅ |
| consolidate | 3m 40s | 3m 44s | +1.8% | ✅ |
| tokenize | 3m 17s | 3m 25s | +4.1% | ✅ |

### Infra

| | Baseline | Treatment |
|---|---|---|
| Preemptions | 0 | 0 |
| Failed shards | 0 | 0 |
| Infra failures | (none) | (none) |
| Task states | succeeded=42 | succeeded=42 |

<details><summary>Raw treatment report</summary>

(JSON contents of /tmp/zephyr-perf/<PR>/treatment-g<N>-perf-report.json)

</details>
```

If the gate returns `❌ fail`, **stop** — post the verdict and don't
escalate. The regression is proven; no point in burning Gate 2 or 3
budget.

### 6. Post one canonical comment

The comment is sentinel-marked so re-runs replace the prior comment
instead of stacking. Two `gh api` calls — find the existing comment,
then patch or post:

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

The comment is the only output — no separate issue is filed on `❌ fail`.
The author decides next steps.

### 7. Clean up

```bash
git worktree remove "$TREATMENT_WT"
```

To wipe stale worktrees from earlier runs:

```bash
shopt -s nullglob
for wt in ../.zephyr_perf_worktrees/${PR_NUMBER}-*; do
  git worktree remove --force "$wt"
done
shopt -u nullglob
```

## Failure modes

- **Treatment flakes**: re-submit treatment; do not call the gate based
  on a single failed run. If it flakes again, escalate to **debug-infra**.
- **Baseline artifact missing or unreadable**: try the GCS mirror at
  `gs://marin-us-central1/infra/datakit/ferry_perf/report_*_tier<N>/perf_report.json`.
  If both are unreachable, post a comment explaining the gap and ping the
  reviewer; do not submit a baseline ferry of your own to fill it.
- **Treatment OOMs at a stage the baseline didn't**: hard fail.
  `treatment.infra_failures.oom > baseline.infra_failures.oom` is
  enough — surface the worker-pool death log with the OOM line in the
  comment so the author can act without re-pulling logs.
- **Agent says out of scope but the reviewer disagrees**: the reviewer
  re-invokes the skill with an explicit `max_gate` override in the
  confirmation reply (step 1a); the agent re-runs at the forced gate.
- **Iris worker preemptions during the run**: spot-VM preemptions
  inflate wall-time, retry counts, and worker-pool churn. Signals:
  `preemption_count` and `task_state_counts.preempted` materially
  higher than baseline. **Action**: mark verdict `⚠ inconclusive`,
  surface the churn in the comment, re-submit treatment. Do not call
  a regression on a single preempted run.
- **Cluster scheduling delay (queue wait, not pipeline wall-time)**:
  the job sits in `JOB_STATE_PENDING`/`JOB_STATE_BUILDING` before any
  pipeline stage starts. This is not a perf signal — the gate measures
  stage wall-times, not end-to-end submit-to-finish. Note the queue
  wait in the comment if notable (>30 min); does not affect the
  verdict.
- **Cluster contention / mixed worker generations**: another large job
  competing for capacity, or the autoscaler bringing up workers on a
  different machine type/zone, can shift baseline timing by 10–30%
  even with no code change. **Action**: if the wall-time delta is in
  the warn band and the contention signal is plausible, mark
  `⚠ inconclusive` rather than `⚠ warn`. Hard-fail thresholds (>+10%,
  new infra failures) still apply — those are too large to be cluster
  noise.
- **TPU/CPU bad-node retries**: surface as
  `infra_failures.hardware_fault`. Same handling as preemptions —
  count toward churn and consider re-running if pervasive.
- **Cached steps in baseline but not treatment (or vice versa)**: a
  step's `stage_wall_seconds` is `0.0` and the step is in
  `cached_steps` when its output already existed in `gs://`. Surface
  "—" for delta; do not penalize. Note in a footnote.
- **Stale tier3 baseline (>1 week old)**: tier3 runs weekly. If the
  latest successful tier3 run on `main` is older than a week (e.g. it
  failed last week), surface the age prominently in the comment so
  the reviewer can decide whether to trust the comparison or wait for
  a fresh baseline.

## Composes with

- `babysit-zephyr` — for monitoring each run while in flight.
- `babysit-job` — for the outer Iris job lifecycle.
- `debug-infra` — when a leg flakes and the cause is unclear.
