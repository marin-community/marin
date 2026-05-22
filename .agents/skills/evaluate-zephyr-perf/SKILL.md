---
name: evaluate-zephyr-perf
description: Run a perf gate on a PR that touches lib/zephyr internals.
---

# Skill: Zephyr Perf Gate

Perf gate for Zephyr-internals PRs. The agent reads the diff, picks a `max_gate`,
submits a treatment ferry, fetches the matching scheduled-baseline perf report,
compares the two JSON reports against the threshold table, and posts one
canonical comment to the PR.

The control side is always the latest successful scheduled
`marin-canary-datakit-tier<N>` workflow run on `main`. Each tier's `Capture perf
report` step uploads a `datakit-tier<N>-perf-report` workflow artifact (90-day
retention) and mirrors the same JSON to
`gs://marin-us-central1/infra/datakit/ferry_perf/`. The agent never submits a
baseline ferry of its own.

Triggers **only** on changes to Zephyr internals (`lib/zephyr/src/zephyr/**`).
Datakit / dedup / normalize / tokenize live in `lib/marin/...` and are out of
scope. If a PR touches both, gate the Zephyr part; the datakit canaries cover
the rest.

## Autonomy

The agent may, without asking:

- Read the PR diff and decide scope + `max_gate`.
- Create a temporary git worktree at the PR head SHA.
- Submit Iris ferry jobs at the same priority as the matching tier workflow.
- Poll job state and pull coordinator logs.
- Post **one** canonical comment on the PR (sentinel-marked, idempotent).
- Escalate up the gate ladder (1 → 2 → 3) when the prior gate passes and
  `max_gate` allows.

The agent does **not** open follow-up issues — even on `❌ fail`. The PR comment
is the artifact; the author owns the response.

Ask before:

- Re-running on a different cluster than `lib/iris/config/marin.yaml`.
- Stopping a ferry that has not crossed its tier wall-time.

## Trigger / Scope

Run when both hold:

1. The PR diff has at least one non-test, non-docs file under
   `lib/zephyr/src/zephyr/**` (or `lib/zephyr/pyproject.toml`), AND
2. The reviewer asked for a perf gate (comment or @-mention).

Out of scope (do **not** trigger):

- `lib/marin/src/marin/processing/classification/deduplication/**` (dedup)
- `lib/marin/src/marin/datakit/normalize/**` (normalize)
- `lib/marin/src/marin/processing/tokenize/**` (tokenize)
- `lib/fray/**` — flag it in the PR comment but don't auto-gate; ask the reviewer
- Docs-only diffs (`*.md`, `lib/zephyr/AGENTS.md`, `lib/zephyr/OPS.md`)
- Test-only changes (`lib/zephyr/tests/**`)

There is no path-glob script — the agent makes the scope call from the diff.
When in doubt, ask the reviewer.

## Gate ladder

| Gate | Tier workflow | Schedule | Ferry / coverage | Wall-time |
|---|---|---|---|---|
| **skip** | — | — | All-trivial diff (comments, docstrings, type hints, renames). Reviewer must concur. | — |
| **1 — smoke** | `marin-canary-datakit-tier1.yaml` | daily 06:30 UTC | `experiments.ferries.datakit_ferry` (FineWeb-Edu sample/10BT). End-to-end pass at small scale. | ~30–60 min |
| **2 — long-tail stress** | `marin-canary-datakit-tier2.yaml` | daily 07:00 UTC | `experiments.ferries.datakit_tier2_skewed_ferry`. Synthetic skewed doc-length distribution (log-normal mean ~5 KB body + Pareto tail + ~100 mega-docs in [128 MB, 256 MB]) — exercises spill, scatter, consolidate under buffer pressure. | ~2.5 h |
| **3 — nemotron** | `marin-canary-datakit-tier3.yaml` | weekly Mon 01:00 UTC | `experiments.ferries.datakit_nemotron_ferry` with `quality=high`, `max_files=1000`. Production-scale bulk filtered web within the GH 6h cap. Runs in europe-west4, non-preemptible. | ~3 h |

**Gate 1 always runs first**, regardless of `max_gate`. If Gate 1 passes and
`max_gate >= 2`, escalate to Gate 2; if Gate 2 passes and `max_gate >= 3`,
escalate to Gate 3. If any gate fails, post the verdict and stop — no point
burning bigger budget on a regression already proven at smaller scale.

The gate is **not** chosen mechanically from file paths. The agent reads the
diff, judges (see *Assess the diff*), and **confirms `max_gate` with the
reviewer before submitting any ferry**. The reviewer can override with a
different `max_gate` (or `skip`) in the confirmation reply. There are no PR
labels — confirmation is a chat exchange in the invoking session.

**Baseline freshness:** tier1/tier2 baselines are <24h old; the tier3 baseline
can be up to a week old (weekly schedule). Surface the baseline age in the
comment.

## Workflow

### 1. Assess the diff

```bash
gh pr diff <PR_NUMBER>          # PRs
git diff <merge_base>...<head>  # local
```

For each touched zephyr file, answer five yes/no questions:

| # | Question | Yes if… |
|---|---|---|
| 1 | Trivial? | comment/docstring/whitespace-only, rename, pure type-hint, log-string text, dead-code removal with no callers. |
| 2 | Affects shuffle? | scatter pipeline (hashing, fanout, combiner, byte-range sidecar), partitioning, k-way merge, chunk routing. |
| 3 | Affects memory? | buffer sizes, in-memory accumulation, chunk shapes, spill thresholds, retained references, RPC payload size. |
| 4 | Affects CPU? | hot loops, serialization paths, sort/merge inner loops, polling intervals, lock contention, JSON/parquet read/write. |
| 5 | Changes zephyr design? | new public API, changed actor protocol, changed stage semantics, changed `.result()` ordering, changed retry/error classification, changed plan/fusion rules. |

Use `lib/zephyr/AGENTS.md` and diff context to identify files most likely to
matter (scatter / planner / executor / sort / spill have the highest prior of
perf impact) — judgment, not a path-glob rule.

**Decision (`max_gate` — Gate 1 always runs first regardless):**

- All-trivial (q1 yes everywhere, q2–q5 no everywhere) → `max_gate = "skip"`.
- Any q2 / q3 / q4 / q5 = yes anywhere → `max_gate = "3"`.
- Otherwise → `max_gate = "1"`.

Record the answers and a one-line rationale per file:

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

Render this assessment as a small table in the final PR comment.

### 1a. Confirm `max_gate` with the reviewer

Before submitting any ferry, post the assessment to the reviewer in the invoking
chat session (**not** as a PR comment) and wait for confirmation:

```
🤖 Zephyr perf gate assessment

Proposed max_gate: <skip|1|2|3>
Rationale: <one-line summary>

Per-file:
- <path>: <one-line summary>
- ...

Reply "go" to run, or override with "max_gate=<skip|1|2|3>".
```

On override, record it in the assessment JSON `rationale` field (e.g.
`"reviewer override: max_gate=2 — only minhash sensitivity matters"`), then
proceed. If the reviewer says `skip`, post a one-line PR comment that the gate
was waived and stop — no ferry.

### 2. Locate the scheduled baseline

Each gate compares against the **latest successful scheduled tier-N run** on
`main`. Capture run id, head SHA, and `createdAt` for comment provenance; the
artifact download happens inside each gate (step 5d).

```bash
gh run list --repo marin-community/marin \
  --workflow=marin-canary-datakit-tier<N>.yaml \
  --branch=main --status=success --limit=1 \
  --json databaseId,headSha,createdAt -q '.[0]'
```

### 3. Set up the treatment worktree

Iris bundles the working directory; submit the treatment from a worktree at the
PR head.

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
WT_DIR="../.zephyr_perf_worktrees"
mkdir -p "$WT_DIR"
TREATMENT_WT="$WT_DIR/${PR_NUMBER}-${TS}-treatment"

gh pr view <PR_NUMBER> --json headRefOid -q .headRefOid > /tmp/pr-head
TREATMENT_SHA=$(cat /tmp/pr-head)
git worktree add "$TREATMENT_WT" "$TREATMENT_SHA"
```

### 4. Run zephyr tests on the treatment worktree

Confirm the treatment compiles, type-checks, and passes the zephyr suite before
paying for ferries:

```bash
( cd "$TREATMENT_WT" && \
  ./infra/pre-commit.py lib/zephyr/ && \
  uv run pyrefly && \
  uv run pytest lib/zephyr/tests/ )
```

Treatment-only — there is no control worktree. CI is assumed green on `main`; if
it isn't, the broken commit is upstream of the gate's concerns — call that out
separately.

If any of these fail, **stop here**. Do not submit ferries. Post a halt comment
using the same sentinel as the verdict (so re-runs upsert in place):

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

### 5. Run a gate (Gate 1 always; Gates 2/3 conditional on prior pass)

Same protocol for all three gates; substitute the tier-N specifics.

**a. Submit the treatment ferry.**

```bash
mkdir -p /tmp/zephyr-perf/<PR>
uv run python scripts/datakit/submit_perf_run.py \
  --gate <N> --pr <PR_NUMBER> --cwd "$TREATMENT_WT" \
  --status-out gs://marin-us-central1/tmp/ttl=7d/zephyr-perf/<PR>/treatment-g<N>.json \
  > /tmp/zephyr-perf/<PR>/submit-g<N>.json
TREATMENT_JOB_ID=$(jq -r .job_id < /tmp/zephyr-perf/<PR>/submit-g<N>.json)
```

`submit_perf_run.py` mirrors the iris CLI shape used by the tier-N workflow YAML
(region, memory, disk, cpu, priority, preemptibility, extra env vars). Drift
between this script and the tier YAML breaks parity — keep them in lockstep.

**b. Babysit until terminal.** Delegate to **babysit-zephyr** (or **babysit-job**
for the outer Iris job). Don't poll tightly — sleep ≥ 10 min between checks for
Gate 1, ≥ 15 min for Gate 2, ≥ 20 min for Gate 3. If the leg flakes (worker pool
wedged, coord zombie), escalate to **debug**; do not silently retry — a flaky
run masks a real regression.

**c. Collect the treatment perf report.** Use the same script the scheduled
workflows use, so the JSON matches the baseline structurally:

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

If the artifact is missing/unreadable (rare — 90d retention), fall back to the
GCS mirror
`gs://marin-us-central1/infra/datakit/ferry_perf/report_*_tier<N>/perf_report.json`
and `gsutil cp` it locally.

**e. Compare and write the verdict.** Read both JSONs and write the verdict
comment **by hand** following the threshold table. There is no compare script —
judgment about cached steps, multi-attempt churn, and infra noise lives in the
agent.

#### Threshold table (apply per gate)

`wall_seconds_total` is the **launcher-task** wall time (max `duration_ms`
across the launcher's own tasks, in seconds). It excludes time spent in
`JOB_STATE_PENDING` / `JOB_STATE_BUILDING` waiting for capacity — that queue
wait isn't a perf signal (see *Failure modes*). `stage_wall_seconds` is the
actual pipeline-step work, derived from the iris job tree.

| Signal | ✅ Pass | ⚠ Warn | ❌ Hard fail |
|---|---|---|---|
| `wall_seconds_total` delta (treatment − baseline) / baseline | ≤ +5% | +5–10% | > +10% |
| Per-step `stage_wall_seconds` delta (any stage) | ≤ +5% | +5–10% | > +10% |
| Any new entry in `infra_failures` (treatment > baseline in any bucket: `oom`, `hardware_fault`, `scheduling_timeout`, `application_failure`, `other`) | — | — | any |
| `failed_shards` strictly higher in treatment | — | — | any |
| `peak_worker_memory_mb` delta | ≤ +5% | +5–15% | > +15% |

#### Inconclusive (infra noise, not a code regression)

Mark `⚠ inconclusive` (not pass/warn/fail) and re-submit the treatment when
**any** of these holds:

- `treatment.preemption_count` materially higher than baseline (e.g. > 3 over
  baseline, or > 0 when baseline is 0). Stage durations split across attempts
  aren't comparable.
- `treatment.task_state_counts.preempted > 0`.
- `treatment.infra_failures.hardware_fault > 0` (TPU/CPU bad-node retry).
- The treatment ran on a visibly different cluster generation than the baseline.

Do **not** call a regression on a single preempted or hardware-flaky run.

#### Cached steps

If a step appears in either report's `cached_steps`, its
`stage_wall_seconds[step]` is `0.0` and the delta is meaningless. Render "—" in
the per-step table; do not count toward the verdict. Note the cache hit in a
footnote.

#### Agent self-check

Before writing the comment, walk both JSONs:

- Treatment and baseline `ferry_module` match (else mis-wired).
- `iris_job_id` differs (else comparing a run to itself).
- `task_state_counts` totals roughly equal (large divergence usually means one
  side did less work — flag it).

#### Comment shape (canonical)

The comment **must** begin with the sentinel and `🤖`.

```markdown
<!-- zephyr-perf-gate -->
🤖 ## Zephyr perf gate — Gate <N> (<tier name>)

**Verdict:** ✅ pass | ⚠ warn | ⚠ inconclusive | ❌ fail

**Baseline:** scheduled run [#<RUN_ID>](<url>), sha=`<sha>`, age=<N>d

**Hard fails:** … (omit if none)
**Warns:** … (omit if none)

### Diff assessment

(per-file table with the five yes/no answers + one-line summary, plus the
overall rationale — rendered from the step 1 assessment JSON)

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

If the gate returns `❌ fail`, **stop** — post the verdict, don't escalate.

### 6. Post one canonical comment

Sentinel-marked so re-runs replace rather than stack — find the existing
comment, then patch or post:

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

No separate issue is filed on `❌ fail`. The author decides next steps.

### 7. Clean up

```bash
git worktree remove "$TREATMENT_WT"
```

Wipe stale worktrees from earlier runs:

```bash
shopt -s nullglob
for wt in ../.zephyr_perf_worktrees/${PR_NUMBER}-*; do
  git worktree remove --force "$wt"
done
shopt -u nullglob
```

## Failure modes

- **Treatment flakes**: re-submit treatment; do not call the gate on a single
  failed run. If it flakes again, escalate to **debug**.
- **Baseline artifact missing/unreadable**: try the GCS mirror
  `gs://marin-us-central1/infra/datakit/ferry_perf/report_*_tier<N>/perf_report.json`.
  If both are unreachable, post a comment explaining the gap and ping the
  reviewer; do not submit a baseline ferry of your own.
- **Treatment OOMs at a stage the baseline didn't**: hard fail.
  `treatment.infra_failures.oom > baseline.infra_failures.oom` is enough —
  surface the worker-pool death log with the OOM line in the comment.
- **Agent says out of scope but the reviewer disagrees**: the reviewer
  re-invokes with an explicit `max_gate` override (step 1a); re-run at the
  forced gate.
- **Iris worker preemptions during the run**: spot-VM preemptions inflate
  wall-time, retry counts, and churn. Signals: `preemption_count` and
  `task_state_counts.preempted` materially higher than baseline. Mark verdict
  `⚠ inconclusive`, surface the churn, re-submit treatment.
- **Cluster scheduling delay (queue wait, not pipeline wall-time)**: the job
  sits in `JOB_STATE_PENDING`/`JOB_STATE_BUILDING` before any stage starts. Not
  a perf signal — the gate measures stage wall-times. Note the queue wait if
  notable (>30 min); does not affect the verdict.
- **Cluster contention / mixed worker generations**: a competing large job or an
  autoscaler bringing up a different machine type/zone can shift baseline timing
  10–30% with no code change. If the wall-time delta is in the warn band and the
  contention signal is plausible, mark `⚠ inconclusive` rather than `⚠ warn`.
  Hard-fail thresholds (>+10%, new infra failures) still apply.
- **TPU/CPU bad-node retries**: surface as `infra_failures.hardware_fault`. Same
  handling as preemptions — count toward churn, re-run if pervasive.
- **Cached steps in baseline but not treatment (or vice versa)**: the step's
  `stage_wall_seconds` is `0.0` and the step is in `cached_steps`. Surface "—"
  for delta; do not penalize. Note in a footnote.
- **Stale tier3 baseline (>1 week old)**: tier3 runs weekly. If the latest
  successful tier3 run on `main` is older than a week, surface the age
  prominently so the reviewer can decide whether to trust the comparison.

## Composes with

- `babysit-zephyr` — monitoring each run while in flight.
- `babysit-job` — the outer Iris job lifecycle.
- `debug` — when a leg flakes and the cause is unclear.
