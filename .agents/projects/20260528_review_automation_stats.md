# Review-automation stats & dashboard — options + plan

**Goal:** measure how effective our review automations (`pre-commit.py --review`,
`/code-review`, `/review-pr`, `ops-claude-review`, security review) are. Daily
dashboard showing: how often each tool fires, how many findings it emits, and
— the headline question — *how often it catches things a human reviewer would
have otherwise had to flag*.

---

## What's actually emitting signal today

| Surface | Trigger | Output | Persisted? |
|---|---|---|---|
| `pre-commit.py --review` | Local, advisory | Free-text findings to stdout, citing `ml-…` codes from `infra/lint.md` | No |
| `/code-review` skill (low/med/high/ultra) | Local on diff; `--comment` posts inline | Text findings; optional inline GH comments | Only if `--comment` |
| `/review-pr` skill | Local or via GHA on PR | Multi-agent findings; `--comment` posts inline | Only if `--comment` |
| `ops-claude-review.yaml` | GHA: PR open/ready, or "claude review this" comment | Calls `/review-pr --comment` → GH inline comments | Yes (in GH API) |
| `marin-lint.yaml` | GHA on every push/PR | pre-commit pass/fail; exit code | GHA logs only |
| Human PR review comments | Reviewers on PRs | GH review/review-comments | Yes (in GH API) |

Existing telemetry plumbing we could reuse: **Finelog** (`lib/finelog`, Arrow
IPC + DuckDB-backed, already used by Iris) and **W&B `marin-monitoring`**
(`infra/github_wandb_metrics.py` already logs weekly GH metrics there).

---

## Options for where to collect

### Option A — GitHub API only, pull-based
Daily cron job that walks merged-in-last-24h PRs via `gh api`, classifies each
comment/review by author (bot vs human, which bot), counts findings, and
emits a report.

- Pros: zero changes to any tool; works retroactively across all history;
  easy to iterate on metrics offline.
- Cons: invisible to `pre-commit.py --review` (never reaches GitHub); can't
  distinguish "agent found a real bug" from "agent posted noise" without a
  resolution signal; bot identity is brittle (depends on the GH App user).

### Option B — Instrument each tool, push-based to Finelog
Each automation writes a structured record at end-of-run: tool name, run id,
PR# (if any), commit sha, finding count, list of `ml-…` codes / severity,
elapsed time, agent model. Stored in a new Finelog table
`code_review_runs`. The dashboard joins this against GitHub comment data.

- Pros: high-fidelity, captures local `pre-commit --review` runs CI never sees,
  schema-validated, queryable via DuckDB.
- Cons: requires touching every tool; local runs need a way to ship rows
  (Finelog write endpoint, or batched GCS append) — and many devs may not
  have credentials to write; risk of dropping data on local-only runs.

### Option C — Hybrid (recommended)
- **CI side (push):** the GHA workflows (`ops-claude-review`, `marin-lint`,
  any future scheduled `/code-review` run) emit a single JSON line per run
  to a known location: either a Finelog `code_review_runs` table, or — if we
  want to ship fast — appended to a daily GCS object
  `gs://marin-stats/code_review/YYYY-MM-DD.jsonl`.
- **GitHub side (pull):** the daily aggregator queries `gh api` for the same
  PRs to attach the human-comment baseline, resolution status (was the bot
  thread resolved? replied to? committed against?), and merge outcome.
- **Local `pre-commit --review`:** add an opt-in `--stats-out <path>` flag
  that writes a one-line JSON receipt. Don't try to ship local runs to the
  cloud yet; let users post-hoc with `weaver` or a `marin stats sync` if we
  ever need them. Bound the scope to CI-visible runs for v1.

- Pros: leverages existing GH data as the human-baseline ground truth, only
  one piece of new plumbing (the JSONL sink), no per-user auth headaches.
- Cons: misses local `pre-commit --review` invocations in v1; coverage grows
  only as we instrument more of CI.

### Option D — Just log to W&B `marin-monitoring`
Reuse `infra/github_wandb_metrics.py`'s existing weekly cron. Add a new run
that logs review-automation tables/metrics there.

- Pros: dashboarding is free (W&B charts), already authed in CI, already on
  weekly cadence.
- Cons: W&B tables aren't great for joining bot findings ↔ human comments;
  granularity is per-week, not per-PR; harder to drill in.

---

## Recommended plan

**Option C with W&B as the dashboard front-end (Option D for presentation).**
That gets us: structured per-PR rows in a queryable store, GitHub as
ground-truth for human comments, and a dashboard we don't have to build from
scratch.

### Headline metric
*"Catch rate"*: of the human-flagged issues on a merged PR, what fraction
had already been flagged by at least one automation on the same lines?
Approximated as:

- `bot_findings_on_pr` — count of inline comments by our bot accounts
- `human_review_comments_on_pr` — count of review comments by human reviewers
- `overlap` — bot+human comments on the same file within ±5 lines (heuristic)
- Catch rate = `overlap / human_review_comments_on_pr`
- Also track: bot precision proxy = `resolved_or_acknowledged_bot_threads / bot_findings`

(Both are proxies. Document the proxies honestly on the dashboard.)

### Steps

1. **Define the schema** (one row per automation invocation):
   `run_id, tool, variant, triggered_by (ci|local|comment), pr_number,
   commit_sha, started_at, elapsed_s, finding_count, findings[]
   {file, line, code, severity, summary}, agent_model, exit_code`.
   Land as `lib/marin/src/marin/stats/review_runs.py` (dataclass) +
   a tiny `write_jsonl(path, row)` helper.

2. **Instrument the cheapest surfaces first** — order of effort, not value:
   - `ops-claude-review.yaml`: after `/review-pr --comment` runs, write
     one JSON line to `gs://marin-stats/code_review/...`. The skill
     already knows what it posted; emit it as part of the harness step.
   - `marin-lint.yaml` running `pre-commit.py --review`: add
     `--stats-out` (writes JSONL of findings + meta) and upload as a GHA
     artifact + to the same GCS path.
   - `/code-review` and `/review-pr` skills locally: same `--stats-out`
     option; off by default, on in CI.

3. **Daily aggregator** — new script
   `infra/codehealth/review.py` (cron'd via GHA, similar shape to
   `github_wandb_metrics.py`):
   - List PRs merged in the last 24h.
   - For each: pull review threads + inline comments via `gh api`
     (`/repos/{owner}/{repo}/pulls/{n}/comments`,
     `/repos/.../pulls/{n}/reviews`).
   - Bucket comments by author class (human, claude-review bot, codex
     bot, dependabot, other).
   - Load the bot's own run rows from GCS for that PR.
   - Compute headline metrics + per-tool breakdown.
   - Write a `wandb.Table` + summary scalars to `marin-monitoring`.

4. **Dashboard** — a W&B Report in `marin-monitoring` with:
   - Daily catch-rate trendline.
   - Per-tool: invocations/day, findings/day, mean findings/PR.
   - Top `ml-…` codes by frequency.
   - Drilldown table: per-PR row with bot findings vs human comments.
   - "Sentinel" callouts: PRs merged with ≥5 human comments and zero bot
     findings (potential blind spots).

5. **Backfill** — once Option-A scraping works, replay it against the last
   ~30 days of merged PRs to seed a baseline of human-comment volume per
   PR (no bot side yet, but useful as denominator).

### Open questions to record but not block on

- **Bot identity.** What GH user account does `ops-claude-review` post as?
  The aggregator needs to recognize it deterministically. (Action: read
  one recent PR comment from the workflow to confirm the login.)
- **Overlap heuristic.** Same-file ±5 lines is a first cut; revisit once
  we have a week of data.
- **Local runs in v1.** Deferred. Revisit if we find devs run
  `pre-commit --review` heavily and we're underreporting catch rate.

### Sequencing (suggested)

1. PR 1: schema + `--stats-out` on `pre-commit.py --review`; wire into
   `marin-lint.yaml`; land aggregator skeleton that just counts findings.
2. PR 2: instrument `/review-pr` + `ops-claude-review.yaml`; add GH-API
   side of the aggregator; first W&B table.
3. PR 3: catch-rate computation + dashboard polish; weekly summary in
   Slack via existing automation.
