# Code-health telemetry and refinement

Marin records agentic-lint activity and human pull-request review activity so a
weekly agent can inspect gaps in the lint catalog.

## Production lint telemetry

`log_stats.py` records every `pre-commit.py --review`, `/code-review`, and
`/review-pr` invocation in Finelog. The two primary namespaces are
`codehealth.autolint.invocations` and `codehealth.autolint.findings`, joined by
`invocation_id`. A successful run that emitted no finding still has an
invocation row with `finding_count = 0`; omitting it would bias firing rates.

The older `human_comments` and `pr_review_outcomes` namespaces remain available
for dashboards. They are classifications produced by `review_quality.py`, not
the refinement agent's source of truth.

## Persistent review workbench

The weekly refinement workflow stores raw GitHub review data and a bounded copy
of lint telemetry in the existing `context` PostgreSQL database on the shared
`marin-metadata` Cloud SQL service. Migration
`infra/echo/migrations/m0014_codehealth_review_store.py` owns the schema.

The store keeps current pull requests, comments, threads, files, commits, and
diffs, plus immutable versions of edited pull requests and review events. It
also stores source windows fetched on demand and the complete provenance of
rule probes. A sync row freezes the 30-day time window. Each reconciled
hydration batch and its checkpoints commit atomically. Later runs compare the
lightweight GitHub scan fingerprint with the stored snapshot. Repository-wide
comment streams are also compared with the stored event body and update
timestamp. The collector hydrates only new or changed pull requests and events.
A failed run resumes the same window from its committed checkpoints. A window
is retried at most three times before it is marked abandoned, preventing one
permanently bad pull request from pinning the weekly job. Exploration commands
fail closed unless the latest sync is complete.

Run the sync from the repository root:

```bash
uv run --frozen python -m infra.codehealth.refinement_sync --days 30
uv run --frozen python -m infra.codehealth.refinement_tools sync-status
```

Collection scans open, merged, and closed pull requests by activity time. It
retains full review bodies and thread state, reconciles paginated GraphQL
connections, verifies that pull-request fingerprints remain stable during
hydration, and bounds GitHub GraphQL and REST usage. GitHub does not expose
deleted comments or prior edited bodies. Raw diffs can be absent when GitHub
returns its oversized-diff response; per-file metadata remains available.
Every sync still performs the bounded GitHub activity scan, edited-comment seed
queries, fingerprint rechecks, and Finelog query. Matching event seeds reuse the
stored pull request after the scan fingerprint passes both observations. New or
edited events, changed pull-request fingerprints, and missing cache entries
trigger full hydration and a diff fetch.

## Agent tools

`infra.codehealth.refinement_tools` is the supported exploration surface:

```bash
uv run --frozen python -m infra.codehealth.refinement_tools list-prs --human --lint
uv run --frozen python -m infra.codehealth.refinement_tools list-comments --pr 8629
uv run --frozen python -m infra.codehealth.refinement_tools context \
  --event-id marin-community/marin:inline_comment:3873201765
uv run --frozen python -m infra.codehealth.refinement_tools list-rules
uv run --frozen python -m infra.codehealth.refinement_tools get-rule \
  --code ml-exception-swallow
uv run --frozen python -m infra.codehealth.refinement_tools rule-activity --days 30
uv run --frozen python -m infra.codehealth.refinement_tools probe \
  --event-id marin-community/marin:inline_comment:3873201765 \
  --rule ml-exception-swallow --model gpt-5.6-luna --effort low \
  --idempotency-key weekly-2026-08-31:3873201765:ml-exception-swallow
uv run --frozen python -m infra.codehealth.refinement_tools validate-rules
uv run --frozen python -m infra.codehealth.refinement_tools post-report \
  --title "Agentic-lint refinement — 2026-08-31" \
  --report /tmp/codehealth-report.md \
  --summary "Reviewed 30 days of feedback; opened PR #1234." \
  --idempotency-key weekly-2026-08-31:report
```

`list-prs` joins human-review and lint activity. `context` returns the complete
thread, pull-request diff, matching lint invocations and findings, and a cached
±100-line source window around an inline comment. Unavailable source windows are
negative-cached; pass `context --refresh-source` to retry one deliberately.
`probe` runs one selected YAML rule against that context with an agent-selected
model and reasoning effort.
Probe output is experimental evidence, not a human label or a recall estimate.
`rule-activity` reports which historical catalog identities have stored
snapshots. The current checkout is snapshotted on every successful sync;
catalog identities without a stored snapshot remain explicitly unknown.

The agent writes its own Markdown report and may use seaborn for charts.
`post-report` publishes a versioned Loom artifact and appends a typed result to
the durable `codehealth-refinement` channel. A Slack delivery can be attached
at the Loom automation boundary without changing report generation. There is no
fixed report renderer and no GitHub gist or corpus archive to manage.

## Structured lint catalog

`infra/lint/catalog.yaml` contains shared reviewer policy and lane settings.
Each rule is independently editable under `infra/lint/rules/<lane>/ml-*.yaml`.
`infra/lint/catalog.py` validates unique codes, lane ownership, filenames,
confidence floors, and deterministic catalog identity, then renders the same
prompts consumed by `infra/linter.py`.

The weekly agent can clarify an existing rule or add a rule directly in YAML.
When evidence supports a catalog change, it opens a normal pull request using
`infra/codehealth/refinement_pr_template.md`. The report still records weak
points when no safe catalog change is justified.

## Weekly ownership

`.github/workflows/ops-codehealth-refinement.yaml` launches one Loom automation
session every Monday. The session:

1. Runs or resumes the 30-day PostgreSQL sync.
2. Explores human feedback and production lint activity through the tools.
3. Probes suspected misses and counterexamples with selected models and effort.
4. Edits and validates structured rules when evidence warrants a change.
5. Opens a normal catalog pull request and publishes an agent-authored report.

The Loom profile uses the `loom-vm` IAM identity for the existing database and
has repository access for the ordinary pull-request workflow. The GitHub Action
only launches the session; it does not fetch, archive, or upload review data.

## Finelog access and writes

Finelog sits behind Iris IAP. `rigging.credentials.iap_provider_for` resolves
the token from a cached Iris login or ambient Google credentials. Namespace
names contain dots and must be double-quoted in SQL.

`review_tables.append_rows` raises unless `Table.flush` reports
`FlushResult.SUCCEEDED`. Every Finelog row type declares a `key_column`; the
server otherwise looks for a nonexistent `timestamp_ms` field.
