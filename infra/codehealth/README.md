# Code-health telemetry

Records what Marin's review automation does and what human reviewers say, so
the two can be compared.

## Tables

Four append-only Finelog namespaces on the `marin` deployment. `review_tables.py`
holds the row types and is the single definition of the layout.

| Namespace | One row per | Written by |
| --- | --- | --- |
| `codehealth.autolint.invocations` | `pre-commit.py --review` / `/code-review` / `/review-pr` run | `log_stats.py` |
| `codehealth.autolint.findings` | finding emitted by a run | `log_stats.py` |
| `codehealth.autolint.human_comments` | classified human review comment | `review.py aggregate` |
| `codehealth.autolint.pr_review_outcomes` | pull request, rolling up the two above | `review.py aggregate` |

`invocations` and `findings` join on `invocation_id`. A run that emitted nothing
still gets an `invocations` row with `finding_count = 0`: that is the signal that
the tool ran and had no objection, and dropping it would bias every rate
computed from the table.

The comment tables are append-only and the aggregator re-emits a pull request's
rows whenever its window covers that pull request again, so a read collapses to
the newest row per natural key using the server-assigned `seq`. `review.py`
exposes that as `LATEST_HUMAN_COMMENTS_SQL` and `LATEST_PR_OUTCOMES_SQL`; a
query that reads the namespace directly will see superseded rows.

## Querying

```bash
uv run finelog query marin <<'SQL'
SELECT date_trunc('week', ts) AS week,
       count(*) AS runs,
       sum(CASE WHEN agent_exit_code = 0 AND NOT timed_out THEN 1 ELSE 0 END) AS clean,
       sum(finding_count) AS findings
FROM "codehealth.autolint.invocations"
WHERE ts >= now() - INTERVAL '30 days'
GROUP BY week ORDER BY week
SQL
```

Namespace names contain dots and must be double-quoted in SQL.

## Writing

`log_stats.py` reads one JSON event on stdin and is invoked fire-and-forget as a
detached subprocess by `infra/linter.py`. It never blocks the caller, but it does
not fail silently: a failed write exits non-zero and explains itself on stderr,
which `linter.py` keeps as `stats.log` in the run's log directory. Set
`MARIN_REVIEW_STATS=0` to skip recording.

A CI runner checks out the synthetic pull-request merge ref, so local git state
describes the merge commit rather than the branch under review, and a row built
from it cannot be joined to its pull request. The harness supplies the real
identity to `linter.py` through `MARIN_REVIEW_TRIGGER`, `MARIN_REVIEW_PR_NUMBER`,
and `MARIN_REVIEW_HEAD_SHA`. Unset, they fall back to `local` and local git,
which is correct on a developer's machine.

`review.py aggregate` classifies reviewer comments on recently-merged pull
requests and appends the two comment tables; `review.py report` renders all four
into a markdown digest. Both need `gh auth login`, and `aggregate` also needs a
logged-in `claude` CLI for the classifier.

## Access

Finelog sits behind Iris IAP. Interactively, `uv run iris --cluster marin login`
caches the token these tools use. Unattended callers (CI, cron) have no desktop
token and instead set `MARIN_FINELOG_IAP_AUDIENCE` to the IAP client id, which
mints a service-account token from ambient Google credentials.

## Verifying a write landed

`Table.flush` reports that the client queue drained, not that the server accepted
the batch — a rejected schema or a non-retryable send is logged by the flush
thread and the rows are dropped. Append through `review_tables.append_rows`,
which compares the namespace row count across the write and raises instead.

Every row type must declare a `key_column`. The server's fallback for an
undeclared key is a column named `timestamp_ms`, which none of these have, so it
rejects the registration rather than defaulting to the declared timestamp.
