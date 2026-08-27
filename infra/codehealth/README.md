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
detached subprocess by `infra/linter.py`, so it never blocks the caller. A failed
write exits non-zero and explains itself on stderr, which `linter.py` keeps as
`stats.log` in the run's log directory. Set `MARIN_REVIEW_STATS=0` to skip
recording.

A CI runner checks out the synthetic pull-request merge ref, so local git state
describes the merge commit rather than the branch under review, and a row built
from it cannot be joined to its pull request. The harness supplies the real
identity to `linter.py` through `MARIN_REVIEW_TRIGGER`, `MARIN_REVIEW_PR_NUMBER`,
and `MARIN_REVIEW_HEAD_SHA`. Unset, they fall back to `local` and local git,
which is correct on a developer's machine.

`review.py aggregate` classifies comments from human reviewers on recently
merged pull requests. Bot comments and agent-authored comments carrying Marin's
required `🤖` prefix are excluded. The command appends the two comment tables;
`review.py report` renders all four
into a markdown digest. Inline comments carry their GitHub diff hunk into the
classifier. Review summaries and issue comments receive a bounded view of the
pull request's changed-file patches. Both commands need `gh auth login`, and
`aggregate` also needs a logged-in `codex` CLI.

`Ops - Code-health Review Data` launches the aggregator through Loom every day.
The Monday run also publishes the rolling 30-day gist. GitHub Actions exchanges
OIDC for a short-lived Loom token; the repository stores no Weaver credential.
The `codehealth` Loom profile owns Finelog access and treats skipped batches,
GitHub fetch failures, and flush failures as run failures.

Run either command from the repository root:

```bash
uv run python -m infra.codehealth.review aggregate --days 7
uv run python -m infra.codehealth.review report --days 30
```

## Access

Finelog sits behind Iris IAP, and `rigging.credentials.iap_provider_for` resolves
the token with no configuration on either path. Interactively, `uv run iris
--cluster marin login` caches a desktop OAuth token. With no cached login the
token is minted from ambient Google credentials for the Marin desktop client id,
which IAP registers as a programmatic client; that identity must hold
`roles/iap.httpsResourceAccessor` on the Iris backend service.

## Verifying a write landed

`Table.flush` returns `FlushResult.SUCCEEDED` only when the rows reached the
server; `DROPPED` means the server refused the batch or the client buffer
overflowed. `review_tables.append_rows` raises on anything but `SUCCEEDED`.

Every row type must declare a `key_column`. With none declared the server looks
for a column named `timestamp_ms`; none of these have one, so registration
fails.
