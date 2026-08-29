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
| `codehealth.autolint.human_comments` | classified human review comment | `review_quality.py aggregate` |
| `codehealth.autolint.pr_review_outcomes` | pull request, rolling up the two above | `review_quality.py aggregate` |

`invocations` and `findings` join on `invocation_id`. A run that emitted nothing
still gets an `invocations` row with `finding_count = 0`: that is the signal that
the tool ran and had no objection, and dropping it would bias every rate
computed from the table.

The comment tables are append-only and the aggregator re-emits a pull request's
rows whenever its window covers that pull request again, so a read collapses to
the newest row per natural key using the server-assigned `seq`. `review_quality.py`
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

`review_quality.py aggregate` classifies comments from human reviewers on recently
merged pull requests. Bot comments and agent-authored comments carrying Marin's
required `🤖` prefix are excluded. The command appends the two comment tables;
`review_quality.py report` renders all four
into a markdown digest. Inline comments carry their GitHub diff hunk into the
classifier. Review summaries and issue comments receive a bounded view of the
pull request's changed-file patches. Both commands need `gh auth login`, and
`aggregate` also needs an `OPENAI_API_KEY`.

Each classified row retains the GitHub source URL, a SHA-256 context hash, and
at most 6,000 characters of diff context. These classifications are dashboard
annotations, not the source of the refinement corpus.

The aggregator fetches at most four pull requests concurrently by default.
Each worker reads that PR's GitHub endpoints sequentially, so the command never
fans one PR into several simultaneous requests. `--github-concurrency` can
lower that cap; GitHub errors fail the run without automatic retries.

`Ops - Code-health Review Data` runs the aggregator directly every day. The
runner uses the repository's CI Google credentials to mint an IAP token for
Finelog, and an OpenAI key mirrored from Secret Manager for classification. No
Iris CLI, gcloud CLI, or SSH connection is involved. Skipped batches, GitHub
fetch failures, and Finelog flush failures fail the job.
Scheduled publication belongs to the refinement job that consumes these rows;
`report` remains available for an operator with gist credentials.

`Ops - Agentic-lint Refinement` freezes a 30-day corpus every Monday. It reads
open, merged, and closed pull requests by GitHub activity time, retains full
review bodies, thread state, changed-file metadata, commits, and GitHub-served
diffs, and adds the matching Finelog automation telemetry and prior comment
annotations. Changed-file metadata is paginated through GraphQL and contains
the path, status, additions, deletions, and total changed lines. Endpoint
counts, GraphQL thread membership, benchmark coverage, and every artifact hash
must validate before the corpus is published.

The workflow uploads the frozen archive to GitHub Actions for 90 days and
attaches the identical archive to one plan-mode Loom session. It deletes the
Google credential file before launching Loom and supplies no OpenAI key. The
runner environment is not forwarded to the Loom session; its environment and
tool access come from the credential-free Loom profile.
Collection stages a GraphQL activity scan before hydrating only pull requests
with in-window human review activity. Initial hydration, connection
continuations, and snapshot fingerprint checks are batched. REST reads the two
repository-wide edit streams and one raw diff per included pull request; its
preflight reserves 150 requests for snapshot retries. Exact cursor and count
reconciliation keeps the complete run within the repository's read-only
Actions-token budget.
The lead agent delegates overlapping pattern mining, complete-catalog matching,
counterexample search, and evidence verification over the local archive. The
session cannot receive GitHub credentials and is instructed not to use live
network sources or mutate the repository or external systems. It publishes a
structured `codehealth-refinement-analysis` artifact, the committed benchmark
predictions, and a rendered `codehealth-refinement-report` artifact. The Loom
artifact is the canonical report: unlike a gist, it is versioned with the
analysis session and does not require another GitHub credential or retention
policy. A short Slack rendering links the Loom report and a catalog PR when one
exists.

Run either command from the repository root:

```bash
uv run python -m infra.codehealth.review_quality aggregate --days 7
uv run python -m infra.codehealth.review_quality report --days 30
uv run python -m infra.codehealth.review_corpus export --days 30 \
  --output /tmp/refinement-corpus
uv run python -m infra.codehealth.review_corpus validate \
  /tmp/refinement-corpus
uv run python -m infra.codehealth.refinement_report \
  --corpus /tmp/refinement-corpus \
  --analysis /tmp/refinement-analysis.json \
  --predictions /tmp/benchmark-predictions.jsonl \
  --report-out /tmp/refinement-report.md \
  --slack-out /tmp/refinement-slack.md \
  --report-url https://loom.example.com/artifacts/codehealth-refinement-report
```

The exporter publishes its directory atomically and refuses to replace an
existing path. `--limit` and `--skip-telemetry` are probe options; either marks
the manifest incomplete, and the validator rejects it by default. GitHub does
not expose deleted comments or prior versions of edited bodies. The single
GitHub-served pull-request diff is the frozen patch context when available;
binary content may be absent. GitHub returns HTTP 406 `too_large` above its
300-file diff render limit. Those pull-request records have a null `diff_path`,
and their GraphQL file metadata is the only changed-file context. GitHub caps
changed-file enumeration at 3,000 files, so the exporter fails above that cap.

The lead agent derives both 7-day and 30-day views from the same complete
snapshot. It treats prior classifications as annotations rather than filters,
requires evidence from three distinct pull requests, and verifies cited
comments against their threads and diffs. The fixed benchmark is isolated from
discovery and scored once per catalog/corpus identity. A rule with a complete
month of catalog presence and no production findings is retirement evidence
even when its catalog-derived benchmark case still passes. When the frozen
telemetry cannot prove month-long presence, the report records an exposure gap
instead of recommending retirement.

`refinement_report.py` validates the corpus identity, proposal evidence, blind
prediction coverage, and catalog rule names before rendering Markdown. It
recomputes 7-day and 30-day production counts from the frozen Finelog rows and
labels the catalog-derived benchmark as synthetic. A proposal is rejected when
its evidence is missing, agent-authored, outside the window, or drawn from fewer
than three pull requests.

The PR fetcher does not maintain a local cache. Finelog already provides the
append-only analytics tables for automation runs, findings, and classified
comments. The weekly corpus is an immutable reproducibility snapshot containing
thread state, diffs, the catalog, and benchmark inputs. Moving these records to
a separate PostgreSQL cache would still require snapshot artifacts and would
add invalidation rules for edited comments, resolved threads, force-pushed
heads, and changed diffs. Add normalized raw-event Finelog tables only if the
measured GitHub collection budget becomes a constraint; the validated survey
used 377 GraphQL points and 453 REST requests.

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
