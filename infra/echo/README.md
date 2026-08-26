# Echo

Echo is Marin's shared agent-context store. The `context` database runs on the
`marin-metadata` Cloud SQL instance. Every 10 minutes, `echo-sync` mirrors the GitHub
and Discord slice of the [marinmirror](https://marinmirror.exe.xyz) corpus and then
attempts one fair turn of the repository corpus. `echo-api` provides an IAP-gated HTTP
interface and browser dashboard.

- `chunks` contains issues, pull requests, comments, and Discord messages. Each row has
  a canonical URL, a weighted PostgreSQL full-text document, and a pgvector embedding.
- `repository_file_chunks` contains the indexed `main` heads of
  `marin-community/marin`, `marin-community/vllm`, `marin-community/tpu-inference`,
  `marin-community/evalchemy`, `marin-community/harbor`, and
  `marin-community/MarinSkyRL`. Results identify both the repository and exact indexed
  commit, and link to that GitHub blob.
- `wiki_entries` contains durable agent-authored notes and incident records with a
  title, a one-sentence `use_when` hint, lowercase kebab-case tags, a body, pgvector
  embeddings, timestamps, attribution, and deliberate-reference counters. Search
  indexes the prose fields and can filter by tags.
- `work_log` is an append-only agent logbook with one row per distilled milestone.
- `search_feedback` records the query, IAP-authenticated caller, and short overall
  explanation. `search_feedback_grades` links 0–10 grades to exact stored search-result
  rows.
- `search_executions` permanently records every API search, including its normalized
  query, mode, selected domains and filters, result count, latency, nullable legacy
  repository commit, and service revision. Each `search_execution_results` row has an
  identity key and snapshots the returned rank, metadata, snippet, and raw reranker
  score. New multi-repository searches leave the scalar repository commit null; their
  per-result IDs and pinned URLs carry provenance.

## CLI

`infra/echo/cli.py` drives the `echo-api` service. Run it in the repo environment:

```bash
uv run infra/echo/cli.py search "expert parallel MoE MFU on B200" --limit 10
uv run infra/echo/cli.py search "ragged_all_to_all" --domain file --domain pr
uv run infra/echo/cli.py search "compare cache implementations" --repository all
uv run infra/echo/cli.py get file:marin-community/marin@main:lib/iris/OPS.md
uv run infra/echo/cli.py feedback --query "how do I deploy Iris?" \
  --execution-id 1234 --grade wiki:730=0 --grade file:731=10 <<'EOF'
The file result answered the question; the wiki result did not.
EOF
uv run infra/echo/cli.py history export > echo-search-history.jsonl
uv run infra/echo/cli.py grep ragged_all_to_all --source discord
uv run infra/echo/cli.py wiki search "grafana access" --tag ops
uv run infra/echo/cli.py wiki add --file note.md          # OKF markdown document
uv run infra/echo/cli.py wiki show <id> > note.md         # export as OKF, edit, then:
uv run infra/echo/cli.py wiki edit <id> --file note.md
```

Authentication reuses Marin's shared IAP login: run `iris login` once and the CLI mints
echo-api tokens from the same cached credential (`~/.config/marin/credentials`), because
echo-api admits the shared Marin desktop OAuth client as an IAP programmatic client. There
is no separate echo login. Agents and CI need no login at all — ambient service-account
credentials (a key, GCE/Cloud Run metadata, or an ADC impersonating a service account) mint
the token instead. `ECHO_API_URL` overrides the target host; `ECHO_LOGIN_CLUSTER` selects
which cached login to reuse.

`search` returns one ranked result set across five domains:

- `wiki` searches durable Echo entries.
- `file` searches the selected configured GitHub `main` branch.
- `discord` searches Discord messages.
- `pr` searches GitHub pull requests and their comments.
- `issue` searches GitHub issues and their comments.

Wiki, files, pull requests, and issues are searched by default. Discord is opt-in
because high-volume conversation is a noisier source of agent context. Repeat
`--domain` to select a subset or add `--domain discord`. When files are selected and
`--repository` is omitted, the CLI infers the configured Marin-community repository
from the current Git checkout. A contributor fork remote with the same repository name
maps to its Marin-community target. Pass `--repository <owner/repo>` to choose one
configured repository or `--repository all` to search all six. An unscoped file search
outside a supported checkout stops before making a request and names both remedies.
Searches without files need no Git checkout. The activity-only `grep` command retains
`--source` and `--kind` filters. The compatibility `GET /api/search` endpoint still
accepts source, kind, and date filters for existing API clients.

The first retrieval stage uses reciprocal-rank fusion with `k=60`. Paths, filenames,
flags, code-like identifiers, and one- or two-token keyword searches use semantic
weight 1 and lexical weight 2. Prose searches of at least three tokens use semantic
weight 2 and lexical weight 1; Markdown and reStructuredText files receive a 1.15 score
multiplier, while test files receive a 0.85 multiplier. This keeps exact-code and terse
keyword lookup lexical-first and favors runbooks over incidental test matches for
questions such as `how do i deploy iris`.

Prose questions that combine logs with training, jobs, runs, or querying add
`finelog iris job logs` to the semantic and cross-encoder inputs. The lexical query
and displayed query remain unchanged. Prose KV-cache questions similarly add
Levanter's `KvPageCache` and `kv_cache` vocabulary. These expansions map user
vocabulary onto Marin subsystem names without changing identifier queries.

Results with a lexical match always qualify. A semantic-only result must have cosine
distance at most 0.45 (similarity at least 0.55), so an unrelated nearest neighbor is
not retained just because it is the closest candidate. File paths, PostgreSQL full-text
matches, and case-insensitive exact file substrings contribute the lexical signal;
exact and partial basename matches receive additional weight. A file with several
independently qualifying chunks gains up to 30% over its best chunk score, so repeated
evidence helps without allowing a large file to dominate. Paraphrases can enter through
BGE semantic retrieval.

The API retrieves at least 20 candidates from each selected domain, takes the best 24
after first-stage fusion, and reranks their complete indexed chunks one at a time with
the INT8 ONNX build of `ms-marco-MiniLM-L-6-v2`. Final rank is reciprocal-rank fusion of the
first-stage rank (weight 0.2) and cross-encoder rank (weight 0.8). Wiki candidates need
a raw cross-encoder score of at least -1; other domains retain the -2 floor. These are
empirical relevance floors, not calibrated probabilities. The wiki cutoff retains 93%
of graded results scoring at least 4 while removing 22% of results below 4 in the 155
execution-linked wiki grades available on 2026-08-18. Raw reranker scores are retained
with new result snapshots so other domain cutoffs can use stable data. A search can
return fewer than the requested limit and returns at most 24 results. `grep` remains a
case-insensitive literal substring scan over activity, newest first.

The API runs four ONNX inference threads on each 4-vCPU Cloud Run instance. Request-based
billing retains one warm instance, admits one request per instance, and scales concurrent
searches out to at most four instances. Startup CPU boost reduces model initialization time
for burst instances without allocating CPU to idle instances continuously.

CLI search prints one result block with an execution-specific grading key, title,
source ID, and one primary source excerpt of at most 240 characters. File results use
their highest-ranked `path:line` reference, wiki results use `use_when`, and activity
results use the matching source excerpt. The detail line is independent of terminal
width, so long source IDs do not consume its display budget. Grading keys use
`<domain>:<numeric-key>` and remain attached
to the stored row even when a later corpus sync replaces an activity chunk ID.
Run `uv run infra/echo/cli.py get <domain:id>` to fetch the full indexed wiki body,
repository file, pull request or issue chunk, or Discord message and its canonical URL.
File IDs have the form `file:<owner>/<repository>@main:<path>`; the repository is part
of the identity, so `file:marin-community/marin@main:README.md` and
`file:marin-community/vllm@main:README.md` are different results.
Wiki summaries use the `use_when` hint; files and activity use the matching source
excerpt. Echo does not generate summaries with an LLM at query time, avoiding added
latency and an additional prompt-injection path.

`search` reports the number of results, elapsed wall-clock time, and the API's
`Server-Timing` stages before its table. The wall-clock measurement covers token
acquisition, the network request, server retrieval and reranking, history persistence,
and response decoding. The server stages separate query embedding, database setup,
selected-domain retrieval, reranking, history persistence, and total application time.

### Record and inspect search feedback

Submit useful or poor results with `feedback`. A search prints its durable execution ID;
pass it with `--execution-id` so each judgment is tied to the exact ranked result set.
Repeat `--grade <result-key>=<0-10>` for
the results you evaluated, where 0 means irrelevant and 10 means directly useful to the
task. The exact query makes each judgment replayable against a future search version.
A short overall explanation is required on stdin. Capture the result set's gestalt
without restating each score. An explanation without grades can describe an empty or
globally poor result set. When an execution ID is supplied, the caller and query must
match and every graded result must have appeared in that recorded execution.

Search history is retained indefinitely for internal search-quality work. The service
does not persist request headers, user agents, or network addresses in these tables.
`history export` pages through the durable records as JSONL, including result-row IDs,
ranked snapshots, and raw reranker scores. New file snapshots contain qualified IDs and
commit-pinned URLs. Existing path-only IDs, stored URLs, and nullable
`repository_commit` values are left unchanged and still export from their stored data;
old path-only IDs are not accepted by the new detail route. Feedback still interprets
path-only historical file IDs as Marin results, but rejects unknown qualified repository
or branch identities instead of falling back to Marin. Historical query manifests replay
through explicit all-repository search, preserving the existing six-repository quality
comparison.

Use the durable export for retrieval analysis. Cloud Logging request lines are operational
telemetry and do not preserve the execution-to-result relationship:

```bash
uv run infra/echo/cli.py history export > echo-search-history.jsonl
```

Every scheduled execution syncs activity first. It then tries the one global repository
lock. A lock loser finishes successfully without consuming a turn. A lock winner saves
the next fair repository cursor before attempting exactly one target. A repository
failure therefore consumes its turn and fails the execution, but cannot starve later
targets. Automatic Cloud Run task retries are disabled. Once incremental attempts finish
within the ten-minute interval, the six targets rotate about once per hour. A long attempt
holds the global lock, so overlapping executions still sync activity but skip their
repository turns.

Each selected turn checks that repository's head. An unchanged head only advances the
check time. A new head uses GitHub's compare API to delete, fetch, and re-embed changed
paths; the first build, a divergent history, or a comparison of at least 300 files falls
back to one repository archive. Builds embed and commit ten files at a time, print
cumulative file progress, and record a durable build watermark in Postgres. A later
turn fetches the same immutable GitHub snapshot and skips paths already committed.
Partial generations are searchable: a full build starts empty and fills in by batch,
while an incremental build keeps unchanged files and replaces changed paths as their
batches finish. Results and index status name the target commit so this temporary
incompleteness is visible. The job reuses the existing `marinmirror-token` GitHub PAT for
authenticated REST calls.

For a cold target that needs longer than the scheduled two-hour attempt, run one manual
task with a six-hour timeout. Choose one of the six configured repository names and
repeat the same command if needed; the durable per-repository checkpoint makes each run
resume safely:

```bash
gcloud run jobs execute echo-sync \
  --project=hai-gcp-models \
  --region=us-central1 \
  --tasks=1 \
  --task-timeout=21600s \
  --update-env-vars=ECHO_REPOSITORY_TARGET=marin-community/vllm \
  --wait
```

The file index accepts source, configuration, and prose files up to 256 KiB. It rejects
binary/non-UTF-8 content, lock files, generated/minified files, build and cache
directories, vendored or external trees, and secret-like names or key/certificate
extensions. A file result shows a title, up to three query-ranked `path:line` excerpts
from the best matching chunk, and a subtitle containing the first line, exact indexed
`main` commit, and index time. Each reference URL is pinned to that commit. Working-tree
and branch changes are absent until they reach `main`; use `rg` locally when the current
checkout matters.

Discord results contain one message, so open the result URL when the surrounding
thread matters. Wiki writes go through the API so it can embed and attribute each
note. Repeat `wiki search --tag <tag>` to require several tags. Tags are normalized
to lowercase, deduplicated, and limited to 20 kebab-case values per entry.

Wiki notes are authored as [Open Knowledge Format](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing)
(OKF) documents — a markdown file with a YAML frontmatter block. `wiki add --file` and
`wiki edit --file` read one, `wiki show` prints one, so an entry round-trips through a `.md`
file (the `--title`/`--use-when`/`--body` flags remain as an alternative to `--file`):

```markdown
---
type: wiki-note
title: chunks.text needs a pg_trgm index for grep
use_when: when grep or ILIKE substring queries over the corpus are slow
tags:
  - echo
  - debugging
---

A pg_trgm GIN index on chunks.text makes the substring match an index scan.
```

Direct SQL access remains available for raw queries through Cloud SQL IAM group
authentication. Members of `eng-all@openathena.ai` inherit `roles/cloudsql.instanceUser`,
`roles/cloudsql.client`, `SELECT` on `chunks`, `repository_file_chunks`,
`repository_index_state`, `repository_index_builds`, `repository_sync_turn`,
`wiki_entries`, `search_feedback`,
`search_feedback_grades`, `search_executions`, and `search_execution_results`, and
`SELECT, INSERT` on `work_log`; the `loom-vm` service
account receives the same access. No database password is shared.
Group membership and IAM changes can take about 15 minutes to propagate.

## Dashboard and HTTP API

OpenAthena accounts can open the IAP-gated `echo-api` Cloud Run service to search both
the activity corpus and wiki notes. The same service exposes OpenAPI documentation at
`/docs` and the following endpoints, all under `/api`:

- `GET /api/search`
- `GET /api/federated-search`
- `GET /api/search-configuration`
- `GET /api/repository-index`
- `GET /api/repository-files/{owner}/{repository}@main:{path}`
- `GET /api/grep`
- `GET /api/chunks/{id}`
- `GET /api/wiki/search`
- `GET /api/wiki/{id}`
- `GET /api/feedback`
- `POST /api/feedback`
- `GET /api/search-executions`
- `POST /api/wiki`
- `PUT /api/wiki/{id}`
- `POST /api/wiki/{id}/references`
- `GET /api/work_log`
- `GET /api/work_log/{id}`
- `POST /api/work_log`

The API connects to PostgreSQL as `echo-api@hai-gcp-models.iam`; callers do not need
direct database access.

`GET /api/search` preserves the activity-only response used by existing clients.
`GET /api/federated-search` accepts repeated
`domain=wiki|file|discord|pr|issue` parameters and an optional `repository` parameter.
Use a configured `<owner>/<repository>` value for one file corpus or `all` for all six.
Omission selects `marin-community/marin`, which is also the dashboard default. The
repository scope applies only to file candidates; wiki remains global and pull requests
and issues are unchanged. Search history records the resolved file scope. The response
shape is unchanged. `GET /api/repository-index` always returns all six targets in
configured order. Each row is `empty`, `building` with completed and total file counts,
or `ready` with its indexed commit and time. The dashboard displays every row on its
landing page.
`GET /api/search-configuration` supplies the domain catalog, defaults, and displayed
commit length used by the dashboard. The CLI's `get` command uses the existing wiki and
activity detail endpoints plus the qualified repository-file route for complete indexed
files.

Every successful search response includes `X-Echo-Search-Execution-ID` and is stored
with its returned result snapshot. `GET /api/search-executions` returns stable ID-ordered
pages of at most 500 records for evaluation exports.

`POST /api/feedback` accepts an exact query, up to 24 unique `{key, grade}` records from
0 through 10, and a required overall explanation of at most 2,000 characters. Grades may be empty
when the search returns no useful results. The API attributes feedback to the
IAP-authenticated caller and optionally links it to a matching search execution.
`GET /api/feedback` returns recent submissions newest-first. Each grade includes its
grading key, source ID, title, and browseable URL from the recorded search snapshot or
current source.
The response also includes explanation-only submissions with an empty grade list.

The dashboard is a Vue single-page app served from the same origin, with client-side
routes at `/` (search), `/wiki` (recently updated notes), `/conversation` (recent agent
work-log entries), `/feedback` (recent result grades), `/wiki/<id>` (a note), and
`/chunk/<id>` (an activity chunk). Conversation details load when an entry is opened.
The feedback table links each grade to its source and keeps explanation-only submissions
visible. The API's catch-all route serves `index.html` for any path that isn't
`/api/...`, `/healthz`, `/static/...`, `/docs`, or `/openapi.json`, so vue-router's
history-mode navigation and reloads resolve correctly.

Dashboard search uses the federated endpoint and exposes checkboxes for files, wiki,
Discord, pull requests, and issues. Discord starts unchecked. Header tabs provide common
domain presets. Wiki note bodies render sanitized Markdown.

For local dashboard development:

```bash
npm --prefix infra/echo/dashboard install
npm --prefix infra/echo/dashboard run dev
```

Rsbuild's development server proxies `/api/...` requests to `http://127.0.0.1:8000`.
Production builds are compiled into the API image and served from the same origin.

## Local Python tests

Echo's Python modules deliberately use flat imports. From `infra/echo`, this command
supplies the runtime and test dependencies and runs every Echo Python test:

```bash
env PYTHONPATH=.:sync:api uv run --no-project \
  --with pytest --with pytest-timeout --with fastapi --with httpx \
  --with 'cloud-sql-python-connector[pg8000]>=1.9' \
  --with 'sqlalchemy>=2' --with 'pgvector>=0.3' \
  --with 'fastembed>=0.8,<0.9' --with requests \
  --with-editable ../../lib/rigging \
  pytest -o addopts='' test_*.py api/test_app.py
```

Infrastructure and durable debugging records are canonical Echo entries tagged
`incident`, `debugging`, `ops`, the subsystem, severity, and resolution. Link the URL
printed by `wiki add` or `wiki edit` from the associated PR or issue.

## Infrastructure

The `marin-echo` Pulumi stack creates the database, IAM database users, Cloud Run
service, and one scheduled sync job with one task. The job mirrors activity every ten
minutes and then advances one globally serialized repository turn. GCP IAM grants for those
resources live in `infra/pulumi/src/iac/gcp/echo.py` and are applied by the `marin`
infrastructure stack;
the Echo stack still owns Cloud SQL users and PostgreSQL grants. Database migrations in
`infra/echo/migrations/` create tables and apply PostgreSQL grants.
`infra/echo/migrate.py` records applied migrations in `schema_migrations`.

Deploy the production stack from the repository root through the shared command.
Pulumi previews the update before asking for confirmation:

```bash
uv run --all-packages --extra deploy marin-deploy echo rollout
```

The rollout applies pending migrations from the operator's machine. It requires ADC
with access to `cloudsql-pulumi-admin-password`. When a release adds tables queried by
new API or sync images, run `infra/echo/migrate.py` before the rollout to avoid a
missing-table window. The first repository build fetches a GitHub archive and embeds
all eligible files in resumable ten-file batches; later hourly runs normally process
only changed paths. Review database grant changes before deploying them.
