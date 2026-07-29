# Echo

Echo is Marin's shared agent-context store. The `context` database runs on the
`marin-metadata` Cloud SQL instance. `echo-sync` mirrors the GitHub and Discord slice of
the [marinmirror](https://marinmirror.exe.xyz) corpus every 10 minutes, and `echo-api`
provides an IAP-gated HTTP interface and browser dashboard.

- `chunks` contains issues, pull requests, comments, and Discord messages. Each row has
  a canonical URL, a weighted PostgreSQL full-text document, and a pgvector embedding.
- `repository_file_chunks` contains the indexed `marin-community/marin` `main` head.
  Results identify the exact indexed commit and link to its GitHub blob.
- `wiki_entries` contains durable agent-authored notes and incident records with a
  title, a one-sentence `use_when` hint, lowercase kebab-case tags, a body, pgvector
  embeddings, timestamps, attribution, and deliberate-reference counters. Search
  indexes the prose fields and can filter by tags.
- `work_log` is an append-only agent logbook with one row per distilled milestone.

## CLI

`infra/echo/cli.py` drives the `echo-api` service. Run it in the repo environment:

```bash
uv run infra/echo/cli.py search "expert parallel MoE MFU on B200" --limit 10
uv run infra/echo/cli.py search "ragged_all_to_all" --domain file --domain pr
uv run infra/echo/cli.py grep ragged_all_to_all --source discord
uv run infra/echo/cli.py show <id>
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
- `file` searches files from the hourly index of GitHub `main`.
- `discord` searches Discord messages.
- `pr` searches GitHub pull requests and their comments.
- `issue` searches GitHub issues and their comments.

All domains are searched by default. Repeat `--domain` to select a subset. The old
activity filters remain available: a search with `--source`, `--kind`, or `--since`
uses the activity-only endpoint and cannot be combined with `--domain`.

All domains use reciprocal-rank fusion with `k=60`: semantic rank has
weight 1 and lexical rank has weight 2. Results with a lexical match always qualify.
A semantic-only result must have cosine distance at most 0.45 (similarity at least
0.55), so an unrelated nearest neighbor is not returned just because it is the
closest candidate. Scores compare rank positions, not calibrated relevance
probabilities. File paths, PostgreSQL full-text matches, and case-insensitive exact
file substrings contribute the lexical signal; paraphrases can enter through BGE
semantic retrieval. `grep` remains a case-insensitive literal substring scan over
activity, newest first.

The scheduled sync checks GitHub at most once per hour. An unchanged head only advances
the check time. A new head uses GitHub's compare API to delete, fetch, and re-embed
changed paths; the first build, a divergent history, or a comparison of at least 300
files falls back to one repository archive. Embeddings are prepared before the database
transaction, then the changed paths and `repository_index_state` watermark are published
atomically. Searches continue to use the previous complete index while a refresh runs.
The job reuses the existing `marinmirror-token` GitHub PAT for authenticated REST calls.

The file index accepts source, configuration, and prose files up to 256 KiB. It rejects
binary/non-UTF-8 content, lock files, generated/minified files, build and cache
directories, vendored or external trees, and secret-like names or key/certificate
extensions. A file result shows a title, matching snippet, and a subtitle containing
`path:line`, the exact indexed `main` commit, and index time. Its URL is pinned to that
commit. Working-tree and branch changes are absent until they reach `main`; use `rg`
locally when the current checkout matters.

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
`repository_index_state`, and `wiki_entries`, and `SELECT, INSERT` on `work_log`; the
`loom-vm` service account receives the same access. No database password is shared.
Group membership and IAM changes can take about 15 minutes to propagate.

## Dashboard and HTTP API

OpenAthena accounts can open the IAP-gated `echo-api` Cloud Run service to search both
the activity corpus and wiki notes. The same service exposes OpenAPI documentation at
`/docs` and the following endpoints, all under `/api`:

- `GET /api/search`
- `GET /api/federated-search`
- `GET /api/grep`
- `GET /api/chunks/{id}`
- `GET /api/wiki/search`
- `GET /api/wiki/{id}`
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
`domain=wiki|file|discord|pr|issue` parameters and returns the common ranked result
shape.

The dashboard is a Vue single-page app served from the same origin, with client-side
routes at `/` (search), `/wiki` (recently updated notes), `/wiki/<id>` (a note), and
`/chunk/<id>` (an activity chunk). The API's catch-all route serves `index.html` for
any path that isn't `/api/...`, `/healthz`, `/static/...`, `/docs`, or `/openapi.json`,
so vue-router's history-mode navigation and reloads resolve correctly.

For local dashboard development:

```bash
npm --prefix infra/echo/dashboard install
npm --prefix infra/echo/dashboard run dev
```

Rsbuild's development server proxies `/api/...` requests to `http://127.0.0.1:8000`.
Production builds are compiled into the API image and served from the same origin.

Infrastructure and durable debugging records are canonical Echo entries tagged
`incident`, `debugging`, `ops`, the subsystem, severity, and resolution. Link the URL
printed by `wiki add` or `wiki edit` from the associated PR or issue.

## Infrastructure

The `marin-echo` Pulumi stack creates the database, IAM database users, Cloud Run
service, and scheduled sync job. The job mirrors activity every ten minutes and gates
its GitHub repository check to once per hour. Database migrations in
`infra/echo/migrations/` create tables and apply PostgreSQL grants.
`infra/echo/migrate.py` records applied migrations in `schema_migrations`.

Preview or deploy from the service directory:

```bash
cd infra/echo
pulumi stack select marin-echo
pulumi preview
pulumi up
```

`pulumi up` applies pending migrations from the operator's machine. It requires ADC
with access to `cloudsql-pulumi-admin-password`. For this change, run
`infra/echo/migrate.py` before `pulumi up` so the new API and sync images never observe
a missing repository table. The first repository build fetches a GitHub archive and
embeds all eligible files; later hourly runs normally process only changed paths.
Review database grant changes before deploying them.
