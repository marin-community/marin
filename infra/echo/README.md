# Echo

Echo is Marin's shared agent-context store. The `context` database runs on the
`marin-metadata` Cloud SQL instance. `echo-sync` mirrors the GitHub and Discord slice of
the [marinmirror](https://marinmirror.exe.xyz) corpus every 10 minutes, and `echo-api`
provides an IAP-gated HTTP interface and browser dashboard.

- `chunks` contains issues, pull requests, comments, and Discord messages. Each row has
  a canonical URL, a weighted PostgreSQL full-text document, and a pgvector embedding.
- `wiki_entries` contains durable agent-authored notes with a title, a one-sentence
  `use_when` hint, a body, pgvector embeddings, timestamps, attribution, and
  deliberate-reference counters. Search indexes all three text fields.
- `work_log` is an append-only agent logbook with one row per distilled milestone.

## CLI

`infra/echo/cli.py` drives the `echo-api` service. Run it in the repo environment:

```bash
uv run infra/echo/cli.py search "expert parallel MoE MFU on B200" --limit 10
uv run infra/echo/cli.py grep ragged_all_to_all --source discord
uv run infra/echo/cli.py show <id>
uv run infra/echo/cli.py wiki search "grafana access"
uv run infra/echo/cli.py wiki add --title "Grafana access" --use-when "inspecting dashboards" --body note.md
uv run infra/echo/cli.py wiki edit <id> --title "Grafana access" --use-when "inspecting dashboards" --body -
```

Authentication reuses Marin's shared IAP login: run `iris login` once and the CLI mints
echo-api tokens from the same cached credential (`~/.config/marin/credentials`), because
echo-api admits the shared Marin desktop OAuth client as an IAP programmatic client. There
is no separate echo login. Agents and CI need no login at all — ambient service-account
credentials (a key, GCE/Cloud Run metadata, or an ADC impersonating a service account) mint
the token instead. `ECHO_API_URL` overrides the target host; `ECHO_LOGIN_CLUSTER` selects
which cached login to reuse.

`search` uses reciprocal-rank fusion over PostgreSQL full-text and BGE semantic candidates;
higher scores are better. Exact identifiers and names receive a strong lexical signal while
paraphrases can enter through the semantic candidates. `grep` is a case-insensitive literal
substring scan, newest first. Discord results contain one message, so open the result URL
when the surrounding thread matters. Wiki writes go through the API so it can embed and
attribute each note; `wiki add`/`edit` read `--body` inline, from a file path, or from stdin
(`-`).

Direct SQL access remains available for raw queries through Cloud SQL IAM group
authentication. Members of `eng-all@openathena.ai` inherit `roles/cloudsql.instanceUser`,
`roles/cloudsql.client`, `SELECT` on `chunks` and `wiki_entries`, and `SELECT, INSERT` on
`work_log`; the `loom-vm` service account receives the same access. No database password is
shared. Group membership and IAM changes can take about 15 minutes to propagate.

## Dashboard and HTTP API

OpenAthena accounts can open the IAP-gated `echo-api` Cloud Run service to search both
the activity corpus and wiki notes. The same service exposes OpenAPI documentation at
`/docs` and the following endpoints:

- `GET /search`
- `GET /grep`
- `GET /chunks/{id}`
- `GET /wiki/search`
- `GET /wiki/{id}`
- `POST /wiki`
- `PUT /wiki/{id}`
- `POST /wiki/{id}/references`
- `GET /work_log`
- `GET /work_log/{id}`
- `POST /work_log`

The API connects to PostgreSQL as `echo-api@hai-gcp-models.iam`; callers do not need
direct database access.

For local dashboard development:

```bash
npm --prefix infra/echo/dashboard install
npm --prefix infra/echo/dashboard run dev
```

Rsbuild's development server proxies Echo API requests to `http://127.0.0.1:8000`.
Production builds are compiled into the API image and served from the same origin.

## Infrastructure

The `marin-echo` Pulumi stack creates the database, IAM database users, Cloud Run
service, and scheduled sync job. Database migrations in `infra/echo/migrations/` create
tables and apply PostgreSQL grants. `infra/echo/migrate.py` records applied migrations
in `schema_migrations`.

Preview or deploy from the service directory:

```bash
cd infra/echo
pulumi stack select marin-echo
pulumi preview
pulumi up
```

`pulumi up` applies pending migrations from the operator's machine. It requires ADC
with access to `cloudsql-pulumi-admin-password`. Review database grant changes before
deploying them.
