# Echo

Echo is Marin's shared agent-context store. The `context` database runs on the
`marin-metadata` Cloud SQL instance. `echo-sync` mirrors the GitHub and Discord slice of
the [marinmirror](https://marinmirror.exe.xyz) corpus every 10 minutes, and `echo-api`
provides an IAP-gated HTTP interface.

- `chunks` contains issues, pull requests, comments, and Discord messages. Each row has
  a canonical URL and a pgvector embedding for semantic search.
- `work_log` is an append-only agent logbook with one row per distilled milestone.

## Search from the CLI

The repository CLI connects directly to Cloud SQL with Application Default
Credentials (ADC):

```bash
gcloud auth application-default login
infra/echo/cli/search.py search "expert parallel MoE MFU on B200" --limit 10
infra/echo/cli/search.py grep "ragged_all_to_all" --source discord
infra/echo/cli/search.py show <id>
```

`search` ranks by cosine distance; lower scores are closer. GitHub hits below about
0.25 are usually on topic. `grep` is a case-insensitive literal substring scan, newest
first. Discord results contain one message, so open the result URL when the surrounding
thread matters.

Direct SQL access uses Cloud SQL IAM group authentication. Members of
`eng-all@openathena.ai` inherit `roles/cloudsql.instanceUser`,
`roles/cloudsql.client`, `SELECT` on `chunks`, and `SELECT, INSERT` on `work_log`.
The `loom-vm` service account receives the same direct database access. No database
password is shared. Group membership and IAM changes can take about 15 minutes to
propagate.

`MARIN_DB_USER` overrides the PostgreSQL username when ADC cannot resolve an
impersonated, external-account, or workforce identity. Service-account usernames omit
the `.gserviceaccount.com` suffix.

## HTTP API

OpenAthena accounts can access the IAP-gated `echo-api` Cloud Run service. It exposes
OpenAPI documentation at `/docs` and the following endpoints:

- `GET /search`
- `GET /grep`
- `GET /chunks/{id}`
- `GET /work_log`
- `GET /work_log/{id}`
- `POST /work_log`

The API connects to PostgreSQL as `echo-api@hai-gcp-models.iam`; callers do not need
direct database access.

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
