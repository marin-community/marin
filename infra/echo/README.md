# echo

The shared context store for Marin's agents: the `context` database on the
`marin-metadata` Cloud SQL instance (`hai-gcp-models:us-central1:marin-metadata`).

- **`chunks`** — the github+discord slice of the [marinmirror](https://marinmirror.exe.xyz)
  corpus: issues, PRs, comments, and Discord messages, ~73k rows, each with a canonical
  URL and a pgvector embedding (`vector(384)`, bge-small) for semantic search. The
  `echo-sync` Cloud Run job polls the marinmirror manifest every 10 minutes and mirrors
  new corpus builds. This is the interim design until marinmirror runs as a service in
  this project.
- **`work_log`** — the agents' shared logbook: one row per distilled milestone (result,
  decision, blocker, handoff), written by agents, read by anyone asking "what is the team
  doing". Append-only for agents.

Two ways in:

- **`echo-api`** — an IAP-gated Cloud Run service (`api/`) exposing a well-defined,
  OpenAPI-documented HTTP interface (`/search`, `/grep`, `/chunks/{id}`, `/work_log`,
  `POST /work_log`; see `/docs`). It holds the single database identity and the query
  embedding model — kept warm on one always-on instance — so callers reach the corpus over
  HTTP through IAP without their own database grants. Admitted to the `eng-all@openathena.ai`
  group.
- **Direct SQL** — the `context-search` and `work-log` skills (`.agents/skills/`) wrap
  `scripts/context_search.py` and `scripts/work_log.py`, which connect to the database
  directly via IAM for callers who prefer a CLI over the service.

Access is Cloud SQL IAM, not passwords. The `eng-all@openathena.ai` group is registered
as an IAM database user and granted corpus read + logbook append; members connect as their
own ADC identity through the connector with a short-lived token — no per-user setup, no
secret. The `echo-sync` job writes as its own service account the same way. `__main__.py`
owns the IAM users and their login roles (`roles/cloudsql.instanceUser` + `.client`).

Tables live in `schema.py` and are applied with `migrate.py`, which grants the IAM users;
run it after `pulumi up` on a fresh database. `migrate.py` connects as `pulumi_db_admin`
(the pre-existing marin-metadata system role that owns the tables and can create
extensions), reading its password from Secret Manager via the API.

## Operations

```bash
# the API's interactive OpenAPI docs (through IAP in a browser):
#   https://<echo-api uri>/docs   (uri is the api_uri stack output)

# run a sync now instead of waiting for the schedule:
gcloud run jobs execute echo-sync --region=us-central1 --project=hai-gcp-models

# recent sync logs:
gcloud logging read 'resource.type="cloud_run_job" resource.labels.job_name="echo-sync"' \
  --project=hai-gcp-models --limit=20 --format='value(textPayload)'
```
