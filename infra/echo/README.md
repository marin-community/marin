# echo

The shared context store for Marin's agents: the `context` database on the
`marin-metadata` Cloud SQL instance (`hai-gcp-models:us-central1:marin-metadata`).

- **`chunks`** — the github+discord slice of the [marinmirror](https://marinmirror.exe.xyz)
  corpus: issues, PRs, comments, and Discord messages, ~73k rows, each with a canonical
  URL and a pgvector embedding (`vector(384)`, bge-small) for semantic search. The
  `echo-sync` Cloud Run job polls the marinmirror manifest every 10 minutes and mirrors
  new corpus builds (upsert by chunk id, delete rows gone upstream). This duplicates what
  marinmirror could push itself; it is the interim design until marinmirror runs as a
  service in this project.
- **`work_log`** — the agents' shared logbook: one row per distilled milestone (result,
  decision, blocker, handoff), written by agents, read by anyone asking "what is the team
  doing". Append-only for agents.

Agents use these through the `context-search` and `work-log` skills
(`.agents/skills/`), which wrap `scripts/context_search.py` and `scripts/work_log.py`.
Unlike mumwelt's local corpus cache, echo needs no per-machine setup or 650MB download —
any agent with gcloud ADC can query it — and it carries the `work_log`, which a local
cache cannot share.

## Access

Two SQL roles, both Pulumi-managed with passwords in Secret Manager:

- `agents` (`cloudsql-agents-password`) — SELECT on `chunks`, SELECT+INSERT on `work_log`.
  What every agent connects as.
- `echo_sync` (`cloudsql-echo-sync-password`) — writer for `chunks`/`sync_state`. The sync
  job only.

Connections go through the Cloud SQL connector/auth-proxy (the instance's public IP has no
authorized networks). A caller needs `roles/cloudsql.client` and accessor on the relevant
password secret:

```python
# deps: cloud-sql-python-connector[pg8000]
import subprocess
from google.cloud.sql.connector import Connector

password = subprocess.run(
    ["gcloud", "secrets", "versions", "access", "latest",
     "--secret=cloudsql-agents-password", "--project=hai-gcp-models"],
    capture_output=True, text=True, check=True).stdout
conn = Connector(quota_project="hai-gcp-models").connect(
    "hai-gcp-models:us-central1:marin-metadata", "pg8000",
    user="agents", password=password, db="context")
```

## Schema and migrations

`schema.py` holds the SQLAlchemy table definitions — the single source of truth.
Migrations live in `migrations/mNNNN_*.py` (each exposes `upgrade(conn)`) and are applied
with `migrate.py`, which records progress in `schema_migrations`:

```bash
infra/echo/migrate.py --list    # applied/pending
infra/echo/migrate.py           # apply pending
```

Grants live in migrations beside the DDL they depend on; Pulumi owns the roles.

## Deploy

`pulumi up` needs a local Cloud SQL auth proxy because the PostgreSQL provider manages the
SQL roles directly (as `pulumi_db_admin`, password in `cloudsql-pulumi-admin-password`):

```bash
uv sync --all-packages --extra deploy               # once: iac + Pulumi providers on the venv

cloud-sql-proxy --port 5433 hai-gcp-models:us-central1:marin-metadata &   # keep running during up

cd infra/echo
pulumi login gs://marin-iac-state
pulumi stack select marin-echo
pulumi preview
pulumi up                                           # builds + pushes the sync image (needs docker)
```

On a fresh database, run `migrate.py` after the first `pulumi up` (roles must exist before
the grants in m0001 apply). The stack uses the shared `marin-iac-key` KMS secrets
provider; the operator needs `roles/cloudkms.cryptoKeyEncrypterDecrypter` on that key.

The one secret Pulumi does not fill is `marinmirror-token`: a GitHub PAT with `read:org`
only, owned by an Open-Athena member — marinmirror authorizes by the token owner's org
membership. Set it with:

```bash
printf '%s' "<PAT>" | gcloud secrets versions add marinmirror-token \
  --project=hai-gcp-models --data-file=-
```

## Operations

```bash
# run a sync now instead of waiting for the schedule:
gcloud run jobs execute echo-sync --region=us-central1 --project=hai-gcp-models

# recent sync logs:
gcloud logging read 'resource.type="cloud_run_job" resource.labels.job_name="echo-sync"' \
  --project=hai-gcp-models --limit=20 --format='value(textPayload)'
```
