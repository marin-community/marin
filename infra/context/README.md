# context

A shared Cloud SQL for PostgreSQL instance, `marin-context`, that gives every team member's
agents a common context store — the `context` database carries:

- **`chunks`** — the github+discord slice of the [marinmirror](https://marinmirror.exe.xyz)
  corpus (issues, PRs, comments, Discord messages), with pgvector embeddings
  (`vector(384)`, `BAAI/bge-small-en-v1.5`, HNSW cosine index) for semantic search. Kept
  current by the `marin-context-sync` Cloud Run job (sync/), triggered by Cloud Scheduler
  every 6 hours: it compares the marinmirror manifest against the `sync_state` watermark,
  and on a new build downloads the corpus, upserts by chunk id, and deletes rows gone
  upstream.
- **`work_log`** — an append-only logbook written by agents (never humans): distilled
  milestones of who is working on what, as shared team context.

The instance has a public IP with no authorized networks, so nothing dials it directly.
Consumers connect through the Cloud SQL connector/auth-proxy; the sync job mounts the
connector socket under `/cloudsql`.

Pulumi owns the instance, the `context` database, the Secret Manager secret *shells*, and
the sync job (service account, Artifact Registry repo + image, Cloud Run job, Scheduler
trigger). It does not own the SQL users, their passwords, or the marinmirror token: a value
passed to Pulumi would land in stack state. Those are set out-of-band with `gcloud` (below).
This directory is its own Pulumi project, runs on the shared repo venv, and shares
`infra/pulumi`'s state backend.

## Deploy

```bash
uv sync --all-packages --extra deploy                     # once: iac + Pulumi providers on the venv

cd infra/context
pulumi login gs://marin-iac-state
pulumi stack select marin-context

pulumi preview                                            # plan; then, once it looks right:
pulumi up                                                 # builds + pushes the sync image (needs docker)
```

The stack uses the shared `marin-iac-key` KMS secrets provider. The operator needs
`roles/cloudkms.cryptoKeyEncrypterDecrypter` on that key; no passphrase is used.

### Adopting pre-existing resources (one-shot, already done)

The instance, database, and password secrets were first created by hand; the stack adopted
them with the shared `marin-iac:import` recon flag (see `infra/pulumi/README.md`):

```bash
pulumi config set marin-iac:import true
pulumi preview          # gate: every pre-existing resource plans `import`, none plan replace/delete
pulumi up
pulumi config rm marin-iac:import   # ONE-SHOT: leaving it set breaks the next up
```

## Users, passwords, and the marinmirror token (out-of-band)

Pulumi creates the secret shells but never a value or a SQL user. The live setup:

```bash
# postgres admin password (the sync job connects with it):
PW="$(python3 -c 'import secrets,sys; sys.stdout.write(secrets.token_urlsafe(32))')"
gcloud sql users set-password postgres --instance=marin-context --project=hai-gcp-models --password="$PW"
printf '%s' "$PW" | gcloud secrets versions add cloudsql-postgres-marin-context \
  --project=hai-gcp-models --data-file=-

# shared agents user (what everyone's agents connect as):
PW="$(python3 -c 'import secrets,sys; sys.stdout.write(secrets.token_urlsafe(32))')"
gcloud sql users create agents --instance=marin-context --project=hai-gcp-models --password="$PW"
printf '%s' "$PW" | gcloud secrets versions add cloudsql-agents-password \
  --project=hai-gcp-models --data-file=-

# marinmirror bearer token: a fine-grained GitHub PAT (read:org) of an Open-Athena member:
printf '%s' "<the PAT>" | gcloud secrets versions add marinmirror-token \
  --project=hai-gcp-models --data-file=-
```

## Database schema (out-of-band)

`chunks` and `sync_state` are owned by sync/main.py (`CREATE ... IF NOT EXISTS` on every
run). `work_log` and the `agents` grants were applied once as `postgres`:

```sql
CREATE TABLE work_log (
  id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  at      timestamptz NOT NULL DEFAULT now(),
  author  text NOT NULL,   -- whose agent wrote it, e.g. 'rav/claude-code'
  project text NOT NULL,   -- stable slug for the thread of work
  title   text NOT NULL,   -- one-line summary
  body    text             -- short markdown; link evidence inline
);
CREATE INDEX ON work_log (project, at DESC);
CREATE INDEX ON work_log (at DESC);

GRANT SELECT ON chunks TO agents;
GRANT SELECT, INSERT ON work_log TO agents;   -- no UPDATE/DELETE: append-only for agents
```

## Connecting (agents)

```python
# deps: cloud-sql-python-connector[pg8000]
import subprocess
from google.cloud.sql.connector import Connector

password = subprocess.run(
    ["gcloud", "secrets", "versions", "access", "latest",
     "--secret=cloudsql-agents-password", "--project=hai-gcp-models"],
    capture_output=True, text=True, check=True).stdout
conn = Connector(quota_project="hai-gcp-models").connect(
    "hai-gcp-models:us-central1:marin-context", "pg8000",
    user="agents", password=password, db="context")
```

The caller needs `roles/cloudsql.client` and accessor on the secret in `hai-gcp-models`.

## Operations

```bash
# run a sync now instead of waiting for the schedule:
gcloud run jobs execute marin-context-sync --region=us-central1 --project=hai-gcp-models

# recent sync logs:
gcloud logging read 'resource.type="cloud_run_job" resource.labels.job_name="marin-context-sync"' \
  --project=hai-gcp-models --limit=20 --format='value(textPayload)'
```
