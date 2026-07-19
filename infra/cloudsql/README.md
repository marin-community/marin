# cloudsql

A shared Cloud SQL for PostgreSQL instance, `marin-metadata`, for Marin's internal metadata:
Grafana's state (the `grafana` database) and eval run records (the `evals` database). It also
declares `marin-eval-metadata`, a GCS bucket for eval run records.

The instance has a public IP with no authorized networks, so nothing dials it directly.
Consumers connect through the Cloud SQL connector/auth-proxy — Cloud Run mounts the socket
under `/cloudsql` (see `iac.gcp.cloud_run.CloudRunService.cloudsql_instances`).

Pulumi owns the instance, the two databases, the two Secret Manager secret *shells*, and the
bucket. It does not own the SQL users or their passwords: a password passed to Pulumi would
land in stack state. Users and secret values are set out-of-band with `gcloud` (below). This
directory is its own Pulumi project, runs on the shared repo venv, and shares `infra/iac`'s
state backend.

## Deploy

```bash
uv sync --all-packages --extra deploy                     # once: iac + Pulumi providers on the venv (pulumi lives behind marin-iac[deploy])

cd infra/cloudsql
pulumi login gs://marin-iac-state
export PULUMI_CONFIG_PASSPHRASE="$(gcloud secrets versions access latest \
  --secret=pulumi-iac-passphrase --project=hai-gcp-models)"
pulumi stack select marin-cloudsql                        # first time: pulumi stack init marin-cloudsql

pulumi preview                                            # plan; then, once it looks right:
pulumi up
```

`pulumi up` creates the instance, the `grafana` and `evals` databases, the
`cloudsql-grafana-password` and `cloudsql-evals-password` secret shells, and the
`marin-eval-metadata` bucket. Outputs: `connection_name` (the `project:region:instance`
connector target), `public_ip`, and `eval_bucket`.

## Users and passwords (out-of-band)

Pulumi creates the secret shells but never a value or a SQL user. After `pulumi up`, create
each native user and store its password in the matching secret. Generate a password, create
the SQL user with it, then add it as the secret's value so consumers read the same string:

```bash
GRAFANA_PW="$(python3 -c 'import secrets,sys; sys.stdout.write(secrets.token_urlsafe(32))')"
gcloud sql users create grafana --instance=marin-metadata --project=hai-gcp-models --password="$GRAFANA_PW"
printf '%s' "$GRAFANA_PW" | gcloud secrets versions add cloudsql-grafana-password \
  --project=hai-gcp-models --data-file=-

EVALS_PW="$(python3 -c 'import secrets,sys; sys.stdout.write(secrets.token_urlsafe(32))')"
gcloud sql users create evals --instance=marin-metadata --project=hai-gcp-models --password="$EVALS_PW"
printf '%s' "$EVALS_PW" | gcloud secrets versions add cloudsql-evals-password \
  --project=hai-gcp-models --data-file=-
```
