# Cloud SQL metadata

`marin-metadata` is the shared Cloud SQL for PostgreSQL instance for Grafana state, eval metadata, and the ops workflow. It contains the `grafana`, `evals`, and `ops` logical databases and declares the `marin-eval-metadata` GCS bucket.

The instance has a public IP with no authorized networks. Consumers connect through the Cloud SQL connector or auth proxy. Cloud Run mounts the connector socket under `/cloudsql` through `CloudRunService.cloudsql_instances`.

Pulumi owns the instance, logical databases, PostgreSQL roles and grants, Secret Manager secrets, and bucket. The two ops login passwords and the deployment administrator password are generated as encrypted Pulumi outputs and published as Secret Manager versions. Grafana and evals retain their existing externally managed passwords.

## Deploy

```bash
uv sync --all-packages --extra deploy
cd infra/cloudsql
pulumi login gs://marin-iac-state
pulumi stack select marin-cloudsql
pulumi preview
pulumi up
```

The stack uses the shared `marin-iac-key` KMS secrets provider. Its outputs are `connection_name`, `public_ip`, `eval_bucket`, and `ops_password_generation`.

The password secrets are:

- `cloudsql-grafana-password`;
- `cloudsql-evals-password`;
- `cloudsql-ops-app-password`;
- `cloudsql-ops-migrator-password`;
- `cloudsql-pulumi-admin-password`.

The deployment identity needs Cloud SQL connection access. The PostgreSQL provider connects through the Cloud SQL connector with Application Default Credentials and uses the dedicated `pulumi_db_admin` database login for cluster and `ops` policy.

## Ops roles and credentials

Cloud SQL grants `cloudsqlsuperuser` to a built-in user created without an explicit database role. The ops logins instead inherit narrow `NOLOGIN` roles created directly in PostgreSQL. This keeps schema ownership separate from the runtime. See Google's [user-role behavior](https://cloud.google.com/sql/docs/postgres/users).

`pulumi-postgresql` manages the two `NOLOGIN` group roles, the two login roles, login-time role selection and search paths, `ops.public` ownership, database and schema grants, and default table and sequence privileges. The login defaults make objects created by `ops_migrator` belong to `ops_migrator_role`; the application never inherits schema-owner privileges.

`OPS_PASSWORD_GENERATION` in `infra/cloudsql/__main__.py` controls coordinated rotation of the two ops passwords. The Cloud SQL stack exports it and the ops stack places it in the Cloud Run environment, so applying `infra/ops` creates a revision that resolves the new `latest` secret versions. Do not change `ADMIN_PASSWORD_GENERATION` in an ordinary update because the PostgreSQL providers authenticate with that password while planning the update.

## Migrations and grants

With the Cloud SQL Auth Proxy listening on `127.0.0.1:55439`, apply every migration as the dedicated migrator before deploying the service:

```bash
OPS_MIGRATOR_PW="$(gcloud secrets versions access latest \
  --secret=cloudsql-ops-migrator-password --project=hai-gcp-models)"
PGPASSWORD="$OPS_MIGRATOR_PW" uv run --project infra/ops ops-workflow migrate \
  --database-url postgresql://ops_migrator@127.0.0.1:55439/ops
unset OPS_MIGRATOR_PW
```

The migration runner takes an advisory lock, records each file digest, and rejects changed applied files. Default privileges grant `ops_app_role` DML on tables and sequence use; the runtime never owns or creates schema objects.

## Verify the privilege boundaries

Connect as each login rather than relying on the owner to emulate its membership. `ops_app` must be able to read and update workflow rows but must fail to create a table or read Grafana. `ops_migrator` is the only ops identity with schema-creation privileges. Neither login user should be a member of `cloudsqlsuperuser`.
