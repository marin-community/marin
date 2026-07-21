# Cloud SQL metadata

`marin-metadata` is the shared Cloud SQL for PostgreSQL instance for Grafana state, eval metadata, and the ops workflow. It contains the `grafana`, `evals`, and `ops` logical databases and declares the `marin-eval-metadata` GCS bucket.

The instance has a public IP with no authorized networks. Consumers connect through the Cloud SQL connector or auth proxy. Cloud Run mounts the connector socket under `/cloudsql` through `CloudRunService.cloudsql_instances`.

Pulumi owns the instance, logical databases, Secret Manager secret shells, and bucket. It does not put password values or native PostgreSQL users into stack state. Operators create users, secret versions, and SQL grants out of band.

## Deploy

```bash
uv sync --all-packages --extra deploy
cd infra/cloudsql
pulumi login gs://marin-iac-state
pulumi stack select marin-cloudsql
pulumi preview
pulumi up
```

The stack uses the shared `marin-iac-key` KMS secrets provider. Its outputs are `connection_name`, `public_ip`, and `eval_bucket`.

The password secret shells are:

- `cloudsql-grafana-password`;
- `cloudsql-evals-password`;
- `cloudsql-ops-app-password`;
- `cloudsql-ops-grafana-reader-password`;
- `cloudsql-ops-migrator-password`.

## Ops roles, users, and secret values

Cloud SQL grants `cloudsqlsuperuser` to a PostgreSQL user created without an explicit database role. The ops users instead inherit narrow `NOLOGIN` roles. This keeps schema ownership separate from the runtime and limits the Grafana reader to two tables. See Google's [user-role behavior](https://cloud.google.com/sql/docs/postgres/users) and [`--database-roles` guidance](https://cloud.google.com/sdk/gcloud/reference/sql/users/create#--database-roles).

These privileges are PostgreSQL objects, not Cloud SQL IAM resources. The GCP Pulumi
provider can assign an existing custom role with `gcp.sql.User.database_roles`, but it
cannot create `NOLOGIN` roles, transfer schema ownership, set login defaults, grant tables,
or manage default privileges. A PostgreSQL Pulumi provider could manage those objects, but
would make every preview and update depend on a live Auth Proxy connection and the database
owner credential. Keep that bootstrap at the SQL boundary.

The two scripts reflect the required order. [`ops_roles.sql`](ops_roles.sql) creates the
custom roles before Cloud SQL creates login users with `--database-roles`;
[`ops_login_roles.sql`](ops_login_roles.sql) sets login defaults after those users exist.

Start the Cloud SQL Auth Proxy on `127.0.0.1:55439`. Retrieve the existing Grafana owner password into an environment variable without printing it. After the Pulumi update has created the `ops` database and secret shells, create the custom roles and grants:

```bash
GRAFANA_ADMIN_PW="$(gcloud secrets versions access latest \
  --secret=cloudsql-grafana-password --project=hai-gcp-models)"
PGPASSWORD="$GRAFANA_ADMIN_PW" psql \
  --host=127.0.0.1 --port=55439 --username=grafana --dbname=grafana \
  --file=infra/cloudsql/ops_roles.sql
```

Generate each password once, create the login with its custom role, and publish the same value as a secret version:

```bash
OPS_MIGRATOR_PW="$(python3 -c 'import secrets,sys; sys.stdout.write(secrets.token_urlsafe(32))')"
gcloud sql users create ops_migrator --instance=marin-metadata \
  --project=hai-gcp-models --database-roles=ops_migrator_role \
  --password="$OPS_MIGRATOR_PW"
printf '%s' "$OPS_MIGRATOR_PW" | gcloud secrets versions add \
  cloudsql-ops-migrator-password --project=hai-gcp-models --data-file=-

OPS_APP_PW="$(python3 -c 'import secrets,sys; sys.stdout.write(secrets.token_urlsafe(32))')"
gcloud sql users create ops_app --instance=marin-metadata \
  --project=hai-gcp-models --database-roles=ops_app_role \
  --password="$OPS_APP_PW"
printf '%s' "$OPS_APP_PW" | gcloud secrets versions add \
  cloudsql-ops-app-password --project=hai-gcp-models --data-file=-

OPS_READER_PW="$(python3 -c 'import secrets,sys; sys.stdout.write(secrets.token_urlsafe(32))')"
gcloud sql users create ops_grafana_reader --instance=marin-metadata \
  --project=hai-gcp-models --database-roles=ops_grafana_reader_role \
  --password="$OPS_READER_PW"
printf '%s' "$OPS_READER_PW" | gcloud secrets versions add \
  cloudsql-ops-grafana-reader-password --project=hai-gcp-models --data-file=-
```

Do not echo generated passwords or pass them through Pulumi configuration. Finish the role setup after all three users exist:

```bash
PGPASSWORD="$GRAFANA_ADMIN_PW" psql \
  --host=127.0.0.1 --port=55439 --username=grafana --dbname=grafana \
  --file=infra/cloudsql/ops_login_roles.sql
unset GRAFANA_ADMIN_PW
```

The login defaults select the corresponding group role at connection time. Objects created by `ops_migrator` therefore belong to the durable `ops_migrator_role`; the application and reader cannot inherit the owner's privileges.

The role bootstrap temporarily grants the schema-owner role to the existing Grafana admin because PostgreSQL requires the current user to be able to `SET ROLE` before transferring schema ownership. The last statement revokes that membership; a failed bootstrap must be rerun through the revocation before continuing.

## Migrations and grants

Apply every migration as the dedicated migrator before deploying the service:

```bash
PGPASSWORD="$OPS_MIGRATOR_PW" uv run --project infra/ops ops-workflow migrate \
  --database-url postgresql://ops_migrator@127.0.0.1:55439/ops
unset OPS_MIGRATOR_PW OPS_APP_PW OPS_READER_PW
```

The migration runner takes an advisory lock, records each file digest, and rejects changed applied files. Default privileges grant `ops_app_role` DML on tables and sequence use; the runtime never owns or creates schema objects.

## Verify the privilege boundaries

Connect as each login rather than relying on the owner to emulate its membership. The reader's positive checks must succeed:

```sql
SELECT session_user, current_user;
SELECT count(*) FROM public.alert_instance;
SELECT count(*) FROM public.alert_rule;
```

`current_user` must be `ops_grafana_reader_role`. Each negative check must fail:

```sql
UPDATE public.alert_instance SET current_state = current_state;
SELECT count(*) FROM public.user;
CREATE TABLE public.reader_must_not_create (id integer);
```

Similarly, `ops_app` must be able to read and update workflow rows but must fail to create a table or read Grafana. `ops_migrator` is the only ops identity with schema-creation privileges. None of the three login users should be a member of `cloudsqlsuperuser`.

Grafana owns its alert tables. If a future Grafana migration recreates them with different grants, rerun `ops_roles.sql` and reverify this policy before restarting the poller. The adapter fails closed on an incompatible schema or serialization.
