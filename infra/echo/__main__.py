# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for echo: Marin's shared agent-context database and its sync job.

Declares the `context` database on the shared `marin-metadata` Cloud SQL instance
(infra/cloudsql owns the instance), the `agents` and `echo_sync` SQL roles through the
PostgreSQL provider, their password secrets, and the `echo-sync` scheduled Cloud Run job
that keeps the corpus mirror current (sync/main.py). Tables and grants are applied by
migrate.py; see README.md for the deploy order.

The PostgreSQL provider connects through a locally running Cloud SQL auth proxy as
`pulumi_db_admin` (password in the cloudsql-pulumi-admin-password secret), so `pulumi up`
for this stack requires the proxy; see README.md.
"""

import pulumi
import pulumi_gcp as gcp
import pulumi_postgresql as postgresql
import pulumi_random as random_
from iac.gcp.cloud_run import SecretEnv
from iac.gcp.cloud_run_job import ScheduledCloudRunJob, ScheduledCloudRunJobArgs

PROJECT = "hai-gcp-models"
REGION = "us-central1"
INSTANCE = "marin-metadata"
CONNECTION_NAME = f"{PROJECT}:{REGION}:{INSTANCE}"
DATABASE = "context"
# Shared read/append SQL role for everyone's agents.
AGENTS_PASSWORD_SECRET = "cloudsql-agents-password"
# The sync job's writer role.
SYNC_PASSWORD_SECRET = "cloudsql-echo-sync-password"
# The PostgreSQL provider's admin role; password set out-of-band, shared with other
# provider users of marin-metadata.
ADMIN_PASSWORD_SECRET = "cloudsql-pulumi-admin-password"
# marinmirror bearer token: a GitHub PAT (read:org) of an Open-Athena member.
MARINMIRROR_TOKEN_SECRET = "marinmirror-token"


def password_secret(resource_name: str, secret_id: str, value: pulumi.Output[str], *, adopt: bool, provider) -> None:
    """A Secret Manager secret holding a role's password, with its current version."""
    secret = gcp.secretmanager.Secret(
        resource_name,
        secret_id=secret_id,
        project=PROJECT,
        replication=gcp.secretmanager.SecretReplicationArgs(auto=gcp.secretmanager.SecretReplicationAutoArgs()),
        opts=pulumi.ResourceOptions(
            provider=provider,
            import_=f"projects/{PROJECT}/secrets/{secret_id}" if adopt else None,
        ),
    )
    gcp.secretmanager.SecretVersion(
        f"{resource_name}-version",
        secret=secret.id,
        secret_data=value,
        opts=pulumi.ResourceOptions(provider=provider),
    )


def main() -> None:
    config = pulumi.Config("marin-iac")
    # Adoption recon: `pulumi config set marin-iac:import true` stamps import_ on
    # pre-existing secrets (one-shot; see infra/pulumi/README.md).
    adopt = config.get_bool("import") or False
    gcp_provider = gcp.Provider("gcp", project=PROJECT)

    pg_config = pulumi.Config("echo")
    admin_password = gcp.secretmanager.get_secret_version_output(
        secret=ADMIN_PASSWORD_SECRET, project=PROJECT, opts=pulumi.InvokeOptions(provider=gcp_provider)
    ).secret_data
    pg_provider = postgresql.Provider(
        "pg",
        host=pg_config.get("pg-host") or "127.0.0.1",
        port=pg_config.get_int("pg-port") or 5433,
        username="pulumi_db_admin",
        password=admin_password,
        sslmode="disable",  # the auth proxy carries TLS; the local hop is loopback
        superuser=False,
    )

    database = gcp.sql.Database(
        "context",
        name=DATABASE,
        instance=INSTANCE,
        project=PROJECT,
        opts=pulumi.ResourceOptions(provider=gcp_provider),
    )

    agents_password = random_.RandomPassword("agents-password", length=32, special=False)
    sync_password = random_.RandomPassword("sync-password", length=32, special=False)
    for role_name, password in (("agents", agents_password), ("echo_sync", sync_password)):
        postgresql.Role(
            role_name,
            name=role_name,
            login=True,
            password=password.result,
            opts=pulumi.ResourceOptions(provider=pg_provider, depends_on=[database]),
        )
    password_secret("agents-secret", AGENTS_PASSWORD_SECRET, agents_password.result, adopt=adopt, provider=gcp_provider)
    password_secret("sync-secret", SYNC_PASSWORD_SECRET, sync_password.result, adopt=False, provider=gcp_provider)

    gcp.secretmanager.Secret(
        "marinmirror-token",
        secret_id=MARINMIRROR_TOKEN_SECRET,
        project=PROJECT,
        replication=gcp.secretmanager.SecretReplicationArgs(auto=gcp.secretmanager.SecretReplicationAutoArgs()),
        opts=pulumi.ResourceOptions(
            provider=gcp_provider,
            import_=f"projects/{PROJECT}/secrets/{MARINMIRROR_TOKEN_SECRET}" if adopt else None,
        ),
    )

    sync = ScheduledCloudRunJob(
        "sync",
        ScheduledCloudRunJobArgs(
            project=PROJECT,
            region=REGION,
            job_name="echo-sync",
            build_context=".",
            dockerfile="sync/Dockerfile",
            # Cheap when nothing changed: the job exits on the manifest watermark check, and
            # only a new upstream corpus build (~every 90 min) triggers the full download+upsert.
            schedule="*/10 * * * *",
            env={
                "CLOUDSQL_CONNECTION": CONNECTION_NAME,
                "PGDATABASE": DATABASE,
                "PGUSER": "echo_sync",
            },
            secrets=(
                SecretEnv(name="PGPASSWORD", secret=SYNC_PASSWORD_SECRET),
                SecretEnv(name="MARINMIRROR_TOKEN", secret=MARINMIRROR_TOKEN_SECRET),
            ),
            cloudsql_instances=(CONNECTION_NAME,),
            # The sync holds a ~650 MB corpus download plus batch buffers in memory.
            memory="2Gi",
        ),
        gcp_provider=gcp_provider,
    )

    pulumi.export("connection_name", CONNECTION_NAME)
    pulumi.export("database", database.name)
    pulumi.export("sync_job", sync.job_name)


main()
