# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for Marin's shared agent-context database.

Declares the `marin-context` PostgreSQL instance through `iac.gcp.cloud_sql.CloudSqlPostgres`
— one instance carrying the `context` database (the marinmirror github+discord corpus in
`chunks`, plus the agents' `work_log`) — and the `marin-context-sync` scheduled Cloud Run job
that keeps `chunks` current against marinmirror (see sync/main.py). SQL users, secret values,
and the `work_log` schema are set out-of-band; see README.md.

Runs on the shared repo venv (plain `python` runtime), which is where `iac` and the Pulumi GCP
provider live; `uv sync --all-packages --extra deploy` first. See README.md.
"""

import pulumi
import pulumi_gcp as gcp
from iac.gcp.cloud_run import SecretEnv
from iac.gcp.cloud_run_job import ScheduledCloudRunJob, ScheduledCloudRunJobArgs
from iac.gcp.cloud_sql import CloudSqlPostgres, CloudSqlPostgresArgs

PROJECT = "hai-gcp-models"
REGION = "us-central1"
INSTANCE = "marin-context"
DATABASE = "context"
CONNECTION_NAME = f"{PROJECT}:{REGION}:{INSTANCE}"
# Shared read/append SQL user for everyone's agents; password value set out-of-band.
AGENTS_PASSWORD_SECRET = "cloudsql-agents-password"
# The postgres admin password; the sync job connects with it to own the chunks schema.
ADMIN_PASSWORD_SECRET = "cloudsql-postgres-marin-context"
# marinmirror bearer token: a GitHub PAT (read:org) of an Open-Athena member; value set
# out-of-band.
MARINMIRROR_TOKEN_SECRET = "marinmirror-token"


def main() -> None:
    # Adoption recon: `pulumi config set marin-iac:import true` stamps import_ on every
    # pre-existing resource the program declares (one-shot; see README.md).
    adopt = pulumi.Config("marin-iac").get_bool("import") or False
    provider = gcp.Provider("gcp", project=PROJECT)

    postgres = CloudSqlPostgres(
        "context",
        CloudSqlPostgresArgs(
            project=PROJECT,
            region=REGION,
            instance_name=INSTANCE,
            databases=(DATABASE,),
            password_secrets=(AGENTS_PASSWORD_SECRET, ADMIN_PASSWORD_SECRET),
            adopt=adopt,
        ),
        gcp_provider=provider,
    )

    gcp.secretmanager.Secret(
        f"secret-{MARINMIRROR_TOKEN_SECRET}",
        secret_id=MARINMIRROR_TOKEN_SECRET,
        project=PROJECT,
        replication=gcp.secretmanager.SecretReplicationArgs(
            auto=gcp.secretmanager.SecretReplicationAutoArgs(),
        ),
        opts=pulumi.ResourceOptions(
            provider=provider,
            import_=f"projects/{PROJECT}/secrets/{MARINMIRROR_TOKEN_SECRET}" if adopt else None,
        ),
    )

    sync = ScheduledCloudRunJob(
        "sync",
        ScheduledCloudRunJobArgs(
            project=PROJECT,
            region=REGION,
            job_name="marin-context-sync",
            build_context="./sync",
            schedule="0 */6 * * *",
            env={
                "CLOUDSQL_CONNECTION": CONNECTION_NAME,
                "PGDATABASE": DATABASE,
                "PGUSER": "postgres",
            },
            secrets=(
                SecretEnv(name="PGPASSWORD", secret=ADMIN_PASSWORD_SECRET),
                SecretEnv(name="MARINMIRROR_TOKEN", secret=MARINMIRROR_TOKEN_SECRET),
            ),
            cloudsql_instances=(CONNECTION_NAME,),
            # The sync holds a ~650 MB corpus download plus batch buffers in memory.
            memory="2Gi",
        ),
        gcp_provider=provider,
    )

    pulumi.export("connection_name", postgres.connection_name)
    pulumi.export("public_ip", postgres.public_ip)
    pulumi.export("sync_job", sync.job_name)


main()
