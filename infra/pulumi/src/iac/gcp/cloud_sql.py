# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A shared Cloud SQL for PostgreSQL instance for Marin's internal metadata.

The generic shape behind Marin's small managed-Postgres needs: one regional Cloud SQL
instance carrying several logical databases (Grafana's state, the eval run records), reached
only through the Cloud SQL connector/auth-proxy with IAM authentication — the public IP admits
no authorized network, so nothing dials it directly.

The component owns the instance, one database per name, and one Secret Manager secret *shell*
per native database user. It never creates the SQL users or holds their passwords: a password
passed to Pulumi would land in stack state. Users and secret values are set out-of-band with
`gcloud` (see infra/cloudsql/README.md), and the secret shells created here are what a consumer
mounts and reads.

Exposes ``connection_name`` (project:region:instance, the connector target) and ``public_ip``.
"""

from dataclasses import dataclass

import pulumi
import pulumi_gcp as gcp

POSTGRES_VERSION = "POSTGRES_16"
# Cloud SQL flag that lets connections authenticate as IAM principals instead of only native
# password users; the connector uses it for password-less service-account access.
IAM_AUTHENTICATION_FLAG = "cloudsql.iam_authentication"
# Daily automated backup window (UTC); off-hours to avoid the busiest period.
BACKUP_START_TIME = "03:00"


@dataclass(frozen=True)
class CloudSqlPostgresArgs:
    project: str
    region: str
    instance_name: str

    # Logical databases created on the instance, one `gcp.sql.Database` each.
    databases: tuple[str, ...]
    # Secret Manager secret shells to create, one per native database user. The component
    # creates the empty secret and never its value; the password version is added out-of-band
    # alongside the matching `gcloud sql users create` (see the project README).
    password_secrets: tuple[str, ...]

    # Machine tier. db-g1-small is a shared-core instance sized for low-traffic metadata.
    tier: str = "db-g1-small"
    # Initial data-disk size in GB; the instance autoresizes upward from here under pressure.
    disk_size: int = 10


class CloudSqlPostgres(pulumi.ComponentResource):
    """Provision a PostgreSQL Cloud SQL instance, its databases, and per-user secret shells.

    Exposes ``connection_name`` (the ``project:region:instance`` connector target) and
    ``public_ip``. It does not create SQL users — those and the secret values are set
    out-of-band so no password enters Pulumi state.
    """

    connection_name: pulumi.Output[str]
    public_ip: pulumi.Output[str]

    def __init__(
        self,
        name: str,
        args: CloudSqlPostgresArgs,
        *,
        gcp_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:gcp:CloudSqlPostgres", name, None, opts)
        child = pulumi.ResourceOptions(parent=self, provider=gcp_provider)

        instance = gcp.sql.DatabaseInstance(
            "instance",
            name=args.instance_name,
            project=args.project,
            region=args.region,
            database_version=POSTGRES_VERSION,
            # Guard the metadata store: a stray `pulumi destroy`/rename must not drop the DB.
            deletion_protection=True,
            settings=gcp.sql.DatabaseInstanceSettingsArgs(
                tier=args.tier,
                # Enterprise (not Enterprise Plus) is the edition that admits shared-core
                # tiers like db-g1-small; the provider otherwise defaults to Plus.
                edition="ENTERPRISE",
                availability_type="ZONAL",
                disk_size=args.disk_size,
                disk_autoresize=True,
                ip_configuration=gcp.sql.DatabaseInstanceSettingsIpConfigurationArgs(
                    # Public IP on, but no authorized networks: nothing reaches it directly.
                    # Consumers connect through the Cloud SQL connector (IAM-authenticated).
                    ipv4_enabled=True,
                ),
                backup_configuration=gcp.sql.DatabaseInstanceSettingsBackupConfigurationArgs(
                    enabled=True,
                    start_time=BACKUP_START_TIME,
                ),
                database_flags=[
                    gcp.sql.DatabaseInstanceSettingsDatabaseFlagArgs(
                        name=IAM_AUTHENTICATION_FLAG,
                        value="on",
                    )
                ],
            ),
            opts=child,
        )

        for database in args.databases:
            gcp.sql.Database(
                f"db-{database}",
                name=database,
                instance=instance.name,
                project=args.project,
                opts=child,
            )

        for secret in args.password_secrets:
            gcp.secretmanager.Secret(
                f"secret-{secret}",
                secret_id=secret,
                project=args.project,
                replication=gcp.secretmanager.SecretReplicationArgs(
                    auto=gcp.secretmanager.SecretReplicationAutoArgs(),
                ),
                opts=child,
            )

        self.connection_name = instance.connection_name
        self.public_ip = instance.public_ip_address
        self.register_outputs({"connection_name": self.connection_name, "public_ip": self.public_ip})
