# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for Marin's shared Cloud SQL metadata instance."""

import pulumi
import pulumi_gcp as gcp
from iac.gcp.cloud_sql import CloudSqlPostgres, CloudSqlPostgresArgs
from iac.gcp.cloud_sql_ops import configure_ops_database

PROJECT = "hai-gcp-models"
REGION = "us-central1"
INSTANCE = "marin-metadata"
EVAL_BUCKET = "marin-eval-metadata"
OPS_PASSWORD_GENERATION = 1
ADMIN_PASSWORD_GENERATION = 1
PULUMI_ADMIN_SECRET = "cloudsql-pulumi-admin-password"


def main() -> None:
    gcp_provider = gcp.Provider("gcp", project=PROJECT)
    postgres = CloudSqlPostgres(
        "metadata",
        CloudSqlPostgresArgs(
            project=PROJECT,
            region=REGION,
            instance_name=INSTANCE,
            databases=("grafana", "evals", "ops"),
            password_secrets=(
                "cloudsql-grafana-password",
                "cloudsql-evals-password",
                "cloudsql-ops-app-password",
                "cloudsql-ops-migrator-password",
                PULUMI_ADMIN_SECRET,
            ),
        ),
        gcp_provider=gcp_provider,
    )

    bucket = gcp.storage.Bucket(
        "eval-metadata",
        name=EVAL_BUCKET,
        project=PROJECT,
        location="US-CENTRAL1",
        uniform_bucket_level_access=True,
        public_access_prevention="enforced",
        opts=pulumi.ResourceOptions(provider=gcp_provider),
    )

    configure_ops_database(
        project=PROJECT,
        postgres=postgres,
        gcp_provider=gcp_provider,
        admin_password_generation=ADMIN_PASSWORD_GENERATION,
        admin_secret=PULUMI_ADMIN_SECRET,
        ops_password_generation=OPS_PASSWORD_GENERATION,
    )

    pulumi.export("connection_name", postgres.connection_name)
    pulumi.export("public_ip", postgres.public_ip)
    pulumi.export("eval_bucket", bucket.name)
    pulumi.export("ops_password_generation", OPS_PASSWORD_GENERATION)


main()
