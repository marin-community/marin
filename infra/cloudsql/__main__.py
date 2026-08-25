# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for Marin's shared Cloud SQL metadata instance.

Declares the `marin-metadata` PostgreSQL instance through the reusable
`iac.gcp.cloud_sql.CloudSqlPostgres` component — one instance carrying the `grafana` and
`evals` databases, with a Secret Manager secret shell per native user — plus the
`marin-eval-metadata` GCS bucket that holds eval run records. SQL users and secret values are
set out-of-band; see README.md.

Runs on the shared repo venv (plain `python` runtime), which is where `iac` and the Pulumi GCP
provider live; `uv sync --all-packages --extra deploy` first. See README.md.
"""

import pulumi
import pulumi_gcp as gcp
from iac.gcp.cloud_sql import CloudSqlPostgres, CloudSqlPostgresArgs

PROJECT = "hai-gcp-models"
REGION = "us-central1"
INSTANCE = "marin-metadata"
# GCS bucket for eval run records.
EVAL_BUCKET = "marin-eval-metadata"


def main() -> None:
    provider = gcp.Provider("gcp", project=PROJECT)
    postgres = CloudSqlPostgres(
        "metadata",
        CloudSqlPostgresArgs(
            project=PROJECT,
            region=REGION,
            instance_name=INSTANCE,
            databases=("grafana", "evals"),
            password_secrets=("cloudsql-grafana-password", "cloudsql-evals-password"),
        ),
        gcp_provider=provider,
    )

    bucket = gcp.storage.Bucket(
        "eval-metadata",
        name=EVAL_BUCKET,
        project=PROJECT,
        location="US-CENTRAL1",
        uniform_bucket_level_access=True,
        public_access_prevention="enforced",
        opts=pulumi.ResourceOptions(provider=provider),
    )

    pulumi.export("connection_name", postgres.connection_name)
    pulumi.export("public_ip", postgres.public_ip)
    pulumi.export("eval_bucket", bucket.name)


main()
