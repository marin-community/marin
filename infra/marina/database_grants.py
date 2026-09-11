# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grant the runtime accounts what Cloud SQL IAM users do not get by default.

Run by the Pulumi program after the databases and IAM users exist, as the native
``pulumi_db_admin`` user whose password lives in Secret Manager. A Cloud SQL IAM user can
connect but cannot create schemas, so the Marina service account gets every privilege on
the ``marina`` database. This script also installs the restricted function that creates
one role and schema for a generated applet UUID. The Loom VM gets CREATE on ``context``'s
public schema for the codehealth workbench tables. It also installs the pgvector extension
Echo needs, which only the Cloud SQL superuser may do. Every statement is idempotent.

    uv run infra/marina/database_grants.py
"""

import subprocess

import sqlalchemy
from google.cloud.sql.connector import Connector
from marina.database_setup import applet_provisioning_statements

PROJECT = "hai-gcp-models"
CONNECTION_NAME = f"{PROJECT}:us-central1:marin-metadata"
ADMIN_USER = "pulumi_db_admin"
ADMIN_PASSWORD_SECRET = "cloudsql-pulumi-admin-password"
MARINA_SERVICE_ROLE = "marina@hai-gcp-models.iam"
GRANTS = {
    "marina": [
        f'GRANT ALL PRIVILEGES ON DATABASE marina TO "{MARINA_SERVICE_ROLE}"',
        # Only the Cloud SQL superuser can install extensions; Echo's search needs pgvector.
        "CREATE EXTENSION IF NOT EXISTS vector",
        *applet_provisioning_statements(MARINA_SERVICE_ROLE),
    ],
    "context": ['GRANT CREATE ON SCHEMA public TO "loom-vm@hai-gcp-models.iam"'],
}


def admin_password() -> str:
    command = ["gcloud", "secrets", "versions", "access", "latest", f"--secret={ADMIN_PASSWORD_SECRET}"]
    return subprocess.run(
        [*command, f"--project={PROJECT}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main() -> None:
    password = admin_password()
    with Connector(refresh_strategy="lazy") as connector:
        for database, statements in GRANTS.items():
            engine = sqlalchemy.create_engine(
                "postgresql+pg8000://",
                creator=lambda database=database: connector.connect(
                    CONNECTION_NAME, "pg8000", user=ADMIN_USER, password=password, db=database
                ),
            )
            with engine.begin() as conn:
                for statement in statements:
                    conn.execute(sqlalchemy.text(statement))
                    print(f"{database}: {statement}")
            engine.dispose()


if __name__ == "__main__":
    main()
