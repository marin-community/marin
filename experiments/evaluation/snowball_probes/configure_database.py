# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grant an existing IAM database login the lab's restricted worker operations."""

import argparse
from uuid import UUID

from google.cloud.sql.connector import Connector
from sqlalchemy import create_engine

from infra.marina.applets.snowball_probes.server.worker_sql import grant_worker_functions
from infra.marina.database_grants import ADMIN_USER, admin_password


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--applet", required=True, type=UUID)
    parser.add_argument("--worker-user", required=True)
    args = parser.parse_args()
    password = admin_password()
    with Connector(refresh_strategy="lazy", timeout=30) as connector:
        engine = create_engine(
            "postgresql+pg8000://",
            hide_parameters=True,
            creator=lambda: connector.connect(
                args.connection, "pg8000", user=ADMIN_USER, password=password, db=args.database
            ),
        )
        try:
            with engine.begin() as connection:
                grant_worker_functions(connection, args.applet, args.worker_user)
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()
