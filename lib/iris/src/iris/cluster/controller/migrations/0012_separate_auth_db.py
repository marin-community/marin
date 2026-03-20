# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marker migration for the auth DB separation.

The actual data migration (copying api_keys and controller_secrets from main
to the auth database) is handled by ControllerDB._init_auth_schema(), which
runs before migrations. This file exists only so the migration numbering
stays contiguous.
"""

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    pass
