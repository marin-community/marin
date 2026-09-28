# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Store the optional application health policy for each job."""


def migrate(raw_conn) -> None:
    columns = {row[1] for row in raw_conn.execute("PRAGMA table_info(job_config)")}
    if "health_check_json" not in columns:
        raw_conn.execute("ALTER TABLE job_config ADD COLUMN health_check_json VARCHAR")
