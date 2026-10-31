# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rename persisted task services and the primary selector to containers.

The environment JSON is shared by local dispatch and federation reconstruction.
Keep container definitions intact and leave already-renamed rows untouched so an
interrupted migration can resume safely.
"""

import json


def migrate(raw_conn) -> None:
    rows = raw_conn.execute("SELECT job_id, environment_json FROM job_config").fetchall()
    for job_id, environment_json in rows:
        environment = json.loads(environment_json)
        if "services" not in environment and "primary_service" not in environment:
            continue
        if "services" in environment:
            environment["containers"] = environment.pop("services")
        if "primary_service" in environment:
            environment["primary_container"] = environment.pop("primary_service")
        raw_conn.execute(
            "UPDATE job_config SET environment_json = ? WHERE job_id = ?",
            (json.dumps(environment), job_id),
        )
