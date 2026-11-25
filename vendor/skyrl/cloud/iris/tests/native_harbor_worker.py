# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import socket
import sys
import traceback

from harbor.models.job.config import JobConfig

expected = json.loads(sys.argv[2])
connection = socket.socket(fileno=int(sys.argv[1]))
with connection.makefile("rwb") as stream:
    for line in stream:
        message = json.loads(line)
        try:
            job = JobConfig.model_validate(message["cohort"]["job"])
            if expected is None:
                assert job.telemetry is None
            else:
                assert job.telemetry is not None
                actual = {
                    "endpoint": job.telemetry.endpoint.get_secret_value(),
                    "root_run_uid": job.telemetry.root_run_uid,
                    "execution_uid": job.telemetry.execution_uid,
                    "serving_job_id": job.telemetry.serving_job_id,
                }
                assert actual == expected, (actual, expected)
            response = {"id": message["id"], "metrics": {"num_attempted": 1, "num_scored": 1, "reward_sum": 1}}
        except Exception:
            response = {"id": message["id"], "error": traceback.format_exc()}
        stream.write((json.dumps(response) + "\n").encode())
        stream.flush()
