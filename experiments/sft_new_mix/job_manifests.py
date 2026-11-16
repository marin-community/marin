# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Store campaign jobs outside actor messages and verify references at workers."""

import hashlib
import json

from rigging.filesystem.storage_path import StoragePath


def stage_job_manifest(job: dict) -> dict:
    content = json.dumps(job, sort_keys=True).encode()
    sha256 = hashlib.sha256(content).hexdigest()
    path = StoragePath(job["output_prefix"]) / "jobs" / (sha256 + ".json")
    path.write_bytes(content)
    return {"job_uri": str(path), "job_sha256": sha256}


def read_job_manifest(reference: dict) -> dict:
    content = StoragePath(reference["job_uri"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != reference["job_sha256"]:
        raise ValueError("Job manifest digest changed")
    return json.loads(content)
