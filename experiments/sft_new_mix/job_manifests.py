# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Store campaign jobs outside actor messages and verify references at workers."""

import hashlib
import json

from rigging.filesystem.storage_path import StoragePath

from experiments.sft_new_mix.exp3_runtime import read_pinned


def stage_job_manifest(job: dict) -> dict:
    content = json.dumps(job, sort_keys=True).encode()
    sha256 = hashlib.sha256(content).hexdigest()
    path = StoragePath(job["output_prefix"]) / "jobs" / (sha256 + ".json")
    path.write_bytes(content)
    return {"job_uri": str(path), "job_sha256": sha256}


def read_job_manifest(reference: dict) -> dict:
    return json.loads(read_pinned(reference["job_uri"], reference["job_sha256"]))
