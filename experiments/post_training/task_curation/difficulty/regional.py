# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regional worker and immutable metadata guards shared by campaign staging."""

import json

from iris.cluster.client.job_info import get_job_info
from rigging.filesystem.storage_path import StoragePath

REGION = "us-central2"


def regional_worker() -> None:
    """Large reads are permitted only from a regional Iris worker."""
    info = get_job_info()
    if info is None or info.worker_region != REGION:
        raise ValueError(f"Staging/serving requires an Iris worker in {REGION}")


def immutable_json(path: StoragePath, value: dict | list) -> None:
    payload = json.dumps(value, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != payload:
        raise ValueError(f"Immutable staging receipt differs: {path}")
    path.parent.mkdirs()
    if not path.exists():
        path.write_text(payload)
    if path.read_text() != payload:
        raise ValueError(f"Staging readback differs: {path}")
