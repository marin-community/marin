# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pin a FineStore autotune manifest when a GPU job requests a tuned set."""

from finestore.commit import EMPTY_SNAPSHOT_ID, read_snapshot
from finestore.layout import FineStoreLayout

from iris.jax.autotune_snapshot import IRIS_AUTOTUNE_SNAPSHOT_ID, IRIS_AUTOTUNE_STORE_ROOT
from iris.rpc import job_pb2


def pin_autotune_snapshot(environment: job_pb2.EnvironmentConfig) -> None:
    """Record one manifest ID in the job environment before submission."""
    root = environment.env_vars.get(IRIS_AUTOTUNE_STORE_ROOT)
    if not root or IRIS_AUTOTUNE_SNAPSHOT_ID in environment.env_vars:
        return
    snapshot = read_snapshot(FineStoreLayout(root))
    environment.env_vars[IRIS_AUTOTUNE_SNAPSHOT_ID] = (
        snapshot.manifest.commit_id if snapshot.token else EMPTY_SNAPSHOT_ID
    )
