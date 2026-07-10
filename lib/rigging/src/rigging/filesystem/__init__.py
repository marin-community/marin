# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin filesystem: storage paths, cluster data config, and cross-region I/O.

Focused submodules with one-directional imports:

- ``storage_path`` — the :class:`StoragePath` value type and its verbs, plus
  ``prefix_join``, ``rebase_file_path``, ``split_gcs_path``.
- ``cluster_config`` — the cluster :class:`DataConfig`, region/prefix resolution,
  region-local temp storage, and GCS-location utils.
- ``cross_region`` — the :class:`TransferBudget` and :class:`CrossRegionGuardedFS`.
- ``factory`` — the guarded ``url_to_fs`` / ``open_url`` / ``filesystem`` entry
  points and ``atomic_rename``.
- ``mirror`` — the ``mirror://`` :class:`MirrorFileSystem`.
- ``distributed_lock`` — lease-based distributed locks (used by ``mirror``).

This module re-exports the public API of the first four so
``from rigging.filesystem import …`` keeps working. ``mirror`` and
``distributed_lock`` are reached by their submodule path; the lazy ``mirror://``
registration below keeps them — and the ``botocore`` import they pull in for the
S3 lock backend — off a plain ``import rigging.filesystem``.
"""

import fsspec

from rigging.filesystem.cluster_config import (
    MARIN_CLUSTER_CONFIG_DIRS,
    PER_USER_CLUSTER_CONFIG_DIR,
    BucketSpec,
    DataConfig,
    StoreType,
    check_gcs_paths_same_region,
    check_path_in_region,
    collect_gcs_paths,
    data_config,
    get_bucket_location,
    load_cluster_config,
    marin_prefix,
    marin_region,
    marin_temp_bucket,
    region_from_metadata,
    region_from_prefix,
    reset_data_config_cache,
    s3_data_buckets,
    use_data_config,
)
from rigging.filesystem.cross_region import (
    CROSS_REGION_TRANSFER_LIMIT_BYTES,
    MARIN_CROSS_REGION_OVERRIDE_ENV,
    MARIN_MIRROR_BUDGET_ENV,
    CrossRegionGuardedFS,
    TransferBudget,
    TransferBudgetExceeded,
    cached_marin_region,
    is_cross_region_url,
    mirror_budget,
    record_transfer,
    reset_mirror_budget,
    set_mirror_budget,
)
from rigging.filesystem.factory import (
    atomic_rename,
    filesystem,
    is_remote_path,
    open_url,
    unique_temp_path,
    url_to_fs,
)
from rigging.filesystem.storage_path import (
    StoragePath,
    prefix_join,
    rebase_file_path,
    split_gcs_path,
)

# Register mirror:// by class path so fsspec imports rigging.filesystem.mirror
# (and the botocore its lock backend pulls in) only on first use, not on import.
fsspec.register_implementation("mirror", "rigging.filesystem.mirror.MirrorFileSystem")
