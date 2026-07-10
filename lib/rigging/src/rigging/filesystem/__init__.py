# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin filesystem: the :class:`StoragePath` value type, cluster data config,
storage-prefix/region resolution, region-local temp storage, cross-region read
guards, guarded fsspec entry points, and the cross-region mirror filesystem.

The package is split into focused modules whose cross-imports run one direction:

- :mod:`rigging.filesystem.storage_path` — the :class:`StoragePath` value type and
  its verbs, plus ``prefix_join``, ``rebase_file_path``, ``split_gcs_path``. A leaf
  module: its verbs import the factory at call time, so it depends on nothing else
  in the package.
- :mod:`rigging.filesystem.cluster_config` — the cluster :class:`DataConfig`,
  region/prefix resolution, region-local temp storage, and GCS-location utils.
- :mod:`rigging.filesystem.cross_region` — the :class:`TransferBudget` and the
  :class:`CrossRegionGuardedFS` read guard, plus ``record_transfer``.
- :mod:`rigging.filesystem.factory` — the guarded ``url_to_fs`` / ``open_url`` /
  ``filesystem`` entry points and ``atomic_rename``.
- :mod:`rigging.filesystem.mirror` — the ``mirror://`` :class:`MirrorFileSystem`.

This module re-exports the public API of the first four, so
``from rigging.filesystem import …`` keeps working. ``MirrorFileSystem`` is the
one component that imports :mod:`rigging.distributed_lock`; the ``mirror://``
protocol is registered *lazily* (by class path) so importing this package does
not import ``mirror`` — and therefore does not pull ``distributed_lock`` under
the package, leaving it free to depend on :class:`StoragePath` (and the other
value/config modules) without an import cycle. Access the class directly via
``rigging.filesystem.mirror`` when you need it by name.
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

# Register the mirror:// protocol lazily by class path. fsspec imports
# rigging.filesystem.mirror on demand the first time a mirror filesystem is
# constructed; until then, distributed_lock stays out from under this package.
fsspec.register_implementation("mirror", "rigging.filesystem.mirror.MirrorFileSystem")
