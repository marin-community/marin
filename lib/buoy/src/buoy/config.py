# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Process-wide configuration for buoy, resolved once from the environment.

Critical knobs are explicit env vars (no silent defaults buried in call sites).
The defaults are tuned for the single-replica Iris service; tests override
``BUOY_CACHE_ROOT`` to point at a local directory.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from rigging.filesystem import marin_temp_bucket

# wandb artifact types that carry a JAX/xprof profile. Run families differ:
# `jax_profile` ships the xprof logdir pre-unpacked (plugins/profile/<ts>/….xplane.pb
# at the root); `profiler` ships a single .tgz of the same logdir that we extract.
PROFILE_ARTIFACT_TYPES = frozenset({"jax_profile", "profiler"})

# GCS cache TTL (days) under marin_temp_bucket; the cache is refetchable, not an
# archive — anything evicted is re-pulled from wandb on next view.
CACHE_TTL_DAYS = 30
CACHE_PREFIX = "buoy"

# Rows per history parquet part. Bounds peak memory during the streaming mirror:
# a full-fidelity run is ~10^5 steps x ~400 keys, far too large to hold at once.
HISTORY_PAGE_ROWS = 5000


@dataclass(frozen=True)
class BuoyConfig:
    """Resolved runtime configuration."""

    cache_root: str
    default_entity: str
    xprof_bin: str | None
    max_xprof_procs: int
    local_profile_dir: str

    @staticmethod
    def from_env() -> BuoyConfig:
        cache_root = os.environ.get("BUOY_CACHE_ROOT") or marin_temp_bucket(CACHE_TTL_DAYS, CACHE_PREFIX)
        return BuoyConfig(
            cache_root=cache_root.rstrip("/"),
            default_entity=os.environ.get("BUOY_DEFAULT_ENTITY", "marin-community"),
            xprof_bin=os.environ.get("BUOY_XPROF_BIN") or shutil.which("xprof"),
            max_xprof_procs=int(os.environ.get("BUOY_MAX_XPROF_PROCS", "4")),
            local_profile_dir=os.environ.get("BUOY_LOCAL_PROFILE_DIR", "/tmp/buoy-profiles"),
        )
