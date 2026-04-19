# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hydrate real ScaleGroupConfig protos from the cluster YAML for the load-test.

Stage 1 fabricated CPU placeholders for every persisted scaling-groups row.
For Stage 2 we need the *real* expanded set — per-(size, zone) TPU groups with
topology-derived ``num_vms`` / ``device_count`` / ``device_variant`` — so that
demand routing, scale-up targeting, and failure injection all use the same
shapes as production.

Re-use ``iris.cluster.config.load_config`` so we do not reimplement
``_expand_tpu_pools``. The return value is a ``dict`` in the same shape
``create_autoscaler`` expects (``name -> ScaleGroupConfig``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from iris.cluster.config import load_config
from iris.rpc import config_pb2

logger = logging.getLogger(__name__)

DEFAULT_MARIN_YAML = Path(__file__).resolve().parents[3] / "examples" / "marin.yaml"

# Captured controller sqlite snapshot used by default. The harness copies this
# to a temp dir before booting so the original is never mutated. Fetched once
# from GCS by hand; see `.agents/projects/autoscaler-loadtest.md`.
DEFAULT_SNAPSHOT_PATH = Path("/tmp/iris-marin.sqlite3")


def load_scale_group_protos(cluster_config_path: Path) -> dict[str, config_pb2.ScaleGroupConfig]:
    """Return the fully-expanded scale-group protos defined by *cluster_config_path*.

    Uses the same pipeline the controller uses at startup
    (:func:`iris.cluster.config.load_config`), which applies defaults,
    expands ``tpu_pools`` and multi-zone groups, and validates resources.
    """
    cluster_config = load_config(cluster_config_path)
    return dict(cluster_config.scale_groups)


def zones_for(cluster_config_path: Path) -> list[str]:
    """Return all GCP zones referenced by the YAML, after expansion."""
    cluster_config = load_config(cluster_config_path)
    if not cluster_config.platform.HasField("gcp"):
        return []
    return list(cluster_config.platform.gcp.zones)
