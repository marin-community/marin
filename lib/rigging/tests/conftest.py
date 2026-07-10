# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import rigging.filesystem.cluster_config as cluster_config
from rigging.config_discovery import find_project_root


@pytest.fixture(autouse=True)
def _hermetic_cluster_config(monkeypatch):
    """Isolate cluster-config resolution from the host's ambient environment.

    ``~/.config/marin/clusters`` is a real per-user override some dev hosts carry
    for connecting to a live Iris cluster. It takes priority over the
    repo-committed ``config/`` dir, and a `data:`-less document there (e.g. a
    pure Iris connection config) silently swaps the loaded ``DataConfig`` out
    from under any test that expects the committed cluster layout — the same
    class of host-dependent flake as an unmocked GCE metadata lookup. Tests see
    only the repo-committed and bundled config dirs.
    """
    monkeypatch.setattr(
        cluster_config,
        "MARIN_CLUSTER_CONFIG_DIRS",
        tuple(p for p in cluster_config.MARIN_CLUSTER_CONFIG_DIRS if p != cluster_config.PER_USER_CLUSTER_CONFIG_DIR),
    )
    find_project_root.cache_clear()
    cluster_config.reset_data_config_cache()
    yield
    find_project_root.cache_clear()
    cluster_config.reset_data_config_cache()
