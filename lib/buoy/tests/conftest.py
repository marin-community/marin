# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for buoy tests.

The wandb public API is replaced by an in-memory ``FakeApi`` (see ``fakes.py``)
so the mirror layer runs against canned history/config/artifacts without network
or auth. The cache is a local ``tmp_path`` directory (fsspec resolves a bare path
to the local filesystem), so no GCS is touched.
"""

from __future__ import annotations

import pytest
from buoy.config import BuoyConfig
from fakes import FakeApi, FakeRun


@pytest.fixture
def cfg(tmp_path) -> BuoyConfig:
    return BuoyConfig(
        cache_root=str(tmp_path / "cache"),
        default_entity="marin-community",
        xprof_bin=None,
        max_xprof_procs=2,
        local_profile_dir=str(tmp_path / "profiles"),
    )


@pytest.fixture
def patch_wandb(monkeypatch):
    """Return a factory that installs a FakeApi wrapping the given FakeRun."""

    def _install(run: FakeRun) -> FakeApi:
        api = FakeApi(run)
        monkeypatch.setattr("buoy.mirror.wandb.Api", lambda: api)
        return api

    return _install


@pytest.fixture
def profile_logdir(tmp_path) -> str:
    """A minimal local xprof logdir to be mirrored as a jax_profile artifact."""
    d = tmp_path / "logdir" / "plugins" / "profile" / "2026_01_01_00_00_00"
    d.mkdir(parents=True)
    (d / "host0.xplane.pb").write_bytes(b"\x00\x01profile-bytes")
    return str(tmp_path / "logdir")
