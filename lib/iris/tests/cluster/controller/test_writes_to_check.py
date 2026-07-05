# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Stage 12 ``@writes_to`` owned-table startup check."""

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.schema import endpoints_table, meta_table
from iris.cluster.controller.writes import (
    REGISTERED_WRITE_FUNCTIONS,
    ConfigurationError,
    validate,
    writes_to,
)


@pytest.fixture
def fresh_db() -> Iterator[ControllerDB]:
    """Yield a ControllerDB so Projection instances can be constructed."""
    tmp = Path(tempfile.mkdtemp(prefix="iris_writes_check_"))
    db = ControllerDB(db_dir=tmp)
    try:
        yield db
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def projections_built(fresh_db: ControllerDB) -> Iterator[ControllerDB]:
    """Construct one of each Projection so the cache registry exposes their owned tables."""
    EndpointsProjection(fresh_db)
    yield fresh_db


@pytest.fixture
def registry_isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Snapshot/restore REGISTERED_WRITE_FUNCTIONS so per-test decorations don't leak."""
    snapshot = list(REGISTERED_WRITE_FUNCTIONS)
    yield
    REGISTERED_WRITE_FUNCTIONS[:] = snapshot


def test_violation_detected(projections_built, registry_isolated):
    """A free function writing to a Projection-owned table must trip the check."""

    @writes_to(endpoints_table)
    def rogue_write(tx) -> None:
        pass

    with pytest.raises(ConfigurationError) as exc_info:
        validate(projections_built.caches)

    msg = str(exc_info.value)
    assert "rogue_write" in msg
    assert "endpoints" in msg
    assert "EndpointsProjection" in msg


def test_cascade_violation_detected(projections_built, registry_isolated):
    """``cascades_into`` over a Projection-owned table is treated as a write."""

    @writes_to(meta_table, cascades_into=(endpoints_table,))
    def rogue_cascade(tx) -> None:
        pass

    with pytest.raises(ConfigurationError) as exc_info:
        validate(projections_built.caches)

    msg = str(exc_info.value)
    assert "rogue_cascade" in msg
    assert "endpoints" in msg
    assert "EndpointsProjection" in msg


def test_projection_method_allowed(projections_built, registry_isolated):
    """A function whose qualified name belongs to the owning Projection is exempt."""

    @writes_to(endpoints_table)
    def fake_method(tx) -> None:
        pass

    # Simulate ``EndpointsProjection.some_write`` so the qualname check exempts it.
    fake_method.__qualname__ = "EndpointsProjection.some_write"

    # Must not raise.
    validate(projections_built.caches)


def test_clean_codebase_passes(fresh_db):
    """The shipping codebase must satisfy the invariant.

    ``ControllerDB.__init__`` already ran the check (and would have raised
    on construction if a violation existed); re-running it here surfaces
    regressions as a normal assertion rather than as fixture-setup failure.
    """
    EndpointsProjection(fresh_db)
    validate(fresh_db.caches)
