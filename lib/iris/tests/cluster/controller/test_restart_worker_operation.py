# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for ``autoscaler.operations.restart_worker`` DB lookup.

The SQLAlchemy Core migration (#5651) removed ``Tx.raw`` but this call site was
missed and stayed wired to the old API, so any invocation of
``cluster controller worker-restart`` raised
``AttributeError: 'Tx' object has no attribute 'raw'`` on the controller. These
tests pin the SA Core lookup so the regression cannot return silently.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
from iris.cluster.controller.autoscaler.operations import restart_worker
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema import workers_table
from iris.cluster.types import WorkerId
from sqlalchemy import insert


@pytest.fixture
def db():
    tmp = Path(tempfile.mkdtemp(prefix="iris_restart_test_"))
    controller_db = ControllerDB(db_dir=tmp)
    try:
        yield controller_db
    finally:
        controller_db.close()
        shutil.rmtree(tmp, ignore_errors=True)


def _insert_worker(controller_db: ControllerDB, worker_id: str, slice_id: str, scale_group: str) -> None:
    with controller_db.transaction() as tx:
        tx.execute(
            insert(workers_table).values(
                worker_id=WorkerId(worker_id),
                address=f"{worker_id}.example:10001",
                slice_id=slice_id,
                scale_group=scale_group,
            )
        )


def test_restart_worker_raises_when_row_missing(db):
    with pytest.raises(ValueError, match="not found in workers table"):
        restart_worker(groups={}, db=db, worker_id="ghost-worker", build_worker_config=lambda g: None)


def test_restart_worker_query_returns_row_and_advances_past_lookup(db):
    """A worker with slice_id+scale_group is returned by the SA Core query; the
    call then fails at the next step (scale group missing) rather than at the
    DB lookup. This proves the lookup itself works."""
    _insert_worker(db, worker_id="w-alive", slice_id="slice-1", scale_group="cpu-group")

    with pytest.raises(ValueError, match="Scale group cpu-group not found"):
        restart_worker(groups={}, db=db, worker_id="w-alive", build_worker_config=lambda g: None)


def test_restart_worker_skips_rows_without_slice(db):
    """Rows with empty slice_id are filtered out (a worker that has not been
    placed onto a slice cannot be restarted)."""
    _insert_worker(db, worker_id="w-unplaced", slice_id="", scale_group="cpu-group")

    with pytest.raises(ValueError, match="not found in workers table"):
        restart_worker(groups={}, db=db, worker_id="w-unplaced", build_worker_config=lambda g: None)
