# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the controller startup DB restore/reuse decision.

Covers _prepare_local_db_dir and _apply_requested_rollback, which choose between
reusing the on-VM local DB, applying a requested rollback (restore the pre-deploy
checkpoint over the local DB and self-clear the rollout record), and restoring the
latest remote checkpoint on start.
"""

from pathlib import Path

import pytest
from iris.cluster.controller import main as controller_main
from iris.cluster.controller.rollout import RolloutPhase, RolloutRecord


def _seed_db(db_dir: Path, marker: str) -> None:
    db_dir.mkdir(parents=True, exist_ok=True)
    (db_dir / "controller.sqlite3").write_text(marker)


def _no_rollout_record(monkeypatch) -> None:
    monkeypatch.setattr(controller_main, "read_rollout_record", lambda remote: None)


def test_fresh_wipes_local_and_skips_restore(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    _seed_db(db_dir, "stale")
    restores = []
    monkeypatch.setattr(controller_main, "download_checkpoint_to_local", lambda *a, **k: restores.append(k) or True)

    controller_main._prepare_local_db_dir(db_dir, "gs://b/state", fresh=True, checkpoint_path=None)

    assert db_dir.exists() and not (db_dir / "controller.sqlite3").exists()
    assert restores == []


def test_rollback_requested_restores_checkpoint_and_marks_rolled_back(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    _seed_db(db_dir, "migrated")
    ckpt = "gs://b/state/controller-state/123"
    monkeypatch.setattr(
        controller_main,
        "read_rollout_record",
        lambda remote: RolloutRecord(phase=RolloutPhase.ROLLBACK_REQUESTED, image="img:old", rollback_checkpoint=ckpt),
    )

    def fake_restore(remote, dbdir, checkpoint_dir=None):
        Path(dbdir).mkdir(parents=True, exist_ok=True)
        (Path(dbdir) / "controller.sqlite3").write_text("restored")
        return True

    captured = {}
    monkeypatch.setattr(
        controller_main,
        "download_checkpoint_to_local",
        lambda remote, dbdir, checkpoint_dir=None: captured.update(checkpoint_dir=checkpoint_dir)
        or fake_restore(remote, dbdir, checkpoint_dir),
    )
    written = []
    monkeypatch.setattr(controller_main, "write_rollout_record", lambda remote, record: written.append(record))
    # The freshness heuristic must not run on the rollback path, or the migrated
    # local DB could win and defeat the rollback.
    monkeypatch.setattr(
        controller_main, "_local_db_epoch_ms", lambda d: pytest.fail("freshness checked on the rollback path")
    )

    controller_main._prepare_local_db_dir(db_dir, "gs://b/state", fresh=False, checkpoint_path=None)

    assert captured["checkpoint_dir"] == ckpt
    assert (db_dir / "controller.sqlite3").read_text() == "restored"
    assert len(written) == 1
    assert written[0].phase is RolloutPhase.ROLLED_BACK
    assert written[0].rollback_checkpoint is None


def test_rollback_requested_missing_checkpoint_falls_through(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    monkeypatch.setattr(
        controller_main,
        "read_rollout_record",
        lambda remote: RolloutRecord(phase=RolloutPhase.ROLLBACK_REQUESTED, image="img:old", rollback_checkpoint=None),
    )
    monkeypatch.setattr(controller_main, "_local_db_epoch_ms", lambda d: 200)
    monkeypatch.setattr(controller_main, "latest_checkpoint_epoch_ms", lambda r: 100)
    written = []
    monkeypatch.setattr(controller_main, "write_rollout_record", lambda remote, record: written.append(record))
    restores = []
    monkeypatch.setattr(controller_main, "download_checkpoint_to_local", lambda *a, **k: restores.append(1) or True)

    controller_main._prepare_local_db_dir(db_dir, "gs://b/state", fresh=False, checkpoint_path=None)

    # No checkpoint recorded: reuse the local DB, leave the record untouched.
    assert restores == []
    assert written == []


def test_local_db_reused_when_at_least_as_fresh_as_remote(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    _no_rollout_record(monkeypatch)
    monkeypatch.setattr(controller_main, "_local_db_epoch_ms", lambda d: 200)
    monkeypatch.setattr(controller_main, "latest_checkpoint_epoch_ms", lambda r: 100)
    restores = []
    monkeypatch.setattr(controller_main, "download_checkpoint_to_local", lambda *a, **k: restores.append(1) or True)

    controller_main._prepare_local_db_dir(db_dir, "gs://b/state", fresh=False, checkpoint_path=None)

    assert restores == []


def test_stale_local_db_restores_latest_remote(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    _no_rollout_record(monkeypatch)
    monkeypatch.setattr(controller_main, "_local_db_epoch_ms", lambda d: 100)
    monkeypatch.setattr(controller_main, "latest_checkpoint_epoch_ms", lambda r: 200)
    captured = {}
    monkeypatch.setattr(
        controller_main,
        "download_checkpoint_to_local",
        lambda remote, dbdir, checkpoint_dir=None: captured.update(checkpoint_dir=checkpoint_dir) or True,
    )

    controller_main._prepare_local_db_dir(db_dir, "gs://b/state", fresh=False, checkpoint_path=None)

    assert captured["checkpoint_dir"] is None


def test_checkpoint_path_restores_when_local_absent(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"  # absent -> _local_db_epoch_ms returns None -> restore
    _no_rollout_record(monkeypatch)
    monkeypatch.setattr(controller_main, "latest_checkpoint_epoch_ms", lambda r: 200)
    captured = {}
    monkeypatch.setattr(
        controller_main,
        "download_checkpoint_to_local",
        lambda remote, dbdir, checkpoint_dir=None: captured.update(checkpoint_dir=checkpoint_dir) or True,
    )

    ckpt = "gs://b/state/controller-state/123"
    controller_main._prepare_local_db_dir(db_dir, "gs://b/state", fresh=False, checkpoint_path=ckpt)

    assert captured["checkpoint_dir"] == ckpt


def test_checkpoint_path_not_found_raises(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    _no_rollout_record(monkeypatch)
    monkeypatch.setattr(controller_main, "latest_checkpoint_epoch_ms", lambda r: 200)
    monkeypatch.setattr(controller_main, "download_checkpoint_to_local", lambda *a, **k: False)

    with pytest.raises(ValueError, match="Checkpoint not found"):
        controller_main._prepare_local_db_dir(
            db_dir, "gs://b/state", fresh=False, checkpoint_path="gs://b/state/controller-state/999"
        )
