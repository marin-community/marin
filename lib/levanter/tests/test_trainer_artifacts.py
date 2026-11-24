# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Trainer.write_artifact must keep artifacts on disk even when the tracker refuses them.

An optional debug artifact (e.g. the train-step JAXPR) must never raise into
training: an oversized artifact is written to the run's log dir and skipped at
the tracker with one warning, while normal artifacts upload as before.
"""

import logging

import optax

from levanter.tracker import NoopConfig
from levanter.tracker.wandb import MAX_WANDB_ARTIFACT_BYTES, WandbTracker
from levanter.trainer import Trainer, TrainerConfig


def test_write_artifact_keeps_oversized_file_and_skips_tracker_upload(tmp_path, caplog):
    class FakeRun:
        def __init__(self):
            self.logged = []

        def log_artifact(self, artifact_path, *, name=None, type=None):
            self.logged.append((artifact_path, name, type))

    run = FakeRun()
    config = TrainerConfig(
        id="artifact-size-guard",
        tracker=NoopConfig(),
        require_accelerator=False,
        log_dir=tmp_path,
    )
    trainer = Trainer(config, optax.sgd(0.0), lambda model, batch: model, add_default_hooks=False)
    trainer.tracker = WandbTracker(run)

    with caplog.at_level(logging.WARNING):
        trainer.write_artifact("train_step.hlo.txt", "x" * (MAX_WANDB_ARTIFACT_BYTES + 1), type="hlo")
        trainer.write_artifact("small.hlo.txt", "ok", type="hlo")

    artifact_dir = tmp_path / "artifact-size-guard" / "artifacts"
    # The oversized file is still written to the run's log dir...
    assert (artifact_dir / "train_step.hlo.txt").stat().st_size == MAX_WANDB_ARTIFACT_BYTES + 1
    # ...and the normal artifact uploads as before.
    assert run.logged == [(artifact_dir / "small.hlo.txt", "small.hlo.txt", "hlo")]
    # The skip is exactly one warning; nothing raised.
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
