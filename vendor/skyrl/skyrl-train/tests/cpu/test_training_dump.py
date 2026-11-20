"""Diagnostic batches retain their native format at local and cloud export paths."""

import pickle
from types import SimpleNamespace

from fsspec.implementations.memory import MemoryFileSystem
import pytest
import torch

from skyrl_train.io import io
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch


@pytest.mark.parametrize("cloud", [False, True])
def test_training_dump_round_trip_at_export_path(tmp_path, monkeypatch, cloud):
    monkeypatch.chdir(tmp_path)
    filesystem = MemoryFileSystem(skip_instance_cache=True)
    if cloud:
        export_path = "s3://training-dump-test/exports"
        monkeypatch.setattr(io, "filesystem_and_path", lambda path: (filesystem, path))
    else:
        export_path = str(tmp_path / "exports")

    # Dumping a CPU batch requires configuration only, without model initialization.
    trainer = object.__new__(RayPPOTrainer)
    trainer.cfg = SimpleNamespace(trainer=SimpleNamespace(export_path=export_path))
    batch = TrainingInputBatch(
        sequences=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        advantages=torch.tensor([[0.25, -0.5], [0.0, 1.0]]),
    )
    batch.metadata = {"response_length": 2}
    trainer.dump_data(batch, "global_step_1_training_input")

    path = export_path + "/dumped_data/global_step_1_training_input.pkl"
    with io.open_file(path, "rb") as source:
        restored = pickle.load(source)
    assert isinstance(restored, TrainingInputBatch)
    assert restored.metadata == batch.metadata
    for key, expected in batch.items():
        torch.testing.assert_close(restored[key], expected, rtol=0, atol=0)
