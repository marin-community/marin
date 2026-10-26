import fsspec
import torch
from skyrl_train.io import io
from skyrl_train.training_batch import TrainingInputBatch, dump_training_batch


def test_training_batch_dump_round_trips_an_object_store_uri(monkeypatch):
    filesystem = fsspec.filesystem("memory")
    monkeypatch.setattr(io, "_get_filesystem", lambda path: filesystem)
    batch = TrainingInputBatch({"advantages": torch.tensor([[1.0, -0.0], [2.0, 3.0]])})
    batch.metadata = {"uids": ["a", "b"]}
    path = "s3://worker-batch-test/verify/dumped_data/global_step_7_training_input.pkl"
    dump_training_batch(batch, "s3://worker-batch-test/verify", "global_step_7_training_input")
    restored = TrainingInputBatch().load(path)
    assert restored["advantages"].numpy().tobytes() == batch["advantages"].numpy().tobytes()
    assert restored.metadata == {"uids": ["a", "b"]}
