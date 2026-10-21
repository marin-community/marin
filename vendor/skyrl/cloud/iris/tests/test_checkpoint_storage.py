from datetime import timedelta
from pathlib import Path
import time

from fsspec.implementations.local import LocalFileSystem
import pytest
import torch
import torch.distributed as dist
from torch.distributed import checkpoint
from torch.distributed.checkpoint import DefaultLoadPlanner, DefaultSavePlanner, FileSystemReader
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed._shard.sharded_tensor import Shard, init_from_local_shards
import torch.multiprocessing as mp

from skyrl_train.io.checkpoint_reader import RecordCheckpointReader
from skyrl_train.io.torch_distributed_checkpoint import MasterWeightsWriter, StreamingFsspecWriter


MODEL_KEY = "decoder.layers.mlp.experts.weight"


def save_checkpoint(path: Path, state: dict, writer_class: type[StreamingFsspecWriter]) -> None:
    checkpoint.save(
        state,
        storage_writer=writer_class(str(path), filesystem=LocalFileSystem()),
        planner=DefaultSavePlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
        no_dist=True,
    )


def load_checkpoint(path: Path, state: dict) -> None:
    checkpoint.load(
        state,
        storage_reader=RecordCheckpointReader(str(path)),
        planner=DefaultLoadPlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
        no_dist=True,
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "optimizer.state.param.",
        "chained_1.optimizer.state.param.",
        "optimizer.state.fp32_param.",
        "chained_1.optimizer.state.fp32_param.",
    ],
)
def test_checkpoint_shares_master_records_and_restores_training_and_model_only(tmp_path: Path, prefix: str):
    master = torch.linspace(-0.9, 0.9, 4096).reshape(64, 64)
    master_key = prefix + MODEL_KEY
    state = {
        MODEL_KEY: master.bfloat16(),
        master_key: master,
        "optimizer.state.exp_avg." + MODEL_KEY: torch.sin(master),
        "optimizer.state.exp_avg_sq." + MODEL_KEY: torch.cos(master).square(),
        "decoder.layers.mlp.router.bias": torch.linspace(-0.3, 0.3, 64),
        "frozen.weight": torch.ones(32, dtype=torch.bfloat16),
        "rng": torch.arange(32, dtype=torch.uint8),
    }
    compact_path = tmp_path / "compact"
    original_path = tmp_path / "original"
    save_checkpoint(compact_path, state, MasterWeightsWriter)
    save_checkpoint(original_path, state, StreamingFsspecWriter)

    metadata = FileSystemReader(compact_path).read_metadata()
    assert (
        metadata.storage_data[MetadataIndex(MODEL_KEY, (0, 0))]
        == metadata.storage_data[MetadataIndex(master_key, (0, 0))]
    )
    assert metadata.state_dict_metadata[MODEL_KEY].properties.dtype == torch.bfloat16
    compact_bytes = sum(path.stat().st_size for path in compact_path.glob("*.distcp"))
    original_bytes = sum(path.stat().st_size for path in original_path.glob("*.distcp"))
    assert original_bytes - compact_bytes >= state[MODEL_KEY].numel() * state[MODEL_KEY].element_size()

    restored = {key: torch.empty_like(value) for key, value in state.items()}
    load_checkpoint(compact_path, restored)
    for key, expected in state.items():
        torch.testing.assert_close(restored[key], expected, rtol=0, atol=0)

    model_only = {
        key: torch.empty_like(state[key]) for key in (MODEL_KEY, "decoder.layers.mlp.router.bias", "frozen.weight")
    }
    load_checkpoint(compact_path, model_only)
    for key, expected in model_only.items():
        torch.testing.assert_close(expected, state[key], rtol=0, atol=0)


def test_checkpoint_preserves_weights_without_a_matching_fp32_master(tmp_path: Path):
    state = {
        "unoptimized.weight": torch.ones(8, dtype=torch.bfloat16),
        "fp32.weight": torch.arange(8, dtype=torch.float32),
        "optimizer.state.param.fp32.weight": torch.arange(8, dtype=torch.float32) + 1,
        "other_shape.weight": torch.ones((2, 4), dtype=torch.bfloat16),
        "optimizer.state.param.other_shape.weight": torch.zeros(8),
        "bf16_master.weight": torch.ones(8, dtype=torch.bfloat16),
        "optimizer.state.param.bf16_master.weight": torch.zeros(8, dtype=torch.bfloat16),
    }
    save_checkpoint(tmp_path, state, MasterWeightsWriter)
    restored = {key: torch.empty_like(value) for key, value in state.items()}
    load_checkpoint(tmp_path, restored)
    for key, expected in state.items():
        torch.testing.assert_close(restored[key], expected, rtol=0, atol=0)


def save_distributed_checkpoint(rank: int, path: str, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        master = torch.linspace(-0.9, 0.9, 64).reshape(2, 4, 8)
        shard = Shard.from_tensor_and_offsets(master[rank : rank + 1].clone(), [rank, 0, 0], rank)
        state = {"chained_1.optimizer.state.param." + MODEL_KEY: init_from_local_shards([shard], *master.shape)}
        if rank == 0:
            state[MODEL_KEY] = master.bfloat16()
        checkpoint.save(
            state,
            storage_writer=MasterWeightsWriter(path, filesystem=LocalFileSystem()),
            planner=DefaultSavePlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
        )
    finally:
        dist.destroy_process_group()


def test_checkpoint_shares_masters_across_ranks_and_reshards_model_only(tmp_path: Path):
    path = tmp_path / "checkpoint"
    processes = mp.spawn(
        save_distributed_checkpoint, args=(str(path), str(tmp_path / "rendezvous")), nprocs=2, join=False
    )
    deadline = time.monotonic() + 45
    try:
        while not processes.join(timeout=max(0, deadline - time.monotonic())):
            if time.monotonic() >= deadline:
                raise TimeoutError("distributed checkpoint save exceeded 45 seconds")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
        for process in processes.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()

    metadata = FileSystemReader(path).read_metadata()
    master_key = "chained_1.optimizer.state.param." + MODEL_KEY
    assert len(metadata.state_dict_metadata[MODEL_KEY].chunks) == 2
    for rank in range(2):
        assert (
            metadata.storage_data[MetadataIndex(MODEL_KEY, (rank, 0, 0))]
            == metadata.storage_data[MetadataIndex(master_key, (rank, 0, 0))]
        )
    model_only = {MODEL_KEY: torch.empty((2, 4, 8), dtype=torch.bfloat16)}
    checkpoint.load(
        model_only,
        storage_reader=FileSystemReader(path),
        planner=DefaultLoadPlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
        no_dist=True,
    )
    master = torch.linspace(-0.9, 0.9, 64).reshape(2, 4, 8)
    torch.testing.assert_close(model_only[MODEL_KEY], master.bfloat16(), rtol=0, atol=0)
    restored = {MODEL_KEY: torch.empty((2, 4, 8), dtype=torch.bfloat16), master_key: torch.empty((2, 4, 8))}
    load_checkpoint(path, restored)
    torch.testing.assert_close(restored[MODEL_KEY], master.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(restored[master_key], master, rtol=0, atol=0)
