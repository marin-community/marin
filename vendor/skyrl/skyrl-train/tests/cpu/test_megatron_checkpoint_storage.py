from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
import pickle
from threading import Event
from weakref import ref
import time

from fsspec.implementations.local import LocalFileSystem
import pytest
import torch
import torch.distributed as dist
from torch.distributed import checkpoint
from torch.distributed.checkpoint import DefaultLoadPlanner, DefaultSavePlanner, FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed._shard.sharded_tensor import Shard, init_from_local_shards
import torch.multiprocessing as mp

from skyrl_train.io.checkpoint_reader import RecordCheckpointReader
from skyrl_train.distributed.megatron.checkpoint_metadata import remote_checkpoint_metadata
from skyrl_train.distributed.megatron.checkpoint_writer import StagedMasterWeightsWriter


MODEL_KEY = "decoder.layers.mlp.experts.weight"
CHECKPOINT_TEST_TIMEOUT = 10
COLLECTIVE_TIMEOUT = timedelta(seconds=30)


def save_checkpoint(path: Path, state: dict) -> None:
    writer = StagedMasterWeightsWriter(str(path), filesystem=LocalFileSystem(), max_staged_bytes=2**20)
    writer.save(
        dict(state),
        planner=DefaultSavePlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
    )


def load_checkpoint(path: Path, state: dict) -> None:
    checkpoint.load(
        state,
        storage_reader=RecordCheckpointReader(str(path)),
        planner=DefaultLoadPlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
        no_dist=True,
    )


@pytest.mark.parametrize("fits_in_buffer", [False, True])
def test_streaming_checkpoint_releases_training_after_cpu_snapshot(tmp_path: Path, fits_in_buffer: bool):
    write_started = Event()
    allow_write = Event()

    class PausedWriter(StagedMasterWeightsWriter):
        def _write_items(self, plan, items):
            def pause_after_first_record_is_staged():
                first = next(items)
                write_started.set()
                if not allow_write.wait(timeout=CHECKPOINT_TEST_TIMEOUT):
                    raise TimeoutError("Test did not release the checkpoint writer")
                yield first
                yield from items

            return super()._write_items(plan, pause_after_first_record_is_staged())

    state = {"policy": torch.arange(64), "optimizer": torch.arange(64) * 2}
    original = {key: value.clone() for key, value in state.items()}
    sources = tuple(ref(value) for value in state.values())
    tensor_bytes = state["policy"].nbytes
    writer = PausedWriter(
        str(tmp_path), filesystem=LocalFileSystem(), max_staged_bytes=tensor_bytes * (2 if fits_in_buffer else 1)
    )
    with ThreadPoolExecutor() as executor:
        save = executor.submit(
            writer.save, dict(state), planner=DefaultSavePlanner(flatten_state_dict=False, flatten_sharded_tensors=False)
        )
        try:
            assert write_started.wait(timeout=CHECKPOINT_TEST_TIMEOUT)
            if not fits_in_buffer:
                assert not writer.staged.done()
                allow_write.set()
            writer.staged.result(timeout=CHECKPOINT_TEST_TIMEOUT)
            while state:
                state.popitem()[1].zero_()
            if fits_in_buffer:
                assert not save.done()
                assert all(source() is None for source in sources)
        finally:
            allow_write.set()
        save.result(timeout=CHECKPOINT_TEST_TIMEOUT)

    restored = {key: torch.empty_like(value) for key, value in original.items()}
    load_checkpoint(tmp_path, restored)
    for key, expected in original.items():
        torch.testing.assert_close(restored[key], expected, rtol=0, atol=0)


def test_streaming_checkpoint_write_failure_releases_staging(tmp_path: Path):
    class FailingWriter(StagedMasterWeightsWriter):
        def _write_items(self, plan, items):
            next(items)
            raise OSError("Destination unavailable")

    state = {"policy": torch.ones(64), "optimizer": torch.zeros(64)}
    writer = FailingWriter(str(tmp_path), filesystem=LocalFileSystem(), max_staged_bytes=state["policy"].nbytes)
    with pytest.raises(checkpoint.CheckpointException) as failed:
        writer.save(state, planner=DefaultSavePlanner(flatten_state_dict=False, flatten_sharded_tensors=False))
    assert isinstance(failed.value.failures[0][0], OSError)
    assert not (tmp_path / ".metadata").exists()


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
    save_checkpoint(compact_path, state)
    checkpoint.save(
        state,
        storage_writer=FileSystemWriter(original_path),
        planner=DefaultSavePlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
        no_dist=True,
    )

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
    save_checkpoint(tmp_path, state)
    restored = {key: torch.empty_like(value) for key, value in state.items()}
    load_checkpoint(tmp_path, restored)
    for key, expected in state.items():
        torch.testing.assert_close(restored[key], expected, rtol=0, atol=0)


def save_distributed_checkpoint(rank: int, path: str, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=COLLECTIVE_TIMEOUT
    )
    try:
        checkpoint_group = dist.new_group(backend="gloo", timeout=COLLECTIVE_TIMEOUT)
        allow_write = Event()

        class PausedWriter(StagedMasterWeightsWriter):
            def _write_items(self, plan, items):
                if not allow_write.wait(timeout=CHECKPOINT_TEST_TIMEOUT):
                    raise TimeoutError("Training collective did not finish while the save was pending")
                return super()._write_items(plan, items)

        master = torch.linspace(-0.9, 0.9, 64).reshape(2, 4, 8)
        shard = Shard.from_tensor_and_offsets(master[rank : rank + 1].clone(), [rank, 0, 0], rank)
        state = {"chained_1.optimizer.state.param." + MODEL_KEY: init_from_local_shards([shard], *master.shape)}
        if rank == 0:
            state[MODEL_KEY] = master.bfloat16()
        writer = PausedWriter(path, filesystem=LocalFileSystem(), max_staged_bytes=4096)
        with ThreadPoolExecutor() as executor:
            save = executor.submit(
                writer.save,
                state,
                planner=DefaultSavePlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
                process_group=checkpoint_group,
            )
            try:
                writer.staged.result(timeout=CHECKPOINT_TEST_TIMEOUT)
                progress = torch.tensor(rank)
                dist.all_reduce(progress)
                assert progress.item() == 1
                shard.tensor.zero_()
            finally:
                allow_write.set()
            save.result(timeout=CHECKPOINT_TEST_TIMEOUT)
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


@pytest.mark.parametrize("optimizer_class", [torch.optim.AdamW, torch.optim.SGD])
def test_checkpoint_metadata_restricts_optimizer_classes_to_the_restore_scope(tmp_path, optimizer_class):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    common_file = checkpoint_path / "common.pt"
    torch.save({"optimizer": {"defaults": {"cpu_optimizer_cls": optimizer_class}}, "step": 22}, common_file)
    source_bytes = common_file.read_bytes()
    with pytest.raises(pickle.UnpicklingError):
        torch.load(common_file, weights_only=True)

    with remote_checkpoint_metadata(str(checkpoint_path), optimizer_classes=(torch.optim.AdamW,)) as read_dir:
        staged = Path(read_dir) / common_file.name
        if optimizer_class is torch.optim.AdamW:
            restored = torch.load(staged, weights_only=True)
            assert restored["optimizer"]["defaults"]["cpu_optimizer_cls"] is torch.optim.AdamW
            assert restored["step"] == 22
        else:
            with pytest.raises(pickle.UnpicklingError):
                torch.load(staged, weights_only=True)

    assert common_file.read_bytes() == source_bytes
    with pytest.raises(pickle.UnpicklingError):
        torch.load(common_file, weights_only=True)
