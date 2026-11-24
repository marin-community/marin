from copy import deepcopy
from datetime import timedelta
import json
import pickle
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from unittest.mock import patch
import time

from fsspec.implementations.local import LocalFileSystem
from loguru import logger
import pytest
import torch
import torch.distributed as dist
from torch.distributed import checkpoint
from torch.distributed.checkpoint import DefaultLoadPlanner, DefaultSavePlanner, FileSystemReader
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed._shard.sharded_tensor import Shard, init_from_local_shards
import torch.multiprocessing as mp

from skyrl_train.distributed.megatron.checkpoint_metadata import remote_checkpoint_metadata
from skyrl_train.io.checkpoint_reader import RecordCheckpointReader
from skyrl_train.distributed.megatron.checkpoint_writer import MasterWeightsWriter
from skyrl_train.io.torch_distributed_checkpoint import StreamingFsspecWriter


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


def run_checkpoint_processes(worker, path: Path, rendezvous: Path) -> None:
    processes = mp.spawn(worker, args=(str(path), str(rendezvous)), nprocs=2, join=False)
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


def test_checkpoint_shares_masters_across_ranks_and_reshards_model_only(tmp_path: Path):
    path = tmp_path / "checkpoint"
    run_checkpoint_processes(save_distributed_checkpoint, path, tmp_path / "rendezvous")

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


@pytest.fixture
def cached_megatron_save(monkeypatch):
    # MCore is an optional Linux trainer dependency. The dedicated checkpoint-cache
    # CPU verification installs the exact pin and executes these cases without GPUs.
    try:
        version("megatron-core")
    except PackageNotFoundError:
        pytest.skip("Real MCore is required; import-only CPU stubs do not exercise checkpoint planning")
    pytest.importorskip("megatron.core", exc_type=ImportError)
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    from skyrl_train.distributed.megatron import direct_checkpoint

    monkeypatch.setattr(direct_checkpoint, "create_s3_filesystem", lambda: LocalFileSystem())
    return direct_checkpoint, dist_checkpointing, ShardedTensor


@pytest.mark.parametrize("cache_plans", [True, False])
@pytest.mark.parametrize("change", ["shape", "dtype", "key"])
def test_checkpoint_saves_restore_across_structure_changes(tmp_path, cached_megatron_save, change, cache_plans):
    direct, serialization, sharded_tensor = cached_megatron_save

    events = []
    sink = logger.add(
        lambda message: events.append(message.record["extra"]["checkpoint_save"])
        if "checkpoint_save" in message.record["extra"]
        else None
    )
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1)
    expected = []
    try:
        strategy = direct.DirectS3TorchDistSaveShardedStrategy(str(tmp_path / "step-1"), cached_metadata=True)
        parallel = direct.CachedFullyParallelSaveStrategy(strategy, dist.group.WORLD)
        for step in range(1, 9):
            size = 12 if change == "shape" and step > 4 else 8
            master = torch.linspace(-0.9, 0.9, size) + step
            master[0], master[1] = -0.0, 0.0
            name = "changed.weight" if change == "key" and step > 4 else MODEL_KEY
            master_key = "optimizer.state.param." + name
            if change == "dtype" and step > 4:
                # The old cached alias must disappear: these are independent BF16 records.
                master = master.bfloat16()
                model = (master + 2).bfloat16()
            else:
                model = master.bfloat16()
            state = {
                "model": sharded_tensor.from_rank_offsets(name, model),
                "master": sharded_tensor.from_rank_offsets(master_key, master),
                "momentum": sharded_tensor.from_rank_offsets(
                    "optimizer.state.exp_avg." + name, torch.sin(master.float())
                ),
                "step": step,
                "rng": torch.arange(7, dtype=torch.uint8) + step,
            }
            path = tmp_path / f"step-{step}"
            path.mkdir()
            if cache_plans:
                strategy.checkpoint_dir = str(path)
            else:
                strategy = direct.DirectS3TorchDistSaveShardedStrategy(str(path), cached_metadata=False)
                parallel = direct.FullyParallelSaveStrategyWrapper(
                    strategy, dist.group.WORLD, do_cache_distribution=False
                )
            serialization.save(state, path, sharded_strategy=parallel)
            expected.append(
                (path, name, master_key, model.clone(), master.clone(), state["momentum"].data.clone(), step)
            )
        # Restore every destination after all saves: cached storage paths must never
        # redirect an older checkpoint to the newest payload or metadata.
        for path, name, master_key, model, master, momentum, step in expected:
            targets = {
                "model": sharded_tensor.from_rank_offsets(name, torch.empty_like(model)),
                "master": sharded_tensor.from_rank_offsets(master_key, torch.empty_like(master)),
                "momentum": sharded_tensor.from_rank_offsets(
                    "optimizer.state.exp_avg." + name, torch.empty_like(momentum)
                ),
            }
            restored = serialization.load(targets, path)
            for key, value in (("model", model), ("master", master), ("momentum", momentum)):
                torch.testing.assert_close(restored[key], value, rtol=0, atol=0)
                assert torch.equal(restored[key].contiguous().view(torch.uint8), value.contiguous().view(torch.uint8))
            assert restored["step"] == step
            torch.testing.assert_close(restored["rng"], torch.arange(7, dtype=torch.uint8) + step, rtol=0, atol=0)
            only_model = {"model": sharded_tensor.from_rank_offsets(name, torch.empty_like(model))}
            restored_model = serialization.load(only_model, path)["model"]
            torch.testing.assert_close(restored_model, model, rtol=0, atol=0)
            assert torch.equal(restored_model.contiguous().view(torch.uint8), model.contiguous().view(torch.uint8))
        if cache_plans:
            assert [event["cache_reused"] for event in events] == [False, False, True, True] * 2
            assert [event["cache_invalidated"] for event in events] == [True, False, False, False] * 2
        else:
            assert all(not event["cache_enabled"] and not event["cache_reused"] for event in events)
    finally:
        logger.remove(sink)
        dist.destroy_process_group()


def test_cached_writer_reapplies_aliases_to_four_fresh_destinations(tmp_path):
    aliases = {}
    saved = []
    cached_plan = metadata = None
    for step in range(1, 5):
        path = tmp_path / f"save-{step}"
        master = torch.arange(32, dtype=torch.float32) * 0.1 + step
        state = {
            MODEL_KEY: master.bfloat16(),
            "optimizer.state.param." + MODEL_KEY: master,
            "optimizer.state.exp_avg." + MODEL_KEY: torch.sin(master),
        }
        writer = MasterWeightsWriter(str(path), filesystem=LocalFileSystem(), model_aliases=aliases)
        planner = DefaultSavePlanner(flatten_state_dict=False, flatten_sharded_tensors=False)
        if cached_plan is None:
            checkpoint.save(state, storage_writer=writer, planner=planner, no_dist=True)
            cached_plan = planner.plan
            metadata = FileSystemReader(path).read_metadata()
        else:
            planner.set_up_planner(state, is_coordinator=True)
            writer.set_up_storage_writer(True)
            writer.prepare_local_plan(cached_plan)
            writer.prepare_write_data(cached_plan, planner)
            writer.finish(deepcopy(metadata), [writer.write_planned_data()])
        aliases = writer.model_aliases
        saved.append((path, state))
    for path, state in saved:
        restored = {key: torch.empty_like(value) for key, value in state.items()}
        load_checkpoint(path, restored)
        for key in state:
            torch.testing.assert_close(restored[key], state[key], rtol=0, atol=0)
        only_model = {MODEL_KEY: torch.empty_like(state[MODEL_KEY])}
        load_checkpoint(path, only_model)
        torch.testing.assert_close(only_model[MODEL_KEY], state[MODEL_KEY], rtol=0, atol=0)


def save_cached_distributed_checkpoints(rank: int, path: str, rendezvous: str) -> None:
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    from skyrl_train.distributed.megatron.direct_checkpoint import (
        CachedFullyParallelSaveStrategy,
        DirectS3TorchDistSaveShardedStrategy,
    )

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    events = []
    sink = logger.add(
        lambda message: events.append(message.record["extra"]["checkpoint_save"])
        if "checkpoint_save" in message.record["extra"]
        else None
    )
    try:
        strategy = DirectS3TorchDistSaveShardedStrategy(path, cached_metadata=True)
        parallel = CachedFullyParallelSaveStrategy(strategy, dist.group.WORLD)
        with patch(
            "skyrl_train.distributed.megatron.direct_checkpoint.create_s3_filesystem", lambda: LocalFileSystem()
        ):
            for step in range(1, 9):
                master = torch.arange(8, dtype=torch.float32) + rank * 8 + step
                state = {
                    "model": ShardedTensor.from_rank_offsets(MODEL_KEY, master.bfloat16(), (0, rank, 2)),
                    "master": ShardedTensor.from_rank_offsets(
                        "optimizer.state.param." + MODEL_KEY, master, (0, rank, 2)
                    ),
                    "step": step,
                }
                # Only rank1 changes: rank0 must invalidate its global metadata too.
                if rank == 1 and step > 4:
                    state["extra"] = ShardedTensor.from_rank_offsets("rank1.extra", torch.ones(3) * step)
                destination = Path(path) / f"step-{step}"
                if rank == 0:
                    destination.mkdir(parents=True)
                dist.barrier()
                strategy.checkpoint_dir = str(destination)
                dist_checkpointing.save(state, destination, sharded_strategy=parallel)
        (Path(path) / f"rank-{rank}-events.json").write_text(json.dumps(events))
    finally:
        logger.remove(sink)
        dist.destroy_process_group()


def test_cached_checkpoint_change_on_one_rank_invalidates_all_ranks(tmp_path, cached_megatron_save):
    run_checkpoint_processes(save_cached_distributed_checkpoints, tmp_path / "distributed", tmp_path / "rdzv")
    for step in range(1, 9):
        master = torch.arange(16, dtype=torch.float32) + step
        state = {
            MODEL_KEY: torch.empty(16, dtype=torch.bfloat16),
            "optimizer.state.param." + MODEL_KEY: torch.empty(16),
        }
        if step > 4:
            state["rank1.extra"] = torch.empty(3)
        load_checkpoint(tmp_path / "distributed" / f"step-{step}", state)
        torch.testing.assert_close(state[MODEL_KEY], master.bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(state["optimizer.state.param." + MODEL_KEY], master, rtol=0, atol=0)
        if step > 4:
            torch.testing.assert_close(state["rank1.extra"], torch.ones(3) * step, rtol=0, atol=0)
    for rank in range(2):
        events = json.loads((tmp_path / "distributed" / f"rank-{rank}-events.json").read_text())
        assert [event["cache_reused"] for event in events] == [False, False, True, True] * 2
        assert [event["cache_invalidated"] for event in events] == [True, False, False, False] * 2


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
        if optimizer_class is torch.optim.AdamW:
            restored = torch.load(Path(read_dir) / "common.pt", weights_only=True)
            assert restored["optimizer"]["defaults"]["cpu_optimizer_cls"] is torch.optim.AdamW
            assert restored["step"] == 22
        else:
            with pytest.raises(pickle.UnpicklingError):
                torch.load(Path(read_dir) / "common.pt", weights_only=True)

    assert common_file.read_bytes() == source_bytes
    with pytest.raises(pickle.UnpicklingError):
        torch.load(common_file, weights_only=True)
