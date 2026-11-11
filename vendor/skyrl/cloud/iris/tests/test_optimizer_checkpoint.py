from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from skyrl_train.distributed.megatron.optimizer_checkpoint import (
    restore_hybrid_adam_bucket_steps,
    restore_in_host_order,
)


def checkpoint_state(steps: list[int | None]) -> dict:
    return {
        "optimizer": {"param_groups": [{"step": step} for step in steps]},
        "param_state_sharding_type": "dp_reshardable",
        "param_state": {
            0: {
                (torch.bfloat16, torch.float32): [
                    [
                        {
                            "param": torch.ones(3),
                            "exp_avg": torch.ones(3),
                            "exp_avg_sq": torch.ones(3),
                            "padding": False,
                        }
                    ]
                ]
            }
        },
    }


def test_bucket_restore_recovers_saved_step_without_copying_parameters_or_moments() -> None:
    state = checkpoint_state([5, 5])
    bucket = state["param_state"][0][(torch.bfloat16, torch.float32)][0][0]
    original = {key: value for key, value in bucket.items()}
    restore_hybrid_adam_bucket_steps({0: state})
    assert bucket["step"].item() == 5
    for key, value in original.items():
        assert bucket[key] is value


@pytest.mark.parametrize("steps", [[None], [4, 5]])
def test_bucket_restore_rejects_missing_or_inconsistent_saved_steps(steps) -> None:
    with pytest.raises(ValueError, match="saved optimizer step"):
        restore_hybrid_adam_bucket_steps(checkpoint_state(steps))


def restore_worker(rank: int, root: str) -> None:
    path = Path(root)
    dist.init_process_group("gloo", init_method=f"file://{path / 'rendezvous'}", rank=rank, world_size=2)
    try:

        def restore():
            # Exclusive creation fails if another rank on this host is restoring.
            with (path / "active-restore").open("x"):
                time.sleep(0.05)
                (path / f"restored-{rank}").write_text("complete")
            (path / "active-restore").unlink()

        restore_in_host_order(restore)
    finally:
        dist.destroy_process_group()


def test_offloaded_restore_runs_one_rank_per_host_at_a_time(tmp_path: Path) -> None:
    mp.spawn(restore_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert [p.read_text() for p in sorted(tmp_path.glob("restored-*"))] == ["complete", "complete"]


def failed_restore_worker(rank: int, root: str) -> None:
    path = Path(root)
    dist.init_process_group("gloo", init_method=f"file://{path / 'rendezvous'}", rank=rank, world_size=2)
    try:

        def restore():
            raise ValueError("invalid optimizer state")

        with pytest.raises(RuntimeError, match="rank 0: ValueError: invalid optimizer state"):
            restore_in_host_order(restore)
        (path / f"failure-reported-{rank}").write_text("failed promptly")
    finally:
        dist.destroy_process_group()


def test_restore_failure_reaches_all_ranks_without_waiting_at_the_next_barrier(tmp_path: Path) -> None:
    mp.spawn(failed_restore_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert [p.read_text() for p in sorted(tmp_path.glob("failure-reported-*"))] == [
        "failed promptly",
        "failed promptly",
    ]
