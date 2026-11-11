from pathlib import Path
import time
import weakref

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from skyrl_train.distributed.megatron.optimizer_checkpoint import (
    rebind_master_parameters,
    restore_hybrid_adam_bucket_steps,
    restore_in_host_order,
)


def test_restored_master_follows_the_live_parameter_and_releases_obsolete_storage() -> None:
    master = torch.nn.Parameter(torch.tensor([0.3, -0.8, 1.2]))
    optimizer = torch.optim.AdamW([master], lr=0.01, weight_decay=0)
    old_master = master.detach().clone()
    old_reference = weakref.ref(old_master)
    parameter = object()
    state = {parameter: {"master_param": old_master}}
    del old_master

    rebind_master_parameters(state, {parameter: master})

    assert old_reference() is None
    master.grad = torch.tensor([0.1, -0.2, 0.4])
    optimizer.step()
    torch.testing.assert_close(state[parameter]["master_param"], master, rtol=0, atol=0)
    assert state[parameter]["master_param"].data_ptr() == master.data_ptr()


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
