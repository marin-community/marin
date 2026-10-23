"""FP32 master export verifies every weight against the checkpoint metadata."""

from pathlib import Path

from fsspec.implementations.local import LocalFileSystem
import pytest
import torch
from torch.distributed import checkpoint
from torch.distributed.checkpoint import DefaultSavePlanner

from skyrl_train.distributed.megatron.checkpoint_writer import MasterWeightsWriter
from skyrl_train.distributed.megatron.master_export import load_fp32_masters

EXPERT_KEY = "decoder.layers.mlp.experts.weight"
ROUTER_KEY = "decoder.layers.mlp.router.weight"


def save_checkpoint(path: Path, state: dict) -> None:
    checkpoint.save(
        state,
        storage_writer=MasterWeightsWriter(str(path), filesystem=LocalFileSystem()),
        planner=DefaultSavePlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
        no_dist=True,
    )


def chained_optimizer_checkpoint() -> tuple[dict, torch.Tensor, torch.Tensor]:
    """One dense optimizer's fp32_param master plus a chained optimizer's param master.

    The BF16 model records share the masters' bytes through the writer; the moment
    records and the frozen router bias have no master at all.
    """
    expert_master = torch.linspace(-0.9, 0.9, 4096).reshape(64, 64)
    router_master = torch.linspace(0.4, -0.4, 64)
    state = {
        EXPERT_KEY: expert_master.bfloat16(),
        "optimizer.state.fp32_param." + EXPERT_KEY: expert_master,
        "optimizer.state.exp_avg." + EXPERT_KEY: torch.sin(expert_master),
        "optimizer.state.exp_avg_sq." + EXPERT_KEY: torch.cos(expert_master).square(),
        ROUTER_KEY: router_master.bfloat16(),
        "chained_1.optimizer.state.param." + ROUTER_KEY: router_master,
        "decoder.layers.mlp.router.bias": torch.linspace(-0.3, 0.3, 64),
    }
    return state, expert_master, router_master


def test_load_returns_only_the_fp32_masters_keyed_by_model_name(tmp_path: Path):
    state, expert_master, router_master = chained_optimizer_checkpoint()
    save_checkpoint(tmp_path, state)

    masters = load_fp32_masters(str(tmp_path))

    assert set(masters) == {EXPERT_KEY, ROUTER_KEY}
    assert masters[EXPERT_KEY].dtype == torch.float32
    torch.testing.assert_close(masters[EXPERT_KEY], expert_master, rtol=0, atol=0)
    torch.testing.assert_close(masters[ROUTER_KEY], router_master, rtol=0, atol=0)


def test_export_refuses_a_checkpoint_without_master_records(tmp_path: Path):
    save_checkpoint(
        tmp_path,
        {
            EXPERT_KEY: torch.ones(8, dtype=torch.bfloat16),
            "optimizer.state.exp_avg." + EXPERT_KEY: torch.ones(8, dtype=torch.bfloat16),
        },
    )

    with pytest.raises(ValueError, match="no FP32 master records"):
        load_fp32_masters(str(tmp_path))


def test_export_refuses_a_master_record_that_is_not_fp32(tmp_path: Path):
    """A BF16 master record means the export would upcast non-master weights."""
    save_checkpoint(
        tmp_path,
        {
            "optimizer.state.param." + EXPERT_KEY: torch.ones(8, dtype=torch.bfloat16),
        },
    )

    with pytest.raises(ValueError, match="stores torch.bfloat16, not float32"):
        load_fp32_masters(str(tmp_path))
