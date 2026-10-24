# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model-space FP32 master records assembled without an optimizer-state gather."""

import platform
import socket
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from torch import distributed as dist

from skyrl_train.distributed.megatron.checkpoint_writer import model_key_for_master
from skyrl_train.distributed.megatron.master_records import full_master, master_record_key, model_space_masters


@dataclass
class FakeShardedTensor:
    key: str
    data: torch.Tensor
    dtype: torch.dtype
    replica_id: tuple


@dataclass
class FakeShardedTensorFactory:
    key: str
    data: torch.Tensor
    replica_id: tuple


class FakeDistributedOptimizer:
    """One gradient buffer whose main params this rank holds in the given slices."""

    def __init__(self, params, local_slices, masters):
        self.data_parallel_group = None  # the default group
        self.buffers = [SimpleNamespace(params=params)]
        param_map = {
            param: {"param": SimpleNamespace(start=span.start, end=span.stop)}
            for param, span in zip(params, local_slices, strict=True)
            if span is not None
        }
        self.gbuf_ranges = [{torch.float32: [{"param_map": param_map}]}]
        self._masters = {id(param): master for param, master in zip(params, masters, strict=True)}

    def _get_main_param_and_optimizer_states(self, param):
        return {"param": self._masters[id(param)], "exp_avg": torch.zeros(1)}


@pytest.fixture
def single_process_group(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def test_every_optimized_parameter_gets_a_master_record_under_its_model_key(single_process_group):
    weight = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.bfloat16))
    bias = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    weight_master = torch.arange(6, dtype=torch.float32) + 0.25
    bias_master = torch.arange(4, dtype=torch.float32) - 0.5
    model_state = {
        "decoder.weight": FakeShardedTensor("decoder.layers.0.linear.weight", weight, torch.bfloat16, (0, 0, 0)),
        "decoder.bias": FakeShardedTensorFactory("decoder.layers.0.linear.bias", bias, (0, 0, 0)),
    }
    optimizer = FakeDistributedOptimizer([weight, bias], [range(0, 6), range(0, 4)], [weight_master, bias_master])

    records = model_space_masters(SimpleNamespace(chained_optimizers=[optimizer]), model_state)

    assert set(records) == set(model_state)
    assert {model_key_for_master(record.key) for record in records.values()} == {
        entry.key for entry in model_state.values()
    }
    assert records["decoder.weight"].key == master_record_key("decoder.layers.0.linear.weight")
    assert records["decoder.weight"].dtype == torch.float32 and records["decoder.weight"].replica_id == (0, 0, 0)
    assert torch.equal(records["decoder.weight"].data, weight_master.view(2, 3))
    assert torch.equal(records["decoder.bias"].data, bias_master)


def test_an_optimized_parameter_without_a_model_record_raises(single_process_group):
    orphan = torch.nn.Parameter(torch.zeros(2))
    optimizer = FakeDistributedOptimizer([orphan], [range(0, 2)], [torch.ones(2)])
    with pytest.raises(ValueError, match="no model checkpoint record"):
        model_space_masters(optimizer, {})


def _assemble(rank: int, world_size: int, port: int, out_dir: str) -> None:
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    full = torch.arange(10, dtype=torch.float32) * 1.5
    # Rank 0 holds [0, 6), rank 1 holds [6, 10), rank 2 holds nothing of this parameter.
    span = {0: range(0, 6), 1: range(6, 10), 2: None}[rank]
    shard = full[span.start : span.stop].clone() if span is not None else None
    torch.save(full_master(10, span, shard, None, torch.device("cpu")), f"{out_dir}/rank{rank}.pt")
    dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.timeout(120)
def test_slices_spread_over_the_dp_group_assemble_bit_for_bit(tmp_path, monkeypatch):
    if platform.system() == "Darwin":  # Gloo cannot pick macOS's loopback interface on its own
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0")
    mp.spawn(_assemble, args=(3, _free_port(), str(tmp_path)), nprocs=3, join=True)
    expected = torch.arange(10, dtype=torch.float32) * 1.5
    for rank in range(3):
        assert torch.equal(torch.load(tmp_path / f"rank{rank}.pt"), expected)
