# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

MPU = "skyrl_train.workers.megatron.megatron_model_wrapper.mpu"


@pytest.mark.parametrize("rank", [0, 1])
def test_chunked_statistics_cp_gather_preserves_values_and_local_gradients(megatron_wrapper, monkeypatch, rank):
    wrapper = megatron_wrapper(packing=True, controller=None, num_experts=8)
    monkeypatch.setattr(f"{MPU}.get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(f"{MPU}.get_context_parallel_rank", lambda: rank)
    monkeypatch.setattr(f"{MPU}.get_context_parallel_group", lambda: "cp", raising=False)
    indices = ([0, 1, 6, 7], [2, 3, 4, 5])
    all_statistics = torch.stack((torch.arange(8.0), torch.arange(8.0) + 10), -1)
    local = all_statistics[list(indices[rank])].unsqueeze(0).clone().requires_grad_()
    channel = 0

    def gather(destinations, source, *, group):
        nonlocal channel
        if not source.is_contiguous():
            raise ValueError("Collective input must be contiguous")
        torch.testing.assert_close(source, all_statistics[list(indices[rank]), channel].unsqueeze(0))
        for peer, destination in enumerate(destinations):
            destination.copy_(all_statistics[list(indices[peer]), channel].unsqueeze(0))
        channel += 1

    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    logprobs, entropy = wrapper._unpack_statistics(
        local, torch.ones((1, 8), dtype=torch.bool), SimpleNamespace(cu_seqlens_q_padded=torch.tensor([0, 8]))
    )
    torch.testing.assert_close(logprobs, torch.arange(7.0).unsqueeze(0))
    torch.testing.assert_close(entropy, (torch.arange(8.0) + 10).unsqueeze(0))
    (logprobs.sum() + entropy.sum()).backward()
    expected = torch.ones_like(local)
    if rank == 0:
        expected[0, -1, 0] = 0
    torch.testing.assert_close(local.grad, expected)
