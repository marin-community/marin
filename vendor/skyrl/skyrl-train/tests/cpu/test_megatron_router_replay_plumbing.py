"""Replay targets keep their values without retaining nonlocal storage."""

import pytest
import torch
from skyrl_train.models.megatron_router_replay import SENTINEL_EXPERT_ID, MegatronRouterReplay

MPU = "skyrl_train.workers.megatron.megatron_model_wrapper.mpu"


def _wrapper(megatron_wrapper, *, packing):
    controller = MegatronRouterReplay(range(3), recompute_enabled=True)
    controller.num_moe_layers_total = 3
    controller.topk = 2
    return megatron_wrapper(packing=packing, controller=controller, num_experts=8)


@pytest.mark.parametrize("packing", [False, True], ids=["unpacked", "packed"])
def test_local_replay_targets_do_not_retain_other_layers(megatron_wrapper, packing):
    wrapper = _wrapper(megatron_wrapper, packing=packing)
    sequences = torch.ones((2, 8), dtype=torch.long)
    routes = (torch.arange(48) % 7).reshape(2, 4, 3, 2)

    targets = wrapper._build_router_replay_targets(sequences, sequences.bool(), routes, 4, (1,))

    expected = torch.full((2, 8, 2), SENTINEL_EXPERT_ID, dtype=routes.dtype)
    expected[:, 3:7] = routes[:, :, 1]
    expected = expected.reshape(-1, 2) if packing else expected.transpose(0, 1).reshape(-1, 2)
    actual = targets.per_layer[1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.untyped_storage().nbytes() == actual.numel() * actual.element_size()


@pytest.mark.parametrize("layer_indices", [(1,), (0, 1, 2)], ids=["local_layer", "all_layers"])
def test_single_router_row_does_not_retain_other_tp_rows(megatron_wrapper, monkeypatch, layer_indices):
    wrapper = _wrapper(megatron_wrapper, packing=True)
    monkeypatch.setattr(f"{MPU}.get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(f"{MPU}.get_tensor_model_parallel_rank", lambda: 0, raising=False)
    sequences = torch.ones((1, 2), dtype=torch.long)
    routes = torch.arange(6).reshape(1, 1, 3, 2)
    targets = wrapper._build_router_replay_targets(sequences, sequences.bool(), routes, 1, layer_indices)

    storages = {}
    for layer, actual in targets.per_layer.items():
        torch.testing.assert_close(actual, routes[0, :, layer], rtol=0, atol=0)
        storages[actual.untyped_storage().data_ptr()] = actual.untyped_storage().nbytes()
    assert sum(storages.values()) == len(layer_indices) * 2 * routes.element_size()
