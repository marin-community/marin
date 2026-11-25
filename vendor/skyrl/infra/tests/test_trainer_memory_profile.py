"""Sizing cannot consume a training update or change the next stochastic update."""

import copy

import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train import trainer_memory_profile
from skyrl_train.config.utils import get_default_config
from skyrl_train.config.validation import validate_megatron_config
from skyrl_train.models.megatron_router_replay import dense_replay_targets, validate_replay_geometry
from skyrl_train.trainer_memory_profile import preserve_training_state, warm_state_profile


def test_sizing_with_router_replay_uses_balanced_valid_targets():
    cfg = get_default_config()
    OmegaConf.set_struct(cfg, False)
    cfg.generator.weight_sync_backend = "nccl"
    cfg.generator.engine_init_kwargs.enable_return_routed_experts = True
    cfg.trainer.policy.optimizer_config.optimizer = "adam"
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    cfg.trainer.policy.megatron_config.memory_profile = {
        "enabled": True,
        "max_sequence_length": 65_536,
        "valid_token_counts": [17],
        "minimum_headroom_fraction": 0.0,
    }
    validate_megatron_config(cfg)

    routes = trainer_memory_profile.sizing_replay_routes(2, 16, 3, 8, 2)
    validate_replay_geometry(
        num_layers_captured=3, expected_moe_layers=3, topk_captured=2, expected_topk=2,
        num_experts=8, targets=routes, response_len=16, num_actions=16,
    )
    dense, mask = dense_replay_targets(routes, batch_size=2, seq_len=17, num_actions=16)
    assert mask[:, :16].all() and not mask[:, 16].any()
    assert (routes[..., 0] != routes[..., 1]).all()
    assert torch.bincount(routes.flatten().long(), minlength=8).tolist() == [24] * 8
    torch.testing.assert_close(dense[:, :16], routes.long())


@pytest.mark.parametrize("fails", [None, "warmup", "measured"])
def test_sizing_restores_next_optimizer_update_and_rng(fails):
    torch.manual_seed(18)
    learning_rate = 0.01
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    initial_model = copy.deepcopy(model.state_dict())
    rng = torch.random.get_rng_state()

    def step(tokens):
        optimizer.zero_grad()
        model(torch.randn(tokens, 3)).square().mean().backward()
        optimizer.step()
        scheduler.step()

    def measure(tokens, run, warm):
        # Adam's moment slots must already exist before the measured forward/backward.
        if warm:
            assert optimizer.state and all(state["step"].item() >= 1 for state in optimizer.state.values())
        run()
        if fails == ("measured" if warm else "warmup"):
            raise RuntimeError("profile failed")
        return {"valid_tokens": tokens, "warm_optimizer": warm}

    failed = False
    try:
        with preserve_training_state([model, optimizer, scheduler], optimizer_parameters=list(model.parameters()), cuda_devices=[]):
            reports = warm_state_profile([5, 9], step=step, measure=measure)
            assert reports == [
                {"valid_tokens": 5, "warm_optimizer": False},
                {"valid_tokens": 5, "warm_optimizer": True},
                {"valid_tokens": 9, "warm_optimizer": True},
            ]
    except RuntimeError as error:
        failed = True
        assert fails and str(error) == "profile failed"
    assert failed == (fails is not None)
    step(7)
    actual = copy.deepcopy(model.state_dict())

    control = torch.nn.Linear(3, 2)
    control.load_state_dict(initial_model)
    control_optimizer = torch.optim.Adam(control.parameters(), lr=learning_rate)
    torch.random.set_rng_state(rng)
    control(torch.randn(7, 3)).square().mean().backward()
    control_optimizer.step()
    for key, value in control.state_dict().items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
    assert scheduler.last_epoch == 1


def test_profile_restores_master_shards_omitted_from_optimizer_state_dict():
    model = torch.nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
    master = torch.nn.Parameter(model.weight.detach().float() + 1e-5)
    optimizer = torch.optim.Adam([master], lr=0.01)
    original_master = master.detach().clone()
    original_weight = model.weight.detach().clone()
    with preserve_training_state([model, optimizer], optimizer_parameters=[master], cuda_devices=[]):
        master.square().sum().backward()
        optimizer.step()
        with torch.no_grad():
            model.weight.copy_(master)
    torch.testing.assert_close(master, original_master, rtol=0, atol=0)
    torch.testing.assert_close(model.weight, original_weight, rtol=0, atol=0)
    assert not optimizer.state
