"""Sizing cannot consume a training update or change the next stochastic update."""

import copy

import pytest
import torch

from skyrl_train.trainer_memory_profile import preserve_training_state, warm_state_profile


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
