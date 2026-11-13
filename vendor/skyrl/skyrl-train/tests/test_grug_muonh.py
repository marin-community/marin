"""Hero EP numerical and expert-parallel regressions, using CPU tensors."""

import io
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from skyrl_train.distributed.megatron.grug_muonh import (
    MegatronGrugMuonH,
    megatron_grug_expert_bank,
    megatron_grug_route,
)

FIXTURE = Path(__file__).with_name("fixtures") / "grug_muonh_hero_ep" / "golden.npz"


def tensor(value):
    return torch.from_numpy(np.asarray(value).copy())


def optimizers(parameters, routes, *, offload_momentum=True, expert_parallel_group=None):
    groups = [
        {"params": [parameter], "optimizer": megatron_grug_route(name, parameter)}
        for name, parameter in parameters.items()
        if routes[name] != "adam"
    ]
    muon = MegatronGrugMuonH(
        groups,
        lr=0.03,
        betas=(0.9, 0.95),
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        eps=1e-8,
        muon_eps=1e-8,
        qkv_split_shapes=(4, 2, 2),
        offload_momentum=offload_momentum,
        parameter_banks={
            parameter: bank
            for name, parameter in parameters.items()
            if (bank := megatron_grug_expert_bank(name)) is not None
        },
        expert_parallel_group=expert_parallel_group,
    )
    adam = torch.optim.Adam(
        [parameter for name, parameter in parameters.items() if routes[name] == "adam"],
        lr=0.004,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return muon, adam


@pytest.mark.parametrize("offload_momentum", [False, True])
def test_hero_ep_routes_updates_moments_and_restart_match_pinned_jax(offload_momentum):
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        names = fixture["metadata_names"].tolist()
        routes = dict(zip(names, fixture["metadata_routes"].tolist(), strict=True))
        parameters = {name: torch.nn.Parameter(tensor(fixture[f"initial__{name}"])) for name in names}
        for name, parameter in parameters.items():
            actual_route = megatron_grug_route(name, parameter)
            assert (
                "adam" if actual_route == "adam" else "adamh" if actual_route == "grug_adamh" else "muonh"
            ) == routes[name]
        muon, adam = optimizers(parameters, routes, offload_momentum=offload_momentum)
        for number in range(1, 4):
            previous = {name: parameter.detach().clone() for name, parameter in parameters.items()}
            for name, parameter in parameters.items():
                parameter.grad = tensor(fixture[f"gradient_{number}__{name}"])
            muon.step()
            adam.step()
            for name, parameter in parameters.items():
                route = routes[name]
                expected = tensor(fixture[f"parameter_{number}__{name}"])
                expected_previous = tensor(
                    fixture[f"{'initial' if number == 1 else f'parameter_{number - 1}'}__{name}"]
                )
                actual_delta = parameter.detach() - previous[name]
                expected_delta = expected - expected_previous
                relative_error = torch.linalg.vector_norm(actual_delta - expected_delta) / torch.linalg.vector_norm(
                    expected_delta
                )
                # Separate updates from weights so a small learning rate cannot
                # hide a wrong route, norm sphere or matrix split.
                assert relative_error < 1e-3, (name, number, relative_error.item())
                if route == "muonh":
                    # Preserve the existing Grug BF16 golden-step tolerances.
                    torch.testing.assert_close(parameter, expected, rtol=3e-3, atol=1.5e-3)
                    state = muon.state[parameter]
                    torch.testing.assert_close(
                        state["momentum_buffer"], tensor(fixture[f"momentum_{number}__{name}"]), rtol=2e-6, atol=5e-7
                    )
                else:
                    torch.testing.assert_close(parameter, expected, rtol=2e-6, atol=5e-7)
                    state = (adam if route == "adam" else muon).state[parameter]
                    torch.testing.assert_close(
                        state["exp_avg"], tensor(fixture[f"{route}_mu_{number}__{name}"]), rtol=2e-6, atol=5e-7
                    )
                    torch.testing.assert_close(
                        state["exp_avg_sq"], tensor(fixture[f"{route}_nu_{number}__{name}"]), rtol=2e-6, atol=5e-7
                    )
                    assert int(state["step"].item()) == number == int(fixture[f"{route}_step_{number}"])
            if number == 1:
                saved = io.BytesIO()
                torch.save(
                    {
                        "parameters": {name: p.detach() for name, p in parameters.items()},
                        "muon": muon.state_dict(),
                        "adam": adam.state_dict(),
                    },
                    saved,
                )
                saved.seek(0)
                restored = torch.load(saved, weights_only=True)
                parameters = {name: torch.nn.Parameter(value) for name, value in restored["parameters"].items()}
                muon, adam = optimizers(parameters, routes, offload_momentum=offload_momentum)
                muon.initialize_state()
                muon.load_state_dict(restored["muon"])
                adam.load_state_dict(restored["adam"])


def expert_bank_worker(rank, root):
    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    try:
        with np.load(FIXTURE, allow_pickle=False) as fixture:
            # The two owners have unequal expert counts and unequal expert norms.
            names = [
                f"decoder.layers.0.mlp.experts.linear_fc2.weight{index}" for index in ((0,) if rank == 0 else (1, 2))
            ]
            parameters = {name: torch.nn.Parameter(tensor(fixture[f"initial__{name}"])) for name in names}
            optimizer = MegatronGrugMuonH(
                [{"params": list(parameters.values()), "optimizer": "grug_muonh"}],
                lr=0.03,
                betas=(0.9, 0.95),
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                eps=1e-8,
                muon_eps=1e-8,
                qkv_split_shapes=(4, 2, 2),
                offload_momentum=True,
                parameter_banks={parameter: "layer0-down" for parameter in parameters.values()},
                expert_parallel_group=dist.group.WORLD,
            )
            for number in range(1, 4):
                for name, parameter in parameters.items():
                    parameter.grad = tensor(fixture[f"gradient_{number}__{name}"])
                optimizer.step()
            torch.save({name: p.detach() for name, p in parameters.items()}, Path(root) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_hero_expert_parallel_bank_matches_jax_with_unequal_local_banks(tmp_path):
    mp.spawn(expert_bank_worker, args=(str(tmp_path),), nprocs=2, join=True)
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        for path in sorted(tmp_path.glob("rank-*.pt")):
            for name, value in torch.load(path, weights_only=True).items():
                torch.testing.assert_close(value, tensor(fixture[f"parameter_3__{name}"]), rtol=3e-3, atol=1.5e-3)
