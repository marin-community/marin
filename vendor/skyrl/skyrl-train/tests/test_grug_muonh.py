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


def optimizers(parameters, routes, *, offload_momentum=True, expert_parallel_group=None, normalization="all_experts"):
    groups = [
        {"params": [parameter], "optimizer": megatron_grug_route(name, parameter)}
        for name, parameter in parameters.items()
        if routes[name] != "adam"
    ]
    banks = {}
    if normalization == "all_experts":
        for name, parameter in parameters.items():
            # The multiprocess case exercises the alternate grouped-GEMM names.
            if expert_parallel_group is not None:
                name = name.replace(".mlp.experts.", ".mlp.experts.experts.")
            if (bank := megatron_grug_expert_bank(name)) is not None:
                banks[parameter] = bank
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
        parameter_banks=banks,
        expert_parallel_group=expert_parallel_group,
    )
    adam = torch.optim.Adam(
        [parameter for name, parameter in parameters.items() if routes[name] == "adam"],
        lr=0.004,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return muon, adam


def oracle_update_error(offload_momentum, normalization, *, expert_parallel_group=None, expert_rank=None):
    fixture_path = FIXTURE if normalization == "all_experts" else FIXTURE.with_name("golden_per_expert.npz")
    with np.load(fixture_path, allow_pickle=False) as fixture:
        names = fixture["metadata_names"].tolist()
        routes = dict(zip(names, fixture["metadata_routes"].tolist(), strict=True))
        if expert_rank is not None:
            owned = (0,) if expert_rank == 0 else (1, 2)
            names = [name for name in names if ".mlp.experts." not in name or int(name.rsplit("weight", 1)[1]) in owned]
        parameters = {name: torch.nn.Parameter(torch.tensor(fixture[f"initial__{name}"])) for name in names}
        for name, parameter in parameters.items():
            actual_route = megatron_grug_route(name, parameter)
            assert (
                "adam" if actual_route == "adam" else "adamh" if actual_route == "grug_adamh" else "muonh"
            ) == routes[name]
        muon, adam = optimizers(
            parameters,
            routes,
            offload_momentum=offload_momentum,
            normalization=normalization,
            expert_parallel_group=expert_parallel_group,
        )
        # Megatron replaces the original parameters with FP32 masters after
        # constructing the optimizer. Bank identity must survive that replacement.
        replacements = {p: torch.nn.Parameter(p.detach().clone()) for p in parameters.values()}
        for optimizer in (muon, adam):
            for group in optimizer.param_groups:
                group["params"] = [replacements[p] for p in group["params"]]
        parameters = {name: replacements[p] for name, p in parameters.items()}
        errors = []
        for number in range(1, 4):
            previous = {name: parameter.detach().clone() for name, parameter in parameters.items()}
            for name, parameter in parameters.items():
                parameter.grad = torch.tensor(fixture[f"gradient_{number}__{name}"])
            muon.step()
            adam.step()
            for name, parameter in parameters.items():
                route = routes[name]
                expected = torch.tensor(fixture[f"parameter_{number}__{name}"])
                expected_previous = torch.tensor(
                    fixture[f"{'initial' if number == 1 else f'parameter_{number - 1}'}__{name}"]
                )
                actual_delta = parameter.detach() - previous[name]
                expected_delta = expected - expected_previous
                relative_error = torch.linalg.vector_norm(actual_delta - expected_delta) / torch.linalg.vector_norm(
                    expected_delta
                )
                # Separate updates from weights so a small learning rate cannot
                # hide a wrong route, norm sphere or matrix split.
                errors.append((relative_error.item(), name, number))
                if route == "muonh":
                    torch.testing.assert_close(parameter, expected, rtol=3e-3, atol=1.5e-3)
                    state = muon.state[parameter]
                    torch.testing.assert_close(
                        state["momentum_buffer"],
                        torch.tensor(fixture[f"momentum_{number}__{name}"]),
                        rtol=2e-6,
                        atol=5e-7,
                    )
                else:
                    torch.testing.assert_close(parameter, expected, rtol=2e-6, atol=5e-7)
                    state = (adam if route == "adam" else muon).state[parameter]
                    torch.testing.assert_close(
                        state["exp_avg"], torch.tensor(fixture[f"{route}_mu_{number}__{name}"]), rtol=2e-6, atol=5e-7
                    )
                    torch.testing.assert_close(
                        state["exp_avg_sq"], torch.tensor(fixture[f"{route}_nu_{number}__{name}"]), rtol=2e-6, atol=5e-7
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
                muon, adam = optimizers(
                    parameters,
                    routes,
                    offload_momentum=offload_momentum,
                    normalization=normalization,
                    expert_parallel_group=expert_parallel_group,
                )
                muon.initialize_state()
                muon.load_state_dict(restored["muon"])
                adam.load_state_dict(restored["adam"])

        return max(errors)


@pytest.mark.parametrize("normalization", ["all_experts", "per_expert"])
@pytest.mark.parametrize("offload_momentum", [False, True])
def test_hero_ep_routes_updates_moments_and_restart_match_pinned_jax(offload_momentum, normalization):
    error, name, step = oracle_update_error(offload_momentum, normalization)
    assert error < 1e-3, (name, step, error)


def expert_bank_worker(rank, root):
    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    try:
        error = oracle_update_error(True, "all_experts", expert_parallel_group=dist.group.WORLD, expert_rank=rank)
        torch.save(error, Path(root) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_hero_expert_parallel_bank_matches_jax_with_unequal_local_banks(tmp_path):
    mp.spawn(expert_bank_worker, args=(str(tmp_path),), nprocs=2, join=True)
    for rank in range(2):
        error, name, step = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        assert error < 1e-3, (rank, name, step, error)


def test_whole_bank_rejects_unrecognized_routed_expert_names():
    with pytest.raises(ValueError):
        megatron_grug_expert_bank("decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight")
