"""CPU checkpoint continuation through the pinned Megatron optimizer loader."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
from megatron.core.optimizer.optimizer_config import OptimizerConfig

from skyrl_train.distributed.megatron.checkpoint_restore import load_optimizer_checkpoint


def cpu_optimizer(parameter, optimizer_type=torch.optim.AdamW):
    group = dict(params=[parameter], wd_mult=1.0, lr_mult=1.0, is_expert_parallel=False, is_decoupled_lr=False)
    inner = optimizer_type([group], lr=0.002, weight_decay=0.03, foreach=False)
    # MCore's constructor allocates CUDA buffers. Supply one CPU bucket while
    # retaining its real state_dict, load_state_dict and bucket-loading methods.
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer.optimizer = inner
    optimizer.config = OptimizerConfig(
        optimizer="adam",
        optimizer_cpu_offload=True,
        optimizer_offload_fraction=1.0,
        use_distributed_optimizer=True,
        bf16=True,
    )
    optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    optimizer.grad_scaler = None
    optimizer.is_stub_optimizer = False
    optimizer.model_chunks = []
    optimizer.model_param_group_index_map = {parameter: (0, 0)}
    extent = Range(0, parameter.numel())
    optimizer.gbuf_ranges = [
        {(torch.bfloat16, torch.float32): [{"param_map": {parameter: {"gbuf_world": extent, "gbuf_local": extent}}}]}
    ]
    optimizer.per_bucket_numel = [{(torch.bfloat16, torch.float32): [parameter.numel()]}]
    optimizer.per_bucket_numel_unpadded = optimizer.per_bucket_numel
    return optimizer


def unsharded_cpu_optimizer(parameter):
    group = dict(params=[parameter], wd_mult=1.0, lr_mult=1.0, is_expert_parallel=False, is_decoupled_lr=False)
    optimizer = Float16OptimizerWithFloat16Params.__new__(Float16OptimizerWithFloat16Params)
    optimizer.optimizer = torch.optim.AdamW([group], lr=0.002, weight_decay=0.03, foreach=False)
    optimizer.config = OptimizerConfig(optimizer="adam", optimizer_cpu_offload=True, bf16=True)
    optimizer.is_stub_optimizer = False
    optimizer.grad_scaler = None
    optimizer.fp32_from_float16_groups = [[parameter]]
    return optimizer


def advance(optimizer, parameter, iteration):
    parameter.grad = torch.tensor([0.3, -0.5, 0.9]) * iteration
    optimizer.optimizer.step()


def saved_checkpoint(optimizer, parameter):
    state = optimizer.state_dict()
    tensors = optimizer._get_main_param_and_optimizer_states(parameter)
    # The affected bucket format persists weights and moments; scalar steps
    # are saved separately in the optimizer's parameter-group metadata.
    bucket = {key: tensors[key].detach().clone() for key in ("param", "exp_avg", "exp_avg_sq")}
    bucket["padding"] = False
    state.update(
        param_state_sharding_type="dp_reshardable",
        param_state={
            "per_bucket_numel_unpadded": deepcopy(optimizer.per_bucket_numel_unpadded),
            0: {(torch.bfloat16, torch.float32): [[bucket, {"padding": True}]]},
        },
    )
    return state


def assert_optimizer_matches(actual, actual_parameter, expected, expected_parameter):
    torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)
    for name in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            actual.optimizer.state[actual_parameter][name],
            expected.optimizer.state[expected_parameter][name],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("optimizer_type", [torch.optim.Adam, torch.optim.AdamW])
@pytest.mark.parametrize("local_bucket_step", [False, True])
def test_optimizer_resume_preserves_state_and_next_update(tmp_path, optimizer_type, local_bucket_step):
    reference_parameter = torch.nn.Parameter(torch.tensor([0.1, -0.8, 1.3]))
    reference = cpu_optimizer(reference_parameter, optimizer_type)
    for iteration in range(1, 6):
        advance(reference, reference_parameter, iteration)
    path = tmp_path / "optimizer.pt"
    torch.save(saved_checkpoint(reference, reference_parameter), path)
    state = torch.load(path, weights_only=False)
    bucket = state["param_state"][0][(torch.bfloat16, torch.float32)][0][0]

    parameter = torch.nn.Parameter(torch.zeros_like(reference_parameter))
    resumed = cpu_optimizer(parameter, optimizer_type)
    for _ in range(5):
        advance(resumed, parameter, 0)
    if local_bucket_step:
        # MCore restores nonpersistent bucket counters from the receiver's local template.
        bucket["step"] = resumed.optimizer.state[parameter]["step"]
    with torch.no_grad():
        load_optimizer_checkpoint(resumed, state)
    assert_optimizer_matches(resumed, parameter, reference, reference_parameter)

    # The reference has never loaded a checkpoint or shared tensors with resumed.
    advance(reference, reference_parameter, 6)
    advance(resumed, parameter, 6)
    assert_optimizer_matches(resumed, parameter, reference, reference_parameter)


@pytest.mark.parametrize("steps", [[None], [5, 6]])
def test_ambiguous_saved_counter_fails_before_optimizer_restore(steps):
    source_parameter = torch.nn.Parameter(torch.tensor([0.1, -0.8, 1.3]))
    source = cpu_optimizer(source_parameter)
    advance(source, source_parameter, 5)
    state = saved_checkpoint(source, source_parameter)
    group = state["optimizer"]["param_groups"][0]
    state["optimizer"]["param_groups"] = [{**group, "step": step} for step in steps]
    parameter = torch.nn.Parameter(torch.zeros_like(source_parameter))
    receiver = cpu_optimizer(parameter)
    advance(receiver, parameter, 0)
    with pytest.raises(ValueError, match="one saved step shared by all parameter groups"):
        load_optimizer_checkpoint(receiver, state)
    torch.testing.assert_close(parameter, torch.zeros_like(parameter), rtol=0, atol=0)
    assert receiver.optimizer.state[parameter]["step"].item() == 1


def test_chained_optimizer_recovers_each_childs_saved_counter(tmp_path):
    references, reference_parameters, states = [], [], {}
    receivers, receiver_parameters = [], []
    for index in range(2):
        reference_parameter = torch.nn.Parameter(torch.tensor([0.1, -0.8, 1.3]) + index)
        reference = cpu_optimizer(reference_parameter)
        for iteration in range(1, 6):
            advance(reference, reference_parameter, iteration)
        references.append(reference)
        reference_parameters.append(reference_parameter)
        states[index] = saved_checkpoint(reference, reference_parameter)
        parameter = torch.nn.Parameter(torch.zeros_like(reference_parameter))
        receiver = cpu_optimizer(parameter)
        advance(receiver, parameter, 0)
        receivers.append(receiver)
        receiver_parameters.append(parameter)
    path = tmp_path / "optimizer.pt"
    torch.save(states, path)
    with torch.no_grad():
        load_optimizer_checkpoint(ChainedOptimizer(receivers), torch.load(path, weights_only=False))
    for reference, reference_parameter, receiver, parameter in zip(
        references, reference_parameters, receivers, receiver_parameters, strict=True
    ):
        assert_optimizer_matches(receiver, parameter, reference, reference_parameter)
        advance(reference, reference_parameter, 7)
        advance(receiver, parameter, 7)
        assert_optimizer_matches(receiver, parameter, reference, reference_parameter)


def test_rank_without_trainable_parameters_can_restore_its_empty_state():
    # MCore's empty chain has no config; restoring it must still work on ranks
    # without trainable parameters.
    optimizer = ChainedOptimizer([])
    load_optimizer_checkpoint(optimizer, {})
    assert optimizer.state_dict() == []


def test_unsharded_optimizer_preserves_ordinary_checkpoint_continuation(tmp_path):
    reference_parameter = torch.nn.Parameter(torch.tensor([0.1, -0.8, 1.3]))
    reference = unsharded_cpu_optimizer(reference_parameter)
    for iteration in range(1, 6):
        advance(reference, reference_parameter, iteration)
    path = tmp_path / "optimizer.pt"
    torch.save(reference.state_dict(), path)
    parameter = torch.nn.Parameter(torch.zeros_like(reference_parameter))
    resumed = unsharded_cpu_optimizer(parameter)
    with torch.no_grad():
        load_optimizer_checkpoint(resumed, torch.load(path, weights_only=False))
    assert_optimizer_matches(resumed, parameter, reference, reference_parameter)
    advance(reference, reference_parameter, 6)
    advance(resumed, parameter, 6)
    assert_optimizer_matches(resumed, parameter, reference, reference_parameter)
