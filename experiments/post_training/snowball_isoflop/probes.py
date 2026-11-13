# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Size-specific native SFTSpec batch, layout and optimizer ablations."""

import math
from dataclasses import dataclass, replace

import draccus
from levanter.optim.config import OptimizerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import StepContext
from marin.training.training import TrainLmOnPodConfig, apply_output_path

from experiments.post_training.snowball_isoflop.config import (
    PREFIX,
    SEQ_LEN,
    VERSION,
    Cell,
    Mask,
    baseline_optimizer,
    build_spec,
    resources,
    training_flops_per_position,
)


@dataclass(frozen=True)
class Layout:
    context: int = 4
    data: int = 2
    expert: int = 1
    replica: int = 1
    dispatch: str = "sonic"
    offload: bool = True

    def __post_init__(self):
        if min(self.context, self.data, self.expert, self.replica) <= 0:
            raise ValueError("Mesh dimensions must be positive")
        if self.context * self.data * self.expert * self.replica != 8:
            raise ValueError("Every probe uses exactly one eight-GPU node")
        if self.dispatch not in ("sonic", "ring_gather_combine"):
            raise ValueError("Only the two approved MoE dispatch backends are compared")

    @property
    def label(self) -> str:
        return (
            f"cp{self.context}-dp{self.data}-ep{self.expert}-r{self.replica}-"
            f"{self.dispatch}-offload{int(self.offload)}"
        )

    @property
    def batch_shards(self) -> int:
        return self.data * self.expert * self.replica

    def mesh(self) -> MeshConfig:
        return MeshConfig(
            axes={
                "context": self.context,
                "data": self.data,
                "expert": self.expert,
                "replica": self.replica,
                "model": 1,
            },
            dcn_axes={"replica_dcn": 1},
            compute_mapping={"batch": ["replica_dcn", "replica", "data", "expert"], "position": "context"},
            param_mapping={"embed": "data"},
        )


@dataclass(frozen=True)
class Probe:
    phase: str
    size: int
    microbatch: int
    accumulation: int
    layout: Layout = Layout()
    steps: int = 8
    seed: int = 0
    optimizer: str = "grug_moe_muonh_v1"
    learning_rate: float = 5e-5

    @property
    def batch_size(self) -> int:
        return self.microbatch * self.accumulation * self.layout.batch_shards

    @property
    def cell(self) -> Cell:
        label = (
            f"{self.phase}-{self.layout.label}-mb{self.microbatch}-acc{self.accumulation}-"
            f"{self.optimizer}-lr{self.learning_rate:g}"
        )
        return Cell(self.size, Mask.ASSISTANT, self.seed, self.steps, (self.steps,), label)


def batch_probes(size: int) -> tuple[Probe, ...]:
    # The accumulation-only probe diagnoses overhead; five microbatch increases
    # select a training batch with at least the approved baseline exposure.
    return (Probe("B-accumulation", size, 1, 1), *(Probe("B", size, microbatch, 4) for microbatch in (1, 2, 4, 8, 16)))


def layout_probes(batch_probe: Probe) -> tuple[Probe, ...]:
    layouts = (
        Layout(),
        Layout(dispatch="ring_gather_combine"),
        Layout(offload=False),
        Layout(context=8, data=1),
        Layout(context=4, data=1, expert=2, dispatch="ring_gather_combine"),
        Layout(context=4, data=1, replica=2),
    )
    batch_size = batch_probe.batch_size
    microbatch = batch_probe.microbatch
    if any(batch_size % (microbatch * layout.batch_shards) for layout in layouts):
        raise ValueError("The B-selected batch and microbatch must divide every C layout")
    return tuple(
        Probe(
            "C-throughput", batch_probe.size, microbatch, batch_size // (microbatch * layout.batch_shards), layout=layout
        )
        for layout in layouts
    )


def at_budget(probe: Probe, budget: float, phase: str) -> Probe:
    steps = math.floor(budget / (training_flops_per_position(probe.size) * SEQ_LEN * probe.batch_size))
    if steps < 1:
        raise ValueError("A budget must cover at least one complete update")
    return replace(probe, phase=phase, steps=steps)


def build_probe_pod(probe: Probe) -> TrainLmOnPodConfig:
    spec = build_spec(probe.cell)
    assert spec.trainer_config is not None and spec.model_config is not None and spec.snowball_sft is not None
    optimizer = draccus.encode(baseline_optimizer(probe.size))
    optimizer["learning_rate"] = probe.learning_rate
    if probe.optimizer == "grug_moe_muonh_v1":
        optimizer["adam_lr"] = probe.learning_rate * 3 / 13
    elif probe.optimizer == "adam":
        for key in ("adam_lr", "momentum", "backend_steps", "muon_epsilon", "coefficient_type"):
            optimizer.pop(key)
        optimizer["nesterov"] = False
        # Pretrain's Muon-H implementation ignored its declared decay field.
        optimizer["weight_decay"] = 0.0
    else:
        raise ValueError(f"Unregistered ablation optimizer {probe.optimizer}")
    optimizer["type"] = probe.optimizer
    spec = replace(
        spec,
        model_config=replace(
            spec.model_config, moe_implementation=probe.layout.dispatch, moe_capacity_factor=float(probe.layout.expert)
        ),
        trainer_config=replace(
            spec.trainer_config,
            offload_opt_state=probe.layout.offload,
            record_device_memory=True,
            max_eval_batches=(
                0
                if probe.phase.startswith("B") or probe.phase == "C-throughput"
                else spec.trainer_config.max_eval_batches
            ),
        ),
        snowball_sft=replace(spec.snowball_sft, record_dispatch_counts=probe.phase.startswith("C")),
        mesh=probe.layout.mesh(),
        batch_size=probe.batch_size,
        per_device_parallelism=probe.microbatch,
        optimizer=draccus.decode(OptimizerConfig, optimizer),
    )
    output = f"{PREFIX}/{VERSION}/{probe.cell.name}"
    pod = spec.model.build_train_config(
        StepContext.for_run(output, PREFIX), spec, spec.prepared_data, resources(), probe.steps
    )
    return replace(pod, train_config=apply_output_path(pod.train_config, output))
