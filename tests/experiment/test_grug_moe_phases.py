# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The grug-moe phase chain (pretrain -> midtrain -> SFT -> RL) must lower to a
checkpoint lineage where each phase initializes from its parent's checkpoint
directory, resolved without the executor.
"""

from marin.execution.lazy import Checkpoint, lower, materialized_config

from experiments.grug.moe.phases_lazy import phase_chain


def _parent(handle: Checkpoint) -> Checkpoint | None:
    return next((dep for dep in handle.recipe.deps if isinstance(dep, Checkpoint)), None)


def test_phase_chain_lowers_to_a_pretrain_to_rl_lineage():
    rl = phase_chain()
    spec = lower(rl)

    lineage = []
    current = rl
    while current is not None:
        lineage.append(current.name)
        current = _parent(current)

    assert lineage == [
        "grug/phases/rl",
        "grug/phases/sft",
        "grug/phases/midtrain",
        "grug/phases/pretrain",
    ]
    # Each phase also depends on the shared data cache so it materializes first.
    assert {dep.name for dep in spec.deps} == {"grug/phases/sft", "fineweb-edu-10M"}


def test_each_phase_initializes_from_its_parents_checkpoints():
    prefix = "gs://marin-golden"
    rl = phase_chain()

    init_by_name = {}
    current = rl
    while current is not None:
        init_by_name[current.name] = materialized_config(current, prefix).init_from
        current = _parent(current)

    assert init_by_name["grug/phases/pretrain"] is None
    assert init_by_name["grug/phases/midtrain"] == f"{prefix}/grug/phases/pretrain/v1/checkpoints"
    assert init_by_name["grug/phases/sft"] == f"{prefix}/grug/phases/midtrain/v1/checkpoints"
    assert init_by_name["grug/phases/rl"] == f"{prefix}/grug/phases/sft/v1/checkpoints"
