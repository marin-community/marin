# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.domain_phase_mix import launch_delphi_fairness_repeats_3e18 as fairness
from experiments.domain_phase_mix import launch_delphi_frozen_procedure_validation_3e18 as frozen
from experiments.domain_phase_mix import launch_delphi_kl_ablation_3e18 as kl_ablation
from experiments.domain_phase_mix import launch_delphi_olmix_kl005_repeats_3e18 as olmix_repeats
from experiments.domain_phase_mix import launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18 as sweep
from experiments.domain_phase_mix import launch_delphi_three_component_optimum_3e18 as launch
from tests.test_delphi_one_phase_dsp_epoch_cap_sweep_3e18 import _template_spec


def _run_ids(module) -> set[int]:
    template = _template_spec()
    return {
        spec.run_id
        for _, _, specs in module.build_validation_run_specs(
            template=template, tpu_type=sweep.TPU_TYPE, tpu_region=sweep.TPU_REGION, tpu_zone=sweep.TPU_ZONE
        )
        for spec in specs
    }


def test_three_component_optimum_trains_one_grid_mixture_at_three_trainer_seeds():
    groups = launch.selected_candidates()
    assert [(d.common_data_seed, d.trainer_seed, tuple(c.candidate_id for c in s)) for d, s in groups] == [
        (launch.UNCHEATABLE_DATA_SEED, seed, (launch.THREE_COMPONENT_OPTIMUM,)) for seed in launch.TRAINER_SEEDS
    ]
    candidate = groups[0][1][0]
    assert sum(candidate.runtime_counts.values()) == sweep.MIXTURE_BLOCK_SIZE
    assert 5.6 < candidate.max_materialized_epoch <= candidate.epoch_cap == 6
    assert candidate.runtime_counts["dolma3_stack_edu"] == candidate.runtime_counts["dolmino_stack_edu_fim"] == 0


def test_three_component_optimum_eval_names_are_unique_bounded_and_ids_collide_with_no_earlier_batch():
    template = _template_spec()
    groups = launch.build_validation_run_specs(
        template=template, tpu_type=sweep.TPU_TYPE, tpu_region=sweep.TPU_REGION, tpu_zone=sweep.TPU_ZONE
    )
    names = [sweep.table9_eval_step_name(definition, spec) for definition, _, specs in groups for spec in specs]
    assert len(set(names)) == launch.MAX_CONCURRENT and max(map(len, names)) <= 32
    ids = _run_ids(launch)
    assert len(ids) == launch.MAX_CONCURRENT
    for other in (fairness, frozen, kl_ablation, olmix_repeats):
        assert not ids & _run_ids(other)
