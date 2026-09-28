# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.domain_phase_mix import launch_delphi_fairness_repeats_3e18 as fairness
from experiments.domain_phase_mix import launch_delphi_frozen_procedure_validation_3e18 as frozen
from experiments.domain_phase_mix import launch_delphi_kl_ablation_3e18 as kl_ablation
from experiments.domain_phase_mix import launch_delphi_matched_olmix_3e18 as launch
from experiments.domain_phase_mix import launch_delphi_olmix_kl005_repeats_3e18 as kl005
from experiments.domain_phase_mix import launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18 as sweep
from experiments.domain_phase_mix import launch_delphi_three_component_optimum_3e18 as three_component
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


def test_matched_olmix_groups_follow_the_seed_design():
    groups = launch.selected_candidates()
    assert [(d.common_data_seed, d.trainer_seed, tuple(c.candidate_id for c in s)) for d, s in groups] == [
        (launch.UNCHEATABLE_DATA_SEED, seed, launch.UNCHEATABLE_GROUPS[seed]) for seed in launch.TRAINER_SEEDS
    ] + [(launch.TABLE9_DATA_SEED, seed, launch.TABLE9_GROUPS[seed]) for seed in launch.TRAINER_SEEDS]
    assert sum(len(candidates) for _, candidates in groups) == launch.MAX_CONCURRENT == 11
    seen = {}
    for definition, candidates in groups:
        for candidate in candidates:
            seen[candidate.candidate_id] = candidate
            assert candidate.target == definition.experiment_name.rsplit("/", 1)[-1].split("_seed")[0]
            assert sum(candidate.runtime_counts.values()) == sweep.MIXTURE_BLOCK_SIZE
            assert candidate.max_materialized_epoch <= candidate.epoch_cap == 4
    assert set(seen) == set(launch.ALL_CANDIDATE_IDS)
    assert seen[launch.UNCHEATABLE_KL0].max_materialized_epoch > 3.9
    assert seen[launch.TABLE9_KL0P005].max_materialized_epoch > 3.9
    assert seen[launch.UNCHEATABLE_KL0P05].max_materialized_epoch < 3.5


def test_matched_olmix_eval_names_are_unique_bounded_and_ids_collide_with_no_earlier_batch():
    template = _template_spec()
    groups = launch.build_validation_run_specs(
        template=template, tpu_type=sweep.TPU_TYPE, tpu_region=sweep.TPU_REGION, tpu_zone=sweep.TPU_ZONE
    )
    names = [sweep.table9_eval_step_name(definition, spec) for definition, _, specs in groups for spec in specs]
    assert len(set(names)) == launch.MAX_CONCURRENT and max(map(len, names)) <= 32
    ids = _run_ids(launch)
    assert len(ids) == launch.MAX_CONCURRENT
    for other in (fairness, frozen, kl_ablation, kl005, three_component):
        assert not ids & _run_ids(other)
