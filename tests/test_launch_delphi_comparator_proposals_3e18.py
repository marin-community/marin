# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.domain_phase_mix import launch_delphi_comparator_proposals_3e18 as launch
from experiments.domain_phase_mix import launch_delphi_fairness_repeats_3e18 as fairness
from experiments.domain_phase_mix import launch_delphi_frozen_procedure_validation_3e18 as frozen
from experiments.domain_phase_mix import launch_delphi_kl_ablation_3e18 as kl_ablation
from experiments.domain_phase_mix import launch_delphi_matched_olmix_3e18 as matched_olmix
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


def test_comparator_groups_follow_the_seed_design():
    groups = launch.selected_candidates()
    assert [(d.common_data_seed, d.trainer_seed, tuple(c.candidate_id for c in s)) for d, s in groups] == [
        (launch.UNCHEATABLE_DATA_SEED, seed, launch.UNCHEATABLE_GROUPS[seed]) for seed in launch.TRAINER_SEEDS
    ] + [(launch.TABLE9_DATA_SEED, seed, launch.TABLE9_GROUPS[seed]) for seed in launch.TRAINER_SEEDS]
    assert sum(len(candidates) for _, candidates in groups) == launch.MAX_CONCURRENT
    seen = {}
    for definition, candidates in groups:
        for candidate in candidates:
            seen[candidate.candidate_id] = candidate
            assert candidate.target == definition.experiment_name.rsplit("/", 1)[-1].split("_seed")[0]
            assert sum(candidate.runtime_counts.values()) == sweep.MIXTURE_BLOCK_SIZE
            # The caps are nominal: every proposal was optimized without one and lies strictly inside its cap.
            assert candidate.max_materialized_epoch < candidate.epoch_cap
    # The Uncheatable power-one proposal is the frozen mixture and is listed but not retrained.
    assert set(seen) == set(launch.ALL_CANDIDATE_IDS) - {launch.UNCHEATABLE_MARINER_POWER1}


def test_comparator_proposals_differ_from_the_frozen_procedure():
    frozen_groups = frozen.selected_candidates()
    frozen_weights = {
        candidate.candidate_id: candidate.weights for _, candidates in frozen_groups for candidate in candidates
    }
    reference = {"uncheatable": frozen_weights["lwspu_u_snc_cap06"], "table9": frozen_weights["lwspu_t9_snc_cap08"]}
    for _, candidates in launch.selected_candidates():
        for candidate in candidates:
            distance = sum(abs(candidate.weights[d] - reference[candidate.target][d]) for d in candidate.weights) / 2
            assert distance > 0.01, candidate.candidate_id


def test_comparator_eval_names_are_unique_bounded_and_ids_collide_with_no_earlier_batch():
    template = _template_spec()
    groups = launch.build_validation_run_specs(
        template=template, tpu_type=sweep.TPU_TYPE, tpu_region=sweep.TPU_REGION, tpu_zone=sweep.TPU_ZONE
    )
    names = [sweep.table9_eval_step_name(definition, spec) for definition, _, specs in groups for spec in specs]
    assert len(set(names)) == launch.MAX_CONCURRENT and max(map(len, names)) <= 32
    ids = _run_ids(launch)
    assert len(ids) == launch.MAX_CONCURRENT
    for other in (fairness, frozen, kl_ablation, kl005, matched_olmix, three_component):
        assert not ids & _run_ids(other)
