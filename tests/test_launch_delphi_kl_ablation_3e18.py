# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


from experiments.domain_phase_mix import launch_delphi_frozen_procedure_validation_3e18 as frozen
from experiments.domain_phase_mix import launch_delphi_kappa_floor_flat_validation_3e18 as flat
from experiments.domain_phase_mix import launch_delphi_kl_ablation_3e18 as launch
from experiments.domain_phase_mix import launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18 as sweep
from experiments.domain_phase_mix import launch_delphi_wspu_coupling_validation_3e18 as coupling
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


def test_kl_ablation_table_holds_eight_penalized_proposals_per_objective_under_inactive_caps():
    groups = launch.selected_candidates()
    candidates = [candidate for _, selected in groups for candidate in selected]
    assert [candidate.candidate_id for candidate in candidates] == list(launch.ALL_CANDIDATE_IDS)
    assert len({tuple(candidate.runtime_counts.values()) for candidate in candidates}) == 16
    assert all(sum(candidate.runtime_counts.values()) == sweep.MIXTURE_BLOCK_SIZE for candidate in candidates)
    assert all(candidate.max_materialized_epoch < candidate.epoch_cap for candidate in candidates)
    assert [(candidate.target, candidate.epoch_cap) for candidate in candidates] == [("uncheatable", 6)] * 8 + [
        ("table9", 8)
    ] * 8


def test_kl_ablation_repetition_falls_with_the_penalty():
    for _, selected in launch.selected_candidates():
        epochs = [candidate.max_materialized_epoch for candidate in selected]
        assert epochs == sorted(epochs, reverse=True)


def test_kl_ablation_eval_names_are_unique_and_bounded():
    template = _template_spec()
    names = [
        sweep.table9_eval_step_name(definition, spec)
        for definition, _, specs in launch.build_validation_run_specs(
            template=template, tpu_type=sweep.TPU_TYPE, tpu_region=sweep.TPU_REGION, tpu_zone=sweep.TPU_ZONE
        )
        for spec in specs
    ]
    assert len(set(names)) == launch.MAX_CONCURRENT
    assert max(map(len, names)) <= 32


def test_kl_ablation_shares_the_control_seeds_and_collides_with_no_earlier_batch():
    template = _template_spec()
    groups = launch.build_validation_run_specs(
        template=template, tpu_type=sweep.TPU_TYPE, tpu_region=sweep.TPU_REGION, tpu_zone=sweep.TPU_ZONE
    )
    assert {definition.common_data_seed for definition, _, _ in groups} == {
        frozen.UNCHEATABLE_DATA_SEED,
        frozen.TABLE9_DATA_SEED,
    }
    assert all(spec.trainer_seed == frozen.TRAINER_SEED for _, _, specs in groups for spec in specs)
    ids = _run_ids(launch)
    assert len(ids) == launch.MAX_CONCURRENT
    assert not ids & _run_ids(frozen) and not ids & _run_ids(flat) and not ids & _run_ids(coupling)
