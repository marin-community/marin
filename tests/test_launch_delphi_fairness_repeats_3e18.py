# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


from experiments.domain_phase_mix import launch_delphi_fairness_repeats_3e18 as launch
from experiments.domain_phase_mix import launch_delphi_frozen_procedure_validation_3e18 as frozen
from experiments.domain_phase_mix import launch_delphi_kappa_floor_flat_validation_3e18 as flat
from experiments.domain_phase_mix import launch_delphi_kl_ablation_3e18 as kl_ablation
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


def test_fairness_repeats_hold_twelve_seed_matched_rows():
    groups = launch.selected_candidates()
    layout = [
        (definition.common_data_seed, definition.trainer_seed, tuple(c.candidate_id for c in selected))
        for definition, selected in groups
    ]
    assert layout == [
        (launch.UNCHEATABLE_DATA_SEED, 0, (launch.OLMIX_UNCHEATABLE,)),
        (launch.UNCHEATABLE_DATA_SEED, 1, (launch.OLMIX_UNCHEATABLE, launch.OURS_UNCHEATABLE)),
        (launch.UNCHEATABLE_DATA_SEED, 2, (launch.OLMIX_UNCHEATABLE, launch.OURS_UNCHEATABLE)),
        (launch.TABLE9_DATA_SEED, 0, (launch.OLMIX_TABLE9,)),
        (launch.TABLE9_DATA_SEED, 1, (launch.OLMIX_TABLE9, *launch.OURS_TABLE9)),
        (launch.TABLE9_DATA_SEED, 2, (launch.OLMIX_TABLE9, *launch.OURS_TABLE9)),
    ]
    assert sum(len(selected) for _, selected in groups) == launch.MAX_CONCURRENT


def test_fairness_repeats_reuse_the_frozen_procedure_mixtures_verbatim():
    ours = {c.candidate_id: c for _, selected in launch.selected_candidates() for c in selected}
    for _, selected in frozen.selected_candidates():
        for candidate in selected:
            assert ours[candidate.candidate_id].runtime_counts == candidate.runtime_counts
            assert ours[candidate.candidate_id].epoch_cap == candidate.epoch_cap


def test_fairness_repeats_olmix_rows_are_grid_mixtures_at_olmix_repetition():
    olmix = {
        c.candidate_id: c
        for _, selected in launch.selected_candidates()
        for c in selected
        if c.candidate_id.startswith("olmix")
    }
    assert set(olmix) == {launch.OLMIX_UNCHEATABLE, launch.OLMIX_TABLE9}
    for candidate in olmix.values():
        assert sum(candidate.runtime_counts.values()) == sweep.MIXTURE_BLOCK_SIZE
        assert 3.9 < candidate.max_materialized_epoch < 4.1


def test_fairness_repeats_eval_names_are_unique_and_bounded():
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


def test_fairness_repeats_run_ids_collide_with_no_earlier_batch():
    ids = _run_ids(launch)
    assert len(ids) == launch.MAX_CONCURRENT
    for other in (frozen, flat, coupling, kl_ablation):
        assert not ids & _run_ids(other)
