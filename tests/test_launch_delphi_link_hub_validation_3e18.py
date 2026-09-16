# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from experiments.domain_phase_mix import launch_delphi_link_hub_validation_3e18 as launch
from experiments.domain_phase_mix import launch_delphi_link_validation_3e18 as link
from experiments.domain_phase_mix import launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18 as sweep
from experiments.domain_phase_mix import launch_delphi_wspu_coupling_validation_3e18 as coupling
from tests.test_delphi_one_phase_dsp_epoch_cap_sweep_3e18 import _template_spec


def test_hub_candidate_table_matches_the_frozen_two_policies():
    groups = launch.selected_candidates()

    assert [candidate.candidate_id for _, selected in groups for candidate in selected] == list(launch.ALL_CANDIDATE_IDS)
    candidates = [candidate for _, selected in groups for candidate in selected]
    assert len({tuple(candidate.runtime_counts.values()) for candidate in candidates}) == 2
    assert all(sum(candidate.runtime_counts.values()) == sweep.MIXTURE_BLOCK_SIZE for candidate in candidates)
    assert all(candidate.max_materialized_epoch <= candidate.epoch_cap for candidate in candidates)
    assert [(candidate.target, candidate.epoch_cap) for candidate in candidates] == [("table9", 6), ("table9", 8)]


def test_hub_table9_eval_names_are_unique_and_bounded():
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


def test_hub_run_specs_bind_to_the_table9_comparator_seed():
    template = _template_spec()
    groups = launch.build_validation_run_specs(
        template=template,
        tpu_type=sweep.TPU_TYPE,
        tpu_region=sweep.TPU_REGION,
        tpu_zone=sweep.TPU_ZONE,
    )
    seeds = {definition.common_data_seed for definition, _, _ in groups}
    assert seeds == {launch.TABLE9_DATA_SEED}
    all_specs = [spec for _, _, specs in groups for spec in specs]
    assert len(all_specs) == launch.MAX_CONCURRENT
    assert len({spec.run_id for spec in all_specs}) == launch.MAX_CONCURRENT
    assert len({spec.run_name for spec in all_specs}) == launch.MAX_CONCURRENT
    for definition, _, specs in groups:
        assert {spec.data_seed for spec in specs} == {definition.common_data_seed}
        assert {spec.trainer_seed for spec in specs} == {launch.TRAINER_SEED}
    assert all(spec.phase_weights["phase_0"] == spec.phase_weights["phase_1"] for spec in all_specs)
    assert all(
        replace(
            spec,
            run_order=template.run_order,
            run_id=template.run_id,
            run_name=template.run_name,
            source_run_name=template.source_run_name,
            source_experiment=template.source_experiment,
            panel_source=template.panel_source,
            tpu_type=template.tpu_type,
            tpu_region=template.tpu_region,
            tpu_zone=template.tpu_zone,
            tensor_parallel_size=template.tensor_parallel_size,
            data_seed=template.data_seed,
            trainer_seed=template.trainer_seed,
            max_simulated_epoch=template.max_simulated_epoch,
            q95_simulated_epoch=template.q95_simulated_epoch,
            mean_phase_tv_to_proportional=template.mean_phase_tv_to_proportional,
            phase_weights=template.phase_weights,
        )
        == template
        for spec in all_specs
    )


def test_hub_run_ids_do_not_collide_with_the_other_validations():
    template = _template_spec()
    hub_ids = {
        spec.run_id
        for _, _, specs in launch.build_validation_run_specs(
            template=template, tpu_type=sweep.TPU_TYPE, tpu_region=sweep.TPU_REGION, tpu_zone=sweep.TPU_ZONE
        )
        for spec in specs
    }
    other_ids = {
        spec.run_id
        for module in (coupling, link)
        for _, _, specs in module.build_validation_run_specs(
            template=template, tpu_type=sweep.TPU_TYPE, tpu_region=sweep.TPU_REGION, tpu_zone=sweep.TPU_ZONE
        )
        for spec in specs
    }
    assert not hub_ids & other_ids
