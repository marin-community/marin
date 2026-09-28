# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from experiments.domain_phase_mix import launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18 as sweep
from experiments.domain_phase_mix import launch_delphi_one_phase_weibull_softplus_epoch_cap_sweep_3e18 as launch
from tests.test_delphi_one_phase_dsp_epoch_cap_sweep_3e18 import _template_spec


def test_successor_candidate_contract_deduplicates_only_the_uncheatable_plateau():
    candidates, alias_map = sweep.load_candidate_mixtures(
        launch.DEFAULT_CANDIDATE_WEIGHTS,
        launch.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
        definition=launch.SWEEP_DEFINITION,
    )

    expected_ids = [
        *(f"wspu_uncheatable_cap{cap:02d}" for cap in range(2, 7)),
        *(f"wspu_table9_cap{cap:02d}" for cap in range(2, 9)),
    ]
    assert [candidate.candidate_id for candidate in candidates] == expected_ids
    assert alias_map == launch.EXPECTED_ALIAS_MAP
    assert len({tuple(candidate.runtime_counts.values()) for candidate in candidates}) == 12
    assert all(sum(candidate.runtime_counts.values()) == sweep.MIXTURE_BLOCK_SIZE for candidate in candidates)
    assert all(candidate.max_materialized_epoch <= candidate.epoch_cap for candidate in candidates)


def test_successor_run_specs_change_only_policy_and_runtime_identity():
    candidates, _ = sweep.load_candidate_mixtures(
        launch.DEFAULT_CANDIDATE_WEIGHTS,
        launch.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
        definition=launch.SWEEP_DEFINITION,
    )
    template = _template_spec()
    specs = sweep.build_run_specs(
        template=template,
        candidates=candidates,
        tpu_type=sweep.TPU_TYPE,
        tpu_region=sweep.TPU_REGION,
        tpu_zone=sweep.TPU_ZONE,
        definition=launch.SWEEP_DEFINITION,
    )

    assert len(specs) == launch.EXPECTED_RUN_COUNT
    assert len({spec.run_id for spec in specs}) == launch.EXPECTED_RUN_COUNT
    assert {spec.data_seed for spec in specs} == {sweep.COMMON_DATA_SEED}
    assert {spec.trainer_seed for spec in specs} == {sweep.TRAINER_SEED}
    assert all(spec.phase_weights["phase_0"] == spec.phase_weights["phase_1"] for spec in specs)
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
        for spec in specs
    )


def test_successor_table9_eval_names_are_unique_and_bounded():
    candidates, _ = sweep.load_candidate_mixtures(
        launch.DEFAULT_CANDIDATE_WEIGHTS,
        launch.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
        definition=launch.SWEEP_DEFINITION,
    )
    specs = sweep.build_run_specs(
        template=_template_spec(),
        candidates=candidates,
        tpu_type=sweep.TPU_TYPE,
        tpu_region=sweep.TPU_REGION,
        tpu_zone=sweep.TPU_ZONE,
        definition=launch.SWEEP_DEFINITION,
    )

    names = [sweep.table9_eval_step_name(launch.SWEEP_DEFINITION, spec) for spec in specs]
    assert len(set(names)) == launch.EXPECTED_RUN_COUNT
    assert max(map(len, names)) <= 32
