# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from experiments.domain_phase_mix import launch_delphi_frontier_factorial_3e18 as launch
from experiments.domain_phase_mix import launch_delphi_link_hub_validation_3e18 as hub
from experiments.domain_phase_mix import launch_delphi_link_validation_3e18 as link
from experiments.domain_phase_mix import launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18 as sweep
from experiments.domain_phase_mix import launch_delphi_wspu_coupling_validation_3e18 as coupling
from tests.test_delphi_one_phase_dsp_epoch_cap_sweep_3e18 import _template_spec


def test_factorial_table_matches_the_frozen_design():
    groups = launch.selected_candidates()
    candidates = [candidate for _, selected in groups for candidate in selected]

    assert [candidate.candidate_id for candidate in candidates] == list(launch.ALL_CANDIDATE_IDS)
    assert len(candidates) == 18
    assert len({tuple(candidate.runtime_counts.values()) for candidate in candidates}) == 17
    assert all(sum(candidate.runtime_counts.values()) == sweep.MIXTURE_BLOCK_SIZE for candidate in candidates)
    assert all(candidate.max_materialized_epoch <= candidate.epoch_cap for candidate in candidates)
    design = pd.read_csv(launch.DEFAULT_CANDIDATE_DIR / "design.csv").set_index("candidate_id")
    signs = design[[column for column in design.columns if column.startswith("factor_")]]
    corners = signs.loc[list(launch.FACTORIAL_CORNER_IDS)]
    assert (corners.abs() == 1).all().all()
    assert (corners.iloc[:, :4].prod(axis=1) == corners.iloc[:, 4]).all()
    assert (corners.sum(axis=0) == 0).all()


def test_factorial_run_specs_share_the_data_seed_and_split_trainer_seeds():
    template = _template_spec()
    groups = launch.build_validation_run_specs(
        template=template,
        tpu_type=sweep.TPU_TYPE,
        tpu_region=sweep.TPU_REGION,
        tpu_zone=sweep.TPU_ZONE,
    )
    all_specs = [spec for _, _, specs in groups for spec in specs]
    assert len(all_specs) == launch.MAX_CONCURRENT
    assert len({spec.run_id for spec in all_specs}) == launch.MAX_CONCURRENT
    assert len({spec.run_name for spec in all_specs}) == launch.MAX_CONCURRENT
    assert {spec.data_seed for spec in all_specs} == {launch.TABLE9_DATA_SEED}
    trainer_seeds = {definition.trainer_seed: len(specs) for definition, _, specs in groups}
    assert trainer_seeds == {0: 17, 1: 1}
    names = [sweep.table9_eval_step_name(definition, spec) for definition, _, specs in groups for spec in specs]
    assert len(set(names)) == launch.MAX_CONCURRENT
    assert max(map(len, names)) <= 32


def test_factorial_run_ids_do_not_collide_with_the_validations():
    template = _template_spec()
    factorial_ids = {
        spec.run_id
        for _, _, specs in launch.build_validation_run_specs(
            template=template, tpu_type=sweep.TPU_TYPE, tpu_region=sweep.TPU_REGION, tpu_zone=sweep.TPU_ZONE
        )
        for spec in specs
    }
    other_ids = {
        spec.run_id
        for module in (coupling, link, hub)
        for _, _, specs in module.build_validation_run_specs(
            template=template, tpu_type=sweep.TPU_TYPE, tpu_region=sweep.TPU_REGION, tpu_zone=sweep.TPU_ZONE
        )
        for spec in specs
    }
    assert not factorial_ids & other_ids
