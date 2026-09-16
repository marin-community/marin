# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from experiments.domain_phase_mix import launch_delphi_floor_replicates_3e18 as launch
from experiments.domain_phase_mix import launch_delphi_frontier_factorial_3e18 as factorial
from experiments.domain_phase_mix import launch_delphi_link_hub_validation_3e18 as hub
from experiments.domain_phase_mix import launch_delphi_link_validation_3e18 as link
from experiments.domain_phase_mix import launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18 as sweep
from experiments.domain_phase_mix import launch_delphi_wspu_coupling_validation_3e18 as coupling
from tests.test_delphi_one_phase_dsp_epoch_cap_sweep_3e18 import _template_spec


def test_floor_tables_hold_the_five_design_rows_twice():
    groups = launch.selected_candidates()
    assert [definition.trainer_seed for definition, _ in groups] == [0, 1]
    design = pd.read_csv(launch.DEFAULT_CANDIDATE_DIR / "design.csv")
    for _, candidates in groups:
        assert [candidate.candidate_id for candidate in candidates] == list(design.candidate_id)
        assert len({tuple(candidate.runtime_counts.values()) for candidate in candidates}) == 5
        assert all(sum(candidate.runtime_counts.values()) == sweep.MIXTURE_BLOCK_SIZE for candidate in candidates)
        assert all(candidate.max_materialized_epoch <= candidate.epoch_cap for candidate in candidates)
    first, second = (candidates for _, candidates in groups)
    assert [c.runtime_counts for c in first] == [c.runtime_counts for c in second]


def test_floor_run_specs_share_the_data_seed_and_split_trainer_seeds():
    template = _template_spec()
    groups = launch.build_validation_run_specs(
        template=template,
        tpu_type=sweep.TPU_TYPE,
        tpu_region=sweep.TPU_REGION,
        tpu_zone=sweep.TPU_ZONE,
    )
    all_specs = [spec for _, _, specs in groups for spec in specs]
    assert len(all_specs) == launch.MAX_CONCURRENT == 10
    assert len({spec.run_id for spec in all_specs}) == 10
    assert len({spec.run_name for spec in all_specs}) == 10
    assert {spec.data_seed for spec in all_specs} == {launch.TABLE9_DATA_SEED}
    assert {definition.trainer_seed: len(specs) for definition, _, specs in groups} == {0: 5, 1: 5}
    names = [sweep.table9_eval_step_name(definition, spec) for definition, _, specs in groups for spec in specs]
    assert len(set(names)) == 10
    assert max(map(len, names)) <= 32


def test_floor_run_ids_do_not_collide_with_the_other_launches():
    template = _template_spec()

    def run_ids(module):
        return {
            spec.run_id
            for _, _, specs in module.build_validation_run_specs(
                template=template, tpu_type=sweep.TPU_TYPE, tpu_region=sweep.TPU_REGION, tpu_zone=sweep.TPU_ZONE
            )
            for spec in specs
        }

    mine = run_ids(launch)
    for module in (coupling, link, hub, factorial):
        assert not mine & run_ids(module)
