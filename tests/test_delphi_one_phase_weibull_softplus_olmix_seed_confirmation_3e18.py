# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.domain_phase_mix import (
    launch_delphi_one_phase_weibull_softplus_olmix_seed_confirmation_3e18 as confirmation,
)
from tests.test_delphi_one_phase_dsp_epoch_cap_sweep_3e18 import _template_spec


def test_confirmation_selects_only_requested_frozen_coordinates():
    groups = confirmation.selected_candidates()

    assert [[candidate.candidate_id for candidate in candidates] for _, candidates in groups] == [
        ["wspu_uncheatable_cap06"],
        ["wspu_table9_cap06", "wspu_table9_cap07", "wspu_table9_cap08"],
    ]


def test_confirmation_uses_presented_olmix_data_seeds():
    groups = confirmation.build_confirmation_run_specs(
        template=_template_spec(),
        tpu_type="v6e-8",
        tpu_region="us-east5",
        tpu_zone="us-east5-b",
    )
    specs = [run_spec for _, _, run_specs in groups for run_spec in run_specs]

    assert len(specs) == 4
    assert len({spec.run_id for spec in specs}) == 4
    assert {spec.trainer_seed for spec in specs} == {0}
    assert {spec.data_seed for spec in specs if spec.source_run_name.startswith("wspu_uncheatable")} == {666_200}
    assert {spec.data_seed for spec in specs if spec.source_run_name.startswith("wspu_table9")} == {662_009}
    assert all(spec.phase_weights["phase_0"] == spec.phase_weights["phase_1"] for spec in specs)


def test_confirmation_eval_names_are_unique_and_bounded():
    groups = confirmation.build_confirmation_run_specs(
        template=_template_spec(),
        tpu_type="v6e-8",
        tpu_region="us-east5",
        tpu_zone="us-east5-b",
    )
    names = [
        confirmation.sweep.table9_eval_step_name(definition, run_spec)
        for definition, _, run_specs in groups
        for run_spec in run_specs
    ]

    assert len(set(names)) == 4
    assert max(map(len, names)) <= 32
