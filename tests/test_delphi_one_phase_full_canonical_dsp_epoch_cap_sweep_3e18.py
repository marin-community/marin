# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.domain_phase_mix import launch_delphi_one_phase_dsp_epoch_cap_sweep_3e18 as sweep
from experiments.domain_phase_mix import launch_delphi_one_phase_full_canonical_dsp_epoch_cap_sweep_3e18 as full
from tests.test_delphi_one_phase_dsp_epoch_cap_sweep_3e18 import _template_spec


def test_full_canonical_candidate_table_is_exact_and_cap_bounded():
    candidates, alias_map = sweep.load_candidate_mixtures(
        full.DEFAULT_CANDIDATE_WEIGHTS,
        full.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
        definition=full.SWEEP_DEFINITION,
    )

    assert [candidate.candidate_id for candidate in candidates] == list(full.NOMINAL_CANDIDATE_IDS)
    assert alias_map == full.EXPECTED_ALIAS_MAP
    assert len(candidates) == 16
    assert all(candidate.max_materialized_epoch <= candidate.epoch_cap for candidate in candidates)


def test_full_canonical_run_specs_preserve_common_random_numbers():
    candidates, _ = sweep.load_candidate_mixtures(
        full.DEFAULT_CANDIDATE_WEIGHTS,
        full.EXPECTED_CANDIDATE_WEIGHTS_SHA256,
        definition=full.SWEEP_DEFINITION,
    )
    specs = sweep.build_run_specs(
        template=_template_spec(),
        candidates=candidates,
        tpu_type=sweep.TPU_TYPE,
        tpu_region=sweep.TPU_REGION,
        tpu_zone=sweep.TPU_ZONE,
        definition=full.SWEEP_DEFINITION,
    )

    assert len(specs) == 16
    assert {spec.data_seed for spec in specs} == {sweep.COMMON_DATA_SEED}
    assert {spec.trainer_seed for spec in specs} == {sweep.TRAINER_SEED}
    assert all(spec.phase_weights["phase_0"] == spec.phase_weights["phase_1"] for spec in specs)
    assert all(spec.run_name.startswith("onephase_fullcanonical_dsp_") for spec in specs)
