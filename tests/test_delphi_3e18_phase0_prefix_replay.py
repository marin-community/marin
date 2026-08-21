# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_replay as replay


def test_phase_0_boundary_matches_original_delphi_horizon():
    prefix_steps, hf_step = replay.phase_0_boundary(
        replay.EXPECTED_FULL_TRAIN_STEPS,
        replay.base.TARGET_BATCH_SIZE,
    )

    assert prefix_steps == replay.EXPECTED_PREFIX_TRAIN_STEPS
    assert hf_step == replay.EXPECTED_PREFIX_HF_STEP
    assert prefix_steps * replay.base.TARGET_BATCH_SIZE * replay.base.SEQ_LEN_DELPHI == (
        replay.EXPECTED_PREFIX_TRAIN_TOKENS
    )
