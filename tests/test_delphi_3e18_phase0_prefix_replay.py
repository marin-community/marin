# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_replay as replay


def test_phase_0_boundary_matches_original_delphi_horizon() -> None:
    prefix_steps, hf_step = replay.phase_0_boundary(
        replay.EXPECTED_FULL_TRAIN_STEPS,
        replay.base.TARGET_BATCH_SIZE,
    )

    assert prefix_steps == replay.EXPECTED_PREFIX_TRAIN_STEPS
    assert hf_step == replay.EXPECTED_PREFIX_HF_STEP
    assert prefix_steps * replay.base.TARGET_BATCH_SIZE * replay.base.SEQ_LEN_DELPHI == (
        replay.EXPECTED_PREFIX_TRAIN_TOKENS
    )


def test_replay_commit_accepts_explicit_sha_without_git_metadata() -> None:
    requested_commit = "a" * 40

    assert replay.validate_replay_code_commit(requested_commit, None) == requested_commit


def test_replay_commit_rejects_local_head_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match the local workspace HEAD"):
        replay.validate_replay_code_commit("a" * 40, "b" * 40)


@pytest.mark.parametrize("requested_commit", ["", "a" * 39, "A" * 40, "g" * 40])
def test_replay_commit_requires_full_lowercase_sha(requested_commit: str) -> None:
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        replay.validate_replay_code_commit(requested_commit, None)
