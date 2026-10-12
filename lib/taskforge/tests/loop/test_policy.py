# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A run root's policy is written once, normalised, and a relaunch under another policy is refused."""

from dataclasses import replace

import pytest

from taskforge.llm.policy import LLMPolicy
from taskforge.loop.policy import POLICY_FILE, LoopPolicy, install_policy
from taskforge.review.rules import BandChoice, BandRules
from taskforge.validate.calibration import CalibrationBand
from taskforge.validate.run import ValidationPolicy
from taskforge.validate.trials import RetryBackoff, SessionLimits


def code_built_policy() -> LoopPolicy:
    """A policy as a driver writes it: whole numbers in the float fields."""
    backoff = RetryBackoff(initial=1, maximum=4, factor=2, jitter=0)
    return LoopPolicy(
        proposals_per_idea=1,
        max_idea_reproposals=0,
        max_build_revisions=1,
        max_validation_retries=1,
        retry_backoff=backoff,
        band_rules=BandRules(too_easy=BandChoice.ACCEPT, too_hard=BandChoice.REJECT),
        validation=ValidationPolicy(
            k=4,
            band=CalibrationBand(min_solve_rate=0, max_solve_rate=1),
            sampling=LLMPolicy(max_continuations=0),
            limits=SessionLimits(
                max_turns=4,
                model_turn_timeout=20,
                command_timeout=10,
                tool_turn_timeout=20,
                total_turn_timeout=30,
                attempt_timeout=60,
                cleanup_timeout=10,
            ),
            max_retries=0,
            token_contract_retries=0,
            retry_backoff=backoff,
        ),
    )


def test_the_policy_is_written_once_and_a_code_built_relaunch_is_accepted(tmp_path):
    first = install_policy(tmp_path, code_built_policy())
    written = (tmp_path / POLICY_FILE).read_bytes()

    again = install_policy(tmp_path, code_built_policy())

    assert again.digest == first.digest == code_built_policy().digest
    assert (tmp_path / POLICY_FILE).read_bytes() == written
    assert isinstance(first.validation.band.max_solve_rate, float)


def more_session_turns(policy: LoopPolicy) -> LoopPolicy:
    validation = policy.validation
    return replace(policy, validation=replace(validation, limits=replace(validation.limits, max_turns=8)))


@pytest.mark.parametrize(
    "change",
    [lambda policy: replace(policy, max_build_revisions=2), more_session_turns],
    ids=["loop bound", "session limit"],
)
def test_a_relaunch_under_another_policy_is_refused(tmp_path, change):
    install_policy(tmp_path, code_built_policy())

    with pytest.raises(ValueError, match="use a new run root"):
        install_policy(tmp_path, change(code_built_policy()))
