# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.capability_driven_envs.biotasks_library.audit_openrouter_first_wave import (
    _candidate_values,
    _same_json_value,
)


def test_offline_candidate_scan_does_not_change_frozen_submission_contract() -> None:
    text = 'The answer is:\n```json\n{"formula": "H2O", "mass": 18.0}\n```'
    assert _candidate_values(text) == [{"formula": "H2O", "mass": 18.0}]
    assert _same_json_value(_candidate_values(text)[0], {"formula": "H2O", "mass": 18})


def test_candidate_equality_never_interprets_boolean_as_a_number() -> None:
    assert not _same_json_value({"valid": 1}, {"valid": True})
    assert not _same_json_value({"samples": [{"valid": False}]}, {"samples": [{"valid": 0}]})
    assert not _same_json_value({"score": 1, "extra": 2}, {"score": 1})
