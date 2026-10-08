# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``policy.json``: every bound stated, nothing defaulted, and the digest a relaunch is checked against."""

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from taskforge.loop.policy import POLICY
from taskforge.review.rules import BandChoice, BandRule, BandRules


def test_policy_json_round_trips_to_the_same_digest(programs):
    policy = POLICY.validate_json(POLICY.dump_json(programs.policy()))

    assert POLICY.validate_json(POLICY.dump_json(policy)).digest == policy.digest
    assert replace(policy, max_repairs=policy.max_repairs + 1).digest != policy.digest
    accepting = BandRules(BandRule(1, BandChoice.ACCEPT), policy.band_rules.too_hard)
    assert replace(policy, band_rules=accepting).digest != policy.digest
    assert json.loads(POLICY.dump_json(policy))["band_rules"]["too_easy"] == {"repairs": 1, "then": "reject"}


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (lambda p: p.pop("max_repairs"), "missing fields \\['max_repairs'\\]"),
        (lambda p: p["validation"].pop("k"), "missing fields \\['k'\\]"),
        (lambda p: p["retry_backoff"].pop("jitter"), "jitter"),
        (lambda p: p["validation"]["sampling"].pop("temperature"), "missing fields \\['temperature'\\]"),
        (lambda p: p["validation"]["sampling"].update(temprature=0.1), "unknown fields \\['temprature'\\]"),
        (lambda p: p["validation"]["deadlines"].pop("total_turn_timeout"), "missing fields \\['total_turn_timeout'\\]"),
        (lambda p: p["validation"]["band"].update(min=0.1), "unknown fields \\['min'\\]"),
        (lambda p: p.update(max_retries=3), "unknown fields \\['max_retries'\\]"),
        (lambda p: p.pop("band_rules"), "missing fields \\['band_rules'\\]"),
        (lambda p: p["band_rules"].pop("too_hard"), "missing fields \\['too_hard'\\]"),
        (lambda p: p["band_rules"]["too_easy"].update(then="note"), "accept"),
        (lambda p: p["band_rules"]["too_easy"].update(notes=1), "unknown fields \\['notes'\\]"),
        (lambda p: p["validation"].pop("adversary_submissions"), "missing fields \\['adversary_submissions'\\]"),
    ],
)
def test_policy_json_states_every_field_and_no_other(programs, edit, problem):
    document = json.loads(POLICY.dump_json(programs.policy()))
    edit(document)

    with pytest.raises(ValidationError, match=problem):
        POLICY.validate_python(document)
