# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``policy.json``: every bound stated, nothing defaulted, and the digest a relaunch is checked against."""

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from taskforge.loop.policy import POLICY


def test_policy_json_round_trips_to_the_same_digest(programs):
    policy = POLICY.validate_json(POLICY.dump_json(programs.policy()))

    assert POLICY.validate_json(POLICY.dump_json(policy)).digest == policy.digest
    assert replace(policy, max_repairs=policy.max_repairs + 1).digest != policy.digest


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (lambda p: p.pop("max_repairs"), "missing fields \\['max_repairs'\\]"),
        (lambda p: p["validation"].pop("k"), "missing fields \\['k'\\]"),
        (lambda p: p["retry_backoff"].pop("jitter"), "jitter"),
        (lambda p: p.update(max_retries=3), "unknown fields \\['max_retries'\\]"),
    ],
)
def test_policy_json_states_every_field_and_no_other(programs, edit, problem):
    document = json.loads(POLICY.dump_json(programs.policy()))
    edit(document)

    with pytest.raises(ValidationError, match=problem):
        POLICY.validate_python(document)
