# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from scripts.ci.claude_runner import validate_agent_policy


@pytest.mark.parametrize(
    "args, message",
    [
        (["--effort=low"], "full model identifier"),
        (["--model=opus", "--effort=high"], "full model identifier"),
        (["--model=claude-opus-4-8"], "effort tier"),
    ],
)
def test_validate_agent_policy_rejects_inherited_or_alias_defaults(args, message):
    with pytest.raises(ValueError, match=message):
        validate_agent_policy(args)


def test_validate_agent_policy_accepts_explicit_model_and_effort():
    validate_agent_policy(["--model=claude-opus-4-8", "--effort=high"])
