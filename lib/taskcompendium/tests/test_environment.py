# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Environment values passed to task and verifier machines."""

import pytest

from taskcompendium.runtime.environment import resolve_env_vars


def test_task_environment_resolves_host_values_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASK_TOKEN", "host-token")
    monkeypatch.delenv("TASK_REGION", raising=False)

    assert resolve_env_vars(
        {
            "TOKEN": "${TASK_TOKEN}",
            "REGION": "${TASK_REGION:-us-east-1}",
            "LITERAL": "prefix-${TASK_TOKEN}",
        }
    ) == {"TOKEN": "host-token", "REGION": "us-east-1", "LITERAL": "prefix-${TASK_TOKEN}"}

    with pytest.raises(ValueError, match="TASK_REGION"):
        resolve_env_vars({"REGION": "${TASK_REGION}"})
