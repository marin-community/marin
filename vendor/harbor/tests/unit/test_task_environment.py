# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task capabilities are separate from the harness's storage credentials."""

import pytest

from harbor.utils.env import (
    resolve_env_vars,
    task_environment,
    validate_task_environment,
)


def test_task_alias_cannot_resolve_ambient_storage_credentials(monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "storage-private-canary")
    monkeypatch.setenv("INNOCENT_ALIAS", "storage-private-canary")
    for source in ("AWS_SECRET_ACCESS_KEY", "INNOCENT_ALIAS"):
        with pytest.raises(ValueError):
            resolve_env_vars({"TASK_VALUE": "${" + source + "}"})
    with pytest.raises(ValueError):
        validate_task_environment({"INNOCENT_ALIAS": "storage-private-canary"})


@pytest.mark.parametrize(
    "key", ["HUGGINGFACE_API_KEY", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"]
)
def test_huggingface_host_aliases_cannot_be_forwarded(monkeypatch, key):
    monkeypatch.setenv(key, "host-huggingface-token")
    with pytest.raises(ValueError):
        validate_task_environment({key: "host-huggingface-token"})
    with pytest.raises(ValueError):
        validate_task_environment({"RENAMED_TOKEN": "host-huggingface-token"})


def test_task_model_capabilities_resolve_without_exposing_parent_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "real-provider-secret")
    with task_environment(
        {"OPENAI_API_KEY": "EMPTY", "OPENAI_BASE_URL": "http://127.0.0.1:8787/v1"}
    ):
        resolved = resolve_env_vars(
            {
                "MODEL_TOKEN": "${OPENAI_API_KEY}",
                "MODEL_ENDPOINT": "${OPENAI_BASE_URL}",
                "MODE": "${OPTION:-local}",
            }
        )
    assert resolved == {
        "MODEL_TOKEN": "EMPTY",
        "MODEL_ENDPOINT": "http://127.0.0.1:8787/v1",
        "MODE": "local",
    }
    with pytest.raises(ValueError):
        resolve_env_vars({"MODEL_TOKEN": "${OPENAI_API_KEY}"})


@pytest.mark.parametrize(
    "key",
    [
        "AWS_ACCESS_KEY_ID",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "CW_KEY_SECRET",
        "IRIS_TASK_TOKEN",
    ],
)
def test_storage_and_control_plane_credentials_cannot_be_explicit_task_capabilities(
    key,
):
    with pytest.raises(ValueError):
        with task_environment({key: "private"}):
            resolve_env_vars({key: "private"})
