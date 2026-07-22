# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "preflight.py"
SPEC = importlib.util.spec_from_file_location("loom_preflight", MODULE_PATH)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def app() -> dict:
    return {"owner": {"login": "marin-community"}}


def installations() -> dict:
    return {
        "installations": [
            {
                "app_slug": "loom-oa-dev",
                "target_type": "Organization",
                "repository_selection": "all",
                "permissions": {
                    "contents": "write",
                    "issues": "write",
                    "metadata": "read",
                    "pull_requests": "write",
                },
                "events": ["issue_comment", "pull_request_review_comment"],
            }
        ]
    }


def test_github_app_preflight_accepts_the_expected_installation() -> None:
    installation = preflight.validate_github_app(app(), installations(), "marin-community", "loom-oa-dev")
    assert installation["app_slug"] == "loom-oa-dev"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda app, _install: app["owner"].update(login="someone-else"), "owned"),
        (
            lambda _app, install: install["installations"][0].update(repository_selection="selected"),
            "cover all",
        ),
        (
            lambda _app, install: install["installations"][0]["permissions"].pop("issues"),
            "permissions",
        ),
    ],
)
def test_github_app_preflight_rejects_drift(mutation, message: str) -> None:
    current_app = app()
    current_installations = installations()
    mutation(current_app, current_installations)
    with pytest.raises(ValueError, match=message):
        preflight.validate_github_app(
            current_app,
            current_installations,
            "marin-community",
            "loom-oa-dev",
        )


def test_buildx_preflight_requires_linux_amd64() -> None:
    with patch.object(
        preflight.subprocess,
        "run",
        return_value=subprocess.CompletedProcess([], 0, "Name: default\nPlatforms: linux/amd64, linux/arm64\n", ""),
    ):
        preflight.validate_buildx(None)

    with (
        patch.object(
            preflight.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, "Platforms: linux/arm64\n", ""),
        ),
        pytest.raises(ValueError, match="does not support"),
    ):
        preflight.validate_buildx(None)


def test_registry_preflight_accepts_inline_registry_auth() -> None:
    config = {"auths": {"us-central1-docker.pkg.dev": {"auth": "opaque"}}}
    with patch.object(preflight.subprocess, "run") as run:
        preflight.validate_registry_credentials(config, "us-central1-docker.pkg.dev")
    run.assert_not_called()


@pytest.mark.parametrize(
    ("config", "helper"),
    [
        ({"credHelpers": {"us-central1-docker.pkg.dev": "gcloud"}}, "gcloud"),
        ({"credsStore": "desktop"}, "desktop"),
    ],
)
def test_registry_preflight_invokes_the_selected_credential_helper(config: dict, helper: str) -> None:
    with patch.object(
        preflight.subprocess,
        "run",
        return_value=subprocess.CompletedProcess([], 0, '{"Username":"token","Secret":"redacted"}', ""),
    ) as run:
        preflight.validate_registry_credentials(config, "us-central1-docker.pkg.dev")
    assert run.call_args.args[0] == [f"docker-credential-{helper}", "get"]
    assert run.call_args.kwargs["input"] == "us-central1-docker.pkg.dev\n"


def test_registry_preflight_rejects_missing_credentials() -> None:
    with pytest.raises(ValueError, match="gcloud auth configure-docker"):
        preflight.validate_registry_credentials({}, "us-central1-docker.pkg.dev")


def test_stack_build_config_reads_the_selected_stack() -> None:
    completed = subprocess.CompletedProcess(
        [],
        0,
        (
            '{"marin-loom:buildxBuilder":{"value":"remote-amd64","secret":false},'
            '"marin-loom:region":{"value":"us-east1","secret":false}}'
        ),
        "",
    )
    with patch.object(preflight.subprocess, "run", return_value=completed):
        assert preflight.stack_build_config("marin-loom") == (
            "remote-amd64",
            "us-east1-docker.pkg.dev",
        )
