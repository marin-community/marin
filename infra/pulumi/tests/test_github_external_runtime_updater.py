# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pulumi
import pytest
from iac.github import external_runtime_updater
from scripts.ci.external_runtime_update import REQUIRED_CHECKS


def test_pulumi_and_merge_gate_require_the_same_checks() -> None:
    assert external_runtime_updater.REQUIRED_MAIN_CHECKS == REQUIRED_CHECKS


def test_registers_dedicated_app_credentials_and_layered_main_rulesets(monkeypatch) -> None:
    calls: list[tuple[str, str, dict]] = []

    def resource_type(kind: str):
        def register(resource_name: str, **kwargs):
            calls.append((kind, resource_name, kwargs))
            return object.__new__(pulumi.CustomResource)

        return register

    monkeypatch.setattr(
        external_runtime_updater.github,
        "AppInstallationRepository",
        resource_type("installation"),
    )
    monkeypatch.setattr(
        external_runtime_updater.github,
        "ActionsVariable",
        resource_type("variable"),
    )
    monkeypatch.setattr(
        external_runtime_updater.github,
        "ActionsEnvironmentSecret",
        resource_type("secret"),
    )
    monkeypatch.setattr(
        external_runtime_updater.github,
        "RepositoryRuleset",
        resource_type("ruleset"),
    )
    config = external_runtime_updater.ExternalRuntimeUpdaterConfig(
        organization="marin-community",
        repository="marin-community/marin",
        app_id=1234,
        app_slug="marin-external-runtime-updater",
        installation_id=5678,
        actions_key_id="test-actions-key-id",
        encrypted_private_key="test-encrypted-private-key",
        review_ruleset_id=785435,
    )

    environment = object.__new__(pulumi.CustomResource)
    resources = external_runtime_updater.register_external_runtime_updater(config, environment)

    assert len(resources) == 6
    installation = next(kwargs for kind, _, kwargs in calls if kind == "installation")
    assert installation["installation_id"] == "5678"
    assert installation["repository"] == "marin"
    assert installation["opts"].import_ == "5678:marin"

    variables = {kwargs["variable_name"]: kwargs for kind, _, kwargs in calls if kind == "variable"}
    assert variables["EXTERNAL_RUNTIME_UPDATER_APP_ID"]["value"] == "1234"
    assert variables["EXTERNAL_RUNTIME_UPDATER_APP_SLUG"]["value"] == "marin-external-runtime-updater"

    secret = next(kwargs for kind, _, kwargs in calls if kind == "secret")
    assert {key: value for key, value in secret.items() if key != "opts"} == {
        "repository": "marin",
        "environment": "external-runtime-updater",
        "secret_name": "EXTERNAL_RUNTIME_UPDATER_PRIVATE_KEY",
        "key_id": "test-actions-key-id",
        "value_encrypted": "test-encrypted-private-key",
    }
    assert secret["opts"].depends_on == [environment]

    rulesets = {kwargs["name"]: kwargs for kind, _, kwargs in calls if kind == "ruleset"}
    review = rulesets["protect main"]
    assert review["opts"].import_ == "marin:785435"
    assert len(review["bypass_actors"]) == 1
    bypass = review["bypass_actors"][0]
    assert bypass.actor_id == 1234
    assert bypass.actor_type == "Integration"
    assert bypass.bypass_mode == "pull_request"
    assert review["rules"].pull_request.required_approving_review_count == 1

    required_ci = rulesets["require main CI"]
    assert "bypass_actors" not in required_ci
    checks = required_ci["rules"].required_status_checks.required_checks
    assert tuple(check.context for check in checks) == external_runtime_updater.REQUIRED_MAIN_CHECKS
    assert {check.integration_id for check in checks} == {external_runtime_updater.GITHUB_ACTIONS_APP_ID}


def test_rejects_an_app_installation_for_another_organization() -> None:
    config = external_runtime_updater.ExternalRuntimeUpdaterConfig(
        organization="marin-community",
        repository="elsewhere/marin",
        app_id=1234,
        app_slug="marin-external-runtime-updater",
        installation_id=5678,
        actions_key_id="test-actions-key-id",
        encrypted_private_key="test-encrypted-private-key",
        review_ruleset_id=785435,
    )

    with pytest.raises(ValueError, match="not owned by"):
        external_runtime_updater.register_external_runtime_updater(config, object.__new__(pulumi.CustomResource))


def test_stack_config_allows_an_explicit_disabled_bootstrap_state() -> None:
    assert (
        external_runtime_updater.external_runtime_updater_config(
            organization="marin-community",
            settings={"enabled": False},
        )
        is None
    )


def test_enabled_stack_config_requires_the_environment_sealed_private_key() -> None:
    with pytest.raises(ValueError, match="exactly"):
        external_runtime_updater.external_runtime_updater_config(
            organization="marin-community",
            settings={
                "enabled": True,
                "repository": "marin-community/marin",
                "appId": 1234,
                "appSlug": "marin-external-runtime-updater",
                "installationId": 5678,
                "reviewRulesetId": 785435,
            },
        )


def test_registers_a_protected_main_only_actions_environment(monkeypatch) -> None:
    calls: list[dict] = []

    def register(_resource_name: str, **kwargs):
        calls.append(kwargs)
        return object.__new__(pulumi.CustomResource)

    monkeypatch.setattr(external_runtime_updater.github, "RepositoryEnvironment", register)

    external_runtime_updater.register_external_runtime_updater_environment(
        "marin-community", "marin-community/marin"
    )

    assert len(calls) == 1
    environment = calls[0]
    assert environment["repository"] == "marin"
    assert environment["environment"] == "external-runtime-updater"
    assert environment["can_admins_bypass"] is False
    assert environment["deployment_branch_policy"].protected_branches is True
    assert environment["deployment_branch_policy"].custom_branch_policies is False
