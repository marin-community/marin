# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from iac.github.dependency_updater import (
    DependencyUpdaterConfig,
    DependencyUpdaterInstallation,
    DependencyUpdaterRepositoryConfig,
    RequiredChecksPolicy,
    RulesetBypassActorPlan,
    SealedPrivateKeyConfig,
    dependency_updater_config,
    dependency_updater_installation_id,
    dependency_updater_plan,
    validate_classic_branch_protection,
)

from scripts.ci.dependency_update_policy import GITHUB_ACTIONS_APP_ID


def _repository(**overrides) -> DependencyUpdaterRepositoryConfig:
    values = {
        "repository": "marin-community/marin",
        "review_ruleset_id": 785435,
        "classic_required_checks": ("marin-lint", "unit-tests"),
        "required_checks_policy": RequiredChecksPolicy.MARIN,
        "private_key": SealedPrivateKeyConfig(
            actions_key_id="test-actions-key-id",
            encrypted_private_key="test-encrypted-private-key",
        ),
    }
    values.update(overrides)
    return DependencyUpdaterRepositoryConfig(**values)


def _config(*repositories: DependencyUpdaterRepositoryConfig) -> DependencyUpdaterConfig:
    return DependencyUpdaterConfig(
        organization="marin-community",
        app_id=1234,
        client_id="Iv23test-client-id",
        app_slug="marin-external-runtime-updater",
        repositories=repositories or (_repository(),),
    )


def _installation(**overrides) -> DependencyUpdaterInstallation:
    values = {
        "installation_id": 5678,
        "app_id": 1234,
        "client_id": "Iv23test-client-id",
        "app_slug": "marin-external-runtime-updater",
        "permissions": {"contents": "write", "pull_requests": "write", "workflows": "write"},
        "repository_selection": "selected",
        "suspended": False,
    }
    values.update(overrides)
    return DependencyUpdaterInstallation(**values)


def test_plan_separates_review_bypass_from_required_ci() -> None:
    plan = dependency_updater_plan(_config(), _repository())

    admin = RulesetBypassActorPlan(actor_type="OrganizationAdmin", bypass_mode="always")
    updater = RulesetBypassActorPlan(actor_type="Integration", bypass_mode="pull_request", actor_id=1234)
    assert plan.review_bypass_actors == (admin, updater)
    assert plan.required_ci_bypass_actors == (admin,)
    assert plan.review_ruleset_import == "marin:785435"
    assert plan.classic_branch_protection_import == "marin:main"
    assert plan.classic_review_bypass_apps == ("marin-external-runtime-updater",)
    assert {check.integration_id for check in plan.required_checks} == {GITHUB_ACTIONS_APP_ID}
    assert {check.integration_id for check in plan.classic_required_checks} == {GITHUB_ACTIONS_APP_ID}


def test_plan_creates_review_policy_without_inventing_ci_or_classic_protection() -> None:
    repository = _repository(
        repository="marin-community/axolotl",
        review_ruleset_id=None,
        classic_required_checks=None,
        required_checks_policy=None,
        private_key=None,
    )

    plan = dependency_updater_plan(_config(repository), repository)

    assert plan.repository == "axolotl"
    assert plan.review_ruleset_import is None
    assert plan.classic_branch_protection_import is None
    assert plan.required_checks == ()


def test_installation_requires_selected_scope_and_workflow_write_access() -> None:
    config = _config()
    assert dependency_updater_installation_id(config, [_installation()]) == 5678

    permissions = dict(_installation().permissions)
    permissions.pop("workflows")
    with pytest.raises(ValueError):
        dependency_updater_installation_id(config, [_installation(permissions=permissions)])
    with pytest.raises(ValueError):
        dependency_updater_installation_id(config, [_installation(repository_selection="all")])


def test_classic_protection_must_be_adopted_before_updater_registration() -> None:
    unmanaged = replace(_repository(), classic_required_checks=None)
    with pytest.raises(ValueError):
        validate_classic_branch_protection(unmanaged, ["main"])

    managed = _repository()
    with pytest.raises(ValueError):
        validate_classic_branch_protection(managed, [])
    with pytest.raises(ValueError):
        validate_classic_branch_protection(managed, ["*"])


def test_config_rejects_incomplete_environment_ciphertext() -> None:
    with pytest.raises(ValueError):
        dependency_updater_config(
            organization="marin-community",
            settings={
                "appId": 1234,
                "clientId": "Iv23test-client-id",
                "appSlug": "marin-external-runtime-updater",
                "repositories": [
                    {
                        "repository": "marin-community/marin",
                        "privateKey": {"actionsKeyId": "test-actions-key-id"},
                    }
                ],
            },
        )
