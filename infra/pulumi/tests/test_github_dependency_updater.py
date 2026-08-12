# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iac.github.dependency_updater import (
    DependencyUpdaterConfig,
    RulesetBypassActorPlan,
    dependency_updater_config,
    dependency_updater_plan,
)

from scripts.ci.dependency_update_policy import GITHUB_ACTIONS_APP_ID, REQUIRED_CHECKS


def _config(**overrides) -> DependencyUpdaterConfig:
    values = {
        "organization": "marin-community",
        "repository": "marin-community/marin",
        "app_id": 1234,
        "app_slug": "marin-external-runtime-updater",
        "actions_key_id": "test-actions-key-id",
        "encrypted_private_key": "test-encrypted-private-key",
        "review_ruleset_id": 785435,
    }
    values.update(overrides)
    return DependencyUpdaterConfig(**values)


def test_plan_preserves_admin_override_and_limits_the_updater_to_review_bypass() -> None:
    plan = dependency_updater_plan(_config())

    admin = RulesetBypassActorPlan(actor_type="OrganizationAdmin", bypass_mode="always")
    updater = RulesetBypassActorPlan(actor_type="Integration", bypass_mode="pull_request", actor_id=1234)
    assert plan.review_bypass_actors == (admin, updater)
    assert plan.required_ci_bypass_actors == (admin,)
    assert plan.review_ruleset_import == "marin:785435"


def test_plan_binds_required_checks_to_github_actions() -> None:
    plan = dependency_updater_plan(_config())

    assert tuple(check.context for check in plan.required_checks) == REQUIRED_CHECKS
    assert {check.integration_id for check in plan.required_checks} == {GITHUB_ACTIONS_APP_ID}


def test_plan_targets_marins_dependency_update_environment() -> None:
    plan = dependency_updater_plan(_config())

    assert plan.repository == "marin"
    assert plan.environment == "external-runtime-updater"


def test_plan_rejects_an_app_installation_for_another_organization() -> None:
    with pytest.raises(ValueError):
        dependency_updater_plan(_config(repository="elsewhere/marin"))


def test_active_stack_config_requires_complete_sealed_key_metadata() -> None:
    with pytest.raises(ValueError):
        dependency_updater_config(
            organization="marin-community",
            settings={
                "repository": "marin-community/marin",
                "appId": 1234,
                "appSlug": "marin-external-runtime-updater",
                "reviewRulesetId": 785435,
            },
        )
