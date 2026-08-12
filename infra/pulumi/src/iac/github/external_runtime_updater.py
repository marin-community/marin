# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GitHub App credentials and repository rules for dependency updates."""

import re
from dataclasses import dataclass

import pulumi
import pulumi_github as github

from iac.github.resources import repository_name
from scripts.ci.dependency_update_policy import GITHUB_ACTIONS_APP_ID, REQUIRED_CHECKS

APP_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
UPDATER_ENVIRONMENT = "external-runtime-updater"
UPDATER_BRANCH = "main"


@dataclass(frozen=True)
class ExternalRuntimeUpdaterConfig:
    """Active configuration for the dedicated dependency updater app."""

    organization: str
    repository: str
    app_id: int
    app_slug: str
    actions_key_id: str
    encrypted_private_key: str
    review_ruleset_id: int


@dataclass(frozen=True)
class RequiredCheckPlan:
    context: str
    integration_id: int


@dataclass(frozen=True)
class RulesetBypassActorPlan:
    actor_type: str
    bypass_mode: str
    actor_id: int | None = None


@dataclass(frozen=True)
class ExternalRuntimeUpdaterPlan:
    repository: str
    environment: str
    review_ruleset_import: str
    review_bypass_actors: tuple[RulesetBypassActorPlan, ...]
    required_ci_bypass_actors: tuple[RulesetBypassActorPlan, ...]
    required_checks: tuple[RequiredCheckPlan, ...]


def _positive_int(settings: dict[str, object], name: str) -> int:
    value = settings[name]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"externalRuntimeUpdater.{name} must be a positive integer")
    return value


def external_runtime_updater_config(
    *,
    organization: str,
    settings: dict[str, object],
) -> ExternalRuntimeUpdaterConfig:
    """Validate the active updater app settings."""
    repository = settings.get("repository")
    if not isinstance(repository, str):
        raise ValueError("externalRuntimeUpdater.repository must be a string")
    repository_name(organization, repository)
    expected_keys = {
        "actionsKeyId",
        "appId",
        "appSlug",
        "encryptedPrivateKey",
        "repository",
        "reviewRulesetId",
    }
    if set(settings) != expected_keys:
        raise ValueError(
            "externalRuntimeUpdater settings require exactly " f"{sorted(expected_keys)!r}; found {sorted(settings)!r}"
        )

    app_slug = settings["appSlug"]
    if not isinstance(app_slug, str) or APP_SLUG.fullmatch(app_slug) is None:
        raise ValueError("externalRuntimeUpdater.appSlug must be a lowercase GitHub App slug")
    actions_key_id = settings["actionsKeyId"]
    if not isinstance(actions_key_id, str) or not actions_key_id:
        raise ValueError("externalRuntimeUpdater.actionsKeyId must be a non-empty string")
    encrypted_private_key = settings["encryptedPrivateKey"]
    if not isinstance(encrypted_private_key, str) or not encrypted_private_key:
        raise ValueError("externalRuntimeUpdater.encryptedPrivateKey must be a non-empty string")
    return ExternalRuntimeUpdaterConfig(
        organization=organization,
        repository=repository,
        app_id=_positive_int(settings, "appId"),
        app_slug=app_slug,
        actions_key_id=actions_key_id,
        encrypted_private_key=encrypted_private_key,
        review_ruleset_id=_positive_int(settings, "reviewRulesetId"),
    )


def external_runtime_updater_plan(config: ExternalRuntimeUpdaterConfig) -> ExternalRuntimeUpdaterPlan:
    """Compute the app's credentials and non-overlapping merge rules."""
    repository = repository_name(config.organization, config.repository)
    organization_admin = RulesetBypassActorPlan(
        actor_type="OrganizationAdmin",
        bypass_mode="always",
    )
    return ExternalRuntimeUpdaterPlan(
        repository=repository,
        environment=UPDATER_ENVIRONMENT,
        review_ruleset_import=f"{repository}:{config.review_ruleset_id}",
        review_bypass_actors=(
            organization_admin,
            RulesetBypassActorPlan(
                actor_id=config.app_id,
                actor_type="Integration",
                bypass_mode="pull_request",
            ),
        ),
        required_ci_bypass_actors=(organization_admin,),
        required_checks=tuple(
            RequiredCheckPlan(context=context, integration_id=GITHUB_ACTIONS_APP_ID) for context in REQUIRED_CHECKS
        ),
    )


def _default_branch_conditions() -> github.RepositoryRulesetConditionsArgs:
    return github.RepositoryRulesetConditionsArgs(
        ref_name=github.RepositoryRulesetConditionsRefNameArgs(
            excludes=[],
            includes=["~DEFAULT_BRANCH", "refs/heads/main"],
        )
    )


def _bypass_actor_args(actor: RulesetBypassActorPlan) -> github.RepositoryRulesetBypassActorArgs:
    return github.RepositoryRulesetBypassActorArgs(
        actor_id=actor.actor_id,
        actor_type=actor.actor_type,
        bypass_mode=actor.bypass_mode,
    )


def register_external_runtime_updater(
    config: ExternalRuntimeUpdaterConfig,
    deployment_policy: pulumi.CustomResource,
) -> tuple[pulumi.CustomResource, ...]:
    """Register the app credentials and layered main-branch rules."""
    plan = external_runtime_updater_plan(config)
    app_id = github.ActionsVariable(
        "external-runtime-updater-app-id",
        repository=plan.repository,
        variable_name="DEPENDENCY_UPDATER_APP_ID",
        value=str(config.app_id),
    )
    app_slug = github.ActionsVariable(
        "external-runtime-updater-app-slug",
        repository=plan.repository,
        variable_name="DEPENDENCY_UPDATER_APP_SLUG",
        value=config.app_slug,
    )
    private_key = github.ActionsEnvironmentSecret(
        "external-runtime-updater-private-key",
        repository=plan.repository,
        environment=plan.environment,
        secret_name="DEPENDENCY_UPDATER_PRIVATE_KEY",
        key_id=config.actions_key_id,
        value_encrypted=config.encrypted_private_key,
        opts=pulumi.ResourceOptions(depends_on=[deployment_policy]),
    )
    review_ruleset = github.RepositoryRuleset(
        "protect-main",
        repository=plan.repository,
        name="protect main",
        target="branch",
        enforcement="active",
        conditions=_default_branch_conditions(),
        bypass_actors=[_bypass_actor_args(actor) for actor in plan.review_bypass_actors],
        rules=github.RepositoryRulesetRulesArgs(
            deletion=True,
            non_fast_forward=True,
            pull_request=github.RepositoryRulesetRulesPullRequestArgs(
                allowed_merge_methods=["squash"],
                dismiss_stale_reviews_on_push=False,
                require_code_owner_review=False,
                require_last_push_approval=False,
                required_approving_review_count=1,
                required_review_thread_resolution=False,
            ),
        ),
        opts=pulumi.ResourceOptions(
            import_=plan.review_ruleset_import,
        ),
    )
    required_ci_ruleset = github.RepositoryRuleset(
        "require-main-ci",
        repository=plan.repository,
        name="require main CI",
        target="branch",
        enforcement="active",
        conditions=_default_branch_conditions(),
        bypass_actors=[_bypass_actor_args(actor) for actor in plan.required_ci_bypass_actors],
        rules=github.RepositoryRulesetRulesArgs(
            required_status_checks=github.RepositoryRulesetRulesRequiredStatusChecksArgs(
                do_not_enforce_on_create=False,
                required_checks=[
                    github.RepositoryRulesetRulesRequiredStatusChecksRequiredCheckArgs(
                        context=check.context,
                        integration_id=check.integration_id,
                    )
                    for check in plan.required_checks
                ],
                strict_required_status_checks_policy=False,
            )
        ),
    )
    return app_id, app_slug, private_key, review_ruleset, required_ci_ruleset


def register_external_runtime_updater_environment(
    organization: str,
    repository: str,
) -> tuple[pulumi.CustomResource, pulumi.CustomResource]:
    """Create the environment and restrict it to the main branch."""
    normalized_repository = repository_name(organization, repository)
    environment = github.RepositoryEnvironment(
        "external-runtime-updater-environment",
        repository=normalized_repository,
        environment=UPDATER_ENVIRONMENT,
        can_admins_bypass=False,
        deployment_branch_policy=github.RepositoryEnvironmentDeploymentBranchPolicyArgs(
            custom_branch_policies=True,
            protected_branches=False,
        ),
    )
    deployment_policy = github.RepositoryEnvironmentDeploymentPolicy(
        "external-runtime-updater-main-policy",
        repository=normalized_repository,
        environment=UPDATER_ENVIRONMENT,
        branch_pattern=UPDATER_BRANCH,
        opts=pulumi.ResourceOptions(depends_on=[environment]),
    )
    return environment, deployment_policy
