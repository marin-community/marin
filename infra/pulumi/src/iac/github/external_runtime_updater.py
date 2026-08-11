# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GitHub App credentials and repository rules for external runtime updates."""

import re
from dataclasses import dataclass

import pulumi
import pulumi_github as github

REQUIRED_MAIN_CHECKS = ("marin-integration", "marin-lint", "rust-checks", "unit-tests")
GITHUB_ACTIONS_APP_ID = 15368
UPDATER_ENVIRONMENT = "external-runtime-updater"
APP_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
DEFAULT_BRANCH_CONDITIONS = github.RepositoryRulesetConditionsArgs(
    ref_name=github.RepositoryRulesetConditionsRefNameArgs(
        excludes=[],
        includes=["~DEFAULT_BRANCH", "refs/heads/main"],
    )
)


@dataclass(frozen=True)
class ExternalRuntimeUpdaterConfig:
    """Configuration produced by the dedicated GitHub App bootstrap."""

    organization: str
    repository: str
    app_id: int
    app_slug: str
    installation_id: int
    actions_key_id: str
    encrypted_private_key: str
    review_ruleset_id: int


def external_runtime_updater_config(
    *,
    organization: str,
    settings: dict[str, object],
) -> ExternalRuntimeUpdaterConfig | None:
    """Validate stack settings, returning ``None`` for the bootstrap state."""
    enabled = settings.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("externalRuntimeUpdater.enabled must be a boolean")
    if not enabled:
        return None
    expected_keys = {
        "actionsKeyId",
        "appId",
        "appSlug",
        "enabled",
        "encryptedPrivateKey",
        "installationId",
        "repository",
        "reviewRulesetId",
    }
    if set(settings) != expected_keys:
        raise ValueError(
            "enabled externalRuntimeUpdater settings require exactly "
            f"{sorted(expected_keys)!r}; found {sorted(settings)!r}"
        )

    def positive_int(name: str) -> int:
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"externalRuntimeUpdater.{name} must be a positive integer")
        return value

    repository = settings["repository"]
    if not isinstance(repository, str):
        raise ValueError("externalRuntimeUpdater.repository must be a string")
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
        app_id=positive_int("appId"),
        app_slug=app_slug,
        installation_id=positive_int("installationId"),
        actions_key_id=actions_key_id,
        encrypted_private_key=encrypted_private_key,
        review_ruleset_id=positive_int("reviewRulesetId"),
    )


def _repository_name(organization: str, repository: str) -> str:
    prefix = f"{organization}/"
    if not repository.startswith(prefix):
        raise ValueError(f"repository {repository!r} is not owned by {organization!r}")
    return repository.removeprefix(prefix)


def register_external_runtime_updater(
    config: ExternalRuntimeUpdaterConfig,
    environment: pulumi.CustomResource,
) -> tuple[pulumi.CustomResource, ...]:
    """Register the app installation, credentials, and layered main-branch rules."""
    repository = _repository_name(config.organization, config.repository)
    installation = github.AppInstallationRepository(
        "external-runtime-updater-installation",
        installation_id=str(config.installation_id),
        repository=repository,
        opts=pulumi.ResourceOptions(import_=f"{config.installation_id}:{repository}"),
    )
    app_id = github.ActionsVariable(
        "external-runtime-updater-app-id",
        repository=repository,
        variable_name="EXTERNAL_RUNTIME_UPDATER_APP_ID",
        value=str(config.app_id),
    )
    app_slug = github.ActionsVariable(
        "external-runtime-updater-app-slug",
        repository=repository,
        variable_name="EXTERNAL_RUNTIME_UPDATER_APP_SLUG",
        value=config.app_slug,
    )
    private_key = github.ActionsEnvironmentSecret(
        "external-runtime-updater-private-key",
        repository=repository,
        environment=UPDATER_ENVIRONMENT,
        secret_name="EXTERNAL_RUNTIME_UPDATER_PRIVATE_KEY",
        key_id=config.actions_key_id,
        value_encrypted=config.encrypted_private_key,
        opts=pulumi.ResourceOptions(depends_on=[environment]),
    )
    review_ruleset = github.RepositoryRuleset(
        "protect-main",
        repository=repository,
        name="protect main",
        target="branch",
        enforcement="active",
        conditions=DEFAULT_BRANCH_CONDITIONS,
        bypass_actors=[
            github.RepositoryRulesetBypassActorArgs(
                actor_id=config.app_id,
                actor_type="Integration",
                bypass_mode="pull_request",
            )
        ],
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
            depends_on=[installation],
            import_=f"{repository}:{config.review_ruleset_id}",
        ),
    )
    required_ci_ruleset = github.RepositoryRuleset(
        "require-main-ci",
        repository=repository,
        name="require main CI",
        target="branch",
        enforcement="active",
        conditions=DEFAULT_BRANCH_CONDITIONS,
        rules=github.RepositoryRulesetRulesArgs(
            required_status_checks=github.RepositoryRulesetRulesRequiredStatusChecksArgs(
                do_not_enforce_on_create=False,
                required_checks=[
                    github.RepositoryRulesetRulesRequiredStatusChecksRequiredCheckArgs(
                        context=context,
                        integration_id=GITHUB_ACTIONS_APP_ID,
                    )
                    for context in REQUIRED_MAIN_CHECKS
                ],
                strict_required_status_checks_policy=False,
            )
        ),
    )
    return installation, app_id, app_slug, private_key, review_ruleset, required_ci_ruleset


def register_external_runtime_updater_environment(organization: str, repository: str) -> pulumi.CustomResource:
    """Create the protected environment that releases the app key only on main."""
    repository_name = _repository_name(organization, repository)
    return github.RepositoryEnvironment(
        "external-runtime-updater-environment",
        repository=repository_name,
        environment=UPDATER_ENVIRONMENT,
        can_admins_bypass=False,
        deployment_branch_policy=github.RepositoryEnvironmentDeploymentBranchPolicyArgs(
            custom_branch_policies=False,
            protected_branches=True,
        ),
    )
