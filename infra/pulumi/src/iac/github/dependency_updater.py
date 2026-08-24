# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GitHub App credentials and repository rules for dependency updates."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase

import pulumi
import pulumi_github as github

from iac.github.resources import repository_name
from scripts.ci.dependency_update_policy import GITHUB_ACTIONS_APP_ID

APP_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
UPDATER_ENVIRONMENT = "external-runtime-updater"
UPDATER_BRANCH = "main"
REQUIRED_APP_PERMISSIONS = {
    "contents": "write",
    "pull_requests": "write",
    "workflows": "write",
}


@dataclass(frozen=True)
class SealedPrivateKeyConfig:
    """Private key ciphertext sealed to one Actions environment."""

    actions_key_id: str
    encrypted_private_key: str


@dataclass(frozen=True)
class DependencyUpdaterRepositoryConfig:
    """Repository-specific updater credentials and protection declarations."""

    repository: str
    review_ruleset_id: int | None
    classic_required_checks: tuple[str, ...] | None
    required_checks: tuple[str, ...]
    private_key: SealedPrivateKeyConfig | None


@dataclass(frozen=True)
class DependencyUpdaterConfig:
    """Active configuration for the dedicated dependency updater app."""

    organization: str
    app_id: int
    client_id: str
    app_slug: str
    repositories: tuple[DependencyUpdaterRepositoryConfig, ...]


@dataclass(frozen=True)
class DependencyUpdaterInstallation:
    """GitHub App installation fields needed to validate the updater."""

    installation_id: int
    app_id: int
    client_id: str
    app_slug: str
    permissions: Mapping[str, str]
    repository_selection: str
    suspended: bool


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
class DependencyUpdaterPlan:
    repository: str
    environment: str
    classic_branch_protection_import: str | None
    classic_review_bypass_apps: tuple[str, ...]
    classic_required_checks: tuple[RequiredCheckPlan, ...]
    review_ruleset_import: str | None
    review_bypass_actors: tuple[RulesetBypassActorPlan, ...]
    required_ci_bypass_actors: tuple[RulesetBypassActorPlan, ...]
    required_checks: tuple[RequiredCheckPlan, ...]


def _positive_int(settings: dict[str, object], name: str, path: str) -> int:
    value = settings[name]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path}.{name} must be a positive integer")
    return value


def _optional_positive_int(settings: dict[str, object], name: str, path: str) -> int | None:
    if name not in settings:
        return None
    return _positive_int(settings, name, path)


def _optional_string_tuple(settings: dict[str, object], name: str, path: str) -> tuple[str, ...] | None:
    if name not in settings:
        return None
    value = settings[name]
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{path}.{name} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{path}.{name} must not contain duplicate checks")
    return tuple(value)


def _repository_config(
    *,
    organization: str,
    settings: dict[str, object],
    index: int,
) -> DependencyUpdaterRepositoryConfig:
    path = f"dependencyUpdater.repositories[{index}]"
    expected_keys = {
        "classicRequiredChecks",
        "privateKey",
        "repository",
        "requiredChecks",
        "reviewRulesetId",
    }
    unexpected_keys = set(settings) - expected_keys
    if unexpected_keys:
        raise ValueError(f"{path} contains unexpected settings {sorted(unexpected_keys)!r}")

    repository = settings.get("repository")
    if not isinstance(repository, str):
        raise ValueError(f"{path}.repository must be a string")
    repository_name(organization, repository)
    private_key_settings = settings.get("privateKey")
    private_key = None
    if private_key_settings is not None:
        if not isinstance(private_key_settings, dict) or set(private_key_settings) != {
            "actionsKeyId",
            "encryptedPrivateKey",
        }:
            raise ValueError(f"{path}.privateKey must contain actionsKeyId and encryptedPrivateKey")
        actions_key_id = private_key_settings["actionsKeyId"]
        encrypted_private_key = private_key_settings["encryptedPrivateKey"]
        if not isinstance(actions_key_id, str) or not actions_key_id:
            raise ValueError(f"{path}.privateKey.actionsKeyId must be a non-empty string")
        if not isinstance(encrypted_private_key, str) or not encrypted_private_key:
            raise ValueError(f"{path}.privateKey.encryptedPrivateKey must be a non-empty string")
        private_key = SealedPrivateKeyConfig(
            actions_key_id=actions_key_id,
            encrypted_private_key=encrypted_private_key,
        )
    return DependencyUpdaterRepositoryConfig(
        repository=repository,
        review_ruleset_id=_optional_positive_int(settings, "reviewRulesetId", path),
        classic_required_checks=_optional_string_tuple(settings, "classicRequiredChecks", path),
        required_checks=_optional_string_tuple(settings, "requiredChecks", path) or (),
        private_key=private_key,
    )


def dependency_updater_config(
    *,
    organization: str,
    settings: dict[str, object],
) -> DependencyUpdaterConfig:
    """Validate updater app and repository settings."""
    expected_keys = {"appId", "clientId", "appSlug", "repositories"}
    if set(settings) != expected_keys:
        raise ValueError(
            "dependencyUpdater settings require exactly " f"{sorted(expected_keys)!r}; found {sorted(settings)!r}"
        )

    app_slug = settings["appSlug"]
    if not isinstance(app_slug, str) or APP_SLUG.fullmatch(app_slug) is None:
        raise ValueError("dependencyUpdater.appSlug must be a lowercase GitHub App slug")
    client_id = settings["clientId"]
    if not isinstance(client_id, str) or not client_id:
        raise ValueError("dependencyUpdater.clientId must be a non-empty string")
    repository_settings = settings["repositories"]
    if not isinstance(repository_settings, list) or not repository_settings:
        raise ValueError("dependencyUpdater.repositories must be a non-empty list")
    if not all(isinstance(repository, dict) for repository in repository_settings):
        raise ValueError("dependencyUpdater.repositories entries must be mappings")
    repositories = tuple(
        _repository_config(organization=organization, settings=repository, index=index)
        for index, repository in enumerate(repository_settings)
    )
    repository_keys = [repository.repository.casefold() for repository in repositories]
    if len(repository_keys) != len(set(repository_keys)):
        raise ValueError("dependencyUpdater.repositories must not contain duplicate repositories")
    return DependencyUpdaterConfig(
        organization=organization,
        app_id=_positive_int(settings, "appId", "dependencyUpdater"),
        client_id=client_id,
        app_slug=app_slug,
        repositories=repositories,
    )


def dependency_updater_installation_id(
    config: DependencyUpdaterConfig,
    installations: Sequence[DependencyUpdaterInstallation],
) -> int:
    """Return the validated organization installation for the updater app."""
    matching = [installation for installation in installations if installation.app_id == config.app_id]
    if len(matching) != 1:
        raise ValueError(f"expected one GitHub App installation for app ID {config.app_id}; found {len(matching)}")
    installation = matching[0]
    if installation.app_slug != config.app_slug or installation.client_id != config.client_id:
        raise ValueError("dependency updater installation identity does not match the configured app")
    if installation.suspended:
        raise ValueError("dependency updater installation is suspended")
    if installation.repository_selection != "selected":
        raise ValueError("dependency updater installation must use selected-repository access")
    missing_permissions = {
        permission: required
        for permission, required in REQUIRED_APP_PERMISSIONS.items()
        if installation.permissions.get(permission) != required
    }
    if missing_permissions:
        raise ValueError(f"dependency updater installation is missing write permissions: {sorted(missing_permissions)}")
    return installation.installation_id


def validate_classic_branch_protection(
    config: DependencyUpdaterRepositoryConfig,
    patterns: Sequence[str],
) -> None:
    """Reject main protection that the repository declaration does not adopt."""
    matching_patterns = tuple(sorted(pattern for pattern in patterns if fnmatchcase(UPDATER_BRANCH, pattern)))
    if config.classic_required_checks is None and matching_patterns:
        raise ValueError(
            f"{config.repository} has unmanaged classic protection matching {UPDATER_BRANCH!r}: {matching_patterns!r}"
        )
    if config.classic_required_checks is not None and matching_patterns != (UPDATER_BRANCH,):
        raise ValueError(
            f"{config.repository} must have one exact {UPDATER_BRANCH!r} classic protection before import; "
            f"found {matching_patterns!r}"
        )


def dependency_updater_plan(
    config: DependencyUpdaterConfig,
    repository_config: DependencyUpdaterRepositoryConfig,
) -> DependencyUpdaterPlan:
    """Compute one repository's credentials and non-overlapping merge rules."""
    repository = repository_name(config.organization, repository_config.repository)
    organization_admin = RulesetBypassActorPlan(
        actor_type="OrganizationAdmin",
        bypass_mode="always",
    )
    return DependencyUpdaterPlan(
        repository=repository,
        environment=UPDATER_ENVIRONMENT,
        classic_branch_protection_import=(
            f"{repository}:{UPDATER_BRANCH}" if repository_config.classic_required_checks is not None else None
        ),
        classic_review_bypass_apps=(config.app_slug,),
        classic_required_checks=tuple(
            RequiredCheckPlan(context=context, integration_id=GITHUB_ACTIONS_APP_ID)
            for context in repository_config.classic_required_checks or ()
        ),
        review_ruleset_import=(
            f"{repository}:{repository_config.review_ruleset_id}"
            if repository_config.review_ruleset_id is not None
            else None
        ),
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
            RequiredCheckPlan(context=context, integration_id=GITHUB_ACTIONS_APP_ID)
            for context in repository_config.required_checks
        ),
    )


def _resource_name(repository: str, name: str) -> str:
    if repository.casefold() == "marin":
        return name
    repository_prefix = re.sub(r"[^a-z0-9-]+", "-", repository.casefold()).strip("-")
    return f"{repository_prefix}-{name}"


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


def register_dependency_updater_installation(
    config: DependencyUpdaterConfig,
    installation_id: int,
) -> github.AppInstallationRepositories:
    """Restrict the updater app installation to declared repositories."""
    return github.AppInstallationRepositories(
        "external-runtime-updater-repositories",
        installation_id=str(installation_id),
        selected_repositories=[repository_name(config.organization, item.repository) for item in config.repositories],
        opts=pulumi.ResourceOptions(protect=True),
    )


def register_dependency_updater_repository(
    config: DependencyUpdaterConfig,
    repository_config: DependencyUpdaterRepositoryConfig,
    deployment_policy: pulumi.CustomResource,
    installation: github.AppInstallationRepositories,
) -> tuple[pulumi.CustomResource, ...]:
    """Register updater credentials and layered main-branch rules for one repository."""
    plan = dependency_updater_plan(config, repository_config)
    resources: list[pulumi.CustomResource] = [
        github.ActionsVariable(
            _resource_name(plan.repository, "external-runtime-updater-client-id"),
            repository=plan.repository,
            variable_name="DEPENDENCY_UPDATER_CLIENT_ID",
            value=config.client_id,
        ),
        github.ActionsVariable(
            _resource_name(plan.repository, "external-runtime-updater-app-slug"),
            repository=plan.repository,
            variable_name="DEPENDENCY_UPDATER_APP_SLUG",
            value=config.app_slug,
        ),
    ]
    if repository_config.private_key is not None:
        resources.append(
            github.ActionsEnvironmentSecret(
                _resource_name(plan.repository, "external-runtime-updater-private-key"),
                repository=plan.repository,
                environment=plan.environment,
                secret_name="DEPENDENCY_UPDATER_PRIVATE_KEY",
                key_id=repository_config.private_key.actions_key_id,
                value_encrypted=repository_config.private_key.encrypted_private_key,
                opts=pulumi.ResourceOptions(depends_on=[deployment_policy]),
            )
        )
    if plan.classic_branch_protection_import is not None:
        resources.append(
            github.BranchProtectionV3(
                _resource_name(plan.repository, "main-classic-protection"),
                repository=plan.repository,
                branch=UPDATER_BRANCH,
                enforce_admins=False,
                require_conversation_resolution=False,
                require_signed_commits=False,
                required_status_checks=github.BranchProtectionV3RequiredStatusChecksArgs(
                    checks=[f"{check.context}:{check.integration_id}" for check in plan.classic_required_checks],
                    strict=False,
                ),
                required_pull_request_reviews=github.BranchProtectionV3RequiredPullRequestReviewsArgs(
                    bypass_pull_request_allowances=github.BranchProtectionV3RequiredPullRequestReviewsBypassPullRequestAllowancesArgs(
                        apps=list(plan.classic_review_bypass_apps),
                        teams=[],
                        users=[],
                    ),
                    dismiss_stale_reviews=False,
                    require_code_owner_reviews=False,
                    require_last_push_approval=False,
                    required_approving_review_count=1,
                ),
                opts=pulumi.ResourceOptions(
                    import_=plan.classic_branch_protection_import,
                    depends_on=[installation],
                ),
            )
        )
    resources.append(
        github.RepositoryRuleset(
            _resource_name(plan.repository, "protect-main"),
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
                depends_on=[installation],
            ),
        )
    )
    if plan.required_checks:
        resources.append(
            github.RepositoryRuleset(
                _resource_name(plan.repository, "require-main-ci"),
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
        )
    return tuple(resources)


def register_dependency_updater_environment(
    organization: str,
    repository: str,
) -> tuple[pulumi.CustomResource, pulumi.CustomResource]:
    """Create one updater environment and restrict it to the main branch."""
    normalized_repository = repository_name(organization, repository)
    environment = github.RepositoryEnvironment(
        _resource_name(normalized_repository, "external-runtime-updater-environment"),
        repository=normalized_repository,
        environment=UPDATER_ENVIRONMENT,
        can_admins_bypass=False,
        deployment_branch_policy=github.RepositoryEnvironmentDeploymentBranchPolicyArgs(
            custom_branch_policies=True,
            protected_branches=False,
        ),
    )
    deployment_policy = github.RepositoryEnvironmentDeploymentPolicy(
        _resource_name(normalized_repository, "external-runtime-updater-main-policy"),
        repository=normalized_repository,
        environment=UPDATER_ENVIRONMENT,
        branch_pattern=UPDATER_BRANCH,
        opts=pulumi.ResourceOptions(depends_on=[environment]),
    )
    return environment, deployment_policy
