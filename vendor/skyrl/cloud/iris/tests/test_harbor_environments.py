"""Behavior tests for explicit original-trial environment provisioning."""

import pytest
from harbor_config.models.trial.config import EnvironmentConfig, TrialConfig

from marinskyrl.harbor_environments import (
    DeferredVerifierEnvironment,
    EnvironmentResourceOverrides,
    HarborEnvironmentSettings,
    HarborTaskBinding,
    apply_task_binding,
    original_trial_environment,
)


def test_native_provisioning_builds_source_with_task_scoped_resources():
    settings = HarborEnvironmentSettings(
        provisioning="native",
        task_bindings={
            "large-task": HarborTaskBinding(
                environment_resources=EnvironmentResourceOverrides(override_storage_mb=10240)
            )
        },
    )
    original = EnvironmentConfig(type="daytona", override_storage_mb=2048, kwargs={"api_key_env": "DAYTONA_KEY"})

    actual = original_trial_environment(
        original,
        "shared-image",
        settings,
        settings.task_bindings["large-task"].environment_resources,
    )

    assert actual.override_storage_mb == 10240
    assert actual.kwargs == {"api_key_env": "DAYTONA_KEY"}
    assert "snapshot_template_name" not in actual.kwargs
    assert original.override_storage_mb == 2048


def test_approved_snapshots_reject_unapproved_sources():
    settings = HarborEnvironmentSettings(
        provisioning="approved_snapshots",
        approved_snapshots={"approved-image": "snapshot-1"},
    )
    environment = EnvironmentConfig(type="daytona", kwargs={"api_key_env": "DAYTONA_KEY"})

    approved = original_trial_environment(environment, "approved-image", settings, EnvironmentResourceOverrides())
    assert approved.kwargs == {
        "api_key_env": "DAYTONA_KEY",
        "snapshot_template_name": "snapshot-1",
        "auto_snapshot": False,
        "assume_global_snapshot": True,
    }
    with pytest.raises(ValueError, match="no approved snapshot"):
        original_trial_environment(environment, "other-image", settings, EnvironmentResourceOverrides())


def test_task_binding_overlays_agent_and_verifier_policy():
    config = TrialConfig(
        task={"path": "unused"},
        agent={"name": "pi", "mcp_servers": [{"name": "existing", "transport": "stdio", "command": "existing"}]},
        environment={"type": "daytona"},
        verifier={"env": {"EXISTING": "value"}},
    )
    binding = HarborTaskBinding(
        agent_mcp_servers=[{"name": "tools", "transport": "stdio", "command": "tool-server"}],
        verifier_environment=DeferredVerifierEnvironment(
            values={"MODEL_NAME": "judge-model"},
            variables={"OPENAI_API_KEY": "JUDGE_API_KEY"},
        ),
    )

    actual = apply_task_binding(config, binding)

    assert actual.agent.mcp_servers[0].name == "tools"
    assert actual.verifier.env == {
        "EXISTING": "value",
        "MODEL_NAME": "judge-model",
        "OPENAI_API_KEY": "${JUDGE_API_KEY}",
    }

    resource_only = apply_task_binding(
        config,
        HarborTaskBinding(environment_resources=EnvironmentResourceOverrides(override_storage_mb=10240)),
    )
    assert resource_only.agent.mcp_servers[0].name == "existing"
    assert resource_only.verifier.env == {"EXISTING": "value"}
