"""Behavior tests for explicit original-trial environment provisioning."""

from harbor_config.models.trial.config import TrialConfig

from marinskyrl.harbor_environments import (
    EnvironmentResourceOverrides,
    HarborTaskBinding,
    apply_task_binding,
)


def test_task_binding_overlays_agent_and_verifier_policy():
    config = TrialConfig(
        task={"path": "unused"},
        agent={"name": "pi", "mcp_servers": [{"name": "existing", "transport": "stdio", "command": "existing"}]},
        environment={"type": "daytona"},
        verifier={"env": {"EXISTING": "value"}},
    )
    binding = HarborTaskBinding(
        agent_mcp_servers=[{"name": "tools", "transport": "stdio", "command": "tool-server"}],
        verifier_environment={"MODEL_NAME": "judge-model"},
    )

    actual = apply_task_binding(config, binding)

    assert actual.agent.mcp_servers[0].name == "tools"
    assert actual.verifier.env == {
        "EXISTING": "value",
        "MODEL_NAME": "judge-model",
    }

    resource_only = apply_task_binding(
        config,
        HarborTaskBinding(environment_resources=EnvironmentResourceOverrides(override_storage_mb=10240)),
    )
    assert resource_only.agent.mcp_servers[0].name == "existing"
    assert resource_only.verifier.env == {"EXISTING": "value"}
