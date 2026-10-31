"""Snapshot approval preserves the source-specific policy resource budget."""

from harbor_config.models.trial.config import EnvironmentConfig
from marinskyrl.harbor_environments import EnvironmentResourceOverrides, original_trial_environment


def test_bixbench_policy_override_does_not_replace_other_task_resources():
    snapshots = {"bixbench": "approved-bix", "small-task": "approved-small"}
    overrides = {"bixbench": EnvironmentResourceOverrides(override_storage_mb=10240)}
    original = EnvironmentConfig(type="daytona", override_storage_mb=20480, kwargs={"api_key_env": "DAYTONA_TEST_KEY"})
    bixbench = original_trial_environment(original, "bixbench", snapshots, overrides)
    small = original_trial_environment(
        original.model_copy(update={"override_storage_mb": 2048}), "small-task", snapshots, overrides
    )
    assert bixbench.override_storage_mb == 10240
    assert small.override_storage_mb == 2048
    assert original.override_storage_mb == 20480
    assert bixbench.kwargs == {
        "api_key_env": "DAYTONA_TEST_KEY",
        "snapshot_template_name": "approved-bix",
        "auto_snapshot": False,
        "assume_global_snapshot": True,
    }
    assert small.kwargs["snapshot_template_name"] == "approved-small"
    assert bixbench.force_build is False


def test_native_daytona_trial_does_not_require_a_snapshot_map():
    environment = EnvironmentConfig(type="daytona", kwargs={"api_key_env": "DAYTONA_EVAL_API_KEY"})
    actual = original_trial_environment(
        environment, "native-image", {}, {"native-image": EnvironmentResourceOverrides(override_storage_mb=10240)}
    )
    assert actual.override_storage_mb == 10240
    assert actual.kwargs == {"api_key_env": "DAYTONA_EVAL_API_KEY"}
    assert actual.type == environment.type
    assert environment.override_storage_mb is None
