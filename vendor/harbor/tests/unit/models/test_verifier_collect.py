import pytest
from pydantic import ValidationError

from harbor.models.task.config import TaskConfig


def test_deepswe_collect_hook_toml_parses_for_task_and_step() -> None:
    config = TaskConfig.model_validate_toml(
        """
[[verifier.collect]]
command = "cd /app && git diff --binary HEAD > /logs/artifacts/model.patch"
timeout_sec = 300.0
user = 1000

[[steps]]
name = "finalize"

[[steps.verifier.collect]]
command = "write-step-evidence"
"""
    )

    task_hook = config.verifier.collect[0]
    assert task_hook.command.endswith("/logs/artifacts/model.patch")
    assert task_hook.timeout_sec == 300.0
    assert task_hook.user == 1000
    assert task_hook.service == "main"
    assert config.steps is not None
    assert config.steps[0].verifier.collect[0].command == "write-step-evidence"


@pytest.mark.parametrize("timeout_sec", [0, -0.1])
def test_collect_hook_timeout_must_be_positive(timeout_sec: float) -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        TaskConfig.model_validate(
            {
                "verifier": {
                    "collect": [{"command": "snapshot", "timeout_sec": timeout_sec}]
                }
            }
        )


def test_collect_hook_rejects_sidecar_service_in_scoped_port() -> None:
    with pytest.raises(ValidationError, match="main"):
        TaskConfig.model_validate(
            {"verifier": {"collect": [{"service": "postgres", "command": "pg_dump"}]}}
        )
