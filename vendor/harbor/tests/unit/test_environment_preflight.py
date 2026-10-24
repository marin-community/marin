"""The environment factory admits only maintained Shellbox adapters."""

import pytest
from pydantic import ValidationError

from harbor.environments.factory import EnvironmentFactory
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
from harbor.models.trial.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from harbor_config import load_config_document


@pytest.mark.parametrize(
    "environment_type",
    [EnvironmentType.IRIS, EnvironmentType.QEMU, EnvironmentType.SHELLSIM],
)
def test_factory_constructs_the_selected_shellbox_adapter(tmp_path, environment_type):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    trial_paths = TrialPaths(trial_dir=tmp_path / "trial")
    trial_paths.mkdir()
    task = TaskEnvironmentConfig(allow_internet=False, workdir="/workspace")
    if environment_type is EnvironmentType.IRIS:
        task.docker_image = "example.test/task@sha256:" + "a" * 64
        kwargs = {"controller_url": "http://controller.invalid"}
    elif environment_type is EnvironmentType.QEMU:
        kwargs = {"guest_bundle": str(environment_dir)}
    else:
        kwargs = {}
    environment = EnvironmentFactory.create_environment_from_config(
        EnvironmentConfig(type=environment_type, kwargs=kwargs),
        environment_dir=environment_dir,
        environment_name="task",
        session_id="trial",
        trial_paths=trial_paths,
        task_env_config=task,
    )
    assert environment.type() == environment_type
    assert environment.capabilities.disable_internet


@pytest.mark.parametrize(
    "environment", [{"type": "docker"}, {"import_path": "untrusted.module:Environment"}]
)
def test_config_rejects_execution_outside_shellbox_before_loading_a_backend(
    environment,
):
    with pytest.raises(ValidationError):
        load_config_document({"environment": environment})
