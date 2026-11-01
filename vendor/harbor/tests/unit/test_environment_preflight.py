"""The environment factory admits only maintained Shellbox adapters."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from harbor.job import Job
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import EnvironmentConfig, TaskConfig, TrialConfig, VerifierConfig
from harbor.trial.trial import Trial
from harbor_config import load_config_document


@pytest.mark.parametrize(
    "environment", [{"type": "docker"}, {"import_path": "untrusted.module:Environment"}]
)
def test_config_rejects_execution_outside_shellbox_before_loading_a_backend(
    environment,
):
    with pytest.raises(ValidationError):
        load_config_document({"environment": environment})


@pytest.fixture
def retention_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    def forbidden_backend(*args, **kwargs):
        pytest.fail("A retention request reached backend construction")

    monkeypatch.setattr("harbor.environments.local.LocalMachineFactory", forbidden_backend)
    task = tmp_path / "task"
    task.mkdir()
    (task / "environment").mkdir()
    (task / "task.toml").write_text(
        '[environment]\ndocker_image = "registry.invalid/task@sha256:' + "a" * 64 + '"\n'
    )
    (task / "instruction.md").write_text("Return the answer.\n")
    (task / "tests").mkdir()
    (task / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
    return task


async def test_job_rejects_retention_before_backend_construction(tmp_path: Path, retention_task: Path):
    config = JobConfig(jobs_dir=tmp_path / "jobs", tasks=[TaskConfig(path=retention_task)])
    # CLI overrides mutate this field after the schema has been validated.
    config.environment.delete = False

    with pytest.raises(ValueError, match="Shellbox environments cannot be retained"):
        await Job.create(config)

    assert not (tmp_path / "jobs").exists()


async def test_direct_trial_rejects_retention_before_backend_construction(tmp_path: Path, retention_task: Path):
    config = TrialConfig(
        task=TaskConfig(path=retention_task),
        trials_dir=tmp_path / "trials",
        environment=EnvironmentConfig(delete=False),
        verifier=VerifierConfig(disable=True),
    )

    with pytest.raises(ValueError, match="Shellbox environments cannot be retained"):
        await Trial.create(config)
