import json
import logging
from types import SimpleNamespace

import pytest
from upath import UPath

from harbor.job import Job
from harbor.cli.jobs import resume
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import ArtifactAuthorityConfig, TaskConfig, TrialConfig
from harbor.trial.single_step import SingleStepTrial
from harbor.utils.artifact_authority import (
    ArtifactAuthorityError,
    validate_artifact_authority,
)


def _authority(record_path, *, authority_id="launch-123", mode="require"):
    return ArtifactAuthorityConfig(
        authority_id=authority_id,
        record_path=record_path,
        mode=mode,
    )


@pytest.mark.unit
def test_matching_precreated_authority_record_succeeds(tmp_path) -> None:
    mount_root = tmp_path / "artifact-store"
    jobs_dir = mount_root / "jobs"
    mount_root.mkdir()
    record_path = mount_root / ".harbor-authority.json"
    record_path.write_text(
        json.dumps(
            {
                "authority_id": "launch-123",
                "hostname": "diagnostic-only",
                "device": 42,
            }
        )
    )

    validate_artifact_authority(
        jobs_dir,
        _authority(record_path),
        operation="test access",
    )


@pytest.mark.unit
@pytest.mark.parametrize("record_contents", [None, "not json", '{"hostname": "x"}'])
def test_required_authority_rejects_missing_or_malformed_record(
    tmp_path, record_contents
) -> None:
    root = tmp_path / "artifact-store"
    root.mkdir()
    record_path = root / ".harbor-authority.json"
    if record_contents is not None:
        record_path.write_text(record_contents)

    with pytest.raises(ArtifactAuthorityError, match="authority record"):
        validate_artifact_authority(
            root,
            _authority(record_path),
            operation="test access",
        )


@pytest.mark.unit
def test_required_authority_rejects_mismatching_id(tmp_path) -> None:
    root = tmp_path / "artifact-store"
    root.mkdir()
    record_path = root / ".harbor-authority.json"
    record_path.write_text(json.dumps({"authority_id": "another-launch"}))

    with pytest.raises(ArtifactAuthorityError, match="does not match"):
        validate_artifact_authority(
            root,
            _authority(record_path),
            operation="test access",
        )


@pytest.mark.unit
def test_warning_mode_reports_failure_and_continues(tmp_path, caplog) -> None:
    root = tmp_path / "artifact-store"
    record_path = root / ".harbor-authority.json"

    with caplog.at_level(logging.WARNING):
        validate_artifact_authority(
            root,
            _authority(record_path, mode="warn"),
            operation="trial startup",
        )

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING


@pytest.mark.unit
def test_unprotected_local_and_remote_roots_are_unchanged(tmp_path) -> None:
    validate_artifact_authority(tmp_path, None, operation="local access")
    validate_artifact_authority(
        UPath("memory://artifact-store/jobs"),
        None,
        operation="remote access",
    )


@pytest.mark.unit
def test_protected_remote_root_is_rejected(tmp_path) -> None:
    with pytest.raises(ArtifactAuthorityError, match="only local artifact roots"):
        validate_artifact_authority(
            UPath("memory://artifact-store/jobs"),
            _authority(tmp_path / ".harbor-authority.json"),
            operation="remote access",
        )


@pytest.mark.unit
def test_authority_record_path_must_be_absolute_and_above_root(tmp_path) -> None:
    root = tmp_path / "artifact-store" / "jobs"
    outside_record = tmp_path / "other-store" / ".harbor-authority.json"

    with pytest.raises(ArtifactAuthorityError, match="must be absolute"):
        validate_artifact_authority(
            root,
            _authority(".harbor-authority.json"),
            operation="test access",
        )
    with pytest.raises(ArtifactAuthorityError, match="lexical ancestor"):
        validate_artifact_authority(
            root,
            _authority(outside_record),
            operation="test access",
        )


@pytest.mark.unit
def test_authority_record_must_not_be_a_symlink(tmp_path) -> None:
    root = tmp_path / "artifact-store"
    root.mkdir()
    target = root / "record-target.json"
    target.write_text(json.dumps({"authority_id": "launch-123"}))
    record_path = root / ".harbor-authority.json"
    record_path.symlink_to(target)

    with pytest.raises(ArtifactAuthorityError, match="must not be a symlink"):
        validate_artifact_authority(
            root,
            _authority(record_path),
            operation="test access",
        )


@pytest.mark.unit
def test_job_checks_authority_before_creating_job_directory(tmp_path) -> None:
    jobs_dir = tmp_path / "artifact-store"
    config = JobConfig(
        job_name="protected-job",
        jobs_dir=jobs_dir,
        artifact_authority=_authority(jobs_dir / ".harbor-authority.json"),
    )

    with pytest.raises(ArtifactAuthorityError, match="before job startup"):
        Job(config, _task_configs=[], _metrics={})

    assert not (jobs_dir / "protected-job").exists()


@pytest.mark.unit
def test_direct_trial_checks_authority_before_creating_trial_directory(
    tmp_path,
) -> None:
    trials_dir = tmp_path / "artifact-store"
    config = TrialConfig(
        task=TaskConfig(path=tmp_path / "task"),
        trial_name="protected-trial",
        trials_dir=trials_dir,
        artifact_authority=_authority(trials_dir / ".harbor-authority.json"),
    )

    with pytest.raises(ArtifactAuthorityError, match="before trial startup"):
        SingleStepTrial(config, _task=SimpleNamespace(has_steps=False))

    assert not (trials_dir / "protected-trial").exists()


@pytest.mark.unit
def test_job_propagates_authority_to_trial_configs(tmp_path) -> None:
    jobs_dir = tmp_path / "artifact-store"
    jobs_dir.mkdir()
    record_path = jobs_dir / ".harbor-authority.json"
    record_path.write_text(json.dumps({"authority_id": "launch-123"}))
    task_config = TaskConfig(path=tmp_path / "task")
    authority = _authority(record_path)
    config = JobConfig(
        job_name="protected-job",
        jobs_dir=jobs_dir,
        artifact_authority=authority,
        tasks=[task_config],
    )

    job = Job(config, _task_configs=[task_config], _metrics={})
    try:
        assert len(job._trial_configs) == 1
        assert job._trial_configs[0].artifact_authority == authority
    finally:
        job._close_logger_handlers()


@pytest.mark.unit
def test_authority_config_round_trips_with_job_and_trial_configs(tmp_path) -> None:
    authority = _authority(
        tmp_path / "artifact-store" / ".harbor-authority.json",
        mode="warn",
    )
    job_config = JobConfig(artifact_authority=authority)
    trial_config = TrialConfig(
        task=TaskConfig(path=tmp_path / "task"),
        artifact_authority=authority,
    )

    assert (
        JobConfig.model_validate_json(job_config.model_dump_json()).artifact_authority
        == authority
    )
    assert (
        TrialConfig.model_validate_json(
            trial_config.model_dump_json()
        ).artifact_authority
        == authority
    )


@pytest.mark.unit
def test_resume_checks_authority_before_pruning_trial_directories(tmp_path) -> None:
    job_dir = tmp_path / "protected-job"
    trial_dir = job_dir / "unscored-trial"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text("not valid json")
    config = JobConfig(
        job_name=job_dir.name,
        jobs_dir=tmp_path,
        artifact_authority=_authority(tmp_path / ".harbor-authority.json"),
    )
    (job_dir / "config.json").write_text(config.model_dump_json())

    with pytest.raises(ArtifactAuthorityError, match="before job resume"):
        resume(job_path=job_dir)

    assert trial_dir.exists()
