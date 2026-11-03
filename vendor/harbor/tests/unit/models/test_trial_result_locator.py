from pathlib import Path

import pytest

from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, TrialResult
from harbor.trial.attempt import trial_relative_path


def _result(tmp_path: Path, *, trial_relpath: str | None) -> TrialResult:
    config = TrialConfig(
        task=TaskConfig(path=tmp_path / "task"),
        trial_name="trial-1",
        trials_dir=tmp_path / "original",
    )
    return TrialResult(
        task_name="task",
        trial_name="trial-1",
        trial_uri=(tmp_path / "original" / "trial-1").as_uri(),
        trial_relpath=trial_relpath,
        task_id=config.task.get_task_id(),
        task_checksum="checksum",
        config=config,
        agent_info=AgentInfo(name="agent", version="1"),
    )


@pytest.mark.unit
def test_trial_result_resolves_portably_under_supplied_root(tmp_path) -> None:
    result = _result(tmp_path, trial_relpath="trial-1/attempts/002")
    exported_root = tmp_path / "exported"

    assert str(result.resolve_trial_path(exported_root)) == str(
        exported_root / "trial-1" / "attempts" / "002"
    )


@pytest.mark.unit
def test_legacy_result_without_relative_locator_uses_trial_uri(tmp_path) -> None:
    result = _result(tmp_path, trial_relpath=None)
    legacy_json = result.model_dump_json(exclude={"trial_relpath"})

    restored = TrialResult.model_validate_json(legacy_json)

    assert restored.trial_relpath is None
    assert str(restored.resolve_trial_path(tmp_path / "exported")) == str(
        result.resolve_trial_path()
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "trial_relpath",
    ["", ".", "/absolute/trial", "../trial", "trial/../other", "trial\\result"],
)
def test_trial_relative_locator_rejects_nonportable_paths(
    tmp_path, trial_relpath
) -> None:
    with pytest.raises(ValueError, match="normalized relative POSIX path"):
        _result(tmp_path, trial_relpath=trial_relpath)


@pytest.mark.unit
def test_trial_relative_path_includes_attempt_scope(tmp_path) -> None:
    config = TrialConfig(
        task=TaskConfig(path=tmp_path / "task"),
        trial_name="trial-1",
    )

    assert trial_relative_path(config) == "trial-1"
    assert trial_relative_path(config, 7) == "trial-1/attempts/007"
