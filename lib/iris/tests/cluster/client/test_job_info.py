# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.cluster.client.job_info import JobInfo, get_job_info, resolve_job_user, set_job_info
from iris.cluster.types import JobName


@pytest.fixture(autouse=True)
def _reset_job_info():
    """Clear the JobInfo contextvar between tests so state doesn't leak."""
    set_job_info(None)
    yield
    set_job_info(None)


# IRIS_USER and .marin.yaml isolation is suite-wide: the autouse
# _isolate_marin_user_config fixture in tests/conftest.py points
# MARIN_CONFIG_PATH at <tmp_path>/.marin.yaml, so tests below write there.


def test_job_info_user_derives_from_task_id():
    info = JobInfo(task_id=JobName.from_wire("/alice/train/0"))
    assert info.user == "alice"


def test_resolve_job_user_prefers_explicit_value():
    assert resolve_job_user("alice") == "alice"


def test_resolve_job_user_uses_current_job_info_before_os_user(monkeypatch):
    set_job_info(JobInfo(task_id=JobName.from_wire("/alice/train/0")))
    monkeypatch.setattr("getpass.getuser", lambda: "local-user")
    assert resolve_job_user() == "alice"


def test_resolve_job_user_falls_back_to_os_user(monkeypatch):
    monkeypatch.setattr("getpass.getuser", lambda: "local-user")
    assert resolve_job_user() == "local-user"


def test_resolve_job_user_falls_back_to_root_when_os_user_lookup_fails(monkeypatch):
    def _raise():
        raise OSError("no passwd entry")

    monkeypatch.setattr("getpass.getuser", _raise)
    assert resolve_job_user() == "root"


def test_resolve_job_user_reads_and_strips_iris_user_env(monkeypatch):
    monkeypatch.setenv("IRIS_USER", " mwittmann ")
    monkeypatch.setattr("getpass.getuser", lambda: "local-user")
    assert resolve_job_user() == "mwittmann"


def test_resolve_job_user_iris_user_env_beats_current_job_info(monkeypatch):
    """A deliberate per-shell identity survives shared dev pods (which are Iris jobs)."""
    set_job_info(JobInfo(task_id=JobName.from_wire("/alice/train/0")))
    monkeypatch.setenv("IRIS_USER", "mwittmann")
    assert resolve_job_user() == "mwittmann"


def test_resolve_job_user_current_job_info_beats_marin_yaml(tmp_path):
    """A .marin.yaml riding along in a job bundle must not re-attribute in-job submissions."""
    set_job_info(JobInfo(task_id=JobName.from_wire("/alice/train/0")))
    (tmp_path / ".marin.yaml").write_text("user: mwittmann\n")
    assert resolve_job_user() == "alice"


def test_resolve_job_user_reads_marin_yaml_user(monkeypatch, tmp_path):
    (tmp_path / ".marin.yaml").write_text("user: mwittmann\n")
    monkeypatch.setattr("getpass.getuser", lambda: "local-user")
    assert resolve_job_user() == "mwittmann"


def test_resolve_job_user_ignores_marin_yaml_without_user_key(monkeypatch, tmp_path):
    """Existing .marin.yaml files that only configure env: keep OS-user attribution."""
    (tmp_path / ".marin.yaml").write_text("env:\n  WANDB_API_KEY: abc\n")
    monkeypatch.setattr("getpass.getuser", lambda: "local-user")
    assert resolve_job_user() == "local-user"


@pytest.mark.parametrize("value", ["   ", "team/alice"])
def test_resolve_job_user_rejects_invalid_iris_user_env(monkeypatch, value):
    monkeypatch.setenv("IRIS_USER", value)
    with pytest.raises(ValueError, match="IRIS_USER"):
        resolve_job_user()


@pytest.mark.parametrize(
    "content",
    [
        "user: [not, a, string]\n",
        "user: team/alice\n",
        "user: [unclosed\n",  # YAML syntax error must surface as ValueError naming the file
    ],
)
def test_resolve_job_user_rejects_invalid_marin_yaml_user(tmp_path, content):
    (tmp_path / ".marin.yaml").write_text(content)
    with pytest.raises(ValueError, match=r"\.marin\.yaml"):
        resolve_job_user()


def test_worker_region_from_env(monkeypatch):
    """IRIS_WORKER_REGION is read into JobInfo.worker_region (regression for #5541)."""
    monkeypatch.setenv("IRIS_TASK_ID", "/test-user/my-job/0:1")
    monkeypatch.setenv("IRIS_WORKER_REGION", "us-central1")
    info = get_job_info()
    assert info is not None
    assert info.worker_region == "us-central1"


def test_worker_region_absent_when_env_not_set(monkeypatch):
    """worker_region is None when IRIS_WORKER_REGION is not set."""
    monkeypatch.setenv("IRIS_TASK_ID", "/test-user/my-job/0:1")
    monkeypatch.delenv("IRIS_WORKER_REGION", raising=False)
    info = get_job_info()
    assert info is not None
    assert info.worker_region is None


def test_default_jobinfo_exposes_worker_region_attribute():
    """JobInfo.worker_region defaults to None so attribute access never raises."""
    info = JobInfo(task_id=JobName.from_wire("/alice/train/0"))
    assert info.worker_region is None
