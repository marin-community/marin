from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iris.cluster.client.job_info import JobInfo  # noqa: E402
from iris.cluster.types import JobName  # noqa: E402

from cloud.iris import telemetry_env  # noqa: E402
from marinskyrl.environment_contract import (  # noqa: E402
    EXECUTION_UID_ENV,
    RUN_ID_ENV,
    TELEMETRY_ENDPOINT_ENV,
    TRAINING_TYPE_ENV,
    TrainingType,
)
from skyrl_train.telemetry import TelemetryConfig  # noqa: E402


_ATTEMPT_UID = "01JABCDEF0123456789"


def _job_info() -> JobInfo:
    return JobInfo(
        task_id=JobName.from_wire("/atqamar/iceball-micro-0/1"),
        attempt_id=2,
        attempt_uid=_ATTEMPT_UID,
    )


class _Client:
    def resolve_endpoint(self, _name: str) -> str:
        return "http://finelog.marin.svc.cluster.local:8080/"


class _UnreachableClient:
    def resolve_endpoint(self, _name: str) -> str:
        raise ConnectionError("controller is unreachable")


@dataclass(frozen=True)
class _Context:
    client: _Client | _UnreachableClient


def _in_cluster(monkeypatch) -> None:
    monkeypatch.setattr(telemetry_env, "get_job_info", _job_info)
    monkeypatch.setattr(telemetry_env, "get_iris_ctx", lambda: _Context(_Client()))


def test_telemetry_environment_in_cluster_returns_endpoint_and_identity(monkeypatch) -> None:
    _in_cluster(monkeypatch)

    assert telemetry_env.telemetry_environment() == {
        TELEMETRY_ENDPOINT_ENV: "http://finelog.marin.svc.cluster.local:8080/v1/telemetry",
        RUN_ID_ENV: "/atqamar/iceball-micro-0",
        EXECUTION_UID_ENV: f"iris:{_ATTEMPT_UID}",
    }


def test_telemetry_environment_run_id_sources_follow_precedence(monkeypatch) -> None:
    _in_cluster(monkeypatch)

    assert telemetry_env.telemetry_environment()[RUN_ID_ENV] == "/atqamar/iceball-micro-0"
    monkeypatch.setenv(RUN_ID_ENV, "inherited-run")
    assert telemetry_env.telemetry_environment()[RUN_ID_ENV] == "inherited-run"
    assert telemetry_env.telemetry_environment(run_id="explicit-run")[RUN_ID_ENV] == "explicit-run"


def test_telemetry_environment_without_cluster_context_returns_nothing(monkeypatch) -> None:
    monkeypatch.setattr(telemetry_env, "get_job_info", lambda: None)
    monkeypatch.setattr(telemetry_env, "get_iris_ctx", lambda: None)

    assert telemetry_env.telemetry_environment() == {}


def test_telemetry_environment_unreachable_controller_returns_nothing(monkeypatch) -> None:
    monkeypatch.setattr(telemetry_env, "get_job_info", _job_info)
    monkeypatch.setattr(telemetry_env, "get_iris_ctx", lambda: _Context(_UnreachableClient()))

    assert telemetry_env.telemetry_environment() == {}


def test_telemetry_environment_round_trips_through_trainer_config(monkeypatch) -> None:
    _in_cluster(monkeypatch)
    exported = telemetry_env.telemetry_environment(training_type=TrainingType.ASYNC)
    for name, value in exported.items():
        monkeypatch.setenv(name, value)

    config = TelemetryConfig.from_environment()
    assert config.endpoint == exported[TELEMETRY_ENDPOINT_ENV]
    assert config.run_id == exported[RUN_ID_ENV]
    assert config.execution_uid == exported[EXECUTION_UID_ENV]
    assert config.training_type is TrainingType.ASYNC
    assert exported[TRAINING_TYPE_ENV] == "async"


def test_telemetry_environment_omits_an_unset_training_type(monkeypatch) -> None:
    _in_cluster(monkeypatch)
    monkeypatch.delenv(TRAINING_TYPE_ENV, raising=False)
    assert TRAINING_TYPE_ENV not in telemetry_env.telemetry_environment()
