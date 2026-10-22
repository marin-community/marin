from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.iris import list_iris_jobs


@pytest.fixture
def iris_command(monkeypatch):
    recorder = SimpleNamespace(
        calls=[],
        response=SimpleNamespace(
            returncode=0,
            stdout="job_id,state,submitted_at_ms,started_at_ms,finished_at_ms,error,exit_code\n",
            stderr="",
        ),
    )

    def fake_command(arguments, **kwargs):
        recorder.calls.append((arguments, kwargs))
        return recorder.response

    monkeypatch.setattr(list_iris_jobs, "run_iris_command", fake_command)
    return recorder


def test_job_inventory_classifies_and_orders_controller_rows(iris_command):
    output = """info line
job_id,state,submitted_at_ms,started_at_ms,finished_at_ms,error,exit_code
/benjaminfeuer/tracegen-a,4,1000,1100,1200,,0
/benjaminfeuer/rl-run,3,2000,2100,,,
/benjaminfeuer/eval-b,6,3000,3100,3200,manual stop,1
"""
    iris_command.response.stdout = output
    rows = list_iris_jobs.query_jobs(user="benjaminfeuer", hours=24, cluster="cw", now_ms=86_400_000)
    assert [row["job_id"] for row in rows] == [
        "/benjaminfeuer/tracegen-a",
        "/benjaminfeuer/rl-run",
        "/benjaminfeuer/eval-b",
    ]
    values = [list_iris_jobs.job_filter_values(row, now_ms=3_800_000) for row in rows]
    assert [value["type"] for value in values] == ["datagen", "RL", "eval"]
    assert [value["state"] for value in values] == ["succeeded", "running", "terminated"]


def test_job_inventory_formats_running_and_finished_durations():
    finished = {
        "submitted_at_ms": "0",
        "started_at_ms": "60_000",
        "finished_at_ms": "7_320_000",
    }
    pending = {"submitted_at_ms": "0", "started_at_ms": "", "finished_at_ms": ""}

    assert list_iris_jobs.job_filter_values(finished)["duration"] == "2h 1m"
    assert list_iris_jobs.job_filter_values(pending, now_ms=90_000)["duration"] == "1m"


def test_job_inventory_filters_regex_fields():
    rows = [
        {
            "job_id": "/benjaminfeuer/glm52-running",
            "state": "3",
            "cluster": "cw-rno2a",
        },
        {
            "job_id": "/benjaminfeuer/other-running",
            "state": "3",
            "cluster": "marin",
        },
        {
            "job_id": "/benjaminfeuer/glm52-failed",
            "state": "5",
            "cluster": "cw-rno2a",
        },
    ]
    filters = list_iris_jobs.parse_regex_filters(
        ["state=RUNNING", "name=^glm52", "cluster=^cw-"],
        {
            "cluster",
            "submitted",
            "job",
            "name",
            "type",
            "state",
            "duration",
            "exit",
            "error",
        },
    )

    filtered = list_iris_jobs.filter_records(rows, filters, list_iris_jobs.job_filter_values)

    assert [row["job_id"] for row in filtered] == ["/benjaminfeuer/glm52-running"]


def test_job_inventory_overrides_an_inherited_non_coreweave_kubeconfig(monkeypatch, iris_command):
    monkeypatch.setenv("KUBECONFIG", "/tmp/other-kubeconfig")

    list_iris_jobs.query_jobs(user="benjaminfeuer", hours=24, cluster="cw-us-east-02a")

    assert iris_command.calls[0][1]["environment"]["KUBECONFIG"] == str(
        list_iris_jobs.COREWEAVE_CLUSTERS["cw-us-east-02a"].kubeconfig
    )


def test_job_inventory_rejects_an_invalid_user_before_query():
    with pytest.raises(ValueError, match="Invalid Iris user"):
        list_iris_jobs.query_jobs(user="x' OR 1=1", hours=24, cluster="cw")
