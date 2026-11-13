# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.post_training.tmax_reserve import harbor_job
from experiments.post_training.tmax_reserve.harbor_job import sandbox_seconds
from experiments.post_training.tmax_reserve.null_run import ALLOWED_HOST, PROBE_KEYSTROKES, ScriptedModel, summarize

INSTRUCTIONS = {"a": "Count the lines in data.txt.", "b": "Sort the CSV by date."}


def _first(task):
    return [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": f"Task:\n{INSTRUCTIONS[task]}\n"},
    ]


def test_scripted_model_probes_then_completes_and_records_the_probe():
    model = ScriptedModel(INSTRUCTIONS, controls={})
    first = json.loads(model.answer(_first("a")))
    assert first["commands"][0]["keystrokes"] == PROBE_KEYSTROKES and not first["task_complete"]
    screen = f"TMUX tmux 3.2a\nEGRESS https://github.com/ URLError\nEGRESS {ALLOWED_HOST} 200\nPROBE_DONE\n"
    second = json.loads(
        model.answer([*_first("a"), {"role": "assistant", "content": "x"}, {"role": "user", "content": screen}])
    )
    assert second["task_complete"] and second["commands"] == []
    assert model.probes["a"] == [
        {"tmux": ["tmux 3.2a"], "egress": {"https://github.com/": "URLError", ALLOWED_HOST: "200"}, "probe_done": True}
    ]


def test_scripted_model_runs_control_commands_instead_of_the_probe():
    model = ScriptedModel(INSTRUCTIONS, controls={"b": "sort data.csv > out.csv\n"})
    assert json.loads(model.answer(_first("b")))["commands"][0]["keystrokes"] == "sort data.csv > out.csv\n"


def test_summary_flags_egress_beyond_the_allowed_host_and_nonzero_null_rewards():
    probes = {
        "a": [
            {"tmux": ["tmux 3.2a"], "egress": {"https://github.com/": "200", ALLOWED_HOST: "200"}, "probe_done": True}
        ],
        "c": [{"tmux": ["missing"], "egress": {"https://github.com/": "URLError"}, "probe_done": False}],
    }
    trials = [
        {"task": "a", "reward": 0.0, "exception": None},
        {"task": "c", "reward": 1.0, "exception": None},
        {"task": "b", "reward": 1.0, "exception": None},
        {"task": "d", "reward": None, "exception": "DaytonaError"},
    ]
    summary = summarize(trials, probes, controls={"b"})
    assert summary["egress_violations"] == ["https://github.com/"]
    assert summary["null_tasks_scoring_nonzero"] == ["c"]
    assert summary["tasks_without_reward"] == ["d"]
    assert summary["control_rewards"] == {"b": 1.0}
    assert summary["probes_with_tmux"] == 1 and summary["probes_incomplete"] == 1


def test_sandbox_seconds_counts_trials_that_reached_environment_setup(tmp_path):
    def trial(name, data):
        (tmp_path / "job" / name).mkdir(parents=True)
        (tmp_path / "job" / name / "result.json").write_text(json.dumps(data))

    span = {"started_at": "2026-11-13T00:00:00+00:00", "finished_at": "2026-11-13T00:02:00+00:00"}
    trial("ran", {"environment_setup": span, "agent_setup": span, "agent_execution": span, "verifier": None})
    trial("never_started", {"environment_setup": None, "exception_info": {"exception_type": "DaytonaError"}})
    assert sandbox_seconds(tmp_path) == 60 + 360


def _job(tmp_path, monkeypatch, *, preexisting, non_general, driver_error=None, prepare_error=None):
    dataset = tmp_path / "tasks"
    for task in ("t1", "t2", "t3"):
        (dataset / task).mkdir(parents=True)
    calls = []

    def snapshots(command, *arguments, env):
        if command == "prepare" and prepare_error:
            raise prepare_error
        calls.append(
            (command, json.loads(Path(arguments[0]).read_text()) if command in ("existing", "delete") else None)
        )
        return {
            "names": {"t1": "snap-a", "t2": "snap-a", "t3": "snap-b"},
            "existing": preexisting,
            "usage": {"non_general": non_general},
            "prepare": {"snap-a": "snap-a", "snap-b": "snap-b"},
            "delete": [],
        }[command]

    def driver(*_):
        calls.append(("driver", None))
        if driver_error:
            raise driver_error

    monkeypatch.setattr(harbor_job, "snapshot_command", snapshots)
    monkeypatch.setattr(harbor_job, "run_harbor_driver", driver)
    overlay = SimpleNamespace(dataset_path=str(dataset), jobs_dir=str(tmp_path / "jobs"))
    return calls, lambda: harbor_job.run_harbor_job(None, overlay, {}, None, 45, tmp_path)


def test_harbor_job_deletes_only_snapshots_it_created_even_when_the_job_fails(tmp_path, monkeypatch):
    calls, run = _job(tmp_path, monkeypatch, preexisting=["snap-a"], non_general=44, driver_error=RuntimeError("x"))

    with pytest.raises(RuntimeError, match="x"):
        run()

    assert ("existing", ["snap-a", "snap-b"]) in calls
    # Snapshots are built before the trials start, so concurrent trials never race their creates.
    assert calls[-3:] == [("prepare", None), ("driver", None), ("delete", ["snap-b"])]


def test_harbor_job_deletes_created_snapshots_when_prepare_fails(tmp_path, monkeypatch):
    calls, run = _job(
        tmp_path, monkeypatch, preexisting=[], non_general=40, prepare_error=RuntimeError("quota exceeded")
    )

    with pytest.raises(RuntimeError, match="quota exceeded"):
        run()

    assert ("driver", None) not in calls
    assert calls[-1] == ("delete", ["snap-a", "snap-b"])


def test_harbor_job_refuses_to_start_without_a_free_slot_per_new_snapshot(tmp_path, monkeypatch):
    calls, run = _job(tmp_path, monkeypatch, preexisting=[], non_general=44)

    with pytest.raises(RuntimeError, match="2 new snapshots needed but only 1 of 45"):
        run()

    assert all(command not in ("driver", "delete") for command, _ in calls)
