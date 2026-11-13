# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

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
