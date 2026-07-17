# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SIMULATOR = Path(__file__).with_name("trc_simulator.py")


def run_simulator(*args: object, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SIMULATOR), *(str(arg) for arg in args)],
        check=check,
        capture_output=True,
        text=True,
    )


def output(process: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(process.stdout)


def initialize(
    tmp_path: Path,
    name: str = "state.json",
    seed: int = 7,
    full_exploitation_level: int = 64,
    scenario: str = "stable-interior",
    time_horizon: float = 504,
) -> Path:
    state = tmp_path / name
    run_simulator(
        "init",
        state,
        "--seed",
        seed,
        "--full-exploitation-level",
        full_exploitation_level,
        "--scenario",
        scenario,
        "--time-horizon",
        time_horizon,
    )
    return state


def launch(state: Path, *, rung: int, point: list[int], target: str) -> dict:
    actions = json.dumps([{"rung": rung, "point": point, "target": target}])
    return output(run_simulator("launch", state, actions))


def test_seed_reproduces_public_execution(tmp_path):
    first = initialize(tmp_path, "first.json", seed=19)
    second = initialize(tmp_path, "second.json", seed=19)
    for state in (first, second):
        launch(state, rung=0, point=[2, 2], target="us-east5/v5p-32")
        run_simulator("advance", state, "--steps", 5)

    assert output(run_simulator("status", first)) == output(run_simulator("status", second))


def test_report_rejects_unconverged_state(tmp_path):
    state = initialize(tmp_path)

    result = run_simulator("report", state, check=False)

    assert result.returncode == 2
    assert set(json.loads(result.stderr)) == {"error"}


def test_launch_rejects_duplicate_and_over_budget_work_without_mutation(tmp_path):
    state = initialize(tmp_path)
    launch(state, rung=0, point=[0, 0], target="us-east5/v5p-32")

    duplicate = run_simulator(
        "launch",
        state,
        json.dumps([{"rung": 0, "point": [0, 0], "target": "us-central1/v5p-32"}]),
        check=False,
    )
    over_budget = run_simulator(
        "launch",
        state,
        json.dumps([{"rung": 0, "point": [1, 0], "target": "us-east5/v5p-128"}]),
        check=False,
    )
    status = output(run_simulator("status", state))

    assert duplicate.returncode == 2
    assert over_budget.returncode == 2
    assert status["inflight_chips"] == 16
    assert [(trial["rung"], trial["point"]) for trial in status["active"]] == [(0, [0, 0])]


def test_wandb_registration_distinguishes_queueing_from_execution(tmp_path):
    queued = initialize(tmp_path, "queued.json")
    launch(queued, rung=0, point=[0, 0], target="us-east5/v5p-128")
    run_simulator("advance", queued, "--steps", 4)

    running = initialize(tmp_path, "running.json")
    launch(running, rung=0, point=[0, 0], target="us-east5/v5p-32")
    run_simulator("advance", running, "--steps", 1)

    queued_recovery = output(run_simulator("targets", queued, 0))["recovery"][0]
    running_recovery = output(run_simulator("targets", running, 0))["recovery"][0]

    assert queued_recovery["condition"] == "awaiting_wandb"
    assert queued_recovery["eligible_action"] == "startup_relocation"
    assert running_recovery["condition"] == "wandb_registered"
    assert running_recovery["eligible_action"] == "observe"


def test_region_moves_preserve_local_state_and_reset_new_region(tmp_path):
    state = initialize(tmp_path)
    launched = launch(state, rung=0, point=[0, 0], target="us-east5/v5p-32")
    trial_id = launched["launched"][0]["trial_id"]
    run_simulator("advance", state, "--steps", 1)
    east_progress = output(run_simulator("status", state))["active"][0]["progress"]

    same_region = output(
        run_simulator(
            "relocate",
            state,
            json.dumps({"trial_id": trial_id, "target": "us-east5/v5p-64"}),
        )
    )
    cross_region = output(
        run_simulator(
            "relocate",
            state,
            json.dumps({"trial_id": trial_id, "target": "us-central1/v5p-32"}),
        )
    )
    returned = output(
        run_simulator(
            "relocate",
            state,
            json.dumps({"trial_id": trial_id, "target": "us-east5/v5p-32"}),
        )
    )

    assert same_region["progress"] == east_progress
    assert cross_region["progress"] == 0
    assert cross_region["regional_run_id"] != same_region["regional_run_id"]
    assert returned["progress"] == east_progress
    assert returned["regional_run_id"] == same_region["regional_run_id"]


def test_retryable_failure_preserves_progress_and_logical_trial(tmp_path):
    state = initialize(tmp_path)
    launched = launch(state, rung=0, point=[2, 2], target="us-east5/v5p-32")["launched"][0]

    advance = output(run_simulator("advance", state, "--steps", 20))

    raw_state = json.loads(state.read_text())
    retry = raw_state["retry_events"][0]
    completed = next(trial for trial in raw_state["completed"] if trial["trial_id"] == launched["trial_id"])
    assert retry["progress"] > 0
    assert retry["dispatch_id"] == launched["dispatch_id"]
    assert retry["target"] == launched["target"]
    assert completed["dispatch_id"] == launched["dispatch_id"]
    assert completed["retry_count"] >= 1
    assert completed["objective"] is not None
    assert advance["retry_events"][0] == retry
    assert next(item for item in advance["completed"] if item["trial_id"] == launched["trial_id"])["retry_count"] >= 1


def test_trc_profiles_have_explicit_constant_and_linear_throughput(tmp_path):
    state = initialize(tmp_path, time_horizon=100)
    truth = json.loads(state.read_text())["truth"]["trc"]
    profiles = truth["targets"]

    constant = [profile for profile in profiles.values() if profile["start_rate"] == profile["end_rate"]]
    linear = [profile for profile in profiles.values() if profile["start_rate"] != profile["end_rate"]]
    assert truth["time_horizon"] == 100
    assert len(constant) == 2
    assert len(linear) == 3


def test_region_outage_stalls_every_slice_in_region(tmp_path):
    state = initialize(tmp_path, time_horizon=100)
    raw_state = json.loads(state.read_text())
    raw_state["now"] = 46
    state.write_text(json.dumps(raw_state))
    launch(state, rung=0, point=[0, 0], target="us-east5/v5p-32")
    launch(state, rung=0, point=[1, 0], target="us-east5/v5p-64")

    stalled = output(run_simulator("advance", state))
    raw_state = json.loads(state.read_text())
    raw_state["now"] = 51
    state.write_text(json.dumps(raw_state))
    resumed = output(run_simulator("advance", state))

    assert all(trial["progress"] == 0 for trial in stalled["active"])
    assert all(trial["progress"] > 0 for trial in resumed["active"])


def test_full_exploitation_level_controls_target_exploration_depth(tmp_path):
    early = initialize(tmp_path, "early.json", full_exploitation_level=16)
    late = initialize(tmp_path, "late.json", full_exploitation_level=64)

    early_depths = [output(run_simulator("targets", early, rung))["exploration_depth"] for rung in range(4)]
    late_depths = [output(run_simulator("targets", late, rung))["exploration_depth"] for rung in range(4)]

    assert early_depths == [5, 3, 1, 1]
    assert late_depths == [5, 3, 2, 1]


def test_predictable_off_grid_scenario_requires_ordered_grid_expansion(tmp_path):
    state = initialize(tmp_path, scenario="predictable-off-grid")
    initial = json.loads(state.read_text())

    assert initial["axes"][0]["values"] == [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    assert initial["truth"]["experiment"]["centers"] == [[1, 3], [3, 3], [5, 3], [6, 3]]

    outside = run_simulator(
        "launch",
        state,
        json.dumps([{"rung": 2, "point": [5, 3], "target": "us-east5/v5p-32"}]),
        check=False,
    )
    skipped = run_simulator(
        "expand-grid",
        state,
        json.dumps({"axis": "learning_rate", "value": 1e-2}),
        check=False,
    )
    first = output(
        run_simulator(
            "expand-grid",
            state,
            json.dumps({"axis": "learning_rate", "value": 3e-3}),
        )
    )
    launched = launch(state, rung=2, point=[5, 3], target="us-east5/v5p-32")
    second = output(
        run_simulator(
            "expand-grid",
            state,
            json.dumps({"axis": "learning_rate", "value": 1e-2}),
        )
    )

    assert outside.returncode == 2
    assert skipped.returncode == 2
    assert first["axes"][0]["values"][-1] == 3e-3
    assert launched["launched"][0]["point"] == [5, 3]
    assert second["axes"][0]["values"][-1] == 1e-2


def test_grid_expansion_allows_recorded_resolution_deviation(tmp_path):
    state = initialize(tmp_path, scenario="predictable-off-grid")

    expanded = output(
        run_simulator(
            "expand-grid",
            state,
            json.dumps(
                {
                    "axis": "learning_rate",
                    "value": 1e-2,
                    "reason": "test a predicted endpoint before the intermediate value",
                }
            ),
        )
    )
    launched = launch(state, rung=3, point=[5, 3], target="us-east5/v5p-32")

    assert expanded["deviation_reason"] == "test a predicted endpoint before the intermediate value"
    assert expanded["axes"][0]["values"][-1] == 1e-2
    assert launched["launched"][0]["point"] == [5, 3]
