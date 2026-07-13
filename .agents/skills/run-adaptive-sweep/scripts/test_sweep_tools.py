# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = runpy.run_path(Path(__file__).with_name("sweep_tools.py"))
check_convergence = TOOLS["check_convergence"]
predict_objectives = TOOLS["predict_objectives"]
rank_targets = TOOLS["rank_targets"]


def axis(values, *, domain=None):
    return {
        "name": "x",
        "values": values,
        "scale": "linear",
        "preferred_max_gap": 1,
        "domain": domain or {"min": values[0], "max": values[-1]},
    }


def convergence_request(values, trials, *, domain=None):
    return {
        "axes": [axis(values, domain=domain)],
        "resource_levels": [1],
        "objective": {"direction": "minimize"},
        "trials": trials,
    }


def completed(point, objective, rung=0):
    return {"rung": rung, "point": point, "objective": objective}


def test_strict_interior_neighbor_dominance_converges():
    result = check_convergence(
        convergence_request(
            [0, 1, 2],
            [completed([0], 1), completed([1], 0), completed([2], 1)],
        )
    )

    assert result["all_rungs_converged"]
    assert result["snapshots"][0]["dominant_points"] == [{"point": [1], "objective": 0.0}]


def test_missing_neighbor_cannot_produce_false_positive():
    result = check_convergence(
        convergence_request(
            [0, 1, 2],
            [completed([0], 1), completed([1], 0)],
        )
    )

    assert not result["all_rungs_converged"]


def test_better_neighbor_prevents_convergence():
    result = check_convergence(
        convergence_request(
            [0, 1, 2],
            [completed([0], 1), completed([1], 0.5), completed([2], 0)],
            domain={"min": 0, "max": 3},
        )
    )

    assert not result["snapshots"][0]["converged"]


def test_tied_neighbors_satisfy_non_strict_comparison():
    result = check_convergence(
        convergence_request(
            [0, 1, 2],
            [completed([0], 1), completed([1], 0), completed([2], 0)],
        )
    )

    assert result["snapshots"][0]["converged"]


def test_hard_domain_boundary_replaces_missing_side():
    result = check_convergence(
        convergence_request(
            [0, 1],
            [completed([0], 0), completed([1], 1)],
            domain={"min": 0, "max": 2},
        )
    )

    assert result["snapshots"][0]["dominant_points"] == [{"point": [0], "objective": 0.0}]


def test_extendable_grid_edge_cannot_converge():
    result = check_convergence(
        convergence_request(
            [0, 1],
            [completed([0], 1), completed([1], 0)],
            domain={"min": 0, "max": 2},
        )
    )

    assert not result["snapshots"][0]["converged"]


def test_near_domain_boundary_is_still_extendable():
    result = check_convergence(
        convergence_request(
            [0, 1],
            [completed([0], 1), completed([1], 0)],
            domain={"min": 0, "max": 1.000000000001},
        )
    )

    assert not result["snapshots"][0]["converged"]


def test_point_values_remain_stable_when_grid_expands_below_existing_values():
    before_expansion = check_convergence(
        convergence_request(
            [10, 20],
            [completed([10], 0), completed([20], 1)],
            domain={"min": 5, "max": 20},
        )
    )
    after_expansion = check_convergence(
        convergence_request(
            [5, 10, 20],
            [completed([5], 1), completed([10], 0), completed([20], 1)],
        )
    )

    assert not before_expansion["all_rungs_converged"]
    assert after_expansion["snapshots"][0]["dominant_points"] == [{"point": [10.0], "objective": 0.0}]


def test_axes_are_evaluated_jointly():
    trials = [
        completed([1, 1], 0),
        completed([0, 1], 1),
        completed([2, 1], 1),
        completed([1, 0], -1),
        completed([1, 2], 1),
    ]
    result = check_convergence(
        {
            "axes": [axis([0, 1, 2]), {**axis([0, 1, 2]), "name": "y"}],
            "resource_levels": [1],
            "objective": {"direction": "minimize"},
            "trials": trials,
        }
    )

    assert not result["snapshots"][0]["converged"]


def test_gradient_boosting_transfers_candidate_order_across_rungs():
    trials = [
        completed([0], 1.0, rung=0),
        completed([1], 0.0, rung=0),
        completed([2], 1.0, rung=0),
        completed([0], 0.8, rung=1),
        completed([1], -0.2, rung=1),
        completed([2], 0.8, rung=1),
    ]
    result = predict_objectives(
        {
            "axes": [axis([0, 1, 2])],
            "resource_levels": [1, 4, 16],
            "objective": {"direction": "minimize"},
            "trials": trials,
            "candidates": [
                {"candidate_id": "left", "rung": 2, "point": [0]},
                {"candidate_id": "center", "rung": 2, "point": [1]},
                {"candidate_id": "right", "rung": 2, "point": [2]},
            ],
        }
    )

    predictions = {prediction["candidate_id"]: prediction for prediction in result["predictions"]}
    assert predictions["center"]["rank_within_rung"] == 1
    assert predictions["center"]["predicted_objective"] < predictions["left"]["predicted_objective"]
    assert predictions["center"]["predicted_objective"] < predictions["right"]["predicted_objective"]


def test_prediction_is_unavailable_without_completed_trials():
    result = predict_objectives(
        {
            "axes": [axis([0, 1])],
            "resource_levels": [1],
            "objective": {"direction": "minimize"},
            "trials": [],
            "candidates": [{"candidate_id": "a", "rung": 0, "point": [0]}],
        }
    )

    assert result["status"] == "unavailable"
    assert result["predictions"][0]["predicted_objective"] is None


def observation(
    dispatch_id,
    target,
    rung,
    observed_at,
    progress,
    *,
    state="running",
    submitted_at=0,
    regional_run_id=None,
    trial_id=None,
    wandb_run_id=None,
):
    return {
        "dispatch_id": dispatch_id,
        "regional_run_id": regional_run_id or dispatch_id,
        "trial_id": trial_id or dispatch_id,
        "rung": rung,
        "target": target,
        "state": state,
        "submitted_at": submitted_at,
        "observed_at": observed_at,
        "wandb_run_id": wandb_run_id,
        "run_progress": progress,
    }


def target_request(observations, *, now=10, current_rung=0, max_chips=64, wall_time_limit=20):
    return {
        "now": now,
        "resource_levels": [1, 4, 16],
        "resource_ratios": [1, 4, 16],
        "wall_time_limit": wall_time_limit,
        "max_inflight_chips": max_chips,
        "current_rung": current_rung,
        "full_exploitation_rung": 2,
        "stagnation": {
            "initial_wandb_timeout": 1,
            "progress_stall_timeout": 4,
            "cross_region_restart_timeout": 8,
        },
        "targets": [
            {"target": "small", "region": "east", "tpu_slice": "v5p-8", "chips": 4},
            {"target": "large", "region": "east", "tpu_slice": "v5p-64", "chips": 32},
            {"target": "central", "region": "central", "tpu_slice": "v5p-32", "chips": 16},
        ],
        "observations": observations,
    }


def test_throughput_normalizes_progress_across_resource_rungs():
    observations = [
        observation("small-job", "small", 0, 0, 0),
        observation("small-job", "small", 0, 10, 1, state="succeeded"),
        observation("large-job", "large", 1, 0, 0),
        observation("large-job", "large", 1, 10, 0.25, state="succeeded"),
    ]
    result = rank_targets(target_request(observations))
    targets = {target["target"]: target for target in result["targets"]}

    assert targets["small"]["normalized_progress_throughput"] == pytest.approx(0.1)
    assert targets["large"]["normalized_progress_throughput"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("rung", "expected_fraction", "expected_depth"),
    [(0, 1.0, 3), (1, 0.5, 2), (2, 0.0, 1)],
)
def test_target_exploration_decays_to_one_target_at_selected_rung(rung, expected_fraction, expected_depth):
    result = rank_targets(target_request([], current_rung=rung))

    assert result["exploration_fraction"] == expected_fraction
    assert result["exploration_depth"] == expected_depth
    assert len(result["selection_pool"]) == expected_depth


def test_stalled_target_throughput_falls_below_progressing_target():
    observations = [
        observation("stalled", "small", 0, 0, 0),
        observation("stalled", "small", 0, 5, 0.5),
        observation("stalled", "small", 0, 15, 0.5),
        observation("progressing", "large", 0, 0, 0),
        observation("progressing", "large", 0, 15, 1),
    ]
    result = rank_targets(target_request(observations, now=15))

    assert result["targets"][0]["target"] == "large"


def test_throughput_uses_wall_time_cosine_decay():
    observations = [
        observation("old-burst", "small", 0, 0, 0),
        observation("old-burst", "small", 0, 5, 1),
        observation("old-burst", "small", 0, 10, 1, state="succeeded"),
        observation("steady", "large", 0, 0, 0),
        observation("steady", "large", 0, 10, 0.5, state="succeeded"),
    ]
    result = rank_targets(target_request(observations, now=10, wall_time_limit=10))
    targets = {target["target"]: target for target in result["targets"]}

    assert 0 < targets["small"]["normalized_progress_throughput"] < 0.05
    assert targets["large"]["normalized_progress_throughput"] == pytest.approx(0.05)
    assert result["targets"][0]["target"] == "large"


def test_throughput_evidence_reaches_zero_at_wall_time_limit():
    observations = [
        observation("expired", "small", 0, -10, 0, submitted_at=-10),
        observation("expired", "small", 0, 0, 1, state="succeeded", submitted_at=-10),
    ]
    result = rank_targets(target_request(observations, now=10, wall_time_limit=10))
    target = next(target for target in result["targets"] if target["target"] == "small")

    assert target["normalized_progress_throughput"] is None


def test_throughput_weighting_is_invariant_to_polling_frequency():
    sparse = [
        observation("job", "small", 0, 0, 0),
        observation("job", "small", 0, 10, 1, state="succeeded"),
    ]
    frequent = [
        observation("job", "small", 0, time, time / 10, state="succeeded" if time == 10 else "running")
        for time in (0, 2, 5, 7, 10)
    ]

    sparse_result = rank_targets(target_request(sparse, now=10, wall_time_limit=10))
    frequent_result = rank_targets(target_request(frequent, now=10, wall_time_limit=10))
    sparse_rate = next(target for target in sparse_result["targets"] if target["target"] == "small")
    frequent_rate = next(target for target in frequent_result["targets"] if target["target"] == "small")

    assert sparse_rate["normalized_progress_throughput"] == pytest.approx(0.1)
    assert frequent_rate["normalized_progress_throughput"] == pytest.approx(0.1)


def test_one_queued_dispatch_makes_target_unattractive_for_more_work():
    observations = [
        observation("small-running", "small", 0, 0, 0),
        observation("small-running", "small", 0, 9, 0.9, state="succeeded"),
        observation("small-queued", "small", 0, 10, None, state="submitted", submitted_at=10),
        observation("large-running", "large", 0, 0, 0),
        observation("large-running", "large", 0, 10, 0.5, state="succeeded"),
    ]
    result = rank_targets(target_request(observations))
    targets = {target["target"]: target for target in result["targets"]}

    assert targets["small"]["mean_normalized_progress_throughput"] == pytest.approx(0.1)
    assert targets["small"]["normalized_progress_throughput"] == 0
    assert result["targets"][0]["target"] == "large"


def test_unknown_progress_is_accepted_before_wandb_registration():
    observations = [
        observation("queued", "small", 0, 2, None, state="submitted", submitted_at=2),
    ]

    result = rank_targets(target_request(observations, now=2))

    assert result["current_inflight_chips"] == 4
    assert result["stagnation"][0]["condition"] == "awaiting_wandb"


def test_unknown_progress_is_accepted_after_wandb_registration():
    observations = [
        observation("starting", "small", 0, 0, None, wandb_run_id="wb"),
        observation("starting", "small", 0, 4, None, wandb_run_id="wb"),
    ]

    result = rank_targets(target_request(observations, now=4))

    assert result["stagnation"][0]["condition"] == "wandb_registered"
    assert result["stagnation"][0]["eligible_action"] == "stalled_same_region_move"


def test_chip_budget_filters_selection_pool():
    observations = [
        observation("occupied", "large", 0, 0, 0),
        observation("occupied", "large", 0, 1, 0),
    ]
    result = rank_targets(target_request(observations, now=1, max_chips=36))

    assert result["available_chips"] == 4
    assert result["selection_pool"] == ["small"]


@pytest.mark.parametrize(
    ("now", "observations", "expected_action"),
    [
        (
            2,
            [
                observation("d1", "small", 0, 0, 0, regional_run_id="run"),
                observation("d1", "small", 0, 2, 0, regional_run_id="run"),
            ],
            "initial_same_region_move",
        ),
        (
            5,
            [
                observation("d1", "small", 0, 0, 0, regional_run_id="run", wandb_run_id="wb"),
                observation("d1", "small", 0, 5, 0, regional_run_id="run", wandb_run_id="wb"),
            ],
            "stalled_same_region_move",
        ),
        (
            9,
            [
                observation("d1", "small", 0, 0, 0, regional_run_id="run", wandb_run_id="wb"),
                observation("d1", "small", 0, 5, 0, state="stopped", regional_run_id="run", wandb_run_id="wb"),
                observation(
                    "d2",
                    "large",
                    0,
                    5,
                    0,
                    submitted_at=5,
                    regional_run_id="run",
                    wandb_run_id="wb",
                ),
                observation(
                    "d2",
                    "large",
                    0,
                    9,
                    0,
                    submitted_at=5,
                    regional_run_id="run",
                    wandb_run_id="wb",
                ),
            ],
            "cross_region_restart",
        ),
    ],
)
def test_stagnation_stages(now, observations, expected_action):
    result = rank_targets(target_request(observations, now=now))

    assert result["stagnation"][0]["eligible_action"] == expected_action


def test_cross_region_restart_waits_for_replacement_dispatch_to_stall():
    observations = [
        observation("d1", "small", 0, 0, 0, regional_run_id="run", wandb_run_id="wb"),
        observation("d1", "small", 0, 7, 0, state="stopped", regional_run_id="run", wandb_run_id="wb"),
        observation(
            "d2",
            "large",
            0,
            7,
            0,
            submitted_at=7,
            regional_run_id="run",
            wandb_run_id="wb",
        ),
        observation(
            "d2",
            "large",
            0,
            9,
            0,
            submitted_at=7,
            regional_run_id="run",
            wandb_run_id="wb",
        ),
    ]

    result = rank_targets(target_request(observations, now=9))

    assert result["stagnation"][0]["eligible_action"] == "observe"


def test_missing_wandb_can_restart_cross_region_after_failed_same_region_move():
    observations = [
        observation("d1", "small", 0, 0, None, state="stopped", regional_run_id="run"),
        observation("d2", "large", 0, 1, None, submitted_at=1, regional_run_id="run"),
        observation("d2", "large", 0, 9, None, submitted_at=1, regional_run_id="run"),
    ]

    result = rank_targets(target_request(observations, now=9))

    stagnation = result["stagnation"][0]
    assert stagnation["condition"] == "awaiting_wandb"
    assert stagnation["eligible_action"] == "cross_region_restart"
    assert stagnation["eligible_targets"] == ["central"]


def test_same_region_move_is_not_advised_without_an_alternative_slice():
    observations = [
        observation("d1", "central", 0, 0, 0, regional_run_id="run"),
        observation("d1", "central", 0, 2, 0, regional_run_id="run"),
    ]
    result = rank_targets(target_request(observations, now=2))

    stagnation = result["stagnation"][0]
    assert stagnation["eligible_action"] == "observe"
    assert stagnation["eligible_targets"] == []
    assert stagnation["blocked_reason"] == "no chip-feasible alternate target in region"


def test_observed_progress_resets_stall_timeout():
    observations = [
        observation("d1", "small", 0, 0, 0, regional_run_id="run", wandb_run_id="wb"),
        observation("d1", "small", 0, 3, 0.5, regional_run_id="run", wandb_run_id="wb"),
        observation("d1", "small", 0, 6, 0.5, regional_run_id="run", wandb_run_id="wb"),
    ]
    result = rank_targets(target_request(observations, now=6))

    stagnation = result["stagnation"][0]
    assert stagnation["eligible_action"] == "observe"
    assert stagnation["inactive_for"] == 3


def test_stalled_same_region_recovery_can_repeat_before_cross_region_timeout():
    observations = [
        observation("d1", "small", 0, 0, 0, regional_run_id="run", wandb_run_id="wb"),
        observation("d1", "small", 0, 4, 0, state="stopped", regional_run_id="run", wandb_run_id="wb"),
        observation("d2", "large", 0, 4, 0, submitted_at=4, regional_run_id="run", wandb_run_id="wb"),
        observation("d2", "large", 0, 9, 0, submitted_at=4, regional_run_id="run", wandb_run_id="wb"),
    ]
    request = target_request(observations, now=9)
    request["stagnation"]["cross_region_restart_timeout"] = 12
    result = rank_targets(request)

    assert result["stagnation"][0]["eligible_action"] == "stalled_same_region_move"


def test_initial_recovery_can_repeat_without_wandb_registration():
    observations = [
        observation("d1", "small", 0, 0, 0, state="stopped", regional_run_id="run"),
        observation("d2", "large", 0, 1, 0, submitted_at=1, regional_run_id="run"),
        observation("d2", "large", 0, 2, 0, submitted_at=1, regional_run_id="run"),
    ]
    result = rank_targets(target_request(observations, now=2))

    stagnation = result["stagnation"][0]
    assert stagnation["condition"] == "awaiting_wandb"
    assert stagnation["eligible_action"] == "initial_same_region_move"


def test_relocation_targets_use_chips_released_by_current_dispatch():
    observations = [
        observation("d1", "large", 0, 0, 0, regional_run_id="run"),
        observation("d1", "large", 0, 2, 0, regional_run_id="run"),
    ]
    result = rank_targets(target_request(observations, now=2, max_chips=32))

    assert result["available_chips"] == 0
    assert result["selection_pool"] == []
    assert result["stagnation"][0]["eligible_targets"] == ["small"]


def test_json_cli_round_trip(tmp_path):
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            convergence_request(
                [0, 1, 2],
                [completed([0], 1), completed([1], 0), completed([2], 1)],
            )
        )
    )
    completed_process = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("sweep_tools.py")), "check-convergence", str(request_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed_process.stdout)["all_rungs_converged"]
