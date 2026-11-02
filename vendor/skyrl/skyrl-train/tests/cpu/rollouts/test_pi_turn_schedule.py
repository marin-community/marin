"""Reproducible per-visited-turn execution independent of model feedback."""

import pytest
from pydantic import ValidationError

from skyrl_train.rollouts.pi_turn_schedule import PiTurnMixtureSchedule


def test_pi_turn_mixture_schedule_replays_the_lease_and_samples_each_turn():
    schedule = PiTurnMixtureSchedule.model_validate(
        {"teacher_execution_probability": {"scheduler": "linear", "start": 1.0, "end": 0.6, "duration_iterations": 2}}
    )
    probability = schedule.teacher_execution_probability
    assert [probability.value(i) for i in range(4)] == pytest.approx([1.0, 0.8, 0.6, 0.6])
    assert schedule.executors(trial_id="physical-0", policy_step=0, max_turns=40) == ("teacher",) * 40
    planned = schedule.executors(trial_id="physical-0", policy_step=2, max_turns=80)
    assert planned == schedule.executors(trial_id="physical-0", policy_step=2, max_turns=80)
    assert "teacher" in planned and "student" in planned
    assert planned != schedule.executors(trial_id="physical-1", policy_step=2, max_turns=80)
    with pytest.raises(ValueError, match="leased policy step"):
        schedule.executors(trial_id="physical-0", policy_step=-1, max_turns=80)
    with pytest.raises(ValueError, match="trial identity"):
        schedule.executors(trial_id="", policy_step=0, max_turns=80)


@pytest.mark.parametrize("start,end,duration", [(1.0, 1.2, 2), (0.2, 0.6, 2), (0.8, -0.1, 2), (1.0, 0.6, 0)])
def test_pi_turn_mixture_rejects_invalid_bounds(start, end, duration):
    with pytest.raises(ValidationError):
        PiTurnMixtureSchedule.model_validate(
            {
                "teacher_execution_probability": {
                    "scheduler": "linear",
                    "start": start,
                    "end": end,
                    "duration_iterations": duration,
                }
            }
        )
