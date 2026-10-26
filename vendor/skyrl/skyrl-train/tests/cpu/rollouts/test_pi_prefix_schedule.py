"""One frozen student-prefix plan per physical trial and leased policy step."""

import pytest
from pydantic import ValidationError

from skyrl_train.rollouts.pi_prefix_schedule import PiPrefixSchedule


def schedule(**overrides):
    raw = {
        "distribution": "uniform_int",
        "min": 0,
        "max": {"scheduler": "linear", "start": 0, "end": 4, "duration_iterations": 4},
    }
    raw.update(overrides)
    return PiPrefixSchedule.model_validate(raw)


def test_prefix_schedule_is_clamped_and_retry_is_deterministic():
    config = schedule()
    assert [config.max.value(index) for index in range(7)] == [0, 1, 2, 3, 4, 4, 4]
    assert config.executors(trial_id="task_0", policy_step=0, max_turns=4) == ("teacher",) * 4
    for step in range(1, 12):
        first = config.executors(trial_id="task_0", policy_step=step, max_turns=4)
        assert first == config.executors(trial_id="task_0", policy_step=step, max_turns=4)
        assert first == ("student",) * first.count("student") + ("teacher",) * first.count("teacher")
        assert len(first) == 4
    assert len({config.sample(trial_id=f"task_{n}", policy_step=4, max_turns=4) for n in range(50)}) > 1


@pytest.mark.parametrize(
    "bad",
    [
        {"scheduler": "cosine", "start": 0, "end": 4, "duration_iterations": 4},
        {"scheduler": "linear", "start": -1, "end": 4, "duration_iterations": 4},
        {"scheduler": "linear", "start": 3, "end": 1, "duration_iterations": 4},
        {"scheduler": "linear", "start": 0, "end": 4, "duration_iterations": 0},
        {"scheduler": "linear", "start": 0, "end": 4, "duration_iterations": True},
        {"scheduler": "linear", "start": 0, "end": 4, "duration_iterations": 4, "unsafe": 1},
    ],
)
def test_prefix_schedule_rejects_invalid_parameters(bad):
    with pytest.raises(ValidationError):
        schedule(max=bad)


def test_prefix_schedule_rejects_out_of_budget_or_unleased_sampling():
    config = schedule()
    with pytest.raises(ValueError, match="turn limit"):
        config.sample(trial_id="task_0", policy_step=4, max_turns=3)
    with pytest.raises(ValueError, match="policy step"):
        config.sample(trial_id="task_0", policy_step=-1, max_turns=4)
    with pytest.raises(ValueError, match="turn limit"):
        config.sample(trial_id="", policy_step=4, max_turns=4)
