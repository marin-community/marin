import pytest
from taskcompendium.grading_result import Outcome
from skyrl_gym.task_records import grade_skipped


@pytest.mark.asyncio
async def test_genrm_session_returns_pending_cohort_reward(nemotron_session, model_turn):
    session = await nemotron_session("genrm_simple_agent", {}, {"genrm": {"default_score": 3.5}})
    result = await session.advance(model_turn("<think>partial", stop_reason="length"))
    assert result.done and result.reward == 3.5


@pytest.mark.parametrize(
    "agent,text,stop_reason,grading",
    [
        ("instruction_following_simple_agent", "<think>B or C?</think>", "stop", "verify"),
        ("mcqa_simple_agent", r"\boxed{C}", "length", "verify"),
        ("instruction_following_simple_agent", "", "stop", "skip"),
    ],
)
@pytest.mark.asyncio
async def test_unanswered_or_length_stopped_ultra_response_scores_zero(
    nemotron_session, model_turn, agent, text, stop_reason, grading
):
    record = (
        {"instruction_id_list": ["keywords:forbidden_words"], "kwargs": [{"forbidden_words": ["banana"]}]}
        if agent == "instruction_following_simple_agent"
        else {"expected_answer": "C", "options": [{"A": "one"}, {"B": "two"}, {"C": "three"}]}
    )
    session = await nemotron_session(agent, record, {"grading": grading})
    result = await session.advance(model_turn(text, stop_reason=stop_reason))
    assert result.done and result.reward == 0.0
    if grading == "skip":
        assert grade_skipped(result.grade) and result.metrics["graded"] == 0.0
    else:
        assert result.grade.status is Outcome.GRADED and result.grade.reward == 0.0
        assert result.metrics["graded"] == 1.0
