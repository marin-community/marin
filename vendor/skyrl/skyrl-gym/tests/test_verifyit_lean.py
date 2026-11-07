import pytest
from taskcompendium.grading_result import Outcome


@pytest.mark.asyncio
async def test_lean_session_empty_proof_allows_correction(nemotron_session, model_turn):
    session = await nemotron_session(
        "math_formal_lean_refinement_agent",
        {"header": "import Mathlib\n", "formal_statement": "example : True := by\n"},
    )
    result = await session.advance(model_turn("", stop_reason="length"))
    assert not result.done and result.reward == 0.0
    assert result.grade.status is Outcome.UNAVAILABLE
