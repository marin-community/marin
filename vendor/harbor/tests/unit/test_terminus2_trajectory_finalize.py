"""Contract tests for Terminus-2 canonical trajectory finalization.

A trial that recorded agent steps must leave a valid ``trajectory.json`` on
every exit path. The gap these tests lock down: when context summarization is
attempted but interrupted (the agent times out mid-summarization), the
summarization count was already incremented while the trajectory-split boundary
dump that writes ``trajectory.json`` never ran. The run()-finally dump then
wrote the full in-memory conversation to ``trajectory.cont-N.json`` instead of
the canonical ``trajectory.json``, so the durable output lost its SFT entry
point even though the conversation was captured.

The fix keys the final dump's continuation index off completed boundary dumps
(``_continuation_segments_persisted``), not attempted summarizations. With no
completed split the in-memory steps are still the full conversation and land in
``trajectory.json`` regardless of how many summarizations were attempted.
"""

import json

import pytest

from harbor.models.trajectories import Step
from harbor.utils.artifact_writer import get_artifact_writer
from tests.unit.agents.terminus_2.bare_agent import make_bare_terminus2


def _recorded_steps() -> list[Step]:
    return [
        Step(step_id=1, source="user", message="initial prompt"),
        Step(step_id=2, source="agent", message="agent response"),
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "segments_persisted, expected_file",
    [
        # No completed split: the finally dump lands the full in-memory
        # conversation in the canonical trajectory.json. This covers an
        # uninterrupted run AND a summarization cancelled before its boundary
        # dump — both leave _continuation_segments_persisted at 0.
        pytest.param(0, "trajectory.json", id="no-completed-split"),
        # One boundary dump completed: the finally dump targets the next
        # continuation segment, not the canonical file.
        pytest.param(1, "trajectory.cont-1.json", id="completed-split"),
    ],
)
def test_final_dump_segment_follows_completed_split_count(
    tmp_path,
    segments_persisted: int,
    expected_file: str,
) -> None:
    agent = make_bare_terminus2(
        tmp_path,
        _linear_history=True,
        _trajectory_steps=_recorded_steps(),
    )
    agent._continuation_segments_persisted = segments_persisted

    agent._persist_final_trajectory()
    get_artifact_writer().flush(prefix=str(tmp_path))

    written = json.loads((tmp_path / expected_file).read_text())
    assert [step["source"] for step in written["steps"]] == ["user", "agent"]
    # The final segment is the last one: it must not claim a continuation that
    # will never be written. A null ref is omitted from the JSON.
    assert "continued_trajectory_ref" not in written
