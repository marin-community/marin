# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validation writes the controls from the grader's stored output, then replays them."""

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError
from rigging.timing import ExponentialBackoff
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, Command, MachineSpec, PackageSpec
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, AssistantToolCalls, PlainText, Source
from verifyit.spec import ExactSpec

from taskforge.ledger.jsonl import JsonlLedger
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger
from taskforge.sandbox.factories import MachineHost
from taskforge.spec import draft
from taskforge.spec.controls import Transcript
from taskforge.stage10_proposal.model import parse
from taskforge.stage40_build.context import BuildFailure
from taskforge.stage40_build.outputs import GradedCandidate, Grader
from taskforge.stage50_validate.control_drafts import (
    CONTROLS_FILE,
    ControlDraft,
    control_payload,
    load_controls,
    output_paths_text,
    shipped_output_paths,
    write_controls,
)
from taskforge.stage50_validate.controls import ControlPlan, ControlVerdict, replay
from taskforge.stage50_validate.lowering import lower
from taskforge.stage50_validate.trials import Deadlines, EngineSettings
from tests.sandbox.fixture_images import FixtureImageFactory, PortlessShellSimFactory
from tests.stage40_build.conftest import FACTORIES, PROPOSAL

TASK = draft.assemble(
    "item",
    "What is 6 * 7? Reply with the number alone.",
    AnswerType.TEXT,
    PlainText(),
    draft.answer_grader(ExactSpec(expected=("42",))),
    Source(dataset="test", revision="r", row="0", importer_revision="test"),
    environment=draft.requirements(image=None),
)
GRADER = Grader(
    package=draft.answer_grader(ExactSpec(expected=("42",))),
    answer_contract="Reply with the number alone.",
    reference_reply="42",
    reference_files=(),
    secret_values=("42",),
    rejection_ceiling=0.0,
    prototyped=(
        GradedCandidate("42", (), Outcome.GRADED, 1.0),
        GradedCandidate("", (), Outcome.GRADED, 0.0),
        GradedCandidate("40", (), Outcome.GRADED, 0.0),
    ),
)


def control(control_id: str, kind: str, category: str, concern: str, reply: str, low, high) -> dict:
    return {
        "id": control_id,
        "kind": kind,
        "category": category,
        "concern": concern,
        "final_reply": reply,
        "files": [],
        "reward_min": low,
        "reward_max": high,
        "rationale": "r",
    }


def controls(wrong: str) -> str:
    return json.dumps(
        {
            "controls": [
                control("P-reference", "positive", "known_correct", "reference", "42", 0.99, None),
                control("N1-shortcut", "negative", "task_specific_shortcut", "shortcut", "6 * 7", None, 0.0),
                control("N2-off-by-two", "negative", "plausible_wrong", "acceptance", wrong, None, 0.0),
                control("N3", "malformed", "empty_or_malformed", "extraction", "", None, 0.0),
            ]
        }
    )


def engine_settings() -> EngineSettings:
    return EngineSettings(
        factories={Backend.SHELLSIM.value: PortlessShellSimFactory(), Backend.DOCKER.value: FixtureImageFactory()},
        max_turns=4,
        command_timeout=10,
        tool_turn_timeout=20,
        model_turn_timeout=30,
        cleanup_timeout=10,
    )


async def test_validation_writes_controls_the_grader_has_not_seen_and_replays_them(
    glm_client, fake_glm, ledger, tmp_path
):
    # "40" is a candidate the grader step graded, so the first set goes back to the model.
    fake_glm.stream(content=controls("40"))
    fake_glm.stream(content=controls("41"))

    written = await write_controls(
        TASK, GRADER, parse(PROPOSAL), glm_client, LLMPolicy(), CallLedger(ledger, "item", 0, "controls"), tmp_path
    )

    assert (
        "controls ['N2-off-by-two'] are candidates the grader step graded"
        in fake_glm.requests[1]["messages"][-2]["content"]
    )
    assert load_controls(tmp_path) == written
    assert [c.id for c in written] == ["P-reference", "N1-shortcut", "N2-off-by-two", "N3"]
    plan = ControlPlan(
        item_id="item",
        round=0,
        deadlines=Deadlines(total_turn_timeout=30, attempt_timeout=60),
        max_retries=0,
        retry_backoff=ExponentialBackoff(initial=0.001, maximum=0.001),
        evidence_dir=tmp_path / "evidence",
        ledger=JsonlLedger(tmp_path / "ledger"),
        first_attempts={},
    )
    outcomes = await replay(
        lower(TASK, MachineHost.LAPTOP, FACTORIES), written, plan, engine_settings()
    )
    assert {o.control.id: o.verdict for o in outcomes} == {c.id: ControlVerdict.MET for c in written}


async def test_a_control_set_the_model_never_gets_right_fails_as_the_controls_step(
    glm_client, fake_glm, ledger, tmp_path
):
    for _ in range(3):
        fake_glm.stream(content=controls("40"))

    with pytest.raises(BuildFailure) as failure:
        await write_controls(
            TASK, GRADER, parse(PROPOSAL), glm_client, LLMPolicy(), CallLedger(ledger, "item", 0, "controls"), tmp_path
        )

    assert failure.value.step == "controls"
    assert not (tmp_path / CONTROLS_FILE).exists()


async def test_control_bounds_follow_the_graders_rejection_ceiling(glm_client, fake_glm, ledger, tmp_path):
    """A grader with partial credit lets a negative control demand up to its ceiling and no more."""
    lenient = replace(GRADER, rejection_ceiling=0.4)
    over = json.loads(controls("41"))
    over["controls"][1]["reward_max"] = 0.5
    within = json.loads(controls("41"))
    within["controls"][1]["reward_max"] = 0.4
    fake_glm.stream(content=json.dumps(over))
    fake_glm.stream(content=json.dumps(within))

    written = await write_controls(
        TASK, lenient, parse(PROPOSAL), glm_client, LLMPolicy(), CallLedger(ledger, "item", 0, "controls"), tmp_path
    )

    request = fake_glm.requests[0]["messages"][-2]["content"]
    assert "lets a submission wrong in substance score up to 0.4" in request
    assert "wrong in substance" in request and "follows the answer contract exactly" in request
    repair = fake_glm.requests[1]["messages"][-2]["content"]
    assert "N1-shortcut must demand a reward at most the grader's rejection ceiling 0.4, not 0.5" in repair
    assert written[1].expect.reward_max == 0.4


async def test_a_controls_reply_the_output_limit_cut_is_asked_again_shorter(glm_client, fake_glm, ledger, tmp_path):
    """A cut reply is one failed submission, not an unhandled error ending the item."""
    fake_glm.stream(content='{"controls": [{"id": "P-reference", "kind": "positive"', finish="length")
    fake_glm.stream(content=controls("41"))

    written = await write_controls(
        TASK, GRADER, parse(PROPOSAL), glm_client, LLMPolicy(), CallLedger(ledger, "item", 0, "controls"), tmp_path
    )

    retry = fake_glm.requests[1]["messages"]
    assert retry[-3]["role"] == "assistant" and retry[-3]["content"].endswith("...[cut by the output limit]")
    assert "the reply was cut off by the output limit" in retry[-2]["content"]
    assert [c.id for c in written] == ["P-reference", "N1-shortcut", "N2-off-by-two", "N3"]


async def test_controls_cut_at_every_attempt_fail_as_the_controls_step(glm_client, fake_glm, ledger, tmp_path):
    for _ in range(3):
        fake_glm.stream(content='{"controls": [', finish="length")

    with pytest.raises(BuildFailure, match="cut off by the output limit") as failure:
        await write_controls(
            TASK, GRADER, parse(PROPOSAL), glm_client, LLMPolicy(), CallLedger(ledger, "item", 0, "controls"), tmp_path
        )

    assert failure.value.step == "controls"


async def test_a_control_writing_files_outside_the_output_paths_goes_back_to_the_model(
    glm_client, fake_glm, ledger, tmp_path
):
    written_out = json.loads(controls("41"))
    written_out["controls"][0]["files"] = [{"path": "/workspace/answer.txt", "content": "42\n"}]
    fake_glm.stream(content=json.dumps(written_out))
    fake_glm.stream(content=controls("41"))

    written = await write_controls(
        TASK, GRADER, parse(PROPOSAL), glm_client, LLMPolicy(), CallLedger(ledger, "item", 0, "controls"), tmp_path
    )

    assert "the task has no output paths" in fake_glm.requests[0]["messages"][-2]["content"].lower()
    repair = fake_glm.requests[1]["messages"][-2]["content"]
    assert "controls ['P-reference'] write files outside the task's output paths" in repair
    assert [c.id for c in written] == ["P-reference", "N1-shortcut", "N2-off-by-two", "N3"]


async def test_a_control_writes_an_output_path_outside_the_workspace(glm_client, fake_glm, ledger, tmp_path):
    report = "/snapshot/report/x.sql"
    task = draft.assemble(
        "item",
        "What is 6 * 7? Write a query returning it to /snapshot/report/x.sql, then reply with the number alone.",
        AnswerType.TEXT,
        PlainText(),
        draft.answer_grader(ExactSpec(expected=("42",))),
        Source(dataset="test", revision="r", row="0", importer_revision="test"),
        environment=draft.requirements(image=None),
        output_paths=(report,),
    )
    submitted = json.loads(controls("41"))
    submitted["controls"][0]["files"] = [{"path": report, "content": "select 42;\n"}]
    fake_glm.stream(content=json.dumps(submitted))

    written = await write_controls(
        task, GRADER, parse(PROPOSAL), glm_client, LLMPolicy(), CallLedger(ledger, "item", 0, "controls"), tmp_path
    )

    assert report in fake_glm.requests[0]["messages"][-2]["content"]
    assert len(fake_glm.requests) == 1
    payload = written[0].payload
    assert isinstance(payload, Transcript)
    turn = payload.turns[0]
    assert isinstance(turn, AssistantToolCalls)
    assert f"> {report}" in str(turn.calls[0].object_arguments()["command"])


def test_a_relative_control_path_is_refused_as_not_an_output_path():
    submitted = {
        **control("relative", "positive", "known_correct", "reference", "42", 1.0, None),
        "files": [{"path": "snapshot/report/x.sql", "content": "select 42;\n"}],
    }

    with pytest.raises(ValidationError) as refusal:
        ControlDraft.model_validate(submitted)

    assert "the control file path 'snapshot/report/x.sql' is relative" in str(refusal.value)
    assert "control files are the task's output paths, spelled as absolute paths" in str(refusal.value)


async def test_control_files_with_shell_metacharacters_in_their_paths_are_written_verbatim():
    path = "/workspace/it's $HOME/a b.txt"
    submitted = ControlDraft.model_validate(
        {
            **control("quoted", "positive", "known_correct", "reference", "done", 1.0, None),
            "files": [{"path": path, "content": "x'y\n"}],
        }
    )
    turn = control_payload(submitted).turns[0]
    assert isinstance(turn, AssistantToolCalls)
    machine = await ShellSimMachineFactory().create(MachineSpec(source=PackageSpec()))
    try:
        for call in turn.calls:
            result = await machine.run(Command(argv=("sh", "-c", str(call.object_arguments()["command"]))))
            assert result.exit_code == 0, result.stderr
        written = await machine.run(Command(argv=("cat", path)))
    finally:
        await machine.close()
    assert written.stdout == b"x'y\n"


def test_the_controls_request_names_the_output_paths_the_task_ships():
    """A control edits a shipped file in place; one it must create from nothing is listed plainly."""
    task = draft.assemble(
        "controls-paths",
        "Fix the README and write the answer.",
        AnswerType.TEXT,
        PlainText(),
        draft.answer_grader(ExactSpec(expected=("42",))),
        Source(dataset="taskforge-validate-tests", revision="1", row="paths", importer_revision="1"),
        environment=draft.requirements(image=None),
        files=(draft.file("workspace/README.md", "intro"), draft.file("workspace/data.csv", "a,b")),
        output_paths=("/workspace/README.md", "/workspace/answer.txt"),
    )

    assert shipped_output_paths(task) == ("/workspace/README.md",)
    text = output_paths_text(task.output_paths, shipped_output_paths(task))
    assert "/workspace/README.md, /workspace/answer.txt" in text
    assert "the task ships /workspace/README.md" in text
    assert (
        output_paths_text((), ())
        == "The task has no output paths, so every control is a final reply alone with no files."
    )
