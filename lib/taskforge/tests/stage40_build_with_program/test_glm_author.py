# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from taskforge.ledger.records import EntryKind
from taskforge.llm.structured import StructuredOutputError, schema_instruction
from taskforge.stage10_proposal.model import Verification
from taskforge.stage30_author.author import PROGRAM_FILE, AuthorContext, Revision, RevisionKind, VerificationChange
from taskforge.stage40_build_with_program.glm_author import (
    AUTHOR_DIR,
    COMPLETIONS_FILE,
    NOTES_LIMIT,
    REFERENCE_SECTIONS,
    GlmAuthor,
    ProgramSubmission,
)
from taskforge.stage40_build_with_program.program_sdk import load_program
from tests.stage30_author.programs import PROGRAM
from tests.stage40_build_with_program.conftest import SAMPLING, WORKPIECE


def test_each_name_appears_in_one_section_of_the_sdk_reference():
    names = [name for _, _, section in REFERENCE_SECTIONS for name in section]

    assert sorted({name for name in names if names.count(name) > 1}) == []
    for _, module, names in REFERENCE_SECTIONS:
        for name in names:
            getattr(module, name)


def submission(source: str) -> str:
    return json.dumps({"source": source, "notes": "Grades 6 * 7 in process."})


async def test_the_author_repairs_a_refused_program_and_returns_the_accepted_source(
    proposal, glm_client, fake_glm, ledger, tmp_path
):
    refused = PROGRAM.replace("from dataclasses", "import os\nfrom dataclasses")
    fake_glm.stream(content=submission(refused))
    fake_glm.stream(content=submission(PROGRAM))
    workpiece_dir = tmp_path / "workpiece"

    authored = await GlmAuthor(glm_client, SAMPLING, ledger)(
        AuthorContext(WORKPIECE, 0, workpiece_dir), proposal, None, {}
    )

    assert authored.source == PROGRAM
    assert authored.program is load_program(PROGRAM)
    assert (workpiece_dir / PROGRAM_FILE).read_text() == PROGRAM
    assert "may not import 'os'" in fake_glm.requests[1]["messages"][-2]["content"]
    assert (
        json.loads((workpiece_dir / AUTHOR_DIR / "00" / "request.json").read_text()) == fake_glm.requests[0]["messages"]
    )
    schema = fake_glm.requests[0]["response_format"]["json_schema"]["schema"]
    assert list(schema["properties"]) == ["notes", "source"]
    assert schema["properties"]["notes"]["maxLength"] == NOTES_LIMIT == 2000


async def test_a_revision_continues_from_the_program_the_workpiece_holds(
    proposal, glm_client, fake_glm, ledger, tmp_path
):
    workpiece_dir = tmp_path / "workpiece"
    workpiece_dir.mkdir()
    (workpiece_dir / PROGRAM_FILE).write_text(PROGRAM)
    fixed = PROGRAM.replace('"Reply with the number."', '"Reply with the product alone."')
    fake_glm.stream(content=submission(fixed))
    revision = Revision(RevisionKind.BUILD_FAILURE, "grader: the reference scored 0.0")

    authored = await GlmAuthor(glm_client, SAMPLING, ledger)(
        AuthorContext(WORKPIECE, 2, workpiece_dir), proposal, revision, {}
    )

    assert authored.source == fixed
    assert (workpiece_dir / PROGRAM_FILE).read_text() == fixed
    turns = fake_glm.requests[0]["messages"]
    assert turns[-3] == {"role": "assistant", "content": PROGRAM}
    assert "grader: the reference scored 0.0" in turns[-2]["content"]
    assert turns[-1] == {"role": "user", "content": schema_instruction(ProgramSubmission)}
    assert [(e.step, e.round) for e in ledger.entries if e.kind == EntryKind.LLM_CALL] == [("author", 2)]


async def test_a_revisions_verification_change_reaches_the_prompt_and_comes_back_unchanged(
    proposal, glm_client, fake_glm, ledger, tmp_path
):
    workpiece_dir = tmp_path / "workpiece"
    workpiece_dir.mkdir()
    (workpiece_dir / PROGRAM_FILE).write_text(PROGRAM)
    fake_glm.stream(content=submission(PROGRAM))
    change = VerificationChange(Verification.ANSWER_IN_BOX, Verification.DECISION_MODEL, "the reply is prose")
    revision = Revision(RevisionKind.MODIFY, "the grader accepted a hedged reply", change)

    authored = await GlmAuthor(glm_client, SAMPLING, ledger)(
        AuthorContext(WORKPIECE, 1, workpiece_dir), proposal, revision, {}
    )

    messages = fake_glm.requests[0]["messages"]
    asked = messages[-2]["content"]
    assert "answer_in_box" in asked and "decision_model" in asked
    assert '"verification": "decision_model"' in messages[1]["content"]
    assert authored.verification_change == change


async def test_an_author_call_that_never_loads_keeps_its_completions(proposal, glm_client, fake_glm, ledger, tmp_path):
    refused = PROGRAM.replace("from dataclasses", "import os\nfrom dataclasses")
    fake_glm.stream(content=submission(refused))
    fake_glm.stream(content=submission(refused))
    workpiece_dir = tmp_path / "workpiece"

    with pytest.raises(StructuredOutputError):
        await GlmAuthor(glm_client, SAMPLING, ledger)(AuthorContext(WORKPIECE, 0, workpiece_dir), proposal, None, {})

    completions = json.loads((workpiece_dir / AUTHOR_DIR / "00" / COMPLETIONS_FILE).read_text())
    assert [json.loads(c["content"])["source"] for c in completions] == [refused, refused]
    assert not (workpiece_dir / PROGRAM_FILE).exists()


async def test_an_author_reply_cut_off_by_the_output_limit_fails_without_a_repair(
    proposal, glm_client, fake_glm, ledger, tmp_path
):
    fake_glm.stream(content=submission(PROGRAM), finish="length")
    fake_glm.stream(content=submission(PROGRAM))
    workpiece_dir = tmp_path / "workpiece"

    with pytest.raises(StructuredOutputError):
        await GlmAuthor(glm_client, SAMPLING, ledger)(AuthorContext(WORKPIECE, 0, workpiece_dir), proposal, None, {})

    assert len(fake_glm.requests) == 1
    completions = json.loads((workpiece_dir / AUTHOR_DIR / "00" / COMPLETIONS_FILE).read_text())
    assert [c["finish_reason"] for c in completions] == ["length"]
    assert not (workpiece_dir / PROGRAM_FILE).exists()
