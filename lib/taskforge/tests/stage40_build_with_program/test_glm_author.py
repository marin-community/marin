# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from taskforge.ledger.records import EntryKind
from taskforge.llm.structured import StructuredOutputError, schema_instruction
from taskforge.stage30_author.author import PROGRAM_FILE, AuthorContext, Revision, RevisionKind
from taskforge.stage40_build_with_program.glm_author import (
    AUTHOR_DIR,
    COMPLETIONS_FILE,
    NOTES_LIMIT,
    REFERENCE_SECTIONS,
    GlmAuthor,
    ProgramSubmission,
    sdk_reference,
)
from taskforge.stage40_build_with_program.program_sdk import load_program
from tests.stage30_author.programs import PROGRAM
from tests.stage40_build_with_program.conftest import SAMPLING, WORKPIECE


def test_the_sdk_reference_lists_the_fields_of_the_task_types_steps_read():
    reference = sdk_reference()

    assert "class TaskResource(*, path: str, source: taskcompendium.models.InlineFile" in reference
    assert "class EnvironmentRequirements(*, capabilities:" in reference
    assert "docker_image:" in reference
    assert "resource_bytes(resource: taskcompendium.models.TaskResource) -> bytes" in reference


def test_each_name_appears_in_one_section_of_the_sdk_reference():
    names = [name for _, _, section in REFERENCE_SECTIONS for name in section]

    assert sorted({name for name in names if names.count(name) > 1}) == []
    assert "build_machine(services: " in sdk_reference()


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
    assert schema["properties"]["notes"]["description"].startswith("At most 2000 characters;")


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
