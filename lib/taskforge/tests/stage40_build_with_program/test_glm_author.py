# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from taskforge.ledger.records import EntryKind
from taskforge.llm.recording import CallLedger
from taskforge.llm.structured import schema_instruction
from taskforge.stage30_author.author import PROGRAM_FILE, Revision, RevisionKind
from taskforge.stage30_author.loader import load_program_source
from taskforge.stage40_build_with_program.glm_author import AUTHOR_DIR, NOTES_LIMIT, GlmAuthor, ProgramSubmission, sdk_reference
from tests.stage30_author.programs import PROGRAM
from tests.stage40_build_with_program.conftest import WORKPIECE, SAMPLING


def test_the_sdk_reference_lists_the_fields_of_the_task_types_steps_read():
    reference = sdk_reference()

    assert "class TaskResource(*, path: str, source: taskcompendium.models.InlineFile" in reference
    assert "class EnvironmentRequirements(*, capabilities:" in reference
    assert "docker_image:" in reference
    assert "resource_bytes(resource: taskcompendium.models.TaskResource) -> bytes" in reference


def submission(source: str) -> str:
    return json.dumps({"source": source, "notes": "Grades 6 * 7 in process."})


async def test_the_author_repairs_a_refused_program_and_returns_the_accepted_source(
    proposal, glm_client, fake_glm, ledger, tmp_path
):
    refused = PROGRAM.replace("from dataclasses", "import os\nfrom dataclasses")
    fake_glm.stream(content=submission(refused))
    fake_glm.stream(content=submission(PROGRAM))
    item_dir = tmp_path / "item"

    authored = await GlmAuthor(glm_client, SAMPLING, item_dir, CallLedger(ledger, WORKPIECE, 0, "author"))(proposal, None)

    assert authored.source == PROGRAM
    assert authored.program is load_program_source(PROGRAM)
    assert (item_dir / PROGRAM_FILE).read_text() == PROGRAM
    assert "may not import 'os'" in fake_glm.requests[1]["messages"][-2]["content"]
    assert json.loads((item_dir / AUTHOR_DIR / "00" / "request.json").read_text()) == fake_glm.requests[0]["messages"]
    schema = fake_glm.requests[0]["response_format"]["json_schema"]["schema"]
    assert list(schema["properties"]) == ["notes", "source"]
    assert schema["properties"]["notes"]["maxLength"] == NOTES_LIMIT


async def test_a_revision_continues_from_the_program_the_item_holds(proposal, glm_client, fake_glm, ledger, tmp_path):
    item_dir = tmp_path / "item"
    item_dir.mkdir()
    (item_dir / PROGRAM_FILE).write_text(PROGRAM)
    fixed = PROGRAM.replace('"Reply with the number."', '"Reply with the product alone."')
    fake_glm.stream(content=submission(fixed))
    revision = Revision(RevisionKind.BUILD_FAILURE, "grader: the reference scored 0.0")

    authored = await GlmAuthor(glm_client, SAMPLING, item_dir, CallLedger(ledger, WORKPIECE, 2, "author"))(proposal, revision)

    assert authored.source == fixed
    assert (item_dir / PROGRAM_FILE).read_text() == fixed
    turns = fake_glm.requests[0]["messages"]
    assert turns[-3] == {"role": "assistant", "content": PROGRAM}
    assert "grader: the reference scored 0.0" in turns[-2]["content"]
    assert turns[-1] == {"role": "user", "content": schema_instruction(ProgramSubmission)}
    assert [(e.step, e.round) for e in ledger.entries if e.kind == EntryKind.LLM_CALL] == [("author", 2)]
