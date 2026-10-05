# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys

import pytest

from taskforge.build.author import (
    PROGRAM_FILE,
    PROGRAM_RECORD,
    SUBMIT_TOOL,
    ProgramSubmission,
    Revision,
    author,
    compile_program,
    load_program,
    module_name,
)
from taskforge.build.run import item_id_for
from taskforge.build.template import standard


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (lambda s: "import os\n" + s, "may not import 'os'"),
        (lambda s: s + "\nopen('/etc/passwd')\n", "NameError"),
        # Only reached at run time, so it is found statically.
        (lambda s: s.replace("    return spec.environment(", "    exec('pass')\n    return spec.environment("), "exec"),
        (lambda s: s.replace("async def build(", "async def run("), "async def build"),
        (lambda s: s.replace("@step(StepRole.GRADER)", "@step(StepRole.OTHER)"), "GRADER"),
    ],
)
def test_program_outside_the_sdk_surface_is_rejected(program_source, edit, problem):
    source = edit(program_source)
    with pytest.raises(ValueError, match=problem):
        compile_program(source, proposal_digest="p")
    assert module_name(source) not in sys.modules


def test_validating_a_submission_keeps_a_live_program_with_the_same_source(program_source):
    program = compile_program(program_source, proposal_digest="p")

    ProgramSubmission.model_validate({"source": program_source, "notes": "n"})

    assert sys.modules[module_name(program_source)] is program.module


async def test_author_repairs_an_invalid_program_and_stores_the_accepted_one(
    proposal, program_source, tmp_path, services, fake_glm
):
    fake_glm.stream(
        tool_calls=((SUBMIT_TOOL, json.dumps({"source": "import os\n" + program_source, "notes": "n"})),),
        finish="tool_calls",
    )
    fake_glm.stream(
        tool_calls=((SUBMIT_TOOL, json.dumps({"source": program_source, "notes": "builds 6*7"})),),
        finish="tool_calls",
    )
    item_dir = tmp_path / "item"

    async with services(fake_glm.base_url) as s:
        program = await author(proposal, standard, item_dir, s, item_id_for(proposal))

    assert len(fake_glm.requests) == 2
    assert program.source == program_source
    assert (item_dir / PROGRAM_FILE).read_text() == program_source
    record = json.loads((item_dir / PROGRAM_RECORD).read_text())
    assert (record["digest"], record["proposal_digest"]) == (program.digest, proposal.digest)
    assert load_program(item_dir, proposal).digest == program.digest


async def test_revision_continues_from_the_failed_program(
    proposal, program_source, tmp_path, services, fake_glm, ledger
):
    fixed = program_source.replace("timeout=60,", "timeout=90,")
    fake_glm.stream(tool_calls=((SUBMIT_TOOL, json.dumps({"source": fixed, "notes": "n"})),), finish="tool_calls")
    item_dir = tmp_path / "item"

    async with services(fake_glm.base_url) as s:
        revision = Revision(source=program_source, failure="grader: reference scored 0.0")
        program = await author(proposal, standard, item_dir, s, item_id_for(proposal), revision, round=2)

    assert program.source == fixed
    assert [e.round for e in ledger.entries if e.step == "author"] == [2]

    turns = fake_glm.requests[0]["messages"]
    assert turns[-2] == {"role": "assistant", "content": program_source}
    assert "grader: reference scored 0.0" in turns[-1]["content"]
    assert (
        json.loads((item_dir / PROGRAM_RECORD).read_text())["revises"]
        == compile_program(program_source, proposal.digest).digest
    )
    assert load_program(item_dir, proposal).digest == program.digest
