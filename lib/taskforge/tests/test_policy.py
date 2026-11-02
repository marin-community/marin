# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The default policy renders the requests Taskforge sent before its prompts became files, and overlays replace them.

Each sha256 below was computed at e2ee276627 (the head of #10301), from the module constant the prompt file replaced,
rendered with the same fields. A prompt edit changes what the models are sent: update the sha with the edit and say
why in the PR.
"""

import hashlib
import json
from collections.abc import Callable

import pytest

from taskforge.defaults import DEFAULT_POLICY, DEFAULTS_DIR
from taskforge.policy import POLICY_FILE, load_policy
from taskforge.stage10_proposal.model import parse
from taskforge.stage30_author.loader import ALLOWED_IMPORTS, GRADER_STEP
from taskforge.stage40_build_with_program.agentic_author import (
    NOTES_PATH,
    PROGRAM_FILE,
    PROPOSAL_PATH,
    SDK_PATH,
    TEMPLATE_PATH,
    WRITE_CHARS,
    opening_messages,
)
from taskforge.stage40_build_with_program.glm_author import author_messages, sdk_reference
from taskforge.stage50_validate.adversary_prompts import NO_SHORTCUT_LINE, SHORTCUT_PREFIX, TOOLS_WITH_SHELL
from tests.stage40_build_with_program.conftest import PROPOSAL

AUTHOR = DEFAULT_POLICY.author
ADVERSARY = DEFAULT_POLICY.adversary


def sha256(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


BASE_RENDERS: dict[str, tuple[Callable[[], object], str]] = {
    "author.system": (
        lambda: AUTHOR.system.render(grader=GRADER_STEP, imports=", ".join(sorted(ALLOWED_IMPORTS))),
        "0a88c10dd4f59b699048778d4e8243014a870278eac28495060bf02565f7d206",
    ),
    "author.user": (
        lambda: AUTHOR.user.render(proposal="P", reference="R", template="T"),
        "b442879a703b86b5d41a651d92dec4386fadd26ce2a8bbd960f4548fbb6b57e6",
    ),
    "author.revision": (
        lambda: AUTHOR.revision.render(kind="build_failure", detail="D"),
        "05ba3456341a46b36cf1c3efb83a6581d5ca5e3f33b116857c31f63d6dc25270",
    ),
    "author.grading_reference": (
        lambda: AUTHOR.grading_reference.render(),
        "b37eed31e6ac3755f17571722426fbcc120cebab6696544eaf191c2173fa3287",
    ),
    "author.session": (
        lambda: AUTHOR.session.render(
            program=PROGRAM_FILE,
            proposal=PROPOSAL_PATH,
            sdk=SDK_PATH,
            template=TEMPLATE_PATH,
            notes=NOTES_PATH,
            write_chars=WRITE_CHARS,
            turns=40,
        ),
        "07c3a157f5bcc2832afaac4b8966c580bcaaa4c7606f6757f6a8cb47fb408e1c",
    ),
    "author.seed": (
        lambda: AUTHOR.seed.render(program=PROGRAM_FILE, notes=NOTES_PATH, submitted="S\n"),
        "fa40577e26b47e3506ef117e1b5f710e978801c6e7964acb5ab5a0f93715d85a",
    ),
    "author.handoff": (
        lambda: AUTHOR.handoff.render(reason="max_turns"),
        "b05c4bce28be19f5e7719b018bad2815f32505c3db0a5f44f1fdb8ae9ecaa96b",
    ),
    "author.no_revision": (
        lambda: AUTHOR.no_revision.render(),
        "48b6971b6187ba74539a767a83788a6ad765ddcefb3663e3a3ba27bed16dbb01",
    ),
    "author.edit_and_submit": (
        lambda: AUTHOR.edit_and_submit.render(program=PROGRAM_FILE),
        "e61252980ddff5088b6e54b99fa26c5a2199fdeb2e122cd72944730e966988a8",
    ),
    "author.length_nudge": (
        lambda: AUTHOR.length_nudge.render(limit=8192),
        "f14ece3c586889ed57e06a2aed741418e29f363c9cc38a05f9bca5c3d9e1ef99",
    ),
    "author.nudge": (
        lambda: AUTHOR.nudge.render(program=PROGRAM_FILE),
        "1d90543103d8643be6d6fd0e01f4e80c80752aa5d9b92a7a89e640c7b05d9634",
    ),
    "adversary.brief": (
        lambda: ADVERSARY.brief.render(
            tools=TOOLS_WITH_SHELL,
            context=" ",
            submissions=3,
            probes=2,
            no_shortcut=NO_SHORTCUT_LINE,
            shortcut_prefix=SHORTCUT_PREFIX,
            line_clause="",
        ),
        "e89c2f9afdd21c4ecbe3f6ad9f3f12e4bc6234609f02b89f5b5e2875d39cb164",
    ),
    "adversary.whitebox": (
        lambda: ADVERSARY.whitebox.render(),
        "2c1bf2c15549dc3f29a0f56c0b64cf6609d5a60eb52b63aef03356b1e181ee7d",
    ),
    "adversary.submit": (
        lambda: ADVERSARY.submit.render(),
        "530fae48386fefb8ad927e978707bf6dbfa2969cebc794ab3c28e826a4ca905f",
    ),
    # The whole requests: the author's, and the agentic author's opening, which appends `author.session`.
    "sdk_reference": (
        lambda: sdk_reference(AUTHOR.grading_reference),
        "891a7071bd6dd1d54484249cd1a270f6449b341222043630d83ad8b2e06f0602",
    ),
    "author_messages": (
        lambda: author_messages(parse(PROPOSAL), AUTHOR),
        "46d6a3640eadae533ffaec9af68321200f647284bdfb097a77e7ad1ee5899d34",
    ),
    "opening_messages, 40 turns": (
        lambda: opening_messages(parse(PROPOSAL), AUTHOR, 40),
        "1f91531f9bc90818a8a49a1a322c32673a02b9a4cd2cb2baec99d5cda6d201f0",
    ),
    "opening_messages, no turn bound": (
        lambda: opening_messages(parse(PROPOSAL), AUTHOR, None),
        "7f8425b7deeaba816321661b08e950d3be016ab49d3b1757ab23bba8311cb648",
    ),
}


@pytest.mark.parametrize("name", list(BASE_RENDERS))
def test_default_prompt_renders_the_text_sent_before_prompts_were_files(name):
    rendered, expected = BASE_RENDERS[name]

    assert sha256(rendered()) == expected


def test_an_overlay_replaces_only_the_fields_it_names_and_changes_the_digest(tmp_path):
    (tmp_path / "nudge.md").write_text("Call a tool; submit `{program}` when it builds.\n")
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({"author.nudge": "nudge.md"}))

    policy = load_policy(DEFAULTS_DIR, overlay)

    assert policy.author.nudge.render(program="program.py") == "Call a tool; submit `program.py` when it builds."
    assert policy.author.system == DEFAULT_POLICY.author.system
    assert policy.proposer == DEFAULT_POLICY.proposer
    assert policy.digest != DEFAULT_POLICY.digest
    assert load_policy(DEFAULTS_DIR).digest == DEFAULT_POLICY.digest


def test_a_policy_file_naming_an_unknown_field_or_missing_one_is_refused(tmp_path):
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({"author.turns": "turns.md"}))
    with pytest.raises(ValueError, match=r"names no such policy field: author\.turns"):
        load_policy(DEFAULTS_DIR, overlay)

    defaults = json.loads((DEFAULTS_DIR / POLICY_FILE).read_text())
    del defaults["author.nudge"]
    (tmp_path / POLICY_FILE).write_text(json.dumps(defaults))
    with pytest.raises(ValueError, match=r"has no file for author\.nudge"):
        load_policy(tmp_path)
