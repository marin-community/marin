# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from infra.loom.gdoc.cli import AgentStatus, agent_status_request, describe_comment
from infra.loom.gdoc.docs_format import Tab


def header_paragraph(*parts: dict) -> dict:
    return {"startIndex": 1, "endIndex": 2, "paragraph": {"elements": list(parts)}}


def chip(value: str) -> dict:
    """A dropdown chip of the agent-status definition, with value selected."""
    return {
        "dropdown": {
            "dropdownId": "kix.chip",
            "dropdownProperties": {
                "dropdownDefinitionId": "kix.agent",
                "selectedOptionId": "dropdownItem.x",
                "displayValue": value,
            },
        }
    }


AGENT_STATUS_DEFINITION = {
    "kix.agent": {
        "dropdownDefinitionProperties": {
            "options": [
                {"optionId": "dropdownItem.unknown", "displayValue": "unknown"},
                {"optionId": "dropdownItem.input", "displayValue": "input needed"},
            ]
        }
    }
}


def test_a_comment_without_the_resolved_field_is_open_and_a_resolve_only_reply_shows_its_action():
    # Drive leaves `resolved` out of a comment that is not resolved, and content out of a resolve-only reply.
    new = {"id": "c9", "modifiedTime": "t9", "content": "Approve the restart?", "author": {"displayName": "loom"}}
    resolved = {**new, "resolved": True, "replies": [{"action": "resolve", "author": {"displayName": "Rav"}}]}

    assert describe_comment(new) == "- [c9] (open) loom: Approve the restart?"
    assert describe_comment(resolved).splitlines() == [
        "- [c9] (resolved) loom: Approve the restart?",
        "  - reply from Rav (resolved): ",
    ]


def test_agent_status_selects_the_chip_option_by_its_name_and_a_doc_without_the_chip_is_an_error():
    header = [header_paragraph({"textRun": {"content": "Agent status: "}}, chip("unknown"))]
    tab = Tab("t.0", "Main", header, dropdown_definitions=AGENT_STATUS_DEFINITION)

    assert agent_status_request(tab, AgentStatus.INPUT_NEEDED) == {
        "updateDropdownProperties": {
            "tabId": "t.0",
            "dropdownId": "kix.chip",
            "dropdownProperties": {"selectedOptionId": "dropdownItem.input"},
            "fields": "selectedOptionId",
        }
    }
    with pytest.raises(ValueError, match="no option 'goal achieved'"):
        agent_status_request(tab, AgentStatus.GOAL_ACHIEVED)
    with pytest.raises(ValueError, match="no Agent status chip"):
        agent_status_request(
            Tab("t.0", "Main", [header_paragraph({"textRun": {"content": "Status: draft\n"}})]), AgentStatus.WORKING
        )
