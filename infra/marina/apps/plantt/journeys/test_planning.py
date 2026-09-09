# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import re

import pytest


@pytest.mark.timeout(120)
def test_create_example_plan(journey) -> None:
    journey.visit("/").sees("Database-backed plans").click("Use example")
    journey.sees("Example Project Plan").sees("Compute capacity").sees("Launch")
    journey.shoot("example-plan")


@pytest.mark.timeout(120)
def test_agent_panel_reads_the_current_plan_and_restores_after_refresh(journey, scripted_loom) -> None:
    journey.visit("/").click("Open an example")
    journey.sees("Example Project Plan").click("Ask Marina")
    journey.sees("Example Project Plan · revision 1")
    journey.click("What is on the critical path?").sees("Thinking…")

    (launch,) = scripted_loom.recorded("/api/sessions/launch")
    assert launch.body["profile"] == "marina"
    assert launch.body["repo"] == "marin-community/marin"
    assert launch.body["protocol"] == "acp"
    goal = launch.body["goal"]
    assert isinstance(goal, str)
    context_match = re.search(r"<marina_page_context_json>\n(.*)\n</marina_page_context_json>", goal)
    assert context_match is not None
    page_context = json.loads(context_match.group(1))
    assert page_context["app"] == "plantt"
    assert page_context["contextKey"].startswith("chart:")
    assert page_context["state"] == {
        "chart_id": page_context["contextKey"].removeprefix("chart:"),
        "conflict": False,
        "dirty": False,
        "revision": 1,
        "saving": False,
        "selected_item": None,
    }
    assert "document" not in page_context["state"]

    scripted_loom.release(
        "tool",
        {
            "turn": 1,
            "tool_call_id": "read-chart-1",
            "title": "Reading Plantt chart",
            "tool_kind": "mcp",
            "status": "running",
            "content": [],
            "locations": [],
        },
    )
    journey.sees("Reading Plantt chart · running").shoot("agent-reading")

    scripted_loom.release(
        "block",
        {
            "turn": 1,
            "seq": 1,
            "kind": "tool_call",
            "payload": {
                "tool_call_id": "read-chart-1",
                "title": "Read Plantt chart",
                "tool_kind": "mcp",
                "status": "completed",
                "content": [],
                "locations": [],
            },
            "created_at": "2026-09-09T12:00:01Z",
        },
    )
    answer = (
        "The launch path is **Foundations → Service v1 → Scale-up → Launch**. "
        "The quality branch finishes earlier at the July 1 quality gate."
    )
    scripted_loom.release(
        "block",
        {
            "turn": 1,
            "seq": 2,
            "kind": "agent_message",
            "payload": {"text": answer},
            "created_at": "2026-09-09T12:00:02Z",
        },
    )
    scripted_loom.release("turn", {"turn": 1, "state": "ended", "stop_reason": "end_turn"})
    journey.sees("The launch path is").shoot("agent-answer").widths("agent-complete")

    journey.page.reload(wait_until="domcontentloaded")
    journey.sees("Example Project Plan").click("Ask Marina").sees("The launch path is")
    assert len(scripted_loom.recorded("/api/sessions/launch")) == 1
    assert len(scripted_loom.recorded("/api/sessions/get")) == 1
