# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A reader follows the cleanup report into the clean Parquet viewer."""

from typing import Any, cast

import pytest
from marina.journeys import Journey
from playwright.sync_api import expect


def test_cleanup_report_explains_the_pipeline(journey: Journey) -> None:
    journey.visit("/").sees("A task collection with explicit, testable rewards")
    journey.sees("Cleanup pipeline").sees("1,449,686").sees("clean tasks")
    journey.sees("s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.9/tasks/part-00000.parquet")
    journey.sees("What happened to every source").sees("DCAgent__exp_rle_adversarial-v6")
    journey.click("Source audit").fill("Find a source", "nl2bash")
    journey.sees("Run the oracle command and compare its sandbox effects.").sees("script")
    journey.widths("cleanup-report")


@pytest.mark.timeout(90)
def test_browser_reads_the_parquet_and_opens_a_task(journey: Journey) -> None:
    journey.visit("/").click("Browse the clean Parquet")
    journey.sees("Parquet viewer")
    journey.select("Source", "DCAgent2__nl2bash-tasks-cleaned-oracle-v2 (1,497)")
    journey.sees("task_5407", timeout=60).shoot("filtered-table").click("task_5407")
    journey.sees("tests/verifier.toml").click("tests/verifier.toml")
    journey.sees('mode = "script"').shoot("script-task")


@pytest.mark.timeout(180)
def test_browser_filters_by_grader_and_tag(journey: Journey) -> None:
    journey.visit("/browse").select("Grader", "reasoning-gym (13,712)")
    journey.sees("reasoning-gym-cd49ef60ff40.tar.gz", timeout=60)
    journey.fill("Tag", "course-schedule").click("Apply")
    journey.sees("reasoning-gym-51ad80354b6a.tar.gz", timeout=60).sees("course-schedule")


def test_shell_knows_the_app_and_the_caller(journey: Journey) -> None:
    journey.visit("/")
    apps = cast(dict[str, Any], journey.api("/api/marina/apps"))
    assert any(app["name"] == "tasktrove" for app in apps["apps"])
    assert journey.api("/api/marina/me") == {"user": "anonymous", "role": "admin"}


@pytest.mark.parametrize("width", [390, 1440])
def test_app_switcher_handles_a_large_directory(journey: Journey, width: int) -> None:
    apps = [
        {"name": f"app-{i}", "title": f"Research tool {i}", "description": "Inspect experiments.", "path": f"/app-{i}/"}
        for i in range(30)
    ]
    apps.append({"name": "tasktrove", "title": "TaskTrove", "description": "Browse tasks.", "path": "/tasktrove/"})
    page = journey.page
    page.route("**/api/marina/apps", lambda route: route.fulfill(json={"apps": apps}))
    page.set_viewport_size({"width": width, "height": 900})
    journey.visit("/")
    launcher = page.get_by_role("button", name="Open apps")
    launcher.click()
    search = page.get_by_role("searchbox", name="Find an app")
    expect(search).to_be_focused()
    links = page.get_by_role("navigation", name="Apps", exact=True).get_by_role("link")
    expect(links).to_have_count(31)
    search.fill("tool 29")
    expect(links).to_have_count(1)
    expect(links).to_have_attribute("href", "/app-29/")
    search.fill("missing app")
    expect(links).to_have_count(0)
    page.keyboard.press("Escape")
    expect(page.get_by_role("dialog")).not_to_be_visible()
    expect(launcher).to_be_focused()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    launcher.click()
    search.fill("TaskTrove")
    expect(links).to_have_attribute("aria-current", "page")
    links.click()
    expect(page.get_by_role("button", name="Open apps")).to_be_visible()
    expect(page.get_by_role("dialog")).not_to_be_visible()
