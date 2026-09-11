# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A reader follows the cleanup report into the clean Parquet viewer."""

from typing import Any, cast

import pytest
from marina.journeys import Journey


def test_cleanup_report_explains_the_pipeline(journey: Journey) -> None:
    journey.visit("/").sees("A task collection with explicit, testable rewards")
    journey.sees("Cleanup pipeline").sees("1,450,969").sees("clean tasks")
    journey.sees("What happened to every source").sees("DCAgent__exp_rle_adversarial-v6")
    journey.click("Source audit").fill("Find a source", "nl2bash")
    journey.sees("Run the oracle command and compare its sandbox effects.").sees("script")
    journey.widths("cleanup-report")


@pytest.mark.timeout(90)
def test_browser_reads_the_parquet_and_opens_a_task(journey: Journey) -> None:
    journey.visit("/").click("Browse the clean Parquet")
    journey.sees("Parquet viewer")
    journey.select("Source", "DCAgent2__nl2bash-tasks-cleaned-oracle-v2 (1,498)")
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
