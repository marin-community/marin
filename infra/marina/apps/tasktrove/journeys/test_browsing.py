# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A reader follows the cleanup report into the clean Parquet viewer."""

from typing import Any, cast

from marina.journeys import Journey


def test_cleanup_report_explains_the_pipeline(journey: Journey) -> None:
    journey.visit("/").sees("A task collection with explicit, testable rewards")
    journey.sees("From archive to training-ready task").sees("1,399,813").sees("clean tasks")
    journey.widths("cleanup-report")


def test_browser_reads_the_parquet_and_opens_a_task(journey: Journey) -> None:
    journey.visit("/").click("Browse the clean Parquet")
    journey.sees("Parquet viewer")
    journey.sees("example-script-task").click("example-script-task")
    journey.sees("tests/verifier.toml").click("tests/verifier.toml")
    journey.sees('mode = "script"').shoot("script-task")


def test_shell_knows_the_app_and_the_caller(journey: Journey) -> None:
    journey.visit("/")
    apps = cast(dict[str, Any], journey.api("/api/marina/apps"))
    assert any(app["name"] == "tasktrove" for app in apps["apps"])
    assert journey.api("/api/marina/me") == {"user": "anonymous", "role": "admin"}
