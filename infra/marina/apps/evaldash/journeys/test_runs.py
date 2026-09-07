# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Someone opens the run log, finds a launch, and reads what one of its evals scored."""

from typing import Any, cast

from marina.journeys import Journey

API = "/evaldash/api"


def test_the_run_log_lists_launches_and_one_run_opens(journey: Journey) -> None:
    journey.visit("/runs").shoot("runs")
    # Every facet value is also a hidden <option> in the filter bar, so read the page rather than
    # look for the first element carrying the text.
    assert "snowball" in journey.reads()

    journey.click("All runs")
    journey.sees("detail →")
    newest: str = cast(list[dict[str, Any]], journey.api(f"{API}/runs?limit=1"))[0]["run_id"]

    journey.click("detail →")
    journey.sees(newest).sees("Grade").sees("Metrics").shoot("run-detail")
    assert journey.page.url == journey.url(f"/runs/{newest}")


def test_the_shell_bar_carries_the_app_and_its_own_navigation(journey: Journey) -> None:
    journey.visit("/runs")
    assert journey.api("/api/marina/me") == {"user": "anonymous", "role": "admin"}
    offers = journey.offers()
    assert "a:EvalDash" in offers
    assert {"a:Panel", "a:Runs", "a:Debug"} <= set(offers)


def test_the_panel_serves_the_committed_catalog(journey: Journey) -> None:
    journey.visit("/").shoot("panel")
    assert "snowball" in journey.reads()
    store = cast(dict[str, Any], journey.api(f"{API}/status"))["store"]
    assert store["backend"] == "postgres"
    assert store["record_count"] == 15
