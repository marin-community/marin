# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re

from marina.journeys import Journey
from playwright.sync_api import expect


def test_launch_opens_its_runs_and_exposes_the_grade(journey: Journey) -> None:
    journey.visit()
    page = journey.page
    navigation = page.get_by_role("navigation", name="This app", exact=True)
    expect(navigation.get_by_role("link")).to_have_text(["Runs", "Compare", "Benchmarks", "Models"])
    more = navigation.get_by_text("More", exact=True)
    more.press("Enter")
    expect(navigation.get_by_role("link", name="Playground reports", exact=True)).to_be_visible()
    diagnostics = navigation.get_by_role("link", name="Diagnostics", exact=True)
    expect(diagnostics).to_be_visible()
    more.press("Escape")
    expect(diagnostics).to_be_hidden()
    expect(page.get_by_role("heading", name="Evaluation runs")).to_be_visible()
    expect(page.get_by_role("button", name="Launches", exact=True)).to_have_attribute("aria-pressed", "true")
    launches = page.get_by_role("region", name="Evaluation launches", exact=True)
    launch = launches.get_by_role("listitem").filter(has_text="Nightly baseline sweep")
    expect(launch).to_contain_text("snowball")
    journey.shoot("launches")
    journey.widths("launches", widths=(390,))

    launch.get_by_role("link", name="View runs for snowball", exact=True).click()
    expect(page.get_by_role("button", name="All runs", exact=True)).to_have_attribute("aria-pressed", "true")
    expect(page.get_by_role("table").get_by_role("link", name="mmlu", exact=True)).to_be_visible()
    expect(page.get_by_role("table")).not_to_contain_text("tootsie-8b")
    expect(page).to_have_url(re.compile(r"[?&]group=snowball-2026\.07\.20"))

    page.get_by_role("link", name="aime", exact=True).click()
    expect(page.get_by_role("heading", name="snowball-2026.07.20-aime", exact=True)).to_be_visible()
    journey.sees(re.compile(r"\b33\.3\s*%")).sees("attempted 10 of 10").sees("graded 9 of 10")
    expect(page.get_by_role("row").filter(has_text="AgentTimeoutError")).to_contain_text("1")
    journey.shoot("run-results")


def test_run_search_finds_older_failures_and_restores_filters(journey: Journey) -> None:
    journey.visit("/runs?view=runs&limit=1")
    page = journey.page
    table = page.get_by_role("table")
    expect(table).to_contain_text("native-qwen3-8b")

    search = page.get_by_role("searchbox", name="Search runs")
    search.fill("TOOTSIE 8B CHECKPOINT")
    search.press("Enter")
    expect(table).to_contain_text("tootsie-8b")
    page.get_by_role("button", name="Next", exact=True).click()
    journey.sees(re.compile(r"2\u20132 of \d+ runs"))

    page.get_by_role("button", name="Filters", exact=True).click()
    page.get_by_role("combobox", name=re.compile(r"^Status\b")).select_option("failed")
    expect(table.get_by_role("link", name="math500", exact=True)).to_be_visible()
    journey.sees("1\u20131 of 1 run")
    expect(page).to_have_url(re.compile(r"[?&]status=failed"))
    page.reload(wait_until="domcontentloaded")
    expect(search).to_have_value("TOOTSIE 8B CHECKPOINT")
    expect(table.get_by_role("link", name="math500", exact=True)).to_be_visible()
    journey.shoot("run-search")

    search.fill("")
    search.press("Enter")
    expect(page).not_to_have_url(re.compile(r"[?&]q="))
