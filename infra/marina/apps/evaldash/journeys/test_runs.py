# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re

from marina.journeys import Journey
from playwright.sync_api import expect


def test_launch_opens_its_runs_and_exposes_the_grade(journey: Journey) -> None:
    journey.visit()
    page = journey.page
    expect(page.get_by_role("heading", name="Evaluation runs")).to_be_visible()
    expect(page.get_by_role("button", name="Launches", exact=True)).to_have_attribute("aria-pressed", "true")
    launches = page.get_by_role("region", name="Evaluation launches", exact=True)
    launch = launches.get_by_role("listitem").filter(has_text="Nightly baseline sweep")
    expect(launch).to_contain_text("snowball")
    journey.shoot("launches")
    journey.widths("launches", widths=(390,))

    navigation = page.get_by_role("navigation", name="This app", exact=True)
    reports = navigation.get_by_role("link", name="Reports", exact=True)
    diagnostics = navigation.get_by_role("link", name="Diagnostics", exact=True)
    expect(reports).to_have_attribute("href", "/evaldash/reports")
    expect(diagnostics).to_have_attribute("href", "/evaldash/debug")
    expect(diagnostics).to_be_visible()

    launch.get_by_role("link", name="Nightly baseline sweep", exact=True).click()
    expect(page.get_by_role("button", name="All runs", exact=True)).to_have_attribute("aria-pressed", "true")
    expect(page.get_by_role("table").get_by_role("link", name="mmlu", exact=True)).to_be_visible()
    expect(page.get_by_role("table")).not_to_contain_text("tootsie-8b")
    expect(page).to_have_url(re.compile(r"[?&]group=snowball-2026\.07\.20"))

    page.get_by_role("link", name="aime", exact=True).click()
    expect(page.get_by_role("heading", name="snowball-2026.07.20-aime", exact=True)).to_be_visible()
    grade = page.get_by_text("Grade", exact=True).locator("xpath=..")
    expect(grade.get_by_text("0.333", exact=True)).to_be_visible()
    expect(grade).to_contain_text("attempted 10 of 10")
    expect(grade).to_contain_text("graded 9 of 10")
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


def test_run_search_draft_preserves_pagination_history(journey: Journey) -> None:
    page = journey.page
    page.clock.install(time="2026-07-20T12:00:00Z")
    journey.visit("/runs?view=runs&q=snowball&limit=1")
    page.get_by_role("button", name=re.compile("Next")).click()
    expect(page).to_have_url(re.compile(r"[?&]offset=1(?:&|$)"))

    page.clock.pause_at("2026-07-20T12:01:00Z")
    search = page.get_by_role("searchbox", name="Search runs")
    search.fill("unsubmitted draft")
    page.go_back()
    expect(page).not_to_have_url(re.compile(r"[?&]offset="))
    page.clock.run_for(1000)
    expect(search).to_have_value("snowball")
    expect(page).to_have_url(re.compile(r"[?&]q=snowball(?:&|$)"))

    page.go_forward()
    expect(page).to_have_url(re.compile(r"[?&]offset=1(?:&|$)"))
    search.fill("tootsie")
    page.get_by_role("button", name="Launches", exact=True).click()
    expect(page).to_have_url(re.compile(r"[?&]q=tootsie(?:&|$)"))
    expect(page).not_to_have_url(re.compile(r"[?&]offset="))


def test_eval_detail_links_to_the_native_marina_attempt(journey: Journey) -> None:
    journey.visit("/runs/snowball-2026.07.20-mmlu")
    link = journey.page.get_by_role("link", name="RL attempt wanted")
    expect(link).to_have_attribute(
        "href", "/a/6c971887-1e22-45b1-9dcb-dd775829df57/#run=native%2Ffixture&attempt=wanted"
    )
