# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from evaldash.playground_reports import FeedbackRecord, FeedbackReport, ingest_reports
from marina.db import DatabaseSpec, engine_for
from marina.journeys import Journey
from playwright.sync_api import expect

MODEL = "marin/demo"
ISSUE = "The inventory export used UTC despite the requested timezone."
OTHER_ISSUE = "The response included inactive items in the stock count."
PROMPT = "Show the cutoff for the inventory export in Europe/Paris."
RESPONSE = "The inventory export cutoff is 18:00 UTC."


@pytest.fixture
def report_id(marina_database: DatabaseSpec, seeded_records: str) -> str:
    prefix = Path(seeded_records) / "model-feedback"
    report = FeedbackRecord(
        version=1,
        id="150000-inventory",
        created_at=datetime(2026, 11, 17, 15, tzinfo=UTC),
        serving={"model": MODEL, "backend": "vllm", "tensor_parallel_size": 4},
        report=FeedbackReport(
            model=MODEL,
            reporter="Casey",
            description=ISSUE,
            conversation={
                "title": "Inventory export cutoff",
                "messages": [
                    {"role": "user", "content": PROMPT},
                    {"role": "assistant", "content": RESPONSE},
                ],
            },
        ),
    )
    unrelated_report = report.model_copy(
        update={
            "id": "153000-stock-count",
            "created_at": datetime(2026, 11, 17, 15, 30, tzinfo=UTC),
            "report": FeedbackReport(model=MODEL, description=OTHER_ISSUE, conversation=None),
        }
    )
    for record in (report, unrelated_report):
        folder = prefix / record.created_at.date().isoformat()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{record.id}.json").write_text(record.model_dump_json(), encoding="utf-8")

    engine = engine_for(marina_database, "evaldash")
    try:
        assert ingest_reports(engine, (str(prefix),)) == ()
    finally:
        engine.dispose()
    return f"{report.created_at.date().isoformat()}-{report.id}"


def test_report_search_opens_shareable_conversation_and_restores_filters(journey: Journey, report_id: str) -> None:
    journey.visit()
    page = journey.page
    navigation = page.get_by_role("navigation", name="This app", exact=True)
    reports_tab = navigation.get_by_role("link", name="Reports", exact=True)
    reports_tab.click()
    expect(page.get_by_role("heading", name="Playground reports", exact=True)).to_be_visible()
    reports = page.get_by_role("table", name="Playground reports", exact=True)
    issue = reports.get_by_role("row").filter(has_text=ISSUE)
    unrelated = reports.get_by_role("row").filter(has_text=OTHER_ISSUE)
    expect(issue).to_be_visible()
    expect(unrelated).to_be_visible()
    desktop_viewport = page.viewport_size
    assert desktop_viewport is not None
    page.set_viewport_size({"width": 390, "height": 844})
    issue_link = issue.get_by_role("link", name=ISSUE, exact=True)
    expect(issue_link).to_be_in_viewport(ratio=1.0)
    journey.shoot("playground-reports-mobile")

    search = page.get_by_role("searchbox", name="Search", exact=True)
    search.fill("INVENTORY TIMEZONE")
    expect(unrelated).to_have_count(0)
    search.fill("")
    expect(unrelated).to_be_visible()
    filters = page.get_by_role("button", name=re.compile(r"^Filters\b"))
    filters.click()
    reporter = page.get_by_role("combobox", name=re.compile(r"^Reporter\b"))
    reporter.select_option("Casey")
    expect(unrelated).to_have_count(0)
    search.fill("INVENTORY TIMEZONE")
    issue_link.click()
    expect(page.get_by_role("heading", name=MODEL, exact=True)).to_be_visible()
    expect(page).to_have_url(re.compile(rf"/reports/{report_id}(?:\?|$)"))
    expect(reports_tab).to_have_attribute("aria-current", "page")
    page.set_viewport_size(desktop_viewport)
    journey.sees(ISSUE).sees(PROMPT).sees(RESPONSE).shoot("playground-report")

    page.get_by_role("link", name=re.compile(r"All reports$")).click()
    expect(search).to_have_value("INVENTORY TIMEZONE")
    expect(issue).to_be_visible()
    expect(unrelated).to_have_count(0)
    filters.click()
    expect(reporter).to_have_value("Casey")

    journey.visit(f"/reports/{report_id}")
    expect(page.get_by_role("heading", name=MODEL, exact=True)).to_be_visible()
    journey.sees(PROMPT).sees(RESPONSE)
