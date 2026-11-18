# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source evidence remains usable when an independent ledger stalls or fails."""

import pytest
from marina.journeys import Journey
from playwright.sync_api import expect

APP_ROOT = "a/fb11c931-5861-4878-8bb5-a964d652b45f"
AUXILIARIES = [
    "measurements/content-progress.json",
    "measurements/oracle-inventory.json",
    "api/source-labels",
    "measurements/pending-filtered-releases.json",
]


@pytest.mark.parametrize("auxiliary", AUXILIARIES)
def test_table_source_evidence_finishes_when_auxiliary_never_settles(journey: Journey, auxiliary: str) -> None:
    page = journey.page
    page.clock.install()
    held = []
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    page.locator("#show-everything").click()
    page.route("**/" + auxiliary, lambda route: held.append(route))
    page.locator('tr[data-source-id="Marin:clean"] a[href^="source.html"]').click()
    expect(page.locator("#title")).to_have_text("clean", timeout=5000)
    expect(page.locator("#sections")).to_have_attribute("aria-busy", "false", timeout=5000)
    expect(page.get_by_role("heading", name="Oversight evidence", exact=True)).to_be_visible()
    expect(page.get_by_text("Loading supplementary evidence…", exact=True)).to_have_count(
        0 if auxiliary == "api/source-labels" else 1
    )
    expect(page.get_by_text("Loading overlap labels…", exact=True)).to_have_count(
        1 if auxiliary == "api/source-labels" else 0
    )
    assert held, f"The test must stall the real {auxiliary} HTTP request"
    page.clock.run_for(8100)
    expect(page.locator("#sections .error")).to_contain_text("Request timed out: " + auxiliary)
    expect(page.locator("#sections")).not_to_contain_text("Loading supplementary evidence")
    expect(page.locator("#sections")).not_to_contain_text("Loading overlap labels")


def test_source_api_timeout_replaces_loading_with_visible_error(journey: Journey) -> None:
    page = journey.page
    page.clock.install()
    held = []
    page.route("**/api/source?*", lambda route: held.append(route))
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    page.goto(page.url + "source.html#source_id=Marin%3Aclean")
    expect(page.locator("#title")).to_have_text("Loading source…")
    assert held
    page.clock.run_for(15100)
    expect(page.locator("#title")).to_have_text("Source evidence")
    expect(page.locator("#message.error")).to_contain_text("Request timed out: api/source?")
    expect(page.locator("#sections")).to_have_attribute("aria-busy", "false")


def test_removed_archive_page_preserves_archive_api(journey: Journey, readability_catalog: None) -> None:
    page = journey.page
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    base = page.url
    expect(page.get_by_role("navigation", name="Atlas views")).not_to_contain_text("Review archive")
    assert page.request.get(base + "archive.html").status == 404
    archive = page.request.get(base + "api/archive")
    assert archive.status == 200
    assert archive.json()["reviews"], "Removing the page must retain stored review collections"
