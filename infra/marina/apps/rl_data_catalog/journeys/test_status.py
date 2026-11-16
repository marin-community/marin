# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A reader sees every source's status, reasons and evidence on the status page."""

import json
import re
from pathlib import Path

import pytest
from marina.journeys import Journey
from playwright.sync_api import expect

APP_ROOT = "a/fb11c931-5861-4878-8bb5-a964d652b45f"


def extras() -> dict:
    return json.loads((Path(__file__).parents[1] / "dist/measurements/source-status-extras.json").read_text())


@pytest.fixture
def seeded_extras(journey: Journey):
    """Serve the real extras plus one fixture pending row and a needs-ruling row."""
    payload = extras()
    payload["pending"] = [
        {
            "id": "Fixture:queued-source",
            "name": "queued-source",
            "display_name": "Fixture queued source",
            "status": "Pending",
            "stage": "Intake queued",
            "origin": "Marin",
            "upstream_dataset": "fixture/queued",
            "upstream_url": "https://example.org/queued",
            "notes": "Awaiting licensed intake conversion.",
            "reasons": [],
        }
    ]
    payload["metadata_exclusions"] = [
        {
            "id": "Fixture:metadata-excluded",
            "name": "metadata-excluded",
            "display_name": "Fixture metadata exclusion",
            "status": "Excluded",
            "stage": None,
            "origin": "Marin",
            "upstream_dataset": "fixture/excluded",
            "upstream_url": "https://example.org/excluded",
            "dataset_id": "fixture/excluded",
            "dataset_revision": "a" * 40,
            "notes": "Declaration metadata only.",
            "reasons": [
                {
                    "reason": "license-ban",
                    "evidence": {
                        "kind": "issue-link",
                        "message": "Upstream task-data permission has not been established.",
                        "uri": "https://github.com/marin-community/marin/issues/11497",
                        "sha256": None,
                        "expected": None,
                    },
                }
            ],
        },
        {
            "id": "Fixture:needs-ruling",
            "name": "needs-ruling",
            "display_name": "Fixture needs ruling",
            "status": "Excluded",
            "stage": None,
            "origin": "Marin",
            "upstream_dataset": "fixture/ruling",
            "upstream_url": "https://example.org/ruling",
            "dataset_id": "fixture/ruling",
            "dataset_revision": "a" * 40,
            "notes": "Legacy tag.",
            "reasons": [
                {
                    "reason": "unspecified-legacy",
                    "evidence": {
                        "kind": "note",
                        "message": "No explicit exclusion ruling is recorded for the tag.",
                        "uri": None,
                        "sha256": None,
                        "expected": None,
                    },
                }
            ],
        },
    ]
    page = journey.page
    page.route("**/measurements/source-status-extras.json", lambda route: route.fulfill(json=payload))
    return payload


def test_default_view_hides_excluded_sources_and_counts_what_is_hidden(journey: Journey) -> None:
    page = journey.page
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    expect(page.locator("#rows tr").first).to_be_visible()
    excluded_row = page.locator('tr[data-source-id="Marin:filtered-zero"]')
    expect(excluded_row).to_have_count(0)
    footer = page.locator("#visible-count")
    expect(footer).to_contain_text("excluded sources hidden")
    expect(page.locator("#task-reconciliation")).to_contain_text("hidden (excluded status)")

    page.locator("#status").select_option("Excluded")
    expect(excluded_row).to_have_count(1)
    expect(footer).not_to_contain_text("excluded sources hidden")
    page.screenshot(path=str(journey.shots / "default-view-excluded-filter.png"))


def test_status_page_lists_every_status_with_clickable_reasons(journey: Journey, seeded_extras: dict) -> None:
    page = journey.page
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    expect(page.locator("#rows tr").first).to_be_visible()
    base = page.url.rstrip("/") + "/"
    page.goto(base + "status.html")
    expect(page.locator("#rows tr").first).to_be_visible()
    expect(page.locator("#message")).to_have_text("")

    counts = page.locator("#status-counts")
    expect(counts.locator("button", has_text="Pending")).to_contain_text("1")
    expect(counts.locator("button", has_text="Excluded")).to_contain_text(re.compile(r"Excluded [1-9]"))

    queued = page.locator('tr[data-source-id="Fixture:queued-source"]')
    expect(queued.locator("td").nth(1)).to_contain_text("Pending")
    expect(queued.locator(".status-stage")).to_have_text("Intake queued")

    excluded = page.locator('tr[data-source-id="Fixture:metadata-excluded"]')
    reason = excluded.locator(".status-reasons summary")
    expect(reason).to_have_text("License ban")
    explanation = excluded.locator(".status-explanation")
    expect(explanation).not_to_be_visible()
    reason.click()
    expect(explanation).to_be_visible()
    expect(explanation.locator(".meaning")).to_contain_text("banned for training use")
    expect(explanation).to_contain_text("Cleared by:")
    link = excluded.locator(".status-evidence a")
    expect(link).to_have_attribute("href", "https://github.com/marin-community/marin/issues/11497")

    page.locator("#status-filter").select_option("Available")
    expect(page.locator('tr[data-source-id="Fixture:metadata-excluded"]')).to_have_count(0)
    page.locator("#status-filter").select_option("all")
    page.locator("#reason-filter").select_option("license-ban")
    expect(page.locator('tr[data-source-id="Fixture:queued-source"]')).to_have_count(0)
    expect(page.locator('tr[data-source-id="Fixture:metadata-excluded"]')).to_have_count(1)
    page.locator("#clear-filters").click()

    ruling = page.locator("#needs-ruling")
    expect(ruling).to_be_visible()
    expect(ruling.locator("li").first).to_contain_text("Fixture needs ruling")

    excluded.screenshot(path=str(journey.shots / "status-reason-expanded.png"))
    page.screenshot(path=str(journey.shots / "status-page.png"), full_page=True)
