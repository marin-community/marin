# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quality tooltips and Evidence expose the same factual review explanation."""

import pytest
from marina.journeys import Journey
from playwright.sync_api import expect

APP_ROOT = "a/fb11c931-5861-4878-8bb5-a964d652b45f"


@pytest.mark.parametrize("theme,width", [("light", 1440), ("light", 390), ("dark", 1440), ("dark", 390)])
def test_source_publication_attribution_links_upstream_and_cc_by(journey: Journey, theme: str, width: int) -> None:
    page = journey.page
    page.set_viewport_size({"width": width, "height": 1000})
    page.emulate_media(color_scheme=theme)
    upstream = "https://huggingface.co/datasets/ricdomolm/CalibForge-decontaminated"

    def source(route):
        response = route.fetch()
        payload = response.json()
        payload["declaration"]["publication_scope"] = {
            "license_id": "cc-by-4.0",
            "upstream_name": "ricdomolm/CalibForge-decontaminated",
            "upstream_url": upstream,
        }
        route.fulfill(response=response, json=payload)

    page.route("**/api/source?*", source)
    page.goto(f"{journey.origin}/{APP_ROOT}/source.html#source_id=Marin%3Aclean")
    expect(page.locator("#sections")).to_have_attribute("aria-busy", "false")
    field = page.locator(".field").filter(has=page.get_by_text("License", exact=True))
    expect(field).to_contain_text("CC-BY-4.0")
    expect(field.get_by_role("link", name="ricdomolm/CalibForge-decontaminated ↗")).to_have_attribute("href", upstream)
    expect(field.get_by_role("link", name="CC BY 4.0 ↗")).to_have_attribute(
        "href", "https://creativecommons.org/licenses/by/4.0/"
    )


@pytest.mark.parametrize("theme,width", [("light", 1440), ("light", 390), ("dark", 1440), ("dark", 390)])
@pytest.mark.parametrize("reason", ["no_recorded_full_review", "full_review_other_input", "full_review_incomplete"])
def test_quality_explanation_tooltip_and_evidence(journey: Journey, theme: str, width: int, reason: str) -> None:
    page = journey.page
    page.set_viewport_size({"width": width, "height": 1000})
    page.emulate_media(color_scheme=theme)
    explanation = {
        "code": reason,
        "sample": {"reviewed": 96, "selected": 100, "population": 1000, "receipt": "s3://fixture/sample-review"},
        "full": None,
    }
    if reason != "no_recorded_full_review":
        explanation["full"] = {
            "reviewed": 800,
            "population": 1000,
            "receipt": {"uri": "s3://fixture/full-review"},
            "input": {"dataset_id": "org/data", "dataset_revision": "a" * 40},
            "different_fields": ["dataset_revision"],
        }

    def inventory(route):
        response = route.fetch()
        payload = response.json()
        row = next(row for row in payload["sources"] if row["id"] == "Marin:clean")
        row.update(quality="SAMPLE", quality_explanation=explanation)
        route.fulfill(response=response, json=payload)

    def source(route):
        response = route.fetch()
        payload = response.json()
        payload.update(quality="SAMPLE", quality_explanation=explanation)
        route.fulfill(response=response, json=payload)

    page.route("**/api/sources?projection=table", inventory)
    page.route("**/api/source?*", source)
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    page.locator("#show-everything").click()
    row = page.locator('tr[data-source-id="Marin:clean"]')
    tooltip = row.locator(".level").get_attribute("title")
    assert tooltip is not None
    assert "96 of 100 selected tasks reviewed (1,000 retained tasks)" in tooltip
    expected = {
        "no_recorded_full_review": "No recorded FULL review",
        "full_review_other_input": "another input",
        "full_review_incomplete": "800 of 1,000 tasks reviewed",
    }[reason]
    assert expected in tooltip
    assert "excluded from" not in tooltip and "scheduled" not in tooltip
    row.locator('a[href^="source.html"]').click()
    expect(page.locator("#sections")).to_have_attribute("aria-busy", "false")
    oversight = page.locator(".assessment-card").filter(
        has=page.get_by_role("heading", name="Oversight evidence", exact=True)
    )
    expect(oversight.locator(".field").filter(has=page.get_by_text("Why not FULL", exact=True))).to_contain_text(
        expected
    )
    expect(page.locator("#labels .level")).to_have_attribute("title", tooltip)
    expect(oversight.locator('a[href*="s3%3A%2F%2Ffixture%2Fsample-review"]')).to_have_count(1)
    if reason != "no_recorded_full_review":
        expect(oversight.locator('a[href*="s3%3A%2F%2Ffixture%2Ffull-review"]')).to_have_count(1)
