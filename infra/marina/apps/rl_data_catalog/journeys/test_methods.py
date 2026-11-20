# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marina.journeys import Journey
from playwright.sync_api import expect

from infra.marina.apps.rl_data_catalog.tests.method_fixtures import sft_variant

APP_ROOT = "a/fb11c931-5861-4878-8bb5-a964d652b45f"


def test_method_filter_and_variant_receipt_evidence(journey: Journey):
    page = journey.page
    child = sft_variant("Marin:clean").model_dump(mode="json")

    def inventory(route):
        response = route.fetch()
        payload = response.json()
        row = next(row for row in payload["sources"] if row["id"] == "Marin:clean")
        row["method_tags"] = ["rl", "sft"]
        route.fulfill(response=response, json=payload)

    def evidence(route):
        response = route.fetch()
        payload = response.json()
        payload["method_variants"].append(child)
        payload["method_tags"] = ["rl", "sft"]
        route.fulfill(response=response, json=payload)

    page.route("**/api/sources?projection=table", inventory)
    page.route("**/api/source?*", evidence)
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    page.locator("#show-everything").click()
    page.locator("#method").select_option("sft")
    expect(page.locator("#rows > tr")).to_have_count(1)
    expect(page.locator(".method-cell")).to_have_text("RL / SFT")
    headers = page.locator("#headers th").all_text_contents()
    assert headers.index("Use") + 1 == headers.index("Method")
    page.locator('#rows a[href^="source.html"]').click()
    expect(page.locator("#sections")).to_have_attribute("aria-busy", "false")
    variant = page.locator(".assessment-card").filter(has=page.get_by_text("Dataset variant: sft-v1", exact=True))
    expect(variant).to_contain_text("SFT")
    expect(variant.get_by_role("link", name="Open in Filer ↗", exact=True)).to_have_count(2)
    page.screenshot(path=str(journey.shots / "method-variant-evidence.png"))
