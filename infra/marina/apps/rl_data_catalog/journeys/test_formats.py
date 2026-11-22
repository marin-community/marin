# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marina.journeys import Journey
from playwright.sync_api import expect

APP_ROOT = "a/fb11c931-5861-4878-8bb5-a964d652b45f"


@pytest.mark.parametrize("theme,width", [("light", 1440), ("dark", 1440), ("light", 390), ("dark", 390)])
def test_format_column_and_filter(journey: Journey, theme: str, width: int):
    page = journey.page
    page.emulate_media(color_scheme=theme)
    page.set_viewport_size({"width": width, "height": 1000})

    # Prepare the intercepted payload through the request context BEFORE registering the
    # route: in-handler route.fetch() deadlocks the sync-API event loop (release-blocker
    # fix for the journey added in f10cd8ff92; assertions unchanged).
    payload = page.request.get(f"{journey.origin}/{APP_ROOT}/api/sources?projection=table").json()
    row = next(row for row in payload["sources"] if row["id"] == "Marin:clean")
    row["task_format"] = "harbor"

    page.route("**/api/sources?projection=table", lambda route: route.fulfill(json=payload))
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    page.locator("#show-everything").click()
    page.locator("#format").select_option("harbor")
    expect(page.locator("#rows > tr")).to_have_count(1)
    expect(page.locator(".format-cell")).to_have_text("Harbor")
    headers = page.locator("#headers th").all_text_contents()
    assert headers.index("Method") + 1 == headers.index("Format")
    page.locator("#format").select_option("taskspec")
    expect(page.locator("#rows > tr")).to_have_count(0)
    page.locator("#format").select_option("all")
    expect(page.locator(".format-cell").filter(has_text="—").first).to_be_visible()
    page.screenshot(path=str(journey.shots / f"format-column-{theme}-{width}.png"))
