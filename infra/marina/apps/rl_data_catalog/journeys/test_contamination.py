# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A reader distinguishes measured overlap, remediation and missing evidence."""

import pytest
from marina.journeys import Journey
from playwright.sync_api import expect


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_contamination_chips_counts_and_phone_evidence(journey: Journey, theme: str) -> None:
    page = journey.page
    page.emulate_media(color_scheme=theme)
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    page.locator("#show-all").check()
    expect(page.locator("#rows tr")).to_have_count(9)
    expected = {
        "clean": "NO ISSUES",
        "remediated": "REMEDIATED",
        "eval": "MEASURED, EVAL",
        "required": "REMEDIATION REQUIRED",
        "unmeasured": "NOT MEASURED",
        "stale-pin": "REMEDIATION REQUIRED",
        "eval-zero": "NO ISSUES",
        "public-release": "REMEDIATION REQUIRED",
        "receipt-eval": "MEASURED, EVAL",
    }
    for name, status in expected.items():
        chip = page.locator(f'tr[data-source-id="Marin:{name}"] .contamination-chip')
        expect(chip).to_have_text(status)
        expect(chip).to_be_visible()
    expect(page.locator('tr[data-source-id="Marin:eval"] .overlap-cell')).to_have_text("1")
    expect(page.locator('tr[data-source-id="Marin:eval"] .issue-rate-cell')).to_have_text("0.00%")
    expect(page.locator('tr[data-source-id="Marin:unmeasured"] .overlap-cell')).to_have_text("—")
    expect(page.locator('tr[data-source-id="Marin:unmeasured"] .issue-rate-cell')).to_have_text("—")
    expect(page.locator('tr[data-source-id="Marin:eval"] .result-provisional')).to_have_text("Good · provisional")
    assert page.locator("#rows td:has(> .chip)").evaluate_all(
        "cells => cells.every(cell => cell.children.length === 1 && "
        "cell.textContent === cell.firstElementChild.textContent)"
    )
    warning = page.locator('tr[data-source-id="Marin:required"] .contamination-chip')
    assert warning.evaluate("chip => getComputedStyle(chip).borderTopWidth") == "2px"
    colors = page.locator("#rows .contamination-chip").evaluate_all(
        "chips => [...new Set(chips.map(chip => getComputedStyle(chip).backgroundColor))]"
    )
    assert len(colors) == 5
    page.locator("#contamination-legend summary").click()
    expect(page.locator("#contamination-legend")).to_contain_text("Cleanliness is unknown")
    for width in (1440, 390):
        page.set_viewport_size({"width": width, "height": 900})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator('tr[data-source-id="Marin:eval"] .source-name').click()
        expect(page.locator("#details .contamination-chip")).to_have_text("MEASURED, EVAL")
        assert page.locator("#details").evaluate("dialog => dialog.scrollWidth <= dialog.clientWidth")
        expect(page.locator("#details")).to_contain_text("1 / 3 overlapping tasks")
        page.locator("#close-details").click()
    page.locator('tr[data-source-id="Marin:eval"] .source-name').click()
    page.get_by_role("link", name="All quality evidence", exact=True).click()
    expect(page.locator("#labels .contamination-chip")).to_have_text("MEASURED, EVAL")
    expect(page.locator("#sections")).to_contain_text("Flagged tasks")
    page.get_by_role("link", name="Getting started", exact=True).click()
    expect(page.locator("#contamination")).to_contain_text("NOT MEASURED")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
