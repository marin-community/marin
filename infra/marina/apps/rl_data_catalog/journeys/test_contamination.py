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
    expect(page.locator("#rows tr")).to_have_count(23)
    expected = {
        "clean": "NO ISSUES",
        "stale-clean": "NOT MEASURED",
        "clean-subset": "NO ISSUES",
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
    expect(page.locator('tr[data-source-id="Marin:stale-clean"] .overlap-cell')).to_have_text("—")
    expect(page.locator('tr[data-source-id="Marin:unmeasured"] .issue-rate-cell')).to_have_text("—")
    expect(page.locator('tr[data-source-id="Marin:eval"] .result-provisional')).to_have_count(0)
    assert page.locator("#rows td:has(> .chip)").evaluate_all(
        "cells => cells.every(cell => cell.children.length === 1 && "
        "cell.textContent === cell.firstElementChild.textContent)"
    )
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


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_complete_sample_is_final_and_unfinished_reviews_are_provisional(journey: Journey, theme: str) -> None:
    page = journey.page
    page.emulate_media(color_scheme=theme)
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    page.locator("#show-all").check()
    complete = page.locator('tr[data-source-id="Marin:sample-complete"]')
    content_only = page.locator('tr[data-source-id="Marin:sample-content"]')
    expect(content_only.locator(".level")).to_have_text("SAMPLE")
    expect(content_only.locator(".result-provisional")).to_have_count(0)
    expect(complete.locator(".level")).to_have_text("SAMPLE")
    expect(complete.locator(".result-provisional")).to_have_count(0)
    expect(complete.locator("td").nth(4)).to_have_text("Good")
    for name in ("sample-uncertain", "full-inflight", "full-uncertain", "full-false-complete", "sample-false-waiver"):
        expect(page.locator(f'tr[data-source-id="Marin:{name}"] .result-provisional')).to_have_text("Good · provisional")
    for name in ("full-false-complete", "sample-false-waiver"):
        expect(page.locator(f'tr[data-source-id="Marin:{name}"] .level')).to_have_text("SAMPLE")
    page.set_viewport_size({"width": 390, "height": 844})
    page.locator('tr[data-source-id="Marin:sample-uncertain"] .source-name').click()
    result_field = page.locator("#detail-fields .field").filter(has=page.get_by_text("Result", exact=True)).locator("dd")
    expect(result_field).to_have_text("Good · provisional")
    page.locator("#close-details").click()
    complete.locator(".source-name").click()
    expect(result_field).to_have_text("Good")
    page.get_by_role("link", name="All quality evidence", exact=True).click()
    expect(page.locator("#labels .result-provisional")).to_have_count(0)
    expect(page.locator("#sections")).to_contain_text("Required content review complete")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_pending_review_is_quick_without_a_result(journey: Journey, theme: str) -> None:
    page = journey.page
    page.emulate_media(color_scheme=theme)
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    pending = page.locator('tr[data-source-id="Marin:review-pending"]')
    expect(pending).to_have_count(0)
    page.locator("#show-all").check()
    expect(pending.locator(".level")).to_have_text("QUICK")
    expect(pending.locator(".level")).to_have_attribute("title", "SAMPLE review in progress")
    expect(pending.locator("td").nth(4)).to_have_text("Not assessed")
    expect(pending.locator(".issue-rate-cell")).to_have_text("—")
    for width in (1440, 390):
        page.set_viewport_size({"width": width, "height": 844})
        pending.locator(".source-name").click()
        expect(page.locator("#detail-fields")).to_contain_text("SAMPLE review in progress")
        assert page.locator("#details").evaluate("d=>d.scrollWidth<=d.clientWidth")
        page.locator("#close-details").click()
    pending.locator(".source-name").click()
    page.get_by_role("link", name="All quality evidence", exact=True).click()
    expect(page.locator("#labels")).to_contain_text("Quality QUICK")
    expect(page.locator("#labels")).to_contain_text("Result not assessed")
    expect(page.locator("#sections")).to_contain_text("SAMPLE review in progress")
    expect(page.locator("#sections")).not_to_contain_text("Required content review complete")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_whole_source_waiver_is_full_final_with_incomplete_evidence(journey: Journey, theme: str) -> None:
    page = journey.page
    page.emulate_media(color_scheme=theme)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    page.locator("#show-all").check()
    row = page.locator('tr[data-source-id="Marin:full-waiver"]')
    expect(row.locator(".level")).to_have_text("FULL")
    expect(row.locator(".result-provisional")).to_have_count(0)
    row.locator(".source-name").click()
    page.get_by_role("link", name="All quality evidence", exact=True).click()
    expect(page.locator("#sections")).to_contain_text("Final with accepted whole-source waiver")
    coverage = page.locator(".assessment-card").filter(has=page.get_by_role("heading", name="Content review coverage"))
    expect(coverage).to_contain_text("Strict coverage complete")
    expect(coverage).to_contain_text("false")
    expect(coverage).to_contain_text("Unreviewed tasks")
    expect(coverage).to_contain_text("1")
    expect(coverage.locator('a[href="https://example.org/waiver-with-exact-reasons.json"]')).to_be_visible()
    expect(coverage).to_contain_text("a" * 64)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1440, 390])
def test_wide_enum_columns_scroll_without_overlapping(journey: Journey, theme: str, width: int) -> None:
    page = journey.page
    page.emulate_media(color_scheme=theme)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    page.locator("#show-all").check()
    expect(page.locator("#rows tr")).to_have_count(23)
    chip = page.locator('tr[data-source-id="Marin:clean"] td:nth-child(9) .chip')
    expect(chip).to_have_text("UNSAFE: too_many_snapshots, unbuilt_image")
    expect(page.get_by_role("columnheader", name="Backends")).to_be_visible()
    expect(page.get_by_role("columnheader", name="Grading")).to_be_visible()
    expect(page.get_by_role("columnheader", name="Oracle pass")).to_have_count(0)
    clean = page.locator('tr[data-source-id="Marin:clean"]')
    assert set(clean.locator(".backend-chips .chip").all_text_contents()) == {
        "Docker",
        "LOCAL: PARTIAL: docker_image",
        "Daytona: UNSUPPORTED: packages_lock",
    }
    expect(clean.locator(".grading-chips .chip")).to_have_text("SEPARATE")
    expect(clean.locator("td").nth(1)).to_have_text("Pending")
    assert page.locator(".enum-chips").evaluate_all(
        "groups => groups.every(g => [...g.children].every(c => c.classList.contains('chip')) && "
        "g.textContent === [...g.children].map(c => c.textContent).join(''))"
    )
    scroller = page.locator(".table-scroller")
    assert scroller.evaluate("element => element.scrollWidth > element.clientWidth")
    for right_edge in (False, True):
        scroller.evaluate("(element, right) => element.scrollLeft = right ? element.scrollWidth : 0", right_edge)
        assert page.locator(".source-table").evaluate(
            """table => {
          const headers = [...table.tHead.rows[0].cells].map(cell => cell.getBoundingClientRect());
          return [...table.tBodies[0].rows].every(row => {
            const cells = [...row.cells];
            return cells.every((cell, index) => {
              const bounds = cell.getBoundingClientRect();
              const next = cells[index + 1]?.getBoundingClientRect();
              return (!next || bounds.right <= next.left + 1) &&
                Math.abs(bounds.left - headers[index].left) <= 1 &&
                Math.abs(bounds.right - headers[index].right) <= 1 &&
                [...cell.querySelectorAll('.chip')].every(chip => {
                  const rect = chip.getBoundingClientRect();
                  return rect.left >= bounds.left && rect.right <= bounds.right &&
                    getComputedStyle(chip).whiteSpace === 'nowrap' &&
                    chip.scrollWidth <= chip.clientWidth;
                });
            });
          });
        }"""
        )
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert page.locator("#rows td:has(> .chip)").evaluate_all(
        "cells => cells.every(cell => cell.children.length === 1 && "
        "cell.textContent === cell.firstElementChild.textContent)"
    )
    journey.shoot(f"wide-enums-{theme}-{width}")
    page.locator('tr[data-source-id="Marin:clean"] .source-name').click()
    page.get_by_role("link", name="All quality evidence", exact=True).click()
    expect(page.locator("#labels")).to_contain_text("too_many_snapshots, unbuilt_image")
    ledger = page.locator(".assessment-card").filter(has=page.get_by_role("heading", name="Filtration ledger"))
    expect(ledger).to_contain_text("N/A: complete census found no shipped oracle solutions")
    expect(ledger).to_contain_text("Pending: post-oracle admission is not measured")
    expect(page.locator(".backend-table tbody tr")).to_have_count(3)
    expect(page.locator(".backend-table")).to_contain_text("packages lock")
    expect(page.locator(".backend-table")).to_contain_text("docker image")
    for right_edge in (False, True):
        page.locator(".backend-table").locator("xpath=..").evaluate(
            "(element, right) => element.scrollLeft = right ? element.scrollWidth : 0", right_edge
        )
        assert page.locator(".backend-table").evaluate(
            """table => {
          const headers = [...table.tHead.rows[0].cells].map(cell => cell.getBoundingClientRect());
          return [...table.tBodies[0].rows].every(row => [...row.cells].every((cell, index, cells) => {
            const bounds = cell.getBoundingClientRect(), next = cells[index + 1]?.getBoundingClientRect();
            return (!next || bounds.right <= next.left + 1) &&
              Math.abs(bounds.left - headers[index].left) <= 1 &&
              Math.abs(bounds.right - headers[index].right) <= 1;
          }));
        }"""
        )
    assert page.locator(".field").evaluate_all(
        """fields => fields.every(field => {
      const label = field.querySelector('dt').getBoundingClientRect();
      const value = field.querySelector('dd').getBoundingClientRect();
      return label.right <= value.left && value.right <= field.getBoundingClientRect().right + 1;
    })"""
    )
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    journey.shoot(f"wide-evidence-{theme}-{width}")
    page.get_by_role("link", name="By source", exact=True).click()
    page.locator("#show-all").check()
    page.locator('tr[data-source-id="Marin:stale-clean"] .source-name').click()
    page.get_by_role("link", name="All quality evidence", exact=True).click()
    stale_ledger = page.locator(".assessment-card").filter(has=page.get_by_role("heading", name="Filtration ledger"))
    expect(stale_ledger).to_contain_text("Pending: no complete population-wide oracle-filter receipt is recorded")
    expect(stale_ledger).not_to_contain_text("N/A: complete census found no shipped oracle solutions")
