# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A reader distinguishes measured overlap, remediation and missing evidence."""

import re

import pytest
from marina.journeys import Journey
from playwright.sync_api import expect


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_difficulty_model_sort_tooltips_and_source_details(journey: Journey, theme: str) -> None:
    page = journey.page
    page.emulate_media(color_scheme=theme)
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    page.locator("#show-all").check()
    for solver, first in [("small", "Marin:clean"), ("large", "Marin:eval")]:
        page.locator("#difficulty-model").select_option(solver)
        expect(page.locator("#rows tr").first).to_have_attribute("data-source-id", first)
    row = page.locator('tr[data-source-id="Marin:clean"]')
    expect(row.locator(".difficulty-model-label")).to_have_text(["Small", "Large", "Hosted", "Snowball v0.1"])
    tooltip = row.locator(".difficulty-small").get_attribute("title")
    assert tooltip is not None
    assert "n=3" in tooltip and "Wilson 95% CI" in tooltip and "Pre-oracle-filter" in tooltip
    for width in (1440, 390):
        page.set_viewport_size({"width": width, "height": 900})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert page.locator(".source-table").evaluate(
            """table => {
            const headers = [...table.tHead.rows[0].cells].map(c => c.getBoundingClientRect());
            return [...table.tBodies[0].rows].every(row => [...row.cells].every((cell, index, cells) => {
                const bounds = cell.getBoundingClientRect(), next = cells[index + 1]?.getBoundingClientRect();
                return (!next || bounds.right <= next.left + 1) &&
                    Math.abs(bounds.left - headers[index].left) <= 1 &&
                    Math.abs(bounds.right - headers[index].right) <= 1 &&
                    [...cell.querySelectorAll('.chip,.difficulty-model')].every(child => {
                        const box = child.getBoundingClientRect();
                        return box.left >= bounds.left && box.right <= bounds.right &&
                            child.scrollWidth <= child.clientWidth;
                    });
            }));
        }"""
        )
        row.locator(".source-name").click()
        expect(page.locator("#detail-difficulty .difficulty-model")).to_have_count(4)
        assert page.locator("#details").evaluate("dialog => dialog.scrollWidth <= dialog.clientWidth")
        page.locator("#close-details").click()
    row.locator(".source-name").click()
    page.get_by_role("link", name="All quality evidence", exact=True).click()
    difficulty = page.locator(".assessment-card").filter(has=page.get_by_role("heading", name="Difficulty", exact=True))
    expect(difficulty).to_contain_text("resampling required after filtering")
    expect(difficulty.locator(".difficulty-model")).to_have_count(4)


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
    expect(page.locator('tr[data-source-id="Marin:required"] .use-cell')).to_have_text("Training")
    expect(page.locator('tr[data-source-id="Marin:eval"] .use-cell')).to_have_text("Evaluation")
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
    assert set(page.locator("#contamination-legend dt .chip").all_text_contents()) == set(expected.values())
    for width in (1440, 390):
        page.set_viewport_size({"width": width, "height": 900})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator('tr[data-source-id="Marin:eval"] .source-name').click()
        expect(page.locator("#details .contamination-chip")).to_have_text("MEASURED, EVAL")
        assert page.locator("#details").evaluate("dialog => dialog.scrollWidth <= dialog.clientWidth")
        expect(page.locator("#details")).to_contain_text("1 / 3 overlapping tasks")
        page.locator("#close-details").click()
        page.locator('tr[data-source-id="Marin:required"] .source-name').click()
        pending = (
            page.locator("#detail-fields .field")
            .filter(has=page.get_by_text("Lexically contaminated tasks removed", exact=True))
            .locator("dd")
        )
        expect(pending).to_contain_text("Pending")
        expect(pending).to_contain_text(re.compile(r"\b1\b"))
        page.locator("#close-details").click()
    page.locator('tr[data-source-id="Marin:eval"] .source-name').click()
    page.get_by_role("link", name="All quality evidence", exact=True).click()
    expect(page.locator("#labels .contamination-chip")).to_have_text("MEASURED, EVAL")
    expect(page.locator("#sections")).to_contain_text("Flagged tasks")
    lexical_drops = page.locator(".assessment-card").filter(has=page.get_by_role("heading", name="Filtration ledger"))
    expect(
        lexical_drops.locator(".field")
        .filter(has=page.get_by_text("Lexically contaminated tasks removed", exact=True))
        .locator("dd")
    ).to_have_text("0")
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
    expect(page.get_by_role("columnheader", name="Examples", exact=True).get_by_role("button")).to_have_count(0)
    source_header = page.locator("#headers th").first
    expect(source_header).to_have_attribute("aria-sort", "ascending")
    before = page.locator("#rows tr").evaluate_all("rows => rows.map(row => row.dataset.sourceId)")
    source_header.get_by_role("button").click()
    expect(source_header).to_have_attribute("aria-sort", "descending")
    assert page.locator("#rows tr").evaluate_all("rows => rows.map(row => row.dataset.sourceId)") == list(
        reversed(before)
    )
    source_header.get_by_role("button").click()
    clean = page.locator('tr[data-source-id="Marin:clean"]')
    expect(clean.locator(".backends-cell .chip")).to_have_count(0)
    expect(clean.locator(".backends-link")).to_have_text("Backends \u203a")
    assert clean.locator(".backends-cell").evaluate("cell => cell.getBoundingClientRect().width < 150")
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
    oracle_field = (
        ledger.locator(".field").filter(has=page.get_by_text("Oracle-failing tasks removed", exact=True)).locator("dd")
    )
    admitted_field = (
        ledger.locator(".field").filter(has=page.get_by_text("Admitted after all filtration", exact=True)).locator("dd")
    )
    expect(oracle_field).to_have_text(re.compile(r"^N/A\b"))
    expect(admitted_field).to_have_text(re.compile(r"^Pending\b"))
    executability_field = (
        ledger.locator(".field")
        .filter(has=page.get_by_text("Unexecutable on production backend dropped (by typed reason)", exact=True))
        .locator("dd")
    )
    expect(executability_field).to_have_text(re.compile(r"^Pending\b"))
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
    stale_oracle = (
        stale_ledger.locator(".field")
        .filter(has=page.get_by_text("Oracle-failing tasks removed", exact=True))
        .locator("dd")
    )
    expect(stale_oracle).to_have_text(re.compile(r"^Pending\b"))
    original_oracles = (
        stale_ledger.locator(".field")
        .filter(has=page.get_by_text("Original oracle-bearing tasks", exact=True))
        .locator("dd")
    )
    expect(original_oracles).to_have_text("Not measured")


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1440, 390])
def test_backend_link_opens_role_fit_and_survey_provenance(journey: Journey, theme: str, width: int) -> None:
    page = journey.page
    page.emulate_media(color_scheme=theme)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    page.locator("#show-all").check()
    page.locator('tr[data-source-id="Marin:clean"] .backends-link').click()
    expect(page).to_have_url(re.compile(r"/backends\.html#source_id=Marin%3Aclean$"))
    expect(page.locator("#title")).to_have_text("clean")
    expect(page.locator("#fit .backend-table tbody tr")).to_have_count(3)
    expect(page.locator("#fit")).to_contain_text("docker image")
    expect(page.locator("#fit")).to_contain_text("packages lock")
    expect(page.locator("#provenance")).to_contain_text("survey-fixture")
    expect(page.locator("#provenance")).to_contain_text("b" * 40)
    expect(page.locator("#provenance")).to_contain_text("c" * 64)
    expect(page.locator("#provenance")).to_contain_text("2026-11-14T10:00:00Z")
    expect(page.locator("#census")).to_contain_text("d" * 64)
    expect(
        page.locator('#census a[href="https://filer.marina.oa.dev/#s3%3A%2F%2Ftest%2Ftask-specs.parquet"]')
    ).to_be_visible()
    for right in (False, True):
        page.locator("#fit .table-wrap").evaluate("(e,right)=>e.scrollLeft=right?e.scrollWidth:0", right)
        assert page.locator("#fit table").evaluate(
            """table => [...table.tBodies[0].rows].every(row => [...row.cells].every((cell,i,cells) => {
              const bounds=cell.getBoundingClientRect(), next=cells[i+1]?.getBoundingClientRect();
              const head=table.tHead.rows[0].cells[i].getBoundingClientRect();
              return (!next || bounds.right<=next.left+1) && Math.abs(bounds.left-head.left)<=1 &&
                Math.abs(bounds.right-head.right)<=1;
            }))"""
        )
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    journey.shoot(f"backends-{theme}-{width}")
    page.get_by_role("link", name="By source", exact=True).click()
    page.locator("#show-all").check()
    page.locator('tr[data-source-id="Marin:eval"] .backends-link').click()
    local = page.locator("#fit tbody tr").filter(has=page.get_by_role("cell", name="LOCAL", exact=True))
    expect(local.locator("td").nth(2)).to_have_text("N/A (no role): 3")
    expect(local.locator("td").nth(3)).to_have_text("supported: 3")
    shellsim = page.locator("#fit tbody tr").filter(has=page.get_by_role("cell", name="ShellSim", exact=True))
    expect(shellsim.locator(".chip")).to_have_text("N/A")
    expect(shellsim.locator("td").nth(4)).to_have_text("grader:host image: 3")
    expect(shellsim.locator("td").nth(5)).to_have_text("None measured")
    page.get_by_role("link", name="By source", exact=True).click()
    page.locator("#show-all").check()
    page.locator('tr[data-source-id="Marin:stale-clean"] .backends-link').click()
    qemu = page.locator("#fit tbody tr").filter(has=page.get_by_role("cell", name="QEMU", exact=True))
    expect(qemu.locator("td").nth(2)).to_contain_text("unsupported: 2")
    expect(qemu.locator("td").nth(2)).to_contain_text("supported: 1")
    expect(qemu.locator("td").nth(3)).to_have_text("Not measured")
    expect(qemu.locator("td").nth(4)).to_have_text("None measured")
    expect(qemu.locator("td").nth(5)).to_have_text("agent:packages lock: 2")
    expect(qemu.locator("td").nth(6)).to_have_text("agent:packages lock: 1")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    journey.shoot(f"backend-role-na-{theme}-{width}")
    page.get_by_role("link", name="By source", exact=True).click()
    page.locator("#show-all").check()
    page.locator('tr[data-source-id="Marin:clean-subset"] .backends-link').click()
    expect(page.locator("#fit .chip")).to_have_text("N/A (no role)")
    page.get_by_role("link", name="By source", exact=True).click()
    page.locator("#show-all").check()
    page.locator('tr[data-source-id="Marin:unmeasured"] .backends-link').click()
    expect(page.locator("#fit .backend-table")).to_have_count(0)
    expect(page.locator("#fit .chip")).to_have_count(0)
    basis = page.locator("#census .field").filter(has=page.get_by_text("Measurement basis", exact=True))
    expect(basis.locator("dd")).to_have_text("Not measured")
    converter = page.locator("#census .field").filter(has=page.get_by_text("Converter version", exact=True))
    expect(converter.locator("dd")).to_have_text("Not recorded")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
