# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A reader distinguishes measured overlap, remediation and missing evidence."""

import json
import re
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote, urlencode
from uuid import UUID

import pytest
from marina.applets import applet_schema
from marina.db import DatabaseSpec, engine_for
from marina.journeys import Journey
from playwright.sync_api import Page, expect
from sqlalchemy import text
from taskcompendium.pipeline.models import Confidence, Defect, Quality, ReferenceStatus, ReviewRecord, ReviewVerdict

from experiments.post_training.task_curation.review_quality import quality_result as recompute_review_result
from experiments.post_training.task_curation.tests.license_fixtures import fixture_license
from infra.marina.apps.rl_data_catalog.readability import inspect_state
from infra.marina.apps.rl_data_catalog.release_browser import (
    TABLE_BOUNDS,
    check_evidence,
    check_view,
    curriculum_full_sweep,
    curriculum_sweep,
    evidence_sweep,
)
from infra.marina.apps.rl_data_catalog.server.app import migrate
from infra.marina.apps.rl_data_catalog.server.quality import review_completeness
from infra.marina.apps.rl_data_catalog.tests.review_fixtures import review_declarations
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence

CATALOG_APP = "fb11c931-5861-4878-8bb5-a964d652b45f"


def served_source_ids(journey: Journey) -> list[str]:
    """The served source inventory, in API order.

    Geometry expectations derive from this inventory instead of a hardcoded
    row count, so fixture additions change the expectation with the data.
    """
    response = journey.page.request.get(f"{journey.origin}/a/{CATALOG_APP}/api/sources")
    assert response.ok, f"{response.status}: {response.text()[:200]}"
    return [source["id"] for source in response.json()["sources"]]


@pytest.mark.parametrize("interval", [None, {"low": 0.2, "high": 0.8}])
def test_partial_measured_difficulty_row_renders_without_optional_provenance(page: Page, interval: dict | None) -> None:
    page.set_content("<main></main>")
    page.add_script_tag(path=str(Path(__file__).parents[1] / "dist" / "difficulty.js"))
    page.evaluate(
        """interval => document.querySelector('main').append(AtlasDifficulty.comparison({models: [{
            solver: 'small', status: 'measured', solved: 1, n: 2, solve_rate: 0.5,
            wilson_95: interval, model: null, sample: null
        }]}))""",
        interval,
    )
    expect(page.locator(".difficulty-model")).to_have_count(4)
    title = page.locator(".difficulty-small").get_attribute("title")
    assert title is not None and "50.0% solved" in title
    assert ("Wilson 95% CI" in title) is (interval is not None)
    assert "undefined" not in title


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1440, 390, 320])
def test_pending_private_filter_evidence_keeps_catalog_admission_and_quality(
    journey: Journey, theme: str, width: int
) -> None:
    page = journey.page
    page.emulate_media(color_scheme=theme)
    page.set_viewport_size({"width": width, "height": 1000})
    root = f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/"
    page.goto(root)
    expect(page.locator("#rows tr").first).to_be_visible()
    root = page.url
    before = page.locator(".metrics").inner_text()
    detail = page.evaluate("async()=> (await fetch('api/source?source_id=Marin%3Aclean')).json()")
    manifest = json.loads((Path(__file__).parents[1] / "dist/measurements/pending-filtered-releases.json").read_text())
    candidate = manifest["sources"][0]
    candidate["source_id"] = "Marin:clean"
    page.route(
        "**/measurements/pending-filtered-releases.json",
        lambda route: route.fulfill(json={**manifest, "sources": [candidate]}),
    )
    page.goto(root + "source.html#source_id=Marin%3Aclean")
    card = page.locator(".pending-filtered-release")
    expect(card).to_have_count(1)
    expect(card.get_by_role("heading")).to_have_text("Filtered private release, pending qualification")
    release = candidate["release"]
    expect(card.get_by_role("link", name=f"{release['repo']} @ {release['revision']}")).to_have_attribute(
        "href", release["url"]
    )
    expect(card.locator(".field").filter(has=page.get_by_text("Oracle-filtered tasks", exact=True))).to_contain_text(
        f"{candidate['filtered_tasks']:,} of {candidate['original_tasks']:,}"
    )
    sample = candidate["sample"]
    receipt = card.locator(".field").filter(has=page.get_by_text("SAMPLE receipt", exact=True))
    expect(receipt.get_by_role("link")).to_have_attribute(
        "href", "https://filer.marina.oa.dev/#" + quote(sample["receipt"]["uri"], safe="")
    )
    reviewed = card.locator(".field").filter(has=page.get_by_text("SAMPLE reviewed", exact=True))
    unavailable = card.locator(".field").filter(has=page.get_by_text("SAMPLE unavailable", exact=True))
    expect(reviewed.locator("dd")).to_have_text(str(sample["reviewed"]))
    expect(unavailable.locator("dd")).to_have_text(str(sample["unavailable"]))
    expect(page.locator("#labels .level")).to_have_text(f"Quality {detail['quality']}")
    current_pin = page.locator(".field").filter(has=page.get_by_text("Current release pin", exact=True))
    current = detail["declaration"]["current_release"]
    expect(current_pin).to_contain_text(f"{current['repo']} @ {current['revision']}")
    check_evidence(page)
    assert not inspect_state(page)["findings"]
    card.screenshot(path=str(journey.shots / f"pending-filtered-{theme}-{width}.png"))
    page.goto(root)
    expect(page.locator("#rows tr").first).to_be_visible()
    assert page.locator(".metrics").inner_text() == before


def test_filtering_canonical_keeps_unmeasured_siblings_separate(journey: Journey) -> None:
    page = journey.page
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    expect(page.locator("#rows tr").first).to_be_visible()
    payload = page.evaluate("async()=> (await fetch('api/sources')).json()")
    rows = [row for row in payload["sources"] if row["id"] in {"Marin:clean", "Marin:eval", "Marin:required"}]
    group = {
        "group_id": "containment-star",
        "canonical_source_id": "Marin:clean",
        "members": [
            {"source_id": "Marin:clean", "overlap_with_canonical": 1},
            {"source_id": "Marin:eval", "overlap_with_canonical": 0.97},
            {"source_id": "Marin:required", "overlap_with_canonical": 0.96},
        ],
    }
    for row in rows:
        row.update(
            duplicate_group=group,
            quality="SAMPLE",
            result="GOOD",
            intended_use="train" if row["id"] == "Marin:clean" else "eval",
        )
    page.route("**/api/sources*", lambda route: route.fulfill(json={**payload, "sources": rows}))
    page.reload()
    expect(page.locator("#rows tr")).to_have_count(1)
    expect(page.locator("#rows tr")).to_have_attribute("data-source-id", "Marin:clean")
    page.locator("#use").select_option("eval")
    check_view(page, rows, use="eval")
    expect(page.locator("#rows tr")).to_have_count(2)
    expect(page.locator(".duplicate-toggle")).to_have_count(0)
    page.locator('tr[data-source-id="Marin:eval"] .source-name').click()
    expect(page.locator("#detail-fields .overlap-note a")).to_have_attribute(
        "href", "source.html#source_id=Marin%3Aclean"
    )
    page.locator("#close-details").click()
    page.locator("#use").select_option("all")
    check_view(page, rows)
    page.locator(".duplicate-toggle").click()
    check_view(page, rows, expanded=True)
    expect(page.locator("#rows tr").first).to_have_attribute("data-source-id", "Marin:clean")


def test_table_bounds_rejects_header_words_wrapped_into_vertical_letters(journey: Journey) -> None:
    page = journey.page
    page.set_content(
        """<style>
        table {table-layout:fixed;width:40px;}
        th {font:16px/20px sans-serif;overflow-wrap:anywhere;}
        .header-label {display:block;width:15px;white-space:normal;}
        </style><table><thead><tr><th><span class="header-label">Examples</span></th></tr></thead>
        <tbody></tbody></table>"""
    )
    failures = page.locator("table").evaluate(TABLE_BOUNDS)
    assert any(row[:2] == ["header", "Examples"] for row in failures)
    page.locator("table").evaluate("e=>e.style.width='200px'")
    page.locator(".header-label").evaluate("e=>{e.style.whiteSpace='nowrap';e.style.width='auto';}")
    assert not page.locator("table").evaluate(TABLE_BOUNDS)


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1440, 390])
def test_source_cells_are_names_only_and_overlap_evidence_qualifies_colliding_names(
    journey: Journey, theme: str, width: int
) -> None:
    page = journey.page
    page.emulate_media(color_scheme=theme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    expect(page.locator("#rows tr").first).to_be_visible()
    payload = page.evaluate("async()=> (await fetch('api/sources')).json()")
    detail = page.evaluate("async()=> (await fetch('api/source?source_id=Marin%3Aclean')).json()")
    rows = [row for row in payload["sources"] if row["id"] in {"Marin:clean", "Marin:eval"}]
    for row in rows:
        row.update(display_name="aime24", duplicate_group=None, quality="SAMPLE", result="GOOD", partial_overlaps=[])
        row["consumer_findings"] = ["provider_schema_mismatch"] if row["id"] == "Marin:clean" else []
        row.update(
            origin="HuggingFaceH4" if row["id"] == "Marin:clean" else "Evalchemy",
            build_path="MarinSkyRL" if row["id"] == "Marin:clean" else "Evalchemy",
        )
        if row["id"] == "Marin:clean":
            row["partial_overlaps"] = [{"other_id": "Marin:eval", "source_fraction": 0.23, "other_fraction": 0.04}]
            detail["declaration"].update({key: row[key] for key in ["display_name", "origin", "build_path"]})
            detail.update(duplicate_group=None, partial_overlaps=row["partial_overlaps"])
    labels = [{key: row[key] for key in ["id", "display_name", "origin", "build_path"]} for row in rows]
    page.route("**/api/sources*", lambda route: route.fulfill(json={**payload, "sources": rows}))
    page.route("**/api/source-labels", lambda route: route.fulfill(json={"sources": labels}))
    page.route("**/api/source?source_id=Marin%3Aclean", lambda route: route.fulfill(json=detail))
    page.reload()
    check_view(page, rows)
    expect(page.locator("#rows tr")).to_have_count(2)
    expect(page.locator("#rows .overlap-note")).to_have_count(0)
    advisory = page.locator('tr[data-source-id="Marin:clean"] .evidence-link:not(.backends-link)')
    expect(advisory.locator(".consumer-advisory")).to_have_text("⚑")
    expect(advisory).to_have_attribute("title", "Consumer advisory: provider_schema_mismatch")
    expect(advisory).to_have_attribute("aria-label", "Evidence: consumer advisory")
    expect(page.locator('tr[data-source-id="Marin:eval"] .consumer-advisory')).to_have_count(0)
    cell = page.locator('tr[data-source-id="Marin:clean"] td').first
    cell.evaluate("e=>e.append(document.createTextNode('TaskTrove · Task Trove'))")
    with pytest.raises(ValueError):
        check_view(page, rows)
    cell.evaluate("e=>e.lastChild.remove()")
    cell.locator(".source-name").click()
    expect(page.locator("#detail-fields .overlap-note")).to_contain_text("HuggingFaceH4 · MarinSkyRL aime24")
    expect(page.locator("#detail-fields .overlap-note")).to_contain_text("Evalchemy · aime24")
    page.get_by_role("link", name="All quality evidence", exact=True).click()
    expect(page.locator("#title")).to_have_text("aime24")
    note = page.locator("#sections .overlap-note")
    expect(note).to_contain_text("23% of HuggingFaceH4 · MarinSkyRL aime24's tasks are in Evalchemy · aime24")
    expect(note).to_contain_text("4% of Evalchemy · aime24's tasks are in HuggingFaceH4 · MarinSkyRL aime24")
    expect(note.locator("a").first).to_have_attribute("href", "source.html#source_id=Marin%3Aeval")
    check_evidence(page)


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1440, 390])
def test_column_drag_keyboard_persistence_and_reset_preserve_table_geometry(
    journey: Journey, theme: str, width: int
) -> None:
    page = journey.page
    page.emulate_media(color_scheme=theme)
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    page.locator("#show-everything").click()
    expected_ids = served_source_ids(journey)
    expect(page.locator("#rows tr")).to_have_count(len(expected_ids))
    header = page.locator("#headers th").first
    before = header.evaluate("e=>e.getBoundingClientRect().width")
    rows = page.locator("#rows tr").evaluate_all("rows=>rows.map(r=>r.dataset.sourceId)")
    handle = page.get_by_role("separator", name="Resize Source column", exact=True)
    handle.scroll_into_view_if_needed()
    page.locator(".table-scroller").evaluate(
        "e=>{const h=e.querySelector('.column-resize-handle').getBoundingClientRect();"
        "e.scrollLeft+=h.left-e.getBoundingClientRect().left-e.clientWidth/2;}"
    )
    box = handle.bounding_box()
    assert box is not None
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] / 2 + 80, box["y"] + box["height"] / 2)
    page.mouse.up()
    changed = header.evaluate("e=>e.getBoundingClientRect().width")
    assert abs(changed - before - 80) < 2
    expect(header).to_have_attribute("aria-sort", "ascending")
    assert page.locator("#rows tr").evaluate_all("rows=>rows.map(r=>r.dataset.sourceId)") == rows
    page.reload()
    expected_ids = served_source_ids(journey)
    expect(page.locator("#rows tr")).to_have_count(len(expected_ids))
    assert abs(header.evaluate("e=>e.getBoundingClientRect().width") - changed) < 2
    handle.press("ArrowLeft")
    assert abs(header.evaluate("e=>e.getBoundingClientRect().width") - changed + 16) < 2
    for right in (False, True):
        page.locator(".table-scroller").evaluate("(e,r)=>e.scrollLeft=r?e.scrollWidth:0", right)
        assert page.locator(".source-table").evaluate(TABLE_BOUNDS) == []
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
    page.locator("#reset-column-widths").click()
    page.reload()
    expected_ids = served_source_ids(journey)
    expect(page.locator("#rows tr")).to_have_count(len(expected_ids))
    assert abs(header.evaluate("e=>e.getBoundingClientRect().width") - before) < 2


def test_column_resizing_works_when_viewer_storage_is_unavailable(journey: Journey) -> None:
    browser = journey.page.context.browser
    assert browser is not None
    with browser.new_context(viewport={"width": 390, "height": 1000}) as context:
        context.add_init_script(
            "Object.defineProperty(window,'localStorage',{get(){throw new DOMException('Denied','SecurityError')}})"
        )
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
        expect(page.locator("#rows tr").first).to_be_visible()
        handle = page.get_by_role("separator", name="Resize Source column", exact=True)
        initial = int(handle.get_attribute("aria-valuenow"))
        handle.press("ArrowRight")
        expect(handle).to_have_attribute("aria-valuenow", str(initial + 16))
        page.locator("#reset-column-widths").click()
        assert not errors
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")


def test_release_checker_accepts_real_catalog_and_rejects_corrupted_rendered_totals(journey: Journey) -> None:
    page = journey.page
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    expect(page.locator("#rows tr").first).to_be_visible()
    sources = page.evaluate("async()=> (await (await fetch('api/sources')).json()).sources")
    check_view(page, sources)
    page.locator("#show-everything").click()
    result = check_view(page, sources, include_hidden=True, expanded=True)
    assert result["total"] == result["shown"]
    page.locator("#metric-sources").evaluate("e=>e.textContent='0 / 0'")
    with pytest.raises(AssertionError):
        check_view(page, sources, include_hidden=True, expanded=True)
    root = page.url.split("#")[0]
    evidence_sweep(page, root, [row for row in sources if row["id"] in {"Marin:clean", "Marin:eval"}])
    page.locator(".field dd").first.evaluate("e=>e.style.transform='translateX(-100px)'")
    with pytest.raises(ValueError, match="Evidence overlap"):
        check_evidence(page)


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
    expected_ids = served_source_ids(journey)
    expect(page.locator("#rows tr")).to_have_count(len(expected_ids))
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
    catalog = page.request.get(f"{journey.origin}/a/{CATALOG_APP}/api/sources").json()
    expect(page.locator("#snapshot-info")).to_contain_text(catalog["snapshot_id"])
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_complete_sample_is_final_and_unfinished_reviews_are_provisional(journey: Journey, theme: str) -> None:
    page = journey.page
    page.emulate_media(color_scheme=theme)
    sources_url = f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/"
    page.goto(sources_url)
    page.locator("#show-all").check()
    complete = page.locator('tr[data-source-id="Marin:sample-complete"]')
    content_only = page.locator('tr[data-source-id="Marin:sample-content"]')
    expect(content_only.locator(".level")).to_have_text("SAMPLE")
    expect(content_only.locator(".result-provisional")).to_have_count(0)
    expect(complete.locator(".level")).to_have_text("SAMPLE")
    expect(complete.locator(".result-provisional")).to_have_count(0)
    expect(complete.locator(".result-cell")).to_have_text("Good")
    for name in (
        "sample-uncertain",
        "full-inflight",
        "full-uncertain",
        "full-false-complete",
        "sample-false-waiver",
        "sample-missing-six",
        "full-missing-five",
    ):
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
    page.goto(sources_url)
    page.locator("#show-all").check()
    tolerated = page.locator('tr[data-source-id="Marin:sample-missing-five"]')
    expect(tolerated.locator(".level")).to_have_text("SAMPLE")
    expect(tolerated.locator(".result-provisional")).to_have_count(0)
    tolerated.locator(".source-name").click()
    page.get_by_role("link", name="All quality evidence", exact=True).click()
    coverage = page.locator(".assessment-card").filter(has=page.get_by_role("heading", name="Content review coverage"))
    for label, value in (
        ("Status", "Final"),
        ("Strict coverage complete", "false"),
        ("Conclusive reviewed contracts", "95"),
        ("Unreviewed tasks", "5"),
    ):
        expect(coverage.locator(".field").filter(has=page.get_by_text(label, exact=True)).locator("dd")).to_have_text(
            value
        )
    expect(coverage).not_to_contain_text("5%")
    journey.shoot(f"sample-final-missing-{theme}")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_pending_review_is_quick_without_a_result(journey: Journey, theme: str) -> None:
    page = journey.page
    page.emulate_media(color_scheme=theme)
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    pending = page.locator('tr[data-source-id="Marin:review-pending"]')
    expect(pending).to_have_count(0)
    page.locator("#show-all").check()
    expect(pending.locator(".level")).to_have_text("QUICK")
    expect(pending.locator(".level")).to_have_attribute("title", re.compile("SAMPLE review in progress"))
    expect(pending.locator(".result-cell")).to_have_text("Not assessed")
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
@pytest.mark.parametrize("width", [1440, 390, 320])
def test_wide_enum_columns_scroll_without_overlapping(journey: Journey, theme: str, width: int) -> None:
    page = journey.page
    page.emulate_media(color_scheme=theme)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    page.locator("#show-all").check()
    expected_ids = served_source_ids(journey)
    expect(page.locator("#rows tr")).to_have_count(len(expected_ids))
    tooltip = page.locator('tr[data-source-id="Marin:clean"] .backends-link').get_attribute("title")
    assert tooltip is not None and "UNSAFE: too_many_snapshots, unbuilt_image" in tooltip
    assert "Unique actor snapshot identities: 11" in tooltip and "Snapshot cap: 10" in tooltip
    expect(page.get_by_role("columnheader", name="Daytona", exact=True)).to_have_count(0)
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
    expect(clean.locator("td").nth(1)).to_have_text("2")
    zero = page.locator('tr[data-source-id="Marin:filtered-zero"]')
    expect(zero.locator("td").nth(1)).to_have_text("0")
    expect(zero.locator(".filtration-cell .chip")).to_have_text("EXCLUDED")
    pending = page.locator('tr[data-source-id="Marin:filtered-pending"]')
    expect(pending.locator("td").nth(1)).to_have_text("Pending")
    expect(pending.locator(".filtration-cell .chip")).to_have_text("IN PROGRESS")
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
    expect(admitted_field).to_have_text("2")
    # Only the production backends render (dev-only survey rows never do; see test_backends).
    expect(page.locator(".backend-table tbody tr")).to_have_count(2)
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
    page.get_by_role("link", name="By source", exact=True).click()
    page.locator("#show-all").check()
    page.locator('tr[data-source-id="Marin:filtered-zero"] .source-name').click()
    page.get_by_role("link", name="All quality evidence", exact=True).click()
    zero_ledger = page.locator(".assessment-card").filter(has=page.get_by_role("heading", name="Filtration ledger"))
    expect(zero_ledger.get_by_text("EXCLUDED", exact=True)).to_be_visible()
    expect(
        zero_ledger.locator(".field")
        .filter(has=page.get_by_text("Admitted after all filtration", exact=True))
        .locator("dd")
    ).to_have_text("0")
    expect(zero_ledger).to_contain_text("unpinned environment: 3")
    expect(zero_ledger.get_by_role("link", name="Canonical oracle filter receipt \u2197")).to_have_attribute(
        "href", "https://filer.marina.oa.dev/#s3%3A%2F%2Ftest%2Foracle-filter.json"
    )
    expect(zero_ledger).to_contain_text("b" * 64)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


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
    expect(page.locator("#title")).to_have_text("Marin · clean")
    expect(page.locator("#daytona-evidence")).to_contain_text("UNSAFE: too_many_snapshots, unbuilt_image")
    fields = page.locator("#daytona-evidence .field")
    expect(
        fields.filter(has=page.get_by_text("Unique actor snapshot identities", exact=True)).locator("dd")
    ).to_have_text("11")
    expect(fields.filter(has=page.get_by_text("Snapshot cap", exact=True)).locator("dd")).to_have_text("10")
    expect(page.locator("#daytona-evidence")).to_contain_text("One actor declares an unbuilt image")
    expect(page.locator("#daytona-evidence")).to_contain_text("f" * 64)
    expect(page.get_by_role("link", name="How to make it safe: Daytona operations ↗")).to_have_attribute(
        "href",
        "https://github.com/marin-community/marin/blob/main/.agents/ops/daytona.md#snapshot-preflight-and-reclamation",
    )
    expect(page.locator("#fit .backend-table tbody tr")).to_have_count(2)
    expect(page.locator("#fit")).to_contain_text("docker image")
    expect(page.locator("#fit")).to_contain_text("packages lock")
    api = page.evaluate("async () => (await (await fetch('api/source?source_id=Marin%3Aclean')).json())")
    recorded = api["backend_capabilities"]["backends"]["docker-gvisor"]
    assert recorded["support"] == "partial"
    assert recorded["roles"]["agent"] == {"partial": 3}
    assert recorded["blockers"] == {"agent:cpus": 3}
    # The API keeps the dev-only survey rows; the fit table renders production backends only.
    expect(
        page.locator("#fit tbody tr").filter(has=page.get_by_role("cell", name="Docker + runsc (local dev)"))
    ).to_have_count(0)
    unsupported = page.locator("#fit tbody tr").filter(has=page.get_by_role("cell", name="LOCAL", exact=True))
    expect(unsupported.locator(".chip.quality-bad")).to_have_text("UNSUPPORTED")
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
    page.evaluate("location.hash = 'source_id=Marin%3Aeval'")
    expect(page.locator("#title")).to_have_text("Marin · eval")
    expect(page.locator("#daytona-evidence")).to_contain_text("Pending census")
    local = page.locator("#fit tbody tr").filter(has=page.get_by_role("cell", name="LOCAL", exact=True))
    # Columns: Backend, Overall, Reasons, Agent, Grader, Affected, Blockers, Validation errors.
    expect(local.locator("td").nth(3)).to_have_text("N/A (no role): 3")
    expect(local.locator("td").nth(4)).to_have_text("supported: 3")
    page.go_back()
    expect(page.locator("#title")).to_have_text("Marin · clean")
    expect(page.locator("#fit .backend-table tbody tr")).to_have_count(2)
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
    page.get_by_role("link", name="All quality evidence \u203a").click()
    expect(page.locator("#title")).to_have_text("Marin · unmeasured")
    page.evaluate("location.hash = 'source_id=Marin%3Afiltered-zero'")
    expect(page.locator("#title")).to_have_text("Marin · filtered-zero")
    expect(page.locator("#sections")).to_contain_text("EXCLUDED")
    page.evaluate("location.hash = 'source_id=Marin%3Asample-complete'")
    expect(page.locator("#title")).to_have_text("Marin · sample-complete")
    expect(page.locator("#labels .level")).to_have_text("Quality SAMPLE")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1440, 390])
def test_curriculum_parents_equal_displayed_children_and_remainder(journey: Journey, theme: str, width: int) -> None:
    page = journey.page
    page.emulate_media(color_scheme=theme)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    page.get_by_role("link", name="By curriculum", exact=True).click()
    expect(page.locator(".domain-card")).to_have_count(2)
    earth = page.locator('[data-subject="D05"]')
    expect(earth.locator(".domain-count")).to_have_text("7,088")
    expect(earth.locator(".map-remainder .map-area-count")).to_have_text("7,078")

    def assert_partition() -> None:
        assert (
            page.locator("#curriculum-map").evaluate(
                """map => {
                const count = element => Number(element.textContent.replaceAll(',', ''));
                const failures = [];
                const check = (parent, children) => {
                    const sum = [...children.children].reduce((total, child) => total + count(
                        child.matches('.map-remainder') ? child.querySelector('.map-area-count') :
                            child.querySelector(':scope > .map-area > .map-area-count')), 0);
                    if (count(parent) !== sum) failures.push([parent.textContent, sum]);
                };
                for (const card of map.querySelectorAll('.domain-card'))
                    check(card.querySelector('.domain-count'), card.querySelector('.map-topics'));
                for (const branch of map.querySelectorAll('.map-branch')) {
                    const children = branch.querySelector(':scope > .map-children');
                    if (children) check(branch.querySelector(':scope > .map-area > .map-area-count'), children);
                }
                return failures;
            }"""
            )
            == []
        )
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")

    assert_partition()
    earth.get_by_role("button", name="Other/unlisted (subject only)", exact=True).click()
    expect(page.locator("#area-summary")).to_contain_text("7,078 task instances")
    expect(page.locator("#area-sources b")).to_have_text(["7,076", "2"])
    expect(page.locator("#source")).to_have_value("")
    expect(page.locator("#task-list .task-row")).to_have_count(2)
    expect(page.locator("#task-list .task-column-label")).to_have_text(["Source", "Source"])
    expect(page.locator("#source-breakdown")).not_to_have_attribute("open", "")
    expect(page.locator("#task-list")).to_be_in_viewport()
    page.locator("#source-breakdown summary").click()
    with page.expect_navigation():
        page.get_by_role("link", name="Source evidence ↗", exact=True).first.click()
    expect(page.locator("#title")).to_have_text("Marin · clean")
    page.go_back()
    expect(page.locator(".domain-card")).to_have_count(2)
    earth.get_by_role("button", name="Other/unlisted (subject only)", exact=True).click()
    expect(page.locator("#task-list .task-row")).to_have_count(2)
    expect(page.locator("#task-list")).to_contain_text("earth-only")
    expect(page.locator("#task-list")).to_contain_text("other-source-example")
    page.locator("#task-list .task-row").filter(has_text="earth-only").click()
    expect(page.locator("#detail-task-id")).to_have_text("earth-only")
    expect(page.locator("#detail-matches .score").first).to_be_visible()
    assert page.locator("#detail-matches .score, #detail-matches th").evaluate_all(
        """cells => cells.filter(e=>e.checkVisibility()).every(e=>{
          const range=document.createRange(); range.selectNodeContents(e);
          return range.getClientRects().length===1;
        })"""
    )
    page.locator("#close-detail").click()
    page.locator("#close-tasks").click()
    page.locator("#domain").select_option("D05")
    expect(page.locator('[data-node-id="earth"] > .map-area .map-area-name')).to_have_text(
        "Capability groups for Earth, Atmospheric, Ocean & Environmental Sciences"
    )
    empty = page.locator('[data-node-id="weather"]')
    expect(empty.locator(".map-area-count")).to_have_text("0")
    expect(empty.locator(".map-area")).to_have_class(re.compile(r"zero"))
    empty.get_by_role("button", name="Weather prediction", exact=True).click()
    expect(page.locator("#area-summary")).to_contain_text("0 task instances")
    expect(page.locator("#task-list .task-row")).to_have_count(0)
    expect(page.locator("#task-message")).to_contain_text("No task examples")
    page.locator("#area-sort").select_option("count")
    assert_partition()
    page.locator("#domain").select_option("")
    page.locator("#search").fill("Addition")
    expect(page.locator(".domain-card")).to_have_count(1)
    expect(page.locator(".map-topics > .map-remainder:not(.map-search-hidden) .map-area-count")).to_have_text("4")
    expect(page.locator(".map-search-hidden .map-area-count")).to_have_text("8")
    assert_partition()
    page.locator("#search").fill("")
    page.locator("#domain").select_option("D01")
    expect(page.locator(".domain-card")).to_have_count(1)
    expect(page.locator(".map-topics > .map-remainder .map-area-count")).to_have_text("4")
    assert_partition()


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1440, 390])
def test_header_distinguishes_catalog_totals_from_displayed_rows(journey: Journey, theme: str, width: int) -> None:
    page = journey.page
    page.emulate_media(color_scheme=theme)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    expect(page.locator("#rows tr").first).to_be_visible()
    sources = page.evaluate("async () => (await (await fetch('api/sources')).json()).sources")
    # A complete 110-task partition, including hidden admissions and folded EVAL tasks.
    selected = {source["id"]: source for source in sources}
    rows = []
    for name, population, admitted, excluded, quality, use in [
        ("clean", 40, 30, 10, "SAMPLE", "train"),
        ("eval", 30, 20, 10, "FULL", "eval"),
        ("required", 20, None, None, "SAMPLE", "train"),
        ("filtered-zero", 10, 0, 10, "QUICK", "train"),
        ("stale-clean", 10, 10, 0, "QUICK", "train"),
    ]:
        row = selected[f"Marin:{name}"]
        row.update(quality=quality, result="BAD" if name == "filtered-zero" else "GOOD", intended_use=use)
        row["filtration"] = {
            "population_count": population,
            "admitted_count": admitted,
            "example_count": admitted,
            "excluded_count": excluded,
            "state": "pending_oracle" if admitted is None else "static_retained",
            "reason": "Count reconciliation fixture",
        }
        if name in {"clean", "eval"}:
            row["duplicate_group"] = {
                "group_id": "fixture-duplicates",
                "canonical_source_id": "Marin:clean",
                "members": [
                    {"source_id": "Marin:clean", "overlap_with_canonical": 1},
                    {"source_id": "Marin:eval", "overlap_with_canonical": 1},
                ],
            }
        rows.append(row)
    page.route("**/api/sources*", lambda route: route.fulfill(json={"sources": rows, "synced_at": "2026-11-14"}))
    page.reload()
    total = "60 tasks admitted (TRAIN 40 · EVAL 20) · 20 tasks pending oracle filtration · 30 tasks excluded"
    expect(page.locator("#metric-examples")).to_have_text(total)
    expect(page.locator("#metric-sources")).to_have_text("110 / 60")
    expect(page.locator("#metric-quality")).to_have_text("20 / 60 / 30")
    expect(page.locator("#metric-quality-shown")).to_have_text("Shown 0 / 60 / 0")
    expect(page.locator("#task-reconciliation")).to_have_text(
        "10 tasks hidden (excluded status); 0 admitted · 10 tasks hidden (QUICK/BAD); 10 admitted · "
        "30 tasks folded (duplicates/subsets); 20 admitted"
    )
    expect(page.locator("#metric-examples-shown")).to_have_text(
        "Shown 30 tasks admitted (TRAIN 30 · EVAL 0) · 20 tasks pending oracle filtration · 10 tasks excluded"
    )
    # Read the rendered counts; both populations must reconcile, not only the text labels.
    shown = [int(value) for value in re.findall(r"\d+", page.locator("#metric-sources").inner_text())]
    omitted = [int(value) for value in re.findall(r"\d+", page.locator("#task-reconciliation").inner_text())]
    assert shown[1] + omitted[0] + omitted[2] + omitted[4] == shown[0]
    admitted_shown = int(re.findall(r"\d+", page.locator("#metric-examples-shown").inner_text())[0])
    admitted_total = int(re.findall(r"\d+", page.locator("#metric-examples").inner_text())[0])
    assert admitted_shown + omitted[1] + omitted[3] + omitted[5] == admitted_total
    page.locator("#difficulty-model").select_option("small")
    page.locator("#use").select_option("train")
    expect(page.locator("#task-reconciliation")).to_have_text(
        "10 tasks hidden (excluded status); 0 admitted · 10 tasks hidden (QUICK/BAD); 10 admitted · "
        "30 tasks filtered; 20 admitted"
    )
    page.locator("#show-everything").click()
    expect(page.locator("#difficulty-model")).to_have_value("all")
    expect(page.locator("#use")).to_have_value("all")
    expect(page.locator("#rows tr")).to_have_count(5)
    expect(page.locator("#metric-sources")).to_have_text("110 / 110")
    expect(page.locator("#metric-examples-shown")).to_have_text("Shown " + total)
    expect(page.locator("#metric-quality-shown")).to_have_text("Shown 20 / 60 / 30")
    page.reload()
    expect(page.locator("#rows tr")).to_have_count(5)
    expect(page.locator("#metric-sources")).to_have_text("110 / 110")
    expect(page.locator("#metric-examples-shown")).to_have_text("Shown " + total)
    page.locator("#clear-filters").click()
    expect(page.locator("#rows tr")).to_have_count(2)
    expect(page.locator("#difficulty-model")).to_have_value("all")
    assert page.locator(".metrics > div").evaluate_all(
        """cards => cards.every((card, i) => {
            const box = card.getBoundingClientRect(), next = cards[i + 1]?.getBoundingClientRect();
            if (next && Math.abs(box.top - next.top) < 1 && box.right > next.left + 1) return false;
            return [...card.children].every(child => {
                const bounds = child.getBoundingClientRect();
                return bounds.left >= box.left && bounds.right <= box.right && child.scrollWidth <= child.clientWidth;
            });
        })"""
    )
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.unroute("**/api/sources*")
    page.locator(".atlas-tabs").get_by_role("link", name="Getting started", exact=True).click()
    count = page.locator('button.count[data-leaf="sample"]')
    expect(count).to_have_text(re.compile(r"^\d+$"))
    count.click()
    table = page.locator(".leaf-sources")
    expect(table.locator("thead th")).to_have_text(["Source", "Origin", "Execution oversight", "Daytona"])
    expect(page.locator("#leaf-rows tr").first).to_be_visible()
    assert table.evaluate(
        "t => [...t.tBodies[0].rows].every(r => r.cells.length === 4 && [...r.cells].every((c,i,a) => "
        "!a[i+1] || c.getBoundingClientRect().right <= a[i+1].getBoundingClientRect().left + 1))"
    )
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1440, 390])
def test_confidence_changes_counts_and_pooled_samples_without_requests(journey: Journey, theme: str, width: int) -> None:
    page = journey.page
    page.emulate_media(color_scheme=theme)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    page.get_by_role("link", name="By curriculum", exact=True).click()
    page.goto(page.url.split("#")[0] + "#release=confidence")
    expect(page.locator("#confidence option")).to_have_count(5)
    expect(page.locator("#capability-count")).to_have_text("7,112")
    page.locator('[data-subject="D05"] .domain-count').click()
    expect(page.locator("#task-list .task-row")).to_have_count(3)
    requests = []
    page.on("request", lambda request: requests.append(request.url) if "/api/" in request.url else None)
    for index in range(5):
        page.locator("#confidence").select_option(str(index))
        expect(page.locator("#capability-count")).to_have_text("7,112" if index < 2 else "30")
        expect(page.locator("#task-list .task-row")).to_have_count(3)
        assert (
            page.locator("#outcomes b").evaluate_all(
                "nodes => nodes.reduce((sum, node) => sum + Number(node.textContent.replaceAll(',', '')), 0)"
            )
            == 7112
        )
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert requests == []
    page.locator('[data-subject="D05"]').get_by_role("button", name="Other/unlisted (subject only)", exact=True).click()
    expect(page.locator("#task-list .task-row")).to_have_count(2)
    expect(page.locator("#area-summary")).to_contain_text("7,078 task instances")
    requests.clear()
    page.locator("#confidence").select_option("0")
    expect(page.locator("#task-list .task-row")).to_have_count(0)
    expect(page.locator("#area-summary")).to_contain_text("0 task instances")
    page.locator("#confidence").select_option("4")
    expect(page.locator("#task-list .task-row")).to_have_count(2)
    assert requests == []


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1440, 390])
def test_grader_filter_and_mixed_profile_keep_counts_on_evidence(journey: Journey, theme: str, width: int) -> None:
    page = journey.page
    page.emulate_media(color_scheme=theme)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    page.locator("#show-all").check()
    clean = page.locator('tr[data-source-id="Marin:clean"]')
    expect(clean.locator(".grader-cell")).to_have_text("Mixed")
    expect(clean.locator(".grader-cell .chip")).to_have_count(1)
    page.locator("#grader").select_option("mixed")
    expect(page.locator("#rows tr")).to_have_count(1)
    clean.get_by_role("link", name="Evidence →").click()
    profile = page.locator(".assessment-card").filter(has=page.get_by_role("heading", name="Grader profile", exact=True))
    expect(profile).to_contain_text("script")
    expect(profile).to_contain_text("llm-judge")
    expect(profile.locator('a[href="https://example.org/derived-receipt.json"]')).to_have_attribute(
        "href", "https://example.org/derived-receipt.json"
    )
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


@pytest.mark.parametrize("full", [False, True], ids=["representative", "full_browser"])
def test_release_checker_sweeps_exported_confidence_levels_and_historical_population(journey: Journey, tmp_path, full):
    page = journey.page
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    expect(page.locator("#rows tr").first).to_be_visible()
    root = page.url.split("#")[0]
    releases = page.evaluate("async () => (await fetch('api/curriculum/releases')).json()")["releases"]
    views = (curriculum_full_sweep if full else curriculum_sweep)(page, root, releases, tmp_path)
    if not full:
        assert len(views) == 4
        assert {view["use"] for view in views} == {"all"}
        assert all(view["domain"] and not view["failures"] for view in views)
        assert len(list(tmp_path.glob("curriculum-*.png"))) == 4
        return
    levels = [view for view in views if "confidence" in view]
    assert len(levels) == 5 * 3 * 4
    assert {view["use"] for view in levels} == {"all", "train", "eval"}
    assert {(view["theme"], view["width"]) for view in levels} == {
        ("light", 1440),
        ("light", 390),
        ("dark", 1440),
        ("dark", 390),
    }
    assert all(not view["failures"] for view in views)


def test_frozen_declaration_oversight_cards_show_derived_level_receipts_and_notes(journey: Journey) -> None:
    """Frozen pre-derivation declarations keep one derived level, real receipt links and historical notes."""
    page = journey.page
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    expect(page.locator("#rows tr").first).to_be_visible()
    root = page.url.split("#")[0]
    base = page.evaluate("async()=> (await fetch('api/source?source_id=Marin%3Afull-complete')).json()")
    full_binding = {
        "source_id": "Marin:frozen-full",
        "review_receipt": {"uri": "s3://marin-us-east-02a/frozen/full/review.json", "sha256": "d" * 64},
        "coverage_complete": True,
        "finalization": "complete",
    }
    historical = {
        "campaign": "Frozen campaign",
        "recipe_revision": "2026.01.01",
        "verification": "Hand-written historical verification prose",
        "reviewed": 3,
        "admitted": 2,
    }
    cases = [
        ("full", "FULL", "full", full_binding, None),
        ("sample", "SAMPLE", "sample", None, "https://example.org/frozen-sample-review.json"),
        # A stale hand-written FULL claim must never override the effective QUICK level.
        ("quick", "QUICK", "full", None, "https://example.org/frozen-release-qc.json"),
    ]
    for slug, quality, declared, binding, qc_url in cases:
        detail = json.loads(json.dumps(base))
        detail["quality"] = quality
        declaration = detail["declaration"]
        declaration.pop("recipe_qa_history", None)
        declaration.pop("full_content_result_binding", None)
        declaration["oversight"] = declared
        declaration["oversight_evidence"] = {**historical, "level": declared}
        if binding is not None:
            declaration["full_content_result_binding"] = binding
        declaration["independent_qc"] = (
            None
            if qc_url is None
            else {"verdict": "good", "method": "Fixture QC", "reviewed": 2, "evidence_url": qc_url}
        )
        source_id = f"Marin:frozen-{slug}"
        page.route(
            f"**/api/source?source_id=Marin%3Afrozen-{slug}",
            lambda route, _request, detail=detail: route.fulfill(json=detail),
        )
        page.goto(root + "source.html#" + urlencode({"source_id": source_id}))
        expect(page.locator("#labels .chip").first).to_have_text(f"Quality {quality}")
        oversight = page.locator("#sections .assessment-card").filter(
            has=page.get_by_role("heading", name="Oversight evidence")
        )
        notes = page.locator("#sections .assessment-card").filter(
            has=page.get_by_role("heading", name="Historical recipe QA notes")
        )
        expect(notes.locator("dt")).to_have_text(["Campaign", "Recipe revision", "Verification", "Reviewed", "Admitted"])
        expect(notes).to_contain_text(historical["verification"])
        if quality == "QUICK":
            content = oversight.locator(".field").filter(has=page.locator("dt", has_text="Content review"))
            expect(content.locator("dd")).to_contain_text("Conversion-only")
            expect(
                oversight.locator(".field").filter(has=page.locator("dt", has_text=re.compile("Receipt|QA evidence")))
            ).to_have_count(0)
            expect(oversight).not_to_contain_text("frozen-release-qc.json")
        elif binding is not None:
            receipt = oversight.locator(".field").filter(has=page.locator("dt", has_text="Receipt"))
            expect(receipt).to_have_count(1)
            expect(receipt.locator("dd")).to_contain_text(full_binding["review_receipt"]["uri"])
        else:
            evidence = oversight.locator(".field").filter(has=page.locator("dt", has_text="QA evidence"))
            expect(evidence).to_have_count(1)
            expect(evidence.locator("dd a").first).to_have_attribute("href", qc_url)
        check_evidence(page)


def test_default_source_column_bounds_long_names_and_viewer_can_expand(journey: Journey) -> None:
    page = journey.page
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    payload = page.evaluate("async () => (await fetch('api/sources')).json()")
    source = next(row for row in payload["sources"] if row["id"] == "Marin:clean")
    source["display_name"] = "A long descriptive source name with provenance encoded in its dataset name " * 4
    page.route("**/api/sources*", lambda route: route.fulfill(json=payload))
    page.reload()
    page.locator("#show-all").check()
    header = page.locator("#headers th").first
    expect(page.locator('tr[data-source-id="Marin:clean"] .source-name')).to_have_text(source["display_name"].strip())
    before = header.evaluate("element => element.getBoundingClientRect().width")
    assert before <= 330
    handle = page.get_by_role("separator", name="Resize Source column", exact=True)
    handle.press("ArrowRight")
    handle.press("Shift+ArrowRight")
    assert header.evaluate("element => element.getBoundingClientRect().width") >= before + 78
    assert page.locator(".source-table").evaluate(TABLE_BOUNDS) == []


def test_confidence_and_map_remain_usable_while_sample_pages_stream(journey: Journey) -> None:
    page = journey.page
    page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
    page.get_by_role("link", name="By curriculum", exact=True).click()
    page.goto(page.url.split("#")[0] + "#release=confidence")
    expect(page.locator("#confidence option")).to_have_count(5)
    payload = page.evaluate(
        "async () => (await fetch('api/curriculum/tasks?release=confidence&sample_pool=true&' + "
        "'subject_id=D05&limit=1000')).json()"
    )
    pending = []

    def sample_page(route) -> None:
        if "cursor=held" in route.request.url:
            pending.append(route)
        else:
            route.fulfill(json={**payload, "tasks": payload["tasks"][:2], "next_cursor": "held"})

    page.route("**/api/curriculum/tasks?**", sample_page)
    page.locator("#domain").select_option("D05")
    expect(page.locator("#task-list .task-row")).to_have_count(2)
    expect(page.locator("#task-list")).to_have_attribute("aria-busy", "true")
    expect(page.locator("#curriculum-map")).not_to_have_attribute("aria-busy", "true")
    page.get_by_role("button", name="Climate models", exact=True).click()
    expect(page.locator("#task-list .task-row")).to_have_count(2)
    requests = []
    page.on("request", lambda request: requests.append(request.url) if "/api/" in request.url else None)
    page.locator("#confidence").select_option("4")
    expect(page.locator("#task-list .task-row")).to_have_count(1)
    assert len(pending) == 1
    pending[0].fulfill(json={**payload, "tasks": payload["tasks"][2:], "next_cursor": ""})
    expect(page.locator("#task-list")).to_have_attribute("aria-busy", "false")
    expect(page.locator("#task-list .task-row")).to_have_count(1)
    page.locator("#confidence").select_option("0")
    expect(page.locator("#task-list .task-row")).to_have_count(3)
    assert requests == []


FULL_BROWSE_APP = "fb11c931-5861-4878-8bb5-a964d652b45f"
FULL_BROWSE_SOURCE = "Marin:full-browse"
FULL_BROWSE_KEY = "full-browse"
FULL_BROWSE_POPULATION = 260
FULL_BROWSE_CYCLE = ["missing_context", "wrong_reference", "answer_leakage", "malformed", "rubric_mismatch", "ambiguity"]


def full_browse_records() -> list:
    """One whole-population FULL panel: 260 typed reviews, two unavailable, no packets."""
    records = []
    for index in range(FULL_BROWSE_POPULATION):
        task_id = f"task-{index:04d}"
        if index % 130 == 129:
            records.append(
                ReviewRecord(task_id=task_id, status="unavailable", verdict=None, detail="Response truncated.")
            )
            continue
        defects = [Defect(FULL_BROWSE_CYCLE[index % len(FULL_BROWSE_CYCLE)])]
        if index % 13 == 5:
            verdict = ReviewVerdict(
                task_id=task_id,
                quality=Quality.GOOD,
                confidence=Confidence.HIGH,
                reference_status=ReferenceStatus.CONSISTENT,
                defects=[],
                evidence="Clean retained task.",
            )
        else:
            verdict = ReviewVerdict(
                task_id=task_id,
                quality=Quality.ISSUES,
                confidence=Confidence.LOW if index % 3 == 0 else Confidence.HIGH,
                reference_status=ReferenceStatus.CONSISTENT,
                defects=defects,
                evidence=f"Recorded finding for {task_id}.",
            )
        records.append(ReviewRecord(task_id=task_id, status="reviewed", verdict=verdict, detail=""))
    return records


@pytest.fixture
def full_review_catalog(marina_kernel, marina_database: DatabaseSpec) -> Iterator[None]:
    """One source whose whole-population review is bound with an accepted waiver."""
    records = full_browse_records()
    result = recompute_review_result(records)
    waiver = {"uri": "https://example.org/full-browse-waiver.json", "sha256": "a" * 64}
    base = next(declaration for _, declaration in review_declarations() if declaration.id == "Marin:full-waiver")
    declaration = base.model_copy(
        update={
            "id": FULL_BROWSE_SOURCE,
            "license_receipt": fixture_license(FULL_BROWSE_SOURCE, base.dataset_id, base.dataset_revision),
            "name": "full-browse",
            "display_name": "full-browse",
            "upstream_dataset": "full-browse",
            "upstream_name": "full-browse",
            "task_count": FULL_BROWSE_POPULATION,
            "within_source_dedup": deduplication_evidence("fixture", "1", "1", FULL_BROWSE_POPULATION),
            "quality_result": result,
            "quality_result_finalization": "accepted_incomplete",
            "quality_result_waiver": waiver,
            "independent_qc": {
                "verdict": result.verdict.value,
                "method": "Whole-population review bound with the finalization waiver",
                "reviewed": result.reviewed + result.uncertain_count,
                "evidence_url": "https://example.org/full-browse-review",
            },
        }
    )
    completeness = review_completeness(declaration, [], [])
    engine = engine_for(marina_database, applet_schema(UUID(FULL_BROWSE_APP)))
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO atlas_sources (load_id, id, declaration, assessment_key, latest_attempt_key, "
                "latest_attempt_status, reason, curation, diagnostics, review_completeness) "
                "VALUES ('lexical', :id, CAST(:declaration AS JSONB), :key, NULL, 'skipped', NULL, NULL, '[]'::jsonb, "
                "CAST(:completeness AS JSONB))"
            ),
            {
                "id": FULL_BROWSE_SOURCE,
                "declaration": declaration.model_dump_json(),
                "key": FULL_BROWSE_KEY,
                "completeness": json.dumps(completeness),
            },
        )
        connection.execute(
            text(
                "INSERT INTO atlas_assessments VALUES ('lexical', :key, :source, 'full', "
                '\'{"model": "glm", "revision": "pinned", "max_tokens": 8192, "max_attempts": 3, '
                "\"retry_max_tokens\": 16384}', 'Review every retained task.', "
                '\'{"status": "complete", "verification": null, "admission_counts": {"admitted": 0, '
                "\"excluded\": 0}}', 'https://example.org/full-browse-review', '[]'::jsonb)"
            ),
            {"key": FULL_BROWSE_KEY, "source": FULL_BROWSE_SOURCE},
        )
        connection.execute(
            text("INSERT INTO atlas_populations VALUES ('lexical', :key, 'admitted', :population, :population)"),
            {"key": FULL_BROWSE_KEY, "population": FULL_BROWSE_POPULATION},
        )
        connection.execute(
            text("INSERT INTO atlas_populations VALUES ('lexical', :key, 'excluded', 0, 0)"), {"key": FULL_BROWSE_KEY}
        )
        for record in records:
            verdict = record.verdict
            reviewed = record.status == "reviewed" and verdict is not None
            connection.execute(
                text(
                    """
                INSERT INTO atlas_items (load_id, pipeline_key, task_id, cohort, status, verdict, explanation,
                    evidence_ids, packet_uri, has_task, omissions, exclusion_reason,
                    confidence, reference_status, defects, provenance, replaced_task_sha256, repair_advisories)
                VALUES ('lexical', :key, :task, 'admitted', :status, :verdict, :explanation, '[]'::jsonb,
                    '', false, '[]'::jsonb, NULL, :confidence, :reference, CAST(:defects AS JSONB), NULL, NULL,
                    '[]'::jsonb)
            """
                ),
                {
                    "key": FULL_BROWSE_KEY,
                    "task": record.task_id,
                    "status": "reviewed" if reviewed else "unavailable",
                    "verdict": str(verdict.quality) if reviewed else "unknown",
                    "explanation": verdict.evidence if reviewed else record.detail or "Judgment unavailable",
                    "confidence": str(verdict.confidence) if reviewed else None,
                    "reference": str(verdict.reference_status) if reviewed else None,
                    "defects": json.dumps([str(defect) for defect in verdict.defects]) if reviewed else "[]",
                },
            )
    try:
        with engine.begin() as connection:
            migrate(connection)
        yield
    finally:
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM atlas_sources WHERE load_id='lexical' AND id=:source"), {"source": FULL_BROWSE_SOURCE}
            )
            for table in ["atlas_audit_judgments", "atlas_audit_sample_tasks"]:
                connection.execute(
                    text(
                        f"DELETE FROM {table} WHERE sample_id IN "
                        "(SELECT sample_id FROM atlas_audit_samples WHERE pipeline_key = :key)"
                    ),
                    {"key": FULL_BROWSE_KEY},
                )
            connection.execute(
                text("DELETE FROM atlas_audit_samples WHERE pipeline_key = :key"), {"key": FULL_BROWSE_KEY}
            )
            for table in ["atlas_items", "atlas_populations", "atlas_assessments"]:
                connection.execute(
                    text(f"DELETE FROM {table} WHERE load_id='lexical' AND pipeline_key=:key"), {"key": FULL_BROWSE_KEY}
                )
        engine.dispose()


def test_full_review_browse_paging_filters_and_audit(journey: Journey, full_review_catalog: None) -> None:
    """A person pages through a whole-population review, filters by defect and audits it."""
    page = journey.page
    page.goto(
        f"{journey.origin}/a/{FULL_BROWSE_APP}/v/1/assessment.html?source={quote(FULL_BROWSE_SOURCE)}",
        wait_until="domcontentloaded",
    )
    page.evaluate("() => document.fonts.ready")
    journey.sees("Review scope: FULL")
    journey.sees("whole-source waiver accepted for the unreviewed remainder")
    journey.sees("1-100 of 260 matching reviews")
    journey.shoot("full-browse-page-1")

    # Paging moves through the whole population without reloading everything at once.
    journey.click("Next page →")
    journey.sees("101-200 of 260 matching reviews")
    journey.sees("Some issues · task-0101")
    journey.shoot("full-browse-page-2")

    # The defect toggles carry whole-population counts; filtering narrows server-side.
    ambiguous = sum(
        1
        for record in full_browse_records()
        if record.verdict and "ambiguity" in [str(defect) for defect in record.verdict.defects]
    )
    journey.click(re.compile(r"Ambiguity \(\d+\)"))
    journey.sees(f"1-{ambiguous} of {ambiguous} matching reviews")
    journey.shoot("full-browse-filtered")
    journey.click(re.compile(r"Ambiguity \(\d+\)"))

    # A task's evidence opens from the row itself: these whole-population rows have no packet.
    journey.click("Some issues · task-0002")
    journey.sees("Recorded finding for task-0002.")
    journey.shoot("full-browse-evidence")

    # Audit mode draws its seeded sample from the full reviewed population.
    journey.fill("Seed", "journey-full-260")
    journey.click("Draw sample")
    journey.sees("Sample stratified · seed journey-full-260 · 40 tasks")
    journey.sees("No judgments recorded yet.")
    journey.shoot("full-browse-audit")


def test_table_projection_preserves_every_rendered_source_and_dialog(journey: Journey) -> None:
    page = journey.page
    root = f"{journey.origin}/a/{CATALOG_APP}/"
    complete = page.request.get(root + "api/sources").json()
    compact = page.request.get(root + "api/sources?projection=table").json()
    assert len(compact["sources"]) == len(complete["sources"])
    assert all("quality_result" not in source for source in compact["sources"])

    def capture(payload):
        page.route("**/api/sources?projection=table", lambda route: route.fulfill(json=payload))
        page.goto(root)
        expect(page.locator("#rows tr").first).to_be_visible()
        page.locator("#show-everything").click()
        table = page.locator("#rows").evaluate(
            "e=>({text:e.innerText,titles:[...e.querySelectorAll('[title]')].map(x=>x.title),"
            "links:[...e.querySelectorAll('a')].map(x=>x.getAttribute('href'))})"
        )
        metrics = page.locator(".metrics").inner_text()
        dialogs = []
        for button in page.locator("#rows .source-name").all():
            button.click()
            dialogs.append(
                page.locator("#details").evaluate(
                    "e=>({text:e.innerText,titles:[...e.querySelectorAll('[title]')].map(x=>x.title),"
                    "links:[...e.querySelectorAll('a')].map(x=>x.getAttribute('href'))})"
                )
            )
            page.locator("#close-details").click()
        page.unroute("**/api/sources?projection=table")
        return table, metrics, dialogs

    assert capture(compact) == capture(complete)
