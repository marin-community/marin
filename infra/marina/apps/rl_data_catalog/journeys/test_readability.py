# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Every Atlas page remains readable in both themes and at narrow widths."""

import pytest
from marina.journeys import Journey
from playwright.sync_api import expect

from infra.marina.apps.rl_data_catalog.readability import AUDIT, readability_smoke, readability_sweep


@pytest.mark.timeout(600)
def test_rendered_contrast_readability_and_reflow(journey: Journey, readability_catalog: None) -> None:
    page = journey.page.context.new_page()
    report = readability_smoke(
        page, f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/", journey.shots / "readability-smoke"
    )
    page.close()
    assert report["violations"] == 0


@pytest.mark.timeout(600)
def test_full_browser_contrast_readability_and_reflow(journey: Journey, readability_catalog: None) -> None:
    page = journey.page.context.new_page()
    report = readability_sweep(
        page, f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/", journey.shots / "readability"
    )
    page.close()
    assert (
        report["violations"] == 0
    ), f"{report['violations']} readability violations; see {journey.shots}/readability/readability.json"


def test_checker_measures_composited_text_and_rejects_the_instruction_regression(journey: Journey) -> None:
    page = journey.page
    page.set_content(
        """<style>
        body {background:white; color:black; font:16px/1.5 sans-serif;}
        #alpha {color:rgba(0,0,0,.5);}
        dialog {background:white; color:black;}
        #instructions {background:rgb(241,244,239); color:rgb(217,229,221);}
    </style><p id="solid">Readable text</p><p id="alpha">Translucent text</p>
    <dialog><pre id="instructions">Public instructions</pre></dialog>"""
    )
    result = page.evaluate(AUDIT, {})
    solid = next(row for row in result["measurements"] if row["selector"] == "#solid" and row["rule"] == "text-contrast")
    alpha = next(row for row in result["findings"] if row["selector"] == "#alpha" and row["rule"] == "text-contrast")
    assert solid["measured"] == 21
    assert alpha["measured"] == pytest.approx(3.9766530249)
    page.locator("dialog").evaluate("e=>e.showModal()")
    regression = page.evaluate(AUDIT, {})
    instruction = next(
        row for row in regression["findings"] if row["selector"] == "#instructions" and row["rule"] == "text-contrast"
    )
    assert instruction["measured"] == pytest.approx(1.16816697065)
    assert not any(row["selector"] == "#solid" for row in regression["measurements"])


def test_focus_checker_distinguishes_partial_and_complete_occlusion(journey: Journey) -> None:
    page = journey.page
    page.set_content(
        """<style>
        body {margin:0; background:white; color:black;}
        button {position:fixed; left:20px; top:20px; width:120px; height:60px; outline:3px solid black;}
        #overlay {position:fixed; left:0; top:45px; width:200px; height:100px; background:white; z-index:10;}
        </style><button>Target</button><div id="overlay"></div>"""
    )
    page.keyboard.press("Tab")
    partial = page.evaluate(AUDIT, {"mode": "focus"})
    assert not any(row["rule"] == "focus-obscured" for row in partial["findings"])
    page.locator("#overlay").evaluate("e=>{e.style.top='0';e.style.height='200px';}")
    complete = page.evaluate(AUDIT, {"mode": "focus"})
    assert any(row["rule"] == "focus-obscured" for row in complete["findings"])


def test_focus_checker_measures_disjoint_wrapped_link_fragments(journey: Journey) -> None:
    page = journey.page
    page.set_viewport_size({"width": 300, "height": 200})
    page.set_content(
        """<style>
        body {margin:0;width:300px;background:white;color:black;font:16px/24px monospace;}
        .prefix {display:inline-block;width:280px;}
        a {outline:3px solid black;outline-offset:3px;}
        #overlay {display:none;position:fixed;background:white;z-index:10;}
        </style><span class="prefix">Prefix</span><a href="#">X Y</a><div id="overlay"></div>"""
    )
    page.keyboard.press("Tab")
    exposed = page.evaluate(AUDIT, {"mode": "focus"})
    assert not any(row["rule"] == "focus-obscured" for row in exposed["findings"])
    page.locator("#overlay").evaluate("e=>{e.style.cssText+='display:block;left:270px;top:0;width:30px;height:24px;';}")
    partial = page.evaluate(AUDIT, {"mode": "focus"})
    assert not any(row["rule"] == "focus-obscured" for row in partial["findings"])
    page.locator("#overlay").evaluate("e=>{e.style.left='0';e.style.width='300px';e.style.height='80px';}")
    covered = page.evaluate(AUDIT, {"mode": "focus"})
    assert any(row["rule"] == "focus-obscured" for row in covered["findings"])


@pytest.mark.timeout(600)
def test_header_sort_and_resize_controls_stay_visible_when_focused(journey: Journey, readability_catalog: None) -> None:
    """Every header control a keyboard user reaches is fully on screen and receives the point.

    The all-sources table is wider than the viewport, so a focused sort button or resize
    handle past the scroller's visible right edge must be revealed by the scroller: a
    clipped or covered control is unreadable and its center point hits nothing.
    """
    page = journey.page.context.new_page()
    try:
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.emulate_media(color_scheme="light")
        page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/#all=1")
        expect(page.locator("#rows tr").first).to_be_visible(timeout=60000)
        page.evaluate(
            "() => {document.body.tabIndex=-1; document.body.focus(); document.body.removeAttribute('tabindex');}"
        )
        entered = False
        for _ in range(200):
            page.keyboard.press("Tab")
            state = page.evaluate(
                """() => {
                const e = document.activeElement;
                const th = e && e.closest ? e.closest('#headers th') : null;
                if (!th) return null;
                const r = e.getBoundingClientRect();
                const hit = document.elementFromPoint((r.left + r.right) / 2, (r.top + r.bottom) / 2);
                return {column: [...th.parentElement.children].indexOf(th) + 1, tag: e.tagName,
                        inside: r.left >= 0 && r.right <= innerWidth && r.top >= 0 && r.bottom <= innerHeight,
                        hit: !!hit && (e === hit || e.contains(hit) || hit.contains(e))};
            }"""
            )
            if state is None:
                if entered:
                    break
                continue
            entered = True
            assert (
                state["inside"] and state["hit"]
            ), f"header control in column {state['column']} ({state['tag']}) is clipped or obscured when focused"
        assert entered, "the focus walk never reached the header row"
    finally:
        page.close()


def test_origin_tabs_stay_visible_when_focused(journey: Journey, readability_catalog: None) -> None:
    """Regression (R40 #61): with more origins than the strip shows, a focused origin tab
    past its right edge stayed offscreen (the post-publish sweep's focus-obscured finding)."""
    page = journey.page.context.new_page()
    try:
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.emulate_media(color_scheme="light")
        page.goto(f"{journey.origin}/a/fb11c931-5861-4878-8bb5-a964d652b45f/")
        expect(page.locator("#origin-tabs .tab").first).to_be_visible(timeout=60000)
        strip = page.locator("#origin-tabs")
        # The fixture has few origins; narrow the strip so they overflow it like R40's 33 did.
        strip.evaluate("e => { e.style.maxWidth = '200px'; }")
        assert strip.evaluate("e => e.scrollWidth > e.clientWidth"), "the origin strip must overflow"
        page.locator("#origin-tabs .tab").first.focus()
        for index in range(1, page.locator("#origin-tabs .tab").count()):
            page.keyboard.press("Tab")
            state = page.evaluate(
                """() => {
                const e = document.activeElement, strip = document.getElementById('origin-tabs');
                const r = e.getBoundingClientRect(), box = strip.getBoundingClientRect();
                const hit = document.elementFromPoint((r.left + r.right) / 2, (r.top + r.bottom) / 2);
                return {tab: e.classList.contains('tab'), inside: r.left >= box.left && r.right <= box.right,
                        hit: !!hit && (e === hit || e.contains(hit))};
            }"""
            )
            assert state["tab"], f"focus left the origin strip at tab {index}"
            assert state["inside"] and state["hit"], f"origin tab {index} is clipped or obscured when focused"
    finally:
        page.close()
