# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Every Atlas page remains readable in both themes and at narrow widths."""

import pytest
from marina.journeys import Journey

from infra.marina.apps.rl_data_catalog.readability import AUDIT, readability_sweep


@pytest.mark.timeout(600)
def test_rendered_contrast_readability_and_reflow(journey: Journey, readability_catalog: None) -> None:
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
