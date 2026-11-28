# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Browser journeys for the Cluster Occupancy app.

The app opens on the live view: every cluster's current holders, read from the
fixture-backed fake of the controllers. Dragging the slider left of "now"
enters history from the Finelog mirror fixture, where ``T0 = 06:00 UTC`` is
``1:00 AM EST`` and rno2a has no mirror at all.
"""

import tomllib
from pathlib import Path

from marina.journeys import Journey
from playwright.sync_api import expect

# The app is unnamed, so Marina serves it at /a/<uuid>/ rather than /<name>/.
APPLET_ID = tomllib.loads((Path(__file__).resolve().parents[1] / "applet.toml").read_text())["applet_id"]

# Text a person reads, and the elements whose contrast a reader depends on.
CONTRAST_SELECTORS = (
    "h1",
    "h2",
    "h3",
    ".freshness",
    ".window-label",
    ".window-title",
    ".picker-label span",
    ".legend-user",
    ".legend-hours",
    ".legend-total",
    ".node-name",
    ".no-data span",
    ".status",
    ".hc-title span:not(.swatch)",
    ".hc-job",
    ".hc-label",
    ".hc-value",
    ".hc-utc",
)

_CONTRAST_SCRIPT = """(selectors) => {
  const parse = (color) => {
    const match = color.match(/rgba?\\(([^)]+)\\)/);
    if (!match) return null;
    const parts = match[1].split(',').map((part) => parseFloat(part));
    return { r: parts[0], g: parts[1], b: parts[2], a: parts.length > 3 ? parts[3] : 1 };
  };
  const linear = (channel) => {
    channel /= 255;
    return channel <= 0.04045 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.055, 2.4);
  };
  const luminance = (rgb) => rgb == null ? null :
    0.2126 * linear(rgb.r) + 0.7152 * linear(rgb.g) + 0.0722 * linear(rgb.b);
  const backgroundOf = (element) => {
    for (let node = element; node; node = node.parentElement) {
      const rgb = parse(getComputedStyle(node).backgroundColor);
      if (rgb && rgb.a > 0.9) return rgb;
    }
    return null;
  };
  const ratio = (fg, bg) => {
    const one = luminance(fg), two = luminance(bg);
    if (one == null || two == null) return null;
    return (Math.max(one, two) + 0.05) / (Math.min(one, two) + 0.05);
  };
  const results = [];
  for (const selector of selectors) {
    for (const element of document.querySelectorAll(selector)) {
      if (!(element.offsetWidth || element.offsetHeight)) continue;
      const style = getComputedStyle(element);
      const value = ratio(parse(style.color), backgroundOf(element));
      if (value != null) {
        results.push({ selector, value: Math.round(value * 100) / 100 });
      }
    }
  }
  return results;
}"""


def visit_app(journey: Journey) -> None:
    journey.page.goto(f"{journey.origin}/a/{APPLET_ID}/", wait_until="domcontentloaded")
    journey.page.evaluate("() => document.fonts.ready")


def assert_readable(journey: Journey) -> None:
    results = journey.page.evaluate(_CONTRAST_SCRIPT, list(CONTRAST_SELECTORS))
    assert results, "the contrast audit found no text to check"
    weak = [row for row in results if row["value"] < 4.5]
    assert not weak, f"contrast below WCAG AA for normal text: {weak}"


def test_load_opens_on_the_live_view(journey: Journey) -> None:
    visit_app(journey)
    journey.sees("Now — as of")
    journey.sees("Live from the controllers")
    for cluster in ("cw-us-east-08a", "cw-us-east-02a", "cw-rno2a"):
        journey.sees(cluster)
    # Live rows still group by NVLink domain.
    journey.sees("DH1-122-US-EAST-08A")
    # Live legend counts GPUs held: alice 8, carol 8, dave 8, bob 4.
    journey.sees("28 GPUs held across the fleet, right now")
    expect(journey.page.locator("#node-r1 .segment:not(.idle)")).to_have_count(1), "rno2a maps live"
    expect(journey.page.locator(".tile")).to_have_count(4)
    journey.shoot("live-now")


def test_hover_shows_live_holder_details(journey: Journey) -> None:
    visit_app(journey)
    journey.page.locator("#node-n1 .segment").first.hover()
    card = journey.page.locator(".hover-card")
    expect(card).to_be_visible()
    expect(card).to_contain_text("alice")
    expect(card).to_contain_text("/alice/train-coordinator")
    expect(card).to_contain_text("production")
    # Her two tasks on the node aggregate into one holder.
    expect(card).to_contain_text("8 on n1")
    expect(card).to_contain_text("GB200")
    expect(card).to_contain_text("EST")
    expect(card).to_contain_text("UTC")
    expect(card).to_contain_text("still holding the node")
    journey.shoot("hover-live-holder")


def test_slider_leaves_now_for_mirror_history(journey: Journey) -> None:
    visit_app(journey)
    slider = journey.page.get_by_label("Window start time")
    slider.press("Home")
    for _ in range(6):
        slider.press("ArrowRight")
    # 03:30 UTC is 10:30 PM Eastern in November: mirror history, not live.
    journey.sees("10:30 PM EST \u2013 11:30 PM EST")
    journey.sees("Data through")
    journey.sees("2:00 AM EST")
    journey.sees("No history — cw-rno2a is not mirrored to the hub.")
    journey.shoot("history-02a")
    journey.sees("No data before Nov 23, 2026, 12:20 AM EST — the mirror starts then.")
    # 02a and rno2a show no tiles for an unmirrored period or cluster.
    expect(journey.page.locator("#node-g1")).to_have_count(0)
    expect(journey.page.locator("#node-r1")).to_have_count(0)
    journey.sees("alice")
    journey.sees("16 GPU-h")
    journey.sees("0.5 GPU-h")
    # The live-only user has no history.
    journey.absent("dave")
    journey.shoot("slider-back-history")

    journey.page.locator("#node-n3 .segment.idle").hover()
    card = journey.page.locator(".hover-card")
    expect(card).to_contain_text("Idle")
    expect(card).to_contain_text("n3 holds no GPUs in this window.")
    journey.shoot("hover-idle")


def test_picker_takes_eastern_wall_time(journey: Journey) -> None:
    visit_app(journey)
    picker = journey.page.get_by_label("Window start, US Eastern")
    picker.fill("2026-11-22T22:30")
    # 22:30 Eastern is 03:30 UTC, not 22:30 UTC.
    journey.sees("10:30 PM EST \u2013 11:30 PM EST")
    journey.sees("No data before Nov 23, 2026, 12:20 AM EST — the mirror starts then.")
    # 08a is mirrored in this window, so the conversion worked.
    journey.sees("alice")


def test_unmirrored_history_never_looks_idle(journey: Journey) -> None:
    visit_app(journey)
    slider = journey.page.get_by_label("Window start time")
    slider.press("Home")
    journey.sees("No history — cw-rno2a is not mirrored to the hub.")
    section = journey.page.locator(".cluster").filter(has=journey.page.get_by_role("heading", name="cw-rno2a"))
    expect(section.locator(".tile")).to_have_count(0)
    expect(section.locator(".no-data")).to_be_visible()
    journey.shoot("unmirrored-history")


def test_readability_in_light_and_dark(journey: Journey) -> None:
    visit_app(journey)
    journey.page.locator("#node-n1 .segment").first.hover()
    expect(journey.page.locator(".hover-card")).to_be_visible()
    assert_readable(journey)
    journey.shoot("light")

    journey.page.emulate_media(color_scheme="dark")
    journey.page.wait_for_timeout(250)
    expect(journey.page.locator(".hover-card")).to_be_visible()
    assert_readable(journey)
    journey.shoot("dark")
    journey.page.emulate_media(color_scheme="light")
