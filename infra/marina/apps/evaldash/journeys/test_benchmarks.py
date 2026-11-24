# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
from urllib.parse import parse_qs, urlencode, urlsplit

from marina.journeys import Journey
from playwright.sync_api import expect

GSM8K_ZERO_SHOT = "gsm8k [harness; 0-shot; completion]"


def test_benchmark_selection_carries_settings_and_source_intervals_into_comparison(journey: Journey) -> None:
    journey.visit()
    page = journey.page
    page.get_by_role("link", name="Benchmarks", exact=True).click()
    expect(page.get_by_role("heading", name="Benchmarks", exact=True)).to_be_visible()
    benchmark = page.get_by_role("combobox", name="Benchmark", exact=True)
    benchmark.fill("gsm8k 0-shot")
    page.get_by_role("option", name=GSM8K_ZERO_SHOT, exact=True).click()

    scores = page.get_by_role("table", name="Benchmark results")
    expect(scores.get_by_role("columnheader", name="95% interval", exact=True)).to_be_visible()
    expect(scores.get_by_role("row", name=re.compile(r"\bsnowball\b"))).to_contain_text("56.4")
    expect(scores.get_by_role("row", name=re.compile(r"\bnative-qwen3-8b\b"))).to_contain_text("66.7")
    for model in ("snowball", "native-qwen3-8b"):
        scores.get_by_role("checkbox", name=f"Compare {model}", exact=True).check()
    journey.shoot("benchmark-selection")
    page.get_by_role("link", name="Compare 2 models", exact=True).click()

    expect(page.get_by_role("note", name="Cohort warning", exact=True)).to_be_visible()
    index = page.get_by_role("region", name="Benchmark index", exact=True)
    selected_setting = index.get_by_role("link", name=GSM8K_ZERO_SHOT, exact=True)
    expect(selected_setting).to_be_visible()
    detail = page.get_by_role("region", name="Focused benchmark", exact=True)
    expect(detail.get_by_role("heading", name=GSM8K_ZERO_SHOT, exact=True)).to_be_visible()
    plot = detail.get_by_role("list", name="Model scores", exact=True)
    for model in ("snowball", "native-qwen3-8b"):
        expect(plot.get_by_role("listitem").filter(has_text=model)).to_contain_text(re.compile(r"\d+\.\d\u2013\d+\.\d"))
    sources = detail.get_by_role("region", name="Sources and differences", exact=True)
    historical_source = sources.get_by_role("listitem").filter(has_text="snowball")
    native_source = sources.get_by_role("listitem").filter(has_text="native-qwen3-8b")
    expect(historical_source.get_by_text("2026.07.20")).to_be_visible()
    expect(native_source.get_by_text("native-fixture")).to_be_visible()
    historical_run = historical_source.get_by_role("link", name="View run", exact=True)
    native_run = native_source.get_by_role("link", name="View run", exact=True)
    expect(historical_run).to_have_attribute("href", "/evaldash/runs/snowball-2026.07.20-gsm8k-0shot")
    expect(native_run).to_have_attribute("href", "/evaldash/runs/native-fixture-gsm8k-0shot")
    expect(index.get_by_role("link", name="gsm8k [harness; 5-shot; completion]", exact=True)).to_have_count(0)
    query = parse_qs(urlsplit(page.url).query)
    assert query["benchmarks"] == [GSM8K_ZERO_SHOT]
    assert query["cohort"] == ["all"]
    journey.shoot("benchmark-comparison")

    page.reload(wait_until="domcontentloaded")
    expect(native_run).to_be_visible()
    page.set_viewport_size({"width": 390, "height": 844})
    expect(selected_setting).to_be_visible()
    expect(detail).to_be_hidden()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    journey.shoot("benchmark-comparison-mobile")
    selected_setting.click()
    expect(detail.get_by_role("heading", name=GSM8K_ZERO_SHOT, exact=True)).to_be_visible()
    expect(index).to_be_hidden()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    journey.shoot("benchmark-detail-mobile")
    page.get_by_role("link", name="Back to benchmarks", exact=True).click()
    expect(selected_setting).to_be_focused()
    selected_setting.click()
    native_run.click()
    expect(page.get_by_role("heading", name="native-fixture-gsm8k-0shot", exact=True)).to_be_visible()
    journey.sees("graded 3 of 3")


def test_legacy_panel_link_restores_cohort_and_benchmark_variant(journey: Journey) -> None:
    query = urlencode({"cohort": "2026.07.20", "benchmarks": GSM8K_ZERO_SHOT})
    journey.visit(f"/?{query}")
    page = journey.page
    expect(page).to_have_url(re.compile(r"/benchmarks/all\?"))
    cohort = page.get_by_role("combobox", name=re.compile(r"^Cohort\b"))
    expect(cohort).to_have_value("2026.07.20")
    variant = page.locator("select[title^='Which setting of this benchmark']")
    expect(variant).to_have_value(GSM8K_ZERO_SHOT)

    page.reload(wait_until="domcontentloaded")
    expect(variant).to_have_value(GSM8K_ZERO_SHOT)
    expect(cohort).to_have_value("2026.07.20")
