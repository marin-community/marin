# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
from urllib.parse import urlsplit

import pytest
from marina.journeys import Journey
from playwright.sync_api import expect


@pytest.mark.timeout(60)
def test_large_catalog_keeps_the_benchmark_grid_and_comparison_navigable(journey: Journey, scale_cohort: str) -> None:
    journey.visit(f"/benchmarks/all?cohort={scale_cohort}")
    page = journey.page
    grid = page.get_by_role("table", name="Per-benchmark results", exact=True)
    journey.sees("Models 1\u201320 of 100").sees("Benchmarks 1\u20136 of 50")
    expect(grid.get_by_role("row")).to_have_count(21)
    expect(grid.get_by_role("columnheader", name=re.compile(r"^scale-benchmark-"))).to_have_count(6)
    first_model = grid.get_by_role("button", name=re.compile(r"^scale-model-\d{3}$")).first.inner_text()
    page.get_by_role("button", name="Next models", exact=True).click()
    expect(grid.get_by_role("button", name=first_model, exact=True)).to_have_count(0)
    expect(grid.get_by_role("row")).to_have_count(21)
    page.get_by_role("button", name="Next benchmarks", exact=True).click()
    expect(grid.get_by_role("columnheader", name=re.compile(r"^scale-benchmark-06\b"))).to_be_visible()
    expect(grid.get_by_role("columnheader", name=re.compile(r"^scale-benchmark-00\b"))).to_have_count(0)
    journey.shoot("large-catalog-panel")
    page.get_by_role("combobox", name="Find benchmark", exact=True).fill("scale-benchmark-49")
    page.get_by_role("option", name="scale-benchmark-49", exact=True).click()
    expect(grid.get_by_role("columnheader", name=re.compile(r"^scale-benchmark-49\b"))).to_be_visible()

    comparison_api_requests: list[str] = []
    page.on("request", lambda request: comparison_api_requests.append(urlsplit(request.url).path))
    page.get_by_role("link", name="Compare", exact=True).click()
    expect(page).to_have_url(re.compile(rf"/compare\?cohort={scale_cohort}"))
    picker = page.get_by_role("combobox", name="Add model", exact=True)
    picker.click()
    options = page.get_by_role("listbox", name="Add model", exact=True)
    expect(options.get_by_role("option")).to_have_count(12)
    expect(options.get_by_role("option", name="scale-model-099", exact=True)).to_have_count(0)
    picker.fill("SCALE-MODEL-099")
    picker.press("Enter")
    expect(page.get_by_role("button", name="Remove scale-model-099", exact=True)).to_be_visible()
    for name in ("scale-model-000", "scale-model-050", "scale-model-075"):
        picker.fill(name)
        options.get_by_role("option", name=name, exact=True).click()

    page.get_by_role("button", name="Edit models", exact=True).click()
    index = page.get_by_role("region", name="Benchmark index", exact=True)
    shared = index.get_by_role("button", name="Shared only", exact=True)
    expect(shared).to_have_attribute("aria-pressed", "true")
    expect(index.get_by_role("link", name=re.compile(r"^scale-benchmark-\d{2}$"))).to_have_count(12)
    expect(index.get_by_role("status")).to_contain_text("1\u201312 of 49")
    focus = page.get_by_role("region", name="Focused benchmark", exact=True)
    expect(focus.get_by_role("heading", name="scale-benchmark-00", exact=True)).to_be_visible()
    expect(focus.get_by_text("0.0%", exact=True)).to_be_visible()
    journey.shoot("large-catalog-comparison")
    index.get_by_role("button", name="Next", exact=True).click()
    expect(index.get_by_role("link", name="scale-benchmark-12", exact=True)).to_be_visible()
    expect(index.get_by_role("link", name="scale-benchmark-00", exact=True)).to_have_count(0)

    search = index.get_by_role("searchbox", name="Search benchmarks", exact=True)
    search.fill("scale-benchmark-49")
    sparse = index.get_by_role("link", name="scale-benchmark-49", exact=True)
    expect(index.get_by_role("status")).to_have_text("0 settings")
    expect(sparse).to_have_count(0)
    all_recorded = index.get_by_role("button", name="All recorded", exact=True)
    all_recorded.click()
    expect(all_recorded).to_have_attribute("aria-pressed", "true")
    sparse.click()
    expect(page).to_have_url(re.compile(r"[?&]focus=scale-benchmark-49"))
    benchmark = focus.get_by_role("heading", name="scale-benchmark-49", exact=True)
    expect(benchmark).to_be_visible()
    expect(focus).to_contain_text("3 of 4 models scored")
    expect(focus.get_by_role("listitem").filter(has_text="scale-model-075")).to_contain_text("No recorded score")

    full_scale = focus.get_by_role("button", name="Full scale", exact=True)
    full_scale.click()
    expect(full_scale).to_have_attribute("aria-pressed", "true")
    full_scale_endpoint = focus.get_by_text("100", exact=True)
    expect(full_scale_endpoint).to_be_visible()
    fit = focus.get_by_role("button", name="Fit to scores", exact=True)
    fit.click()
    expect(fit).to_have_attribute("aria-pressed", "true")
    expect(full_scale_endpoint).to_have_count(0)
    journey.shoot("large-catalog-benchmark-focus")

    sources = focus.get_by_role("region", name="Sources and differences", exact=True)
    source = sources.get_by_role("listitem").filter(has_text="scale-model-099")
    source.get_by_role("link", name=scale_cohort, exact=True).click()
    expect(page.get_by_role("heading", name="scale-model-099-scale-benchmark-49", exact=True)).to_be_visible()
    journey.sees(re.compile(r"\b59\.9\s*%"))
    page.go_back(wait_until="domcontentloaded")
    expect(benchmark).to_be_visible()
    expect(search).to_have_value("scale-benchmark-49")
    expect(all_recorded).to_have_attribute("aria-pressed", "true")
    assert "/evaldash/api/comparison-models" in comparison_api_requests
    assert "/evaldash/api/panel" not in comparison_api_requests
