# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Someone opens the run log, finds a launch, and reads what one of its evals scored."""

import re
from typing import Any, cast
from urllib.parse import unquote

import pytest
from marina.journeys import Journey
from playwright.sync_api import expect

API = "/evaldash/api"


def test_the_run_log_lists_launches_and_one_run_opens(journey: Journey) -> None:
    journey.page.route(f"**{API}/runs/*/jobs", lambda route: route.fulfill(json={"roles": []}))
    journey.visit("/runs").shoot("runs")
    # Read the run log itself, including its newest launch.
    assert "snowball" in journey.reads()

    journey.click("All runs")
    journey.sees("detail →")
    newest: str = cast(list[dict[str, Any]], journey.api(f"{API}/runs?limit=1"))[0]["run_id"]

    journey.click("detail →")
    journey.sees(newest)
    expect(journey.page.get_by_text("Grade", exact=True)).to_be_visible()
    journey.sees("Metrics").shoot("run-detail")
    assert journey.page.url == journey.url(f"/runs/{newest}")


def test_the_shell_bar_carries_the_app_and_its_own_navigation(journey: Journey) -> None:
    journey.visit("/runs")
    assert journey.api("/api/marina/me") == {"user": "anonymous", "role": "admin"}
    offers = journey.offers()
    expect(journey.page.get_by_role("banner").get_by_text("EvalDash", exact=True)).to_be_visible()
    expect(journey.page.get_by_role("button", name="Open apps")).to_be_visible()
    assert {"a:Panel", "a:Runs", "a:Debug"} <= set(offers)


def test_the_panel_serves_the_committed_catalog(journey: Journey) -> None:
    journey.visit("/?cohort=2026.07.20").shoot("panel")
    assert "snowball" in journey.reads()
    store = cast(dict[str, Any], journey.api(f"{API}/status"))["store"]
    assert store["backend"] == "postgres"
    assert store["record_count"] == 23


def test_a_benchmark_family_variant_survives_a_shared_panel_url(journey: Journey) -> None:
    journey.visit("/?cohort=2026.07.20")
    picker = journey.page.locator("select[title^='Which setting of this benchmark']").first

    picker.select_option("gsm8k-0shot")
    journey.page.wait_for_url(re.compile(r"[?&]benchmarks="))

    assert "gsm8k-0shot" in unquote(journey.page.url)
    journey.page.reload(wait_until="domcontentloaded")
    expect(picker).to_have_value("gsm8k-0shot")


def test_panel_cohort_survives_reload_and_navigation_to_compare(journey: Journey) -> None:
    journey.visit("/?cohort=2026.07.21")
    cohort = journey.page.locator("label").filter(has_text="Cohort").locator("select")
    expect(cohort).to_have_value("2026.07.21")
    expect(journey.page.locator("tr").filter(has_text="qwen3-8b")).to_have_count(2)
    expect(journey.page.locator("tr").filter(has_text="snowball")).to_have_count(0)

    cohort.select_option("2026.07.20")
    journey.page.wait_for_url(re.compile(r"[?&]cohort=2026.07.20"))
    journey.page.reload(wait_until="domcontentloaded")
    expect(cohort).to_have_value("2026.07.20")
    expect(journey.page.locator("tr").filter(has_text="snowball")).to_have_count(2)
    expect(journey.page.locator("tr").filter(has_text="qwen3-8b")).to_have_count(0)

    journey.page.get_by_role("link", name="Compare", exact=True).click()
    journey.page.wait_for_url(re.compile(r"/compare\?cohort=2026.07.20"))
    journey.page.get_by_role("combobox", name="Add model", exact=True).click()
    expect(journey.page.get_by_role("option", name="snowball", exact=True)).to_be_visible()
    expect(journey.page.get_by_role("option", name="qwen3-8b", exact=True)).to_have_count(0)


def test_compare_picker_excludes_zero_scores_and_other_cohorts(journey: Journey) -> None:
    panel = cast(dict[str, Any], journey.api(f"{API}/panel?cohort=2026.07.20"))
    snowball = next(row for row in panel["rows"] if row["model"] == "snowball")
    panel["rows"].append(
        {
            **snowball,
            "model": "zero-only",
            "cells": {name: {**cell, "value": 0.0} for name, cell in snowball["cells"].items()},
        }
    )
    journey.page.route(f"**{API}/panel?**", lambda route: route.fulfill(json=panel))

    journey.visit("/compare?cohort=2026.07.20&models=snowball,zero-only,qwen3-8b")
    expect(journey.page.get_by_role("button", name="Remove snowball", exact=True)).to_be_visible()
    journey.page.get_by_role("combobox", name="Add model", exact=True).click()
    expect(journey.page.get_by_role("option", name="zero-only", exact=True)).to_have_count(0)
    expect(journey.page.get_by_role("option", name="qwen3-8b", exact=True)).to_have_count(0)
    expect(journey.page.get_by_role("button", name="Remove zero-only", exact=True)).to_have_count(0)
    expect(journey.page.get_by_role("button", name="Remove qwen3-8b", exact=True)).to_have_count(0)
    journey.sees("Pick at least two models to compare.")


@pytest.mark.parametrize("view", ["launches", "runs"])
def test_run_model_search_finds_older_runs_and_survives_reload(journey: Journey, view: str) -> None:
    journey.visit(f"/runs?view={view}&limit=1")
    table = journey.page.get_by_role("table").first
    expect(table).to_contain_text("qwen3-8b")
    model = journey.page.get_by_role("combobox", name="Model", exact=True)
    model.click()
    # The dropdown searches the catalog, even though only the newest result is on screen.
    options = journey.page.get_by_role("listbox", name="Model", exact=True)
    expect(options.get_by_role("option", name="snowball", exact=True)).to_be_visible()
    model.fill("SNOW")
    expect(options.get_by_role("option")).to_have_count(1)
    model.press("ArrowDown")
    model.press("Enter")
    expect(table).to_contain_text("snowball")
    journey.page.wait_for_url(re.compile(r"[?&]model=snowball"))
    journey.page.reload(wait_until="domcontentloaded")
    expect(model).to_have_value("snowball")
    expect(table).to_contain_text("snowball")
    active_view = "By launch" if view == "launches" else "All runs"
    expect(journey.page.get_by_role("button", name=active_view, exact=True)).to_have_attribute("aria-pressed", "true")
    expect(journey.page.get_by_label("Result limit", exact=True)).to_have_value("1")


def test_run_model_dropdown_click_escape_and_clear(journey: Journey) -> None:
    journey.visit("/runs?limit=1")
    model = journey.page.get_by_role("combobox", name="Model", exact=True)
    journey.page.get_by_role("button", name="Browse model options", exact=True).click()
    options = journey.page.get_by_role("listbox", name="Model", exact=True)
    options.get_by_role("option", name="snowball", exact=True).click()
    expect(journey.page.get_by_role("table").first).to_contain_text("snowball")
    model.click()
    model.fill("no-such-model")
    expect(options.get_by_role("option")).to_have_count(0)
    model.press("Enter")
    expect(journey.page.get_by_role("table").first).to_contain_text("snowball")
    model.press("Escape")
    expect(model).to_have_value("snowball")
    expect(options).to_have_count(0)
    journey.page.get_by_role("button", name="Clear model filter", exact=True).click()
    expect(journey.page.get_by_role("table").first).to_contain_text("qwen3-8b")
    assert "model=" not in journey.page.url


def test_compare_search_keeps_variants_distinct_and_caps_selection(journey: Journey) -> None:
    panel = cast(dict[str, Any], journey.api(f"{API}/panel?cohort=2026.07.20"))
    reference = next(row for row in panel["rows"] if row["model"] == "snowball")
    long_name = "open-athena/preview-snowball-checkpoint-" + "1234567890" * 8
    variants = [f"{long_name}@aaaaaaaaaaaa", f"{long_name}@bbbbbbbbbbbb"]
    models = ["snowball", "tootsie-8b", *variants]
    panel["rows"] = [{**reference, "model": model} for model in models]
    comparison = cast(dict[str, Any], journey.api(f"{API}/compare?cohort=2026.07.20&models=snowball,tootsie-8b"))
    comparison["aggregates"] = {model: comparison["aggregates"]["snowball"] for model in models}
    for row in comparison["rows"]:
        if "snowball" in row["cells"]:
            row["cells"] = {model: row["cells"]["snowball"] for model in models}
        row["differences"] = {}
    journey.page.route(f"**{API}/compare?**", lambda route: route.fulfill(json=comparison))
    journey.page.route(f"**{API}/panel?**", lambda route: route.fulfill(json=panel))
    journey.visit("/compare?cohort=2026.07.20&benchmarks=gsm8k&models=snowball,snowball")
    expect(journey.page.get_by_role("button", name="Remove snowball", exact=True)).to_have_count(1)
    picker = journey.page.get_by_role("combobox", name="Add model", exact=True)
    picker.click()
    expect(journey.page.get_by_role("option", name="snowball", exact=True)).to_have_count(0)
    picker.fill("AAAAAAAAAAAA")
    expect(journey.page.get_by_role("option")).to_have_count(1)
    picker.press("ArrowDown")
    picker.press("Enter")
    expect(journey.page.get_by_role("button", name=f"Remove {variants[0]}", exact=True)).to_be_visible()
    picker.click()
    picker.fill("bbbbbbbbbbbb")
    journey.page.get_by_role("option", name=variants[1], exact=True).click()
    picker.click()
    journey.page.get_by_role("option", name="tootsie-8b", exact=True).click()
    expect(picker).to_have_count(0)
    journey.page.reload(wait_until="domcontentloaded")
    for model in models:
        expect(journey.page.get_by_role("button", name=f"Remove {model}", exact=True)).to_be_visible()
    assert "cohort=2026.07.20" in journey.page.url
    assert "benchmarks=gsm8k" in journey.page.url
    journey.page.set_viewport_size({"width": 390, "height": 844})
    assert journey.page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    expect(journey.page.get_by_role("list", name="Chart models")).to_contain_text("@aaaaaaaaaaaa")
    journey.page.get_by_role("button", name=f"Remove {variants[0]}", exact=True).click()
    picker.click()
    expect(journey.page.get_by_role("option", name=variants[0], exact=True)).to_be_visible()
    picker.press("Escape")
    journey.page.get_by_role("button", name="Clear models", exact=True).click()
    journey.page.wait_for_url(re.compile(r"/compare\?(?!.*models=)"))
    assert "benchmarks=gsm8k" in journey.page.url
    expect(journey.page.get_by_role("table")).to_have_count(0)


def test_model_index_reports_failed_load_and_recovers(journey: Journey) -> None:
    panel = cast(dict[str, Any], journey.api(f"{API}/panel?cohort=2026.07.20"))
    name = "open-athena/preview-snowball-checkpoint-" + "1234567890" * 8 + "@aaaaaaaaaaaa"
    panel["rows"] = [{**panel["rows"][0], "model": name}]
    journey.page.route(f"**{API}/panel?**", lambda route: route.fulfill(status=503))
    journey.visit("/models?cohort=2026.07.20")
    expect(journey.page.get_by_role("alert")).to_be_visible()
    # This request deliberately failed. Keep every unexpected page or API error visible to finish().
    expected = (
        f"503 {API}/panel?cohort=2026.07.20",
        "console Failed to load resource: the server responded with a status of 503",
    )
    journey.refusals[:] = [failure for failure in journey.refusals if not failure.startswith(expected)]
    journey.page.unroute(f"**{API}/panel?**")
    journey.page.route(f"**{API}/panel?**", lambda route: route.fulfill(json=panel))
    journey.page.get_by_role("button", name="Retry", exact=True).click()
    card = journey.page.get_by_role("button", name=name, exact=True)
    expect(card).to_be_visible()
    expect(journey.page.get_by_role("alert")).to_have_count(0)
    journey.page.set_viewport_size({"width": 390, "height": 844})
    assert journey.page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    expect(card).to_contain_text("@aaaaaaaaaaaa")
    # The rendered name wraps rather than clipping the checkpoint or config identity.
    assert card.locator("[title]").evaluate("el => el.scrollWidth <= el.clientWidth")


def test_models_from_different_cohorts_compare_behind_the_mixed_settings_warning(journey: Journey) -> None:
    journey.visit("/?cohort=all")
    journey.sees("All cohorts may mix evaluation settings")
    for model in ("snowball", "qwen3-8b"):
        journey.page.locator("tr").filter(has_text=model).first.locator("input[type=checkbox]").check()

    journey.page.get_by_role("button", name=re.compile(r"Compare \d+ models")).click()
    journey.page.wait_for_url(re.compile(r"/compare\?.*cohort=all"))
    journey.sees("All cohorts may mix evaluation settings")
    expect(journey.page.get_by_role("button", name="Remove qwen3-8b", exact=True)).to_be_visible()


def test_unversioned_runs_have_their_own_cohort_entry(journey: Journey) -> None:
    journey.visit("/?cohort=unversioned")
    cohort = journey.page.locator("label").filter(has_text="Cohort").locator("select")
    expect(cohort).to_have_value("unversioned")
    journey.sees("Runs launched without a version label")
    expect(journey.page.locator("tr").filter(has_text="llama3-8b")).to_have_count(2)
    expect(journey.page.locator("tr").filter(has_text="snowball")).to_have_count(0)


def test_links_without_a_cohort_open_on_the_newest_scores_across_cohorts(journey: Journey) -> None:
    journey.visit("/compare?models=snowball,qwen3-8b")
    journey.page.wait_for_url(re.compile(r"[?&]cohort=all"))
    journey.sees("All cohorts may mix evaluation settings")
    expect(journey.page.get_by_role("button", name="Remove qwen3-8b", exact=True)).to_be_visible()

    journey.visit("/models/snowball")
    journey.page.wait_for_url(re.compile(r"[?&]cohort=all"))
    journey.sees("cohort all runs")


def test_a_cross_cohort_comparison_names_each_cell_cohort_and_links_its_run(journey: Journey) -> None:
    comparison = cast(dict[str, Any], journey.api(f"{API}/compare?cohort=all&models=snowball,qwen3-8b"))
    row = next(row for row in comparison["rows"] if len(row["cells"]) == 2)
    cell = row["cells"]["qwen3-8b"]

    journey.visit("/compare?cohort=all&models=snowball,qwen3-8b")
    link = journey.page.get_by_role("link", name=cell["version"], exact=True).first
    expect(link).to_be_visible()
    link.click()
    journey.page.wait_for_url(re.compile(r"/runs/"))


def test_historical_and_native_scores_compare_across_cohorts_and_browse_terminal_samples(journey: Journey) -> None:
    journey.visit("/?cohort=all&benchmarks=mmlu,humaneval")
    for model in ("snowball", "native-qwen3-8b"):
        row = journey.page.get_by_text(model, exact=True).first.locator("xpath=ancestor::tr")
        expect(row.get_by_role("checkbox")).to_be_enabled()
        row.get_by_role("checkbox").check()
    journey.page.get_by_role("button", name="Compare 2 models →", exact=True).click()
    journey.page.wait_for_url(re.compile(r"/compare\?.*cohort=all"))
    expect(journey.page.get_by_role("button", name="Remove native-qwen3-8b", exact=True)).to_be_visible()
    journey.sees("All cohorts may mix evaluation settings.")
    comparison = cast(
        dict[str, Any],
        journey.api(f"{API}/compare?cohort=all&benchmarks=mmlu,humaneval&models=snowball,native-qwen3-8b"),
    )
    assert set(comparison["shared"]) == {"mmlu", "humaneval"}
    for row in comparison["rows"]:
        assert {cell["metric"] for cell in row["cells"].values()} == {"accuracy"}
        assert row["cells"]["native-qwen3-8b"]["value"] == pytest.approx(2 / 3)
        assert row["cells"]["native-qwen3-8b"]["n_scored"] == 3
    journey.shoot("historical-native-compare")
    journey.page.reload(wait_until="domcontentloaded")
    expect(journey.page.get_by_role("button", name="Remove snowball", exact=True)).to_be_visible()
    journey.page.set_viewport_size({"width": 390, "height": 844})
    assert journey.page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    journey.shoot("historical-native-compare-mobile")
    journey.page.set_viewport_size({"width": 1280, "height": 900})
    journey.visit("/runs/native-fixture-mmlu")
    all_samples = journey.page.get_by_role("button", name=re.compile(r"^all.*3", re.IGNORECASE))
    expect(all_samples).to_be_visible()
    journey.page.get_by_role("button", name=re.compile(r"^incorrect.*1", re.IGNORECASE)).click()
    journey.sees("1\u20131 of 1")
    page = cast(dict[str, Any], journey.api(f"{API}/runs/native-fixture-mmlu/samples?task=mmlu&correct=all"))
    assert page["total"] == 3
    assert page["counts"] == {"all": 3, "correct": 2, "incorrect": 1, "ungraded": 0}
    assert [row["grading"]["score"] for row in page["rows"]].count(0) == 1
    all_samples.click()
    journey.page.get_by_role("button", name="Open viewer ⛶", exact=True).click()
    journey.sees("Which option is correct").shoot("native-sample-viewer")
