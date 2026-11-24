# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re
from typing import TypedDict, cast

from marina.journeys import Journey
from playwright.sync_api import expect

MODELS_API = "/evaldash/api/catalog/models"
COMPARE_MODELS_API = "/evaldash/api/comparison-models"


class ComparisonModelChoices(TypedDict):
    models: list[str]
    default_cohort: str
    verified_cohorts: list[str]
    versions: list[str]


def test_compare_picker_keeps_configurations_distinct_and_limits_selection(journey: Journey) -> None:
    path = f"{COMPARE_MODELS_API}?cohort=2026.07.20&benchmarks=gsm8k"
    model_catalog = cast(ComparisonModelChoices, journey.api(path))
    checkpoint = "open-athena/preview-snowball-checkpoint-" + "1234567890" * 8
    variants = [f"{checkpoint}@aaaaaaaaaaaa", f"{checkpoint}@bbbbbbbbbbbb"]
    model_catalog["models"].extend(variants)
    page = journey.page
    page.route(f"**{COMPARE_MODELS_API}**", lambda route: route.fulfill(json=model_catalog))
    journey.visit("/compare?cohort=2026.07.20&benchmarks=gsm8k&models=snowball,snowball,qwen3-8b")

    expect(page.get_by_role("button", name="Remove snowball", exact=True)).to_have_count(1)
    picker = page.get_by_role("combobox", name="Add model", exact=True)
    picker.click()
    for excluded in ("snowball", "qwen3-8b"):
        expect(page.get_by_role("option", name=excluded, exact=True)).to_have_count(0)

    picker.fill("AAAAAAAAAAAA")
    picker.press("ArrowDown")
    picker.press("Enter")
    expect(page.get_by_role("button", name=f"Remove {variants[0]}", exact=True)).to_be_visible()
    picker.fill("bbbbbbbbbbbb")
    page.get_by_role("option", name=variants[1], exact=True).click()
    picker.click()
    page.get_by_role("option", name="tootsie-8b", exact=True).click()
    expect(picker).to_have_count(0)

    page.reload(wait_until="domcontentloaded")
    editor = page.get_by_role("button", name="Edit models", exact=True)
    expect(editor).to_have_attribute("aria-expanded", "false")
    editor.click()
    for name in ("snowball", "tootsie-8b", *variants):
        expect(page.get_by_role("button", name=f"Remove {name}", exact=True)).to_be_visible()
    page.get_by_role("button", name=f"Remove {variants[0]}", exact=True).click()
    picker.click()
    expect(page.get_by_role("option", name=variants[0], exact=True)).to_be_visible()
    picker.press("Escape")
    page.get_by_role("button", name="Clear models", exact=True).click()
    expect(page).not_to_have_url(re.compile(r"[?&]models="))
    expect(page).to_have_url(re.compile(r"[?&]benchmarks=gsm8k"))
    expect(page.get_by_role("region", name="Benchmark index", exact=True)).to_be_hidden()


def test_model_index_recovers_from_a_catalog_failure(journey: Journey) -> None:
    page = journey.page
    page.route(f"**{MODELS_API}?**", lambda route: route.fulfill(status=503), times=1)
    journey.visit("/models?cohort=2026.07.20")
    expect(page.get_by_role("alert")).to_be_visible()
    page.get_by_role("button", name="Retry", exact=True).click()
    expect(page.get_by_role("link", name="snowball", exact=True)).to_be_visible()
    expect(page.get_by_role("alert")).to_have_count(0)

    expected = (
        f"503 {MODELS_API}?cohort=2026.07.20",
        "console Failed to load resource: the server responded with a status of 503",
    )
    journey.refusals[:] = [failure for failure in journey.refusals if not failure.startswith(expected)]
