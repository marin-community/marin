# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A reader sees every source's status, reasons and evidence on the status page."""

import json
import re
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlencode

import pytest
from marina.journeys import Journey
from playwright.sync_api import expect

from infra.marina.apps.rl_data_catalog.readability import inspect_status_focus

APP_ROOT = "a/fb11c931-5861-4878-8bb5-a964d652b45f"


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [390, 320])
def test_status_keyboard_keeps_deep_reason_and_evidence_targets_visible(
    journey: Journey, seeded_extras: dict, theme: str, width: int
) -> None:
    # The real failing rows had two reasons, with the long grading label first.
    # Include multiple such rows and evidence links to exercise cross-row Tab.
    for index in range(3):
        row = deepcopy(seeded_extras["metadata_exclusions"][0])
        row.update(id=f"Fixture:focus-{index}", display_name=f"Fixture focus {index}")
        evidence = row["reasons"][0]["evidence"]
        row["reasons"] = [
            {"reason": reason, "evidence": evidence}
            for reason in ["grading-census-incomplete", "lexical-receipt-missing"]
        ]
        seeded_extras["metadata_exclusions"].append(row)
    page = journey.page
    page.emulate_media(color_scheme=theme, reduced_motion="reduce")
    page.set_viewport_size({"width": width, "height": 1000})
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    page.goto(page.url.rstrip("/") + "/status.html")
    expect(page.locator('tr[data-source-id="Fixture:focus-2"]')).to_have_count(1)
    for opened in [False, True]:
        page.locator("#rows details").evaluate_all("(es, open) => es.forEach(e => e.open = open)", opened)
        result = inspect_status_focus(page)
        assert not result["findings"], result["findings"]
    page.screenshot(path=str(journey.shots / f"status-keyboard-{theme}-{width}.png"))


def test_all_pages_share_navigation_labels_order_and_current_view(journey: Journey, readability_catalog: None) -> None:
    page = journey.page
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    base = page.url
    source = "Marin:clean"
    evidence = page.request.get(base + "api/assessment?" + urlencode({"source_id": source})).json()
    task = evidence["report"]["evidence"]["items"][0]["task_id"]
    models = page.request.get(base + "api/source?" + urlencode({"source_id": source})).json()["difficulty"]["models"]
    measurement = next(model["measurement_id"] for model in models if model.get("measurement_id"))
    attempts = page.request.get(
        base + "api/difficulty-attempts?" + urlencode({"source_id": source, "measurement_id": measurement})
    ).json()
    attempt = attempts["items"][0]["attempt_id"]
    labels = ["Status", "By source", "By curriculum", "Difficulty", "Getting started"]
    views = [
        ("", "By source"),
        ("status.html", "Status"),
        ("curriculum.html", "By curriculum"),
        ("difficulty.html", "Difficulty"),
        ("guide.html", "Getting started"),
        ("source.html#source_id=Marin%3Aclean", "By source"),
        ("backends.html#source_id=Marin%3Aclean", "By source"),
        ("assessment.html?source=Marin%3Aclean", "By source"),
        ("review.html?" + urlencode({"source": source, "task": task}), "By source"),
        ("defects.html", "By source"),
        ("attempts.html?source=Marin%3Aclean", "Difficulty"),
        ("rollout.html?" + urlencode({"source": source, "measurement": measurement, "attempt": attempt}), "Difficulty"),
    ]
    for path, current in views:
        page.goto(base + path)
        navigation = page.get_by_role("navigation", name="Atlas views", exact=True)
        expect(navigation.get_by_role("link")).to_have_text(labels)
        expect(navigation.locator('[aria-current="page"]')).to_have_text(current)
        assert navigation.get_by_role("link").evaluate_all("links => links.map(link => link.getAttribute('href'))") == [
            "status.html",
            "./",
            "curriculum.html",
            "difficulty.html",
            "guide.html",
        ]


def extras() -> dict:
    return json.loads((Path(__file__).parents[1] / "dist/measurements/source-status-extras.json").read_text())


@pytest.fixture
def seeded_extras(journey: Journey):
    """Serve the real extras plus one fixture pending row and a needs-ruling row."""
    payload = extras()
    payload["pending"] = [
        {
            "id": "Fixture:queued-source",
            "display_name": "Fixture · queued-source",
            "status": "Pending",
            "stage": "Intake queued",
            "origin": "Marin",
            "upstream_dataset": "fixture/queued",
            "upstream_url": "https://example.org/queued",
            "notes": "Awaiting licensed intake conversion.",
            "use": "train",
            "method_tags": ["RL"],
            "reasons": [],
        }
    ]
    payload["metadata_exclusions"] = [
        {
            "id": "Fixture:metadata-excluded",
            "display_name": "Fixture · metadata-excluded",
            "status": "Excluded",
            "stage": None,
            "origin": "Marin",
            "upstream_dataset": "fixture/excluded",
            "upstream_url": "https://example.org/excluded",
            "dataset_id": "fixture/excluded",
            "dataset_revision": "a" * 40,
            "notes": "Declaration metadata only.",
            "reasons": [
                {
                    "reason": "license-ban",
                    "evidence": {
                        "kind": "issue-link",
                        "message": "Upstream task-data permission has not been established.",
                        "uri": "https://github.com/marin-community/marin/issues/11497",
                        "sha256": None,
                        "expected": None,
                    },
                }
            ],
        },
        {
            "id": "Fixture:needs-ruling",
            "display_name": "Fixture · needs-ruling",
            "status": "Excluded",
            "stage": None,
            "origin": "Marin",
            "upstream_dataset": "fixture/ruling",
            "upstream_url": "https://example.org/ruling",
            "dataset_id": "fixture/ruling",
            "dataset_revision": "a" * 40,
            "notes": "Legacy tag.",
            "reasons": [
                {
                    "reason": "unspecified-legacy",
                    "evidence": {
                        "kind": "note",
                        "message": "No explicit exclusion ruling is recorded for the tag.",
                        "uri": None,
                        "sha256": None,
                        "expected": None,
                    },
                }
            ],
        },
        {
            "id": "Fixture:superseded",
            "display_name": "Fixture · superseded",
            "status": "Deprecated",
            "stage": None,
            "origin": "Marin",
            "upstream_dataset": "fixture/superseded",
            "upstream_url": "https://example.org/superseded",
            "dataset_id": "fixture/superseded",
            "dataset_revision": "a" * 40,
            "notes": "Superseded by its repaired release.",
            "replacement_source_id": "Fixture:metadata-excluded",
            "reasons": [],
        },
    ]
    page = journey.page
    page.route("**/measurements/source-status-extras.json", lambda route: route.fulfill(json=payload))
    return payload


def test_default_view_hides_excluded_sources_and_counts_what_is_hidden(journey: Journey) -> None:
    page = journey.page
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    expect(page.locator("#rows tr").first).to_be_visible()
    sources = page.evaluate("async () => (await (await fetch('api/sources')).json()).sources")
    sources = [dict(row, status="Pending" if row["id"] == "Marin:stale-clean" else row["status"]) for row in sources]
    page.route("**/api/sources*", lambda route: route.fulfill(json={"sources": sources, "synced_at": "2026-11-14"}))
    page.reload()
    expect(page.locator("#rows tr").first).to_be_visible()
    excluded_row = page.locator('tr[data-source-id="Marin:filtered-zero"]')
    pending_row = page.locator('tr[data-source-id="Marin:stale-clean"]')
    expect(excluded_row).to_have_count(0)
    expect(pending_row).to_have_count(0)
    footer = page.locator("#visible-count")
    expect(footer).to_contain_text("excluded sources hidden")
    expect(footer).to_contain_text("pending sources hidden")
    expect(page.locator("#task-reconciliation")).to_contain_text("hidden (excluded status)")
    expect(page.locator("#task-reconciliation")).to_contain_text("hidden (pending status)")

    page.locator("#status").select_option("Excluded")
    expect(excluded_row).to_have_count(1)
    expect(pending_row).to_have_count(0)
    page.locator("#status").select_option("Pending")
    expect(pending_row).to_have_count(1)
    expect(excluded_row).to_have_count(0)
    expect(footer).not_to_contain_text("pending sources hidden")
    page.screenshot(path=str(journey.shots / "default-view-excluded-filter.png"))


def test_status_table_keeps_human_columns_and_rows(journey: Journey, seeded_extras: dict) -> None:
    """Real rows keep readable columns: sources stay nameable and rows stay scannable.

    The reasons cell is a plain container inside its td; a nested td lets the auto
    table layout explode (1724px reasons, 50px source, 861px first row).
    """
    page = journey.page
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    expect(page.locator("#rows tr").first).to_be_visible()
    base = page.url.rstrip("/") + "/"
    page.goto(base + "status.html")
    expect(page.locator("#rows tr").first).to_be_visible()

    assert page.locator("#rows td td").count() == 0
    widths = page.evaluate(
        "() => [...document.querySelectorAll('#status-table th')].map("
        "th => Math.round(th.getBoundingClientRect().width))"
    )
    table = page.evaluate("() => Math.round(document.getElementById('status-table').getBoundingClientRect().width)")
    first_row = page.evaluate("() => Math.round(document.querySelector('#rows tr').getBoundingClientRect().height)")
    assert widths[0] >= 200, f"Source column too narrow: {widths}"
    assert widths[7] >= 80, f"Last-change column too narrow: {widths}"
    assert table <= 1500, f"status table is {table}px wide at a 1440px viewport"
    assert first_row <= 320, f"first status row is {first_row}px tall"


def test_status_page_lists_every_status_with_clickable_reasons(journey: Journey, seeded_extras: dict) -> None:
    page = journey.page
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    expect(page.locator("#rows tr").first).to_be_visible()
    base = page.url.rstrip("/") + "/"
    page.goto(base + "status.html")
    expect(page.locator("#rows tr").first).to_be_visible()
    expect(page.locator("#message")).to_have_text("")
    layout = page.evaluate(
        """() => {
        const row = document.querySelector('#rows tr');
        const widths = [...row.cells].map(cell => cell.getBoundingClientRect().width);
        return {height: row.getBoundingClientRect().height, widths};
    }"""
    )
    assert layout["widths"][0] >= 260, f"Source column starved: {layout}"
    assert layout["widths"][5] <= 560, f"Reasons column unclamped: {layout}"
    assert layout["widths"][7] >= 60, f"Last-change column starved: {layout}"
    assert layout["height"] <= 200, f"First row too tall: {layout}"

    counts = page.locator("#status-counts")
    expect(counts.locator("button", has_text="Pending")).to_contain_text("1")
    expect(counts.locator("button", has_text="Excluded")).to_contain_text(re.compile(r"Excluded [1-9]"))

    queued = page.locator('tr[data-source-id="Fixture:queued-source"]')
    expect(queued.locator("td").nth(1)).to_contain_text("Pending")
    expect(queued.locator(".status-chips summary")).to_have_text("Intake queued")

    excluded = page.locator('tr[data-source-id="Fixture:metadata-excluded"]')
    reason = excluded.locator(".status-reasons summary")
    expect(reason).to_have_text("License ban")
    explanation = excluded.locator(".status-explanation")
    expect(explanation).not_to_be_visible()
    reason.click()
    expect(explanation).to_be_visible()
    expect(explanation.locator(".meaning")).to_contain_text("banned for training use")
    expect(explanation).to_contain_text("Cleared by:")
    link = excluded.locator(".status-evidence a")
    expect(link).to_have_attribute("href", "https://github.com/marin-community/marin/issues/11497")

    page.locator("#status-filter").select_option("Available")
    expect(page.locator('tr[data-source-id="Fixture:metadata-excluded"]')).to_have_count(0)
    page.locator("#status-filter").select_option("all")
    page.locator("#reason-filter").select_option("license-ban")
    expect(page.locator('tr[data-source-id="Fixture:queued-source"]')).to_have_count(0)
    expect(page.locator('tr[data-source-id="Fixture:metadata-excluded"]')).to_have_count(1)
    page.locator("#clear-filters").click()

    ruling = page.locator("#needs-ruling")
    expect(ruling).to_be_visible()
    expect(ruling.locator("li").first).to_contain_text("Fixture · needs-ruling")

    excluded.screenshot(path=str(journey.shots / "status-reason-expanded.png"))
    page.screenshot(path=str(journey.shots / "status-page.png"), full_page=True)


def test_status_page_renders_typed_columns_and_chip_filters(journey: Journey, seeded_extras: dict) -> None:
    """The Use, Method, Format and stage columns render from the canonical enums.

    Every label comes from the served taxonomy: the queued fixture row shows
    its typed Use, Method and Format chips, the stage chip opens its canonical
    explanation, and each chip filter row filters the table.
    """
    page = journey.page
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    base = page.url.rstrip("/") + "/"
    page.goto(base + "status.html")
    expect(page.locator("#rows tr").first).to_be_visible()
    expect(page.locator("#message")).to_have_text("")
    headers = page.locator("#status-table th").all_text_contents()
    assert headers == ["Source", "Status", "Use", "Method", "Format", "Reasons", "Evidence", "Last change"]

    queued = page.locator('tr[data-source-id="Fixture:queued-source"]')
    expect(queued.locator("td").nth(2)).to_have_text("Training")
    expect(queued.locator("td").nth(3)).to_have_text("RL")
    expect(queued.locator("td").nth(4)).to_have_text("Harbor")
    # A row without a typed column renders an explicit absence, never free text.
    excluded = page.locator('tr[data-source-id="Fixture:metadata-excluded"]')
    expect(excluded.locator("td").nth(2)).to_have_text("—")
    expect(excluded.locator("td").nth(3)).to_have_text("—")
    expect(excluded.locator("td").nth(4)).to_have_text("TaskSpec")

    stage_chip = queued.locator(".status-chips summary")
    stage_chip.click()
    explanation = queued.locator(".status-explanation")
    expect(explanation).to_be_visible()
    expect(explanation.locator(".meaning")).to_contain_text("no standard measurement receipt")
    expect(explanation).to_contain_text("Advanced by:")

    # Each chip filter row filters the table; the pressed chip is marked, and a
    # combination that matches nothing shows the empty state.
    expect(page.locator("#stage-filters button", has_text="Intake queued")).to_contain_text("1")
    page.locator("#format-filters button", has_text="Harbor").click()
    expect(page.locator('tr[data-source-id="Fixture:queued-source"]')).to_have_count(1)
    expect(page.locator('tr[data-source-id="Fixture:metadata-excluded"]')).to_have_count(0)
    assert page.locator("#format-filters button", has_text="Harbor").get_attribute("aria-pressed") == "true"
    page.locator("#format-filters button", has_text="TaskSpec").click()
    page.locator("#stage-filters button", has_text="Intake queued").click()
    expect(page.locator("#rows tr")).to_have_count(0)
    expect(page.locator("#empty")).to_be_visible()
    page.locator("#clear-filters").click()
    expect(page.locator('tr[data-source-id="Fixture:metadata-excluded"]')).to_have_count(1)
    page.screenshot(path=str(journey.shots / "status-typed-columns.png"), full_page=True)


def test_untyped_status_fails_loudly_instead_of_rendering_free_text(journey: Journey, seeded_extras: dict) -> None:
    """An out-of-enum status value is served-catalog drift: the page refuses it.

    The status page renders Status only from the canonical taxonomy. A row whose
    status leaves the enum gets the loud error banner naming the value and the
    source, and the table never renders the raw string as a label.
    """
    page = journey.page
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    expect(page.locator("#rows tr").first).to_be_visible()
    base = page.url.rstrip("/") + "/"
    payload = deepcopy(seeded_extras)
    payload["pending"][0]["status"] = "Not selectable"
    page.route("**/measurements/source-status-extras.json", lambda route: route.fulfill(json=payload))
    page.goto(base + "status.html")
    banner = page.locator("#message")
    expect(banner).to_contain_text("Untyped status value")
    expect(banner).to_contain_text("Not selectable")
    expect(page.locator("#rows tr")).to_have_count(0)


def test_superseded_row_links_its_replacement_with_the_replacement_status(journey: Journey, seeded_extras: dict) -> None:
    """A Deprecated superseded row carries its typed replacement link.

    The replacement's own status renders beside the link, and the link opens
    the replacement's source page.
    """
    page = journey.page
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    expect(page.locator("#rows tr").first).to_be_visible()
    base = page.url.rstrip("/") + "/"
    page.goto(base + "status.html")
    row = page.locator('tr[data-source-id="Fixture:superseded"]')
    expect(row).to_have_count(1)
    link = row.locator("a.chip", has_text="Fixture:metadata-excluded")
    expect(link).to_have_count(1)
    href = link.get_attribute("href")
    assert href is not None and "source.html" in href and "Fixture%3Ametadata-excluded" in href
    status = row.locator("span.chip", has_text="Excluded")
    expect(status).to_have_count(1)
    page.screenshot(path=str(journey.shots / "status-replacement-link.png"))
