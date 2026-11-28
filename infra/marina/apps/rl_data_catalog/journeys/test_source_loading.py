# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source evidence remains usable when an independent ledger stalls or fails."""

import pytest
from marina.journeys import Journey
from playwright.sync_api import Error, expect

from infra.marina.apps.rl_data_catalog.readability import AUDIT
from infra.marina.apps.rl_data_catalog.release_browser import browser_json, report_failed_request, report_failed_response

APP_ROOT = "a/fb11c931-5861-4878-8bb5-a964d652b45f"
AUXILIARIES = [
    "measurements/content-progress.json",
    "measurements/oracle-inventory.json",
    "api/source-labels",
    "measurements/pending-filtered-releases.json",
]


@pytest.mark.parametrize("auxiliary", AUXILIARIES)
def test_table_source_evidence_finishes_when_auxiliary_never_settles(journey: Journey, auxiliary: str) -> None:
    page = journey.page
    page.clock.install()
    held = []
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    page.locator("#show-everything").click()
    page.route("**/" + auxiliary, lambda route: held.append(route))
    page.locator('tr[data-source-id="Marin:clean"] a[href^="source.html"]').click()
    expect(page.locator("#title")).to_have_text("Marin · clean", timeout=5000)
    expect(page.locator("#sections")).to_have_attribute("aria-busy", "false", timeout=5000)
    expect(page.get_by_role("heading", name="Oversight evidence", exact=True)).to_be_visible()
    expect(page.get_by_text("Loading supplementary evidence…", exact=True)).to_have_count(
        0 if auxiliary == "api/source-labels" else 1
    )
    expect(page.get_by_text("Loading overlap labels…", exact=True)).to_have_count(
        1 if auxiliary == "api/source-labels" else 0
    )
    assert held, f"The test must stall the real {auxiliary} HTTP request"
    page.clock.run_for(8100)
    expect(page.locator("#sections .error")).to_contain_text("Request timed out: " + auxiliary)
    expect(page.locator("#sections")).not_to_contain_text("Loading supplementary evidence")
    expect(page.locator("#sections")).not_to_contain_text("Loading overlap labels")


def test_source_api_timeout_replaces_loading_with_visible_error(journey: Journey) -> None:
    page = journey.page
    page.clock.install()
    held = []
    page.route("**/api/source?*", lambda route: held.append(route))
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    page.goto(page.url + "source.html#source_id=Marin%3Aclean")
    expect(page.locator("#title")).to_have_text("Loading source…")
    assert held
    page.clock.run_for(15100)
    expect(page.locator("#title")).to_have_text("Source evidence")
    expect(page.locator("#message.error")).to_contain_text("Request timed out: api/source?")
    expect(page.locator("#sections")).to_have_attribute("aria-busy", "false")


def test_removed_archive_page_preserves_archive_api(journey: Journey, readability_catalog: None) -> None:
    page = journey.page
    page.goto(f"{journey.origin}/{APP_ROOT}/")
    base = page.url
    expect(page.get_by_role("navigation", name="Atlas views")).not_to_contain_text("Review archive")
    assert page.request.get(base + "archive.html").status == 404
    archive = page.request.get(base + "api/archive")
    assert archive.status == 200
    assert archive.json()["reviews"], "Removing the page must retain stored review collections"


@pytest.mark.parametrize(
    "status,body",
    [(500, "An internal server error"), (404, "Missing ledger"), (200, "not JSON"), (500, "server error " + "x" * 1000)],
)
def test_auxiliary_audit_failure_reports_url_status_and_body(
    journey: Journey, status: int, body: str, caplog: pytest.LogCaptureFixture
) -> None:
    page = journey.page.context.new_page()
    try:
        page.goto(f"{journey.origin}/{APP_ROOT}/")
        url = page.url + "measurements/audit-fixture.json?source=example"
        page.route("**/measurements/audit-fixture.json?*", lambda route: route.fulfill(status=status, body=body))
        with page.expect_response(url) as response:
            with pytest.raises(Error) as failure:
                browser_json(page, url)
        assert url in str(failure.value)
        assert f"HTTP {status}" in str(failure.value)
        prefix = body.encode()[:256].decode()
        assert prefix in str(failure.value)
        if len(body) > 256:
            assert body not in str(failure.value)
        report_failed_response(response.value)
        if status >= 400:
            assert url in caplog.text and f"HTTP {status}" in caplog.text and prefix in caplog.text
    finally:
        page.close()


def test_auxiliary_audit_returns_valid_json(journey: Journey) -> None:
    page = journey.page.context.new_page()
    try:
        page.goto(f"{journey.origin}/{APP_ROOT}/")
        page.route("**/measurements/audit-fixture.json", lambda route: route.fulfill(json={"sources": []}))
        assert browser_json(page, "measurements/audit-fixture.json") == {"sources": []}
    finally:
        page.close()


def test_auxiliary_audit_network_failure_identifies_request(journey: Journey, caplog: pytest.LogCaptureFixture) -> None:
    page = journey.page.context.new_page()
    try:
        page.goto(f"{journey.origin}/{APP_ROOT}/")
        url = page.url + "measurements/audit-fixture.json"
        page.route("**/measurements/audit-fixture.json", lambda route: route.abort("connectionfailed"))
        with page.expect_event("requestfailed") as request:
            with pytest.raises(Error) as failure:
                browser_json(page, url)
        assert url in str(failure.value) and "HTTP unavailable" in str(failure.value)
        report_failed_request(request.value)
        assert url in caplog.text and "body prefix unavailable" in caplog.text
    finally:
        page.close()


@pytest.mark.parametrize("theme,width", [("light", 1440), ("light", 390), ("dark", 1440), ("dark", 390)])
@pytest.mark.parametrize("asset", ["common.js", "difficulty.js", "source.js", "source-bootstrap.js"])
def test_failed_source_asset_shows_error_and_reload_recovers_same_source(
    journey: Journey, theme: str, width: int, asset: str
) -> None:
    page = journey.page.context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        page.set_viewport_size({"width": width, "height": 1000})
        page.emulate_media(color_scheme=theme)
        page.goto(f"{journey.origin}/{APP_ROOT}/")
        root = page.url
        page.route("**/" + asset, lambda route: route.fulfill(status=500, body=""))
        page.goto(root + "source.html#source_id=Marin%3Aclean")
        expect(page.locator("#title")).to_have_text("Source evidence", timeout=5000)
        expect(page.locator("#message.error")).to_be_visible()
        expect(page.locator("#message")).not_to_have_text("")
        reload = page.get_by_role("link", name="Reload page", exact=True)
        expect(reload).to_be_visible()
        assert not errors, errors
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        report = page.evaluate(AUDIT, {})
        assert not report["findings"], report["findings"]
        reload.focus()
        assert not page.evaluate(AUDIT, {"mode": "focus"})["findings"]
        page.screenshot(path=str(journey.shots / f"asset-error-{asset}-{theme}-{width}.png"))
        page.unroute("**/" + asset)
        with page.expect_navigation(wait_until="domcontentloaded"):
            reload.click()
        expect(page.locator("#title")).to_have_text("Marin · clean", timeout=5000)
        expect(page.locator("#sections")).to_have_attribute("aria-busy", "false", timeout=5000)
        expect(page.get_by_role("link", name="Reload page", exact=True)).to_be_hidden()
        assert page.url.endswith("#source_id=Marin%3Aclean")
    finally:
        page.close()
