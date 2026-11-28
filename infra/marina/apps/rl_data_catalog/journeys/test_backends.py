# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Backend Evidence page renders the production-backend enum and typed reasons."""

import re
from urllib.parse import urlencode

from marina.journeys import Journey
from playwright.sync_api import expect

APP_ID = "fb11c931-5861-4878-8bb5-a964d652b45f"


def test_backend_evidence_renders_the_production_enum_and_typed_reasons(journey: Journey) -> None:
    """Dev-only backends never render; unsupported production backends show typed reason chips.

    The fixture survey measures Marin:clean on docker, docker-gvisor, qemu, local
    and daytona; the fit table keeps the production enum members (local,
    daytona) only, and each unsupported row's reasons come from the canonical
    backend-reason table.
    """
    page = journey.page
    page.goto(f"{journey.origin}/a/{APP_ID}/")
    expect(page.locator("#rows tr").first).to_be_visible()
    base = page.url.rstrip("/") + "/"
    page.goto(base + "backends.html#" + urlencode({"source_id": "Marin:clean"}))
    rows = page.locator("#fit table tbody tr")
    expect(rows).to_have_count(2)
    backends = rows.locator("td:first-child").all_text_contents()
    assert backends == ["LOCAL", "Daytona"]
    # Every production row renders typed reason chips from the canonical table.
    local_reasons = rows.nth(0).locator(".backends-cell .chip").all_text_contents()
    daytona_reasons = rows.nth(1).locator(".backends-cell .chip").all_text_contents()
    assert local_reasons == ["docker_image"], local_reasons
    assert daytona_reasons == ["packages_lock"], daytona_reasons
    page.screenshot(path=str(journey.shots / "backends-production-enum.png"), full_page=True)


def test_backend_evidence_fails_loudly_on_an_unmapped_blocker(journey: Journey) -> None:
    """A blocker key outside the reason vocabulary is drift: loud error, no free-text reason."""
    page = journey.page
    page.goto(f"{journey.origin}/a/{APP_ID}/")
    expect(page.locator("#rows tr").first).to_be_visible()
    source = page.evaluate("async () => (await (await fetch('api/source?source_id=Marin%3Aclean')).json())")
    capabilities = dict(source["backend_capabilities"])
    fit = dict(capabilities["backends"]["local"])
    fit["blockers"] = {"agent:some_mystery_demand": 3}
    capabilities["backends"] = {**capabilities["backends"], "local": fit}
    page.route(
        "**/api/source?*",
        lambda route: route.fulfill(json={**source, "backend_capabilities": capabilities}),
    )
    base = page.url.rstrip("/") + "/"
    page.goto(base + "backends.html#" + urlencode({"source_id": "Marin:clean"}))
    banner = page.locator("#message")
    expect(banner).to_contain_text("some_mystery_demand")
    expect(page.locator("#fit table")).to_have_count(0)


def test_backend_evidence_types_an_unsupported_environment_kind(journey: Journey) -> None:
    """R40: a production backend's not_applicable environment kind (host image) refused the whole page."""
    page = journey.page
    page.goto(f"{journey.origin}/a/{APP_ID}/")
    expect(page.locator("#rows tr").first).to_be_visible()
    source = page.evaluate("async () => (await (await fetch('api/source?source_id=Marin%3Aclean')).json())")
    capabilities = dict(source["backend_capabilities"])
    fit = dict(capabilities["backends"]["local"])
    fit["not_applicable"] = {"grader:host_image": 3}
    capabilities["backends"] = {**capabilities["backends"], "local": fit}
    page.route(
        "**/api/source?*",
        lambda route: route.fulfill(json={**source, "backend_capabilities": capabilities}),
    )
    base = page.url.rstrip("/") + "/"
    page.goto(base + "backends.html#" + urlencode({"source_id": "Marin:clean"}))
    expect(page.locator("#title")).to_have_text("Marin · clean")
    expect(page.locator("#message")).not_to_have_class(re.compile("error"))
    local = page.locator("#fit table tbody tr").filter(has=page.get_by_role("cell", name="LOCAL", exact=True))
    expect(local.locator(".backends-cell .chip", has_text="Unsupported environment kind")).to_have_count(1)
