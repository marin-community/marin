# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A person browses typed QC findings and manually audits one source's reviews."""

import json
from collections.abc import Iterator
from uuid import UUID

import pytest
from marina.applets import applet_schema
from marina.db import DatabaseSpec, engine_for
from marina.journeys import Journey
from playwright.sync_api import expect
from sqlalchemy import text

from infra.marina.apps.rl_data_catalog.server.app import migrate
from infra.marina.apps.rl_data_catalog.server.content_access import backend_metadata
from infra.marina.apps.rl_data_catalog.server.evidence_content import pack_content

APP_ID = UUID("fb11c931-5861-4878-8bb5-a964d652b45f")
SOURCE = "Marin:sample-content"
KEY = "qc-audit"

# (task_id, cohort, verdict, defects, confidence, reference_status)
ITEMS = [
    ("good-a", "admitted", "good", [], "high", "consistent"),
    ("good-b", "admitted", "good", [], "medium", "consistent"),
    ("good-c", "admitted", "good", [], "low", "unknown"),
    ("issues-a", "admitted", "some_issues", ["ambiguity"], "medium", "consistent"),
    ("issues-b", "admitted", "some_issues", ["ambiguity", "malformed"], "low", "consistent"),
    ("bad-a", "admitted", "bad", ["answer_leakage"], "high", "conflict"),
    ("drop-a", "excluded", "bad", ["missing_context"], "medium", "consistent"),
    ("drop-b", "excluded", "unknown", [], None, "unknown"),
]


@pytest.fixture
def qc_catalog(marina_kernel, marina_database: DatabaseSpec) -> Iterator[None]:
    """One source with typed SAMPLE judgments, packets and a review archive."""
    engine = engine_for(marina_database, applet_schema(APP_ID))
    with engine.begin() as connection:
        connection.execute(
            text("UPDATE atlas_sources SET assessment_key=:key WHERE load_id='lexical' AND id=:source"),
            {"key": KEY, "source": SOURCE},
        )
        connection.execute(
            text(
                """
            INSERT INTO atlas_assessments VALUES ('lexical',:key,:source,'sample',
                '{"model": "fixture","revision": "pinned","max_tokens": 4096,"max_attempts": 3,
                  "retry_max_tokens": 8192}','Assess task quality.',
                '{"status": "complete","verification": null,"admission_counts": {"admitted": 6,"excluded": 2}}',
                'https://example.org/curation.json','[]')
        """
            ),
            {"key": KEY, "source": SOURCE},
        )
        connection.execute(text("INSERT INTO atlas_populations VALUES ('lexical',:key,'admitted',40,6)"), {"key": KEY})
        connection.execute(text("INSERT INTO atlas_populations VALUES ('lexical',:key,'excluded',12,2)"), {"key": KEY})
        for task_id, cohort, verdict, defects, confidence, reference in ITEMS:
            packet = {"task": {"id": task_id, "prompt": "Solve the sampled problem."}, "instructions": "Attempt it."}
            connection.execute(
                text(
                    """
                INSERT INTO atlas_items (load_id, pipeline_key, task_id, cohort, status, verdict, explanation,
                    evidence_ids, packet_uri, has_task, omissions, exclusion_reason,
                    confidence, reference_status, defects, provenance, replaced_task_sha256, repair_advisories)
                VALUES ('lexical',:key,:task,:cohort,'reviewed',:verdict,:explanation,'["task"]',
                    'https://example.org/packet.json',true,'[]',NULL,:confidence,:reference,
                    CAST(:defects AS JSONB),NULL,NULL,'[]'::jsonb)
            """
                ),
                {
                    "key": KEY,
                    "task": task_id,
                    "cohort": cohort,
                    "verdict": verdict,
                    "explanation": f"Recorded finding for {task_id}.",
                    "confidence": confidence,
                    "reference": reference,
                    "defects": json.dumps(defects),
                },
            )
            connection.execute(
                text("INSERT INTO atlas_packets VALUES ('lexical',:key,:task,CAST(:packet AS JSONB))"),
                {"key": KEY, "task": task_id, "packet": json.dumps(packet)},
            )
            connection.execute(
                text(
                    "INSERT INTO atlas_raw_reviews (load_id,pipeline_key,task_id,record) "
                    "VALUES ('lexical',:key,:task,CAST(:record AS JSONB))"
                ),
                {
                    "key": KEY,
                    "task": task_id,
                    "record": json.dumps(
                        pack_content(
                            {
                                "input": packet,
                                "output": {"verdict": verdict, "explanation": f"Recorded finding for {task_id}."},
                                "provenance": {"uri": "s3://fixture/review.json", "sha256": "a" * 64},
                            }
                        )
                    ),
                },
            )
    try:
        with engine.begin() as connection:
            migrate(connection)
        yield
    finally:
        with engine.begin() as connection:
            connection.execute(
                text("UPDATE atlas_sources SET assessment_key=NULL WHERE load_id='lexical' AND id=:source"),
                {"source": SOURCE},
            )
            for table in ["atlas_raw_reviews", "atlas_packets", "atlas_items", "atlas_populations", "atlas_assessments"]:
                connection.execute(
                    text(f"DELETE FROM {table} WHERE load_id='lexical' AND pipeline_key=:key"), {"key": KEY}
                )
        engine.dispose()


@pytest.mark.timeout(600)
def test_browse_filter_and_manual_audit(journey: Journey, qc_catalog: None) -> None:
    page = journey.page
    page.on(
        "response",
        lambda response: (
            print("HTTP404-DEBUG", response.status, response.url)
            if response.status >= 400 and response.url.startswith(journey.origin)
            else None
        ),
    )
    probe = page.request.get(f"{journey.origin}/a/{APP_ID}/v/1/api/assessment?source={SOURCE}")
    print("API-DEBUG", probe.status, probe.text()[:300])
    page.goto(f"{journey.origin}/a/{APP_ID}/v/1/assessment.html?source={SOURCE}", wait_until="domcontentloaded")
    page.evaluate("() => document.fonts.ready")
    journey.sees("Retained task opinions")
    journey.sees("Ambiguity 2")
    journey.sees("Answer leakage 1 · 16.7%")
    journey.sees("Reviewer confidence: High 2 · Medium 2 · Low 2")
    journey.sees("High 0 · Medium 1 · Low 0 · unrecorded 1")
    journey.shoot("qc-assessment")

    # Filtering by one defect narrows the review list to matching tasks.
    journey.click("Answer leakage (1)")
    journey.sees("1-1 of 1 matching reviews")
    journey.shoot("qc-filtered")
    journey.click("Answer leakage (1)")

    # A person draws a seeded audit sample straight from the page.
    journey.fill("Seed", "journey-2026-11-16")
    journey.click("Draw sample")
    journey.sees("Sample stratified · seed journey-2026-11-16 · 40 tasks")
    journey.sees("No judgments recorded yet.")
    journey.shoot("qc-audit-sample")

    # They disagree with the reviewer and record a corrected judgment.
    journey.click("Audit · Some issues · issues-a")
    journey.click("Disagree")
    journey.click("Record judgment")
    journey.sees("Judged 1 tasks · agreement 0%")
    journey.sees("Judged 1 tasks · agreement 0%")
    journey.shoot("qc-audit-judged")


@pytest.mark.parametrize("width,theme", [(1440, "light"), (1440, "dark"), (390, "light"), (390, "dark")])
def test_quality_chip_reaches_review_list_and_raw_review(
    journey: Journey, qc_catalog: None, width: int, theme: str
) -> None:
    page = journey.page
    page.set_viewport_size({"width": width, "height": 1000})
    page.emulate_media(color_scheme=theme)
    page.goto(f"{journey.origin}/a/{APP_ID}/")
    row = page.locator(f'tr[data-source-id="{SOURCE}"]')
    page.get_by_role("button", name="Show everything", exact=True).click()
    row.get_by_role("link", name="SAMPLE: browse task reviews", exact=True).click()
    expect(page.get_by_role("heading", name="Browse reviews", exact=True)).to_be_visible()
    summary = page.locator("#reviews summary").first
    chips = summary.locator(".chip").count()
    summary.locator("span").first.click()
    expect(summary.locator(".chip")).to_have_count(chips)
    summary.locator("span").first.click()
    expect(summary.locator(".chip")).to_have_count(chips)
    page.get_by_role("link", name="Open raw review →").first.click()
    expect(page.get_by_role("heading", name="Reviewer output", exact=True)).to_be_visible()
    expect(page.get_by_role("heading", name="Evidence the reviewer saw", exact=True)).to_be_visible()
    expect(page.locator("#content")).to_contain_text("Solve the sampled problem.")
    page.screenshot(path=str(journey.shots / f"qc-raw-{theme}-{width}.png"))


@pytest.mark.parametrize("entry", ["table", "source"])
@pytest.mark.parametrize("width,theme", [(1440, "light"), (1440, "dark"), (390, "light"), (390, "dark")])
@pytest.mark.timeout(600)
def test_source_solver_links_reach_attempt_and_rollout(journey: Journey, width: int, theme: str, entry: str) -> None:
    page = journey.page
    page.set_viewport_size({"width": width, "height": 1000})
    page.emulate_media(color_scheme=theme)
    page.goto(f"{journey.origin}/a/{APP_ID}/")
    if entry == "table":
        page.locator("#show-everything").click()
        page.locator('tr[data-source-id="Marin:clean"] td.difficulty-cell a').click(timeout=5000)
    else:
        page.goto(page.url + "source.html#source_id=Marin%3Aclean")
        page.get_by_role("link", name="Browse task attempts and rollouts →").first.click(timeout=5000)
    expect(page.get_by_role("heading", name="Difficulty attempts · Marin · clean")).to_be_visible()
    page.get_by_role("link", name="0", exact=True).click()
    expect(page.get_by_role("heading", name="Transcript and tool calls")).to_be_visible()

    # The conversation renders as role-labeled turns, not raw JSON.
    transcript = page.locator(".tr-transcript").first
    expect(transcript.locator(".tr-turn")).to_have_count(6)
    for label in ("system", "user", "assistant", "tool result"):
        expect(transcript.locator(".tr-turn-label", has_text=label).first).to_be_visible()
    expect(transcript.locator(".tr-tool-call", has_text="terminal")).to_have_count(1)
    expect(transcript.locator(".tr-tool-call", has_text="python")).to_have_count(1)

    # Long text folds: the user turn and the big tool output expand and collapse again.
    transcript.locator(".tr-toggle", has_text="Show full message").first.click()
    expect(transcript).to_contain_text("Mention the checksum of the listing explicitly")
    transcript.locator(".tr-toggle", has_text="Collapse").first.click()
    expect(transcript).not_to_contain_text("Mention the checksum of the listing explicitly")
    transcript.locator(".tr-toggle", has_text="Show full output").first.click()
    expect(transcript.locator(".tr-toggle", has_text="Fold output")).to_have_count(1)
    transcript.locator(".tr-toggle", has_text="Fold output").first.click()
    expect(transcript.locator(".tr-toggle", has_text="Show full output")).to_have_count(1)

    # Reasoning rides beside the answer, collapsed until opened.
    reasoning = transcript.locator(".tr-reasoning").first
    expect(reasoning.locator("summary")).to_be_visible()
    expect(reasoning.locator(".tr-reasoning-body")).not_to_be_visible()
    reasoning.locator("summary").click()
    expect(reasoning.locator(".tr-reasoning-body")).to_be_visible()
    reasoning.locator("summary").click()

    # The original bytes stay one click away for both transcript-shaped sections.
    for section_title in ("Transcript and tool calls", "Model input"):
        section = page.locator(".assessment-card", has=page.get_by_role("heading", name=section_title)).first
        summary = section.locator(".raw-json summary")
        summary.click()
        expect(section.locator(".raw-json-view")).to_contain_text('"events"')
        summary.click()

    expect(page.get_by_role("heading", name="Verifier output")).to_be_visible()
    expect(page.get_by_role("link", name="Open in Filer ↗")).to_have_count(2)
    page.screenshot(path=str(journey.shots / f"rollout-{entry}-{theme}-{width}.png"))


@pytest.mark.parametrize("width,theme", [(1440, "light"), (1440, "dark"), (390, "light"), (390, "dark")])
def test_license_withheld_pages_do_not_request_task_content(journey: Journey, width: int, theme: str) -> None:
    page = journey.page
    page.set_viewport_size({"width": width, "height": 1000})
    page.emulate_media(color_scheme=theme)
    marker = "Content withheld: this source has a license exclusion."
    requests = []
    page.on("request", lambda request: requests.append(request.url))
    page.route(
        "**/api/source?**",
        lambda route: route.fulfill(
            json={
                "declaration": {
                    "id": "Marin:clean",
                    "display_name": "clean",
                    "data_restriction": "open",
                    "content_withheld": True,
                },
                "marker": marker,
            }
        ),
    )
    for path in (
        "attempts.html?source=Marin%3Aclean",
        "rollout.html?source=Marin%3Aclean&measurement=x&attempt=y",
        "assessment.html?source=Marin%3Aclean",
    ):
        page.goto(f"{journey.origin}/a/{APP_ID}/v/1/{path}")
        expect(page.locator("#message")).to_have_text(marker)
        page.screenshot(path=str(journey.shots / f"withheld-{path.split('.')[0]}-{theme}-{width}.png"))
    capabilities = backend_metadata(
        {
            "source_id": "Marin:clean",
            "basis": "converted_spec_census",
            "task_count": 1,
            "backends": {
                "daytona": {
                    "support": "unsupported",
                    # A complete survey row: both roles measured, the blocker typed.
                    "roles": {"agent": {"unsupported": 1}, "grader": {"supported": 1}},
                    "task_counts": {"unsupported": 1},
                    "blockers": {"agent:docker_image": 1},
                    "validation_errors": {"private task text": 1},
                }
            },
        }
    )
    # Backend labels render from the served taxonomy; this page has no fit table of its own.
    fits = page.evaluate(
        "async source => { await loadTaxonomy(); return backendLabels(source); }",
        {"backend_capabilities": capabilities},
    )
    assert fits[0]["fit"]["support"] == "unsupported"
    assert fits[0]["fit"]["roles"]["agent"] == {"unsupported": 1}
    assert not any(
        "/api/difficulty-attempts?" in url or "/api/difficulty-rollout?" in url or "/api/assessment?" in url
        for url in requests
    )
