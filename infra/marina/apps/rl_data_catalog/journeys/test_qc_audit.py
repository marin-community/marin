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
from sqlalchemy import text

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
    try:
        yield
    finally:
        with engine.begin() as connection:
            connection.execute(
                text("UPDATE atlas_sources SET assessment_key=NULL WHERE load_id='lexical' AND id=:source"),
                {"source": SOURCE},
            )
            for table in ["atlas_packets", "atlas_items", "atlas_populations", "atlas_assessments"]:
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
