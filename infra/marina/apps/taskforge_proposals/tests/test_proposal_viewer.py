# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The viewer's schema, ingest upsert and API against a real Postgres."""

import json

from fastapi.testclient import TestClient
from marina.db import UrlDatabase, engine_for
from marina.testing import SourceServices
from sqlalchemy import text
from taskforge_proposals import ingest
from taskforge_proposals.server import app as viewer
from taskforge_proposals.server.proposal import body_sections

APP = "taskforge_proposals"

BODY = (
    "## Task\n\nSolve the cooling recurrence. More text.\n\n```\n## not a heading\n```\n\n"
    "## Grader design and controls\n\nExact match."
)


def proposal_row(capability: str, slot: int, **overrides):
    row = {name: None for name in ingest.names(ingest.PROPOSAL_COLUMNS)}
    row.update(
        capability_id=capability,
        slot=slot,
        subject_id="D01",
        domain="Mathematics",
        status="proposed",
        title=f"Reactor cooling {slot}",
        verification="code",
        environment="reasoning",
        verdict="accept",
        cut_off=False,
        superseded=False,
        completion_tokens=1000 * slot,
        reasoning_tokens=500 * slot,
        prompt_tokens=100,
        proposal={"header": {"id": f"{capability}/{slot}"}, "body": BODY},
        verdict_record={"decision": "accept"},
        source_key="k",
    )
    row.update(overrides)
    return row


def test_body_sections_ignore_fenced_headings():
    sections = body_sections(BODY)
    assert [s["heading"] for s in sections] == ["Task", "Grader design and controls"]
    assert "## not a heading" in sections[0]["text"]


def test_api_filters_pages_and_downloads(database_url: str) -> None:
    engine = engine_for(UrlDatabase(url=database_url), APP)
    with engine.begin() as connection:
        viewer.migrate(connection)
    with engine.begin() as connection:
        viewer.migrate(connection)
        rows = [
            proposal_row("d01.a", 1),
            proposal_row("d01.a", 2, verdict="reject", verification="answer_in_box", title="Gear ratio"),
            proposal_row("d01.a", 3, verdict=None, status="cut_off", cut_off=True, proposal=None, verdict_record=None),
        ]
        sql = ingest.upsert_sql("proposals", ingest.PROPOSAL_COLUMNS, "capability_id, slot")
        for body in ingest.batches(rows):
            connection.execute(text(sql), {"rows": body})
        # Upserting again replaces rows by capability and slot.
        connection.execute(text(sql), {"rows": json.dumps([proposal_row("d01.a", 1, title="Reactor cooling one")])})
        connection.execute(
            text(
                "INSERT INTO capabilities (id, subject_id, domain, status, slots_planned, source_key) "
                "VALUES ('d01.a', 'D01', 'Mathematics', 'done', 3, 'k')"
            )
        )
    client = TestClient(viewer.create_api(SourceServices(name=APP, data_url="", database=UrlDatabase(url=database_url))))

    everything = client.get("/proposals").json()
    assert everything["total"] == 3
    assert everything["rows"][0]["title"] == "Reactor cooling one"
    assert client.get("/proposals", params={"q": "cooling recurrence"}).json()["total"] == 2
    assert client.get("/proposals", params={"q": "d01.a"}).json()["total"] == 3
    assert client.get("/proposals", params={"q": "'; DROP TABLE proposals; --"}).json()["total"] == 0
    assert [r["slot"] for r in client.get("/proposals", params={"verdict": ["reject", "none"]}).json()["rows"]] == [2, 3]
    assert client.get("/proposals", params={"kind": "code", "cut_off": "false"}).json()["total"] == 1
    assert client.get("/proposals", params={"completion_tokens_min": 2000}).json()["total"] == 2
    page = client.get(
        "/proposals", params={"sort": "completion_tokens", "dir": "desc", "page": 2, "page_size": 1}
    ).json()
    assert [r["slot"] for r in page["rows"]] == [2]
    assert client.get("/proposals", params={"sort": "slot; DROP"}).status_code == 400

    stats = client.get("/stats", params={"domain": "D01"}).json()
    assert stats["totals"] == stats["totals"] | {"n": 3, "accepted": 1, "rejected": 1, "cut_off": 1}
    assert sum(stats["histograms"]["completion_tokens"]["bins"]) == 3
    assert {(c["kind"], c["n"]) for c in stats["crosstab"]} == {("code", 2), ("answer_in_box", 1)}

    detail = client.get("/proposals/d01.a/1").json()
    assert [s["slot"] for s in detail["siblings"]] == [1, 2, 3]
    assert detail["proposal"]["sections"][0]["heading"] == "Task"
    raw = client.get("/proposals/d01.a/1/raw/proposal", params={"download": "true"})
    assert raw.headers["content-disposition"].endswith('d01.a.slot-01.proposal.json"')
    assert json.loads(raw.text)["body"] == BODY
    assert client.get("/proposals/d01.a/3/raw/proposal").status_code == 404
    assert client.get("/capabilities/d01.a").json()["capability"]["slots_planned"] == 3
    assert client.get("/meta").json()["counts"] == {"capabilities": 1, "proposals": 3}
