# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Someone opens Echo, reads the search page, looks up a string, and opens the wiki.

The query embedder and the reranker are ONNX models this journey deliberately never wakes:
``grep`` is the exact-substring search, and the wiki list is the most recent notes, so the
walk needs the database and nothing else.
"""

from marina.journeys import Journey


def test_search_page_is_ready_to_take_a_query(journey: Journey) -> None:
    journey.visit("/").sees("Search across Marin.").shoot("search")
    offered = journey.offers()
    assert "input:Identifier, incident, question, or phrase…" in offered
    assert "button:Search" in offered
    # Discord is the one domain a search does not select for you.
    assert "Discord" in journey.reads()


def test_grep_finds_a_seeded_activity_chunk(journey: Journey) -> None:
    journey.visit("/")
    hits = journey.api("/echo/api/grep?pattern=ragged_all_to_all")
    assert isinstance(hits, list)
    assert [hit["url"] for hit in hits] == ["https://github.com/marin-community/marin/issues/1"]
    assert hits[0]["title"] == "ragged_all_to_all overflows on the second shard"


def test_wiki_notes_open_from_the_shell_nav(journey: Journey) -> None:
    journey.visit("/").click("Wiki").sees("Grafana access for a new engineer")
    assert journey.page.url.endswith("/echo/wiki")
    journey.click("Grafana access for a new engineer").sees("membership takes 15 minutes")
    journey.shoot("wiki-note")
