# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from typing import cast

import pytest
from fastapi.testclient import TestClient
from marina.applets import AppletServices
from marina.client import AppletQuery
from marina.testing import EngineServices, engine_query, migrated_schema
from sqlalchemy.engine import Engine

from infra.marina.apps.sft_glitch_gallery.categories import Category, Example, Severity
from infra.marina.apps.sft_glitch_gallery.detect import (
    AFTER_CHARS,
    BEFORE_CHARS,
    HEAD_CHARS,
    Match,
    detect,
    window,
)
from infra.marina.apps.sft_glitch_gallery.load import load_gallery
from infra.marina.apps.sft_glitch_gallery.server.app import create_api, migrate

SYSTEM = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>Reasoning: /think<|eot_id|>"


def turn(role: str, content: str) -> str:
    return f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"


def found(source: str, text: str) -> dict[str, Example]:
    return {example.category: example for example in detect(iter([(source, 7, 100, text)]), 5)}


@pytest.fixture
def gallery_database(database_url: str) -> Iterator[Engine]:
    with migrated_schema(database_url, "sft_glitch_test", migrate) as engine:
        yield engine


@pytest.fixture
def gallery_query(gallery_database: Engine) -> AppletQuery:
    return engine_query(gallery_database)


def test_quoted_instruction_is_phantom_only_when_the_user_never_gave_it() -> None:
    source = "nemotron_sft_v3/competitive_programming_v2/competitive_coding_python"
    reasoning = "<|start_think|>The instruction says: Never use sys.stdin.buffer here.<|end_think|>print(1)"
    stripped = SYSTEM + turn("user", "Read n and print it.") + turn("assistant", reasoning)
    intact = SYSTEM + turn("user", "Read n. Do not use sys.stdin.buffer.") + turn("assistant", reasoning)

    example = found(source, stripped)["phantom_instructions"]

    assert example.text[example.match_start :][: example.match_length] == "Never use sys.stdin.buffer"
    assert "phantom_instructions" not in found(source, intact)


def test_harness_warning_counts_only_after_a_turn_that_is_valid_json() -> None:
    warning = "Previous response had warnings:\nWARNINGS: - Extra text detected before JSON object"
    clean = SYSTEM + turn("user", "task") + turn("assistant", '{"commands": []}') + turn("user", warning)
    deserved = SYSTEM + turn("user", "task") + turn("assistant", 'Sure! {"commands": []}') + turn("user", warning)

    assert found("agenttrove", clean)["harness_warning"].note == "the assistant turn above it is valid JSON"
    assert "harness_warning" not in found("agenttrove", deserved)


def test_give_up_in_the_reasoning_tail_is_found_and_a_plain_solution_is_not() -> None:
    source = "nemotron_sft_v3/math_v3/train"
    guessed = "<|start_think|>Hard. Given the time, I'll answer 1.<|end_think|>\\boxed{1}"
    solved = "<|start_think|>2 + 2 = 4.<|end_think|>\\boxed{4}"

    example = found(source, SYSTEM + turn("user", "Solve.") + turn("assistant", guessed))["give_up"]

    assert example.note == "at the end of the reasoning"
    assert "give_up" not in found(source, SYSTEM + turn("user", "Solve.") + turn("assistant", solved))


def test_window_keeps_the_head_and_the_match_of_a_long_document() -> None:
    text = "H" * HEAD_CHARS + "x" * 50_000 + "FLAW" + "y" * 50_000

    clipped, start = window(text, Match(HEAD_CHARS + 50_000, 4, ""))

    assert clipped[start:][:4] == "FLAW"
    assert clipped.startswith("H" * HEAD_CHARS + "\n\n[… ")
    assert len(clipped) < HEAD_CHARS + BEFORE_CHARS + AFTER_CHARS + 200


def test_loaded_gallery_serves_categories_in_order_and_replaces_the_previous_load(
    gallery_database: Engine, gallery_query: AppletQuery
) -> None:
    categories = (
        Category("loops", "Loops", Severity.HIGH, "Repeats.", "3 of 100"),
        Category("quiet", "Quiet", Severity.LOW, "Nothing found.", "0 of 100"),
    )
    example = Example("loops", "math/a", 5, 900, "a loop", "before LOOP LOOP after", 7, 9)
    earlier_by_name = Example("loops", "aaa/z", 1, 50, "listed second", "x", 0, 1)
    load_gallery(categories, [example, earlier_by_name], gallery_query)

    with TestClient(create_api(cast(AppletServices, EngineServices(gallery_database)))) as client:
        listed = client.get("/categories").json()
        assert [(category["key"], len(category["examples"])) for category in listed] == [("loops", 2), ("quiet", 0)]
        assert listed[0]["examples"] == [
            {"source": "math/a", "doc_index": 5, "tokens": 900, "note": "a loop"},
            {"source": "aaa/z", "doc_index": 1, "tokens": 50, "note": "listed second"},
        ]
        document = {"category": "loops", "source": "math/a", "doc_index": 5}
        served = client.get("/example", params=document).json()
        assert served["text"][served["match_start"] :][: served["match_length"]] == "LOOP LOOP"

        load_gallery(categories[:1], [], gallery_query)
        assert [category["key"] for category in client.get("/categories").json()] == ["loops"]
        assert client.get("/example", params=document).status_code == 404


def test_load_rejects_examples_of_an_undefined_category(gallery_query: AppletQuery) -> None:
    with pytest.raises(ValueError, match="mystery"):
        load_gallery((), [Example("mystery", "s", 1, 1, "", "text", 0, 1)], gallery_query)
