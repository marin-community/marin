# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AmazonScience/massive function-calling converter."""

from __future__ import annotations

import json
import tarfile

import pytest

from marin.datakit.download import massive
from marin.datakit.download.massive import (
    MAX_DISTRACTORS,
    MIN_DISTRACTORS,
    TOOLS,
    _INTENT_SLOTS,
    parse_annot_utt,
    render_function_call,
    row_to_doc,
    select_tools,
    transform_staged_massive,
)


def test_parse_annot_utt_extracts_pairs_in_document_order():
    annot = "wake me up at [time : nine am] on [date : friday]"
    assert parse_annot_utt(annot) == [("time", "nine am"), ("date", "friday")]


def test_parse_annot_utt_handles_no_slots():
    assert parse_annot_utt("olly quiet") == []


def test_parse_annot_utt_preserves_repeated_slots():
    annot = "if it is [timeofday : noon] in [place_name : virginia] what time is it in [place_name : california]"
    assert parse_annot_utt(annot) == [
        ("timeofday", "noon"),
        ("place_name", "virginia"),
        ("place_name", "california"),
    ]


def test_parse_annot_utt_keeps_internal_punctuation():
    # Apostrophes and dotted abbreviations live inside slot values.
    annot = "order from [business_name : byron's] at [time : g. m. t. plus five]"
    assert parse_annot_utt(annot) == [
        ("business_name", "byron's"),
        ("time", "g. m. t. plus five"),
    ]


def test_parse_annot_utt_handles_unicode_values():
    # Non-Latin slot values should pass through unchanged.
    annot = "[place_name : 北京] [time : 9時]"
    assert parse_annot_utt(annot) == [("place_name", "北京"), ("time", "9時")]


def test_parse_annot_utt_bails_on_malformed_marker():
    # A stray '[' with no matching ':' or ']' should not raise; we just stop parsing.
    annot = "good [date : today] then [oops nothing here"
    assert parse_annot_utt(annot) == [("date", "today")]


def test_tools_registry_has_one_entry_per_intent():
    assert len(TOOLS) == len(_INTENT_SLOTS) == 60
    names = [t["name"] for t in TOOLS]
    assert names == sorted(_INTENT_SLOTS)
    assert len(set(names)) == 60


def test_tool_schema_uses_responses_api_flat_shape():
    # Responses API tools[] entries are flat (name/description/parameters at top
    # level, no nested ``function`` wrapper). Every slot is array-of-string so
    # multi-value utterances round-trip without a union type at the schema level.
    for tool in TOOLS:
        assert tool["type"] == "function"
        assert "function" not in tool, "Responses API has no nested 'function' key"
        assert "name" in tool and "parameters" in tool
        props = tool["parameters"]["properties"]
        assert props, f"intent {tool['name']} has no slots"
        for spec in props.values():
            assert spec["type"] == "array"
            assert spec["items"]["type"] == "string"


def _row(intent: str, **overrides) -> dict:
    """Build a fake MASSIVE row matching the upstream raw-JSONL shape.

    Raw MASSIVE files store ``intent`` as a string name; ``row_to_doc`` and
    the zephyr workers consume that directly.
    """
    base = {
        "id": "1",
        "locale": "en-US",
        "partition": "train",
        "intent": intent,
        "scenario": 0,
        "utt": "wake me up at nine am on friday",
        "annot_utt": "wake me up at [time : nine am] on [date : friday]",
    }
    base.update(overrides)
    return base


def test_row_to_doc_renders_tools_request_tool_call_text():
    [doc] = row_to_doc(_row("alarm_set"))
    tools_line, request_line, tool_call_line = doc["text"].split("\n")

    assert tools_line.startswith("Tools: ")
    embedded_tools = json.loads(tools_line.removeprefix("Tools: "))
    embedded_names = {t["name"] for t in embedded_tools}
    # Gold tool always present.
    assert "alarm_set" in embedded_names
    # Subset is in the configured size band: 1 gold + [MIN, MAX] distractors,
    # capped by what's available outside the gold (59).
    max_total = 1 + min(MAX_DISTRACTORS, len(TOOLS) - 1)
    assert 1 + MIN_DISTRACTORS <= len(embedded_tools) <= max_total
    assert embedded_names <= {t["name"] for t in TOOLS}

    assert request_line == "Request: wake me up at nine am on friday"

    assert tool_call_line.startswith("tool_call: ")
    call = json.loads(tool_call_line.removeprefix("tool_call: "))
    assert call["type"] == "function_call"
    assert call["call_id"] == "call_en-US_1"
    assert call["name"] == "alarm_set"
    # Responses API encodes ``arguments`` as a JSON string, not a nested object.
    assert json.loads(call["arguments"]) == {"date": ["friday"], "time": ["nine am"]}

    assert doc["intent"] == "alarm_set"
    assert doc["locale"] == "en-US"
    assert doc["split"] == "train"
    assert "tools" not in doc, "tools live in `text`, not as a sibling column"


def test_select_tools_is_deterministic_per_seed():
    a = select_tools("alarm_set", "en-US/1/train")
    b = select_tools("alarm_set", "en-US/1/train")
    assert a == b, "same seed must produce identical selection AND ordering"


def test_select_tools_always_includes_gold():
    for intent in ("alarm_set", "audio_volume_mute", "general_greet", "qa_definition"):
        for seed in ("en-US/1/train", "ja-JP/42/test", "ar-SA/9999/validation"):
            names = [t["name"] for t in select_tools(intent, seed)]
            assert intent in names, f"gold {intent!r} missing for seed {seed!r}"


def test_select_tools_size_in_distractor_band():
    sizes = {len(select_tools("alarm_set", f"seed/{i}/train")) for i in range(200)}
    # 1 gold + [MIN, MAX_clamped] distractors, where MAX_clamped = min(MAX_DISTRACTORS, len(TOOLS)-1).
    max_total = 1 + min(MAX_DISTRACTORS, len(TOOLS) - 1)
    assert min(sizes) >= 1 + MIN_DISTRACTORS
    assert max(sizes) <= max_total
    # 200 seeds should hit a wide spread of the band.
    assert len(sizes) > 5, f"distractor count appears stuck at a single value: {sizes}"


def test_select_tools_permutes_within_a_seed_family():
    a = [t["name"] for t in select_tools("alarm_set", "en-US/1/train")]
    b = [t["name"] for t in select_tools("alarm_set", "en-US/2/train")]
    assert a != b, "different seeds should not produce identical orderings"


def test_render_function_call_matches_responses_api_shape():
    rendered = render_function_call("call_42", "alarm_set", {"time": ["nine am"], "date": ["friday"]})
    call = json.loads(rendered)
    assert call == {
        "type": "function_call",
        "call_id": "call_42",
        "name": "alarm_set",
        "arguments": '{"date": ["friday"], "time": ["nine am"]}',
    }


def _parse_tool_call(doc: dict) -> dict:
    return json.loads(doc["text"].rsplit("\ntool_call: ", 1)[1])


def test_row_to_doc_groups_repeated_slots_into_array():
    row = _row(
        "datetime_convert",
        utt="if it is noon in virginia what time is it in california",
        annot_utt=(
            "if it is [timeofday : noon] in [place_name : virginia] what time is it in [place_name : california]"
        ),
    )
    [doc] = row_to_doc(row)
    call = _parse_tool_call(doc)
    assert json.loads(call["arguments"]) == {
        "timeofday": ["noon"],
        "place_name": ["virginia", "california"],
    }


def test_row_to_doc_emits_empty_arguments_for_intent_without_slots():
    row = _row("audio_volume_mute", utt="olly quiet", annot_utt="olly quiet")
    [doc] = row_to_doc(row)
    call = _parse_tool_call(doc)
    assert call["name"] == "audio_volume_mute"
    assert json.loads(call["arguments"]) == {}


def test_row_to_doc_remaps_partition_dev_to_validation():
    [doc] = row_to_doc(_row("alarm_set", partition="dev"))
    assert doc["split"] == "validation"


def test_row_to_doc_id_is_stable_across_locale_and_split():
    [en] = row_to_doc(_row("alarm_set", locale="en-US"))
    [de] = row_to_doc(_row("alarm_set", locale="de-DE"))
    assert en["id"] != de["id"]
    assert en["id"].startswith("en-US/")
    assert de["id"].startswith("de-DE/")


def _build_fake_tarball(path, locale_rows: dict[str, list[dict]]) -> None:
    """Write a ``.tar.gz`` mimicking the upstream MASSIVE archive layout.

    Each member is ``1.1/data/{locale}.jsonl`` containing JSON-per-line of
    raw MASSIVE rows (intent as a string, partition tag baked in).
    """
    import io

    with tarfile.open(path, "w:gz") as tar:
        for locale, rows in locale_rows.items():
            buf = io.BytesIO()
            for r in rows:
                buf.write(json.dumps(r).encode("utf-8"))
                buf.write(b"\n")
            data = buf.getvalue()
            info = tarfile.TarInfo(name=f"1.1/data/{locale}.jsonl")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_extract_jsonl_files_writes_per_locale(tmp_path):
    tarball = tmp_path / "massive.tar.gz"
    _build_fake_tarball(
        tarball,
        {
            "en-US": [_row("alarm_set", id="1"), _row("alarm_set", id="2", partition="test")],
            "de-DE": [_row("alarm_set", id="100", locale="de-DE")],
        },
    )

    output = tmp_path / "staged"
    massive._extract_jsonl_files(str(tarball), str(output))

    files = sorted(p.name for p in output.iterdir() if p.suffix == ".jsonl")
    assert files == ["de-DE.jsonl", "en-US.jsonl"]

    en_us_rows = [json.loads(line) for line in (output / "en-US.jsonl").read_text().splitlines()]
    assert [r["id"] for r in en_us_rows] == ["1", "2"]
    assert [r["partition"] for r in en_us_rows] == ["train", "test"]


def test_extract_jsonl_files_skips_existing(tmp_path):
    tarball = tmp_path / "massive.tar.gz"
    _build_fake_tarball(tarball, {"en-US": [_row("alarm_set")]})

    output = tmp_path / "staged"
    output.mkdir()
    (output / "en-US.jsonl").write_text("preexisting\n")

    massive._extract_jsonl_files(str(tarball), str(output))
    assert (output / "en-US.jsonl").read_text() == "preexisting\n"


def test_transform_staged_massive_writes_parquet_via_zephyr(tmp_path):
    """End-to-end: per-locale JSONL → zephyr transform → parquet shards."""
    staged = tmp_path / "staged"
    staged.mkdir()
    rows_by_locale = {
        "en-US": [
            _row("alarm_set", id="1"),
            _row("audio_volume_mute", id="2", utt="olly quiet", annot_utt="olly quiet"),
            _row("alarm_set", id="3", partition="test"),
        ],
        "de-DE": [_row("alarm_set", id="100", locale="de-DE", partition="dev")],
    }
    for locale, rows in rows_by_locale.items():
        with (staged / f"{locale}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r))
                f.write("\n")

    transformed = tmp_path / "transformed"
    transform_staged_massive(str(staged), str(transformed))

    import pyarrow.parquet as pq

    parquet_files = sorted(transformed.rglob("*.parquet"))
    assert parquet_files, "transform step should write at least one parquet shard"
    docs = []
    for path in parquet_files:
        docs.extend(pq.read_table(path).to_pylist())

    assert {d["id"] for d in docs} == {
        "en-US/1/train",
        "en-US/2/train",
        "en-US/3/test",
        "de-DE/100/validation",
    }
    for d in docs:
        assert d["text"].startswith("Tools: ")
        assert "\nRequest: " in d["text"]
        assert "\ntool_call: " in d["text"]
        assert d["source"] == massive.HF_DATASET_ID


def test_row_to_doc_rejects_unknown_intent():
    with pytest.raises(ValueError, match="Unknown intent"):
        row_to_doc(_row("not_a_real_intent"))
