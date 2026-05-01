# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sanity tests for the AmazonScience/massive function-calling converter."""

from __future__ import annotations

import io
import json
import tarfile

from marin.datakit.download import massive
from marin.datakit.download.massive import (
    parse_annot_utt,
    row_to_doc,
    transform_staged_massive,
)


def test_parse_annot_utt_handles_messy_real_world_input():
    # Multi-slot, repeated slot names, internal punctuation, and unicode values
    # — covers the main parser paths in one shot.
    annot = (
        "order from [business_name : byron's] in [place_name : 北京] "
        "and also [place_name : tokyo] at [time : g. m. t. plus five]"
    )
    assert parse_annot_utt(annot) == [
        ("business_name", "byron's"),
        ("place_name", "北京"),
        ("place_name", "tokyo"),
        ("time", "g. m. t. plus five"),
    ]


def _row(intent: str, **overrides) -> dict:
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


def test_row_to_doc_renders_full_training_document():
    [doc] = row_to_doc(_row("alarm_set"))
    tools_line, request_line, tool_call_line = doc["text"].split("\n")

    embedded = json.loads(tools_line.removeprefix("Tools: "))
    assert "alarm_set" in {t["name"] for t in embedded}, "gold tool must be in the prompt"

    assert request_line == "Request: wake me up at nine am on friday"

    call = json.loads(tool_call_line.removeprefix("tool_call: "))
    assert call["type"] == "function_call"
    assert call["name"] == "alarm_set"
    # Responses API encodes ``arguments`` as a JSON string, not a nested object.
    assert json.loads(call["arguments"]) == {"date": ["friday"], "time": ["nine am"]}


def test_transform_staged_massive_end_to_end(tmp_path):
    """Tarball → extracted JSONL → zephyr transform → parquet of FC docs."""
    rows_by_locale = {
        "en-US": [_row("alarm_set", id="1"), _row("alarm_set", id="2", partition="test")],
        "de-DE": [_row("alarm_set", id="100", locale="de-DE", partition="dev")],
    }

    tarball = tmp_path / "massive.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        for locale, rows in rows_by_locale.items():
            data = ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")
            info = tarfile.TarInfo(name=f"1.1/data/{locale}.jsonl")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    staged = tmp_path / "staged"
    massive._extract_jsonl_files(str(tarball), str(staged))

    transformed = tmp_path / "transformed"
    transform_staged_massive(str(staged), str(transformed))

    import pyarrow.parquet as pq

    docs = [r for path in transformed.rglob("*.parquet") for r in pq.read_table(path).to_pylist()]
    assert {d["id"] for d in docs} == {
        "en-US/1/train",
        "en-US/2/test",
        "de-DE/100/validation",
    }
    for d in docs:
        assert d["text"].startswith("Tools: ")
        assert "\nRequest: " in d["text"]
        assert "\ntool_call: " in d["text"]
