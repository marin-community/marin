# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pyarrow as pa
import pytest
from marin.datakit.decon_keys import (
    LOCATOR_COLUMNS,
    WINDOW,
    KeyStream,
    KeyTable,
    Reading,
    Windows,
    _located,
    hit_indices,
    item_stream,
    key_streams,
    keys_table,
    record_hit_indices,
    record_pairs,
    record_readings,
    text_stream,
    window_hashes,
)


def test_text_stream_ignores_case_spacing_markup_and_latex_commands():
    assert text_stream("**What is $\\frac{1}{3}$ of 12?**") == text_stream("what is 1/3 of 12")
    # Full-width letters and digits fold to their ASCII forms.
    assert text_stream("\uff31\uff11: \uff21") == "q1a"


def test_hit_indices_find_a_key_inside_other_text():
    key = text_stream("the stylomastoid foramen will cause ipsilateral")[:WINDOW]
    keys = np.sort(window_hashes(key))
    assert len(hit_indices(text_stream("Notes. The stylomastoid foramen will cause ipsilateral palsy."), keys)) == 1
    assert len(hit_indices(text_stream("An unrelated sentence about the weather."), keys)) == 0


def test_a_copy_inside_json_once_or_twice_holds_every_window_of_the_item():
    item = (
        "from typing import List\n\n\ndef has_close_elements(numbers: List[float]) -> bool:\n"
        '    """ Check $\\theta \\neq \\nu$'
    )
    keys = np.sort(window_hashes(text_stream(item)))
    once = json.dumps({"prompt": item})
    twice = json.dumps({"row": once}, ensure_ascii=True)
    assert len(hit_indices(text_stream(once), keys)) < len(keys)
    for record in (once, twice):
        assert len(record_hit_indices("Data: " + record, keys)) == len(keys)


def test_a_short_question_is_located_by_its_key_with_the_choices():
    question, choices = "Which is it?", ["the first rather long choice", "the second rather long choice"]
    keyed = item_stream(question, choices)
    offset = len(keyed) - WINDOW
    record = (
        "Which is it? "
        + "filler words " * 400
        + "Which is it?\nA. the first rather long choice\nB. the second rather long choice"
    )
    reading, at = _located(record_readings(record), text_stream(question), keyed[offset : offset + WINDOW], offset)
    stream = reading.stream
    # The copy with its choices starts after the filler, not at the record's first "Which is it?".
    assert at == stream.rindex(text_stream(question))


def test_a_short_item_has_a_letter_free_key_and_a_passage_item_a_passage_key():
    short = key_streams("Which is it?", ["the first rather long choice", "the second rather long choice"], "")
    assert set(short) == {KeyStream.QUESTION, KeyStream.UNLETTERED}
    # A copy that lists the choices without letters holds every window of the letter-free stream.
    keys = set(window_hashes(short[KeyStream.UNLETTERED]).tolist())
    copy = "- Which is it?\n- the first rather long choice\n- the second rather long choice"
    assert keys <= set(window_hashes(text_stream(copy)).tolist())
    assert not set(window_hashes(short[KeyStream.QUESTION]).tolist()) <= set(window_hashes(text_stream(copy)).tolist())
    with_passage = key_streams("How many?", [], "Ann has three red apples and two green pears today.")
    assert set(with_passage) == {KeyStream.QUESTION, KeyStream.PASSAGE}


def test_an_extra_key_is_kept_only_when_no_reference_record_holds_it():
    # Streams: 0 = question of item a, 1 = passage of item a, 2 = passage of item b.
    windows = Windows(
        hashes=np.array([11, 12, 21, 22, 31], dtype=np.uint64),
        owners=np.array([0, 0, 1, 1, 2]),
        offsets=np.array([0, 25, 0, 25, 0]),
        ids=np.array(["a", "a", "b"]),
        kinds=np.array([KeyStream.QUESTION, KeyStream.PASSAGE, KeyStream.PASSAGE]),
    )
    rows = keys_table(windows, np.array([3, 5, 2, 4, 0])).to_pandas().set_index(["id", "kind"])
    # The question keeps its rarest window even though reference records hold it.
    assert rows.loc[("a", KeyStream.QUESTION), "key1_hash"] == 11
    # A passage whose rarest window occurs in reference text (a generic stem) gets no key.
    assert rows.loc[("a", KeyStream.PASSAGE), "key1_hash"] == 0
    assert rows.loc[("b", KeyStream.PASSAGE), "key1_hash"] == 31


def test_a_catalog_record_yields_one_pair_per_item_with_the_text_around_it():
    question = "Which planet of the solar system has the most moons in total?"
    stream = text_stream(question)
    key, offset = stream[10:30], 10
    rows = {
        "item": ["planets-0", "planets-0"],
        "kind": [KeyStream.QUESTION, KeyStream.PASSAGE],
        "eval": ["planets", "planets"],
        "key1_hash": [int(window_hashes(key)[0]), int(window_hashes(stream[25:45])[0])],
        "key": [key, stream[25:45]],
        "offset": [offset, 25],
        "question": [stream, stream],
        "question_chars": [len(question), len(question)],
    }
    locators = pa.table(
        {name: pa.array(rows[name], pa.uint64() if name == "key1_hash" else None) for name in LOCATOR_COLUMNS}
    )
    table = KeyTable(np.array(rows["key1_hash"], dtype=np.uint64), np.arange(2))
    record = "intro " * 400 + "Q: " + question + " A: Saturn." + " outro" * 900
    pairs = record_pairs("r1", record, table, locators)
    # Both streams of the item match, but the record gets one pair for the item.
    assert [pair["item"] for pair in pairs] == ["planets-0"]
    assert question in pairs[0]["right"] and len(pairs[0]["right"]) < len(record)
    escaped = json.dumps({"text": "Q: " + question.replace(" ", "\n")})
    assert [pair["item"] for pair in record_pairs("r2", escaped, table, locators)] == ["planets-0"]
    assert record_pairs("r3", "nothing about planets here " * 20, table, locators) == []


def _index(question: str, choices: list[str]) -> tuple[KeyTable, pa.Table]:
    """A key table over one item's key streams, each keyed by its middle window."""
    rows = {name: [] for name in LOCATOR_COLUMNS}
    for kind, stream in key_streams(question, choices, "").items():
        offset = (len(stream) - WINDOW) // 2
        key = stream[offset : offset + WINDOW]
        for name, value in zip(
            LOCATOR_COLUMNS,
            ("item-0", kind, "eval", int(window_hashes(key)[0]), key, offset, text_stream(question), len(question)),
            strict=True,
        ):
            rows[name].append(value)
    table = pa.table(
        {name: pa.array(values, pa.uint64() if name == "key1_hash" else None) for name, values in rows.items()}
    )
    return KeyTable(table.column("key1_hash").to_numpy(), np.arange(table.num_rows)), table


LONG = ("Which gas do plants take in from the air during photosynthesis?", ["Oxygen", "Carbon dioxide", "Nitrogen"])
SHORT = ("Which is it?", ["the first rather long choice", "the second rather long choice"])
PAIR = ("她把杯子清洗了\uff0c就去\u3002", ["她把杯子清洗了\uff0c就去\u3002", "杯子被她清洗了\uff0c就去\u3002"])


def _layouts(question: str, choices: list[str]) -> dict[str, str]:
    """The copy forms of gate G2, with the item's choices in order."""
    lettered = question + "\n" + "\n".join(f"{'ABCD'[k]}. {c}" for k, c in enumerate(choices))
    return {
        "plain": lettered,
        "parenthesized": question + "\n" + "\n".join(f"({'ABCD'[k]}) {c}" for k, c in enumerate(choices)),
        "extra_option": lettered + "\nE. none of the above",
        "no_line_breaks": lettered.replace("\n", " "),
        "upper": lettered.upper(),
        "no_punctuation": "".join(ch for ch in lettered if ch.isalnum() or ch.isspace()),
        "markdown": f"**{question}**\n" + "\n".join(f"- {c}" for c in choices),
        "escaped_once": json.dumps({"text": lettered}),
        "escaped_twice": json.dumps({"row": json.dumps({"text": lettered})}),
        "numbered": question + "\n" + "\n".join(f"{k + 1}. {c}" for k, c in enumerate(choices)),
        "json": json.dumps({"question": question, "choices": choices}, ensure_ascii=False),
    }


# Forms whose copies stage 1 misses when an item's text is shorter than a key, so that every key window
# crosses from the question into a choice: a number or a JSON field name between them breaks the window
# (ZhoBLiMP sentence pairs; a listed limit).
MISSED = {("short", "numbered"), ("pair", "numbered"), ("pair", "json")}


@pytest.mark.parametrize("name, item", [("long", LONG), ("short", SHORT), ("pair", PAIR)])
def test_each_planted_form_of_gate_g2_is_found_or_a_listed_miss(name, item):
    table, locators = _index(*item)
    host = "Some unrelated record text. " * 40
    for form, copy in _layouts(*item).items():
        found = [pair["item"] for pair in record_pairs("r", f"{host}\n{copy}\n{host}", table, locators)]
        assert found == ([] if (name, form) in MISSED else ["item-0"]), form


def test_a_scan_rejects_most_window_hashes_with_the_prefilter_and_finds_the_same_keys():
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 2**63, size=5000, dtype=np.uint64)
    table = KeyTable(keys, np.arange(len(keys)))
    probes = np.concatenate([rng.integers(0, 2**63, size=20000, dtype=np.uint64), keys[:7]])
    assert sorted(table.unique[table.found(probes)].tolist()) == sorted(keys[:7].tolist())


PIECES = [
    "word ",
    "word",
    "\u00dcn\u00efc\u00f6d\u00e9",
    "e\u0301",
    "\u1100\u1161\u11a8",
    "\u0b47\u0b3e",
    "\\\\frac{1}{2}",
    "\\frac{1}{2}",
    "\\alpha",
    "x\\ny",
    "\\\\alpha",
    "\uff3c\uff46\uff52\uff41\uff43",
    "\ufb01x",
    "\u0391\u03a3",
    "\u0130",
    "x\\\\ny",
    "\t",
    "\u6570\u5b66\u9898",
    "\uac00\ub098",
    "\uff21\uff22\uff23",
    "12",
    ", ",
]


def _full_search(text: str, position: int) -> int:
    low, high = 0, len(text)
    while low < high:
        middle = (low + high) // 2
        if len(text_stream(text[:middle])) > position:
            high = middle
        else:
            low = middle + 1
    return low


@pytest.mark.parametrize("seed", range(8))
def test_a_reading_maps_stream_positions_to_text_offsets_as_a_full_search_does(seed):
    rng = np.random.default_rng(seed)
    text = "".join(rng.choice(PIECES, size=9000))
    reading = Reading(text)
    for position in rng.integers(0, len(reading.stream) + 5, size=40).tolist():
        assert reading.text_offset(position) == _full_search(text, position)


def test_a_reading_finds_the_first_place_of_a_piece_as_str_find_does():
    rng = np.random.default_rng(0)
    reading = Reading("".join(rng.choice(["alpha beta ", "gamma ", "alpha beta gamma ", "delta "], size=3000)))
    for start in rng.integers(0, len(reading.stream) - 40, size=50).tolist():
        piece = reading.stream[start : start + 30]
        assert reading.first(piece) == reading.stream.find(piece)
    assert reading.first("zzzzzzzzzzzzzzzzzzzzzzzz") == -1


def test_an_offset_in_a_stretch_without_safe_characters_is_estimated_within_the_stretch():
    text = "intro " + "a" * 300_000 + " outro"
    reading = Reading(text)
    for position in (10, 150_000, 299_990):
        offset = reading.text_offset(position)
        assert abs(offset - _full_search(text, position)) <= 1 + 300_000 // 4
