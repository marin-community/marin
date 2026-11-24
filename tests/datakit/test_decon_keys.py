# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import random
import time

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
    candidate_windows,
    excerpt,
    hit_indices,
    item_stream,
    json_unescaped,
    key_streams,
    keys_table,
    long_record_pairs,
    pieces,
    readings,
    record_hit_indices,
    record_pairs,
    record_readings,
    template_bounds,
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
    assert [pair.item for pair in pairs] == ["planets-0"]
    assert question in pairs[0].right and len(pairs[0].right) < len(record)
    escaped = json.dumps({"text": "Q: " + question.replace(" ", "\n")})
    assert [pair.item for pair in record_pairs("r2", escaped, table, locators)] == ["planets-0"]
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
    """The planted copy forms (as ``experiments/datakit/decontam/encoder/planted.py`` writes them), with the
    item's choices in order."""
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
def test_each_planted_layout_is_found_or_a_listed_miss(name, item):
    table, locators = _index(*item)
    host = "Some unrelated record text. " * 40
    for form, copy in _layouts(*item).items():
        found = [pair.item for pair in record_pairs("r", f"{host}\n{copy}\n{host}", table, locators)]
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


@pytest.mark.parametrize(
    "text",
    [
        "\\" + "a" * 100_000 + " " + "Which planet of the solar system has the most moons?",
        "a" * 5000 + "uniquemarkerquestionwithoutspaces" + "a" * 5000 + "\u0301" * 60_000,
    ],
)
def test_an_offset_after_or_inside_a_long_stretch_without_safe_characters_is_exact(text):
    reading = Reading(text)
    for position in (0, len(reading.stream) // 2, len(reading.stream) - 1):
        assert reading.text_offset(position) == _full_search(text, position)


def test_a_question_past_the_checked_places_of_its_first_window_is_located_by_its_key():
    question = "a" * 20 + "question tail words about photosynthesis in plants"
    stream = text_stream(question)
    record = "intro " + ("a" * 20 + "b ") * 70 + question + " outro"
    readings_of = record_readings(record)
    # The first window of the question occurs at 71 places; the full question only at the last one.
    assert readings_of[0].first(stream) == -1 and readings_of[0].stream.find(stream) > 0
    right, _, at = excerpt(readings_of, len(question), stream, stream[20:40], 20)
    assert at == readings_of[0].stream.find(stream) and question in right


@pytest.mark.parametrize("header", ["intro ", "\u00a9 intro ", "\u201cintro\u201d ", "\u201cintro\u201d"])
def test_a_long_plain_ascii_stretch_maps_offsets_by_counting(header):
    text = header + "abc" * 30_000 + " outro"
    reading = Reading(text)
    for position in (3, 45_000, 89_999, len(reading.stream) - 1):
        assert reading.text_offset(position) == _full_search(text, position)


def test_offset_lookups_in_a_long_stretch_without_spaces_after_a_header_take_one_pass():
    reading = Reading("\u00a9 intro " + "abc" * 300_000 + " outro")
    # A search of the stretch per lookup takes about a second; one pass for all 500 lookups takes about 0.1 s.
    deadline = time.monotonic() + 5
    for position in range(0, len(reading.stream), len(reading.stream) // 500):
        reading.text_offset(position)
        assert time.monotonic() < deadline, "offset lookups search the whole stretch"


def test_keys_from_outside_the_eval_template_find_a_copy_without_the_template():
    questions = [
        "Question: Henry made two stops during his 60-mile bike trip.\nAnswer:",
        "Question: A robe takes 2 bolts of blue fiber and half that much white fiber.\nAnswer:",
        "Question: Short.\nAnswer:",
    ]
    streams = [text_stream(q) for q in questions]
    bounds = template_bounds(streams, ["gsm8k"] * 3)
    assert bounds[0] == (len("question"), len(streams[0]) - len("answer"))
    # A stream whose own part is shorter than a window keeps its whole range.
    assert bounds[2] == (0, len(streams[2]))
    hashes, owners, offsets = candidate_windows(streams[:1], bounds=bounds[:1])
    windows = Windows(hashes, owners, offsets, np.array(["gsm8k-0"]), np.array([KeyStream.QUESTION]))
    key = keys_table(windows, np.zeros(len(hashes), dtype=np.int64)).column("key1_hash").to_pylist()[0]
    record = "Solve this problem: Henry made two stops during his 60-mile bike trip. Show your work."
    assert key in set(window_hashes(text_stream(record)).tolist())


@pytest.mark.parametrize("length", [0, 5, 30, 31, 64, 100, 257])
def test_pieces_hold_every_stretch_up_to_the_overlap_in_one_piece(length):
    rng = random.Random(length)
    text = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(length))
    size, overlap = 100, 8
    spans, at = [], 0
    for piece in pieces(text, size, overlap):
        assert len(piece) <= size
        at = text.index(piece, at)
        spans.append((at, at + len(piece)))
    assert spans[-1][1] == len(text)
    for begin in range(len(text)):
        end = min(begin + overlap, len(text))
        assert any(start <= begin and end <= stop for start, stop in spans), (begin, end)


def test_a_key_spread_over_a_run_of_spaces_at_a_piece_edge_lies_in_one_piece():
    key = "abcdefghijklmnopqrst"
    text = "z" * 110 + key[:10] + " " * 12 + key[10:] + "y" * 100
    assert any(key in text_stream(piece) for piece in pieces(text, size=128, overlap=8))


def test_a_key_split_by_escaped_spaces_at_a_piece_edge_lies_in_one_piece():
    key = "abcdefghijklmnopqrst"
    text = "z" * 220 + key[:10] + "\\u0020" * 12 + key[10:] + "y" * 150
    assert key not in text_stream(text) and key in text_stream(json_unescaped(text))
    assert any(key in text_stream(piece) for reading in readings(text) for piece in pieces(reading, size=256, overlap=8))


def test_no_piece_edge_cuts_a_json_escape_inside_a_key():
    key = "abcdefghijklmnopqrst"
    for escape in ("\\u0020", "\\u005c\\u0075\\u0030\\u0030\\u0032\\u0030"):  # a space, and a space escaped twice
        text = "z" * 219 + key[:3] + escape + key[3:19] + " " * 12 + key[19] + "y" * 150
        assert any(key in text_stream(reading) for reading in readings(text))
        assert any(
            key in text_stream(piece) for reading in readings(text) for piece in pieces(reading, size=256, overlap=8)
        )


def test_a_long_record_keeps_the_whole_excerpt_of_a_hit_near_a_piece_edge():
    table, locators = _index(*LONG)
    question, choices = LONG
    copy = f"{question} A) {choices[0]} B) {choices[1]} C) {choices[2]}"
    filler = "lorem ipsum dolor sit amet. " * 1000
    text = filler[:11_000] + copy + filler[: 30_000 - 11_000]
    whole = record_pairs("r", text, table, locators)
    pieced = long_record_pairs("r", text, table, locators, size=12_000, overlap=4_600)
    assert {p.item for p in pieced} == {"item-0"}
    assert whole[0].right in [p.right for p in pieced]
    assert all(p.record_chars == len(text) for p in pieced)


def test_a_long_record_scores_a_later_exact_copy_after_an_edited_one():
    table, locators = _index(*LONG)
    question, choices = LONG
    edited = f"{question} A) a red fox B) a slow turtle C) a tall giraffe"
    exact = f"{question} A) {choices[0]} B) {choices[1]} C) {choices[2]}"
    filler = "lorem ipsum dolor sit amet. " * 1000
    text = filler[:3_000] + edited + filler[:20_000] + exact + filler[:3_000]
    pieced = long_record_pairs("r", text, table, locators, size=12_000, overlap=4_600)
    assert any(choices[0] in p.right for p in pieced)


def test_key_choice_takes_the_rarest_window_and_a_second_one_that_does_not_overlap():
    streams = [
        text_stream("Which of the following best describes the fallacy of equivocation in arguments?"),
        text_stream("short"),
    ]
    hashes, owners, offsets = candidate_windows(streams)
    windows = Windows(hashes, owners, offsets, np.array(["a", "b"], dtype=object), np.array([KeyStream.QUESTION] * 2))
    # Later windows are rarer in the reference sample.
    keys = keys_table(windows, np.arange(len(hashes), 0, -1)).to_pandas().set_index("id")
    assert keys.loc["a", "key1_offset"] == offsets[owners == 0].max()
    assert keys.loc["a", "key2_hash"] != 0 and keys.loc["a", "key2_hash"] != keys.loc["a", "key1_hash"]
    assert keys.loc["b", "key1_hash"] == 0
