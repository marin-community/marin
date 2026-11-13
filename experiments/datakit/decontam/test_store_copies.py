# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.datakit.decontam.store_copies import (
    Matcher,
    SourceFile,
    file_matches,
    match_path,
    normalized,
    record_summary,
    search_shard,
    targets_for,
    write_targets,
)

QUESTION = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May."
ANSWER = "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May."
RECORDS = [{"id": "q1", "question": QUESTION, "answer": ANSWER, "label": "72"}]


def _targets():
    return targets_for(RECORDS, "id", ["question", "answer", "label"], min_words=8)


def _found(text: str) -> set[str]:
    return {t.field for t in Matcher(_targets()).found(text)}


def test_copies_are_found_through_case_whitespace_and_string_escapes():
    reflowed = f"Problem:\n{QUESTION.upper()}\n\nSolution:   {ANSWER.replace(chr(10), '   ')}"
    as_json_line = json.dumps({"q": QUESTION, "a": ANSWER, "note": "café"})
    as_python_literal = "x = '" + ANSWER.replace("\n", "\\n") + "'"

    assert _found(reflowed) == {"question", "answer"}
    assert _found(as_json_line) == {"question", "answer"}
    assert _found(as_python_literal) == {"answer"}


def test_partial_or_reworded_copies_and_short_fields_are_not_found():
    partial = "Natalia sold clips to 48 of her friends in April. The answer is 72."
    reworded = QUESTION.replace("half as many", "twice as many")

    assert _found(partial) == set()
    assert _found(reworded) == set()
    assert {t.field for t in _targets()} == {"question", "answer"}


def test_matcher_finds_exactly_the_targets_a_substring_search_finds():
    rng = random.Random(0)
    vocab = ["ab", "abc", "abcd", "bcd", "cd", "x", "the", "of"]
    records = [{"id": str(i), "text": " ".join(rng.choices(vocab, k=rng.randint(3, 6)))} for i in range(200)]
    targets = targets_for(records, "id", ["text"], min_words=3)
    matcher = Matcher(targets)
    for _ in range(200):
        doc = " ".join(rng.choices(vocab, k=rng.randint(5, 40)))
        expected = {t.record_id for t in targets if t.text in normalized(doc)}
        assert {t.record_id for t in matcher.found(doc)} == expected


def test_chunked_search_finds_copies_that_cross_chunk_boundaries():
    rng = random.Random(1)
    vocab = ["ab", "abc", "abcd", "bcd", "cd", "x", "the", "of"]
    records = [{"id": str(i), "text": " ".join(rng.choices(vocab, k=rng.randint(3, 8)))} for i in range(200)]
    targets = targets_for(records, "id", ["text"], min_words=3)
    # Chunks must hold the longest copy: 8 words of up to 4 letters with 3-character gaps.
    whole, chunked = Matcher(targets), Matcher(targets, chunk_chars=60)
    for _ in range(200):
        doc = (" \\n " if rng.random() < 0.3 else " \n ").join(rng.choices(vocab, k=rng.randint(5, 80)))
        assert {t.record_id for t in chunked.found(doc)} == {t.record_id for t in whole.found(doc)}


def _write(path: Path, rows: list[dict]) -> str:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return str(path)


def test_file_matches_flag_documents_the_store_dropped(tmp_path: Path):
    copy = f"{QUESTION}\n{ANSWER}"
    docs = [
        {"id": "kept", "text": copy},
        {"id": "marked", "text": copy},
        {"id": "exact", "text": copy},
        {"id": "fuzzy", "text": copy},
        {"id": "clean", "text": "An unrelated page about the weather in the valley this week."},
    ]
    file = SourceFile(
        source="src",
        path=_write(tmp_path / "docs.parquet", docs),
        decon=_write(tmp_path / "decon.parquet", [{"id": d["id"], "contaminated": d["id"] == "marked"} for d in docs]),
        exact=_write(tmp_path / "exact.parquet", [{"id": "exact"}]),
        fuzzy=_write(tmp_path / "fuzzy.parquet", [{"id": "fuzzy"}]),
    )

    rows = file_matches(file, Matcher(_targets()))

    kept = {(r["doc_id"], r["field"]): r["kept"] for r in rows}
    assert kept == {
        (doc, field): doc == "kept" for doc in ("kept", "marked", "exact", "fuzzy") for field in ("question", "answer")
    }


def test_file_matches_treat_a_missing_dedup_file_as_listing_no_documents(tmp_path: Path):
    file = SourceFile(
        source="src",
        path=_write(tmp_path / "docs.parquet", [{"id": "d", "text": QUESTION}]),
        decon=_write(tmp_path / "decon.parquet", [{"id": "d", "contaminated": False}]),
        exact=str(tmp_path / "absent.parquet"),
        fuzzy=None,
    )

    assert [r["kept"] for r in file_matches(file, Matcher(_targets()))] == [True]


def test_a_rerun_with_other_targets_under_the_same_prefix_searches_again(tmp_path: Path):
    file = SourceFile(
        source="src",
        path=_write(tmp_path / "docs.parquet", [{"id": "d", "text": QUESTION}]),
        decon=_write(tmp_path / "decon.parquet", [{"id": "d", "contaminated": False}]),
        exact=str(tmp_path / "absent.parquet"),
        fuzzy=None,
    )
    prefix = str(tmp_path / "out")

    def matched_records(records: list[dict]) -> set[str]:
        run = write_targets(prefix, targets_for(records, "id", ["question"], min_words=8))
        list(search_shard(run)(iter([vars(file)]), None))
        return {r["record_id"] for r in pq.read_table(str(match_path(run, file.path))).to_pylist()}

    absent = "A question about the rainfall in the valley that occurs in no document at all."
    assert matched_records([{"id": "q1", "question": QUESTION}]) == {"q1"}
    assert matched_records([{"id": "q1", "question": absent}]) == set()


def test_a_record_is_copied_only_when_one_kept_document_holds_every_field():
    def match(doc: str, field: str, kept: bool) -> dict:
        return {"record_id": "q1", "source": "src", "path": "f", "doc_id": doc, "field": field, "kept": kept}

    split_across_documents = [match("a", "question", True), match("b", "answer", True)]
    whole_but_dropped = [match("c", "question", False), match("c", "answer", False)]
    whole_and_kept = [match("d", "question", True), match("d", "answer", True)]

    [split] = record_summary(_targets(), split_across_documents)
    [dropped] = record_summary(_targets(), whole_but_dropped)
    [copied] = record_summary(_targets(), split_across_documents + whole_and_kept)

    assert (split["in_any_doc"], split["in_kept_doc"]) == (False, False)
    assert (dropped["in_any_doc"], dropped["in_kept_doc"]) == (True, False)
    assert (copied["in_kept_doc"], copied["kept_docs"], copied["sample_kept_docs"]) == (True, 1, ["f#d"])
