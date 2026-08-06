# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the 100k oracle-labeled PDF sample.

The pieces worth pinning are the ones that decide *which* rows exist and *what* a row means: the
key that lets three artifacts written by different jobs join at all, the left join that keeps
documents the Docling route dropped, the segmentation the oracle scores, and the score parsing
that turns a free-text reply into a label. The stages themselves are storage-bound and are covered
by running them.
"""

import asyncio

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import experiments.build_pdf_source.quality.build_oracle_sample as oracle_module
from experiments.build_pdf_source.quality.build_oracle_sample import (
    DOCLING_COLUMNS,
    FAILED_SCORE,
    MIN_TOKENS_FOR_ALL_SEGMENTS,
    ORACLE_MAX_ATTEMPTS,
    RECORD_KEY,
    SEGMENT_TOKENS,
    SEGMENTS,
    OracleWork,
    label_columns,
    parse_score,
    record_key,
    segment_windows,
)

_WARC = "projects/cc/CC-SUPPLEMENTAL-2026-22/segments/20260524215335/warc/CC-SUPPL-0001.warc.gz"


def test_record_key_reproduces_the_clean_corpora_source_id():
    """The clean routes set ``source_id`` to exactly this, which is why the join can be derived.

    Verified against the real corpus: it held for all 13,548 rows of OCR shard 0. Pinning it here
    keeps the fetch-side derivation and the corpus-side column from silently drifting apart.
    """
    frame = pl.DataFrame(
        {
            "warc_filename": [_WARC, _WARC],
            "warc_record_offset": [1332730185, 2668560737],
            "source_id": [f"{_WARC}:1332730185", f"{_WARC}:2668560737"],
        }
    )
    keyed = frame.with_columns(record_key())
    assert keyed[RECORD_KEY].to_list() == keyed["source_id"].to_list()


def test_record_key_ignores_the_fetch_artifacts_own_source_id():
    """The fetch artifact's ``source_id`` is the WARC record UUID and shares no values with it.

    Keying the PDF join on the column name matched nothing at all -- the whole sample came back
    empty -- so this pins that the key comes from the WARC columns instead.
    """
    fetched = pl.DataFrame(
        {
            "warc_filename": [_WARC],
            "warc_record_offset": [960],
            "source_id": ["<urn:uuid:49aafdd1-273c-4361-bda1-121a39bbc91c>"],
        }
    )
    assert fetched.with_columns(record_key())[RECORD_KEY].to_list() == [f"{_WARC}:960"]


def test_docling_left_join_keeps_a_document_the_docling_route_dropped():
    """The sample is drawn over the OCR corpus, so rows Docling lost must survive with nulls.

    Dropping them is the failure mode this guards: it would silently turn the dataset back into a
    sample of the PDFs both routes agree on, which is the sample that was not asked for.
    """
    ocr = pl.LazyFrame({RECORD_KEY: ["a", "b", "c"], "text": ["A", "B", "C"]})
    docling = pl.LazyFrame({RECORD_KEY: ["a", "c"], "docling_text": ["dA", "dC"], "docling_page_offsets": [[1, 2], [3]]})
    joined = ocr.join(docling, on=RECORD_KEY, how="left").sort(RECORD_KEY).collect()

    assert joined.height == 3
    assert joined["docling_text"].to_list() == ["dA", None, "dC"]
    assert joined["docling_page_offsets"].to_list() == [[1, 2], None, [3]]


def test_docling_rename_map_covers_every_column_the_output_promises():
    """Both routes name their columns the same, so the Docling side must be renamed on the way in."""
    collisions = {renamed for renamed in DOCLING_COLUMNS.values() if renamed in {"id", "text", "num_pages"}}
    assert not collisions, f"unrenamed Docling columns would collide with the OCR route: {collisions}"


def test_score_comes_from_the_last_verdict_in_the_reply():
    """The rubric is restated in the prompt, and a reply can quote a score before concluding."""
    assert parse_score("The extract could reach Educational score: 2 but\nEducational score: 4") == 4
    assert parse_score("Educational score: 0") == 0


def test_reply_without_a_verdict_is_not_a_zero():
    """A refusal or a truncated reply must not enter the training set as the lowest grade."""
    assert parse_score("I cannot evaluate this extract.") == FAILED_SCORE


def test_long_document_is_scored_on_three_disjoint_windows():
    windows = segment_windows(list(range(4 * SEGMENT_TOKENS)))
    assert [segments for segments, _ in windows] == [["begin"], ["middle"], ["end"]]
    assert all(len(window) == SEGMENT_TOKENS for _, window in windows)
    covered = [token for _, window in windows for token in window]
    assert len(set(covered)) == len(covered)


def test_shortest_document_with_room_for_three_windows_still_gets_three():
    """At exactly three windows the slices tile the document without overlapping."""
    windows = segment_windows(list(range(MIN_TOKENS_FOR_ALL_SEGMENTS)))
    assert len(windows) == 3
    assert [token for _, window in windows for token in window] == list(range(MIN_TOKENS_FOR_ALL_SEGMENTS))


@pytest.mark.parametrize("tokens", [1, SEGMENT_TOKENS, MIN_TOKENS_FOR_ALL_SEGMENTS - 1])
def test_short_document_is_bought_once_and_the_verdict_shared(tokens):
    """Below three windows the slices would overlap, so paying three times buys near-duplicates."""
    windows = segment_windows(list(range(tokens)))
    assert len(windows) == 1
    segments, window = windows[0]
    assert sorted(segments) == sorted(SEGMENTS)
    assert window == list(range(min(tokens, SEGMENT_TOKENS)))


def test_label_columns_spreads_a_shared_verdict_across_all_three_segments():
    """A short document is scored once; all three columns must still carry that score and window."""
    work = OracleWork(
        requests=[{RECORD_KEY: "a", "segments": list(SEGMENTS), "text": "the whole short document"}],
        doc_tokens={"a": 40},
    )
    scores = pl.DataFrame(
        {
            RECORD_KEY: ["a"] * 3,
            "segment": list(SEGMENTS),
            "score": pl.Series([3, 3, 3], dtype=pl.Int8),
            "reason": ["because"] * 3,
        }
    )
    labels = label_columns(work, scores)

    assert labels.height == 1
    assert labels["doc_tokens"].to_list() == [40]
    for segment in SEGMENTS:
        assert labels[f"edu_score_v2_{segment}"].to_list() == [3]
        assert labels[f"edu_segment_v2_{segment}"].to_list() == ["the whole short document"]


def test_label_columns_keeps_each_window_with_its_own_verdict():
    """A long document's three windows are distinct text and must not be transposed."""
    work = OracleWork(
        requests=[{RECORD_KEY: "a", "segments": [segment], "text": segment.upper()} for segment in SEGMENTS],
        doc_tokens={"a": 5000},
    )
    scores = pl.DataFrame(
        {
            RECORD_KEY: ["a"] * 3,
            "segment": list(SEGMENTS),
            "score": pl.Series([1, 2, 3], dtype=pl.Int8),
            "reason": ["r1", "r2", "r3"],
        }
    )
    labels = label_columns(work, scores)

    assert [labels[f"edu_score_v2_{segment}"].item() for segment in SEGMENTS] == [1, 2, 3]
    assert [labels[f"edu_segment_v2_{segment}"].item() for segment in SEGMENTS] == ["BEGIN", "MIDDLE", "END"]


def test_label_columns_leaves_an_unscored_document_null_rather_than_zero():
    """An unanswered request must be visibly missing, not silently the lowest grade."""
    work = OracleWork(requests=[{RECORD_KEY: "a", "segments": list(SEGMENTS), "text": "extract"}], doc_tokens={"a": 40})
    empty = pl.DataFrame(schema={RECORD_KEY: pl.String, "segment": pl.String, "score": pl.Int8, "reason": pl.String})
    labels = label_columns(work, empty)
    assert labels["edu_score_v2_begin"].to_list() == [None]


def test_merge_join_drops_the_derived_key_but_keeps_the_corpus_identity():
    """``record_key`` is plumbing; ``source_id`` is the identity the dataset promises."""
    labeled = pl.LazyFrame({RECORD_KEY: ["a", "b"], "source_id": ["a", "b"], "text": ["A", "B"]})
    pdfs = pl.LazyFrame({RECORD_KEY: ["b"], "pdf": [b"%PDF-1.7"]})
    merged = labeled.join(pdfs, on=RECORD_KEY, how="inner").drop(RECORD_KEY).collect()

    assert RECORD_KEY not in merged.columns
    assert_frame_equal(merged, pl.DataFrame({"source_id": ["b"], "text": ["B"], "pdf": [b"%PDF-1.7"]}))


class _Replies:
    """An oracle stub: yields the queued outcomes, raising the ones that are exceptions."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def __call__(self, client, prompt, extract):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _run_score_one(monkeypatch, replies: _Replies, segments=("begin",)) -> list[dict]:
    # ``oracle_module.asyncio`` is the one asyncio module, so the backoff has to be neutralized
    # against a captured reference rather than by delegating back to the name being patched.
    unpatched_sleep = asyncio.sleep
    monkeypatch.setattr(oracle_module, "ask_oracle", replies)
    monkeypatch.setattr(oracle_module.asyncio, "sleep", lambda _delay: unpatched_sleep(0))
    item = {RECORD_KEY: "warc:1", "segments": list(segments), "text": "extract"}
    return asyncio.run(oracle_module.score_one(client=None, prompt="{example}", item=item))


def test_a_transient_failure_is_retried_and_the_answer_kept(monkeypatch):
    replies = _Replies(ValueError("flaky"), "Educational score: 3")
    assert [row["score"] for row in _run_score_one(monkeypatch, replies)] == [3]
    assert replies.calls == 2


def test_exhausting_the_attempts_records_a_failure_rather_than_a_score(monkeypatch):
    """The checkpoint is the resume log, so an unanswered request must be visibly unanswered."""
    replies = _Replies(*[ValueError("down")] * ORACLE_MAX_ATTEMPTS)
    rows = _run_score_one(monkeypatch, replies)
    assert replies.calls == ORACLE_MAX_ATTEMPTS
    assert [row["score"] for row in rows] == [FAILED_SCORE]
    assert rows[0]["reason"].startswith("ERROR:")


def test_one_reply_fans_out_to_every_segment_sharing_the_text(monkeypatch):
    replies = _Replies("Educational score: 2")
    rows = _run_score_one(monkeypatch, replies, segments=SEGMENTS)
    assert replies.calls == 1
    assert {row["segment"] for row in rows} == set(SEGMENTS)
    assert {row["score"] for row in rows} == {2}
